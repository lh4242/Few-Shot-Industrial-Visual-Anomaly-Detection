"""Evaluate AnomalyDINO with the shared metrics, entirely in memory.

Why this exists
---------------
The official ``run_anomalydino.py --eval_segm`` dumps a *full-resolution* float
map per test image (~11 GB for the six VisA classes) and then spends minutes
computing AUPRO. We only need image AUROC and pixel AUROC, so this script:

  * reuses the upstream building blocks unchanged (backbone wrapper, rotation
    augmentation, PCA background masking, FAISS kNN, ``mean_top1p`` scoring);
  * resizes each patch-distance map to a fixed ``--eval-size`` (default 448, the
    same resolution the template baseline works at) instead of the original
    image resolution;
  * feeds scores and maps straight into ``src/metrics.py``, writing only a small
    JSON summary.

Deviation from the official protocol, to be stated in the report: pixel-level
AUROC is computed on maps resampled to 448 px rather than at the original image
resolution. Image-level AUROC is identical to the official implementation.

Usage (run from the repository root):

    python scripts/run_anomalydino_mem.py \
        --anomalydino-root /path/to/AnomalyDINO \
        --data-root /path/to/VisA_pytorch/1cls_fs \
        --protocol configs/refs.json --out results/anomalydino.json

Importing this script puts ``--anomalydino-root`` on ``sys.path`` so that the
upstream ``src`` package wins over ours; our ``src/metrics.py`` is therefore
loaded by file path under the name ``p2_metrics``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from typing import Dict, List

import cv2
import numpy as np


def _load_metrics(repo_root: str):
    """Import our metrics module without colliding with AnomalyDINO's `src`."""
    path = os.path.join(repo_root, "src", "metrics.py")
    spec = importlib.util.spec_from_file_location("p2_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dump_partial(path: str, runs) -> None:
    """Write everything finished so far, so a long sweep survives an interruption."""
    partial = os.path.splitext(path)[0] + ".partial.json"
    os.makedirs(os.path.dirname(os.path.abspath(partial)) or ".", exist_ok=True)
    with open(partial, "w", encoding="utf-8") as fh:
        json.dump({"runs": runs}, fh)


def _enable_offline_hub() -> None:
    """Load DINOv2 from the local torch hub cache instead of github.com.

    ``torch.hub.load('facebookresearch/dinov2', ...)`` first asks the GitHub API
    for the default branch, which fails on machines without GitHub access (the
    course server is intermittent). If the repository was already cached under
    ``$TORCH_HOME/hub/facebookresearch_dinov2_main``, redirect the call there with
    ``source='local'``; nothing is downloaded.
    """
    import torch

    local_repo = os.path.join(torch.hub.get_dir(), "facebookresearch_dinov2_main")
    if not os.path.isdir(local_repo):
        return
    original = torch.hub.load

    def load(repo_or_dir, model, *args, **kwargs):
        if repo_or_dir == "facebookresearch/dinov2":
            return original(local_repo, model, *args, source="local", **kwargs)
        return original(repo_or_dir, model, *args, **kwargs)

    torch.hub.load = load
    print(f"torch.hub: using local DINOv2 repo at {local_repo}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anomalydino-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--protocol", default="configs/refs.json")
    parser.add_argument("--out", default="results/anomalydino.json")
    parser.add_argument("--model-name", default="dinov2_vits14")
    parser.add_argument("--resolution", type=int, default=448)
    parser.add_argument("--eval-size", type=int, default=448)
    parser.add_argument("--shots", nargs="+", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--classes", nargs="+", default=None)
    parser.add_argument("--faiss-on-cpu", action="store_true")
    parser.add_argument("--max-test-images", type=int, default=None)
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ad_root = os.path.abspath(args.anomalydino_root)
    sys.path.insert(0, ad_root)

    import faiss
    import torch
    from src.backbones import get_model
    from src.post_eval import mean_top1p
    from src.utils import augment_image, dists2map, list_images

    p2 = _load_metrics(repo_root)

    protocol = json.load(open(args.protocol, encoding="utf-8"))
    classes = args.classes or list(protocol["classes"])
    shots = args.shots or protocol["shots"]
    seeds = args.seeds or protocol["seeds"]
    n_calibration = protocol.get("n_calibration", 20)

    _enable_offline_hub()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = get_model(args.model_name, device, smaller_edge_size=args.resolution, layers=None)
    # 'agnostic' preprocessing: rotate the references, mask the test images
    rotation, masking = True, True

    runs: List[Dict] = []
    for class_name in classes:
        info = protocol["classes"][class_name]
        ref_dir = os.path.join(args.data_root, class_name, "train", "good")
        good_dir = os.path.join(args.data_root, class_name, "test", "good")
        bad_dir = os.path.join(args.data_root, class_name, "test", "bad")
        gt_dir = os.path.join(args.data_root, class_name, "ground_truth", "bad")

        for shot in shots:
            for seed in seeds:
                key = f"{shot}shot_seed{seed}"
                ref_files = info["refs"].get(key)
                if ref_files is None:
                    continue

                # ---- memory bank
                start = time.perf_counter()
                bank = []
                for name in ref_files:
                    image = cv2.cvtColor(
                        cv2.imread(os.path.join(ref_dir, name), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
                    )
                    for aug in (augment_image(image) if rotation else [image]):
                        tensor, grid = model.prepare_image(aug)
                        feats = model.extract_features(tensor)
                        keep = model.compute_background_mask(
                            feats, grid, threshold=10, masking_type=False
                        )
                        bank.append(feats[keep])
                bank = np.concatenate(bank, axis=0).astype("float32")
                if args.faiss_on_cpu:
                    index = faiss.IndexFlatL2(bank.shape[1])
                else:
                    index = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), bank.shape[1])
                faiss.normalize_L2(bank)
                index.add(bank)
                bank_time = time.perf_counter() - start

                # ---- inference
                scores: List[float] = []
                labels: List[int] = []
                maps: List[np.ndarray] = []
                masks: List[np.ndarray] = []
                times: List[float] = []
                records: List[Dict] = []

                for sub, label in ((good_dir, 0), (bad_dir, 1)):
                    if not os.path.isdir(sub):
                        continue
                    files = list_images(sub)
                    if args.max_test_images:
                        files = files[: args.max_test_images]
                    for name in files:
                        start = time.perf_counter()
                        image = cv2.cvtColor(
                            cv2.imread(os.path.join(sub, name), cv2.IMREAD_COLOR),
                            cv2.COLOR_BGR2RGB,
                        )
                        tensor, grid = model.prepare_image(image)
                        feats = model.extract_features(tensor)
                        keep = model.compute_background_mask(
                            feats, grid, threshold=10, masking_type=masking
                        )
                        feats = feats[keep]
                        faiss.normalize_L2(feats)
                        distances, _ = index.search(feats, 1)
                        distances = distances / 2  # cosine distance

                        patch_scores = np.zeros(len(keep), dtype=float)
                        patch_scores[keep] = distances.squeeze()
                        if device != "cpu":
                            torch.cuda.synchronize()
                        times.append(time.perf_counter() - start)

                        score = float(mean_top1p(patch_scores))
                        scores.append(score)
                        labels.append(label)
                        maps.append(
                            dists2map(
                                patch_scores.reshape(grid), (args.eval_size, args.eval_size)
                            )
                        )

                        gt_path = (
                            os.path.join(gt_dir, os.path.splitext(name)[0] + ".png")
                            if label
                            else None
                        )
                        if gt_path and os.path.exists(gt_path):
                            mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                            mask = cv2.resize(
                                mask, (args.eval_size, args.eval_size),
                                interpolation=cv2.INTER_NEAREST,
                            )
                            masks.append((mask > 0).astype(np.uint8))
                        else:
                            masks.append(np.zeros((args.eval_size, args.eval_size), np.uint8))
                        records.append(
                            {
                                "file": f"{'bad' if label else 'good'}/{name}",
                                "score": round(score, 6),
                            }
                        )

                result = {
                    "class": class_name,
                    "shot": shot,
                    "seed": seed,
                    "references": list(ref_files),
                    "n_calibration_excluded": n_calibration,
                    "image_auroc": p2.image_auroc(scores, labels),
                    "pixel_auroc": p2.pixel_auroc(maps, masks),
                    "mean_time_s": float(np.mean(times)),
                    "memory_bank_time_s": bank_time,
                    "num_test": len(scores),
                    "num_anomalous": int(np.sum(labels)),
                    "records": records,
                }
                runs.append(result)
                _dump_partial(args.out, runs)
                print(
                    f"{class_name:12s} {key:14s} image AUROC={result['image_auroc']*100:6.2f} "
                    f"pixel AUROC={result['pixel_auroc']*100:6.2f} "
                    f"({result['mean_time_s']*1000:.1f} ms/img, bank {bank_time:.1f}s)",
                    flush=True,
                )

    summary: Dict[str, Dict] = {}
    for shot in shots:
        per_seed = [r for r in runs if r["shot"] == shot]
        present = sorted({r["seed"] for r in per_seed})
        image_runs, pixel_runs = [], []
        for seed in present:
            batch = [r for r in per_seed if r["seed"] == seed]
            image_runs.append(p2.unweighted_mean({r["class"]: r["image_auroc"] for r in batch}))
            pixel_runs.append(p2.unweighted_mean({r["class"]: r["pixel_auroc"] for r in batch}))
        image_mean, image_std = p2.mean_std_over_seeds(image_runs)
        pixel_mean, pixel_std = p2.mean_std_over_seeds(pixel_runs)
        summary[f"{shot}-shot"] = {
            "image_auroc_mean": image_mean,
            "image_auroc_std": image_std,
            "pixel_auroc_mean": pixel_mean,
            "pixel_auroc_std": pixel_std,
            "num_seeds": len(present),
        }
        print(
            f"== {shot}-shot: image {image_mean*100:.2f} +- {image_std*100:.2f} | "
            f"pixel {pixel_mean*100:.2f} +- {pixel_std*100:.2f} ({len(present)} seeds)",
            flush=True,
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "method": "anomalydino",
                "config": {
                    "model": args.model_name,
                    "resolution": args.resolution,
                    "eval_size": args.eval_size,
                    "preprocess": "agnostic",
                    "faiss_on_cpu": args.faiss_on_cpu,
                    "data_root": args.data_root,
                    "protocol": args.protocol,
                },
                "summary": summary,
                "runs": runs,
            },
            fh,
            indent=2,
        )
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
