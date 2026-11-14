"""Measure inference cost of the AnomalyDINO pipeline on an idle GPU.

The course requires mean inference time per image and peak GPU memory, measured
on the same device and with the resolution stated. Run this when no other job is
using the GPU; numbers collected under contention are meaningless.

    python scripts/benchmark_efficiency.py \
        --anomalydino-root /path/to/AnomalyDINO \
        --data-root /path/to/VisA_pytorch/1cls \
        --protocol configs/refs.json --shots 1 8 --n-images 50

Reported values:
  * ``memory_bank_s``      - building the bank from the reference images (once per class);
  * ``inference_ms_mean``  - end-to-end per image (read, resize, forward, kNN, map);
  * ``inference_ms_median``- robust version of the above;
  * ``peak_gpu_mb``        - ``torch.cuda.max_memory_allocated`` over the timed loop.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anomalydino-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--protocol", default="configs/refs.json")
    parser.add_argument("--class-name", default="candle")
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 8])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-images", type=int, default=50)
    parser.add_argument("--resolution", type=int, default=448)
    parser.add_argument("--eval-size", type=int, default=448)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--out", default="results/efficiency.json")
    args = parser.parse_args()

    sys.path.insert(0, os.path.abspath(args.anomalydino_root))

    import faiss
    import torch
    from src.backbones import get_model
    from src.post_eval import mean_top1p
    from src.utils import augment_image, dists2map, list_images

    hub_dir = os.path.join(torch.hub.get_dir(), "facebookresearch_dinov2_main")
    if os.path.isdir(hub_dir):
        original = torch.hub.load

        def load(repo_or_dir, model, *a, **kw):
            if repo_or_dir == "facebookresearch/dinov2":
                return original(hub_dir, model, *a, source="local", **kw)
            return original(repo_or_dir, model, *a, **kw)

        torch.hub.load = load

    protocol = json.load(open(args.protocol, encoding="utf-8"))
    info = protocol["classes"][args.class_name]
    ref_dir = os.path.join(args.data_root, args.class_name, "train", "good")
    test_dir = os.path.join(args.data_root, args.class_name, "test", "bad")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = get_model("dinov2_vits14", device, smaller_edge_size=args.resolution, layers=None)

    results = {}
    for shot in args.shots:
        refs = info["refs"][f"{shot}shot_seed{args.seed}"]

        start = time.perf_counter()
        bank = []
        for name in refs:
            image = cv2.cvtColor(
                cv2.imread(os.path.join(ref_dir, name), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
            )
            for aug in augment_image(image):
                tensor, grid = model.prepare_image(aug)
                feats = model.extract_features(tensor)
                keep = model.compute_background_mask(feats, grid, threshold=10, masking_type=False)
                bank.append(feats[keep])
        bank = np.concatenate(bank, axis=0).astype("float32")
        index = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), bank.shape[1])
        faiss.normalize_L2(bank)
        index.add(bank)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        bank_time = time.perf_counter() - start

        files = list_images(test_dir)[: args.n_images + args.warmup]

        def infer(name: str) -> float:
            t0 = time.perf_counter()
            image = cv2.cvtColor(
                cv2.imread(os.path.join(test_dir, name), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
            )
            tensor, grid = model.prepare_image(image)
            feats = model.extract_features(tensor)
            keep = model.compute_background_mask(feats, grid, threshold=10, masking_type=True)
            feats = feats[keep]
            faiss.normalize_L2(feats)
            distances, _ = index.search(feats, 1)
            distances = distances / 2
            patch_scores = np.zeros(len(keep), dtype=float)
            patch_scores[keep] = distances.squeeze()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            mean_top1p(patch_scores)
            dists2map(patch_scores.reshape(grid), (args.eval_size, args.eval_size))
            return elapsed

        for name in files[: args.warmup]:
            infer(name)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        times = [infer(name) for name in files[args.warmup :]]
        peak_mb = (
            torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
        )

        results[f"{shot}-shot"] = {
            "memory_bank_s": round(bank_time, 2),
            "bank_size": int(bank.shape[0]),
            "inference_ms_mean": round(statistics.mean(times) * 1000, 1),
            "inference_ms_median": round(statistics.median(times) * 1000, 1),
            "peak_gpu_mb": round(peak_mb, 1),
            "n_images": len(times),
            "resolution": args.resolution,
        }
        print(f"{shot}-shot: {results[f'{shot}-shot']}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"class": args.class_name, "device": device, "results": results}, fh, indent=2)
    print("written to", args.out)


if __name__ == "__main__":
    main()
