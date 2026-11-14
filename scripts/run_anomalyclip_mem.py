"""Evaluate AnomalyCLIP with the shared metrics, reusing the upstream inference.

AnomalyCLIP is a **zero-shot** method: it needs no reference images, so this row
is not comparable to the 1/2/4/8-shot rows and must be labelled as zero-shot.

The scoring pipeline (text prompts, DPAM multi-scale similarity, anomaly map)
belongs to the AnomalyCLIP authors (Apache-2.0, https://github.com/zqhang/AnomalyCLIP);
this script only reproduces their inference loop and replaces the metric
computation with ``src/metrics.py`` so that every method in the report is scored
by identical code.

    python scripts/run_anomalyclip_mem.py \
        --anomalyclip-root /path/to/AnomalyCLIP \
        --data-root /path/to/VisA_pytorch/1cls \
        --checkpoint checkpoints/9_12_4_multiscale_visa/epoch_15.pth \
        --out results/anomalyclip.json

``--data-root`` must contain the ``meta.json`` that the upstream ``Dataset``
expects (see ``scripts/make_anomalyclip_meta.py``).

Deviations, stated in the report: anomaly maps are resampled to ``--eval-size``
(448) before scoring, and AUPRO is not recomputed here.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np


def _load_metrics(repo_root: str):
    path = os.path.join(repo_root, "src", "metrics.py")
    spec = importlib.util.spec_from_file_location("p2_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anomalyclip-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="results/anomalyclip.json")
    parser.add_argument("--eval-size", type=int, default=448)
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--n-ctx", type=int, default=12)
    parser.add_argument("--t-n-ctx", type=int, default=4)
    parser.add_argument("--sigma", type=int, default=4)
    parser.add_argument("--max-test-images", type=int, default=None)
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ac_root = os.path.abspath(args.anomalyclip_root)
    sys.path.insert(0, ac_root)
    os.chdir(ac_root)  # the checkpoint path is relative to the repo

    import cv2
    import torch
    from scipy.ndimage import gaussian_filter
    from torch.utils.data import DataLoader

    from AnomalyCLIP_lib import compute_similarity, get_similarity_map
    import AnomalyCLIP_lib
    from dataset import Dataset
    from prompt_ensemble import AnomalyCLIP_PromptLearner
    from utils import get_transform

    p2 = _load_metrics(repo_root)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    params = {
        "Prompt_length": args.n_ctx,
        "learnabel_text_embedding_depth": args.depth,
        "learnabel_text_embedding_length": args.t_n_ctx,
    }
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=params)
    model.eval()

    preprocess, target_transform = get_transform(args)
    dataset = Dataset(
        root=args.data_root, transform=preprocess,
        target_transform=target_transform, dataset_name="visa",
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    learner = AnomalyCLIP_PromptLearner(model.to("cpu"), params)
    ckpt = torch.load(
        os.path.join(ac_root, args.checkpoint), map_location="cpu", weights_only=False
    )
    learner.load_state_dict(ckpt["prompt_learner"])
    learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=20)

    prompts, tokenized_prompts, compound = learner(cls_id=None)
    text_features = model.encode_text_learn(prompts, tokenized_prompts, compound).float()
    text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    results = {}
    for idx, items in enumerate(loader):
        cls_name = items["cls_name"][0]
        if args.max_test_images and results.get(cls_name, {}).get("n", 0) >= args.max_test_images:
            continue
        entry = results.setdefault(
            cls_name, {"scores": [], "labels": [], "maps": [], "masks": [], "n": 0}
        )
        image = items["img"].to(device)
        start = time.perf_counter()
        with torch.no_grad():
            image_features, patch_features = model.encode_image(image, [24], DPAM_layer=20)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            probs = (image_features @ text_features.permute(0, 2, 1) / 0.07).softmax(-1)
            image_score = float(probs[:, 0, 1])

            maps = []
            for layer_idx, patch_feature in enumerate(patch_features):
                if layer_idx >= 0:
                    patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                    similarity, _ = compute_similarity(patch_feature, text_features[0])
                    sim_map = get_similarity_map(similarity[:, 1:, :], args.image_size)
                    maps.append((sim_map[..., 1] + 1 - sim_map[..., 0]) / 2.0)
            anomaly_map = torch.stack(maps).sum(dim=0)
        elapsed = time.perf_counter() - start

        amap = anomaly_map.detach().cpu().numpy()
        amap = np.squeeze(amap)
        amap = np.array([gaussian_filter(amap, sigma=args.sigma)])
        amap = cv2.resize(np.squeeze(amap), (args.eval_size, args.eval_size),
                          interpolation=cv2.INTER_LINEAR)

        # defect-free images come back with an empty mask list upstream
        gt_raw = items["img_mask"]
        try:
            gt_np = np.squeeze(np.asarray(gt_raw[0].numpy()))
        except Exception:
            gt_np = None
        if gt_np is None or gt_np.size == 0:
            gt = np.zeros((args.eval_size, args.eval_size), np.uint8)
        else:
            gt = cv2.resize(
                gt_np.astype(np.uint8), (args.eval_size, args.eval_size),
                interpolation=cv2.INTER_NEAREST,
            )

        entry["scores"].append(image_score)
        entry["labels"].append(int(items["anomaly"][0]))
        entry["maps"].append(amap)
        entry["masks"].append((gt > 0).astype(np.uint8))
        entry.setdefault("times", []).append(elapsed)
        entry["n"] += 1

    runs = []
    for cls_name, entry in results.items():
        runs.append({
            "class": cls_name,
            "method": "anomalyclip",
            "shots": 0,
            "image_auroc": p2.image_auroc(entry["scores"], entry["labels"]),
            "pixel_auroc": p2.pixel_auroc(entry["maps"], entry["masks"]),
            "mean_time_s": float(np.mean(entry["times"])),
            "num_test": entry["n"],
            "num_anomalous": int(np.sum(entry["labels"])),
        })
        r = runs[-1]
        print(f"{cls_name:12s} image AUROC={r['image_auroc']*100:6.2f} "
              f"pixel AUROC={r['pixel_auroc']*100:6.2f} ({r['num_test']} imgs)", flush=True)

    image_mean = p2.unweighted_mean({r["class"]: r["image_auroc"] for r in runs})
    pixel_mean = p2.unweighted_mean({r["class"]: r["pixel_auroc"] for r in runs})
    print(f"== AnomalyCLIP (zero-shot): image {image_mean*100:.2f} | pixel {pixel_mean*100:.2f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "method": "anomalyclip",
                "regime": "zero-shot (no reference images)",
                "checkpoint": args.checkpoint,
                "config": {"image_size": args.image_size, "eval_size": args.eval_size,
                           "sigma": args.sigma},
                "summary": {"image_auroc_mean": image_mean, "pixel_auroc_mean": pixel_mean},
                "runs": runs,
            },
            fh,
            indent=2,
        )
    print("written to", args.out)


if __name__ == "__main__":
    main()
