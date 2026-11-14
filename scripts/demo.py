"""Interactive-style demo: one product image in, score + OK/NG + heatmap out.

    python scripts/demo.py \
        --anomalydino-root /path/to/AnomalyDINO \
        --data-root /path/to/VisA_pytorch/1cls \
        --protocol configs/refs.json \
        --class-name candle --shot 1 --seed 0 \
        --image /path/to/test.JPG --out demo_out

Outputs:
  * stdout          - image score, calibrated threshold, OK/NG verdict;
  * <out>/*_overlay.png - input image with the anomaly map overlaid;
  * <out>/*_map.npy - the raw map (448x448) for inspection;
  * <out>/result.json - the same information in machine-readable form.

The threshold is the 95th percentile of the scores of the calibration images
reserved by the protocol (first 20 sorted normal images), so the reference pool
and the calibration images never overlap.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anomalydino-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--protocol", default="configs/refs.json")
    parser.add_argument("--class-name", default="candle")
    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default="demo_out")
    parser.add_argument("--resolution", type=int, default=448)
    parser.add_argument("--eval-size", type=int, default=448)
    args = parser.parse_args()

    sys.path.insert(0, os.path.abspath(args.anomalydino_root))

    import faiss
    import torch
    from src.backbones import get_model
    from src.post_eval import mean_top1p
    from src.utils import augment_image, dists2map

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

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = get_model("dinov2_vits14", device, smaller_edge_size=args.resolution, layers=None)

    def features_of(image: np.ndarray, masking: bool):
        tensor, grid = model.prepare_image(image)
        feats = model.extract_features(tensor)
        keep = model.compute_background_mask(feats, grid, threshold=10, masking_type=masking)
        return feats[keep], keep, grid

    # ---- memory bank from the protocol references (rotated, as in agnostic)
    bank = []
    for name in info["refs"][f"{args.shot}shot_seed{args.seed}"]:
        image = cv2.cvtColor(
            cv2.imread(os.path.join(ref_dir, name), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
        )
        for aug in augment_image(image):
            feats, _, _ = features_of(aug, masking=False)
            bank.append(feats)
    bank = np.concatenate(bank, axis=0).astype("float32")
    index = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), bank.shape[1])
    faiss.normalize_L2(bank)
    index.add(bank)

    def score(path: str, want_map: bool = False):
        image = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        feats, keep, grid = features_of(image, masking=True)
        faiss.normalize_L2(feats)
        distances, _ = index.search(feats, 1)
        patch_scores = np.zeros(len(keep), dtype=float)
        patch_scores[keep] = (distances / 2).squeeze()
        value = float(mean_top1p(patch_scores))
        amap = (
            dists2map(patch_scores.reshape(grid), (args.eval_size, args.eval_size))
            if want_map
            else None
        )
        return value, amap, image

    # ---- calibration (first 20 sorted normal images, excluded from the bank)
    calib_scores = [
        score(os.path.join(ref_dir, name))[0] for name in info["calibration"]
    ]
    threshold = float(np.percentile(calib_scores, 95))

    value, amap, image = score(args.image, want_map=True)
    verdict = "NG (anomalous)" if value > threshold else "OK (normal)"

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image))[0]
    np.save(os.path.join(args.out, f"{stem}_map.npy"), amap)

    # heatmap overlay
    norm = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    resized = cv2.resize(image, (args.eval_size, args.eval_size), interpolation=cv2.INTER_AREA)
    overlay = cv2.addWeighted(heat, 0.45, cv2.cvtColor(resized, cv2.COLOR_RGB2BGR), 0.55, 0)
    cv2.imwrite(os.path.join(args.out, f"{stem}_overlay.png"), overlay)

    result = {
        "class": args.class_name,
        "shot": args.shot,
        "seed": args.seed,
        "image": args.image,
        "score": round(value, 6),
        "threshold_p95_calibration": round(threshold, 6),
        "verdict": verdict,
        "calibration_scores": [round(s, 6) for s in calib_scores],
    }
    with open(os.path.join(args.out, "result.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    print(f"image      : {args.image}")
    print(f"score      : {value:.4f}")
    print(f"threshold  : {threshold:.4f} (95th percentile of 20 calibration images)")
    print(f"verdict    : {verdict}")
    print(f"artifacts  : {args.out}")


if __name__ == "__main__":
    main()
