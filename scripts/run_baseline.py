"""Run the Part 1 template-difference baseline over the few-shot protocol.

    python scripts/run_baseline.py --data-root /path/VisA_pytorch/1cls \
        --protocol configs/refs.json --out results/baseline.json

Every (class, shot, seed) condition is evaluated with the same reference images
that AnomalyDINO uses, so the two rows of the comparison table are directly
comparable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baseline import DEFAULT_FEATURES, aggregate, evaluate_class  # noqa: E402
from src.metrics import mean_std_over_seeds, unweighted_mean  # noqa: E402
from src.splits import load_protocol  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--protocol", default="configs/refs.json")
    parser.add_argument("--out", default="results/baseline.json")
    parser.add_argument("--shots", nargs="+", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--classes", nargs="+", default=None)
    parser.add_argument("--size", type=int, default=448)
    parser.add_argument("--features", nargs="+", default=list(DEFAULT_FEATURES))
    parser.add_argument("--ecc", action="store_true", help="enable ECC euclidean alignment")
    parser.add_argument(
        "--border-crop",
        type=int,
        default=32,
        help="ignore this many border pixels (resizing/SSIM edge artefacts) in scoring and pixel metrics",
    )
    parser.add_argument(
        "--max-test-images",
        type=int,
        default=None,
        help="limit the number of test images per class (debugging only)",
    )
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
    classes = args.classes or list(protocol["classes"])
    shots = args.shots or protocol["shots"]
    seeds = args.seeds or protocol["seeds"]

    all_results: List[Dict] = []
    for class_name in classes:
        info = protocol["classes"][class_name]
        for shot in shots:
            for seed in seeds:
                key = f"{shot}shot_seed{seed}"
                if key not in info["refs"]:
                    print(f"skip {class_name} {key}: not in protocol")
                    continue
                result = evaluate_class(
                    args.data_root,
                    class_name,
                    shot,
                    seed,
                    refs=info["refs"][key],
                    calibration=info["calibration"],
                    size=args.size,
                    features=args.features,
                    use_ecc=args.ecc,
                    border_crop=args.border_crop,
                    max_test_images=args.max_test_images,
                )
                all_results.append(result)
                print(
                    f"{class_name:12s} {key:14s} image AUROC={result['image_auroc']*100:6.2f} "
                    f"pixel AUROC={result['pixel_auroc']*100:6.2f} "
                    f"thr={result['threshold']:.4f} ({result['mean_time_s']*1000:.1f} ms/img)"
                )

    # ---- aggregate: per shot -> mean over classes, then mean/std over seeds
    summary: Dict[str, Dict] = {}
    for shot in shots:
        per_seed = [r for r in all_results if r["shot"] == shot]
        seeds_present = sorted({r["seed"] for r in per_seed})
        image_runs, pixel_runs = [], []
        for seed in seeds_present:
            runs = [r for r in per_seed if r["seed"] == seed]
            image_runs.append(unweighted_mean({r["class"]: r["image_auroc"] for r in runs}))
            pixel_runs.append(unweighted_mean({r["class"]: r["pixel_auroc"] for r in runs}))
        image_mean, image_std = mean_std_over_seeds(image_runs)
        pixel_mean, pixel_std = mean_std_over_seeds(pixel_runs)
        summary[f"{shot}-shot"] = {
            "image_auroc_mean": image_mean,
            "image_auroc_std": image_std,
            "pixel_auroc_mean": pixel_mean,
            "pixel_auroc_std": pixel_std,
            "num_seeds": len(seeds_present),
        }
        print(
            f"== {shot}-shot: image {image_mean*100:.2f} +- {image_std*100:.2f} | "
            f"pixel {pixel_mean*100:.2f} +- {pixel_std*100:.2f} ({len(seeds_present)} seeds)"
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    payload = {
        "method": "template_difference",
        "config": {
            "size": args.size,
            "features": args.features,
            "ecc": args.ecc,
            "border_crop": args.border_crop,
            "data_root": args.data_root,
            "protocol": args.protocol,
        },
        "summary": summary,
        "runs": all_results,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
