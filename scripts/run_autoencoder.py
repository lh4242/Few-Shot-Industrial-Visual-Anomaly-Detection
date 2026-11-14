"""Optional Part 1 extension: train a compact autoencoder per class and evaluate it.

    python scripts/run_autoencoder.py --data-root /path/VisA_pytorch/1cls \
        --protocol configs/refs.json --out results/autoencoder.json

The autoencoder sees many normal images (``--train-count``, default 200, taken
from the sorted training list after removing the 20 calibration images). This is
a different regime from the few-shot rows and is reported as such.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from src.autoencoder import evaluate_class  # noqa: E402
from src.metrics import mean_std_over_seeds, unweighted_mean  # noqa: E402
from src.splits import load_protocol  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--protocol", default="configs/refs.json")
    parser.add_argument("--out", default="results/autoencoder.json")
    parser.add_argument("--classes", nargs="+", default=None)
    parser.add_argument("--train-count", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--eval-size", type=int, default=448)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
    classes = args.classes or list(protocol["classes"])

    runs = []
    for class_name in classes:
        info = protocol["classes"][class_name]
        for seed in args.seeds:
            print(f"[{class_name}] training on {args.train_count} images, seed {seed}", flush=True)
            result = evaluate_class(
                args.data_root,
                class_name,
                calibration=info["calibration"],
                train_count=args.train_count,
                size=args.size,
                eval_size=args.eval_size,
                epochs=args.epochs,
                seed=seed,
            )
            runs.append(result)
            print(
                f"{class_name:12s} image AUROC={result['image_auroc']*100:6.2f} "
                f"pixel AUROC={result['pixel_auroc']*100:6.2f} "
                f"(train {result['train_time_s']:.0f}s, {result['mean_time_s']*1000:.1f} ms/img)",
                flush=True,
            )

    per_seed = {}
    for seed in args.seeds:
        batch = [r for r in runs if r["seed"] == seed]
        per_seed[seed] = {
            "image": unweighted_mean({r["class"]: r["image_auroc"] for r in batch}),
            "pixel": unweighted_mean({r["class"]: r["pixel_auroc"] for r in batch}),
        }
    image_mean, image_std = mean_std_over_seeds(v["image"] for v in per_seed.values())
    pixel_mean, pixel_std = mean_std_over_seeds(v["pixel"] for v in per_seed.values())
    print(
        f"== autoencoder: image {image_mean*100:.2f} +- {image_std*100:.2f} | "
        f"pixel {pixel_mean*100:.2f} +- {pixel_std*100:.2f}"
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "method": "autoencoder",
                "config": {
                    "train_count": args.train_count,
                    "epochs": args.epochs,
                    "size": args.size,
                    "eval_size": args.eval_size,
                    "data_root": args.data_root,
                    "protocol": args.protocol,
                },
                "summary": {
                    "image_auroc_mean": image_mean,
                    "image_auroc_std": image_std,
                    "pixel_auroc_mean": pixel_mean,
                    "pixel_auroc_std": pixel_std,
                },
                "runs": runs,
            },
            fh,
            indent=2,
        )
    print("written to", args.out)


if __name__ == "__main__":
    main()
