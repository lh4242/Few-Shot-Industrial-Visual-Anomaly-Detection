"""Build the reference/calibration protocol and (optionally) the data views.

    python scripts/make_refs.py --data-root /path/VisA_pytorch/1cls --out configs/refs.json

With ``--materialise <dir>`` it also creates ``<dir>/1cls_fs`` (reference pool
without the 20 calibration images) and ``<dir>/calib``, both as symlinks. Point
AnomalyDINO at ``<dir>/1cls_fs`` so that its index-based sampling stays inside
the pool and uses exactly the same references as the baseline.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.splits import (  # noqa: E402
    CLASSES,
    N_CALIBRATION,
    SEEDS,
    SHOTS,
    build_protocol,
    materialise_pool,
    save_protocol,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="MVTec-style VisA 1cls root")
    parser.add_argument("--out", default="configs/refs.json")
    parser.add_argument("--classes", nargs="+", default=CLASSES)
    parser.add_argument("--shots", nargs="+", type=int, default=SHOTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--n-calibration", type=int, default=N_CALIBRATION)
    parser.add_argument(
        "--materialise",
        default=None,
        help="optional output dir for the 1cls_fs / calib symlink views",
    )
    args = parser.parse_args()

    protocol = build_protocol(
        args.data_root,
        classes=args.classes,
        shots=args.shots,
        seeds=args.seeds,
        n_calibration=args.n_calibration,
    )
    save_protocol(protocol, args.out)

    print(f"classes: {', '.join(protocol['classes'])}")
    for name, info in protocol["classes"].items():
        print(
            f"  {name:12s} train={info['n_train']:4d}  pool={info['pool_size']:4d}  "
            f"calibration={len(info['calibration'])}"
        )
    print(f"protocol written to {args.out}")

    if args.materialise:
        materialise_pool(args.data_root, args.materialise, protocol)
        print(f"data views written to {args.materialise}/1cls_fs and {args.materialise}/calib")


if __name__ == "__main__":
    main()
