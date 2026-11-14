"""Reference-image protocol for the few-shot VisA experiments.

AnomalyDINO picks few-shot references deterministically, by index slicing of the
*sorted* list of normal training images:

    refs = sorted(train_good)[seed * shot : (seed + 1) * shot]

The course protocol additionally requires that the first 20 sorted normal images
(per class) are reserved for demo threshold calibration and excluded from every
reference selection. Slicing the full list would therefore hand out calibration
images as references, so both the baseline and AnomalyDINO read from a shifted
pool instead:

    pool = sorted(train_good)[20:]
    refs = pool[seed * shot : (seed + 1) * shot]

This module materialises that pool/calibration split on disk and records the
exact file names in ``configs/refs.json`` so that every method uses the same
references.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Sequence

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

CLASSES: List[str] = [
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "pipe_fryum",
]
SHOTS: List[int] = [1, 2, 4, 8]
SEEDS: List[int] = [0, 1, 2]
N_CALIBRATION: int = 20


def list_images(folder: str) -> List[str]:
    """Sorted image file names in a folder (case-insensitive extension check)."""
    return sorted(
        f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    )


def build_protocol(
    data_root: str,
    classes: Sequence[str] = CLASSES,
    shots: Sequence[int] = SHOTS,
    seeds: Sequence[int] = SEEDS,
    n_calibration: int = N_CALIBRATION,
) -> Dict:
    """Describe calibration images and reference images for every (class, shot, seed)."""
    protocol = {
        "data_root": data_root.rstrip("/"),
        "n_calibration": n_calibration,
        "shots": list(shots),
        "seeds": list(seeds),
        "sampling": "pool = sorted(train/good)[n_calibration:]; "
                    "refs = pool[seed * shot : (seed + 1) * shot]",
        "classes": {},
    }
    for name in classes:
        train_dir = os.path.join(data_root, name, "train", "good")
        train_files = list_images(train_dir)
        if len(train_files) < n_calibration + max(shots) * max(seeds + [0]) + max(shots):
            # keep a readable guard instead of failing deep inside an experiment
            pass
        calibration = train_files[:n_calibration]
        pool = train_files[n_calibration:]

        refs: Dict[str, List[str]] = {}
        for shot in shots:
            for seed in seeds:
                picked = pool[seed * shot : (seed + 1) * shot]
                if len(picked) != shot:
                    raise ValueError(
                        f"{name}: cannot draw {shot} reference images for seed {seed} "
                        f"(pool has {len(pool)} after removing {n_calibration} calibration images)"
                    )
                refs[f"{shot}shot_seed{seed}"] = picked

        protocol["classes"][name] = {
            "n_train": len(train_files),
            "calibration": calibration,
            "pool_size": len(pool),
            "refs": refs,
        }
    return protocol


def save_protocol(protocol: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(protocol, fh, indent=2, ensure_ascii=False)


def load_protocol(path: str) -> Dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def materialise_pool(
    data_root: str,
    out_root: str,
    protocol: Dict,
    link: bool = True,
) -> None:
    """Create ``<out_root>/1cls_fs`` (pool only) and ``<out_root>/calib`` as symlinks.

    ``1cls_fs`` keeps the MVTec layout so that the unmodified AnomalyDINO code can
    be pointed at it: its index slicing then automatically stays inside the pool.
    """
    for name, info in protocol["classes"].items():
        src_train = os.path.join(data_root, name, "train", "good")
        dst_train = os.path.join(out_root, "1cls_fs", name, "train", "good")
        os.makedirs(dst_train, exist_ok=True)
        for fname in list_images(src_train)[protocol["n_calibration"] :]:
            _make_link(os.path.join(src_train, fname), os.path.join(dst_train, fname), link)

        dst_calib = os.path.join(out_root, "calib", name)
        os.makedirs(dst_calib, exist_ok=True)
        for fname in info["calibration"]:
            _make_link(os.path.join(src_train, fname), os.path.join(dst_calib, fname), link)

        # test / ground truth are shared with the official split
        for sub in ("test", "ground_truth"):
            src = os.path.join(data_root, name, sub)
            if not os.path.isdir(src):
                continue
            for dirpath, _, filenames in os.walk(src):
                rel = os.path.relpath(dirpath, src)
                dst_dir = os.path.join(out_root, "1cls_fs", name, sub, rel)
                os.makedirs(dst_dir, exist_ok=True)
                for fname in filenames:
                    _make_link(
                        os.path.join(dirpath, fname), os.path.join(dst_dir, fname), link
                    )


def _make_link(src: str, dst: str, link: bool) -> None:
    if os.path.lexists(dst):
        return
    if link:
        os.symlink(os.path.abspath(src), dst)
    else:
        import shutil

        shutil.copy2(src, dst)
