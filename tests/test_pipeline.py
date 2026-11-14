"""End-to-end smoke test on a synthetic dataset (no VisA download required).

    pytest -q tests/test_pipeline.py
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baseline import evaluate_class  # noqa: E402
from src.metrics import image_auroc, pixel_auroc  # noqa: E402
from src.splits import build_protocol  # noqa: E402

SIZE = 64
CLASS = "candle"


def _background(rng: np.random.Generator) -> np.ndarray:
    base = np.full((SIZE, SIZE, 3), 128, dtype=np.float32)
    base += rng.normal(0, 4, base.shape).astype(np.float32)
    return np.clip(base, 0, 255).astype(np.uint8)


def _write(path: str, img: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


@pytest.fixture()
def synthetic_root(tmp_path):
    rng = np.random.default_rng(0)
    root = tmp_path / "1cls"
    cls_dir = root / CLASS

    # 20 calibration + 12 pool images
    for i in range(32):
        _write(str(cls_dir / "train" / "good" / f"{i:03d}.png"), _background(rng))
    # defect-free test images
    for i in range(6):
        _write(str(cls_dir / "test" / "good" / f"g{i:03d}.png"), _background(rng))
    # anomalous test images: a bright scratch pattern plus mask
    for i in range(6):
        img = _background(rng)
        mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
        mask[20:44, 30:34] = 255
        img[mask > 0] = np.array([10, 10, 250], dtype=np.uint8)
        _write(str(cls_dir / "test" / "bad" / f"b{i:03d}.png"), img)
        _write(str(cls_dir / "ground_truth" / "bad" / f"b{i:03d}.png"), mask)

    return str(root)


def test_protocol_excludes_calibration(synthetic_root):
    protocol = build_protocol(
        synthetic_root, classes=[CLASS], shots=[1, 2, 4], seeds=[0, 1], n_calibration=20
    )
    info = protocol["classes"][CLASS]
    assert info["n_train"] == 32
    assert len(info["calibration"]) == 20
    assert info["pool_size"] == 12
    refs = info["refs"]["2shot_seed1"]
    assert refs == sorted(os.listdir(os.path.join(synthetic_root, CLASS, "train", "good")))[20:][2:4]
    assert not set(refs) & set(info["calibration"])


def test_baseline_separates_synthetic_defects(synthetic_root):
    protocol = build_protocol(
        synthetic_root, classes=[CLASS], shots=[1], seeds=[0], n_calibration=20
    )
    info = protocol["classes"][CLASS]
    result = evaluate_class(
        synthetic_root,
        CLASS,
        shot=1,
        seed=0,
        refs=info["refs"]["1shot_seed0"],
        calibration=info["calibration"],
        size=SIZE,
        use_ecc=False,
    )
    assert result["num_test"] == 12
    assert result["num_anomalous"] == 6
    assert result["image_auroc"] > 0.9
    assert result["pixel_auroc"] > 0.8
    assert result["threshold"] is not None


def test_metrics_are_sane():
    assert image_auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    m1 = np.zeros((8, 8), dtype=np.float32)
    m1[2:4, 2:4] = 1.0
    gt1 = np.zeros((8, 8), dtype=np.uint8)
    gt1[2:4, 2:4] = 1
    assert pixel_auroc([m1], [gt1]) == 1.0
