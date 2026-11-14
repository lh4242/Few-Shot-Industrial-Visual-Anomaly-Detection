"""Evaluation metrics used by every method in this repository.

Course protocol:
  * image level  - AUROC on the raw image scores (before thresholding), reported
    per class and as the unweighted mean over the six classes;
  * pixel level  - AUROC over all test images, including defect-free ones whose
    ground-truth mask is empty; computed per class, then averaged with equal
    class weights.

Both metrics are implemented here so that the template baseline, the
AnomalyDINO reproduction and the improved method go through exactly the same
code.

Attribution:
  * AUROC / AP are taken from ``sklearn.metrics`` (no reimplementation);
  * the metric definition and the MVTec-style evaluation convention follow
    Bergmann et al., "MVTec AD", CVPR 2019 (bib key ``bergmann2019mvtec``);
  * the VisA protocol (all test images, including defect-free ones, are used
    for pixel-level evaluation) follows Zou et al., ECCV 2022
    (bib key ``zou2022spotdiff``).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def image_auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """AUROC for anomaly detection; labels are 1 for anomalous images."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(labels, dtype=np.int64)
    if y.min() == y.max():
        raise ValueError("image AUROC undefined: labels contain a single class")
    return float(roc_auc_score(y, np.asarray(scores, dtype=np.float64)))


def pixel_auroc(maps: Iterable[np.ndarray], masks: Iterable[np.ndarray]) -> float:
    """AUROC over all pixels of a class.

    Maps are flattened as-is, masks are binarised at >0. Defect-free images must
    be passed with an all-zero mask.
    """
    from sklearn.metrics import roc_auc_score

    preds: List[np.ndarray] = []
    gts: List[np.ndarray] = []
    for m, gt in zip(maps, masks):
        if m.shape != gt.shape:
            raise ValueError(f"prediction/mask shape mismatch: {m.shape} vs {gt.shape}")
        preds.append(np.asarray(m, dtype=np.float32).ravel())
        gts.append((np.asarray(gt) > 0).astype(np.uint8).ravel())

    pred = np.concatenate(preds)
    gt = np.concatenate(gts)
    if gt.min() == gt.max():
        raise ValueError("pixel AUROC undefined: masks contain a single class")
    return float(roc_auc_score(gt, pred))


def pixel_average_precision(maps: Iterable[np.ndarray], masks: Iterable[np.ndarray]) -> float:
    """Optional AP, provided for completeness (not part of the required table)."""
    from sklearn.metrics import average_precision_score

    pred = np.concatenate([np.asarray(m, dtype=np.float32).ravel() for m in maps])
    gt = np.concatenate([(np.asarray(g) > 0).astype(np.uint8).ravel() for g in masks])
    return float(average_precision_score(gt, pred))


def unweighted_mean(per_class: Dict[str, float]) -> float:
    """Equal weight per class, as required by the protocol."""
    if not per_class:
        raise ValueError("empty per-class result")
    return float(np.mean(list(per_class.values())))


def mean_std_over_seeds(values: Iterable[float]) -> Tuple[float, float]:
    """Mean and (population) standard deviation over seeds."""
    arr = np.asarray(list(values), dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def summarise_image_level(
    scores: Dict[str, List[float]],
    labels: Dict[str, List[int]],
) -> Dict[str, float]:
    """Per-class image AUROC plus the unweighted mean over classes."""
    per_class = {c: image_auroc(scores[c], labels[c]) for c in scores}
    per_class["mean"] = unweighted_mean(per_class)
    return per_class
