"""Part 1 baseline: template difference ("classical normality modelling").

For a handful of defect-free reference images we compare every test image with
the *closest* reference and turn the pixel-wise difference into an anomaly map.
No learning, no training set beyond the references.

Design choices, all reported in the paper:
  * references and test images are resized to the same square resolution (448 px,
    matching the AnomalyDINO setting) before comparison;
  * optional ECC alignment (euclidean) removes small pose shifts, which is the
    dominant failure mode of a naive template difference;
  * the anomaly map is the pixel-wise maximum over four complementary channels
    (RGB difference, grey-level difference, gradient magnitude difference and
    1 - SSIM), which is then smoothed and morphologically cleaned;
  * the image score is the maximum of the cleaned map;
  * the OK/NG threshold is the 95th percentile of the scores of the 20 calibration
    images, which are excluded from the reference pool.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .metrics import image_auroc, pixel_auroc, unweighted_mean

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
DEFAULT_FEATURES: Tuple[str, ...] = ("rgb", "gray", "gradient", "ssim")


def list_images(folder: str) -> List[str]:
    return sorted(
        f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    )


def read_rgb(path: str, size: int) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def read_mask(path: Optional[str], size: int) -> np.ndarray:
    if path is None or not os.path.exists(path):
        return np.zeros((size, size), dtype=np.uint8)
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros((size, size), dtype=np.uint8)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


def _to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)


def _gradient(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _ssim_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    from skimage.metrics import structural_similarity

    _, ssim_img = structural_similarity(
        a,
        b,
        data_range=255,
        channel_axis=2,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
        full=True,
    )
    # multichannel input yields one SSIM map per channel
    if ssim_img.ndim == 3:
        ssim_img = ssim_img.mean(axis=2)
    return (1.0 - ssim_img).astype(np.float32)


@dataclass
class TemplateBaseline:
    """Nearest-template difference with a calibrated decision threshold."""

    size: int = 448
    features: Sequence[str] = DEFAULT_FEATURES
    smooth_sigma: float = 4.0
    morph_kernel: int = 3
    use_ecc: bool = False
    border_crop: int = 32
    score_percentile: float = 99.5
    ecc_max_iter: int = 50
    ref_images: List[np.ndarray] = field(default_factory=list)
    ref_paths: List[str] = field(default_factory=list)
    threshold: Optional[float] = None
    calibration_scores: List[float] = field(default_factory=list)

    # ---------------------------------------------------------------- fitting
    def fit(self, ref_paths: Sequence[str]) -> "TemplateBaseline":
        self.ref_paths = list(ref_paths)
        self.ref_images = [read_rgb(p, self.size) for p in self.ref_paths]
        return self

    def calibrate(self, calib_paths: Sequence[str]) -> float:
        """Kalibrate the OK/NG threshold on held-out normal images."""
        self.calibration_scores = [self.predict(p)[0] for p in calib_paths]
        self.threshold = float(np.percentile(self.calibration_scores, 95))
        return self.threshold

    # ------------------------------------------------------------- inference
    def predict(self, path: str, align: Optional[bool] = None) -> Tuple[float, np.ndarray]:
        image = read_rgb(path, self.size)
        # alignment (if enabled) happens inside _difference, once per reference
        refs = self.ref_images

        dists = [self._difference(image, ref) for ref in refs]
        anomaly_map = np.minimum.reduce(dists) if len(dists) > 1 else dists[0]
        anomaly_map = self._postprocess(anomaly_map)
        score = float(np.percentile(self.crop(anomaly_map), self.score_percentile))
        return score, anomaly_map

    # --------------------------------------------------------------- helpers
    def crop(self, array: np.ndarray) -> np.ndarray:
        """Drop the border, where resizing and SSIM edge effects dominate.

        The same crop is applied to anomaly maps and ground-truth masks so that
        pixel-level metrics never see the artefact-prone margin (and are not
        inflated by trivially correct border pixels).
        """
        b = self.border_crop
        if not b:
            return array
        h, w = array.shape[:2]
        return array[b : h - b, b : w - b]

    def _difference(self, image: np.ndarray, ref: np.ndarray) -> np.ndarray:
        if self.use_ecc:
            # align the reference to the test image (both already resized)
            ref = _align(ref, image, self.ecc_max_iter)
        test = image.astype(np.float32)
        ref_f = ref.astype(np.float32)
        channels: List[np.ndarray] = []

        if "rgb" in self.features:
            channels.append(np.abs(test - ref_f).mean(axis=2) / 255.0)
        if "gray" in self.features:
            channels.append(np.abs(_to_gray(image) - _to_gray(ref)) / 255.0)
        if "gradient" in self.features:
            gt, gr = _gradient(_to_gray(image)), _gradient(_to_gray(ref))
            scale = max(gt.max(), gr.max(), 1.0)
            channels.append(np.abs(gt - gr) / scale)
        if "ssim" in self.features:
            channels.append(_ssim_map(image, ref))

        if not channels:
            raise ValueError("no difference feature selected")
        return np.max(np.stack(channels, axis=0), axis=0).astype(np.float32)

    def _postprocess(self, anomaly_map: np.ndarray) -> np.ndarray:
        out = anomaly_map
        if self.smooth_sigma > 0:
            out = cv2.GaussianBlur(out, (0, 0), self.smooth_sigma)
        if self.morph_kernel > 1:
            kernel = np.ones((self.morph_kernel, self.morph_kernel), np.uint8)
            out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
            out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
        return np.clip(out, 0.0, None).astype(np.float32)


def _align(ref: np.ndarray, image: np.ndarray, max_iter: int) -> np.ndarray:
    """ECC euclidean alignment of ``ref`` onto ``image``; identity on failure."""
    g_ref = _to_gray(ref) / 255.0
    g_img = _to_gray(image) / 255.0
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, 1e-5)
    try:
        _, warp = cv2.findTransformECC(
            g_img, g_ref, warp, cv2.MOTION_EUCLIDEAN, criteria,
            inputMask=None, gaussFiltSize=5,
        )
    except cv2.error:
        return ref
    return cv2.warpAffine(
        ref, warp, (ref.shape[1], ref.shape[0]),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )


def evaluate_class(
    data_root: str,
    class_name: str,
    shot: int,
    seed: int,
    refs: Sequence[str],
    calibration: Sequence[str],
    size: int = 448,
    features: Sequence[str] = DEFAULT_FEATURES,
    use_ecc: bool = False,
    border_crop: int = 32,
    max_test_images: Optional[int] = None,
) -> Dict:
    """Run the baseline on one (class, shot, seed) condition and return metrics."""
    model = TemplateBaseline(
        size=size, features=features, use_ecc=use_ecc, border_crop=border_crop
    ).fit([os.path.join(data_root, class_name, "train", "good", f) for f in refs])
    model.calibrate(
        [os.path.join(data_root, class_name, "train", "good", f) for f in calibration]
    )

    test_good_dir = os.path.join(data_root, class_name, "test", "good")
    test_bad_dir = os.path.join(data_root, class_name, "test", "bad")
    gt_dir = os.path.join(data_root, class_name, "ground_truth", "bad")

    records: List[Dict] = []
    maps: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    scores: List[float] = []
    labels: List[int] = []
    times: List[float] = []

    for sub, label in ((test_good_dir, 0), (test_bad_dir, 1)):
        if not os.path.isdir(sub):
            continue
        files = list_images(sub)
        if max_test_images:
            files = files[:max_test_images]
        for fname in files:
            start = time.perf_counter()
            score, amap = model.predict(os.path.join(sub, fname))
            times.append(time.perf_counter() - start)

            gt_path = (
                os.path.join(gt_dir, os.path.splitext(fname)[0] + ".png") if label else None
            )
            maps.append(model.crop(amap))
            masks.append(model.crop(read_mask(gt_path, size)))
            scores.append(score)
            labels.append(label)
            records.append(
                {
                    "file": f"{'bad' if label else 'good'}/{fname}",
                    "score": round(score, 6),
                    "pred": int(model.threshold is not None and score > model.threshold),
                }
            )

    result = {
        "class": class_name,
        "shot": shot,
        "seed": seed,
        "references": list(refs),
        "calibration": list(calibration),
        "threshold": model.threshold,
        "image_auroc": image_auroc(scores, labels),
        "pixel_auroc": pixel_auroc(maps, masks),
        "mean_score": float(np.mean(scores)),
        "mean_time_s": float(np.mean(times)),
        "num_test": len(scores),
        "num_anomalous": int(np.sum(labels)),
        "records": records,
    }
    return result


def aggregate(results: Sequence[Dict]) -> Dict:
    """Aggregate per-class results into the reported means."""
    per_class_image = {r["class"]: r["image_auroc"] for r in results}
    per_class_pixel = {r["class"]: r["pixel_auroc"] for r in results}
    return {
        "shot": results[0]["shot"] if results else None,
        "seed": results[0]["seed"] if results else None,
        "image_auroc": per_class_image,
        "image_auroc_mean": unweighted_mean(per_class_image),
        "pixel_auroc": per_class_pixel,
        "pixel_auroc_mean": unweighted_mean(per_class_pixel),
    }
