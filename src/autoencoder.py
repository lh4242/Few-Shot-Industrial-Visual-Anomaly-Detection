"""Optional Part 1 extension: compact convolutional autoencoder.

This is *not* a few-shot method: the autoencoder is trained on normal training
images of the same class and reconstruction error is used as the anomaly map.
The number of training images is therefore reported separately from the 1/2/4/8
reference images and must not be compared with the few-shot rows without saying
so.

Design:
  * input 256x256 (the 448 px used elsewhere is unnecessarily expensive for a
    small autoencoder; maps are resampled to the evaluation size before scoring);
  * three stride-2 convolutions, mirrored decoder, MSE loss;
  * the calibration images reserved by the protocol are excluded from training,
    so the OK/NG threshold is still measured on unseen normal images.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn

from .metrics import image_auroc, pixel_auroc

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_images(folder: str) -> List[str]:
    return sorted(
        f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    )


class ConvAutoencoder(nn.Module):
    def __init__(self, base: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, base, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(base, base * 2, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1), nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def read_image(path: str, size: int) -> np.ndarray:
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


def train_autoencoder(
    paths: Sequence[str],
    size: int = 256,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 2e-3,
    base: int = 32,
    device: str = "cuda",
    seed: int = 0,
) -> ConvAutoencoder:
    torch.manual_seed(seed)
    model = ConvAutoencoder(base=base).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    data = np.stack([read_image(p, size) for p in paths]).astype(np.float32) / 255.0
    tensor = torch.from_numpy(data).permute(0, 3, 1, 2).contiguous()
    n = tensor.shape[0]

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        for start in range(0, n, batch_size):
            batch = tensor[perm[start : start + batch_size]].to(device)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            total += loss.item() * batch.shape[0]
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch + 1:3d}/{epochs}  mse {total / n:.5f}", flush=True)
    model.eval()
    return model


@torch.inference_mode()
def reconstruction_map(model: ConvAutoencoder, path: str, size: int, device: str) -> np.ndarray:
    image = read_image(path, size).astype(np.float32) / 255.0
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
    recon = model(tensor)[0].permute(1, 2, 0).cpu().numpy()
    error = np.abs(image - recon).mean(axis=2)
    return cv2.GaussianBlur(error, (0, 0), 2.0)


def evaluate_class(
    data_root: str,
    class_name: str,
    calibration: Sequence[str],
    train_count: int = 200,
    size: int = 256,
    eval_size: int = 448,
    epochs: int = 30,
    device: str = "cuda",
    seed: int = 0,
) -> Dict:
    train_dir = os.path.join(data_root, class_name, "train", "good")
    calib_set = set(calibration)
    pool = [f for f in list_images(train_dir) if f not in calib_set]
    chosen = pool[:train_count]
    paths = [os.path.join(train_dir, f) for f in chosen]

    start = time.perf_counter()
    model = train_autoencoder(paths, size=size, epochs=epochs, device=device, seed=seed)
    train_time = time.perf_counter() - start

    def score_and_map(path: str):
        amap = reconstruction_map(model, path, size, device)
        amap = cv2.resize(amap, (eval_size, eval_size), interpolation=cv2.INTER_LINEAR)
        return float(np.percentile(amap, 99.5)), amap

    calib_scores = [score_and_map(os.path.join(train_dir, f))[0] for f in calibration]
    threshold = float(np.percentile(calib_scores, 95))

    scores, labels, maps, masks, times, records = [], [], [], [], [], []
    for sub, label in (("test/good", 0), ("test/bad", 1)):
        folder = os.path.join(data_root, class_name, sub)
        if not os.path.isdir(folder):
            continue
        for fname in list_images(folder):
            t0 = time.perf_counter()
            score, amap = score_and_map(os.path.join(folder, fname))
            times.append(time.perf_counter() - t0)
            gt = (
                os.path.join(
                    data_root, class_name, "ground_truth", "bad",
                    os.path.splitext(fname)[0] + ".png",
                )
                if label
                else None
            )
            scores.append(score)
            labels.append(label)
            maps.append(amap)
            masks.append(read_mask(gt, eval_size))
            records.append(
                {
                    "file": f"{'bad' if label else 'good'}/{fname}",
                    "score": round(score, 6),
                    "pred": int(score > threshold),
                }
            )

    return {
        "class": class_name,
        "method": "autoencoder",
        "train_images": len(chosen),
        "shots": None,
        "seed": seed,
        "calibration": list(calibration),
        "threshold": threshold,
        "image_auroc": image_auroc(scores, labels),
        "pixel_auroc": pixel_auroc(maps, masks),
        "mean_time_s": float(np.mean(times)),
        "train_time_s": train_time,
        "num_test": len(scores),
        "num_anomalous": int(np.sum(labels)),
        "records": records,
    }
