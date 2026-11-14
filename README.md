# Few-Shot Industrial Visual Anomaly Detection

Course project for **AIAA 3201 Introduction to Computer Vision** (Spring 2027).
Given a handful of defect-free reference images, detect anomalous product images and localize defects.

Team: Zhang Yifei, Zhang Yiyang

## Task overview

| Part | Content | Status |
| --- | --- | --- |
| Part 1 | Mandatory baseline: template difference (classical normality modeling) | TODO |
| Part 2 | Reproduction: AnomalyDINO (DINOv2 ViT-S/14 patch features + memory bank) | in progress |
| Part 3 | One improvement on one failure mode, compared against an unchanged Part 2 | TODO |
| Optional | AnomalyCLIP, extra robustness tests | TODO |

## Dataset

**VisA**, 6 required classes: `candle`, `capsules`, `cashew`, `chewinggum`, `fryum`, `pipe_fryum`.
Official `1cls` split (normal-only training). All test images and masks are used for evaluation.

The dataset is **not** stored in this repository. On the course GPU server it is already available at
`/data/disk1/shared/aiaa3201/project2_anomaly/VisA` (read-only). Because the server's home partition is
full, we build the MVTec-style layout out of symlinks instead of copying:

```bash
python scripts/server/prep_visa.py     # -> VisA_pytorch/1cls/<class>/{train,test,ground_truth}
```

`scripts/server/prep_visa.py` mirrors the `1cls` split: `train/good`, `test/good`, `test/bad`,
`ground_truth/bad`. Layout expected by AnomalyDINO:

```
VisA_pytorch/1cls/<class>/train/good/*.JPG
VisA_pytorch/1cls/<class>/test/good/*.JPG
VisA_pytorch/1cls/<class>/test/bad/*.JPG
VisA_pytorch/1cls/<class>/ground_truth/bad/*.png
```

## Environment

Python 3.10, PyTorch 2.6.0 + CUDA 12.4 (RTX A6000), FAISS-GPU.

```bash
pip install -r requirements.txt
# torch is pinned to the cu124 wheels
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

`opencv-python` is pinned to 4.11: OpenCV 5 changes the rotation interpolation and shifts the reported
numbers away from the paper.

On the course server the environment lives in `/tmp/yzhang176/envs/ad` and can be rebuilt with
`scripts/server/setup_env.sh`. `scripts/server/run_ad.sh` wraps the common runs.

## Few-shot protocol (shared references)

AnomalyDINO samples references deterministically by index slicing of the sorted
normal training list: `sorted(train/good)[seed*shot : (seed+1)*shot]`.
The course protocol additionally reserves the **first 20 sorted normal images per
class** for threshold calibration and excludes them from every reference
selection. Slicing the full list would hand out calibration images, so both
methods draw from a shifted pool:

```
pool = sorted(train/good)[20:]
refs = pool[seed*shot : (seed+1)*shot]
```

`scripts/make_refs.py` records the exact file names in `configs/refs.json` and can
materialise two symlinked views, `1cls_fs/` (pool only) and `calib/`, so that the
unmodified AnomalyDINO training path stays inside the pool:

```bash
python scripts/make_refs.py --data-root /path/VisA_pytorch/1cls \
    --out configs/refs.json --materialise /path/VisA_pytorch
```

Then point `--data_root` at `/path/VisA_pytorch/1cls_fs`. Because the sampling
rule is unchanged, AnomalyDINO picks exactly the files listed in `refs.json`.

## Part 1 baseline: template difference

```bash
python scripts/run_baseline.py --data-root /path/VisA_pytorch/1cls \
    --protocol configs/refs.json --out results/baseline.json
```

Per (class, shot, seed) the baseline resizes everything to 448 px, optionally
aligns with ECC, and takes the pixel-wise maximum over four difference channels
(RGB, grey, gradient magnitude, `1 - SSIM`). The map is smoothed and
morphologically cleaned; the image score is its maximum, the decision threshold
is the 95th percentile of the 20 calibration images.

## Reproducing AnomalyDINO

Official code: <https://github.com/dammsi/AnomalyDINO> (WACV 2025).
We keep the upstream code unmodified apart from restricting `src/utils.py::get_dataset_info` to the six
required VisA classes.

```bash
python run_anomalydino.py --dataset VisA \
  --data_root /path/to/VisA_pytorch/1cls \
  --shots 1 2 4 8 --num_seeds 3 \
  --preprocess agnostic --resolution 448 \
  --faiss_on_cpu      # only when faiss-gpu is unavailable
```

Protocol: shots `1/2/4/8`, seeds `0/1/2`, `agnostic` preprocessing, 448 px, `dinov2_vits14`.

`--eval_segm` writes one full-resolution map per test image (~11 GB for these six classes) and
then computes AUPRO, which takes minutes. For the two required metrics we use an in-memory
evaluator that reuses the upstream building blocks and feeds scores/maps straight into
`src/metrics.py`:

```bash
python scripts/run_anomalydino_mem.py \
  --anomalydino-root /path/to/AnomalyDINO \
  --data-root /path/to/VisA_pytorch/1cls \
  --protocol configs/refs.json --out results/anomalydino.json
```

It also loads DINOv2 from the local torch-hub cache when GitHub is unreachable.

## Results

Protocol: shots `1/2/4/8` x seeds `0/1/2`, six VisA classes, unweighted mean over classes,
mean +- std over the three seeds. Both methods consume the reference images listed in
`configs/refs.json`, so the only difference is the method itself.

Image-level AUROC (%):

| Shots | Template difference | AnomalyDINO |
| --- | --- | --- |
| 1 | 57.44 +- 1.37 | 94.61 +- 0.84 |
| 2 | 63.43 +- 2.11 | 95.33 +- 0.46 |
| 4 | 60.92 +- 4.70 | 96.15 +- 0.40 |
| 8 | 61.06 +- 0.28 | 96.62 +- 0.57 |

Pixel-level AUROC (%):

| Shots | Template difference | AnomalyDINO |
| --- | --- | --- |
| 1 | 61.70 +- 0.44 | 98.08 +- 0.05 |
| 2 | 65.29 +- 0.62 | 98.22 +- 0.07 |
| 4 | 68.48 +- 0.39 | 98.36 +- 0.03 |
| 8 | 70.43 +- 0.43 | 98.47 +- 0.01 |

Per class, image AUROC (%) averaged over seeds:

| Class | Template 1-shot | AD 1-shot | Template 8-shot | AD 8-shot |
| --- | --- | --- | --- | --- |
| candle | 51.77 | 90.01 | 64.53 | 93.47 |
| capsules | 58.54 | 97.69 | 71.57 | 98.13 |
| cashew | 73.69 | 93.57 | 53.83 | 94.62 |
| chewinggum | 54.58 | 98.50 | 57.31 | 98.87 |
| fryum | 56.11 | 95.73 | 68.55 | 98.11 |
| pipe_fryum | 49.95 | 92.14 | 50.57 | 96.49 |

Observations:

- The template baseline stays close to chance on most classes and does **not** improve
  monotonically with more references (2-shot 63.43 > 4-shot 60.92, std 4.70 at 4 shots).
  Adding templates only helps if the extra images cover pose and lighting variation; when
  they do not, the per-pixel minimum over more templates mainly adds new "normal" appearances
  that are themselves dissimilar to the test object.
- Pixel-level AUROC of the baseline is below chance for `cashew` and `chewinggum`
  (38.23 / 36.38 at 1 shot): the largest differences sit on the object silhouette rather than
  on the defect, which is exactly the failure mode this baseline is meant to expose.
- AnomalyDINO's advantage is largest at 1 shot (94.61 vs 57.44) and its seed variance is an
  order of magnitude smaller, which is the stability the few-shot regime needs.

### Reproduction fidelity

Pointing the official `run_anomalydino.py` at the reference pool (`1cls_fs`, so that its index-based
sampling picks exactly the protocol's references) reproduces the in-memory evaluator on all six
classes, 1 shot, seed 0:

| Class | Official script | `run_anomalydino_mem.py` | Delta |
| --- | --- | --- | --- |
| candle | 90.36 | 90.36 | 0.000 |
| capsules | 98.78 | 98.78 | 0.000 |
| cashew | 92.90 | 92.90 | 0.000 |
| chewinggum | 98.26 | 98.26 | 0.000 |
| fryum | 94.52 | 94.52 | 0.000 |
| pipe_fryum | 92.88 | 92.88 | 0.000 |

### Efficiency (A6000, candle, 60 images after 20 warm-up)

| Shots | Memory bank | Median latency | Peak GPU |
| --- | --- | --- | --- |
| 1 | 0.8 s | 154.5 ms/img | 112.8 MB |
| 4 | 1.8 s | 143.2 ms/img | 112.8 MB |
| 8 | 2.8 s | 161.5 ms/img | 112.8 MB |

Resolution 448, batch 1. The latency includes reading, preprocessing, the forward pass, masked kNN,
`mean_top1p` and building the 448 px map; the upstream timer (76 ms/sample) excludes map
post-processing, so the two numbers are not directly comparable.

### Why the template baseline stays near chance

Three ablations, all at 1 shot / seed 0, six classes (image AUROC %):

| Aggregator over the map | Mean | | Difference channel | Mean |
| --- | --- | --- | --- | --- |
| 99.5th percentile (used) | 56.7 | | `1 - SSIM` | 56.5 |
| max | 53.4 | | RGB | 53.1 |
| mean of top 1% | 56.1 | | grey | 52.7 |
| 99.5th / median | 39.3 | | gradient | 50.4 |

No aggregator and no single channel rescues the method, so the loss is in the representation, not in
the scoring. Splitting by class reveals two different failure modes:

| Class | mean(map in GT) - mean(map outside GT) | per-image pixel AUROC | avg defect area |
| --- | --- | --- | --- |
| candle | +0.189 | 72.1 | 540 px |
| capsules | +0.081 | 68.5 | 1053 px |
| fryum | +0.230 | 81.4 | 6936 px |
| pipe_fryum | +0.521 | 93.9 | 3760 px |
| cashew | -0.185 | 23.9 | 3387 px |
| chewinggum | -0.105 | 38.3 | 1702 px |

For four classes the defect is highlighted locally but the *image-level* statistic is swamped by
global pose/background differences (exactly the failure the assignment predicts). For `cashew` and
`chewinggum` the map is inverted: the object silhouette differs more than the defect, so pixel-level
AUROC drops below chance.

The calibrated demo threshold also transfers poorly: the 95th percentile computed on the 20
calibration images yields a 20.6 % false-positive rate on the test split (60 % for `candle`), i.e. the
calibration and test score distributions differ. This is reported as a limitation and feeds the
reliability discussion, not hidden behind a tuned threshold.

### Demo

```bash
python scripts/demo.py --anomalydino-root /path/to/AnomalyDINO \
    --data-root /path/to/VisA_pytorch/1cls --protocol configs/refs.json \
    --class-name candle --shot 1 --seed 0 --image path/to/product.JPG --out demo_out
```

Prints the image score, the calibrated threshold and the OK/NG verdict, and writes a heatmap overlay,
the raw map and a `result.json`.

Raw outputs: `results/baseline_full.json`, `results/anomalydino_full.json`,
`results/efficiency.json` (not committed, regenerate with the scripts above).

## Repository layout

```
scripts/server/    environment bootstrap, VisA symlink preparation, run wrapper
src/               our code: baseline, improved method, shared evaluation
configs/           experiment configuration
docs/              figures and notes
```

## Notes on evaluation

- Image AUROC is the primary metric; report per class and the unweighted mean over six classes.
- Pixel AUROC is computed with all test images (including defect-free ones) and averaged per class.
- AUPRO is optional; if reported, the official 0.30 FPR cap must be used.
- `--eval_segm` writes full-resolution anomaly maps (~11 GB for the six-class set) and the AUPRO
  computation is slow; run pixel-level evaluation per class or with a streaming variant.

## References and attribution

Every dataset, pretrained model, external repository and borrowed implementation component used by
this repository is listed here. BibTeX keys refer to `references/Project2_references.bib`.

| Used for | Source | Key |
| --- | --- | --- |
| Dataset (VisA, official `1cls` split and masks) | Zou et al., *SPot-the-Difference*, ECCV 2022 | `zou2022spotdiff` |
| Data layout (`train/good`, `test/<type>`, `ground_truth/<type>`) | Bergmann et al., *MVTec AD*, CVPR 2019 | `bergmann2019mvtec` |
| Part 2 method and reference-sampling rule | Damm et al., *AnomalyDINO*, WACV 2025 — <https://github.com/dammsi/AnomalyDINO> | `damm2025anomalydino` |
| Backbone features (`dinov2_vits14`) | Oquab et al., *DINOv2*, TMLR 2024 — <https://github.com/facebookresearch/dinov2> | `oquab2024dinov2` |
| Part 1 `1 - SSIM` difference channel | Wang et al., *Image Quality Assessment: From Error Visibility to Structural Similarity*, IEEE TIP 2004 | `wang2004ssim` |
| Part 1 optional ECC alignment | Evangelidis & Psarakis, *Parametric Image Alignment Using Enhanced Correlation Coefficient Maximization*, IEEE TPAMI 2008 | `evangelidis2008ecc` |
| AUROC / AP implementations | scikit-learn `sklearn.metrics` | — |
| Image I/O, resizing, morphology, ECC | OpenCV `cv2` | — |
| SSIM implementation | scikit-image `skimage.metrics.structural_similarity` | — |

The upstream AnomalyDINO code is used unmodified except for restricting the VisA class list in
`src/utils.py::get_dataset_info` to the six required classes, as noted in that repository's README.

A complete related-work discussion citing all papers of the course reading list belongs to the
report, not to this file.
