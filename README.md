# Fisheye intrinsics regression — CNN vs ViT

Estimate a fisheye camera's intrinsics from **one** image, with no calibration
target: a 224×224 fisheye view goes in, the Unified Camera Model parameters
`(f, ξ)` come out.

The point of the project is the **methodology** — how the data is built, how
the loss is defined, how the split avoids leakage, and how results are
measured against an honest baseline — rather than the network, which is
`timm.create_model(...)` off the shelf for both arms of the comparison.

Training data is rendered from [Poly Haven](https://polyhaven.com/) HDRI
panoramas (see [Credits](#credits)).

---

## The problem

The [Unified Camera Model](https://doi.org/10.1109/ROBOT.2007.364084) projects
a ray at angle `θ` from the optical axis to image radius

```
r(θ) = f · sin θ / (ξ + cos θ)
```

`ξ` controls how the projection bends: `ξ = 0` is a pinhole (`r = f·tan θ`),
`ξ = 1` is stereographic, and larger `ξ` reaches past 180°. The principal
point is fixed at the image centre, so the model has two unknowns.

Two things make this harder than it looks:

- **The projection has a validity boundary.** Beyond `θ_max(ξ) = arccos(−min(ξ, 1/ξ))`
  the radius stops increasing and two directions land on the same pixel. A
  network predicting a wrong `ξ` walks straight out of the valid region, and
  the loss has to cope with that.
- **`f` and `ξ` are partly degenerate.** A larger `ξ` paired with a larger `f`
  mimics the same radial profile. Scoring on parameter error alone is
  misleading; this project scores on **reprojection error** instead.

## Approach

### Data

997 CC0 HDRI panoramas are downloaded, tone-mapped and rendered into fisheye
views with randomly sampled cameras.

- **Tone mapping** is Reinhard 2002 with a `cos(latitude)`-weighted log-average
  luminance. The weighting matters: an equirectangular panorama massively
  over-samples the poles, so an unweighted average lets a bright sky dictate
  the exposure of the whole scene.
- **Camera sampling** draws `(FOV, ξ)` uniformly and rejects anything outside
  the feasible region — including a cap on edge magnification
  `M = (1+ξ)(1+ξcos θ)/(ξ+cos θ)²`, which keeps out views whose rim is smeared
  across half the frame while carrying almost no information. (`M = 4` is a
  120° rectilinear lens.)
- **Rendering** supersamples 4× and reduces with `INTER_AREA`; at 224 px
  covering up to 220°, point sampling aliases badly.

### Splitting

Poly Haven ships many variants of the same capture — the same location at
several times of day, `_puresky` versions, numbered series. Splitting per
asset would put the same scene in both train and test, so assets are grouped
first (by collection tag **or** by filename stem, taking the connected
components of the two relations) and whole groups are assigned to a split,
stratified by scene category.

| split | assets | rendered views |
|---|---:|---:|
| train | 755 | 22,650 |
| val | 94 | 1,880 |
| test | 89 | 1,780 |
| stress (`Pure Skies`) | 59 | 1,180 |

706 groups in total. Nested subsets `p10 ⊂ p30 ⊂ p100` (84 / 226 / 755 assets)
support data-scaling experiments.

### Loss

The reprojection error between two parameter sets reduces to a 1-D comparison
of radial profiles, since the principal point is shared and the model is
radially symmetric:

```
loss = radial + barrier
```

- **radial** — smooth L1 between `r_pred(θ)` and `r_gt(θ)`, over the valid
  angles only.
- **barrier** — a hinge on the two quantities that must stay positive,
  `d1 = ξ + cos θ` and `d2 = 1 + ξ cos θ`.

The barrier is not decoration. Masking the invalid angles removes their
gradient, and clamping the denominator instead would be no better: a clamp is
flat, so its gradient is **zero exactly where a correcting signal is needed**.
The hinge is the part with a slope pointing back into the valid region.

## Results so far

**These numbers are still moving — the learning-rate sweep is not finished.**

Everything is measured as mean reprojection error on val, in pixels.

The floor is a **constant predictor** that never looks at the image. The weak
version answers with the train mean; the strong version grid-searches the
single `(f, ξ)` that minimises train reprojection error — which is *not* the
mean, precisely because of the `f`/`ξ` degeneracy.

| | val reproj (mean) |
|---|---:|
| constant — train mean params | 17.11 px |
| **constant — best single `(f, ξ)`** | **14.56 px** ← the number to beat |
| ResNet-34, lr 1e-4 | 5.56 px |
| ResNet-34, lr 3e-4 | 4.98 px |
| ResNet-34, lr 1e-3 | **4.32 px** |

For scale, 5 px on the median val camera is roughly an 8% focal-length error,
or a `ξ` off by 0.13.

A single-batch overfit test reaches 0.26 px with ResNet-34, so capacity is
not the constraint. But in the real runs the train and validation losses sit
almost on top of each other (about 0.019 vs 0.020), which makes this **high
bias — underfitting — rather than a generalisation gap**. The remaining error
is therefore either undertraining, or samples that are genuinely unreadable:
low-texture frames with no geometric cue, and the `f`/`ξ` degeneracy. The
learning-rate sweep and longer runs are what separate those two.

ViT-S/16 is unstable at the ResNet's learning rate — an overfit run swung
between 0.006 and 0.056 and finished 54× short — so the two architectures get
their own optimiser settings rather than a shared one. Comparing a tuned CNN
against a detuned ViT would measure the tuning, not the architecture.

## Running it

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). A CUDA GPU is
needed for training; everything else runs on CPU.

```bash
uv sync
```

The torch wheel comes from PyTorch's CUDA index (pinned in `pyproject.toml`);
pick the index matching your driver — check `nvidia-smi` first.

```bash
# 1. fetch the panoramas (~6.5 GB for all 997 at 2k)
uv run python -m data.download_polyhaven --out data_root/raw/polyhaven --res 2k

# 2. group-aware stratified split
uv run python -m data.split --manifest data_root/manifest.json \
                            --out data_root/splits.json --seed 0

# 3. render each split
uv run python -m data.render_dataset --config configs/data.yaml --split train --workers 8

# 4. sanity-check the result by eye
uv run python -m tools.visualize_samples --root data_root --split train

# 5. establish the floor, then train
uv run python -m tools.baseline --root data_root
uv run python train.py --config configs/train/resnet34_pre_p100_geo_s0.yaml
```

Diagnostics before committing to a long run:

```bash
uv run python train.py --config <cfg> --overfit-batch 300   # can it fit one batch?
uv run python train.py --config <cfg> --benchmark 100       # img/s and peak VRAM
uv run python sweep.py configs/sweeps/resnet34_lr.txt       # resumable, fault-tolerant
```

Metrics go to a local MLflow SQLite store:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Layout

```
geometry/     UCM projection, rotations, equirectangular sampling, rendering (numpy)
data/         download, split, camera sampling, offline rendering
datasets/     torch Datasets, augmentation, epoch-carrying sampler
metrics/      differentiable reprojection error and losses (torch)
tools/        category report, sample visualisation, constant baselines
train.py      training loop (shared by CNN and ViT)
sweep.py      sequential config runner
configs/      data generation and per-run training configs
tests/        478 tests
```

## Testing

```bash
uv run python -m pytest
```

The suite is the project's main guard against silently-wrong geometry. It
checks the things that fail quietly rather than loudly — round-trips through
projection and unprojection, the `θ_max` validity boundary, supersampling
alignment to sub-pixel accuracy, rendering verified against a
direction-colour-coded panorama, bit-identical output across worker counts,
no group spanning two splits, and augmentations that provably leave the label
untouched.

## Credits

Panoramas are HDRIs from **[Poly Haven](https://polyhaven.com/)**, released
under **[CC0](https://creativecommons.org/publicdomain/zero/1.0/)**. Thanks to
Poly Haven and its contributing artists for making them freely available —
this project would not exist without them.

No panorama or rendered image is redistributed here; `data/download_polyhaven.py`
fetches them from the Poly Haven API at the resolution you ask for.
