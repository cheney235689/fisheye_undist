"""Final evaluation and stratified analysis of the selected runs (SPEC.md 1.4.1).

    uv run python evaluate.py --config configs/eval/final.yaml            # val, train_clean
    uv run python evaluate.py --config configs/eval/final.yaml --final    # + test, stress

**This is the only program in the project allowed to read test or stress**,
and it does so only with ``--final``.  Without it, the analyses that belong to
test -- the paired comparison, the stratification, the worst-case figures --
run on val instead and are labelled as a preview.  That is deliberate: the
test evaluation is meant to happen once, so the analysis code should already
have run end to end on real data before the held-out set is spent.  Val was
used to pick the learning rates and checkpoints, so those preview numbers are
optimistic; they validate the code, not the conclusions.

Every model output goes through the same steps ``train.evaluate`` uses --
``_to_device``, the run's own autocast, ``denormalize`` and
``metrics.geometric.reprojection_error_px`` -- and each run is checked against
its MLflow record before anything else is reported.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from datasets.fisheye_dataset import FisheyeDataset, load_norm_stats
from geometry import ucm
from geometry.render import FisheyeCamera, fisheye_to_perspective
from metrics.geometric import denormalize, reprojection_error_px
from train import (
    DEFAULT_TRACKING_URI,
    PROJECT_ROOT,
    DataConfig,
    OptimConfig,
    TrainConfig,
    _to_device,
    autocast,
    build_model,
    release_loaders,
    uses_channels_last,
)

__all__ = [
    "HELD_OUT_SPLITS",
    "EvalConfig",
    "load_eval_config",
    "evaluated_splits",
    "open_split",
    "load_run",
    "predict",
    "cluster_weights",
    "panorama_bootstrap",
    "FOV_EDGES",
    "fov_bin",
    "texture_bins",
    "quantile_bins",
    "valid_mask",
    "line_scores",
    "predicted_fov",
    "fov_bias",
    "run",
    "main",
]

HELD_OUT_SPLITS = ("test", "stress")
BASE_SPLITS = ("val", "train_clean")
EXPERIMENT_NAME = "fisheye-calib"  # train.py logs into this experiment
SANITY_TOLERANCE_PX = 0.01

# SPEC.md task spec: left-closed bins, the last one closed on both sides so a
# camera at exactly 220 deg lands in it.
FOV_EDGES = (100.0, 130.0, 160.0, 190.0, 220.0)
LOW_TEXTURE = 5.0

BASELINE_MEAN = "baseline_mean_params"
BASELINE_BEST = "baseline_best_constant"

PER_SAMPLE_FIELDS = [
    "index", "asset_id", "view_idx", "f", "xi", "fov_deg", "circular", "texture",
    "line_score", "line_score_all",
    "category", "pred_f", "pred_xi", "reproj_px", "reproj_rel_pct", "invalid_frac",
]

# Line score (SPEC.md 3.14.3 item 5).  Chosen before looking at any score and
# not tuned: they only have to separate "has straight edges that curve under
# the distortion" from "does not".
LINE_MIN_LENGTH_PX = 8.0
LINE_MIN_RADIUS = 0.2       # x r_ref: near the centre every line looks straight
LINE_MIN_ANGLE_DEG = 30.0   # to the radial direction: radial lines stay straight
# Same erosion as data.render_dataset.texture_score: the black fill outside the
# aperture would otherwise be detected as segments along the rim.
LINE_MASK_ERODE_ITERATIONS = 2
DEFAULT_SUPERSAMPLE = 4

FOV_BIAS_EDGES = tuple(float(x) for x in range(100, 221, 10))


# --- config -----------------------------------------------------------------


@dataclass
class EvalConfig:
    data_root: str = "data_root"
    checkpoints_dir: str = "checkpoints"
    groups: dict[str, list[str]] = field(default_factory=dict)
    comparison: tuple[str, str] | None = None
    train_clean_n: int = 2000
    train_clean_seed: int = 0
    bootstrap_n: int = 1000
    bootstrap_seed: int = 0
    baseline_grid: int = 100
    baseline_xi_max: float = 3.0
    batch_size: int | None = None
    num_workers: int = 0
    worst_k: int = 8
    perspective_fov_deg: float = 100.0
    out_dir: str = "results"
    tracking_uri: str = DEFAULT_TRACKING_URI

    def runs(self) -> list[str]:
        return [run for members in self.groups.values() for run in members]


def load_eval_config(path: str | Path) -> EvalConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")

    groups = raw.get("groups") or {}
    if not groups:
        raise ValueError(f"{path}: no groups to evaluate")
    groups = {str(k): [str(r) for r in v] for k, v in groups.items()}
    seen: set[str] = set()
    for members in groups.values():
        for run_name in members:
            if run_name in seen:
                raise ValueError(f"{path}: run {run_name!r} listed twice")
            seen.add(run_name)

    comparison = raw.get("comparison")
    if comparison is not None:
        if len(comparison) != 2 or any(g not in groups for g in comparison):
            raise ValueError(f"{path}: comparison must name two of {sorted(groups)}")
        comparison = (str(comparison[0]), str(comparison[1]))

    train_clean = raw.get("train_clean") or {}
    bootstrap = raw.get("bootstrap") or {}
    baselines = raw.get("baselines") or {}
    return EvalConfig(
        data_root=str(raw.get("data_root", "data_root")),
        checkpoints_dir=str(raw.get("checkpoints_dir", "checkpoints")),
        groups=groups,
        comparison=comparison,
        train_clean_n=int(train_clean.get("n", 2000)),
        train_clean_seed=int(train_clean.get("seed", 0)),
        bootstrap_n=int(bootstrap.get("n", 1000)),
        bootstrap_seed=int(bootstrap.get("seed", 0)),
        baseline_grid=int(baselines.get("grid", 100)),
        baseline_xi_max=float(baselines.get("xi_max", 3.0)),
        batch_size=raw.get("batch_size"),
        num_workers=int(raw.get("num_workers", 0)),
        worst_k=int(raw.get("worst_k", 8)),
        perspective_fov_deg=float(raw.get("perspective_fov_deg", 100.0)),
        out_dir=str(raw.get("out_dir", "results")),
        tracking_uri=str(raw.get("tracking_uri", DEFAULT_TRACKING_URI)),
    )


# --- splits -----------------------------------------------------------------


def evaluated_splits(final: bool) -> list[str]:
    """val and train_clean always; test and stress only with ``--final``."""
    return list(BASE_SPLITS) + (list(HELD_OUT_SPLITS) if final else [])


@dataclass
class SplitData:
    name: str
    dataset: FisheyeDataset
    positions: list[int]          # positions into dataset.rows
    rows: list[dict[str, str]]    # one per evaluated sample, in order

    @property
    def n(self) -> int:
        return len(self.positions)

    def column(self, key: str) -> np.ndarray:
        return np.array([float(r[key]) for r in self.rows], dtype=np.float64)

    def strings(self, key: str) -> list[str]:
        return [r[key] for r in self.rows]

    def flags(self, key: str) -> np.ndarray:
        return np.array([r[key] in ("True", "true", "1") for r in self.rows])


def open_split(
    cfg: EvalConfig, name: str, final: bool, splits_doc: Mapping[str, Any] | None = None
) -> SplitData:
    """Open one evaluation split; refuses the held-out ones without ``--final``.

    This guard is the single point where test and stress can be reached.
    """
    if name in HELD_OUT_SPLITS and not final:
        raise PermissionError(
            f"split {name!r} is held out; evaluate.py reads it only with --final"
        )
    if name not in BASE_SPLITS + HELD_OUT_SPLITS:
        raise ValueError(f"unknown evaluation split {name!r}")

    root = Path(cfg.data_root)
    if name == "train_clean":
        # No transform: this measures fit, so the inputs must be the clean
        # renders, not augmented ones.
        dataset = FisheyeDataset(root, "train", transform=None)
        count = min(cfg.train_clean_n, len(dataset))
        rng = np.random.default_rng(cfg.train_clean_seed)
        positions = sorted(int(i) for i in rng.choice(len(dataset), count, replace=False))
    else:
        dataset = FisheyeDataset(root, name, transform=None)
        positions = list(range(len(dataset)))

    rows = [dict(dataset.rows[p]) for p in positions]
    if name == "stress":
        sibling = (splits_doc or {}).get("stress_sibling_split", {})
        for row in rows:
            value = sibling.get(row["asset_id"])
            row["stress_sibling_split"] = "null" if value is None else str(value)
    return SplitData(name, dataset, positions, rows)


# --- runs -------------------------------------------------------------------


def config_from_flat(flat: Mapping[str, Any]) -> TrainConfig:
    """Rebuild a TrainConfig from the flattened copy stored in the checkpoint."""
    top: dict[str, Any] = {}
    data: dict[str, Any] = {}
    optim: dict[str, Any] = {}
    for key, value in flat.items():
        if key.startswith("data."):
            data[key[5:]] = value
        elif key.startswith("optim."):
            optim[key[6:]] = value
        else:
            top[key] = value
    return TrainConfig(data=DataConfig(**data), optim=OptimConfig(**optim), **top)


def _same_stats(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    return (
        list(a.get("fields", [])) == list(b.get("fields", []))
        and np.allclose(a["mean"], b["mean"], rtol=0.0, atol=1e-12)
        and np.allclose(a["std"], b["std"], rtol=0.0, atol=1e-12)
    )


def load_run(
    run_name: str, cfg: EvalConfig, norm_stats: Mapping[str, Any], device: torch.device
) -> tuple[torch.nn.Module, TrainConfig, dict[str, Any]]:
    """Load ``<run_name>_best.pt`` and refuse it if its statistics are stale."""
    path = Path(cfg.checkpoints_dir) / f"{run_name}_best.pt"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found")
    payload = torch.load(path, map_location="cpu", weights_only=False)

    stored = payload.get("norm_stats")
    if not stored or not _same_stats(stored, norm_stats):
        raise ValueError(
            f"{run_name}: the checkpoint's norm_stats do not match the train "
            "split's meta.json -- the dataset was re-rendered after this run, "
            "so its predictions would be de-standardised wrongly"
        )

    train_cfg = config_from_flat(payload["config"])
    # The weights come from the checkpoint; there is nothing to download.
    train_cfg.pretrained = False
    model = build_model(train_cfg, device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, train_cfg, payload


def mlflow_best(run_name: str, tracking_uri: str) -> float:
    """``best_val_reproj_mean_px`` of the latest FINISHED run with this name."""
    import mlflow

    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise LookupError(f"no MLflow experiment {EXPERIMENT_NAME!r} at {tracking_uri}")
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=(
            f"attributes.run_name = '{run_name}' and attributes.status = 'FINISHED'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise LookupError(f"{run_name}: no FINISHED MLflow run")
    value = runs[0].data.metrics.get("best_val_reproj_mean_px")
    if value is None:
        raise LookupError(f"{run_name}: MLflow run has no best_val_reproj_mean_px")
    return float(value)


# --- inference --------------------------------------------------------------


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    train_cfg: TrainConfig,
    split: SplitData,
    norm_stats: Mapping[str, Any],
    device: torch.device,
    batch_size: int,
    num_workers: int = 0,
) -> dict[str, np.ndarray]:
    """Per-sample predictions and errors, computed exactly as train.evaluate does."""
    loader = DataLoader(
        Subset(split.dataset, split.positions),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    channels_last = uses_channels_last(train_cfg.model)
    model.eval()

    parts: dict[str, list[np.ndarray]] = {k: [] for k in ("pred_f", "pred_xi", "reproj", "invalid")}
    try:
        for batch in loader:
            image, _target, f_gt, xi_gt, theta_ref = _to_device(batch, device, channels_last)
            with autocast(train_cfg, device):
                pred = model(image)
            pred = pred.float()
            f_pred, xi_pred = denormalize(pred, norm_stats)
            error, invalid = reprojection_error_px(f_pred, xi_pred, f_gt, xi_gt, theta_ref)
            parts["pred_f"].append(f_pred.cpu().numpy())
            parts["pred_xi"].append(xi_pred.cpu().numpy())
            parts["reproj"].append(error.cpu().numpy())
            parts["invalid"].append(invalid.cpu().numpy())
    finally:
        release_loaders(loader)

    return {k: np.concatenate(v).astype(np.float64) for k, v in parts.items()}


def constant_predictions(split: SplitData, f: float, xi: float) -> dict[str, np.ndarray]:
    """A constant predictor's per-sample errors, through the same metric."""
    f_gt = torch.tensor(split.column("f"), dtype=torch.float32)
    xi_gt = torch.tensor(split.column("xi"), dtype=torch.float32)
    theta_ref = torch.tensor(np.radians(split.column("fov_deg")) / 2.0, dtype=torch.float32)
    error, invalid = reprojection_error_px(
        torch.full_like(f_gt, float(f)), torch.full_like(xi_gt, float(xi)),
        f_gt, xi_gt, theta_ref,
    )
    return {
        "pred_f": np.full(split.n, float(f)),
        "pred_xi": np.full(split.n, float(xi)),
        "reproj": error.numpy().astype(np.float64),
        "invalid": invalid.numpy().astype(np.float64),
    }


def write_per_sample(path: Path, split: SplitData, result: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(PER_SAMPLE_FIELDS)
    if split.name == "stress":
        fields.append("stress_sibling_split")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for i, row in enumerate(split.rows):
            writer.writerow(
                {
                    **row,
                    "pred_f": f"{result['pred_f'][i]:.6f}",
                    "pred_xi": f"{result['pred_xi'][i]:.6f}",
                    "reproj_px": f"{result['reproj'][i]:.6f}",
                    "reproj_rel_pct": f"{result['rel'][i]:.6f}",
                    "invalid_frac": f"{result['invalid'][i]:.6f}",
                }
            )


# --- line score -------------------------------------------------------------


def _supersample(split: SplitData) -> int:
    """The supersample factor the split was rendered with (its meta.json)."""
    meta_path = Path(split.dataset.images_path).parent / "meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return DEFAULT_SUPERSAMPLE
    config = meta.get("config")
    if isinstance(config, str):
        try:
            config = ast.literal_eval(config)
        except (ValueError, SyntaxError):
            config = None
    if isinstance(config, Mapping) and "supersample" in config:
        return int(config["supersample"])
    return DEFAULT_SUPERSAMPLE


def valid_mask(f: float, xi: float, circular: bool, size: int, supersample: int) -> np.ndarray:
    """The renderer's valid-pixel mask, rebuilt from the true camera.

    The dataset stores no mask; this repeats the mask half of
    ``geometry.render.render_fisheye`` -- unproject at ``supersample x size``,
    clip to ``r_ref`` for a circular fisheye, area-reduce and keep coverage
    >= 0.5 -- so it marks exactly the pixels the renderer filled.
    """
    camera = FisheyeCamera(f=float(f), xi=float(xi), size=int(size), circular=bool(circular))
    s = int(supersample)
    c_hi = s * camera.cx + (s - 1) / 2.0
    idx = np.arange(s * camera.size, dtype=np.float64)
    uu, vv = np.meshgrid(idx, idx, indexing="xy")
    _, valid = ucm.unproject(np.stack([uu, vv], axis=-1), s * camera.f, camera.xi, c_hi, c_hi)
    if camera.circular:
        valid &= np.hypot(uu - c_hi, vv - c_hi) <= s * camera.r_ref
    if s == 1:
        return valid
    coverage = cv2.resize(valid.astype(np.float32), (camera.size, camera.size),
                          interpolation=cv2.INTER_AREA)
    return coverage >= 0.5


def line_scores(
    image: np.ndarray, mask: np.ndarray, r_ref: float, detector: Any = None
) -> tuple[float, float]:
    """``(line_score, line_score_all)``: segment length per 1000 valid pixels.

    Segments come from OpenCV's LSD on the grey image.  Kept: length >=
    ``LINE_MIN_LENGTH_PX`` with the midpoint inside the eroded valid mask.
    ``line_score`` further keeps the non-radial ones -- midpoint at least
    ``LINE_MIN_RADIUS * r_ref`` from the principal point and at least
    ``LINE_MIN_ANGLE_DEG`` off the principal-point-to-midpoint direction --
    because those are the lines the distortion visibly bends.  A radial line
    stays straight under any radial model, and so does one near the centre.
    """
    detector = detector or cv2.createLineSegmentDetector()
    image = np.ascontiguousarray(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    interior = cv2.erode(np.asarray(mask, dtype=np.uint8), np.ones((3, 3), np.uint8),
                         iterations=LINE_MASK_ERODE_ITERATIONS).astype(bool)
    n_valid = int(interior.sum())
    if n_valid == 0:
        return 0.0, 0.0
    found = detector.detect(gray)[0]
    if found is None:
        return 0.0, 0.0

    segments = np.asarray(found, dtype=np.float64).reshape(-1, 4)
    p0, p1 = segments[:, :2], segments[:, 2:]
    length = np.linalg.norm(p1 - p0, axis=1)
    mid = 0.5 * (p0 + p1)
    height, width = gray.shape
    col = np.clip(np.rint(mid[:, 0]).astype(int), 0, width - 1)
    row = np.clip(np.rint(mid[:, 1]).astype(int), 0, height - 1)
    keep = (length >= LINE_MIN_LENGTH_PX) & interior[row, col]

    radial = mid - np.array([(width - 1) / 2.0, (height - 1) / 2.0])
    distance = np.linalg.norm(radial, axis=1)
    cosine = np.abs(np.sum((p1 - p0) * radial, axis=1)) / np.maximum(length * distance, 1e-12)
    angle = np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0)))
    non_radial = (keep & (distance >= LINE_MIN_RADIUS * float(r_ref))
                  & (angle >= LINE_MIN_ANGLE_DEG))

    scale = 1000.0 / n_valid
    return float(length[non_radial].sum() * scale), float(length[keep].sum() * scale)


def add_line_scores(split: SplitData) -> dict[str, Any]:
    """Score every image of the split and store both scores in its rows.

    Also counts the images with non-zero pixels outside the rebuilt mask: the
    renderer zeroes everything outside its mask, so on rendered data that
    count is 0 unless the rebuilt mask has drifted from the renderer's.
    """
    detector = cv2.createLineSegmentDetector()
    supersample = _supersample(split)
    images = split.dataset.images
    outside = 0
    for i, row in enumerate(split.rows):
        image = np.asarray(images[split.dataset.indices[split.positions[i]]])
        mask = valid_mask(float(row["f"]), float(row["xi"]),
                          row["circular"] in ("True", "true", "1"),
                          image.shape[0], supersample)
        if image[~mask].any():
            outside += 1
        score, score_all = line_scores(image, mask, float(row["r_ref"]), detector)
        row["line_score"] = f"{score:.6f}"
        row["line_score_all"] = f"{score_all:.6f}"
    return {"supersample": supersample, "n_images": split.n,
            "n_nonzero_outside_mask": outside}


# --- statistics -------------------------------------------------------------


def add_relative(split: SplitData, result: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Attach ``rel``: the error as a percentage of the sample's reference radius.

    Pixels are not comparable across image shapes: ``r_ref`` is S/2 = 112 px
    for a circular fisheye but S/sqrt(2) = 158 px for a full-frame one, so the
    same angular mistake costs ~41% more pixels in full-frame.  Dividing by
    ``r_ref`` puts both on the scale of the image's own field of view.
    """
    result["rel"] = result["reproj"] / split.column("r_ref") * 100.0
    return result


def split_stats(split: SplitData, result: Mapping[str, np.ndarray]) -> dict[str, float]:
    reproj = result["reproj"]
    rel = result["rel"]
    return {
        "n": int(split.n),
        "reproj_mean": float(np.mean(reproj)),
        "reproj_median": float(np.median(reproj)),
        "reproj_p90": float(np.percentile(reproj, 90)),
        "rel_mean": float(np.mean(rel)),
        "rel_median": float(np.median(rel)),
        "rel_p90": float(np.percentile(rel, 90)),
        "f_mae": float(np.mean(np.abs(result["pred_f"] - split.column("f")))),
        "xi_mae": float(np.mean(np.abs(result["pred_xi"] - split.column("xi")))),
        "invalid_frac": float(np.mean(result["invalid"])),
    }


def aggregate(per_run: Sequence[Mapping[str, float]]) -> dict[str, Any]:
    """Mean and between-seed std of each statistic (std is None for one run)."""
    out: dict[str, Any] = {"n_runs": len(per_run)}
    for key in ("reproj_mean", "reproj_median", "reproj_p90", "rel_mean", "rel_median",
                "rel_p90", "f_mae", "xi_mae", "invalid_frac"):
        values = np.array([r[key] for r in per_run], dtype=np.float64)
        out[key] = float(values.mean())
        out[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else None
    return out


def cluster_weights(
    asset_ids: Sequence[str], n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """``(n_boot, n_samples)`` multiplicities from resampling whole panoramas.

    Views of one panorama share scenery, so their errors are correlated; a
    bootstrap over individual images would treat them as independent and
    report a confidence interval that is too narrow.  Each replicate draws as
    many panoramas as there are, with replacement, and every view of a drawn
    panorama comes along with it.
    """
    unique, inverse = np.unique(np.asarray(asset_ids), return_inverse=True)
    n_clusters = len(unique)
    draws = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
    per_cluster = np.stack([np.bincount(row, minlength=n_clusters) for row in draws])
    return per_cluster[:, inverse].astype(np.float64)


def panorama_bootstrap(
    values: np.ndarray, asset_ids: Sequence[str], n_boot: int, seed: int
) -> dict[str, float]:
    """Mean of ``values`` with a 95% CI from a panorama-level bootstrap."""
    values = np.asarray(values, dtype=np.float64)
    weights = cluster_weights(asset_ids, n_boot, np.random.default_rng(seed))
    replicates = (weights @ values) / weights.sum(axis=1)
    lo, hi = np.percentile(replicates, [2.5, 97.5])
    return {
        "mean": float(values.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_images": int(values.size),
        "n_panoramas": int(len(np.unique(np.asarray(asset_ids)))),
        "n_boot": int(n_boot),
        "seed": int(seed),
    }


def fov_bin(fov_deg: np.ndarray) -> np.ndarray:
    """Bin labels for FOV: [100,130) [130,160) [160,190) [190,220]."""
    fov = np.asarray(fov_deg, dtype=np.float64)
    labels = np.full(fov.shape, "out of range", dtype=object)
    edges = FOV_EDGES
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        last = i == len(edges) - 2
        inside = (fov >= lo) & ((fov <= hi) if last else (fov < hi))
        labels[inside] = f"{lo:.0f}-{hi:.0f}"
    return labels


def texture_bins(texture: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """``< 5``, then quintiles of the rest.

    Quintile edges come from the samples with texture >= 5 only, so the
    featureless frames do not drag the first quintile down.  A value equal to
    an edge falls in the lower bin; exactly 5 belongs to the quintiles, not to
    ``< 5``.
    """
    texture = np.asarray(texture, dtype=np.float64)
    labels = np.full(texture.shape, "", dtype=object)
    low = texture < LOW_TEXTURE
    labels[low] = f"< {LOW_TEXTURE:g}"

    rest = texture[~low]
    if rest.size == 0:
        return labels, []
    edges = [float(q) for q in np.quantile(rest, [0.2, 0.4, 0.6, 0.8])]
    bounds = [LOW_TEXTURE] + edges + [float(rest.max())]
    for k in range(5):
        lo, hi = bounds[k], bounds[k + 1]
        if k == 0:
            inside = (~low) & (texture <= hi)
        else:
            inside = (~low) & (texture > lo) & (texture <= hi)
        labels[inside] = f"Q{k + 1} ({lo:.1f}-{hi:.1f}]" if k else f"Q1 [{lo:.1f}-{hi:.1f}]"
    return labels, edges


def _texture_order(labels: np.ndarray) -> list[str]:
    return [f"< {LOW_TEXTURE:g}"] + sorted(
        {str(l) for l in labels if str(l).startswith("Q")}, key=lambda s: int(s[1])
    )


def quantile_bins(values: np.ndarray, n_bins: int = 5) -> tuple[np.ndarray, list[float], list[str]]:
    """Equal-count bins over all samples; ties at an edge go to the lower bin.

    Returns the labels, the inner edges and the bin labels in order.  A bin
    left empty by ties is dropped from the order.
    """
    values = np.asarray(values, dtype=np.float64)
    edges = [float(q) for q in np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])]
    bounds = [float(values.min())] + edges + [float(values.max())]
    labels = np.full(values.shape, "", dtype=object)
    order = []
    for k in range(n_bins):
        lo, hi = bounds[k], bounds[k + 1]
        if k == 0:
            inside = values <= hi
            label = f"Q1 [{lo:.1f}-{hi:.1f}]"
        else:
            inside = (values > lo) & (values <= hi)
            label = f"Q{k + 1} ({lo:.1f}-{hi:.1f}]"
        if inside.any():
            labels[inside] = label
            order.append(label)
    return labels, edges, order


def binned_means(
    errors: np.ndarray, rel: np.ndarray, labels: np.ndarray,
    order: Sequence[str], assets: np.ndarray,
) -> list[dict[str, Any]]:
    """Mean error per bin, in px and % of r_ref, with the bin's panorama count.

    ``n_panoramas`` is the number to keep in mind when reading a bin: the views
    of one panorama are correlated, so a bin of 100 images drawn from 5
    panoramas carries far less evidence than its image count suggests.
    """
    out = []
    for label in order:
        mask = labels == label
        count = int(mask.sum())
        out.append({
            "bin": str(label),
            "n": count,
            "n_panoramas": int(len(set(assets[mask]))),
            "mean": float(errors[mask].mean()) if count else None,
            "mean_pct": float(rel[mask].mean()) if count else None,
        })
    return out


def stratify(
    split: SplitData,
    group_errors: Mapping[str, np.ndarray],
    group_rel: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Per-group mean error by texture, line score, FOV, image shape and category."""
    assets = np.array(split.strings("asset_id"), dtype=object)
    texture = split.column("texture")
    tex_labels, tex_edges = texture_bins(texture)
    tex_order = _texture_order(tex_labels)
    line_labels, line_edges, line_order = quantile_bins(split.column("line_score"))
    fov_labels = fov_bin(split.column("fov_deg"))
    fov_order = [f"{FOV_EDGES[i]:.0f}-{FOV_EDGES[i + 1]:.0f}" for i in range(len(FOV_EDGES) - 1)]
    if (fov_labels == "out of range").any():
        fov_order.append("out of range")
    shape_labels = np.where(split.flags("circular"), "circular", "full-frame").astype(object)
    category = np.array(split.strings("category"), dtype=object)
    cat_order = sorted(set(category))

    keep = texture >= LOW_TEXTURE
    result: dict[str, Any] = {
        "split": split.name,
        "texture_edges": tex_edges,
        "line_score_edges": line_edges,
        "n_low_texture": int((~keep).sum()),
        "n_low_texture_panoramas": int(len(set(assets[~keep]))),
        "n_panoramas": int(len(set(assets))),
        "by": {},
    }
    for group, errors in group_errors.items():
        rel = group_rel[group]
        result["by"][group] = {
            "texture": binned_means(errors, rel, tex_labels, tex_order, assets),
            "line_score": binned_means(errors, rel, line_labels, line_order, assets),
            "fov": binned_means(errors, rel, fov_labels, fov_order, assets),
            "shape": binned_means(errors, rel, shape_labels, ["full-frame", "circular"], assets),
            "category": binned_means(errors, rel, category, cat_order, assets),
            "excluding_low_texture": {
                "n": int(keep.sum()),
                "mean": float(errors[keep].mean()) if keep.any() else None,
                "mean_pct": float(rel[keep].mean()) if keep.any() else None,
                "mean_all": float(errors.mean()),
                "mean_all_pct": float(rel.mean()),
            },
        }
    return result


def texture_line_crosstab(
    split: SplitData,
    group_errors: Mapping[str, np.ndarray],
    group_rel: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Texture bin x (line score below / at-or-above its median).

    Asks whether the most textured frames are the ones short of straight
    lines -- foliage, rock, water -- which would explain why error does not
    fall with texture.  ``low`` is strictly below the split's median.
    """
    assets = np.array(split.strings("asset_id"), dtype=object)
    line = split.column("line_score")
    median = float(np.median(line))
    low = line < median
    tex_labels, _ = texture_bins(split.column("texture"))

    rows = []
    for label in _texture_order(tex_labels):
        in_bin = tex_labels == label
        n = int(in_bin.sum())
        entry: dict[str, Any] = {
            "bin": label,
            "n": n,
            "n_panoramas": int(len(set(assets[in_bin]))),
            "n_low": int((in_bin & low).sum()),
            "n_panoramas_low": int(len(set(assets[in_bin & low]))),
            "n_panoramas_high": int(len(set(assets[in_bin & ~low]))),
            "frac_low": float((in_bin & low).sum() / n) if n else None,
            "by": {},
        }
        for group, errors in group_errors.items():
            rel = group_rel[group]
            cell = {}
            for side, select in (("low", in_bin & low), ("high", in_bin & ~low)):
                any_ = bool(select.any())
                cell[side] = float(errors[select].mean()) if any_ else None
                cell[f"{side}_pct"] = float(rel[select].mean()) if any_ else None
            entry["by"][group] = cell
        rows.append(entry)
    return {"median": median, "rows": rows}


def predicted_fov(pred_f: np.ndarray, pred_xi: np.ndarray, r_ref: np.ndarray) -> np.ndarray:
    """Full-angle FOV of each predicted camera at the sample's ``r_ref``.

    NaN where it is undefined: a negative ``xi``, or ``xi > 1`` with a focal
    length too short for the projection ever to reach ``r_ref``.
    """
    out = np.full(len(pred_f), np.nan)
    for i, (f, xi, r) in enumerate(zip(pred_f, pred_xi, r_ref)):
        try:
            out[i] = ucm.fov_from_f(float(f), float(xi), float(r))
        except ValueError:
            pass
    return out


def fov_bias(
    true_fov: np.ndarray, bias: np.ndarray, assets: Sequence[str],
    n_boot: int, seed: int, edges: Sequence[float] = FOV_BIAS_EDGES,
) -> dict[str, Any]:
    """Predicted - true FOV by true-FOV bin, and its slope against true FOV.

    Regression to the mean shows as a positive bias at narrow FOVs, a
    negative one at wide FOVs, so a negative slope.  The range is the 2.5th to
    97.5th percentile of the per-image bias in the bin -- the spread of the
    errors, not a confidence interval.  The slope's CI is a panorama
    bootstrap, as for the paired comparison.
    """
    true_fov = np.asarray(true_fov, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    assets = np.asarray(assets, dtype=object)
    defined = np.isfinite(bias)

    bins = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        last = i == len(edges) - 2
        inside = (true_fov >= lo) & ((true_fov <= hi) if last else (true_fov < hi)) & defined
        values = bias[inside]
        bins.append({
            "bin": f"{lo:.0f}-{hi:.0f}",
            "n": int(inside.sum()),
            "n_panoramas": int(len(set(assets[inside]))),
            "mean": float(values.mean()) if values.size else None,
            "p2_5": float(np.percentile(values, 2.5)) if values.size else None,
            "p97_5": float(np.percentile(values, 97.5)) if values.size else None,
        })

    x, y = true_fov[defined], bias[defined]
    slope = None
    if x.size >= 2 and np.ptp(x) > 0:
        def _slope(w: np.ndarray) -> np.ndarray:
            total = w.sum(axis=-1)
            mx, my = (w @ x) / total, (w @ y) / total
            sxx = (w @ (x * x)) / total - mx * mx
            sxy = (w @ (x * y)) / total - mx * my
            return sxy / sxx

        weights = cluster_weights(assets[defined], n_boot, np.random.default_rng(seed))
        replicates = _slope(weights)
        lo, hi = np.nanpercentile(replicates, [2.5, 97.5])
        slope = {"per_deg": float(_slope(np.ones_like(x))), "ci_low": float(lo),
                 "ci_high": float(hi), "n_boot": int(n_boot), "seed": int(seed)}
    return {
        "n_images": int(bias.size),
        "n_undefined": int((~defined).sum()),
        "mean": float(y.mean()) if y.size else None,
        "bins": bins,
        "slope": slope,
    }


# --- figures ----------------------------------------------------------------


def _plot_error_vs(
    path: Path, x: np.ndarray, group_errors: Mapping[str, np.ndarray],
    edges: np.ndarray, xlabel: str, title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    centres = 0.5 * (edges[:-1] + edges[1:])
    index = np.clip(np.digitize(x, edges[1:-1], right=False), 0, len(edges) - 2)
    for group, errors in group_errors.items():
        means = [errors[index == b].mean() if (index == b).any() else np.nan
                 for b in range(len(edges) - 1)]
        axis.plot(centres, means, marker="o", label=group)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("mean reprojection error (px)")
    axis.set_title(title)
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_fov_bias(path: Path, bias_by_group: Mapping[str, Mapping[str, Any]], title: str) -> None:
    """Mean predicted - true FOV per true-FOV bin, shaded over its 95% range."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    for group, result in bias_by_group.items():
        bins = [b for b in result["bins"] if b["mean"] is not None]
        centres = [np.mean([float(v) for v in b["bin"].split("-")]) for b in bins]
        line, = axis.plot(centres, [b["mean"] for b in bins], marker="o", label=group)
        axis.fill_between(centres, [b["p2_5"] for b in bins], [b["p97_5"] for b in bins],
                          color=line.get_color(), alpha=0.12)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("true FOV (deg)")
    axis.set_ylabel("predicted − true FOV (deg)")
    axis.set_title(title)
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _perspective(image: np.ndarray, f: float, xi: float, circular: bool, fov: float):
    """Undistort with the given intrinsics, or None if they cannot be rendered."""
    try:
        camera = FisheyeCamera(f=float(f), xi=max(float(xi), 0.0), size=image.shape[0],
                               circular=bool(circular))
        return fisheye_to_perspective(image, camera, image.shape[0], fov)
    except (ValueError, RuntimeError):
        return None


def plot_worst(
    path: Path, split: SplitData, group: str, errors: np.ndarray,
    first_run: Mapping[str, np.ndarray], k: int, persp_fov: float,
) -> list[dict[str, Any]]:
    """Worst ``k`` images: fisheye, true undistortion, predicted, r(theta)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    worst = np.argsort(-errors)[:k]
    fig, axes = plt.subplots(len(worst), 4, figsize=(13, 3.1 * len(worst)), squeeze=False)
    listed = []
    for row, sample in enumerate(worst):
        info = split.rows[sample]
        image = np.asarray(split.dataset.images[split.dataset.indices[split.positions[sample]]])
        f, xi = float(info["f"]), float(info["xi"])
        circular = info["circular"] in ("True", "true", "1")
        pf, pxi = float(first_run["pred_f"][sample]), float(first_run["pred_xi"][sample])
        theta_ref = math.radians(float(info["fov_deg"])) / 2.0

        panels = [
            (image, f"{info['asset_id']} #{info['view_idx']}\n"
                    f"FOV {float(info['fov_deg']):.0f}  tex {float(info['texture']):.1f}  "
                    f"{'circ' if circular else 'full'}"),
            (_perspective(image, f, xi, circular, persp_fov),
             f"true  f={f:.0f}  xi={xi:.2f}"),
            (_perspective(image, pf, pxi, circular, persp_fov),
             f"pred  f={pf:.0f}  xi={pxi:.2f}"),
        ]
        for col, (picture, caption) in enumerate(panels):
            axis = axes[row, col]
            if picture is None:
                axis.text(0.5, 0.5, "cannot render", ha="center", va="center")
            else:
                axis.imshow(picture)
            axis.set_title(caption, fontsize=8)
            axis.set_xticks([])
            axis.set_yticks([])

        theta = np.linspace(0.0, theta_ref, 128)
        axis = axes[row, 3]
        axis.plot(np.degrees(theta), ucm.radius_from_theta(theta, f, xi), label="true")
        axis.plot(np.degrees(theta), ucm.radius_from_theta(theta, pf, pxi), "--", label="pred")
        axis.set_title(f"r(theta)   mean error {errors[sample]:.1f} px", fontsize=8)
        axis.set_xlabel("theta (deg)", fontsize=7)
        axis.set_ylabel("r (px)", fontsize=7)
        axis.tick_params(labelsize=7)
        axis.legend(fontsize=7)
        listed.append({
            "asset_id": info["asset_id"], "view_idx": int(info["view_idx"]),
            "error_px": float(errors[sample]), "f": f, "xi": xi,
            "pred_f": pf, "pred_xi": pxi, "fov_deg": float(info["fov_deg"]),
            "texture": float(info["texture"]), "circular": circular,
        })

    fig.suptitle(
        f"{group}: worst {len(worst)} on {split.name} (error averaged over seeds; "
        "predictions shown from the first seed)", fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return listed


# --- provenance -------------------------------------------------------------


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10, check=False, cwd=PROJECT_ROOT)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 else "unknown"


# --- orchestration ----------------------------------------------------------


def run(
    cfg: EvalConfig,
    final: bool = False,
    device: torch.device | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Evaluate everything, write the per-sample CSVs, summary and figures."""
    from tools.baseline import load_train_labels, search_best_constant

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()
    out_dir = Path(cfg.out_dir)
    norm_stats = load_norm_stats(cfg.data_root)
    splits_doc = json.loads((Path(cfg.data_root) / "splits.json").read_text("utf-8"))

    split_names = evaluated_splits(final)
    analysis_split = "test" if final else "val"
    log(f"evaluate: splits {split_names}; analysis on {analysis_split}"
        f"{'' if final else ' (preview -- rerun with --final for test)'}")

    splits = {name: open_split(cfg, name, final, splits_doc) for name in split_names}

    # Line scores depend only on the images, so they are computed once per
    # split and travel with the rows into every per-sample CSV.
    line_checks: dict[str, dict[str, Any]] = {}
    for name, split in splits.items():
        line_checks[name] = add_line_scores(split)
        log(f"line scores: {name} ({split.n} images, supersample "
            f"{line_checks[name]['supersample']}; "
            f"{line_checks[name]['n_nonzero_outside_mask']} with pixels outside the mask)")

    # --- baselines, fixed on train
    mean_f = float(np.exp(norm_stats["mean"][0]))
    mean_xi = float(norm_stats["mean"][1])
    labels = load_train_labels(cfg.data_root, device=device)
    best = search_best_constant(labels, n_grid=cfg.baseline_grid, xi_max=cfg.baseline_xi_max)
    baselines = {
        BASELINE_MEAN: {"f": mean_f, "xi": mean_xi},
        BASELINE_BEST: {"f": best["f"], "xi": best["xi"], "at_edge": best["at_edge"]},
    }
    log(f"baselines: mean params f={mean_f:.1f} xi={mean_xi:.3f}; "
        f"best constant f={best['f']:.1f} xi={best['xi']:.3f}")

    # --- every run, every split
    results: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    run_stats: dict[str, dict[str, dict[str, float]]] = {}
    sanity: dict[str, dict[str, float]] = {}

    for run_name in cfg.runs():
        model, train_cfg, payload = load_run(run_name, cfg, norm_stats, device)
        batch_size = int(cfg.batch_size or train_cfg.batch_size)
        results[run_name] = {}
        run_stats[run_name] = {}
        try:
            for name, split in splits.items():
                result = add_relative(split, predict(
                    model, train_cfg, split, norm_stats, device,
                    batch_size, cfg.num_workers,
                ))
                results[run_name][name] = result
                run_stats[run_name][name] = split_stats(split, result)
                write_per_sample(out_dir / "per_sample" / run_name / f"{name}.csv",
                                 split, result)
        finally:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        recomputed = run_stats[run_name]["val"]["reproj_mean"]
        recorded = mlflow_best(run_name, cfg.tracking_uri)
        diff = abs(recomputed - recorded)
        sanity[run_name] = {
            "recomputed_val_px": recomputed,
            "mlflow_best_val_px": recorded,
            "checkpoint_val_px": float(payload.get("metrics", {}).get("val_reproj_mean_px", float("nan"))),
            "abs_diff_px": diff,
            "checkpoint_step": int(payload.get("step", -1)),
        }
        log(f"  {run_name}: val {recomputed:.4f} px vs MLflow {recorded:.4f} "
            f"(|diff| {diff:.5f})")
        if diff >= SANITY_TOLERANCE_PX:
            raise RuntimeError(
                f"sanity check failed for {run_name}: recomputed val "
                f"{recomputed:.4f} px vs MLflow {recorded:.4f} px "
                f"(|diff| {diff:.4f} >= {SANITY_TOLERANCE_PX}); the checkpoint "
                "or the evaluation path does not match the training run"
            )

    for name, values in baselines.items():
        results[name] = {}
        run_stats[name] = {}
        for split_name, split in splits.items():
            result = add_relative(split, constant_predictions(split, values["f"], values["xi"]))
            results[name][split_name] = result
            run_stats[name][split_name] = split_stats(split, result)
            write_per_sample(out_dir / "per_sample" / name / f"{split_name}.csv",
                             split, result)

    # --- groups
    groups: dict[str, list[str]] = dict(cfg.groups)
    for name in baselines:
        groups[name] = [name]

    group_stats = {
        group: {s: aggregate([run_stats[r][s] for r in members]) for s in split_names}
        for group, members in groups.items()
    }

    def seed_mean(group: str, split_name: str, key: str = "reproj") -> np.ndarray:
        return np.mean([results[r][split_name][key] for r in groups[group]], axis=0)

    analysis = splits[analysis_split]
    model_groups = list(cfg.groups)
    per_image = {g: seed_mean(g, analysis_split) for g in groups}
    per_image_rel = {g: seed_mean(g, analysis_split, "rel") for g in groups}

    # (b) paired comparison, in px and in % of r_ref.  The same seed gives the
    # same panorama draws, so the two intervals come from identical resamples.
    comparison = None
    if cfg.comparison:
        first, second = cfg.comparison
        assets = analysis.strings("asset_id")
        px = panorama_bootstrap(per_image[first] - per_image[second], assets,
                                cfg.bootstrap_n, cfg.bootstrap_seed)
        pct = panorama_bootstrap(per_image_rel[first] - per_image_rel[second], assets,
                                 cfg.bootstrap_n, cfg.bootstrap_seed)
        comparison = {
            "split": analysis_split,
            "first": first,
            "second": second,
            **px,  # px at the top level: mean, ci_low, ci_high, counts
            "pct": {k: pct[k] for k in ("mean", "ci_low", "ci_high")},
        }

    # (c) generalisation gap
    gap_splits = [s for s in split_names if s in ("train_clean", "val", "test")]
    gap = {
        group: {s: group_stats[group][s]["reproj_mean"] for s in gap_splits}
        for group in groups
    }
    gap_pct = {
        group: {s: group_stats[group][s]["rel_mean"] for s in gap_splits}
        for group in groups
    }

    # (d) stratification
    strata = stratify(analysis, per_image, per_image_rel)
    crosstab = texture_line_crosstab(
        analysis, {g: per_image[g] for g in model_groups},
        {g: per_image_rel[g] for g in model_groups},
    )

    # (f) FOV bias: per image, predicted - true FOV averaged over the group's
    # seeds; an image whose FOV is undefined under any seed is left out.
    true_fov = analysis.column("fov_deg")
    r_ref = analysis.column("r_ref")
    analysis_assets = analysis.strings("asset_id")
    fov_bias_by_group = {}
    for group in model_groups:
        per_run = [
            predicted_fov(results[r][analysis_split]["pred_f"],
                          results[r][analysis_split]["pred_xi"], r_ref) - true_fov
            for r in groups[group]
        ]
        fov_bias_by_group[group] = fov_bias(
            true_fov, np.mean(per_run, axis=0), analysis_assets,
            cfg.bootstrap_n, cfg.bootstrap_seed,
        )

    # (e) stress
    stress = None
    if "stress" in splits:
        st = splits["stress"]
        sibling = np.array(st.strings("stress_sibling_split"), dtype=object)
        in_train = sibling == "train"
        asset = np.array(st.strings("asset_id"), dtype=object)
        stress = {"by": {}, "n_panoramas_in_train": int(len(set(asset[in_train]))),
                  "n_panoramas_not_in_train": int(len(set(asset[~in_train]))),
                  "n_images_in_train": int(in_train.sum()),
                  "n_images_not_in_train": int((~in_train).sum())}
        for group in groups:
            e = seed_mean(group, "stress")
            r = seed_mean(group, "stress", "rel")
            stress["by"][group] = {
                "all": float(e.mean()),
                "sibling_in_train": float(e[in_train].mean()) if in_train.any() else None,
                "sibling_not_in_train": float(e[~in_train].mean()) if (~in_train).any() else None,
                "all_pct": float(r.mean()),
                "sibling_in_train_pct": float(r[in_train].mean()) if in_train.any() else None,
                "sibling_not_in_train_pct":
                    float(r[~in_train].mean()) if (~in_train).any() else None,
            }

    # --- figures
    figures: dict[str, Any] = {}
    fig_dir = out_dir / "figures"
    pair = [g for g in (cfg.comparison or model_groups[:2])]
    pair_errors = {g: per_image[g] for g in pair}
    texture = analysis.column("texture")
    tex_edges = np.unique(np.quantile(texture, np.linspace(0, 1, 11)))
    path = fig_dir / f"{analysis_split}_error_vs_texture.png"
    _plot_error_vs(path, texture, pair_errors, tex_edges, "texture (mean Sobel)",
                   f"error vs texture ({analysis_split}, decile bins)")
    figures["error_vs_texture"] = path.as_posix()
    path = fig_dir / f"{analysis_split}_error_vs_fov.png"
    _plot_error_vs(path, analysis.column("fov_deg"), pair_errors,
                   np.arange(100.0, 221.0, 10.0), "FOV (deg)",
                   f"error vs FOV ({analysis_split}, 10 deg bins)")
    figures["error_vs_fov"] = path.as_posix()
    line = analysis.column("line_score")
    path = fig_dir / f"{analysis_split}_error_vs_line_score.png"
    _plot_error_vs(path, line, pair_errors, np.unique(np.quantile(line, np.linspace(0, 1, 11))),
                   "line score (non-radial segment length per 1000 valid px)",
                   f"error vs line score ({analysis_split}, decile bins)")
    figures["error_vs_line_score"] = path.as_posix()
    path = fig_dir / f"{analysis_split}_fov_bias.png"
    plot_fov_bias(path, fov_bias_by_group,
                  f"predicted − true FOV ({analysis_split}; band: 2.5–97.5th percentile)")
    figures["fov_bias"] = path.as_posix()

    worst: dict[str, Any] = {}
    for group in model_groups:
        path = fig_dir / f"{analysis_split}_worst_{group}.png"
        worst[group] = plot_worst(path, analysis, group, per_image[group],
                                  results[cfg.groups[group][0]][analysis_split],
                                  cfg.worst_k, cfg.perspective_fov_deg)
        figures[f"worst_{group}"] = path.as_posix()

    summary = {
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_sec": round(time.time() - started, 1),
        "final": bool(final),
        "git_commit": _git_commit(),
        "device": str(device),
        "splits": {name: splits[name].n for name in split_names},
        "analysis_split": analysis_split,
        "groups": groups,
        "baselines": baselines,
        "sanity": sanity,
        "runs": run_stats,
        "group_stats": group_stats,
        "comparison": comparison,
        "generalisation_gap": gap,
        "generalisation_gap_pct": gap_pct,
        "stratification": strata,
        "line_checks": line_checks,
        "texture_line_crosstab": crosstab,
        "fov_bias": fov_bias_by_group,
        "stress": stress,
        "worst": worst,
        "figures": figures,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    log(f"wrote {out_dir / 'summary.md'} and summary.json ({summary['elapsed_sec']}s)")
    return summary


# --- markdown ---------------------------------------------------------------


def _pm(stats: Mapping[str, Any], key: str, digits: int = 3) -> str:
    value = stats.get(key)
    if value is None:
        return "—"
    std = stats.get(f"{key}_std")
    return f"{value:.{digits}f}" if std is None else f"{value:.{digits}f} ± {std:.{digits}f}"


def _both(px: float | None, pct: float | None, signed: bool = False) -> str:
    """``px (pct%)`` in one cell, so every table carries both scales."""
    if px is None or pct is None:
        return "—"
    sign = "+" if signed else ""
    return f"{px:{sign}.3f} ({pct:{sign}.2f}%)"


def render_markdown(s: Mapping[str, Any]) -> str:
    groups = list(s["groups"])
    splits = list(s["splits"])
    analysis = s["analysis_split"]
    preview = not s["final"]
    lines: list[str] = []
    add = lines.append

    add("# Evaluation summary")
    add("")
    add(f"- evaluated at **{s['evaluated_at']}** on `{s['device']}`, commit `{s['git_commit'][:12]}`")
    add(f"- splits: " + ", ".join(f"`{k}` ({v})" for k, v in s["splits"].items()))
    if preview:
        add("- **Not a final evaluation.** test and stress were not read. Sections marked "
            "*preview* run on val, which chose the learning rates and checkpoints, "
            "so they are optimistic -- they validate the analysis, not the conclusions. "
            "Rerun with `--final` for test.")
    else:
        add("- **Final evaluation**: test and stress included.")
    add("")

    add("## Sanity check")
    add("")
    add("Val reprojection error recomputed from each checkpoint against the MLflow record "
        f"(must agree within {SANITY_TOLERANCE_PX} px).")
    add("")
    add("| run | recomputed | MLflow | abs diff | step |")
    add("|---|---:|---:|---:|---:|")
    for run_name, v in s["sanity"].items():
        add(f"| `{run_name}` | {v['recomputed_val_px']:.4f} | {v['mlflow_best_val_px']:.4f} "
            f"| {v['abs_diff_px']:.5f} | {v['checkpoint_step']} |")
    add("")

    add("## a. Main table")
    add("")
    add("Reprojection error in px and as % of the reference radius `r_ref` "
        "(112 px circular, 158 px full-frame). Pixels are not comparable across image "
        "shapes; the percentage is. Multi-seed groups show the mean ± std over seeds "
        "of each statistic.")
    for split in splits:
        add("")
        add(f"### {split} ({s['splits'][split]} images)")
        add("")
        add("| group | runs | mean px | mean % | median px | median % | p90 px | p90 % "
            "| f MAE (px) | ξ MAE | invalid |")
        add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for g in groups:
            st = s["group_stats"][g][split]
            add(f"| {g} | {st['n_runs']} | {_pm(st, 'reproj_mean')} | {_pm(st, 'rel_mean', 2)} "
                f"| {_pm(st, 'reproj_median')} | {_pm(st, 'rel_median', 2)} "
                f"| {_pm(st, 'reproj_p90')} | {_pm(st, 'rel_p90', 2)} "
                f"| {_pm(st, 'f_mae', 2)} | {_pm(st, 'xi_mae')} "
                f"| {_pm(st, 'invalid_frac', 4)} |")
    add("")
    b = s["baselines"]
    add(f"Baselines are fixed on train: mean params f={b[BASELINE_MEAN]['f']:.1f}, "
        f"ξ={b[BASELINE_MEAN]['xi']:.3f}; best constant f={b[BASELINE_BEST]['f']:.1f}, "
        f"ξ={b[BASELINE_BEST]['xi']:.3f}"
        + (f" (on grid edge: {', '.join(b[BASELINE_BEST]['at_edge'])})"
           if b[BASELINE_BEST].get("at_edge") else "") + ".")
    add("")

    c = s.get("comparison")
    if c:
        add(f"## b. Paired comparison: {c['first']} − {c['second']}"
            + (" *(preview on val)*" if preview else ""))
        add("")
        add(f"Per-image error averaged over each group's seeds, then differenced on the "
            f"same {c['n_images']} images. 95% CI from a bootstrap over the "
            f"{c['n_panoramas']} panoramas ({c['n_boot']} replicates, seed {c['seed']}): "
            "views of one panorama are correlated, so the panorama is the resampling unit.")
        add("")
        add("| metric | mean difference | 95% CI |")
        add("|---|---:|---|")
        add(f"| px | {c['mean']:+.3f} | [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] |")
        p = c["pct"]
        add(f"| % of r_ref | {p['mean']:+.2f}% | [{p['ci_low']:+.2f}%, {p['ci_high']:+.2f}%] |")
        add("")
        # Decision rule pre-registered in SPEC 3.14.2: both scales must agree.
        if c["ci_high"] < 0 and p["ci_high"] < 0:
            verdict = f"Both intervals lie below 0: `{c['first']}` is better."
        elif c["ci_low"] > 0 and p["ci_low"] > 0:
            verdict = f"Both intervals lie above 0: `{c['second']}` is better."
        else:
            verdict = "At least one interval includes 0: the groups are indistinguishable."
        add(f"{verdict} (Rule from SPEC 3.14.2; negative favours `{c['first']}`.)")
        add("")

    add("## c. Generalisation gap")
    add("")
    add("Mean error, px (% of r_ref).")
    add("")
    cols = [x for x in ("train_clean", "val", "test") if x in splits]
    add("| group | " + " | ".join(cols) + " | val − train_clean |")
    add("|---|" + "---:|" * (len(cols) + 1))
    for g in groups:
        row = s["generalisation_gap"][g]
        row_pct = s["generalisation_gap_pct"][g]
        cells = [_both(row[x], row_pct[x]) for x in cols]
        cells.append(_both(row["val"] - row["train_clean"],
                           row_pct["val"] - row_pct["train_clean"], signed=True))
        add(f"| {g} | " + " | ".join(cells) + " |")
    add("")

    st = s["stratification"]
    add(f"## d. Stratification on {analysis}" + (" *(preview)*" if preview else ""))
    add("")
    add("Mean reprojection error per bin, px (% of r_ref); multi-seed groups use the "
        "per-image error averaged over seeds. **Read `panoramas`, not `n`, as the "
        f"sample size**: views of one panorama are correlated, and the {analysis} split "
        f"has only {st['n_panoramas']} panoramas in total.")
    for facet, title in (("texture", "Texture"),
                         ("line_score", "Line score (quintiles)"), ("fov", "FOV (deg)"),
                         ("shape", "Image shape"), ("category", "primary_category")):
        add("")
        add(f"### {title}")
        add("")
        first = st["by"][groups[0]][facet]
        add("| bin | n | panoramas | " + " | ".join(groups) + " |")
        add("|---|---:|---:|" + "---:|" * len(groups))
        for i, entry in enumerate(first):
            cells = []
            for g in groups:
                b_ = st["by"][g][facet][i]
                cells.append(_both(b_["mean"], b_["mean_pct"]))
            add(f"| {entry['bin']} | {entry['n']} | {entry['n_panoramas']} | "
                + " | ".join(cells) + " |")
    add("")
    add(f"Excluding texture < {LOW_TEXTURE:g} ({st['n_low_texture']} images from "
        f"{st['n_low_texture_panoramas']} panoramas):")
    add("")
    add("| group | all | texture ≥ 5 |")
    add("|---|---:|---:|")
    for g in groups:
        e = st["by"][g]["excluding_low_texture"]
        add(f"| {g} | {_both(e['mean_all'], e['mean_all_pct'])} "
            f"| {_both(e['mean'], e['mean_pct'])} |")
    add("")

    ct = s["texture_line_crosstab"]
    model_groups = list(ct["rows"][0]["by"]) if ct["rows"] else []
    add("### Texture × line score")
    add("")
    add(f"Line score split at its median on {analysis} ({ct['median']:.2f}): `low` is "
        "strictly below. Checks whether the most textured frames are the ones short "
        "of straight lines (SPEC 3.14.3 item 5b). Cells: mean error px (% of r_ref).")
    add("")
    header = "| texture bin | n | panoramas | low line % | panoramas low / high |"
    for g in model_groups:
        header += f" {g} low | {g} high |"
    add(header)
    add("|---|---:|---:|---:|---:|" + "---:|---:|" * len(model_groups))
    for r in ct["rows"]:
        frac = "—" if r["frac_low"] is None else f"{100 * r['frac_low']:.0f}%"
        cells = [f"{r['bin']}", f"{r['n']}", f"{r['n_panoramas']}", frac,
                 f"{r['n_panoramas_low']} / {r['n_panoramas_high']}"]
        for g in model_groups:
            v = r["by"][g]
            cells += [_both(v["low"], v["low_pct"]), _both(v["high"], v["high_pct"])]
        add("| " + " | ".join(cells) + " |")
    add("")
    checks = s["line_checks"]
    add("Line score: OpenCV LSD on the grey image; segments ≥ "
        f"{LINE_MIN_LENGTH_PX:g} px with the midpoint inside the valid mask (eroded "
        f"{LINE_MASK_ERODE_ITERATIONS}× 3×3); non-radial = midpoint ≥ "
        f"{LINE_MIN_RADIUS:g}·r_ref from the principal point and ≥ "
        f"{LINE_MIN_ANGLE_DEG:g}° off the radial direction; length per 1000 valid "
        "pixels. Mask check (images with non-zero pixels outside the rebuilt mask): "
        + ", ".join(f"{k} {v['n_nonzero_outside_mask']}/{v['n_images']}"
                    for k, v in checks.items()) + ".")
    add("")

    add("## e. Stress (Pure Skies)")
    add("")
    stress = s.get("stress")
    if stress is None:
        add("Not evaluated — stress is read only with `--final`.")
    else:
        add(f"Sibling in train: {stress['n_panoramas_in_train']} panoramas / "
            f"{stress['n_images_in_train']} images. Sibling not in train: "
            f"{stress['n_panoramas_not_in_train']} panoramas / "
            f"{stress['n_images_not_in_train']} images — **too few for a conclusion; "
            "reference only.**")
        add("")
        add("Mean error, px (% of r_ref).")
        add("")
        add("| group | all | sibling in train | sibling not in train |")
        add("|---|---:|---:|---:|")
        for g in groups:
            v = stress["by"][g]
            add(f"| {g} | {_both(v['all'], v['all_pct'])} "
                f"| {_both(v['sibling_in_train'], v['sibling_in_train_pct'])} "
                f"| {_both(v['sibling_not_in_train'], v['sibling_not_in_train_pct'])} |")
    add("")

    add(f"## f. FOV bias on {analysis}" + (" *(preview)*" if preview else ""))
    add("")
    add("Predicted FOV from `geometry.ucm.fov_from_f` at the sample's r_ref, minus the "
        "true FOV, in degrees; per image averaged over the group's seeds. `range` is the "
        "2.5–97.5th percentile of that per-image bias (spread, not a CI). Regression to "
        "the mean shows as positive bias at narrow FOVs, negative at wide ones, i.e. a "
        "negative slope (SPEC 3.14.3 item 6). Slope CI: panorama bootstrap.")
    fb = s["fov_bias"]
    for g, v in fb.items():
        add("")
        add(f"### {g}")
        add("")
        sl = v["slope"]
        slope = ("—" if sl is None else
                 f"{sl['per_deg']:+.3f} deg/deg, 95% CI [{sl['ci_low']:+.3f}, {sl['ci_high']:+.3f}]")
        mean = "—" if v["mean"] is None else f"{v['mean']:+.2f}°"
        add(f"Mean bias {mean}; slope against true FOV {slope}; "
            f"{v['n_undefined']} of {v['n_images']} images with an undefined predicted FOV.")
        add("")
        add("| true FOV | n | panoramas | mean bias (°) | range (°) |")
        add("|---|---:|---:|---:|---|")
        for b in v["bins"]:
            if b["mean"] is None:
                add(f"| {b['bin']} | 0 | 0 | — | — |")
            else:
                add(f"| {b['bin']} | {b['n']} | {b['n_panoramas']} | {b['mean']:+.2f} "
                    f"| [{b['p2_5']:+.2f}, {b['p97_5']:+.2f}] |")
    add("")

    add("## Figures")
    add("")
    for key, path in s["figures"].items():
        rel = Path(path).name
        add(f"- `{key}`: [figures/{rel}](figures/{rel})")
    add("")
    return "\n".join(lines)


# --- CLI --------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Final evaluation and stratified analysis (SPEC.md 1.4.1)."
    )
    parser.add_argument("--config", default="configs/eval/final.yaml")
    parser.add_argument(
        "--final", action="store_true",
        help="also evaluate test and stress (do this once, at the end)",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    run(load_eval_config(args.config), final=args.final)
    return 0


if __name__ == "__main__":
    sys.exit(main())
