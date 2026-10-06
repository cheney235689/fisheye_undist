"""Constant-prediction baselines on the val split.

Two models that never look at the image, both of which a trained run has to
beat before it can claim to have learned anything:

**mean params**
    Always answer with the train split's mean parameters.  In the standardised
    space that is exactly the zero vector, since the targets are standardised
    with the train mean and standard deviation (SPEC.md 4.7).

**best constant**
    The single ``(f, xi)`` that minimises the mean reprojection error *on
    train*, found by a grid search over ``log f`` and ``xi``.  This is the
    stronger and more honest floor: the mean of the parameters is not the
    parameter pair that minimises the error, because the reprojection error is
    not symmetric in them.  Beating "mean params" is easy; beating this means
    the image is actually contributing.

The search reads only ``labels.csv`` -- no images -- and scores candidates with
``metrics.geometric.reprojection_error_px``.  The val numbers come from
``train.evaluate`` driven by a constant module, so they are produced by the
very code path that scores a real run.  Like ``train.py``, this only ever
reads train labels and the val split.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.fisheye_dataset import load_norm_stats  # noqa: E402
from metrics.geometric import reprojection_error_px  # noqa: E402
from train import (  # noqa: E402
    DataConfig,
    TrainConfig,
    build_dataset,
    build_loader,
    evaluate,
)

__all__ = [
    "ConstantPredictor",
    "load_train_labels",
    "constant_reproj_error",
    "search_best_constant",
    "baseline_metrics",
    "main",
]

EVAL_SPLIT = "val"
DEFAULT_GRID = 100
# The sampler draws xi in [0, 1.5] (configs/data.yaml), but the best *constant*
# is not bound by that: a larger xi paired with a larger f mimics the average
# radial profile better, and at 1.5 the search was still improving.  Searching
# past the sampled range gives an honest floor instead of a clipped one.
DEFAULT_XI_MAX = 3.0


class ConstantPredictor(nn.Module):
    """Emits one fixed standardised vector for every input, ignoring the image."""

    def __init__(self, value: Sequence[float] | torch.Tensor | None = None) -> None:
        super().__init__()
        if value is None:
            value = torch.zeros(2)
        self.register_buffer("value", torch.as_tensor(value, dtype=torch.float32))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.value.to(images.device).expand(images.shape[0], 2).contiguous()


# --- the grid search --------------------------------------------------------


def load_train_labels(
    root: str | Path, device: torch.device | None = None
) -> dict[str, torch.Tensor]:
    """Read the train split's ``labels.csv``; no images are touched."""
    path = Path(root) / "rendered" / "train" / "labels.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; render the train split first")

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} is empty")

    device = device or torch.device("cpu")
    f = torch.tensor([float(r["f"]) for r in rows], dtype=torch.float32, device=device)
    xi = torch.tensor([float(r["xi"]) for r in rows], dtype=torch.float32, device=device)
    theta_ref = torch.tensor(
        [math.radians(float(r["fov_deg"])) / 2.0 for r in rows],
        dtype=torch.float32,
        device=device,
    )
    return {"f": f, "xi": xi, "theta_ref": theta_ref}


def constant_reproj_error(
    f: float, xi: float, labels: Mapping[str, torch.Tensor]
) -> float:
    """Mean reprojection error of one constant ``(f, xi)`` over the labels."""
    f_gt = labels["f"]
    error, _invalid = reprojection_error_px(
        torch.full_like(f_gt, float(f)),
        torch.full_like(f_gt, float(xi)),
        f_gt,
        labels["xi"],
        labels["theta_ref"],
    )
    return float(error.mean())


def search_best_constant(
    labels: Mapping[str, torch.Tensor],
    n_grid: int = DEFAULT_GRID,
    xi_max: float = DEFAULT_XI_MAX,
) -> dict[str, Any]:
    """Grid-search ``(log f, xi)`` for the lowest mean train reprojection error.

    ``log f`` is swept because ``f`` spans most of a decade and the regression
    target is ``log f`` anyway (SPEC.md 4.7).  Candidate errors are accumulated
    on device and only compared at the end, so the loop never stalls on a
    host-device sync.

    The result reports which grid edges the optimum landed on, if any: an
    optimum sitting on a boundary means the search was cut off rather than
    converged, and the error it reports is therefore too high.
    """
    if n_grid < 2:
        raise ValueError(f"n_grid must be >= 2, got {n_grid}")
    if xi_max <= 0.0:
        raise ValueError(f"xi_max must be positive, got {xi_max}")

    f_gt = labels["f"]
    device = f_gt.device
    log_f = torch.linspace(
        math.log(float(f_gt.min())), math.log(float(f_gt.max())), n_grid, device=device
    )
    xi_values = torch.linspace(0.0, float(xi_max), n_grid, device=device)

    scores = torch.empty(n_grid * n_grid, device=device)
    position = 0
    for log_f_value in log_f:
        f_candidate = torch.full_like(f_gt, 1.0) * torch.exp(log_f_value)
        for xi_value in xi_values:
            error, _invalid = reprojection_error_px(
                f_candidate,
                torch.full_like(f_gt, 1.0) * xi_value,
                f_gt,
                labels["xi"],
                labels["theta_ref"],
            )
            scores[position] = error.mean()
            position += 1

    best = int(torch.argmin(scores))
    f_index, xi_index = best // n_grid, best % n_grid

    at_edge = []
    if f_index == 0:
        at_edge.append("f=min")
    elif f_index == n_grid - 1:
        at_edge.append("f=max")
    if xi_index == 0:
        at_edge.append("xi=0")
    elif xi_index == n_grid - 1:
        at_edge.append(f"xi={xi_max:g}")

    return {
        "f": float(torch.exp(log_f[f_index])),
        "xi": float(xi_values[xi_index]),
        "train_reproj_mean_px": float(scores[best]),
        "n_grid": n_grid,
        "xi_max": float(xi_max),
        "n_train": int(f_gt.numel()),
        "at_edge": at_edge,
    }


# --- val evaluation ---------------------------------------------------------


def _evaluate_constant(
    cfg: TrainConfig,
    value: Sequence[float],
    norm_stats: Mapping[str, Any],
    device: torch.device,
) -> tuple[dict[str, float], int]:
    # build_dataset refuses anything outside ("train", "val"), so the held-out
    # splits stay untouched (SPEC.md 6).
    dataset = build_dataset(cfg, EVAL_SPLIT, train=False)
    loader, sampler = build_loader(cfg, dataset, train=False)
    metrics = evaluate(
        ConstantPredictor(value).to(device), loader, sampler, norm_stats, device, cfg
    )
    return metrics, len(dataset)


def baseline_metrics(
    root: str | Path = "data_root",
    batch_size: int = 256,
    num_workers: int = 0,
    n_grid: int = DEFAULT_GRID,
    xi_max: float = DEFAULT_XI_MAX,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Score both constant baselines exactly as ``train.py`` scores a model."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm_stats = load_norm_stats(root)
    mean = np.asarray(norm_stats["mean"], dtype=np.float64)
    std = np.asarray(norm_stats["std"], dtype=np.float64)

    cfg = TrainConfig(
        run_name="constant_baseline",
        # Not a timm name: keeps `uses_channels_last` false for a module that
        # never looks at the tensor layout anyway.
        model="constant",
        pretrained=False,
        data=DataConfig(root=str(root), subset=None, mode="offline"),
        loss="geometric",
        precision="fp32",
        batch_size=batch_size,
        num_workers=num_workers,
    ).validate()

    # --- baseline 1: the train mean, i.e. the standardised origin
    mean_metrics, n_val = _evaluate_constant(cfg, [0.0, 0.0], norm_stats, device)

    # --- baseline 2: the best single (f, xi) on train
    labels = load_train_labels(root, device=device)
    search = search_best_constant(labels, n_grid=n_grid, xi_max=xi_max)
    standardised = (np.array([math.log(search["f"]), search["xi"]]) - mean) / std
    best_metrics, _ = _evaluate_constant(
        cfg, standardised.tolist(), norm_stats, device
    )

    mean_f = float(np.exp(mean[0]))
    return {
        "n_val": n_val,
        "n_train": search["n_train"],
        "n_grid": search["n_grid"],
        "xi_max": search["xi_max"],
        "at_edge": search["at_edge"],
        "mean_params": {
            "f": mean_f,
            "xi": float(mean[1]),
            "train_reproj_mean_px": constant_reproj_error(mean_f, mean[1], labels),
            **mean_metrics,
        },
        "best_constant": {
            "f": search["f"],
            "xi": search["xi"],
            "train_reproj_mean_px": search["train_reproj_mean_px"],
            "at_edge": search["at_edge"],
            **best_metrics,
        },
    }


# --- CLI --------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Constant-prediction baselines for the val split."
    )
    parser.add_argument("--root", default="data_root")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--grid", type=int, default=DEFAULT_GRID,
        help=f"grid points per axis for the search (default {DEFAULT_GRID})",
    )
    parser.add_argument(
        "--xi-max", type=float, default=DEFAULT_XI_MAX,
        help=f"upper end of the xi grid (default {DEFAULT_XI_MAX})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    result = baseline_metrics(
        root=args.root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        n_grid=args.grid,
        xi_max=args.xi_max,
    )

    mean_params = result["mean_params"]
    best = result["best_constant"]

    print(
        f"constant-prediction baselines on {EVAL_SPLIT} "
        f"({result['n_val']} samples)"
    )
    print(
        f"  best constant found on train ({result['n_train']} labels, "
        f"{result['n_grid']}x{result['n_grid']} grid, xi in [0, {result['xi_max']:g}])"
    )
    print()
    print(f"  {'':30s} {'mean params':>14s} {'best constant':>14s}")
    print(f"  {'-' * 30} {'-' * 14} {'-' * 14}")
    print(
        f"  {'predicted f (px)':30s} {mean_params['f']:14.2f} {best['f']:14.2f}"
    )
    print(f"  {'predicted xi':30s} {mean_params['xi']:14.4f} {best['xi']:14.4f}")
    print(
        f"  {'train reproj mean (px)':30s} "
        f"{mean_params['train_reproj_mean_px']:14.4f} "
        f"{best['train_reproj_mean_px']:14.4f}"
    )
    print(f"  {'-' * 30} {'-' * 14} {'-' * 14}")
    for key, label in (
        ("val_reproj_mean_px", "val reproj mean (px)"),
        ("val_reproj_median_px", "val reproj median (px)"),
        ("val_invalid_frac", "val invalid fraction"),
        ("val_f_mae_px", "val f MAE (px)"),
        ("val_xi_mae", "val xi MAE"),
    ):
        print(f"  {label:30s} {mean_params[key]:14.4f} {best[key]:14.4f}")
    print()
    print(
        f"  (loss_geometric {mean_params['val_loss_geometric']:.4f} / "
        f"{best['val_loss_geometric']:.4f}, "
        f"loss_param_l1 {mean_params['val_loss_param_l1']:.4f} / "
        f"{best['val_loss_param_l1']:.4f})"
    )

    if result["at_edge"]:
        print()
        print(
            f"  WARNING: the best constant sits on the grid edge "
            f"({', '.join(result['at_edge'])}). The search was cut off, not "
            f"converged, so this floor is too high -- widen the grid "
            f"(e.g. --xi-max {2 * result['xi_max']:g}) and re-run."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
