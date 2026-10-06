"""Train a CNN or a ViT to regress UCM intrinsics (SPEC.md 3.13).

One script for both architectures: ``timm.create_model(name, pretrained,
num_classes=2)`` and nothing architecture-specific beyond the memory format.

Length is counted in optimiser steps, never epochs (SPEC.md 3.13), so the
loop walks the loader repeatedly and stops at ``total_steps``.  The epoch is
handed to :class:`datasets.sampler.EpochSampler` before each pass, which is
what makes it reach a ``persistent_workers`` DataLoader (SPEC.md 4.12.1).

**This script never opens the test or stress split.**  :data:`ALLOWED_SPLITS`
is the whole list it is permitted to touch, and :func:`build_dataset` refuses
anything else.

Everything tunable lives in ``configs/train/*.yaml``; the CLI carries only the
mode flags, so a run is fully described by its config file and that is what
gets logged to MLflow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from datasets import transforms as T
from datasets.fisheye_dataset import FisheyeDataset, load_norm_stats
from datasets.sampler import EpochSampler
from metrics.geometric import (
    denormalize,
    geometric_loss,
    param_l1_loss,
    reprojection_error_px,
)

__all__ = [
    "ALLOWED_SPLITS",
    "TrainConfig",
    "load_config",
    "build_model",
    "build_optimizer",
    "build_dataset",
    "build_loader",
    "evaluate",
    "train",
    "main",
]

# SPEC.md 6: test and stress are not for training to look at, ever.
ALLOWED_SPLITS = ("train", "val")

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TRACKING_URI = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"


# --- config -----------------------------------------------------------------


@dataclass
class DataConfig:
    root: str = "data_root"
    subset: str | None = "p100"
    mode: str = "offline"  # offline | online


@dataclass
class OptimConfig:
    lr: float = 3.0e-4
    weight_decay: float = 1.0e-4
    warmup_steps: int = 500


@dataclass
class TrainConfig:
    """SPEC.md 3.13, plus ``grad_clip`` and ``num_workers``."""

    run_name: str = "run"
    model: str = "resnet34"
    pretrained: bool = True
    data: DataConfig = field(default_factory=DataConfig)
    loss: str = "geometric"  # geometric | param_l1
    optim: OptimConfig = field(default_factory=OptimConfig)
    schedule: str = "cosine"
    total_steps: int = 6000
    batch_size: int = 128
    precision: str = "bf16"
    eval_every: int = 500
    seed: int = 0
    grad_clip: float = 1.0
    num_workers: int = 6

    def validate(self) -> "TrainConfig":
        if self.loss not in ("geometric", "param_l1"):
            raise ValueError(f"loss must be geometric or param_l1, got {self.loss!r}")
        if self.schedule != "cosine":
            raise ValueError(f"only the cosine schedule is implemented, got {self.schedule!r}")
        if self.precision not in ("bf16", "fp32"):
            raise ValueError(f"precision must be bf16 or fp32, got {self.precision!r}")
        if self.data.mode not in ("offline", "online"):
            raise ValueError(f"data.mode must be offline or online, got {self.data.mode!r}")
        for name in ("total_steps", "batch_size", "eval_every"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if self.optim.warmup_steps < 0:
            raise ValueError("optim.warmup_steps must be >= 0")
        if self.num_workers < 0:
            raise ValueError("num_workers must be >= 0")
        return self

    def flat(self) -> dict[str, Any]:
        """The fully resolved config, flattened for MLflow params."""
        out: dict[str, Any] = {}
        for key, value in asdict(self).items():
            if isinstance(value, dict):
                out.update({f"{key}.{k}": v for k, v in value.items()})
            else:
                out[key] = value
        return out


def load_config(path: str | Path | None = None) -> TrainConfig:
    """Read a ``configs/train/*.yaml``; absent keys take the SPEC.md 3.13 defaults."""
    raw: dict[str, Any] = {}
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: expected a mapping at the top level")

    known = {f.name for f in TrainConfig.__dataclass_fields__.values()}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"{path}: unknown config keys {sorted(unknown)}")

    data = DataConfig(**(raw.pop("data", None) or {}))
    optim = OptimConfig(**(raw.pop("optim", None) or {}))
    return TrainConfig(data=data, optim=optim, **raw).validate()


# --- model ------------------------------------------------------------------


def build_model(cfg: TrainConfig, device: torch.device) -> torch.nn.Module:
    """``timm`` model with a two-output head (SPEC.md 3.13)."""
    import timm

    model = timm.create_model(cfg.model, pretrained=cfg.pretrained, num_classes=2)
    model = model.to(device)
    if uses_channels_last(cfg.model):
        # Convolutions hit the tensor cores far harder in NHWC; a ViT is pure
        # matmul on tokens, so the layout buys it nothing.
        model = model.to(memory_format=torch.channels_last)
    return model


def uses_channels_last(model_name: str) -> bool:
    return "resnet" in model_name.lower()


def no_decay_names(model: torch.nn.Module) -> set[str]:
    """Parameter names the model itself asks to be left out of weight decay.

    timm exposes ``no_weight_decay()`` on architectures that have parameters
    decay would damage -- a ViT's ``pos_embed`` and ``cls_token`` are learned
    positional structure, not weights, and shrinking them toward zero degrades
    the embedding for no regularisation benefit.
    """
    getter = getattr(model, "no_weight_decay", None)
    if getter is None:
        return set()
    try:
        return {str(name) for name in getter()}
    except (TypeError, AttributeError):
        return set()


def build_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> torch.optim.AdamW:
    """AdamW with weight decay applied only where it belongs.

    Decaying one-dimensional parameters -- biases, and BatchNorm/LayerNorm
    scales and shifts -- pulls normalisation toward the identity and shifts the
    activation statistics; it is a standard and worthwhile exception, and it
    matters more here because the ViT config runs a decay of 0.05.  Everything
    with two or more dimensions (conv and linear weights) is decayed as the
    config says.
    """
    skip = no_decay_names(model)

    def is_exempt(name: str, param: torch.nn.Parameter) -> bool:
        if param.ndim <= 1:
            return True
        return name in skip or any(name.startswith(f"{entry}.") for entry in skip)

    decayed: list[torch.nn.Parameter] = []
    exempt: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (exempt if is_exempt(name, param) else decayed).append(param)

    groups = [
        {"params": decayed, "weight_decay": cfg.optim.weight_decay},
        {"params": exempt, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.optim.lr)


# --- data -------------------------------------------------------------------


def build_dataset(cfg: TrainConfig, split: str, train: bool) -> FisheyeDataset:
    """Offline dataset for one split, with augmentation only on train."""
    if split not in ALLOWED_SPLITS:
        raise ValueError(
            f"train.py may only read {ALLOWED_SPLITS}, refusing {split!r} "
            "(SPEC.md 6: test and stress stay untouched during training)"
        )
    if cfg.data.mode != "offline":
        raise NotImplementedError(
            "data.mode='online' needs the panorama cache; build it with "
            "`python -m datasets.online_dataset --root <root> --split train`"
        )
    return FisheyeDataset(
        root=cfg.data.root,
        split=split,
        subset=cfg.data.subset if split == "train" else None,
        transform=T.build_train_transform() if train else None,
        aug_seed=cfg.seed,
    )


def build_loader(
    cfg: TrainConfig, dataset: FisheyeDataset, train: bool
) -> tuple[DataLoader, EpochSampler]:
    sampler = EpochSampler(len(dataset), shuffle=train, seed=cfg.seed)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
        drop_last=train,
    )
    return loader, sampler


# --- schedule ---------------------------------------------------------------


def lr_scale(step: int, total_steps: int, warmup_steps: int) -> float:
    """Linear warmup then cosine decay, counted in steps (SPEC.md 3.13)."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))


# --- loss / metrics ---------------------------------------------------------


def _to_device(batch: Mapping[str, Any], device: torch.device, channels_last: bool):
    image = batch["image"].to(device, non_blocking=True)
    if channels_last:
        image = image.to(memory_format=torch.channels_last)
    return (
        image,
        batch["target"].to(device, non_blocking=True),
        batch["f"].to(device, non_blocking=True).float(),
        batch["xi"].to(device, non_blocking=True).float(),
        batch["theta_ref"].to(device, non_blocking=True).float(),
    )


def compute_loss(
    name: str,
    pred: torch.Tensor,
    target: torch.Tensor,
    f_gt: torch.Tensor,
    xi_gt: torch.Tensor,
    theta_ref: torch.Tensor,
    norm_stats: Mapping[str, Any],
) -> torch.Tensor:
    if name == "geometric":
        return geometric_loss(pred, f_gt, xi_gt, theta_ref, norm_stats)
    return param_l1_loss(pred, target)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    sampler: EpochSampler,
    norm_stats: Mapping[str, Any],
    device: torch.device,
    cfg: TrainConfig,
) -> dict[str, float]:
    """Validation metrics (SPEC.md 3.13 and the task spec).

    Both losses are reported regardless of which one is being optimised, so the
    two training objectives stay comparable across runs.
    """
    model.eval()
    sampler.set_epoch(0)  # evaluation is deterministic; no augmentation anyway
    channels_last = uses_channels_last(cfg.model)

    errors: list[torch.Tensor] = []
    invalids: list[torch.Tensor] = []
    f_abs: list[torch.Tensor] = []
    xi_abs: list[torch.Tensor] = []
    geo_sum = l1_sum = 0.0
    seen = 0

    for batch in loader:
        image, target, f_gt, xi_gt, theta_ref = _to_device(batch, device, channels_last)
        with autocast(cfg, device):
            pred = model(image)
        pred = pred.float()

        f_pred, xi_pred = denormalize(pred, norm_stats)
        error, invalid = reprojection_error_px(f_pred, xi_pred, f_gt, xi_gt, theta_ref)

        count = image.shape[0]
        errors.append(error.cpu())
        invalids.append(invalid.cpu())
        f_abs.append((f_pred - f_gt).abs().cpu())
        xi_abs.append((xi_pred - xi_gt).abs().cpu())
        geo_sum += float(
            geometric_loss(pred, f_gt, xi_gt, theta_ref, norm_stats)
        ) * count
        l1_sum += float(param_l1_loss(pred, target)) * count
        seen += count

    model.train()
    all_errors = torch.cat(errors)
    return {
        "val_reproj_mean_px": float(all_errors.mean()),
        "val_reproj_median_px": float(all_errors.median()),
        "val_invalid_frac": float(torch.cat(invalids).mean()),
        "val_f_mae_px": float(torch.cat(f_abs).mean()),
        "val_xi_mae": float(torch.cat(xi_abs).mean()),
        "val_loss_geometric": geo_sum / max(seen, 1),
        "val_loss_param_l1": l1_sum / max(seen, 1),
    }


def autocast(cfg: TrainConfig, device: torch.device):
    if cfg.precision == "bf16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type=device.type, enabled=False)


# --- provenance -------------------------------------------------------------


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            timeout=10, check=False, cwd=PROJECT_ROOT,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 else "unknown"


def meta_hash(root: str | Path) -> str:
    """SHA-256 of the train split's ``meta.json``: which dataset this run saw."""
    path = Path(root) / "rendered" / "train" / "meta.json"
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- training ---------------------------------------------------------------


def _infinite(loader: DataLoader, sampler: EpochSampler) -> Iterator[tuple[int, Any]]:
    """Walk the loader forever, advancing the sampler's epoch each pass."""
    epoch = 0
    while True:
        sampler.set_epoch(epoch)
        for batch in loader:
            yield epoch, batch
        epoch += 1


def train(
    cfg: TrainConfig,
    device: torch.device | None = None,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    out_dir: str | Path | None = None,
    log=print,
) -> dict[str, Any]:
    """Run the full loop and return the best metrics."""
    import mlflow

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    norm_stats = load_norm_stats(cfg.data.root)
    train_set = build_dataset(cfg, "train", train=True)
    val_set = build_dataset(cfg, "val", train=False)
    train_loader, train_sampler = build_loader(cfg, train_set, train=True)
    val_loader, val_sampler = build_loader(cfg, val_set, train=False)

    model = build_model(cfg, device)
    optimiser = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser,
        lambda step: lr_scale(step, cfg.total_steps, cfg.optim.warmup_steps),
    )

    out_dir = Path(out_dir) if out_dir else PROJECT_ROOT / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / f"{cfg.run_name}_best.pt"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("fisheye-calib")

    best = math.inf
    best_metrics: dict[str, float] = {}
    started = time.time()

    with mlflow.start_run(run_name=cfg.run_name):
        mlflow.log_params(cfg.flat())
        mlflow.set_tags(
            {
                "git_commit": git_commit(),
                "train_meta_sha256": meta_hash(cfg.data.root),
                "device": str(device),
                "n_train": len(train_set),
                "n_val": len(val_set),
            }
        )
        log(
            f"{cfg.run_name}: {cfg.model} on {device}, "
            f"{len(train_set)} train / {len(val_set)} val, "
            f"{cfg.total_steps} steps of batch {cfg.batch_size}"
        )

        model.train()
        batches = _infinite(train_loader, train_sampler)
        for step in range(cfg.total_steps):
            epoch, batch = next(batches)
            image, target, f_gt, xi_gt, theta_ref = _to_device(
                batch, device, uses_channels_last(cfg.model)
            )

            with autocast(cfg, device):
                pred = model(image)
            loss = compute_loss(
                cfg.loss, pred.float(), target, f_gt, xi_gt, theta_ref, norm_stats
            )

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.grad_clip
            )
            optimiser.step()
            scheduler.step()

            if step % 50 == 0 or step == cfg.total_steps - 1:
                mlflow.log_metrics(
                    {
                        "train_loss": float(loss.detach()),
                        "grad_norm": float(grad_norm),
                        "lr": float(scheduler.get_last_lr()[0]),
                        "epoch": float(epoch),
                    },
                    step=step,
                )

            is_last = step == cfg.total_steps - 1
            if (step + 1) % cfg.eval_every == 0 or is_last:
                metrics = evaluate(
                    model, val_loader, val_sampler, norm_stats, device, cfg
                )
                mlflow.log_metrics(metrics, step=step)
                log(
                    f"  step {step + 1}/{cfg.total_steps} "
                    f"train_loss {float(loss.detach()):.4f}  "
                    f"val_reproj {metrics['val_reproj_mean_px']:.2f} px  "
                    f"f_mae {metrics['val_f_mae_px']:.2f}  "
                    f"xi_mae {metrics['val_xi_mae']:.4f}"
                )
                if metrics["val_reproj_mean_px"] < best:
                    best = metrics["val_reproj_mean_px"]
                    best_metrics = dict(metrics)
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "config": cfg.flat(),
                            "step": step,
                            "metrics": metrics,
                            "norm_stats": dict(norm_stats),
                        },
                        best_path,
                    )

        if best_path.is_file():
            mlflow.log_artifact(str(best_path))
        mlflow.log_metrics(
            {"best_val_reproj_mean_px": best, "elapsed_sec": time.time() - started}
        )

    log(f"best val reprojection error: {best:.3f} px -> {best_path}")
    return {"best_val_reproj_mean_px": best, "best_metrics": best_metrics,
            "checkpoint": str(best_path)}


# --- diagnostic modes -------------------------------------------------------


def overfit_batch(cfg: TrainConfig, steps: int, device: torch.device | None = None,
                  log=print) -> dict[str, Any]:
    """Drive one fixed batch to zero: a sanity check on the whole loop.

    If the loss does not collapse here, the problem is the model, the loss or
    the label plumbing -- not the data pipeline or the schedule.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    norm_stats = load_norm_stats(cfg.data.root)
    dataset = build_dataset(cfg, "train", train=False)  # no augmentation
    loader, sampler = build_loader(cfg, dataset, train=False)
    sampler.set_epoch(0)
    batch = next(iter(loader))

    model = build_model(cfg, device)
    model.train()
    optimiser = build_optimizer(model, cfg)

    image, target, f_gt, xi_gt, theta_ref = _to_device(
        batch, device, uses_channels_last(cfg.model)
    )
    log(
        f"overfit {cfg.model} ({cfg.loss}) on one batch of {image.shape[0]} "
        f"for {steps} steps, device={device}"
    )

    history: list[float] = []
    started = time.time()
    for step in range(steps):
        with autocast(cfg, device):
            pred = model(image)
        loss = compute_loss(
            cfg.loss, pred.float(), target, f_gt, xi_gt, theta_ref, norm_stats
        )
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimiser.step()

        value = float(loss.detach())
        history.append(value)
        if step % 10 == 0 or step == steps - 1:
            log(f"  step {step:4d}  loss {value:.6f}")

    model.eval()
    with torch.no_grad(), autocast(cfg, device):
        pred = model(image).float()
    f_pred, xi_pred = denormalize(pred, norm_stats)
    error, invalid = reprojection_error_px(f_pred, xi_pred, f_gt, xi_gt, theta_ref)
    log(
        f"  final: loss {history[-1]:.6f}  reproj {float(error.mean()):.3f} px  "
        f"f_mae {float((f_pred - f_gt).abs().mean()):.3f}  "
        f"xi_mae {float((xi_pred - xi_gt).abs().mean()):.5f}  "
        f"invalid {float(invalid.mean()):.4f}  ({time.time() - started:.1f}s)"
    )
    return {"history": history, "final_loss": history[-1],
            "reproj_px": float(error.mean())}


def benchmark(cfg: TrainConfig, steps: int, device: torch.device | None = None,
              log=print) -> dict[str, float]:
    """Steady-state training throughput and peak GPU memory."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    norm_stats = load_norm_stats(cfg.data.root)
    dataset = build_dataset(cfg, "train", train=True)
    loader, sampler = build_loader(cfg, dataset, train=True)

    model = build_model(cfg, device)
    model.train()
    optimiser = build_optimizer(model, cfg)
    channels_last = uses_channels_last(cfg.model)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    warmup = min(10, max(1, steps // 10))
    images = 0
    started = 0.0
    batches = _infinite(loader, sampler)

    for step in range(steps + warmup):
        _epoch, batch = next(batches)
        image, target, f_gt, xi_gt, theta_ref = _to_device(batch, device, channels_last)
        with autocast(cfg, device):
            pred = model(image)
        loss = compute_loss(
            cfg.loss, pred.float(), target, f_gt, xi_gt, theta_ref, norm_stats
        )
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimiser.step()

        if step == warmup - 1:
            # Start the clock only once the pipeline and allocator have settled.
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.time()
        elif step >= warmup:
            images += image.shape[0]

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - started
    peak = (
        torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
    )
    throughput = images / elapsed if elapsed > 0 else float("nan")

    log(
        f"benchmark {cfg.model} ({cfg.loss}), batch {cfg.batch_size}, "
        f"{cfg.num_workers} workers, {cfg.precision} on {device}"
    )
    log(
        f"  {steps} steps / {images} images in {elapsed:.2f}s  "
        f"-> {throughput:.1f} img/s  ({1000 * elapsed / max(steps, 1):.1f} ms/step)"
    )
    log(f"  peak GPU memory: {peak:.2f} GiB")
    return {"img_per_s": throughput, "ms_per_step": 1000 * elapsed / max(steps, 1),
            "peak_gib": peak}


# --- CLI --------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a fisheye intrinsics regressor (SPEC.md 3.13)."
    )
    parser.add_argument("--config", required=True, help="configs/train/*.yaml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--overfit-batch", type=int, default=None, metavar="N",
        help="repeat one fixed batch for N steps and print the loss every 10",
    )
    mode.add_argument(
        "--benchmark", type=int, default=None, metavar="N",
        help="run N steps and report throughput and peak GPU memory",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    cfg = load_config(args.config)

    if args.overfit_batch is not None:
        overfit_batch(cfg, args.overfit_batch)
    elif args.benchmark is not None:
        benchmark(cfg, args.benchmark)
    else:
        train(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
