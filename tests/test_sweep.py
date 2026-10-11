"""Tests for ``sweep.py`` (SPEC.md 3.13).

The two behaviours that make a sweep survivable over several sessions: a
FINISHED run is not repeated, and one failing arm does not take the rest of
the sweep with it.  Both are exercised against a real MLflow store and a real
(tiny) training run, not a mock of the training function.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml

from data.render_dataset import LABEL_FIELDS
from data.sampling import SamplingConfig, sample_camera
from sweep import (
    EXPERIMENT_NAME,
    SweepResult,
    finished_run_exists,
    read_config_list,
    run_sweep,
)

SIZE = 32
VIEWS = 4
TRAIN_ASSETS = ["alpha", "beta"]
VAL_ASSETS = ["gamma"]
TINY_MODEL = "resnet10t"


# --- a miniature data_root --------------------------------------------------


def _write_split(root: Path, split: str, assets: list[str], seed: int, stats: bool):
    out_dir = root / "rendered" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cfg = SamplingConfig(image_size=SIZE, f_min_px=5.0)
    total = len(assets) * VIEWS

    images = np.lib.format.open_memmap(
        out_dir / "images.npy", mode="w+", dtype=np.uint8, shape=(total, SIZE, SIZE, 3)
    )
    rows: list[dict[str, Any]] = []
    for position, asset_id in enumerate(assets):
        for view in range(VIEWS):
            index = position * VIEWS + view
            images[index] = rng.integers(0, 256, (SIZE, SIZE, 3), dtype=np.uint8)
            sample = sample_camera(np.random.default_rng(700 + index), cfg)
            rows.append(
                {
                    "index": index, "asset_id": asset_id, "view_idx": view,
                    "f": sample.f, "xi": sample.xi, "fov_deg": sample.fov_deg,
                    "circular": sample.circular, "r_ref": sample.r_ref,
                    "yaw_deg": sample.yaw_deg, "pitch_deg": sample.pitch_deg,
                    "roll_deg": sample.roll_deg, "category": "Interiors",
                    "texture": 12.0,
                }
            )
    images.flush()
    del images

    with (out_dir / "labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    meta: dict[str, Any] = {"split": split, "image_size": SIZE, "n_images": total}
    if stats:
        log_f = np.log([float(r["f"]) for r in rows])
        xi = np.array([float(r["xi"]) for r in rows])
        stacked = np.stack([log_f, xi], axis=1)
        meta["norm_stats"] = {
            "fields": ["log_f", "xi"],
            "mean": stacked.mean(axis=0).tolist(),
            "std": stacked.std(axis=0).tolist(),
        }
    (out_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


@pytest.fixture(scope="module")
def data_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("sweep_root")
    _write_split(root, "train", TRAIN_ASSETS, seed=0, stats=True)
    _write_split(root, "val", VAL_ASSETS, seed=1, stats=False)
    (root / "splits.json").write_text(
        json.dumps(
            {
                "train": TRAIN_ASSETS, "val": VAL_ASSETS, "test": [], "stress": [],
                "train_subsets": {"p100": TRAIN_ASSETS},
            }
        ),
        encoding="utf-8",
    )
    return root


def write_config(
    directory: Path, run_name: str, data_root: Path, model: str = TINY_MODEL
) -> Path:
    path = directory / f"{run_name}.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "run_name": run_name,
                "model": model,
                "pretrained": False,
                "data": {"root": str(data_root), "subset": "p100", "mode": "offline"},
                "loss": "geometric",
                "optim": {"lr": 1e-3, "weight_decay": 1e-4, "warmup_steps": 1},
                "total_steps": 2,
                "batch_size": 4,
                "precision": "fp32",
                "eval_every": 2,
                "seed": 0,
                "num_workers": 0,
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def tracking_uri(tmp_path: Path) -> str:
    return f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"


def cpu_train(cfg, **kwargs):
    """The real training function, pinned to CPU."""
    from train import train

    return train(cfg, device=torch.device("cpu"), **kwargs)


# --- the list file ----------------------------------------------------------


def test_list_file_ignores_comments_and_blanks(tmp_path: Path) -> None:
    path = tmp_path / "sweep.txt"
    path.write_text(
        "# a comment\n"
        "\n"
        "configs/train/a.yaml\n"
        "   \n"
        "configs/train/b.yaml  # trailing comment\n"
        "  # indented comment\n",
        encoding="utf-8",
    )
    assert read_config_list(path) == [
        Path("configs/train/a.yaml"),
        Path("configs/train/b.yaml"),
    ]


def test_missing_list_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_config_list(tmp_path / "nope.txt")


def test_empty_list_file_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("# nothing here\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lists no configs"):
        read_config_list(path)


def test_shipped_sweep_list_points_at_real_configs() -> None:
    """Every listed arm must exist, load, and test a distinct learning rate.

    Deliberately not pinned to a fixed arm count: the list grows as the sweep
    is extended, and a test that has to be edited for each new arm would just
    get edited without being read.
    """
    configs = read_config_list("configs/sweeps/resnet34_lr.txt")
    assert len(configs) >= 3
    for path in configs:
        assert path.is_file(), path

    from train import load_config

    loaded = [load_config(p) for p in configs]
    rates = [cfg.optim.lr for cfg in loaded]
    assert len(set(rates)) == len(rates), f"duplicate learning rates: {rates}"
    assert all(rate > 0 for rate in rates)

    # The arms that have already been run must stay in the list, or a re-run
    # would silently drop them from the comparison.
    assert {1.0e-4, 3.0e-4, 1.0e-3} <= set(rates)

    # The sweep varies the learning rate and nothing else.
    varying = {
        key
        for key in loaded[0].flat()
        if len({cfg.flat()[key] for cfg in loaded}) > 1
    }
    assert varying == {"run_name", "optim.lr"}, varying


# --- skipping finished runs -------------------------------------------------


def test_finished_run_is_skipped(
    data_root: Path, tmp_path: Path, tracking_uri: str
) -> None:
    """SPEC.md 3.13: a FINISHED run_name is not trained again."""
    config = write_config(tmp_path, "arm_a", data_root)
    listing = tmp_path / "sweep.txt"
    listing.write_text(f"{config}\n", encoding="utf-8")

    first = run_sweep(listing, tracking_uri=tracking_uri, out_dir=tmp_path / "ckpt",
                      log=lambda _m: None, train_fn=cpu_train)
    assert first.done == 1 and first.skipped == 0
    assert first.results[0].best_val_reproj_mean_px is not None

    # Same list again: the arm is now FINISHED, so it must not retrain.
    calls: list[str] = []

    def must_not_run(cfg, **kwargs):
        calls.append(cfg.run_name)
        return cpu_train(cfg, **kwargs)

    second = run_sweep(listing, tracking_uri=tracking_uri, out_dir=tmp_path / "ckpt",
                       log=lambda _m: None, train_fn=must_not_run)
    assert second.skipped == 1 and second.done == 0
    assert calls == [], "a finished run was trained again"
    assert second.results[0].status == "skipped"


def test_skip_message_names_the_run(
    data_root: Path, tmp_path: Path, tracking_uri: str
) -> None:
    config = write_config(tmp_path, "arm_named", data_root)
    listing = tmp_path / "sweep.txt"
    listing.write_text(f"{config}\n", encoding="utf-8")

    run_sweep(listing, tracking_uri=tracking_uri, out_dir=tmp_path / "ckpt",
              log=lambda _m: None, train_fn=cpu_train)

    lines: list[str] = []
    run_sweep(listing, tracking_uri=tracking_uri, out_dir=tmp_path / "ckpt",
              log=lines.append, train_fn=cpu_train)
    joined = "\n".join(lines)
    assert "arm_named" in joined and "skipping" in joined


def test_finished_run_exists_is_specific(
    data_root: Path, tmp_path: Path, tracking_uri: str
) -> None:
    config = write_config(tmp_path, "arm_present", data_root)
    listing = tmp_path / "sweep.txt"
    listing.write_text(f"{config}\n", encoding="utf-8")
    run_sweep(listing, tracking_uri=tracking_uri, out_dir=tmp_path / "ckpt",
              log=lambda _m: None, train_fn=cpu_train)

    assert finished_run_exists("arm_present", tracking_uri, EXPERIMENT_NAME)
    assert not finished_run_exists("arm_absent", tracking_uri, EXPERIMENT_NAME)


def test_no_store_means_nothing_is_finished(tmp_path: Path) -> None:
    empty = f"sqlite:///{(tmp_path / 'fresh.db').as_posix()}"
    assert not finished_run_exists("anything", empty, EXPERIMENT_NAME)


# --- a failing arm must not end the sweep -----------------------------------


def test_failure_does_not_stop_the_sweep(
    data_root: Path, tmp_path: Path, tracking_uri: str
) -> None:
    """A bad arm is reported; the arms after it still run."""
    good_before = write_config(tmp_path, "before", data_root)
    broken = write_config(tmp_path, "broken", data_root, model="not_a_real_model_xyz")
    good_after = write_config(tmp_path, "after", data_root)

    listing = tmp_path / "sweep.txt"
    listing.write_text(
        f"{good_before}\n{broken}\n{good_after}\n", encoding="utf-8"
    )

    lines: list[str] = []
    summary = run_sweep(listing, tracking_uri=tracking_uri,
                        out_dir=tmp_path / "ckpt", log=lines.append,
                        train_fn=cpu_train)

    assert summary.done == 2, [r.status for r in summary.results]
    assert summary.failed == 1
    statuses = {r.run_name: r.status for r in summary.results}
    assert statuses == {"before": "done", "broken": "failed", "after": "done"}

    failed = next(r for r in summary.results if r.status == "failed")
    assert failed.error
    assert failed.best_val_reproj_mean_px is None

    # The arm after the failure really trained.
    after = next(r for r in summary.results if r.run_name == "after")
    assert after.best_val_reproj_mean_px is not None
    assert math.isfinite(after.best_val_reproj_mean_px)


def test_a_broken_yaml_is_just_a_failed_arm(
    data_root: Path, tmp_path: Path, tracking_uri: str
) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"nonsense_key": 1}), encoding="utf-8")
    good = write_config(tmp_path, "survivor", data_root)

    listing = tmp_path / "sweep.txt"
    listing.write_text(f"{bad}\n{good}\n", encoding="utf-8")

    summary = run_sweep(listing, tracking_uri=tracking_uri,
                        out_dir=tmp_path / "ckpt", log=lambda _m: None,
                        train_fn=cpu_train)
    assert summary.failed == 1 and summary.done == 1
    assert summary.results[1].run_name == "survivor"


def test_a_failed_arm_is_retried_next_time(
    data_root: Path, tmp_path: Path, tracking_uri: str
) -> None:
    """Only FINISHED runs are skipped, so a failure is not sticky."""
    broken = write_config(tmp_path, "retry_me", data_root, model="not_a_real_model_xyz")
    listing = tmp_path / "sweep.txt"
    listing.write_text(f"{broken}\n", encoding="utf-8")

    first = run_sweep(listing, tracking_uri=tracking_uri, out_dir=tmp_path / "ckpt",
                      log=lambda _m: None, train_fn=cpu_train)
    assert first.failed == 1

    # Fix the arm and re-run: it must be attempted, not skipped.
    write_config(tmp_path, "retry_me", data_root)
    second = run_sweep(listing, tracking_uri=tracking_uri, out_dir=tmp_path / "ckpt",
                       log=lambda _m: None, train_fn=cpu_train)
    assert second.done == 1 and second.skipped == 0


# --- the summary ------------------------------------------------------------


def test_summary_reports_each_arm(
    data_root: Path, tmp_path: Path, tracking_uri: str
) -> None:
    good = write_config(tmp_path, "sum_ok", data_root)
    broken = write_config(tmp_path, "sum_bad", data_root, model="not_a_real_model_xyz")
    listing = tmp_path / "sweep.txt"
    listing.write_text(f"{good}\n{broken}\n", encoding="utf-8")

    lines: list[str] = []
    summary = run_sweep(listing, tracking_uri=tracking_uri,
                        out_dir=tmp_path / "ckpt", log=lines.append,
                        train_fn=cpu_train)

    joined = "\n".join(lines)
    assert "sweep summary: 1 done, 0 skipped, 1 failed" in joined
    assert "sum_ok" in joined and "sum_bad" in joined
    assert "px" in joined, "the best score should appear in the summary"
    assert summary.done == 1 and summary.failed == 1


def test_summary_counts() -> None:
    from sweep import SweepSummary

    summary = SweepSummary(
        results=[
            SweepResult("a", "a", "done", 1.0),
            SweepResult("b", "b", "skipped"),
            SweepResult("c", "c", "failed", error="boom"),
            SweepResult("d", "d", "done", 2.0),
        ]
    )
    assert (summary.done, summary.skipped, summary.failed) == (2, 1, 1)


def test_cli_exit_code_reflects_failure(
    data_root: Path, tmp_path: Path, tracking_uri: str
) -> None:
    """A non-zero exit lets a shell loop notice that an arm died."""
    import sweep as sweep_module

    broken = write_config(tmp_path, "cli_bad", data_root, model="not_a_real_model_xyz")
    listing = tmp_path / "sweep.txt"
    listing.write_text(f"{broken}\n", encoding="utf-8")

    summary = run_sweep(listing, tracking_uri=tracking_uri,
                        out_dir=tmp_path / "ckpt", log=lambda _m: None,
                        train_fn=cpu_train)
    assert (1 if summary.failed else 0) == 1
    assert sweep_module.build_parser().parse_args(["some/list.txt"]).list == (
        "some/list.txt"
    )
