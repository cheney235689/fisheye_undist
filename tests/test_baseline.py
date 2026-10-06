"""Tests for ``tools/baseline.py``.

The property that makes the second baseline worth computing: the parameter
pair that minimises the reprojection error is not the mean of the parameters,
so the grid search can only ever do at least as well as "mean params".  If it
ever came out worse, the search or the error function would be wrong.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from data.render_dataset import LABEL_FIELDS
from data.sampling import SamplingConfig, sample_camera
from tools.baseline import (
    ConstantPredictor,
    constant_reproj_error,
    load_train_labels,
    search_best_constant,
)


def _write_train_labels(root: Path, count: int = 64, seed: int = 0) -> Path:
    """A train split with labels only -- the search never opens an image."""
    out_dir = root / "rendered" / "train"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = SamplingConfig(image_size=224)
    rng = np.random.default_rng(seed)

    rows = []
    for index in range(count):
        sample = sample_camera(rng, cfg)
        rows.append(
            {
                "index": index, "asset_id": f"a{index // 4}", "view_idx": index % 4,
                "f": sample.f, "xi": sample.xi, "fov_deg": sample.fov_deg,
                "circular": sample.circular, "r_ref": sample.r_ref,
                "yaw_deg": sample.yaw_deg, "pitch_deg": sample.pitch_deg,
                "roll_deg": sample.roll_deg, "category": "Interiors",
                "texture": 10.0,
            }
        )
    path = out_dir / "labels.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture
def labels(tmp_path: Path):
    _write_train_labels(tmp_path)
    return load_train_labels(tmp_path)


def test_best_constant_beats_the_mean_parameters(labels) -> None:
    """The whole point: the search can only improve on the mean."""
    mean_f = float(torch.exp(torch.log(labels["f"]).mean()))
    mean_xi = float(labels["xi"].mean())
    mean_error = constant_reproj_error(mean_f, mean_xi, labels)

    best = search_best_constant(labels, n_grid=40)

    assert best["train_reproj_mean_px"] <= mean_error + 1e-6, (
        best["train_reproj_mean_px"],
        mean_error,
    )
    # And the reported error really is that constant's error.
    assert constant_reproj_error(best["f"], best["xi"], labels) == pytest.approx(
        best["train_reproj_mean_px"], rel=1e-5
    )


def test_best_constant_stays_inside_the_grid(labels) -> None:
    best = search_best_constant(labels, n_grid=20, xi_max=2.5)
    assert float(labels["f"].min()) <= best["f"] <= float(labels["f"].max())
    assert 0.0 <= best["xi"] <= 2.5
    assert best["n_grid"] == 20
    assert best["xi_max"] == pytest.approx(2.5)
    assert best["n_train"] == labels["f"].numel()


def test_edge_hits_are_reported(labels) -> None:
    """An optimum on a boundary means the grid was too narrow, not converged."""
    clipped = search_best_constant(labels, n_grid=20, xi_max=0.2)
    assert "xi=0.2" in clipped["at_edge"], clipped

    wide = search_best_constant(labels, n_grid=40, xi_max=6.0)
    assert "xi=6" not in wide["at_edge"], wide
    # A wider grid can only find a lower (or equal) error.
    assert wide["train_reproj_mean_px"] <= clipped["train_reproj_mean_px"] + 1e-6


def test_search_rejects_a_non_positive_xi_max(labels) -> None:
    with pytest.raises(ValueError, match="xi_max"):
        search_best_constant(labels, n_grid=10, xi_max=0.0)


def test_a_finer_grid_never_scores_worse(labels) -> None:
    coarse = search_best_constant(labels, n_grid=10)
    fine = search_best_constant(labels, n_grid=50)
    assert fine["train_reproj_mean_px"] <= coarse["train_reproj_mean_px"] + 1e-6


def test_perfect_labels_give_a_zero_error_constant(tmp_path: Path) -> None:
    """With one camera repeated, the best constant is that camera, error 0."""
    out_dir = tmp_path / "rendered" / "train"
    out_dir.mkdir(parents=True, exist_ok=True)
    f, xi, fov = 150.0, 0.75, 160.0
    with (out_dir / "labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        for index in range(8):
            writer.writerow(
                {
                    "index": index, "asset_id": "a", "view_idx": index,
                    "f": f, "xi": xi, "fov_deg": fov, "circular": False,
                    "r_ref": 158.4, "yaw_deg": 0.0, "pitch_deg": 0.0,
                    "roll_deg": 0.0, "category": "Interiors", "texture": 10.0,
                }
            )
    labels = load_train_labels(tmp_path)
    assert constant_reproj_error(f, xi, labels) == pytest.approx(0.0, abs=1e-4)
    best = search_best_constant(labels, n_grid=8)
    assert best["train_reproj_mean_px"] < constant_reproj_error(f, 0.0, labels)


def test_load_train_labels_reads_no_images(tmp_path: Path) -> None:
    _write_train_labels(tmp_path, count=8)
    assert not (tmp_path / "rendered" / "train" / "images.npy").exists()
    labels = load_train_labels(tmp_path)
    assert set(labels) == {"f", "xi", "theta_ref"}
    assert all(t.shape == (8,) for t in labels.values())
    assert torch.all(labels["theta_ref"] > 0)


def test_load_train_labels_needs_the_split(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="render the train split"):
        load_train_labels(tmp_path)


def test_theta_ref_is_half_the_fov(tmp_path: Path) -> None:
    _write_train_labels(tmp_path, count=16)
    path = tmp_path / "rendered" / "train" / "labels.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels = load_train_labels(tmp_path)
    for position, row in enumerate(rows):
        expected = math.radians(float(row["fov_deg"])) / 2.0
        assert float(labels["theta_ref"][position]) == pytest.approx(expected, rel=1e-6)


def test_search_rejects_a_degenerate_grid(labels) -> None:
    with pytest.raises(ValueError, match="n_grid"):
        search_best_constant(labels, n_grid=1)


def test_constant_predictor_emits_the_same_vector_for_every_row() -> None:
    predictor = ConstantPredictor([0.25, -0.5])
    out = predictor(torch.zeros(5, 3, 8, 8))
    assert out.shape == (5, 2)
    assert torch.allclose(out, torch.tensor([0.25, -0.5]).expand(5, 2))

    zeros = ConstantPredictor()(torch.zeros(3, 3, 8, 8))
    assert torch.equal(zeros, torch.zeros(3, 2))
