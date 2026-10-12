"""Tests for ``evaluate.py``.

The properties checked here are the ones that would silently invalidate the
final numbers rather than crash: test and stress must not be read without
``--final``; the per-sample errors must be the very numbers ``train.evaluate``
produces; the bootstrap must resample panoramas, not images; and the
stratification bins must put boundary values where the spec says.

Everything runs on CPU with tiny resnets trained for a few steps on a
synthetic data_root, so there are real checkpoints and real MLflow records to
check against.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import evaluate as ev
import train as train_module
from data.render_dataset import LABEL_FIELDS
from data.sampling import SamplingConfig, sample_camera
from datasets.fisheye_dataset import load_norm_stats

SIZE = 32
VIEWS = 4
SPLIT_ASSETS = {
    "train": ["tr_a", "tr_b", "tr_c", "tr_d"],
    "val": ["va_a", "va_b", "va_c"],
    "test": ["te_a", "te_b", "te_c"],
    "stress": ["st_a", "st_b"],
}
GROUPS = {"cnn": ["cnn_s0", "cnn_s1"], "other": ["other_s0"]}


# --- a synthetic data_root with all four splits -----------------------------


def _write_split(root: Path, split: str, assets: list[str], seed: int, stats: bool) -> None:
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
            sample = sample_camera(np.random.default_rng(900 + seed * 100 + index), cfg)
            rows.append({
                "index": index, "asset_id": asset_id, "view_idx": view,
                "f": sample.f, "xi": sample.xi, "fov_deg": sample.fov_deg,
                "circular": sample.circular, "r_ref": sample.r_ref,
                "yaw_deg": sample.yaw_deg, "pitch_deg": sample.pitch_deg,
                "roll_deg": sample.roll_deg,
                "category": "Interiors" if position % 2 else "Coast & Water",
                "texture": round(float(rng.uniform(1.0, 60.0)), 4),
            })
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
        meta["norm_stats"] = {"fields": ["log_f", "xi"],
                              "mean": stacked.mean(axis=0).tolist(),
                              "std": stacked.std(axis=0).tolist()}
    (out_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def _make_root(root: Path, with_held_out: bool = True) -> Path:
    for seed, (split, assets) in enumerate(SPLIT_ASSETS.items()):
        if split in ev.HELD_OUT_SPLITS and not with_held_out:
            continue
        _write_split(root, split, assets, seed=seed, stats=(split == "train"))
    (root / "splits.json").write_text(json.dumps({
        **SPLIT_ASSETS,
        "train_subsets": {"p100": SPLIT_ASSETS["train"]},
        "stress_sibling_split": {"st_a": "train", "st_b": "test"},
    }), encoding="utf-8")
    return root


def _train_tiny(root: Path, run_name: str, seed: int, ckpt: Path, uri: str) -> None:
    cfg = train_module.TrainConfig(
        run_name=run_name, model="resnet10t", pretrained=False,
        data=train_module.DataConfig(root=str(root), subset="p100"),
        optim=train_module.OptimConfig(lr=1e-3, warmup_steps=1),
        total_steps=3, batch_size=4, precision="fp32", eval_every=3, seed=seed,
        num_workers=0,
    ).validate()
    train_module.train(cfg, device=torch.device("cpu"), tracking_uri=uri,
                       out_dir=ckpt, log=lambda _m: None)


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    base = tmp_path_factory.mktemp("eval_world")
    root = _make_root(base / "data_root")
    ckpt = base / "checkpoints"
    uri = f"sqlite:///{(base / 'mlflow.db').as_posix()}"
    for seed, name in enumerate(["cnn_s0", "cnn_s1", "other_s0"]):
        _train_tiny(root, name, seed, ckpt, uri)
    return {"base": base, "root": root, "ckpt": ckpt, "uri": uri}


def eval_config(world: dict[str, Any], out: Path, **overrides: Any) -> ev.EvalConfig:
    cfg = ev.EvalConfig(
        data_root=str(world["root"]), checkpoints_dir=str(world["ckpt"]),
        groups={k: list(v) for k, v in GROUPS.items()}, comparison=("other", "cnn"),
        train_clean_n=10, train_clean_seed=0, bootstrap_n=200, bootstrap_seed=0,
        baseline_grid=12, baseline_xi_max=3.0, batch_size=4, num_workers=0,
        worst_k=3, perspective_fov_deg=90.0, out_dir=str(out), tracking_uri=world["uri"],
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _run(cfg: ev.EvalConfig, final: bool) -> dict[str, Any]:
    return ev.run(cfg, final=final, device=torch.device("cpu"), log=lambda _m: None)


# --- the split rule ---------------------------------------------------------


def test_default_splits_exclude_the_held_out_ones() -> None:
    assert ev.evaluated_splits(final=False) == ["val", "train_clean"]
    assert ev.evaluated_splits(final=True) == ["val", "train_clean", "test", "stress"]


@pytest.mark.parametrize("split", ["test", "stress"])
def test_open_split_refuses_held_out_without_final(world, tmp_path, split) -> None:
    cfg = eval_config(world, tmp_path / "out")
    with pytest.raises(PermissionError, match="--final"):
        ev.open_split(cfg, split, final=False)


def test_without_final_test_and_stress_are_never_opened(
    world, tmp_path, monkeypatch
) -> None:
    """They exist on disk; the run must still not touch them."""
    opened: list[str] = []
    real = ev.FisheyeDataset

    class Recording(real):  # type: ignore[misc, valid-type]
        def __init__(self, root, split, *args, **kwargs):
            opened.append(split)
            super().__init__(root, split, *args, **kwargs)

    monkeypatch.setattr(ev, "FisheyeDataset", Recording)
    summary = _run(eval_config(world, tmp_path / "out"), final=False)

    assert set(opened) <= {"train", "val"}, opened
    assert not set(opened) & set(ev.HELD_OUT_SPLITS)
    assert set(summary["splits"]) == {"val", "train_clean"}
    assert summary["stress"] is None


def test_without_final_runs_even_when_held_out_splits_are_absent(world, tmp_path) -> None:
    """Stronger than not opening them: they need not exist at all."""
    root = _make_root(tmp_path / "data_root", with_held_out=False)
    assert not (root / "rendered" / "test").exists()
    cfg = eval_config(world, tmp_path / "out", data_root=str(root))
    summary = _run(cfg, final=False)
    assert set(summary["splits"]) == {"val", "train_clean"}


def test_final_adds_test_and_stress(world, tmp_path) -> None:
    out = tmp_path / "out"
    summary = _run(eval_config(world, out), final=True)
    assert set(summary["splits"]) == {"val", "train_clean", "test", "stress"}
    assert summary["final"] is True
    assert summary["analysis_split"] == "test"
    assert summary["evaluated_at"]

    with (out / "per_sample" / "cnn_s0" / "stress.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert "stress_sibling_split" in rows[0]
    assert {r["stress_sibling_split"] for r in rows} == {"train", "test"}

    stress = summary["stress"]
    assert stress["n_panoramas_in_train"] == 1
    assert stress["n_panoramas_not_in_train"] == 1


# --- per-sample numbers are train.evaluate's numbers ------------------------


def test_per_sample_mean_equals_train_evaluate(world, tmp_path) -> None:
    """The per-sample errors must average to exactly what training reported."""
    cfg = eval_config(world, tmp_path / "out")
    device = torch.device("cpu")
    norm_stats = load_norm_stats(world["root"])
    model, train_cfg, _payload = ev.load_run("cnn_s0", cfg, norm_stats, device)

    split = ev.open_split(cfg, "val", final=False)
    per_sample = ev.predict(model, train_cfg, split, norm_stats, device, batch_size=4)

    # The checkpoint's own config points at this data_root, so train.py's
    # loader is built exactly as it was during training.
    assert train_cfg.data.root == str(world["root"])
    val_set = train_module.build_dataset(train_cfg, "val", train=False)
    loader, sampler = train_module.build_loader(train_cfg, val_set, train=False)
    reference = train_module.evaluate(model, loader, sampler, norm_stats, device, train_cfg)

    assert per_sample["reproj"].shape == (split.n,)
    assert per_sample["reproj"].mean() == pytest.approx(
        reference["val_reproj_mean_px"], abs=1e-5
    )


def test_per_sample_csv_has_the_required_columns(world, tmp_path) -> None:
    out = tmp_path / "out"
    _run(eval_config(world, out), final=False)
    with (out / "per_sample" / "cnn_s0" / "val.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    required = {"asset_id", "view_idx", "f", "xi", "fov_deg", "circular", "texture",
                "category", "pred_f", "pred_xi", "reproj_px", "invalid_frac"}
    assert required <= set(rows[0])
    assert len(rows) == len(SPLIT_ASSETS["val"]) * VIEWS

    # Baselines are written the same way.
    assert (out / "per_sample" / ev.BASELINE_BEST / "val.csv").is_file()


# --- sanity check and checkpoint integrity -----------------------------------


def test_sanity_check_agrees_with_mlflow(world, tmp_path) -> None:
    summary = _run(eval_config(world, tmp_path / "out"), final=False)
    assert set(summary["sanity"]) == {r for rs in GROUPS.values() for r in rs}
    for run_name, check in summary["sanity"].items():
        assert check["abs_diff_px"] < ev.SANITY_TOLERANCE_PX, (run_name, check)


def test_sanity_check_rejects_a_mismatch(world, tmp_path, monkeypatch) -> None:
    real = ev.mlflow_best
    monkeypatch.setattr(ev, "mlflow_best", lambda name, uri: real(name, uri) + 0.5)
    with pytest.raises(RuntimeError, match="sanity check failed"):
        _run(eval_config(world, tmp_path / "out"), final=False)


def test_stale_norm_stats_are_rejected(world, tmp_path) -> None:
    """A checkpoint trained before a re-render would de-standardise wrongly."""
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    payload = torch.load(world["ckpt"] / "cnn_s0_best.pt", weights_only=False)
    payload["norm_stats"]["mean"] = [m + 0.1 for m in payload["norm_stats"]["mean"]]
    torch.save(payload, ckpt / "cnn_s0_best.pt")

    cfg = eval_config(world, tmp_path / "out", checkpoints_dir=str(ckpt))
    with pytest.raises(ValueError, match="norm_stats"):
        ev.load_run("cnn_s0", cfg, load_norm_stats(world["root"]), torch.device("cpu"))


# --- bootstrap over panoramas -----------------------------------------------


def test_cluster_weights_resample_whole_panoramas() -> None:
    """Every view of a panorama carries the same weight in every replicate."""
    assets = ["a", "a", "a", "b", "c", "c", "d", "d", "d", "d"]
    weights = ev.cluster_weights(assets, n_boot=300, rng=np.random.default_rng(0))
    assert weights.shape == (300, len(assets))

    groups: dict[str, list[int]] = {}
    for i, asset in enumerate(assets):
        groups.setdefault(asset, []).append(i)

    for row in weights:
        for members in groups.values():
            assert len(set(row[members])) == 1, "a panorama was split up"
        # Each replicate draws as many panoramas as there are.
        assert sum(row[members[0]] for members in groups.values()) == len(groups)


def test_panorama_bootstrap_widens_the_interval_for_correlated_views() -> None:
    """The reason to resample panoramas: correlated views understate the error.

    Here the views of a panorama are near-identical, so the data hold ~20
    independent observations, not 200; a per-image bootstrap would pretend
    otherwise and return a much narrower interval.
    """
    rng = np.random.default_rng(1)
    n_panos, views = 20, 10
    pano_effect = rng.normal(0.0, 1.0, n_panos)
    values = np.repeat(pano_effect, views) + rng.normal(0.0, 0.05, n_panos * views)
    assets = [f"p{i}" for i in range(n_panos) for _ in range(views)]
    per_image_ids = [f"img{i}" for i in range(n_panos * views)]

    clustered = ev.panorama_bootstrap(values, assets, n_boot=1000, seed=0)
    naive = ev.panorama_bootstrap(values, per_image_ids, n_boot=1000, seed=0)

    clustered_width = clustered["ci_high"] - clustered["ci_low"]
    naive_width = naive["ci_high"] - naive["ci_low"]
    assert clustered_width > 2.0 * naive_width, (clustered_width, naive_width)
    assert clustered["n_panoramas"] == n_panos


def test_panorama_bootstrap_is_reproducible_and_brackets_the_mean() -> None:
    rng = np.random.default_rng(2)
    values = rng.normal(0.3, 1.0, 60)
    assets = [f"p{i // 3}" for i in range(60)]
    first = ev.panorama_bootstrap(values, assets, n_boot=500, seed=7)
    second = ev.panorama_bootstrap(values, assets, n_boot=500, seed=7)
    assert first == second
    assert first["ci_low"] <= first["mean"] <= first["ci_high"]
    assert ev.panorama_bootstrap(values, assets, n_boot=500, seed=8) != first


# --- stratification bins ----------------------------------------------------


def test_fov_bins_are_left_closed_and_end_at_220() -> None:
    fov = np.array([100.0, 129.999, 130.0, 159.9, 160.0, 189.99, 190.0, 219.9, 220.0])
    assert list(ev.fov_bin(fov)) == [
        "100-130", "100-130", "130-160", "130-160", "160-190",
        "160-190", "190-220", "190-220", "190-220",
    ]
    assert list(ev.fov_bin(np.array([99.9, 220.1]))) == ["out of range"] * 2


def test_texture_bins_split_off_low_texture_then_quintiles() -> None:
    low = np.array([0.5, 1.0, 4.999])
    rest = np.arange(5.0, 105.0)  # 100 values from exactly 5 upward
    labels, edges = ev.texture_bins(np.concatenate([low, rest]))

    assert list(labels[:3]) == ["< 5"] * 3
    # Exactly 5 is not low texture.
    assert labels[3].startswith("Q1")

    # Edges are computed from the >= 5 samples only, and the five bins are even.
    assert edges == pytest.approx(list(np.quantile(rest, [0.2, 0.4, 0.6, 0.8])))
    counts = [int(sum(1 for l in labels if str(l).startswith(f"Q{k}"))) for k in range(1, 6)]
    assert counts == [20, 20, 20, 20, 20], counts
    assert sum(counts) + 3 == len(labels)


def test_texture_bin_edges_assign_ties_to_the_lower_bin() -> None:
    texture = np.array([6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    labels, edges = ev.texture_bins(texture)
    first_edge = edges[0]
    on_edge = texture == first_edge
    if on_edge.any():
        assert all(str(l).startswith("Q1") for l in labels[on_edge])


# --- outputs ----------------------------------------------------------------


def test_summary_and_figures_are_written(world, tmp_path) -> None:
    out = tmp_path / "out"
    summary = _run(eval_config(world, out), final=False)

    assert (out / "summary.md").is_file()
    data = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert data["analysis_split"] == "val"
    for key in ("error_vs_texture", "error_vs_fov", "worst_cnn", "worst_other"):
        assert Path(summary["figures"][key]).is_file(), key

    text = (out / "summary.md").read_text(encoding="utf-8")
    assert "Not a final evaluation" in text
    assert "preview" in text
    # Multi-seed groups are reported as mean +/- std.
    assert "±" in text
    # The two constant baselines are in the main table.
    assert ev.BASELINE_MEAN in text and ev.BASELINE_BEST in text


def test_comparison_uses_per_image_seed_means(world, tmp_path) -> None:
    summary = _run(eval_config(world, tmp_path / "out"), final=False)
    comparison = summary["comparison"]
    assert comparison["first"] == "other" and comparison["second"] == "cnn"
    assert comparison["n_panoramas"] == len(SPLIT_ASSETS["val"])
    assert comparison["ci_low"] <= comparison["mean"] <= comparison["ci_high"]
