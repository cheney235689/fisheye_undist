"""Tests for ``train.py`` (SPEC.md 3.13).

Everything runs on CPU against a miniature rendered data_root and a tiny timm
model with ``pretrained=False``, so no weights are downloaded and no GPU is
needed.  MLflow is pointed at a temporary SQLite file per test, never the
project's ``mlflow.db``.
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

import train as train_module
from data.render_dataset import LABEL_FIELDS
from data.sampling import SamplingConfig, sample_camera
from train import (
    ALLOWED_SPLITS,
    DataConfig,
    OptimConfig,
    TrainConfig,
    benchmark,
    build_dataset,
    build_loader,
    build_optimizer,
    evaluate,
    load_config,
    lr_scale,
    meta_hash,
    no_decay_names,
    overfit_batch,
    train,
    uses_channels_last,
)

SIZE = 32
VIEWS = 6
TRAIN_ASSETS = ["alpha", "beta", "gamma", "delta"]
VAL_ASSETS = ["epsilon", "zeta"]
TINY_MODEL = "resnet10t"  # smallest resnet timm ships; exercises channels_last


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
            sample = sample_camera(np.random.default_rng(500 + index), cfg)
            rows.append(
                {
                    "index": index, "asset_id": asset_id, "view_idx": view,
                    "f": sample.f, "xi": sample.xi, "fov_deg": sample.fov_deg,
                    "circular": sample.circular, "r_ref": sample.r_ref,
                    "yaw_deg": sample.yaw_deg, "pitch_deg": sample.pitch_deg,
                    "roll_deg": sample.roll_deg, "category": "Interiors",
                    "texture": round(float(rng.uniform(1.0, 60.0)), 4),
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
    root = tmp_path_factory.mktemp("train_root")
    _write_split(root, "train", TRAIN_ASSETS, seed=0, stats=True)
    _write_split(root, "val", VAL_ASSETS, seed=1, stats=False)
    # test and stress exist on disk precisely so a test can prove they are not read.
    _write_split(root, "test", ["eta"], seed=2, stats=False)
    _write_split(root, "stress", ["theta"], seed=3, stats=False)
    (root / "splits.json").write_text(
        json.dumps(
            {
                "train": TRAIN_ASSETS, "val": VAL_ASSETS,
                "test": ["eta"], "stress": ["theta"],
                "train_subsets": {
                    "p10": TRAIN_ASSETS[:1],
                    "p30": TRAIN_ASSETS[:2],
                    "p100": TRAIN_ASSETS,
                },
            }
        ),
        encoding="utf-8",
    )
    return root


def tiny_config(data_root: Path, **overrides: Any) -> TrainConfig:
    cfg = TrainConfig(
        run_name="tiny",
        model=TINY_MODEL,
        pretrained=False,
        data=DataConfig(root=str(data_root), subset="p100", mode="offline"),
        loss="geometric",
        optim=OptimConfig(lr=1e-3, weight_decay=1e-4, warmup_steps=2),
        total_steps=5,
        batch_size=4,
        precision="fp32",
        eval_every=5,
        seed=0,
        grad_clip=1.0,
        num_workers=0,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg.validate()


# --- config -----------------------------------------------------------------


def test_defaults_follow_spec_313() -> None:
    cfg = load_config(None)
    assert cfg.model == "resnet34"
    assert cfg.pretrained is True
    assert cfg.loss == "geometric"
    assert cfg.schedule == "cosine"
    assert cfg.total_steps == 6000
    assert cfg.batch_size == 128
    assert cfg.precision == "bf16"
    assert cfg.eval_every == 500
    assert cfg.seed == 0
    assert cfg.optim.lr == pytest.approx(3.0e-4)
    assert cfg.optim.weight_decay == pytest.approx(1.0e-4)
    assert cfg.optim.warmup_steps == 500
    assert cfg.data.root == "data_root"
    assert cfg.data.subset == "p100"
    assert cfg.data.mode == "offline"
    # The two additions this task asked for.
    assert cfg.grad_clip == pytest.approx(1.0)
    assert cfg.num_workers == 6


@pytest.mark.parametrize(
    "name",
    ["resnet34_pre_p100_geo_s0.yaml", "vit_small_pre_p100_geo_s0.yaml"],
)
def test_shipped_configs_load(name: str) -> None:
    cfg = load_config(Path("configs/train") / name)
    assert cfg.run_name.endswith("_s0")
    assert cfg.loss == "geometric"
    assert cfg.data.mode == "offline"
    assert cfg.validate() is cfg


def test_config_round_trips_through_yaml(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "run_name": "x", "model": "resnet18", "pretrained": False,
                "data": {"root": "r", "subset": "p30", "mode": "offline"},
                "loss": "param_l1", "optim": {"lr": 1e-3, "warmup_steps": 7},
                "total_steps": 11, "batch_size": 3, "eval_every": 4,
                "grad_clip": 0.5, "num_workers": 2,
            }
        ),
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.model == "resnet18" and cfg.loss == "param_l1"
    assert cfg.data.subset == "p30"
    assert cfg.optim.lr == pytest.approx(1e-3) and cfg.optim.warmup_steps == 7
    # Unset nested keys keep their defaults.
    assert cfg.optim.weight_decay == pytest.approx(1.0e-4)
    assert cfg.grad_clip == pytest.approx(0.5) and cfg.num_workers == 2


def test_unknown_config_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"lr": 1e-3}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown config keys"):
        load_config(path)


@pytest.mark.parametrize(
    "field,value",
    [("loss", "mse"), ("schedule", "step"), ("precision", "fp16"),
     ("total_steps", 0), ("batch_size", 0), ("num_workers", -1)],
)
def test_invalid_config_is_rejected(field: str, value: Any) -> None:
    cfg = TrainConfig()
    setattr(cfg, field, value)
    with pytest.raises(ValueError):
        cfg.validate()


def test_flat_config_is_mlflow_ready() -> None:
    flat = TrainConfig().flat()
    assert flat["optim.lr"] == pytest.approx(3.0e-4)
    assert flat["data.root"] == "data_root"
    assert flat["grad_clip"] == pytest.approx(1.0)
    assert all(not isinstance(v, (dict, list)) for v in flat.values())


# --- CLI carries modes only -------------------------------------------------


def test_cli_has_no_hyperparameter_overrides() -> None:
    """Every hyperparameter must come from the yaml, so a run is reproducible."""
    options = {
        action.dest for action in train_module.build_parser()._actions
    } - {"help"}
    assert options == {"config", "overfit_batch", "benchmark"}


def test_cli_requires_a_config() -> None:
    with pytest.raises(SystemExit):
        train_module.build_parser().parse_args([])


def test_overfit_and_benchmark_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        train_module.build_parser().parse_args(
            ["--config", "c.yaml", "--overfit-batch", "10", "--benchmark", "10"]
        )


# --- SPEC.md 6: test and stress are off limits ------------------------------


@pytest.mark.parametrize("split", ["test", "stress"])
def test_train_refuses_the_held_out_splits(data_root: Path, split: str) -> None:
    """#6: train.py must not be able to read test or stress, even by mistake."""
    cfg = tiny_config(data_root)
    with pytest.raises(ValueError, match="may only read"):
        build_dataset(cfg, split, train=False)


def test_allowed_splits_is_exactly_train_and_val() -> None:
    assert set(ALLOWED_SPLITS) == {"train", "val"}


def test_train_source_never_names_the_held_out_splits() -> None:
    """A belt-and-braces grep: no stray literal that could open them."""
    source = Path("train.py").read_text(encoding="utf-8")
    for forbidden in ('"test"', "'test'", '"stress"', "'stress'"):
        assert forbidden not in source, forbidden


# --- schedule ---------------------------------------------------------------


def test_warmup_then_cosine() -> None:
    total, warmup = 100, 10
    assert lr_scale(0, total, warmup) == pytest.approx(0.1)
    assert lr_scale(warmup - 1, total, warmup) == pytest.approx(1.0)
    assert lr_scale(warmup, total, warmup) == pytest.approx(1.0)
    assert lr_scale(total, total, warmup) == pytest.approx(0.0, abs=1e-12)
    mid = lr_scale((total + warmup) // 2, total, warmup)
    assert 0.3 < mid < 0.7

    values = [lr_scale(s, total, warmup) for s in range(warmup, total + 1)]
    assert values == sorted(values, reverse=True)


def test_schedule_without_warmup() -> None:
    assert lr_scale(0, 10, 0) == pytest.approx(1.0)
    assert lr_scale(10, 10, 0) == pytest.approx(0.0, abs=1e-12)


def test_channels_last_only_for_resnet() -> None:
    assert uses_channels_last("resnet34")
    assert uses_channels_last("resnet10t")
    assert not uses_channels_last("vit_small_patch16_224")


# --- data plumbing ----------------------------------------------------------


def test_train_split_is_augmented_and_val_is_not(data_root: Path) -> None:
    """SPEC.md 4.9: augmentation on train only."""
    cfg = tiny_config(data_root)
    assert build_dataset(cfg, "train", train=True).transform is not None
    assert build_dataset(cfg, "val", train=False).transform is None


def test_subset_applies_to_train_only(data_root: Path) -> None:
    cfg = tiny_config(data_root, data=DataConfig(root=str(data_root), subset="p10"))
    assert len(build_dataset(cfg, "train", train=True)) == VIEWS
    assert len(build_dataset(cfg, "val", train=False)) == len(VAL_ASSETS) * VIEWS


def test_loader_shuffles_train_and_not_val(data_root: Path) -> None:
    cfg = tiny_config(data_root)
    _loader, train_sampler = build_loader(cfg, build_dataset(cfg, "train", True), True)
    _loader, val_sampler = build_loader(cfg, build_dataset(cfg, "val", False), False)
    assert train_sampler.shuffle is True
    assert val_sampler.shuffle is False
    assert [i for _e, i in val_sampler] == list(range(len(val_sampler)))


def test_online_mode_explains_itself(data_root: Path) -> None:
    cfg = tiny_config(data_root, data=DataConfig(root=str(data_root), mode="online"))
    with pytest.raises(NotImplementedError, match="panorama cache"):
        build_dataset(cfg, "train", train=True)


# --- evaluation -------------------------------------------------------------


def test_evaluate_reports_every_required_metric(data_root: Path) -> None:
    cfg = tiny_config(data_root)
    device = torch.device("cpu")
    model = train_module.build_model(cfg, device)
    dataset = build_dataset(cfg, "val", train=False)
    loader, sampler = build_loader(cfg, dataset, train=False)
    from datasets.fisheye_dataset import load_norm_stats

    metrics = evaluate(model, loader, sampler, load_norm_stats(data_root), device, cfg)

    for key in (
        "val_reproj_mean_px", "val_reproj_median_px", "val_invalid_frac",
        "val_f_mae_px", "val_xi_mae", "val_loss_geometric", "val_loss_param_l1",
    ):
        assert key in metrics, key
        assert math.isfinite(metrics[key]), key

    assert metrics["val_reproj_mean_px"] >= 0.0
    assert 0.0 <= metrics["val_invalid_frac"] <= 1.0
    assert metrics["val_f_mae_px"] >= 0.0


def test_evaluate_logs_both_losses_whichever_is_trained(data_root: Path) -> None:
    """Both objectives are recorded so the two run families stay comparable."""
    device = torch.device("cpu")
    from datasets.fisheye_dataset import load_norm_stats

    stats = load_norm_stats(data_root)
    results = {}
    for loss in ("geometric", "param_l1"):
        cfg = tiny_config(data_root, loss=loss)
        model = train_module.build_model(cfg, device)
        torch.manual_seed(0)
        loader, sampler = build_loader(cfg, build_dataset(cfg, "val", False), False)
        results[loss] = evaluate(model, loader, sampler, stats, device, cfg)
    for loss in ("geometric", "param_l1"):
        assert results[loss]["val_loss_geometric"] >= 0.0
        assert results[loss]["val_loss_param_l1"] >= 0.0


def test_evaluate_leaves_the_model_in_train_mode(data_root: Path) -> None:
    cfg = tiny_config(data_root)
    device = torch.device("cpu")
    model = train_module.build_model(cfg, device)
    model.train()
    loader, sampler = build_loader(cfg, build_dataset(cfg, "val", False), False)
    from datasets.fisheye_dataset import load_norm_stats

    evaluate(model, loader, sampler, load_norm_stats(data_root), device, cfg)
    assert model.training, "evaluate must restore train mode"


# --- the loop end to end ----------------------------------------------------


@pytest.fixture
def tracking_uri(tmp_path: Path) -> str:
    return f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"


def test_five_steps_run_and_checkpoint(
    data_root: Path, tmp_path: Path, tracking_uri: str
) -> None:
    """#9: the loop, the checkpoint and the MLflow record all work."""
    cfg = tiny_config(data_root)
    out_dir = tmp_path / "ckpt"
    result = train(
        cfg,
        device=torch.device("cpu"),
        tracking_uri=tracking_uri,
        out_dir=out_dir,
        log=lambda _m: None,
    )

    assert math.isfinite(result["best_val_reproj_mean_px"])
    checkpoint = Path(result["checkpoint"])
    assert checkpoint.is_file()

    payload = torch.load(checkpoint, weights_only=False)
    assert payload["config"]["model"] == TINY_MODEL
    assert payload["step"] == cfg.total_steps - 1
    assert "norm_stats" in payload and payload["norm_stats"]["fields"] == ["log_f", "xi"]
    assert payload["metrics"]["val_reproj_mean_px"] == pytest.approx(
        result["best_val_reproj_mean_px"]
    )
    # The weights really are the trained ones.
    model = train_module.build_model(cfg, torch.device("cpu"))
    model.load_state_dict(payload["model"])


def test_mlflow_records_config_and_provenance(
    data_root: Path, tmp_path: Path, tracking_uri: str
) -> None:
    import mlflow

    cfg = tiny_config(data_root, run_name="recorded")
    train(cfg, device=torch.device("cpu"), tracking_uri=tracking_uri,
          out_dir=tmp_path / "ckpt", log=lambda _m: None)

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("fisheye-calib")
    runs = client.search_runs([experiment.experiment_id])
    assert runs, "no run was recorded"
    run = next(r for r in runs if r.data.tags.get("mlflow.runName") == "recorded")

    # Full resolved config, not just what the yaml happened to contain.
    for key in ("model", "loss", "total_steps", "batch_size", "seed",
                "grad_clip", "num_workers", "optim.lr", "data.root"):
        assert key in run.data.params, key
    assert run.data.params["model"] == TINY_MODEL

    assert "git_commit" in run.data.tags
    assert run.data.tags["train_meta_sha256"] == meta_hash(data_root)
    assert len(run.data.tags["train_meta_sha256"]) == 64

    for metric in ("train_loss", "val_reproj_mean_px", "val_f_mae_px",
                   "val_xi_mae", "val_invalid_frac", "val_loss_geometric",
                   "val_loss_param_l1", "best_val_reproj_mean_px"):
        assert metric in run.data.metrics, metric


def test_meta_hash_tracks_the_dataset(data_root: Path, tmp_path: Path) -> None:
    first = meta_hash(data_root)
    assert len(first) == 64
    assert meta_hash(data_root) == first
    assert meta_hash(tmp_path) == "missing"


@pytest.mark.parametrize("loss", ["geometric", "param_l1"])
def test_both_losses_train(
    data_root: Path, tmp_path: Path, tracking_uri: str, loss: str
) -> None:
    cfg = tiny_config(data_root, loss=loss, run_name=f"tiny_{loss}")
    result = train(cfg, device=torch.device("cpu"), tracking_uri=tracking_uri,
                   out_dir=tmp_path / loss, log=lambda _m: None)
    assert math.isfinite(result["best_val_reproj_mean_px"])


def test_training_actually_updates_the_weights(
    data_root: Path, tmp_path: Path, tracking_uri: str
) -> None:
    cfg = tiny_config(data_root)
    device = torch.device("cpu")
    torch.manual_seed(cfg.seed)
    before = {
        k: v.clone() for k, v in train_module.build_model(cfg, device).state_dict().items()
    }
    result = train(cfg, device=device, tracking_uri=tracking_uri,
                   out_dir=tmp_path / "ckpt", log=lambda _m: None)
    after = torch.load(result["checkpoint"], weights_only=False)["model"]
    assert any(
        not torch.equal(before[k], after[k])
        for k in before
        if before[k].dtype.is_floating_point
    )


# --- diagnostic modes -------------------------------------------------------


def test_overfit_batch_drives_the_loss_down(data_root: Path) -> None:
    cfg = tiny_config(data_root, total_steps=40)
    result = overfit_batch(cfg, steps=40, device=torch.device("cpu"),
                           log=lambda _m: None)
    assert len(result["history"]) == 40
    assert all(math.isfinite(v) for v in result["history"])
    # A single batch with a free-form model must get easier to fit.
    assert result["history"][-1] < result["history"][0]


def test_overfit_batch_uses_no_augmentation(data_root: Path) -> None:
    """The batch must be fixed, or the loss curve means nothing."""
    cfg = tiny_config(data_root)
    first = overfit_batch(cfg, steps=3, device=torch.device("cpu"), log=lambda _m: None)
    second = overfit_batch(cfg, steps=3, device=torch.device("cpu"), log=lambda _m: None)
    assert first["history"] == pytest.approx(second["history"])


def test_overfit_batch_prints_every_ten_steps(data_root: Path) -> None:
    lines: list[str] = []
    cfg = tiny_config(data_root)
    overfit_batch(cfg, steps=25, device=torch.device("cpu"), log=lines.append)
    step_lines = [line for line in lines if line.strip().startswith("step ")]
    # steps 0, 10, 20 and the final step 24.
    assert len(step_lines) == 4, lines


def test_benchmark_reports_throughput(data_root: Path) -> None:
    cfg = tiny_config(data_root)
    result = benchmark(cfg, steps=5, device=torch.device("cpu"), log=lambda _m: None)
    assert result["img_per_s"] > 0
    assert result["ms_per_step"] > 0
    assert result["peak_gib"] == 0.0  # CPU run allocates no GPU memory


def test_benchmark_output_mentions_memory_and_throughput(data_root: Path) -> None:
    lines: list[str] = []
    benchmark(tiny_config(data_root), steps=3, device=torch.device("cpu"),
              log=lines.append)
    joined = "\n".join(lines)
    assert "img/s" in joined
    assert "peak GPU memory" in joined


# --- weight decay param groups (SPEC.md 3.13.2) -----------------------------


def _groups(model_name: str, weight_decay: float = 0.05):
    """Build the optimiser for a real timm model and return its two groups."""
    import timm

    model = timm.create_model(model_name, pretrained=False, num_classes=2)
    cfg = TrainConfig(
        model=model_name, optim=OptimConfig(lr=1e-4, weight_decay=weight_decay)
    )
    optimiser = build_optimizer(model, cfg)
    assert len(optimiser.param_groups) == 2

    decayed, exempt = optimiser.param_groups
    assert decayed["weight_decay"] == pytest.approx(weight_decay)
    assert exempt["weight_decay"] == 0.0
    return model, decayed["params"], exempt["params"]


@pytest.mark.parametrize("model_name", ["resnet18", "vit_small_patch16_224"])
def test_param_groups_cover_every_parameter_exactly_once(model_name: str) -> None:
    """The two groups must partition the parameters: no gaps, no duplicates."""
    model, decayed, exempt = _groups(model_name)

    expected = [p for p in model.parameters() if p.requires_grad]
    combined = decayed + exempt
    assert len(combined) == len(expected), (len(combined), len(expected))

    ids = [id(p) for p in combined]
    assert len(set(ids)) == len(ids), "a parameter landed in both groups"
    assert set(ids) == {id(p) for p in expected}

    # Both groups are actually used.
    assert decayed and exempt


@pytest.mark.parametrize("model_name", ["resnet18", "vit_small_patch16_224"])
def test_bias_and_norm_parameters_skip_weight_decay(model_name: str) -> None:
    """SPEC.md 3.13.2: nothing with ndim <= 1 may be decayed."""
    model, decayed, exempt = _groups(model_name)
    exempt_ids = {id(p) for p in exempt}
    decayed_ids = {id(p) for p in decayed}

    saw_bias = saw_norm = False
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1:
            assert id(param) in exempt_ids, f"{name} (ndim={param.ndim}) is decayed"
            saw_bias |= name.endswith(".bias")
            saw_norm |= "norm" in name.lower() or "bn" in name.lower()
        else:
            # Every multi-dim parameter is decayed unless the model exempted it.
            if id(param) not in decayed_ids:
                assert id(param) in exempt_ids
    assert saw_bias, "the fixture should contain biases"
    assert saw_norm, "the fixture should contain normalisation parameters"


@pytest.mark.parametrize("model_name", ["resnet18", "vit_small_patch16_224"])
def test_conv_and_linear_weights_are_decayed(model_name: str) -> None:
    model, decayed, _exempt = _groups(model_name)
    decayed_ids = {id(p) for p in decayed}
    skip = no_decay_names(model)

    checked = 0
    for module_name, module in model.named_modules():
        if not isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            continue
        weight = getattr(module, "weight", None)
        if weight is None or not weight.requires_grad:
            continue
        full = f"{module_name}.weight" if module_name else "weight"
        if full in skip:
            continue
        assert id(weight) in decayed_ids, f"{full} is not decayed"
        checked += 1
    assert checked > 0


def test_vit_positional_parameters_are_exempt() -> None:
    """pos_embed and cls_token are 3-D, so only no_weight_decay() can save them."""
    model, decayed, exempt = _groups("vit_small_patch16_224")
    skip = no_decay_names(model)
    assert "pos_embed" in skip and "cls_token" in skip

    exempt_ids = {id(p) for p in exempt}
    decayed_ids = {id(p) for p in decayed}
    for name, param in model.named_parameters():
        if name in skip:
            assert param.ndim > 1, f"{name} would have been exempt on ndim alone"
            assert id(param) in exempt_ids, f"{name} is decayed despite no_weight_decay"
            assert id(param) not in decayed_ids


def test_resnet_has_no_model_level_exemptions() -> None:
    """A plain CNN exempts nothing by name; the ndim rule carries it alone."""
    import timm

    model = timm.create_model("resnet18", pretrained=False, num_classes=2)
    assert no_decay_names(model) == set()


def test_no_decay_names_tolerates_a_model_without_the_hook() -> None:
    assert no_decay_names(torch.nn.Linear(4, 2)) == set()


def test_optimizer_uses_the_config_learning_rate() -> None:
    _model, _decayed, _exempt = _groups("resnet18", weight_decay=0.05)
    import timm

    model = timm.create_model("resnet18", pretrained=False, num_classes=2)
    cfg = TrainConfig(model="resnet18", optim=OptimConfig(lr=7e-5, weight_decay=0.01))
    optimiser = build_optimizer(model, cfg)
    assert all(group["lr"] == pytest.approx(7e-5) for group in optimiser.param_groups)


def test_shipped_configs_tune_each_architecture() -> None:
    """SPEC.md 3.13.1: the optimiser differs between the two runs, on purpose."""
    resnet = load_config("configs/train/resnet34_pre_p100_geo_s0.yaml")
    vit = load_config("configs/train/vit_small_pre_p100_geo_s0.yaml")

    assert resnet.optim.lr == pytest.approx(3.0e-4)
    assert resnet.optim.weight_decay == pytest.approx(1.0e-4)
    assert vit.optim.lr == pytest.approx(1.0e-4)
    assert vit.optim.weight_decay == pytest.approx(0.05)

    # Everything that is meant to be held fixed stays fixed.
    assert resnet.loss == vit.loss
    assert resnet.schedule == vit.schedule
    assert resnet.total_steps == vit.total_steps
    assert resnet.batch_size == vit.batch_size
    assert resnet.optim.warmup_steps == vit.optim.warmup_steps
    assert resnet.data == vit.data
    assert resnet.seed == vit.seed


def test_training_runs_with_the_split_param_groups(
    data_root: Path, tmp_path: Path, tracking_uri: str
) -> None:
    """The real loop, with a decay large enough that a wrong group would show."""
    cfg = tiny_config(data_root, run_name="grouped")
    cfg.optim.weight_decay = 0.05
    result = train(cfg, device=torch.device("cpu"), tracking_uri=tracking_uri,
                   out_dir=tmp_path / "ckpt", log=lambda _m: None)
    assert math.isfinite(result["best_val_reproj_mean_px"])


# --- DataLoader worker cleanup ----------------------------------------------


def _live_children() -> list:
    import multiprocessing

    # active_children() also reaps finished ones, so a non-empty result means
    # processes that are genuinely still running.
    return multiprocessing.active_children()


def test_consecutive_runs_leave_no_worker_processes(
    data_root: Path, tmp_path: Path, tracking_uri: str
) -> None:
    """Two back-to-back runs, as in a sweep, must not accumulate workers.

    Note that on the normal return path CPython's refcounting usually reaps the
    workers on its own, so this test guards the behaviour rather than proving
    the fix; the exception-path test below is the one that fails without it.
    """
    assert _live_children() == []
    for attempt in (1, 2):
        cfg = tiny_config(data_root, run_name=f"workers_{attempt}", num_workers=2)
        train(cfg, device=torch.device("cpu"), tracking_uri=tracking_uri,
              out_dir=tmp_path / "ckpt", log=lambda _m: None)
    assert _live_children() == [], "persistent workers outlived train()"


def test_a_failing_run_does_not_pin_its_workers(
    data_root: Path, tmp_path: Path, tracking_uri: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The case the try/finally exists for.

    A raised exception's traceback keeps train()'s frame alive, and before the
    fix that frame kept the loaders -- and their worker processes -- alive for
    as long as anyone held the exception. A sweep logging the error, or this
    very test holding it via pytest.raises, is exactly such a holder. Without
    the fix this test sees two live workers.
    """
    calls = {"n": 0}
    real = train_module.compute_loss

    def fail_on_third_step(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("simulated failure at step 3")
        return real(*args, **kwargs)

    monkeypatch.setattr(train_module, "compute_loss", fail_on_third_step)
    cfg = tiny_config(data_root, run_name="workers_fail", num_workers=2)

    with pytest.raises(RuntimeError, match="simulated failure") as excinfo:
        train(cfg, device=torch.device("cpu"), tracking_uri=tracking_uri,
              out_dir=tmp_path / "ckpt", log=lambda _m: None)

    assert excinfo.value is not None  # the exception (and its traceback) is held
    assert _live_children() == [], "a failed run left its workers running"


def test_shutdown_loader_is_safe_on_any_loader(data_root: Path) -> None:
    """None, a worker-less loader, and a second call must all be no-ops."""
    train_module.shutdown_loader(None)

    cfg = tiny_config(data_root, num_workers=0)
    loader, _sampler = build_loader(cfg, build_dataset(cfg, "val", False), False)
    next(iter(loader))
    train_module.shutdown_loader(loader)
    train_module.shutdown_loader(loader)


def test_shutdown_loader_stops_persistent_workers(data_root: Path) -> None:
    cfg = tiny_config(data_root, num_workers=2)
    loader, sampler = build_loader(cfg, build_dataset(cfg, "val", False), False)
    sampler.set_epoch(0)
    for _batch in loader:
        pass
    # persistent_workers keeps them alive after the epoch ends ...
    assert len(_live_children()) == 2
    # ... until they are shut down explicitly.
    train_module.shutdown_loader(loader)
    assert _live_children() == []
    assert loader._iterator is None


def test_sweep_collects_garbage_after_every_arm(
    data_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sweep as sweep_module

    listing = tmp_path / "arms.txt"
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"not_a_key": 1}), encoding="utf-8")
    listing.write_text(f"{bad}\n{bad}\n{bad}\n", encoding="utf-8")

    collected = {"n": 0}
    real_collect = sweep_module.gc.collect

    def counting_collect(*args, **kwargs):
        collected["n"] += 1
        return real_collect(*args, **kwargs)

    monkeypatch.setattr(sweep_module.gc, "collect", counting_collect)
    summary = sweep_module.run_sweep(
        listing, tracking_uri=f"sqlite:///{(tmp_path / 'm.db').as_posix()}",
        log=lambda _m: None,
    )
    assert summary.failed == 3
    assert collected["n"] >= 3, "gc.collect() must run once per arm"
