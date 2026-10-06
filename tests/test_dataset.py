"""Acceptance criterion #12 (SPEC.md 5): the datasets and the augmentations.

Shapes and dtypes, val borrowing the train statistics, and no forbidden
augmentation anywhere in the pipeline.  Two further properties get their own
tests because they are the ones that break silently: the memmap must not be
opened until a worker touches it (SPEC.md 4.12), and a transform must leave
the label alone while staying reproducible for a given seed.
"""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import torch

from data.render_dataset import LABEL_FIELDS
from data.sampling import SamplingConfig, sample_camera
from datasets import transforms as T
from datasets.fisheye_dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    FisheyeDataset,
    load_norm_stats,
)
from datasets.online_dataset import (
    INDEX_FILE,
    PANOS_FILE,
    OnlineFisheyeDataset,
    build_pano_cache,
    worker_init_fn,
)
from datasets.sampler import EpochSampler, split_key

SIZE = 32
VIEWS = 4
ASSETS = ["alpha_scene", "beta_scene", "gamma_scene", "delta_scene"]
TRAIN_ASSETS = ASSETS[:3]
VAL_ASSETS = ASSETS[3:]


# --- a small, self-contained data_root --------------------------------------


def _write_split(
    root: Path, split: str, asset_ids: list[str], seed: int, with_stats: bool
) -> None:
    out_dir = root / "rendered" / split
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    cfg = SamplingConfig(image_size=SIZE, f_min_px=5.0)
    total = len(asset_ids) * VIEWS

    images = np.lib.format.open_memmap(
        out_dir / "images.npy", mode="w+", dtype=np.uint8, shape=(total, SIZE, SIZE, 3)
    )
    rows: list[dict[str, Any]] = []
    for position, asset_id in enumerate(asset_ids):
        for view_idx in range(VIEWS):
            index = position * VIEWS + view_idx
            images[index] = rng.integers(0, 256, (SIZE, SIZE, 3), dtype=np.uint8)
            sample = sample_camera(np.random.default_rng(1000 + index), cfg)
            rows.append(
                {
                    "index": index,
                    "asset_id": asset_id,
                    "view_idx": view_idx,
                    "f": sample.f,
                    "xi": sample.xi,
                    "fov_deg": sample.fov_deg,
                    "circular": sample.circular,
                    "r_ref": sample.r_ref,
                    "yaw_deg": sample.yaw_deg,
                    "pitch_deg": sample.pitch_deg,
                    "roll_deg": sample.roll_deg,
                    "category": "Interiors",
                    "texture": round(float(rng.uniform(1.0, 80.0)), 4),
                }
            )
    images.flush()
    del images

    with (out_dir / "labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    meta: dict[str, Any] = {"split": split, "image_size": SIZE, "n_images": total}
    if with_stats:
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
    """A miniature data_root with a train split, a val split and panoramas."""
    root = tmp_path_factory.mktemp("data_root")
    _write_split(root, "train", TRAIN_ASSETS, seed=0, with_stats=True)
    _write_split(root, "val", VAL_ASSETS, seed=1, with_stats=False)

    (root / "splits.json").write_text(
        json.dumps(
            {
                "train": TRAIN_ASSETS,
                "val": VAL_ASSETS,
                "test": [],
                "stress": [],
                "train_subsets": {
                    "p10": TRAIN_ASSETS[:1],
                    "p30": TRAIN_ASSETS[:2],
                    "p100": TRAIN_ASSETS,
                },
            }
        ),
        encoding="utf-8",
    )

    raw = root / "raw" / "polyhaven"
    raw.mkdir(parents=True, exist_ok=True)
    assets = []
    for index, asset_id in enumerate(ASSETS):
        rng = np.random.default_rng(50 + index)
        pano = (rng.random((64, 128, 3)) * 255).astype(np.uint8)
        pano = cv2.GaussianBlur(pano, (0, 0), sigmaX=1.5, sigmaY=1.5)
        assert cv2.imwrite(str(raw / f"{asset_id}.jpg"), pano)
        assets.append(
            {
                "asset_id": asset_id,
                "file": f"{asset_id}.jpg",
                "primary_category": "Interiors",
                "categories": ["indoor"],
            }
        )
    (root / "manifest.json").write_text(json.dumps({"assets": assets}), encoding="utf-8")
    return root


# --- #12 shapes, dtypes and fields ------------------------------------------


def test_item_has_the_declared_shape_and_dtype(data_root: Path) -> None:
    """#12: image (3,S,S) float, target (2,) float, plus the raw parameters."""
    dataset = FisheyeDataset(data_root, "train")
    assert len(dataset) == len(TRAIN_ASSETS) * VIEWS

    item = dataset[0]
    assert item["image"].shape == (3, SIZE, SIZE)
    assert item["image"].dtype == torch.float32
    assert item["target"].shape == (2,)
    assert item["target"].dtype == torch.float32

    assert isinstance(item["f"], float) and item["f"] > 0
    assert isinstance(item["xi"], float)
    assert isinstance(item["r_ref"], float)
    assert isinstance(item["circular"], bool)
    assert isinstance(item["theta_ref"], float)
    assert isinstance(item["asset_id"], str)
    assert isinstance(item["category"], str)


def test_item_carries_texture_for_error_analysis(data_root: Path) -> None:
    dataset = FisheyeDataset(data_root, "train")
    values = [dataset[i]["texture"] for i in range(len(dataset))]
    assert all(isinstance(v, float) and v >= 0.0 for v in values)
    assert len(set(values)) > 1


def test_theta_ref_is_half_the_fov_in_radians(data_root: Path) -> None:
    dataset = FisheyeDataset(data_root, "train")
    with (data_root / "rendered" / "train" / "labels.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for position in range(len(dataset)):
        expected = np.radians(float(rows[position]["fov_deg"])) / 2.0
        assert dataset[position]["theta_ref"] == pytest.approx(expected)


def test_image_is_imagenet_normalised(data_root: Path) -> None:
    dataset = FisheyeDataset(data_root, "train")
    item = dataset[0]
    raw = np.asarray(dataset.images[0], dtype=np.float32) / 255.0
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
    expected = (raw.transpose(2, 0, 1) - mean) / std
    assert np.allclose(item["image"].numpy(), expected, atol=1e-5)


def test_target_is_the_standardised_log_f_and_xi(data_root: Path) -> None:
    stats = load_norm_stats(data_root)
    dataset = FisheyeDataset(data_root, "train")
    for position in (0, 3, 7):
        item = dataset[position]
        raw = np.array([np.log(item["f"]), item["xi"]])
        expected = (raw - np.asarray(stats["mean"])) / np.asarray(stats["std"])
        assert np.allclose(item["target"].numpy(), expected, atol=1e-5)


# --- #12 val must borrow the train statistics -------------------------------


def test_val_uses_the_train_norm_stats(data_root: Path) -> None:
    """#12 / SPEC.md 4.7: val has no statistics of its own and must not invent any."""
    assert not (data_root / "rendered" / "val" / "meta.json").read_text().count(
        "norm_stats"
    )
    train = FisheyeDataset(data_root, "train")
    val = FisheyeDataset(data_root, "val")
    assert val.norm_stats == train.norm_stats == load_norm_stats(data_root)


def test_val_targets_use_the_train_statistics(data_root: Path) -> None:
    stats = load_norm_stats(data_root)
    val = FisheyeDataset(data_root, "val")
    item = val[0]
    raw = np.array([np.log(item["f"]), item["xi"]])
    expected = (raw - np.asarray(stats["mean"])) / np.asarray(stats["std"])
    assert np.allclose(item["target"].numpy(), expected, atol=1e-5)


def test_explicit_norm_stats_win(data_root: Path) -> None:
    custom = {"fields": ["log_f", "xi"], "mean": [0.0, 0.0], "std": [1.0, 1.0]}
    dataset = FisheyeDataset(data_root, "val", norm_stats=custom)
    item = dataset[0]
    assert item["target"][0].item() == pytest.approx(np.log(item["f"]), abs=1e-5)
    assert item["target"][1].item() == pytest.approx(item["xi"], abs=1e-5)


def test_missing_train_stats_is_an_explicit_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="render the train split"):
        load_norm_stats(tmp_path)


# --- subsets ----------------------------------------------------------------


def test_subset_filters_by_asset_id(data_root: Path) -> None:
    sizes = {}
    for name in ("p10", "p30", "p100"):
        dataset = FisheyeDataset(data_root, "train", subset=name)
        assets = {dataset[i]["asset_id"] for i in range(len(dataset))}
        sizes[name] = len(dataset)
        assert len(dataset) == len(assets) * VIEWS
    assert sizes["p10"] < sizes["p30"] < sizes["p100"]
    assert sizes["p100"] == len(TRAIN_ASSETS) * VIEWS


def test_subsets_are_nested(data_root: Path) -> None:
    def assets_of(name: str) -> set[str]:
        dataset = FisheyeDataset(data_root, "train", subset=name)
        return {dataset[i]["asset_id"] for i in range(len(dataset))}

    assert assets_of("p10") <= assets_of("p30") <= assets_of("p100")


def test_unknown_subset_is_rejected(data_root: Path) -> None:
    with pytest.raises(KeyError, match="unknown subset"):
        FisheyeDataset(data_root, "train", subset="p99")


# --- SPEC.md 4.12: the memmap must be opened in the worker -------------------


def test_memmap_is_not_opened_during_construction(data_root: Path) -> None:
    """#12 / SPEC.md 4.12: spawn pickles the dataset, so nothing may be open."""
    dataset = FisheyeDataset(data_root, "train")
    assert dataset._images is None, "__init__ opened the memmap"
    dataset[0]
    assert dataset._images is not None, "first __getitem__ should open it"


def test_pickling_never_carries_an_open_memmap(data_root: Path) -> None:
    dataset = FisheyeDataset(data_root, "train")
    dataset[0]  # open it here
    assert dataset._images is not None

    revived = pickle.loads(pickle.dumps(dataset))
    assert revived._images is None, "a worker would inherit the parent's handle"
    assert len(revived) == len(dataset)
    assert np.allclose(revived[0]["image"].numpy(), dataset[0]["image"].numpy())


def test_dataloader_with_workers_returns_every_sample(data_root: Path) -> None:
    """The spawn path end to end, which is what SPEC.md 4.12 is about."""
    from torch.utils.data import DataLoader

    dataset = FisheyeDataset(data_root, "train")
    loader = DataLoader(dataset, batch_size=3, num_workers=2, shuffle=False)
    seen = [asset for batch in loader for asset in batch["asset_id"]]
    assert len(seen) == len(dataset)
    assert seen == [dataset[i]["asset_id"] for i in range(len(dataset))]


# --- #12 forbidden augmentations --------------------------------------------


def test_train_transform_contains_no_forbidden_op() -> None:
    """#12: nothing that changes the intrinsics may be in the pipeline."""
    transform = T.build_train_transform()
    names = set(transform.op_names())
    assert names <= T.ALLOWED_OPS, sorted(names - T.ALLOWED_OPS)
    assert not names & T.FORBIDDEN_OPS
    assert T.check_no_forbidden_ops(transform) == []


def test_allowed_and_forbidden_sets_are_disjoint() -> None:
    assert not T.ALLOWED_OPS & T.FORBIDDEN_OPS
    for name in ("crop", "resize", "scale", "rotate", "translate", "aspect_ratio"):
        assert name in T.FORBIDDEN_OPS, name


def test_check_no_forbidden_ops_detects_a_bad_pipeline() -> None:
    class Bad(T.Op):
        name = "resize"

        def __call__(self, image, rng):
            return image

    assert T.check_no_forbidden_ops(T.Compose((Bad(),))) == ["resize"]


def test_eval_transform_does_nothing() -> None:
    """SPEC.md 4.9: val/test get no augmentation at all."""
    transform = T.build_eval_transform()
    assert transform.op_names() == []
    image = np.random.default_rng(0).integers(0, 256, (SIZE, SIZE, 3), dtype=np.uint8)
    assert np.array_equal(transform(image, np.random.default_rng(0)), image)


# --- transforms: label-preserving and reproducible ---------------------------


@pytest.fixture
def image() -> np.ndarray:
    rng = np.random.default_rng(7)
    base = (rng.random((SIZE, SIZE, 3)) * 255).astype(np.uint8)
    return cv2.GaussianBlur(base, (0, 0), sigmaX=1.0, sigmaY=1.0)


def test_transform_is_reproducible_for_a_seed(image: np.ndarray) -> None:
    transform = T.build_train_transform()
    first = transform(image, np.random.default_rng(1234))
    second = transform(image, np.random.default_rng(1234))
    assert np.array_equal(first, second)


def test_transform_varies_with_the_seed(image: np.ndarray) -> None:
    transform = T.build_train_transform()
    outputs = {
        transform(image, np.random.default_rng(seed)).tobytes() for seed in range(6)
    }
    assert len(outputs) > 1


def test_transform_preserves_shape_and_dtype(image: np.ndarray) -> None:
    transform = T.build_train_transform()
    for seed in range(12):
        out = transform(image, np.random.default_rng(seed))
        assert out.shape == image.shape
        assert out.dtype == np.uint8


def test_transform_never_touches_the_label(data_root: Path, image: np.ndarray) -> None:
    """The point of SPEC.md 4.9: augmentation must not invalidate (f, xi)."""
    plain = FisheyeDataset(data_root, "train")
    augmented = FisheyeDataset(data_root, "train", transform=T.build_train_transform())

    for position in range(len(plain)):
        a, b = plain[position], augmented[position]
        assert a["f"] == b["f"]
        assert a["xi"] == b["xi"]
        assert a["r_ref"] == b["r_ref"]
        assert a["circular"] == b["circular"]
        assert a["theta_ref"] == b["theta_ref"]
        assert torch.equal(a["target"], b["target"])


def test_dataset_augmentation_is_reproducible(data_root: Path) -> None:
    first = FisheyeDataset(data_root, "train", transform=T.build_train_transform())
    second = FisheyeDataset(data_root, "train", transform=T.build_train_transform())
    assert torch.equal(first[2]["image"], second[2]["image"])


def test_epoch_in_the_key_changes_the_augmentation_not_the_label(
    data_root: Path,
) -> None:
    """SPEC.md 4.12: the epoch rides along with the index, not on the dataset."""
    dataset = FisheyeDataset(data_root, "train", transform=T.build_train_transform())
    first = dataset[(0, 1)]
    second = dataset[(1, 1)]

    assert not torch.equal(first["image"], second["image"])
    assert torch.equal(first["target"], second["target"])
    assert first["f"] == second["f"] and first["xi"] == second["xi"]

    # A bare index is epoch 0, so the dataset still works without a sampler.
    assert torch.equal(dataset[1]["image"], first["image"])


def test_dataset_no_longer_carries_epoch_state(data_root: Path) -> None:
    """The state that persistent workers could never have seen is gone."""
    dataset = FisheyeDataset(data_root, "train")
    assert not hasattr(dataset, "set_epoch")
    assert not hasattr(dataset, "_epoch")


@pytest.mark.parametrize(
    "op",
    [
        T.RandomBrightnessContrast(p=1.0),
        T.RandomGamma(p=1.0),
        T.RandomGaussianBlur(p=1.0),
        T.RandomGaussianNoise(p=1.0),
        T.RandomJPEG(p=1.0),
        T.RandomVignette(p=1.0),
        T.RandomDihedral(p=1.0),
    ],
)
def test_each_op_is_reproducible_and_well_formed(op: T.Op, image: np.ndarray) -> None:
    a = op(image, np.random.default_rng(3))
    b = op(image, np.random.default_rng(3))
    assert np.array_equal(a, b)
    assert a.shape == image.shape and a.dtype == np.uint8
    assert op.name in T.ALLOWED_OPS


def _channel_detail(channel: int) -> np.ndarray:
    """Flat grey with a one-pixel checkerboard injected into one channel."""
    image = np.full((64, 64, 3), 128, dtype=np.int16)
    checker = ((np.arange(64)[:, None] + np.arange(64)[None, :]) % 2) * 90
    image[:, :, channel] = np.clip(image[:, :, channel] + checker - 45, 0, 255)
    return image.astype(np.uint8)


def test_jpeg_keeps_the_channel_order() -> None:
    """RGB must reach the codec as BGR, or luma is computed on the wrong channels.

    JPEG puts most of its bit budget in luma, and red weighs 0.299 there
    against blue's 0.114 -- so red detail must survive better than blue.
    Handing the codec RGB while it assumes BGR swaps those two weights and
    inverts the relationship.  A flat colour patch cannot detect this: encoding
    and decoding under the same wrong assumption cancels out, so the test needs
    chroma detail.
    """
    op = T.RandomJPEG(p=1.0, quality=(30, 30))

    errors = {}
    for channel, label in ((0, "red"), (2, "blue")):
        image = _channel_detail(channel)
        out = op(image, np.random.default_rng(0))
        errors[label] = float(
            np.abs(out.astype(int) - image.astype(int))[:, :, channel].mean()
        )

    assert errors["red"] < errors["blue"], errors


def test_jpeg_does_not_tint_a_flat_colour() -> None:
    op = T.RandomJPEG(p=1.0, quality=(95, 95))
    for colour in ((220, 30, 30), (30, 220, 30), (30, 30, 220)):
        image = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        image[:, :] = colour
        out = op(image, np.random.default_rng(0))
        assert np.abs(out.astype(int) - np.array(colour)).max() <= 6, (
            colour,
            out[0, 0].tolist(),
        )


def test_jpeg_actually_degrades_the_image(image: np.ndarray) -> None:
    op = T.RandomJPEG(p=1.0, quality=(20, 20))
    out = op(image, np.random.default_rng(0))
    assert not np.array_equal(out, image)
    assert np.abs(out.astype(int) - image.astype(int)).mean() < 40


def test_dihedral_only_permutes_pixels(image: np.ndarray) -> None:
    """A square symmetry rearranges pixels; it cannot invent or lose any."""
    op = T.RandomDihedral(p=1.0)
    for seed in range(8):
        out = op(image, np.random.default_rng(seed))
        assert np.array_equal(np.sort(out.ravel()), np.sort(image.ravel()))


def test_dihedral_keeps_the_radial_profile(image: np.ndarray) -> None:
    """Why it is label-safe: every pixel keeps its distance from the centre."""
    op = T.RandomDihedral(p=1.0)
    centre = (SIZE - 1) / 2.0
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    radius = np.round(np.hypot(xx - centre, yy - centre), 6)
    gray = image[:, :, 0].astype(np.int64)

    # numpy 2 returns an inverse shaped like the input, so flatten it.
    bins = np.unique(radius, return_inverse=True)[1].ravel()

    reference = np.bincount(bins, weights=gray.ravel())
    for seed in range(8):
        out = op(image, np.random.default_rng(seed))[:, :, 0].astype(np.int64)
        got = np.bincount(bins, weights=out.ravel())
        assert np.allclose(got, reference)


def test_dihedral_rejects_a_non_square_image() -> None:
    op = T.RandomDihedral(p=1.0)
    with pytest.raises(ValueError, match="square"):
        op(np.zeros((8, 16, 3), dtype=np.uint8), np.random.default_rng(0))


def test_build_train_transform_overrides() -> None:
    assert "jpeg" not in T.build_train_transform(jpeg=None).op_names()
    with pytest.raises(KeyError, match="unknown op"):
        T.build_train_transform(nonsense=1.0)


# --- online dataset ---------------------------------------------------------


@pytest.fixture(scope="module")
def pano_cache(data_root: Path) -> Path:
    return build_pano_cache(data_root, ASSETS, log=lambda _m: None)


def test_pano_cache_is_one_file_plus_an_index(data_root: Path, pano_cache: Path) -> None:
    panos = np.load(pano_cache / PANOS_FILE, mmap_mode="r")
    assert panos.shape == (len(ASSETS), 512, 1024, 3)
    assert panos.dtype == np.uint8

    index = json.loads((pano_cache / INDEX_FILE).read_text(encoding="utf-8"))
    assert index["asset_ids"] == ASSETS
    assert index["pano_size"] == [1024, 512]


def _online(data_root: Path, **kwargs: Any) -> OnlineFisheyeDataset:
    defaults: dict[str, Any] = {
        "root": data_root,
        "asset_ids": TRAIN_ASSETS,
        "views_per_pano": VIEWS,
        "sampling_cfg": SamplingConfig(image_size=SIZE, f_min_px=5.0),
        "base_seed": 0,
        "supersample": 2,
    }
    defaults.update(kwargs)
    return OnlineFisheyeDataset(**defaults)


def test_online_length_is_assets_times_views(data_root: Path, pano_cache: Path) -> None:
    dataset = _online(data_root)
    assert len(dataset) == len(TRAIN_ASSETS) * VIEWS


def test_online_item_matches_the_offline_contract(
    data_root: Path, pano_cache: Path
) -> None:
    item = _online(data_root)[0]
    assert item["image"].shape == (3, SIZE, SIZE)
    assert item["image"].dtype == torch.float32
    assert item["target"].shape == (2,)
    assert item["f"] > 0
    assert isinstance(item["circular"], bool)
    assert item["theta_ref"] > 0


def test_online_memmap_is_opened_lazily(data_root: Path, pano_cache: Path) -> None:
    """SPEC.md 4.12 again: 1.2 GB of panoramas must not be pickled per worker."""
    dataset = _online(data_root)
    assert dataset._panos is None
    dataset[0]
    assert dataset._panos is not None
    assert pickle.loads(pickle.dumps(dataset))._panos is None


def test_online_is_reproducible_within_an_epoch(
    data_root: Path, pano_cache: Path
) -> None:
    a, b = _online(data_root), _online(data_root)
    for position in (0, 5, 11):
        assert torch.equal(a[position]["image"], b[position]["image"])
        assert a[position]["f"] == b[position]["f"]


def test_online_epoch_in_the_key_redraws_the_cameras(
    data_root: Path, pano_cache: Path
) -> None:
    dataset = _online(data_root)
    first = [dataset[(0, i)]["f"] for i in range(len(dataset))]
    second = [dataset[(1, i)]["f"] for i in range(len(dataset))]
    assert first != second
    # Going back to epoch 0 reproduces it exactly.
    assert [dataset[(0, i)]["f"] for i in range(len(dataset))] == first
    assert [dataset[i]["f"] for i in range(len(dataset))] == first

    assert not hasattr(dataset, "set_epoch")


def test_online_renders_what_it_claims(data_root: Path, pano_cache: Path) -> None:
    """The returned image must be the render of the reported parameters."""
    from geometry.render import FisheyeCamera, render_fisheye
    from geometry.rotation import rotation_from_ypr

    dataset = _online(data_root)
    position = 6
    item = dataset[(2, position)]
    sample = dataset.sample_for(position, epoch=2)
    assert sample.f == pytest.approx(item["f"])

    asset_id, _view = dataset.locate(position)
    row = json.loads((pano_cache / INDEX_FILE).read_text("utf-8"))["asset_ids"].index(
        asset_id
    )
    pano = np.ascontiguousarray(np.load(pano_cache / PANOS_FILE, mmap_mode="r")[row])
    camera = FisheyeCamera(
        f=sample.f, xi=sample.xi, size=SIZE, circular=sample.circular
    )
    expected, _mask = render_fisheye(
        pano,
        camera,
        rotation_from_ypr(sample.yaw_deg, sample.pitch_deg, sample.roll_deg),
        supersample=2,
    )
    expected = cv2.cvtColor(expected, cv2.COLOR_BGR2RGB)

    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
    want = (expected.astype(np.float32).transpose(2, 0, 1) / 255.0 - mean) / std
    assert np.allclose(item["image"].numpy(), want, atol=1e-5)


def test_online_target_uses_the_train_statistics(
    data_root: Path, pano_cache: Path
) -> None:
    stats = load_norm_stats(data_root)
    item = _online(data_root)[3]
    raw = np.array([np.log(item["f"]), item["xi"]])
    expected = (raw - np.asarray(stats["mean"])) / np.asarray(stats["std"])
    assert np.allclose(item["target"].numpy(), expected, atol=1e-5)


def test_online_transform_keeps_the_label(data_root: Path, pano_cache: Path) -> None:
    plain = _online(data_root)
    augmented = _online(data_root, transform=T.build_train_transform())
    for position in range(len(plain)):
        assert plain[position]["f"] == augmented[position]["f"]
        assert plain[position]["xi"] == augmented[position]["xi"]
        assert torch.equal(plain[position]["target"], augmented[position]["target"])


def test_online_rejects_an_asset_outside_the_cache(
    data_root: Path, pano_cache: Path
) -> None:
    with pytest.raises(KeyError, match="pano cache"):
        _online(data_root, asset_ids=["not_cached"])


def test_online_requires_the_cache_to_exist(data_root: Path, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="build it with"):
        _online(data_root, cache_dir=tmp_path / "absent")


def test_worker_init_fn_disables_opencv_threads() -> None:
    """SPEC.md 4.11/4.12: one rendering thread per worker, not one per core."""
    previous = cv2.getNumThreads()
    try:
        cv2.setNumThreads(4)
        assert cv2.getNumThreads() > 1
        worker_init_fn(0)
        # OpenCV reports 1 once its thread pool is switched off.
        assert cv2.getNumThreads() <= 1
    finally:
        cv2.setNumThreads(previous)


# --- EpochSampler (SPEC.md 4.12) --------------------------------------------


def test_split_key_accepts_both_forms() -> None:
    assert split_key(5) == (0, 5)
    assert split_key((3, 7)) == (3, 7)
    assert split_key([3, 7]) == (3, 7)
    assert split_key(torch.tensor(4)) == (0, 4)
    with pytest.raises(ValueError, match="epoch, index"):
        split_key((1, 2, 3))


def test_sampler_yields_epoch_index_pairs() -> None:
    sampler = EpochSampler(6, shuffle=True, seed=0)
    first = list(sampler)
    assert len(first) == len(sampler) == 6
    assert sorted(index for _epoch, index in first) == list(range(6))
    assert {epoch for epoch, _index in first} == {0}

    sampler.set_epoch(2)
    second = list(sampler)
    assert {epoch for epoch, _index in second} == {2}
    assert [i for _e, i in second] != [i for _e, i in first]


def test_sampler_order_is_reproducible_and_seed_dependent() -> None:
    def order(seed: int, epoch: int) -> list[int]:
        sampler = EpochSampler(32, shuffle=True, seed=seed)
        sampler.set_epoch(epoch)
        return [index for _epoch, index in sampler]

    assert order(0, 0) == order(0, 0)
    assert order(0, 0) != order(0, 1)
    assert order(0, 1) != order(1, 0), "seed and epoch must not collide"


def test_sampler_without_shuffle_is_sequential() -> None:
    sampler = EpochSampler(4, shuffle=False, seed=0)
    sampler.set_epoch(3)
    assert list(sampler) == [(3, 0), (3, 1), (3, 2), (3, 3)]


def _epoch_images(data_root: Path, epochs: int, seed: int) -> dict[int, torch.Tensor]:
    """Run a persistent-worker DataLoader for `epochs` epochs, keyed by index."""
    from torch.utils.data import DataLoader

    dataset = FisheyeDataset(
        data_root, "train", transform=T.build_train_transform()
    )
    sampler = EpochSampler(len(dataset), shuffle=False, seed=seed)
    loader = DataLoader(
        dataset,
        batch_size=2,
        sampler=sampler,
        num_workers=2,
        persistent_workers=True,
        pin_memory=False,
    )
    captured: dict[int, torch.Tensor] = {}
    try:
        for epoch in range(epochs):
            sampler.set_epoch(epoch)
            for batch in loader:
                captured.setdefault(epoch, batch["image"][0].clone())
    finally:
        del loader
    return captured


def test_persistent_workers_see_the_new_epoch(data_root: Path) -> None:
    """The bug this design exists to prevent.

    With the epoch held on the dataset, a persistent worker keeps its epoch-0
    copy forever and every epoch augments identically.  Carrying the epoch in
    the sampler's output makes the change actually reach the worker.
    """
    captured = _epoch_images(data_root, epochs=2, seed=0)
    assert set(captured) == {0, 1}
    assert not torch.equal(captured[0], captured[1]), (
        "epoch 1 reused epoch 0's augmentation: the worker never saw set_epoch"
    )


def test_persistent_worker_run_is_bit_identical_on_a_rerun(data_root: Path) -> None:
    first = _epoch_images(data_root, epochs=2, seed=0)
    second = _epoch_images(data_root, epochs=2, seed=0)
    assert set(first) == set(second)
    for epoch in first:
        assert torch.equal(first[epoch], second[epoch]), epoch
