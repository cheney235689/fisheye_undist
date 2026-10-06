"""Acceptance criteria #9 and #11 (SPEC.md 5).

#9  Ten thousand draws all satisfy the four feasibility conditions of 4.6.
#11 Rendering the same small dataset with different ``--workers`` counts is
    byte-identical, because a sample's index and seed never depend on the
    schedule (SPEC.md 4.11).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from data.render_dataset import run as render_run
from data.sampling import (
    CameraSample,
    SamplingConfig,
    edge_stretch,
    is_feasible,
    sample_camera,
    sample_camera_with_attempts,
    sample_seed,
)
from geometry import ucm

CFG = SamplingConfig()
N_DRAWS = 10_000


# --- #9 feasibility ---------------------------------------------------------


@pytest.fixture(scope="module")
def draws() -> list[CameraSample]:
    rng = np.random.default_rng(0)
    return [sample_camera(rng, CFG) for _ in range(N_DRAWS)]


def test_every_draw_satisfies_the_first_three_conditions(draws: list[CameraSample]) -> None:
    """#9: SPEC.md 4.6 conditions, recomputed from the returned sample."""
    for sample in draws:
        half = np.radians(sample.fov_deg) / 2.0
        cos_half = float(np.cos(half))

        # 1. f is positive and not about to blow up.
        assert sample.xi + cos_half >= CFG.feasibility_margin

        # 2. inside the monotone range when xi > 1.
        if sample.xi > 1.0:
            assert cos_half >= -1.0 / sample.xi + CFG.feasibility_margin

        # 3. f is large enough to be a sensible image.
        assert sample.f >= CFG.f_min_px


def test_every_draw_respects_max_edge_stretch(draws: list[CameraSample]) -> None:
    """#9 condition 4: edge magnification M <= max_edge_stretch (SPEC.md 4.6)."""
    for sample in draws:
        half = np.radians(sample.fov_deg) / 2.0
        cos_half = float(np.cos(half))
        m = (1 + sample.xi) * (1 + sample.xi * cos_half) / (sample.xi + cos_half) ** 2
        assert m <= CFG.max_edge_stretch, (sample.fov_deg, sample.xi, m)


def test_edge_stretch_matches_the_reference_lenses() -> None:
    """The anchors quoted in SPEC.md 4.6.

    Note the angle convention differs between them: the rectilinear figure is
    a full angle, the stereographic one a half angle.
    """
    # Rectilinear (xi = 0) at a full 120 deg: M = 1/cos^2(60 deg) = 4.
    assert edge_stretch(120.0, 0.0) == pytest.approx(4.0, abs=1e-9)
    # Stereographic (xi = 1) at theta = 110 deg, i.e. a full 220 deg.
    assert edge_stretch(220.0, 1.0) == pytest.approx(3.04, abs=0.01)
    # The centre of the frame is never stretched.
    assert edge_stretch(1e-9, 0.0) == pytest.approx(1.0, abs=1e-6)
    assert edge_stretch(1e-9, 1.0) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize(
    "fov_deg,expected",
    [(119.0, True), (120.0, True), (121.0, False), (140.0, False)],
)
def test_edge_stretch_bounds_the_pinhole_fov(fov_deg: float, expected: bool) -> None:
    """At xi = 0, M = 1/cos^2(theta), so the cap of 4.0 is a 120 deg ceiling."""
    r_ref = CFG.r_ref(circular=False)
    assert is_feasible(fov_deg, 0.0, r_ref, CFG) is expected


def test_max_edge_stretch_is_the_only_reason_121_is_rejected() -> None:
    """Raising the cap admits exactly the sample the default rejects."""
    r_ref = CFG.r_ref(circular=False)
    relaxed = SamplingConfig(max_edge_stretch=5.0)
    assert not is_feasible(121.0, 0.0, r_ref, CFG)
    assert is_feasible(121.0, 0.0, r_ref, relaxed)


def test_draws_stay_inside_the_configured_ranges(draws: list[CameraSample]) -> None:
    fov = np.array([s.fov_deg for s in draws])
    xi = np.array([s.xi for s in draws])
    assert CFG.fov_deg[0] <= fov.min() and fov.max() <= CFG.fov_deg[1]
    assert CFG.xi[0] <= xi.min() and xi.max() <= CFG.xi[1]

    for name, bounds in (
        ("yaw_deg", CFG.yaw_deg), ("pitch_deg", CFG.pitch_deg), ("roll_deg", CFG.roll_deg)
    ):
        values = np.array([getattr(s, name) for s in draws])
        assert bounds[0] <= values.min() and values.max() <= bounds[1], name


def test_f_and_r_ref_agree_with_the_geometry_module(draws: list[CameraSample]) -> None:
    """f must be exactly what geometry/ucm.py derives; no second formula."""
    for sample in draws[:500]:
        expected_r_ref = 224 / 2.0 if sample.circular else 224 / np.sqrt(2.0)
        assert sample.r_ref == pytest.approx(expected_r_ref)
        assert sample.f == pytest.approx(
            ucm.f_from_fov(sample.fov_deg, sample.xi, sample.r_ref), rel=1e-12
        )


def test_both_image_shapes_are_drawn(draws: list[CameraSample]) -> None:
    circular = np.array([s.circular for s in draws])
    assert 0.4 < circular.mean() < 0.6, circular.mean()


def test_is_feasible_rejects_outside_the_region() -> None:
    r_ref = CFG.r_ref(circular=False)
    # xi = 0 with FOV 220 needs cos(110 deg) = -0.34, so xi + cos < margin.
    assert not is_feasible(220.0, 0.0, r_ref, CFG)
    assert is_feasible(220.0, 1.0, r_ref, CFG)
    # A tiny f is rejected by condition 3.
    assert not is_feasible(100.0, 0.0, 1.0, CFG)
    # A wide, low-xi view is rejected by condition 4.
    assert not is_feasible(160.0, 0.2, r_ref, CFG)


def test_rejection_rate_is_reported_and_plausible() -> None:
    rng = np.random.default_rng(1)
    attempts = accepted = 0
    for _ in range(2000):
        _, used = sample_camera_with_attempts(rng, CFG)
        attempts += used
        accepted += 1
    rate = (attempts - accepted) / attempts
    assert 0.0 < rate < 0.5, rate


def test_sampling_is_reproducible_from_the_generator() -> None:
    a = [sample_camera(np.random.default_rng(7), CFG) for _ in range(5)]
    b = [sample_camera(np.random.default_rng(7), CFG) for _ in range(5)]
    assert a == b
    assert sample_camera(np.random.default_rng(8), CFG) != a[0]


# --- sample_seed ------------------------------------------------------------


def test_sample_seed_is_stable_across_processes() -> None:
    """SPEC.md 6 forbids Python's salted hash(); the seed must survive a restart."""
    args = (0, "train", "aarfontein_dawn_2", 7)
    here = sample_seed(*args)

    code = (
        "import sys; sys.path.insert(0, r'%s');"
        "from data.sampling import sample_seed;"
        "print(sample_seed(0, 'train', 'aarfontein_dawn_2', 7))"
        % str(Path(__file__).resolve().parents[1])
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert int(out.stdout.strip()) == here


def test_sample_seed_separates_its_inputs() -> None:
    base = sample_seed(0, "train", "asset", 0)
    assert sample_seed(1, "train", "asset", 0) != base
    assert sample_seed(0, "val", "asset", 0) != base
    assert sample_seed(0, "train", "other", 0) != base
    assert sample_seed(0, "train", "asset", 1) != base


def test_sample_seed_fits_in_int64() -> None:
    for view in range(200):
        seed = sample_seed(0, "train", f"asset_{view}", view)
        assert 0 <= seed < 2**63
        np.random.default_rng(seed)  # must be acceptable to numpy


def test_sample_seed_is_well_spread() -> None:
    seeds = {sample_seed(0, "train", f"asset_{i}", j)
             for i in range(60) for j in range(30)}
    assert len(seeds) == 60 * 30


# --- #11 reproducibility across worker counts -------------------------------


def _make_tiny_dataset(tmp_path: Path, n_panos: int = 3) -> dict:
    """A self-contained data_root with synthetic panoramas."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    assets = []
    for index in range(n_panos):
        asset_id = f"synthetic_{index:02d}"
        rng = np.random.default_rng(100 + index)
        pano = (rng.random((64, 128, 3)) * 255).astype(np.uint8)
        pano = cv2.GaussianBlur(pano, (0, 0), sigmaX=1.5, sigmaY=1.5)
        assert cv2.imwrite(str(raw_dir / f"{asset_id}.jpg"), pano)
        assets.append(
            {
                "asset_id": asset_id,
                "file": f"{asset_id}.jpg",
                "categories": ["outdoor"],
                "primary_category": "Fields & Countryside",
            }
        )

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"assets": assets}), encoding="utf-8")
    splits = tmp_path / "splits.json"
    splits.write_text(
        json.dumps({"train": [a["asset_id"] for a in assets], "val": [], "test": [],
                    "stress": []}),
        encoding="utf-8",
    )

    return {
        "data_root": str(tmp_path),
        "manifest": str(manifest),
        "splits": str(splits),
        "raw_dir": str(raw_dir),
        "image_size": 48,
        "supersample": 2,
        "pano_resize": [128, 64],
        "circular_prob": 0.5,
        "fov_deg": [100, 220],
        "xi": [0.0, 1.5],
        "feasibility_margin": 0.05,
        "f_min_px": 5,
        "yaw_deg": [-180, 180],
        "pitch_deg": [-30, 30],
        "roll_deg": [-15, 15],
        "views_per_pano": {"train": 4},
        "seed": 0,
    }


@pytest.mark.parametrize("workers", [2, 3])
def test_render_is_identical_regardless_of_worker_count(
    tmp_path: Path, workers: int
) -> None:
    """#11: byte-identical images and labels for any --workers (SPEC.md 4.11)."""
    serial_root = tmp_path / "serial"
    parallel_root = tmp_path / "parallel"

    serial_cfg = _make_tiny_dataset(serial_root)
    parallel_cfg = _make_tiny_dataset(parallel_root)

    render_run(serial_cfg, "train", workers=1, log=lambda _m: None)
    render_run(parallel_cfg, "train", workers=workers, log=lambda _m: None)

    a = np.load(serial_root / "rendered" / "train" / "images.npy")
    b = np.load(parallel_root / "rendered" / "train" / "images.npy")
    assert a.shape == b.shape
    assert np.array_equal(a, b), "images differ between worker counts"

    labels_a = (serial_root / "rendered" / "train" / "labels.csv").read_bytes()
    labels_b = (parallel_root / "rendered" / "train" / "labels.csv").read_bytes()
    assert labels_a == labels_b


def test_render_writes_the_expected_artifacts(tmp_path: Path) -> None:
    cfg = _make_tiny_dataset(tmp_path, n_panos=2)
    meta = render_run(cfg, "train", workers=1, log=lambda _m: None)

    out_dir = tmp_path / "rendered" / "train"
    images = np.load(out_dir / "images.npy")
    assert images.shape == (2 * 4, 48, 48, 3)
    assert images.dtype == np.uint8

    assert meta["n_images"] == 8
    assert 0.0 <= meta["rejection_rate"] < 1.0
    # SPEC.md 4.7: the train split carries the standardisation statistics.
    assert meta["norm_stats"]["fields"] == ["log_f", "xi"]
    assert len(meta["norm_stats"]["mean"]) == 2
    assert all(s > 0 for s in meta["norm_stats"]["std"])

    import csv

    with (out_dir / "labels.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8
    assert [int(r["index"]) for r in rows] == list(range(8))
    for field in ("asset_id", "view_idx", "f", "xi", "fov_deg", "circular",
                  "r_ref", "yaw_deg", "pitch_deg", "roll_deg", "category",
                  "texture"):
        assert field in rows[0], field

    # texture is a plain non-negative score on the uint8 gradient scale.
    textures = [float(r["texture"]) for r in rows]
    assert all(value >= 0.0 for value in textures)
    assert any(value > 0.0 for value in textures)


def test_render_rejects_an_unknown_split(tmp_path: Path) -> None:
    cfg = _make_tiny_dataset(tmp_path, n_panos=1)
    with pytest.raises(ValueError, match="split must be one of"):
        render_run(cfg, "nonsense", workers=1, log=lambda _m: None)


# --- texture score ----------------------------------------------------------


def test_texture_score_is_zero_on_a_flat_image() -> None:
    from data.render_dataset import texture_score

    flat = np.full((32, 32, 3), 128, dtype=np.uint8)
    mask = np.ones((32, 32), dtype=bool)
    assert texture_score(flat, mask) == pytest.approx(0.0, abs=1e-6)


def test_texture_score_grows_with_detail() -> None:
    from data.render_dataset import texture_score

    mask = np.ones((64, 64), dtype=bool)
    flat = np.full((64, 64, 3), 128, dtype=np.uint8)

    gradient = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))
    ramp = np.repeat(gradient[:, :, None], 3, axis=2)

    checker = (((np.arange(64)[:, None] // 4) + (np.arange(64)[None, :] // 4)) % 2)
    noisy = np.repeat((checker * 255).astype(np.uint8)[:, :, None], 3, axis=2)

    assert texture_score(flat, mask) < texture_score(ramp, mask)
    assert texture_score(ramp, mask) < texture_score(noisy, mask)


def test_texture_score_only_counts_the_valid_mask() -> None:
    from data.render_dataset import texture_score

    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, 32:] = 255  # all the detail lives on the right half

    # Bands wide enough to survive the two 3x3 erosions texture_score applies.
    left = np.zeros((64, 64), dtype=bool)
    left[:, :26] = True
    right = np.zeros((64, 64), dtype=bool)
    right[:, 26:40] = True

    assert texture_score(image, left) == pytest.approx(0.0, abs=1e-6)
    assert texture_score(image, right) > 100.0


def test_texture_score_ignores_the_aperture_rim() -> None:
    """The erosion must remove the fill-induced gradient at the mask edge."""
    from data.render_dataset import texture_score

    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[16:48, 16:48] = 128  # a grey square on a black fill
    mask = np.zeros((64, 64), dtype=bool)
    mask[16:48, 16:48] = True

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    magnitude = cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    un_eroded = float(magnitude[mask].mean())

    assert un_eroded > 10.0, "the rim should look textured before erosion"
    assert texture_score(image, mask) == pytest.approx(0.0, abs=1e-6)


def test_texture_score_returns_zero_when_erosion_empties_the_mask() -> None:
    from data.render_dataset import texture_score

    image = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))
    image = np.repeat(image[:, :, None], 3, axis=2)
    sliver = np.zeros((64, 64), dtype=bool)
    sliver[30:33, 30:33] = True  # 3x3 cannot survive two erosions

    assert texture_score(image, sliver) == 0.0


def test_texture_score_handles_an_empty_mask() -> None:
    from data.render_dataset import texture_score

    image = np.full((16, 16, 3), 200, dtype=np.uint8)
    assert texture_score(image, np.zeros((16, 16), dtype=bool)) == 0.0


@pytest.mark.parametrize("circular", [False, True])
@pytest.mark.parametrize("fov_deg", [110.0, 160.0, 200.0])
def test_uniform_panorama_scores_no_texture(circular: bool, fov_deg: float) -> None:
    """A featureless scene must score ~0 for both image shapes.

    Without the mask erosion the circular case scores about 9.5: every pixel on
    the aperture rim has the black fill in its Sobel window.
    """
    from data.render_dataset import texture_score
    from geometry import ucm
    from geometry.render import FisheyeCamera, render_fisheye
    from geometry.rotation import rotation_from_ypr

    pano = np.full((512, 1024, 3), 128, dtype=np.uint8)
    xi = 0.5
    probe = FisheyeCamera(f=1.0, xi=xi, size=224, circular=circular)
    camera = FisheyeCamera(
        f=ucm.f_from_fov(fov_deg, xi, probe.r_ref), xi=xi, size=224, circular=circular
    )
    image, mask = render_fisheye(
        pano, camera, rotation_from_ypr(0.0, 0.0, 0.0), supersample=4
    )

    assert mask.any()
    score = texture_score(image, mask)
    assert score < 0.5, f"circular={circular} fov={fov_deg}: texture {score:.4f}"
