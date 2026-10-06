"""Acceptance criterion #6 (SPEC.md 5) plus the equirectangular mapping itself."""

import math

import numpy as np
import pytest

from geometry import equirect, rotation

W, H = 1024, 512


def _angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle between unit vectors, conditioned for near-parallel inputs.

    ``arccos(dot)`` cannot resolve below ~1.5e-8 rad in float64 because its
    slope is infinite at 1; the cross-product form stays accurate there.
    """
    cross = np.linalg.norm(np.cross(a, b), axis=-1)
    dot = np.sum(a * b, axis=-1)
    return np.arctan2(cross, dot)


def _latlon_of(direction: np.ndarray) -> tuple[float, float]:
    """Return ``(latitude_deg, longitude_deg)`` of a world direction."""
    d = np.asarray(direction, dtype=np.float64)
    d = d / np.linalg.norm(d)
    lat = math.degrees(math.asin(float(np.clip(-d[1], -1.0, 1.0))))
    lon = math.degrees(math.atan2(float(d[0]), float(d[2])))
    return lat, lon


# --- #6 rotation semantics --------------------------------------------------

OPTICAL_AXIS = np.array([0.0, 0.0, 1.0])


@pytest.mark.parametrize("pitch_deg", [-30.0, -10.0, 0.0, 10.0, 30.0])
def test_pitch_moves_optical_axis_to_that_latitude(pitch_deg: float) -> None:
    """#6: pitch = +30 deg puts the image centre at latitude +30 deg."""
    R = rotation.rotation_from_ypr(0.0, pitch_deg, 0.0)
    lat, lon = _latlon_of(R @ OPTICAL_AXIS)
    assert lat == pytest.approx(pitch_deg, abs=1e-9)
    assert lon == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("yaw_deg", [-120.0, -45.0, 0.0, 45.0, 120.0])
def test_yaw_moves_optical_axis_to_that_longitude(yaw_deg: float) -> None:
    """#6: yaw = +45 deg puts the image centre at longitude +45 deg."""
    R = rotation.rotation_from_ypr(yaw_deg, 0.0, 0.0)
    lat, lon = _latlon_of(R @ OPTICAL_AXIS)
    assert lat == pytest.approx(0.0, abs=1e-9)
    assert lon == pytest.approx(yaw_deg, abs=1e-9)


def test_pitch_latitude_on_panorama_pixels() -> None:
    """#6 stated in panorama pixels: +30 deg pitch is 1/6 of the height above centre."""
    R = rotation.rotation_from_ypr(0.0, 30.0, 0.0)
    px = equirect.dirs_to_equirect_px(R @ OPTICAL_AXIS, W, H)
    expected_v = (0.5 - 30.0 / 180.0) * H - 0.5
    assert px[0] == pytest.approx((0.5) * W - 0.5, abs=1e-9)
    assert px[1] == pytest.approx(expected_v, abs=1e-9)


def test_positive_yaw_moves_right_on_panorama() -> None:
    """#6: yaw > 0 turns the camera right, i.e. towards larger u."""
    centre_u = equirect.dirs_to_equirect_px(OPTICAL_AXIS, W, H)[0]
    right_u = equirect.dirs_to_equirect_px(
        rotation.rotation_from_ypr(45.0, 0.0, 0.0) @ OPTICAL_AXIS, W, H
    )[0]
    assert right_u > centre_u


def test_roll_keeps_optical_axis_fixed() -> None:
    for roll in (-15.0, 15.0):
        R = rotation.rotation_from_ypr(0.0, 0.0, roll)
        assert np.allclose(R @ OPTICAL_AXIS, OPTICAL_AXIS, atol=1e-12)


def test_rotation_is_orthonormal() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        ypr = rng.uniform(-180.0, 180.0, size=3)
        R = rotation.rotation_from_ypr(*ypr)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)


def test_rotation_composition_order() -> None:
    """SPEC.md 3.3: R = R_yaw @ R_pitch @ R_roll."""
    ypr = (37.0, -12.0, 5.0)
    expected = (
        rotation.rot_y(math.radians(ypr[0]))
        @ rotation.rot_x(math.radians(ypr[1]))
        @ rotation.rot_z(math.radians(ypr[2]))
    )
    assert np.allclose(rotation.rotation_from_ypr(*ypr), expected, atol=1e-15)


# --- equirectangular mapping conventions ------------------------------------


def test_equirect_anchor_points() -> None:
    """SPEC.md 3.1: phi = 0 is the horizontal centre, latitude is up-positive."""
    px = equirect.dirs_to_equirect_px(np.array([0.0, 0.0, 1.0]), W, H)
    assert px[0] == pytest.approx(W / 2.0 - 0.5, abs=1e-9)
    assert px[1] == pytest.approx(H / 2.0 - 0.5, abs=1e-9)

    # Straight up (y negative in OpenCV camera axes) is the top row.
    up = equirect.dirs_to_equirect_px(np.array([0.0, -1.0, 0.0]), W, H)
    assert up[1] == pytest.approx(-0.5, abs=1e-9)
    down = equirect.dirs_to_equirect_px(np.array([0.0, 1.0, 0.0]), W, H)
    assert down[1] == pytest.approx(H - 0.5, abs=1e-9)


def test_px_to_dirs_roundtrip() -> None:
    idx_u = np.arange(0, W, 13, dtype=np.float64)
    idx_v = np.arange(0, H, 7, dtype=np.float64)
    uu, vv = np.meshgrid(idx_u, idx_v, indexing="xy")
    px = np.stack([uu, vv], axis=-1)

    dirs = equirect.equirect_px_to_dirs(px, W, H)
    assert np.abs(np.linalg.norm(dirs, axis=-1) - 1.0).max() < 1e-12

    px_back = equirect.dirs_to_equirect_px(dirs, W, H)
    assert np.abs(px_back - px).max() < 1e-9


def test_dirs_to_px_roundtrip() -> None:
    rng = np.random.default_rng(1)
    dirs = rng.normal(size=(2000, 3))
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

    px = equirect.dirs_to_equirect_px(dirs, W, H)
    dirs_back = equirect.equirect_px_to_dirs(px, W, H)
    ang = _angle_between(dirs_back, dirs)
    assert ang.max() < 1e-9


# --- sampling ---------------------------------------------------------------


def _ramp_pano() -> np.ndarray:
    """Panorama whose red channel encodes the column index."""
    pano = np.zeros((H, W, 3), dtype=np.uint8)
    pano[:, :, 0] = (np.arange(W) % 256).astype(np.uint8)
    pano[:, :, 1] = (np.arange(H) % 256).astype(np.uint8)[:, None]
    return pano


def test_sample_equirect_hits_pixel_centres() -> None:
    pano = _ramp_pano()
    idx_u = np.arange(0, W, 11, dtype=np.float64)
    idx_v = np.arange(0, H, 5, dtype=np.float64)
    uu, vv = np.meshgrid(idx_u, idx_v, indexing="xy")

    out = equirect.sample_equirect(pano, uu, vv)
    assert out.shape == (idx_v.size, idx_u.size, 3)
    assert np.array_equal(out, pano[np.ix_(idx_v.astype(int), idx_u.astype(int))])


def test_sample_equirect_wraps_horizontally() -> None:
    """SPEC.md 4.4: u must wrap around, not clamp."""
    pano = _ramp_pano()
    v = np.full((1, 3), H / 2.0)
    u = np.array([[-1.0, float(W), float(W) + 5.0]])

    out = equirect.sample_equirect(pano, u, v)
    expected_cols = [W - 1, 0, 5]
    for k, col in enumerate(expected_cols):
        assert out[0, k, 0] == pano[H // 2, col, 0]

    # A sample in the seam blends the two edge columns, it does not clamp.
    seam = equirect.sample_equirect(
        pano, np.array([[-0.5]]), np.array([[H / 2.0]])
    )
    blended = 0.5 * (int(pano[H // 2, W - 1, 0]) + int(pano[H // 2, 0, 0]))
    assert abs(int(seam[0, 0, 0]) - blended) <= 1


def test_sample_equirect_replicates_vertically() -> None:
    """SPEC.md 4.4: the poles replicate, they must not wrap to the other pole."""
    pano = _ramp_pano()
    u = np.full((1, 2), 10.0)
    v = np.array([[-5.0, float(H) + 5.0]])

    out = equirect.sample_equirect(pano, u, v)
    assert out[0, 0, 1] == pano[0, 10, 1]
    assert out[0, 1, 1] == pano[H - 1, 10, 1]
