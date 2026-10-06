"""Acceptance criteria #1-#5 (SPEC.md 5): the UCM itself."""

import math

import numpy as np
import pytest

from geometry import ucm

XI_VALUES = [0.0, 0.5, 1.0, 1.5]
SIZE = 224
R_REF_FULL = SIZE / math.sqrt(2.0)
R_REF_CIRCULAR = SIZE / 2.0
CX = CY = (SIZE - 1) / 2.0


def _angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle between unit vectors, conditioned for near-parallel inputs.

    ``arccos(dot)`` bottoms out around 1.5e-8 rad in float64 because its slope
    is infinite at 1; the cross-product form stays accurate there.
    """
    cross = np.linalg.norm(np.cross(a, b), axis=-1)
    dot = np.sum(a * b, axis=-1)
    return np.arctan2(cross, dot)


def _f_for(xi: float, fov_deg: float = 160.0, r_ref: float = R_REF_FULL) -> float:
    return ucm.f_from_fov(fov_deg, xi, r_ref)


def _pixel_grid(size: int = SIZE, step: int = 7) -> np.ndarray:
    idx = np.arange(0, size, step, dtype=np.float64)
    uu, vv = np.meshgrid(idx, idx, indexing="xy")
    return np.stack([uu, vv], axis=-1)


# --- #1 project o unproject -------------------------------------------------


@pytest.mark.parametrize("xi", XI_VALUES)
@pytest.mark.parametrize("r_ref", [R_REF_FULL, R_REF_CIRCULAR])
def test_project_unproject_roundtrip(xi: float, r_ref: float) -> None:
    """#1: pixel -> ray -> pixel is within 1e-4 px on the valid region."""
    f = _f_for(xi, r_ref=r_ref)
    px = _pixel_grid()

    rays, valid = ucm.unproject(px, f, xi, CX, CY)
    assert valid.any(), "no valid pixel to test"

    px_back, valid_back = ucm.project(rays, f, xi, CX, CY)
    # Every pixel the back-projection accepted must project back.
    assert valid_back[valid].all()

    err = np.linalg.norm(px_back[valid] - px[valid], axis=-1)
    assert err.max() < 1e-4, f"max pixel round-trip error {err.max():.3e}"


@pytest.mark.parametrize("xi", XI_VALUES)
def test_unproject_returns_unit_rays(xi: float) -> None:
    f = _f_for(xi)
    rays, valid = ucm.unproject(_pixel_grid(), f, xi, CX, CY)
    norms = np.linalg.norm(rays[valid], axis=-1)
    assert np.abs(norms - 1.0).max() < 1e-12


@pytest.mark.parametrize("xi", XI_VALUES)
def test_invalid_entries_are_finite(xi: float) -> None:
    """SPEC.md 4.2: invalid entries must not leak NaN/inf into the output."""
    # A grid far outside the valid disc for xi > 1, and extreme rays.
    px = _pixel_grid(size=4000, step=97) - 2000.0
    rays, _ = ucm.unproject(px, _f_for(xi), xi, CX, CY)
    assert np.isfinite(rays).all()

    rays_in = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, -1.0], [0.0, 1.0, -5.0]]
    )
    out, _ = ucm.project(rays_in, _f_for(xi), xi, CX, CY)
    assert np.isfinite(out).all()


# --- #2 unproject o project -------------------------------------------------


@pytest.mark.parametrize("xi", XI_VALUES)
def test_unproject_project_roundtrip(xi: float) -> None:
    """#2: ray -> pixel -> ray is within 1e-6 rad for theta < theta_max."""
    f = _f_for(xi)
    t_max = ucm.theta_max(xi)
    theta = np.linspace(0.0, 0.98 * t_max, 40)
    azimuth = np.linspace(0.0, 2.0 * np.pi, 17)[:-1]

    th, az = np.meshgrid(theta, azimuth, indexing="ij")
    rays = np.stack(
        [np.sin(th) * np.cos(az), np.sin(th) * np.sin(az), np.cos(th)], axis=-1
    )

    px, valid = ucm.project(rays, f, xi, CX, CY)
    assert valid.all(), "all sampled rays should be inside theta_max"

    rays_back, valid_back = ucm.unproject(px, f, xi, CX, CY)
    assert valid_back.all()

    ang = _angle_between(rays_back, rays)
    assert ang.max() < 1e-6, f"max angular round-trip error {ang.max():.3e} rad"


# --- #3 pinhole degeneracy --------------------------------------------------


def test_pinhole_degeneracy_radius() -> None:
    """#3: xi = 0 reduces to the pinhole model r = f * tan(theta)."""
    f = 123.456
    theta = np.linspace(0.0, np.radians(85.0), 200)
    r = ucm.radius_from_theta(theta, f, 0.0)
    expected = f * np.tan(theta)
    assert np.abs(r - expected).max() < 1e-9


def test_pinhole_degeneracy_projection() -> None:
    """#3: xi = 0 projection equals the pinhole u = f*x/z + cx."""
    f = 123.456
    rng = np.random.default_rng(0)
    rays = rng.normal(size=(500, 3))
    rays[:, 2] = np.abs(rays[:, 2]) + 0.1  # keep them in front of the camera

    px, valid = ucm.project(rays, f, 0.0, CX, CY)
    assert valid.all()
    expected_u = f * rays[:, 0] / rays[:, 2] + CX
    expected_v = f * rays[:, 1] / rays[:, 2] + CY
    assert np.abs(px[:, 0] - expected_u).max() < 1e-9
    assert np.abs(px[:, 1] - expected_v).max() < 1e-9


def test_theta_max_at_zero_xi() -> None:
    assert ucm.theta_max(0.0) == pytest.approx(np.pi / 2.0, abs=1e-15)


# --- #4 theta_max consistency -----------------------------------------------


@pytest.mark.parametrize("xi", [1.2, 1.5, 2.0, 3.0])
def test_theta_max_matches_unproject_boundary(xi: float) -> None:
    """#4: for xi > 1 the back-projection limit rho_max maps to theta_max."""
    f = 100.0
    rho_max = 1.0 / math.sqrt(xi * xi - 1.0)

    px = np.array([[CX + f * rho_max, CY]])
    rays, valid = ucm.unproject(px, f, xi, CX, CY)
    assert valid.all()

    theta = math.acos(float(np.clip(rays[0, 2], -1.0, 1.0)))
    assert theta == pytest.approx(ucm.theta_max(xi), abs=1e-9)

    # Just outside the limit the back-projection must report invalid.
    px_out = np.array([[CX + f * rho_max * (1.0 + 1e-6), CY]])
    _, valid_out = ucm.unproject(px_out, f, xi, CX, CY)
    assert not valid_out.any()


@pytest.mark.parametrize("xi", [0.0, 0.5, 1.0])
def test_theta_max_is_asymptote_for_small_xi(xi: float) -> None:
    """#4: for xi <= 1 the radius diverges, so theta -> theta_max as rho -> inf."""
    f = 100.0
    px = np.array([[CX + f * 1e8, CY]])
    rays, valid = ucm.unproject(px, f, xi, CX, CY)
    assert valid.all()

    theta = math.acos(float(np.clip(rays[0, 2], -1.0, 1.0)))
    assert theta == pytest.approx(ucm.theta_max(xi), abs=1e-6)


@pytest.mark.parametrize("xi", XI_VALUES)
def test_project_rejects_beyond_theta_max(xi: float) -> None:
    t_max = ucm.theta_max(xi)
    if t_max >= np.pi - 1e-12:
        pytest.skip("theta_max is pi, nothing lies beyond it")
    theta = t_max + np.array([1e-9, 1e-3, 1e-2])
    rays = np.stack([np.sin(theta), np.zeros_like(theta), np.cos(theta)], axis=-1)
    _, valid = ucm.project(rays, 100.0, xi, CX, CY)
    assert not valid.any()


# --- #5 FOV <-> f -----------------------------------------------------------


@pytest.mark.parametrize("xi", XI_VALUES)
@pytest.mark.parametrize("r_ref", [R_REF_FULL, R_REF_CIRCULAR])
def test_fov_f_roundtrip(xi: float, r_ref: float) -> None:
    """#5: fov_from_f(f_from_fov(theta)) == theta within 1e-6 degrees."""
    fov_limit_deg = 2.0 * math.degrees(ucm.theta_max(xi))
    fovs = np.linspace(40.0, 0.95 * fov_limit_deg, 25)

    for fov in fovs:
        f = ucm.f_from_fov(float(fov), xi, r_ref)
        assert f > 0.0
        fov_back = ucm.fov_from_f(f, xi, r_ref)
        assert fov_back == pytest.approx(float(fov), abs=1e-6)


@pytest.mark.parametrize("xi", XI_VALUES)
def test_f_from_fov_puts_fov_at_r_ref(xi: float) -> None:
    """f_from_fov is defined by r(fov/2) == r_ref."""
    fov = 150.0
    f = ucm.f_from_fov(fov, xi, R_REF_FULL)
    r = float(ucm.radius_from_theta(np.radians(fov) / 2.0, f, xi))
    assert r == pytest.approx(R_REF_FULL, rel=1e-12)


@pytest.mark.parametrize("xi", XI_VALUES)
def test_f_from_fov_rejects_infeasible_fov(xi: float) -> None:
    fov_limit_deg = 2.0 * math.degrees(ucm.theta_max(xi))
    with pytest.raises(ValueError):
        ucm.f_from_fov(fov_limit_deg + 1.0, xi, R_REF_FULL)
    with pytest.raises(ValueError):
        ucm.f_from_fov(0.0, xi, R_REF_FULL)
