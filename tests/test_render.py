"""Acceptance criteria #7-#8 (SPEC.md 5): panorama -> fisheye rendering.

Correctness is checked with a *direction-coded* panorama: each panorama pixel
stores its own world direction in its RGB value, linearly scaled from
``[-1, 1]`` to ``[0, 255]``.  The encoding is linear, so bilinear sampling and
``INTER_AREA`` reduction both stay meaningful, and the direction can be read
straight back out of the rendered image and compared with
:func:`geometry.ucm.unproject`.

Note the accuracy floor this imposes: ``render_fisheye`` returns uint8, so a
decoded direction carries up to half a quantisation step (``1/255`` of the
``[-1, 1]`` range) of error per axis, about 0.2 deg.  That is why #7 is stated
as a median.
"""

import math

import cv2
import numpy as np
import pytest

from geometry import equirect, rotation, ucm
from geometry.render import FisheyeCamera, fisheye_to_perspective, render_fisheye

PANO_W, PANO_H = 1024, 512
SIZE = 224

# (xi, fov_deg, circular, (yaw, pitch, roll))
CASES = [
    (0.0, 170.0, False, (0.0, 0.0, 0.0)),
    (0.0, 120.0, True, (30.0, 10.0, -5.0)),
    (0.5, 200.0, False, (-100.0, -20.0, 10.0)),
    (0.5, 160.0, True, (150.0, 25.0, 15.0)),
    (1.0, 220.0, True, (0.0, 30.0, 0.0)),
    (1.0, 180.0, False, (60.0, -30.0, -12.0)),
    (1.5, 210.0, True, (-45.0, 15.0, 8.0)),
    (1.5, 140.0, False, (120.0, -8.0, 3.0)),
]


def _camera(xi: float, fov_deg: float, circular: bool) -> FisheyeCamera:
    probe = FisheyeCamera(f=1.0, xi=xi, size=SIZE, circular=circular)
    f = ucm.f_from_fov(fov_deg, xi, probe.r_ref)
    return FisheyeCamera(f=f, xi=xi, size=SIZE, circular=circular)


def _encode_dirs(dirs: np.ndarray) -> np.ndarray:
    return np.clip(np.rint((dirs + 1.0) * 0.5 * 255.0), 0.0, 255.0).astype(np.uint8)


def _decode_dirs(img: np.ndarray) -> np.ndarray:
    d = img.astype(np.float64) / 255.0 * 2.0 - 1.0
    norm = np.linalg.norm(d, axis=-1, keepdims=True)
    return d / np.where(norm > 0.0, norm, 1.0)


@pytest.fixture(scope="module")
def direction_pano() -> np.ndarray:
    """Panorama whose RGB encodes the world direction of each pixel."""
    uu, vv = np.meshgrid(
        np.arange(PANO_W, dtype=np.float64),
        np.arange(PANO_H, dtype=np.float64),
        indexing="xy",
    )
    dirs = equirect.equirect_px_to_dirs(np.stack([uu, vv], axis=-1), PANO_W, PANO_H)
    return _encode_dirs(dirs)


def _output_grid(size: int = SIZE) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(size, dtype=np.float64)
    return np.meshgrid(idx, idx, indexing="xy")


def _expected_world_dirs(cam: FisheyeCamera, R: np.ndarray):
    """Ground-truth world direction and validity for every output pixel centre."""
    uu, vv = _output_grid(cam.size)
    rays, valid = ucm.unproject(
        np.stack([uu, vv], axis=-1), cam.f, cam.xi, cam.cx, cam.cy
    )
    return rays @ R.T, valid


# --- #7 render round-trip ---------------------------------------------------


@pytest.mark.parametrize("supersample", [1, 2, 3])
@pytest.mark.parametrize("xi,fov_deg,circular,ypr", CASES)
def test_render_direction_roundtrip(
    direction_pano: np.ndarray,
    supersample: int,
    xi: float,
    fov_deg: float,
    circular: bool,
    ypr: tuple[float, float, float],
) -> None:
    """#7: directions decoded from the render match unproject within 0.3 deg (median)."""
    cam = _camera(xi, fov_deg, circular)
    R = rotation.rotation_from_ypr(*ypr)

    image, mask = render_fisheye(direction_pano, cam, R, supersample=supersample)
    assert image.shape == (SIZE, SIZE, 3)
    assert image.dtype == np.uint8
    assert mask.shape == (SIZE, SIZE)
    assert mask.dtype == np.bool_

    expected, valid = _expected_world_dirs(cam, R)
    sel = mask & valid
    assert sel.sum() > 0.5 * mask.sum()

    decoded = _decode_dirs(image)
    dot = np.clip(np.sum(decoded[sel] * expected[sel], axis=-1), -1.0, 1.0)
    ang_deg = np.degrees(np.arccos(dot))

    median = float(np.median(ang_deg))
    assert median < 0.3, f"median angular error {median:.4f} deg"


def test_render_zeroes_outside_circular_aperture(direction_pano: np.ndarray) -> None:
    """SPEC.md 3.5: a circular fisheye fills r > r_ref with 0."""
    cam = _camera(1.0, 200.0, circular=True)
    R = rotation.rotation_from_ypr(0.0, 0.0, 0.0)
    image, mask = render_fisheye(direction_pano, cam, R, supersample=2)

    uu, vv = _output_grid()
    r = np.hypot(uu - cam.cx, vv - cam.cy)
    outside = r > cam.r_ref + 1.0
    assert outside.any()
    assert not mask[outside].any()
    assert (image[outside] == 0).all()


def test_render_full_frame_covers_whole_image(direction_pano: np.ndarray) -> None:
    """SPEC.md 4.3: with r_ref = S/sqrt(2) a full-frame view has content everywhere."""
    cam = _camera(0.5, 180.0, circular=False)
    R = rotation.rotation_from_ypr(0.0, 0.0, 0.0)
    _, mask = render_fisheye(direction_pano, cam, R, supersample=2)
    assert mask.all()


def test_render_respects_rotation(direction_pano: np.ndarray) -> None:
    """The principal point must look along R @ (0, 0, 1).

    ``cx = cy = (S-1)/2`` falls between pixels, so no single pixel sits on the
    optical axis; the four straddling pixels are symmetric about it and their
    mean direction recovers the axis.
    """
    cam = _camera(0.5, 160.0, circular=False)
    lo, hi = int(math.floor(cam.cx)), int(math.ceil(cam.cx))
    for ypr in [(0.0, 0.0, 0.0), (40.0, 20.0, 0.0), (-90.0, -25.0, 10.0)]:
        R = rotation.rotation_from_ypr(*ypr)
        image, mask = render_fisheye(direction_pano, cam, R, supersample=1)
        assert mask[lo : hi + 1, lo : hi + 1].all()

        decoded = _decode_dirs(image[lo : hi + 1, lo : hi + 1]).reshape(-1, 3)
        mean_dir = decoded.mean(axis=0)
        mean_dir /= np.linalg.norm(mean_dir)

        expected = R @ np.array([0.0, 0.0, 1.0])
        ang = math.degrees(
            math.atan2(
                float(np.linalg.norm(np.cross(mean_dir, expected))),
                float(np.dot(mean_dir, expected)),
            )
        )
        assert ang < 0.5, f"ypr={ypr} centre direction off by {ang:.4f} deg"


def test_render_rejects_bad_inputs(direction_pano: np.ndarray) -> None:
    cam = _camera(0.5, 160.0, circular=False)
    R = np.eye(3)
    with pytest.raises(ValueError):
        render_fisheye(direction_pano[:, :, 0], cam, R)
    with pytest.raises(ValueError):
        render_fisheye(direction_pano.astype(np.float32), cam, R)
    with pytest.raises(ValueError):
        render_fisheye(direction_pano, cam, np.eye(4))
    with pytest.raises(ValueError):
        render_fisheye(direction_pano, cam, R, supersample=0)


# --- #8 supersampling consistency -------------------------------------------


@pytest.mark.parametrize("supersample", [2, 3, 4])
@pytest.mark.parametrize("xi,fov_deg,circular,ypr", CASES)
def test_supersample_coordinate_scaling_is_exact(
    supersample: int, xi: float, fov_deg: float, circular: bool, ypr
) -> None:
    """#8: key directions land on the same place at s = 1 and at s > 1.

    SPEC.md 4.5 renders with ``f' = s*f`` and ``cx' = s*cx + (s-1)/2``, which
    under the pixel-centre convention must satisfy ``u' = s*u + (s-1)/2``
    exactly for every direction.  A wrong half-pixel offset here shifts the
    whole rendered image, so this is checked far tighter than the 0.5 px the
    spec asks for.
    """
    cam = _camera(xi, fov_deg, circular)
    s = supersample
    f_hi = s * cam.f
    c_hi = s * cam.cx + (s - 1) / 2.0

    # Key directions: the optical axis plus a theta/azimuth fan out to the FOV.
    theta_ref = math.radians(fov_deg) / 2.0
    theta = np.linspace(0.0, theta_ref, 9)
    azimuth = np.linspace(0.0, 2.0 * np.pi, 13)[:-1]
    th, az = np.meshgrid(theta, azimuth, indexing="ij")
    rays = np.stack(
        [np.sin(th) * np.cos(az), np.sin(th) * np.sin(az), np.cos(th)], axis=-1
    )

    px_lo, valid_lo = ucm.project(rays, cam.f, cam.xi, cam.cx, cam.cy)
    px_hi, valid_hi = ucm.project(rays, f_hi, cam.xi, c_hi, c_hi)
    assert np.array_equal(valid_lo, valid_hi)
    assert valid_lo.any()

    px_hi_in_lo = (px_hi - (s - 1) / 2.0) / s
    err = np.linalg.norm(px_hi_in_lo[valid_lo] - px_lo[valid_lo], axis=-1)
    assert err.max() < 1e-9, f"max key-direction mismatch {err.max():.3e} px"
    assert err.max() < 0.5


@pytest.mark.parametrize("supersample", [1, 2, 3])
@pytest.mark.parametrize("xi,fov_deg,circular,ypr", CASES)
def test_render_has_no_subpixel_shift(
    direction_pano: np.ndarray,
    supersample: int,
    xi: float,
    fov_deg: float,
    circular: bool,
    ypr: tuple[float, float, float],
) -> None:
    """#8: the rendered content sits on the pixel grid it is supposed to.

    Decoding a direction per pixel and projecting it back gives a position
    offset per pixel; uint8 quantisation makes each one noisy but zero-mean, so
    averaging over tens of thousands of pixels exposes any systematic shift
    (translation) or scale error (radial) well below the 0.5 px bound.
    """
    cam = _camera(xi, fov_deg, circular)
    R = rotation.rotation_from_ypr(*ypr)
    image, mask = render_fisheye(direction_pano, cam, R, supersample=supersample)
    expected, valid = _expected_world_dirs(cam, R)

    # Stay clear of the mask boundary, where zeroed pixels bleed into the
    # INTER_AREA reduction and corrupt the colour.
    interior = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    sel = interior & valid
    assert sel.sum() > 1000

    uu, vv = _output_grid()
    decoded_cam = _decode_dirs(image)[sel] @ R  # R.T @ d_world
    px, px_valid = ucm.project(decoded_cam, cam.f, cam.xi, cam.cx, cam.cy)

    du = px[:, 0] - uu[sel]
    dv = px[:, 1] - vv[sel]
    ru, rv = uu[sel] - cam.cx, vv[sel] - cam.cy
    radius = np.hypot(ru, rv)
    ok = px_valid & (radius > 1.0)
    assert ok.sum() > 1000

    shift = math.hypot(float(du[ok].mean()), float(dv[ok].mean()))
    radial = float(((du[ok] * ru[ok] + dv[ok] * rv[ok]) / radius[ok]).mean())

    assert shift < 0.5, f"mean translation {shift:.4f} px"
    assert abs(radial) < 0.5, f"mean radial offset {radial:.4f} px"

    if supersample == 1:
        # No area reduction, so the mapping should be essentially exact.
        assert shift < 0.02, f"s=1 mean translation {shift:.4f} px"
        assert abs(radial) < 0.05, f"s=1 mean radial offset {radial:.4f} px"

    # Unused in this check, kept to show expected is the reference direction.
    assert expected.shape == (SIZE, SIZE, 3)


@pytest.mark.parametrize("xi,fov_deg,circular,ypr", CASES)
def test_supersample_mask_agrees_away_from_boundary(
    direction_pano: np.ndarray,
    xi: float,
    fov_deg: float,
    circular: bool,
    ypr: tuple[float, float, float],
) -> None:
    """#8: the valid masks at s = 1 and s = 2 agree outside the boundary band."""
    cam = _camera(xi, fov_deg, circular)
    R = rotation.rotation_from_ypr(*ypr)
    _, mask1 = render_fisheye(direction_pano, cam, R, supersample=1)
    _, mask2 = render_fisheye(direction_pano, cam, R, supersample=2)

    kernel = np.ones((5, 5), np.uint8)  # a 2 px band on each side

    def boundary_band(mask: np.ndarray) -> np.ndarray:
        m = mask.astype(np.uint8)
        return (cv2.dilate(m, kernel) ^ cv2.erode(m, kernel)).astype(bool)

    band = boundary_band(mask1) | boundary_band(mask2)
    disagreement = (mask1 ^ mask2) & ~band
    assert not disagreement.any(), (
        f"{disagreement.sum()} pixels disagree away from the boundary"
    )

    intersection = (mask1 & mask2).sum()
    union = (mask1 | mask2).sum()
    assert intersection / union > 0.99


SUPERSAMPLE_CONVERGENCE_CASES = [
    (0.5, 220.0, False),
    (0.3, 200.0, True),
]


@pytest.fixture(scope="module")
def noise_pano() -> np.ndarray:
    """Lightly blurred random noise: the hardest case for point sampling."""
    rng = np.random.default_rng(20240517)
    noise = rng.integers(0, 256, size=(PANO_H, PANO_W, 3), dtype=np.uint8)
    return cv2.GaussianBlur(noise, (0, 0), sigmaX=0.7, sigmaY=0.7)


@pytest.mark.parametrize("xi,fov_deg,circular", SUPERSAMPLE_CONVERGENCE_CASES)
def test_supersample_converges_to_reference(
    noise_pano: np.ndarray, xi: float, fov_deg: float, circular: bool
) -> None:
    """SPEC.md 4.5: the default s = 4 must already be close to converged.

    The image centre is the worst region: ``dr/dtheta`` is smallest at
    ``theta = 0``, so one output pixel there integrates the most panorama
    pixels and aliases hardest.  An s = 16 render is the reference.
    """
    cam = _camera(xi, fov_deg, circular)
    R = rotation.rotation_from_ypr(0.0, 0.0, 0.0)

    reference, _ = render_fisheye(noise_pano, cam, R, supersample=16)
    image, _ = render_fisheye(noise_pano, cam, R, supersample=4)

    lo = cam.size // 2 - 20
    hi = cam.size // 2 + 20
    centre_ref = reference[lo:hi, lo:hi].astype(np.float64)
    centre_img = image[lo:hi, lo:hi].astype(np.float64)
    assert centre_ref.shape == (40, 40, 3)

    mae = float(np.abs(centre_img - centre_ref).mean())
    assert mae < 2.5, f"s=4 vs s=16 mean absolute error {mae:.3f} / 255"


# --- FisheyeCamera and the perspective view ---------------------------------


def test_camera_derived_quantities() -> None:
    full = FisheyeCamera(f=100.0, xi=0.5, size=224, circular=False)
    circ = FisheyeCamera(f=100.0, xi=0.5, size=224, circular=True)
    assert full.cx == full.cy == (224 - 1) / 2
    assert full.r_ref == pytest.approx(224 / math.sqrt(2.0))
    assert circ.r_ref == pytest.approx(112.0)


@pytest.mark.parametrize("xi,fov_deg,circular,ypr", CASES)
def test_camera_fov_matches_construction(
    xi: float, fov_deg: float, circular: bool, ypr
) -> None:
    cam = _camera(xi, fov_deg, circular)
    assert cam.fov_deg == pytest.approx(fov_deg, abs=1e-6)


@pytest.mark.parametrize("xi,fov_deg,circular", [(c[0], c[1], c[2]) for c in CASES])
def test_fisheye_to_perspective_recovers_directions(
    direction_pano: np.ndarray, xi: float, fov_deg: float, circular: bool
) -> None:
    """Undistorting with the true intrinsics must yield a true pinhole view."""
    cam = _camera(xi, fov_deg, circular)
    R = rotation.rotation_from_ypr(25.0, -10.0, 5.0)
    image, _ = render_fisheye(direction_pano, cam, R, supersample=2)

    out_size, persp_fov = 128, 90.0
    persp = fisheye_to_perspective(image, cam, out_size, persp_fov)
    assert persp.shape == (out_size, out_size, 3)
    assert persp.dtype == np.uint8

    c_out = (out_size - 1) / 2.0
    f_out = c_out / math.tan(math.radians(persp_fov) / 2.0)
    uu, vv = _output_grid(out_size)
    rays = np.stack(
        [(uu - c_out) / f_out, (vv - c_out) / f_out, np.ones_like(uu)], axis=-1
    )
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    expected = rays @ R.T

    decoded = _decode_dirs(persp)
    filled = persp.any(axis=-1)
    assert filled.mean() > 0.9

    dot = np.clip(np.sum(decoded[filled] * expected[filled], axis=-1), -1.0, 1.0)
    median = float(np.degrees(np.median(np.arccos(dot))))
    assert median < 0.3, f"median undistortion error {median:.4f} deg"
