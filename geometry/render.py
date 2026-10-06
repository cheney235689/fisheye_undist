"""Panorama to fisheye rendering, and fisheye to perspective undistortion.

SPEC.md 3.5 / 4.5.  Rendering is done by back-projecting every output pixel
through the UCM, rotating the ray into the world frame and sampling the
panorama.  Supersampling renders at ``s`` times the resolution and reduces
with ``INTER_AREA``, which is what keeps a 224 px image covering up to 220
degrees from aliasing badly.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np

from . import equirect, ucm

__all__ = ["FisheyeCamera", "render_fisheye", "fisheye_to_perspective"]


@dataclass(frozen=True)
class FisheyeCamera:
    """A square, centred UCM fisheye camera.

    ``r_ref`` and the principal point are derived from ``size`` and
    ``circular`` rather than stored, so a camera object can never hold an
    inconsistent combination (SPEC.md 3.5 / 4.3).
    """

    f: float  # pixels, at the output resolution `size`
    xi: float
    size: int = 224  # S
    circular: bool = False

    @property
    def cx(self) -> float:
        """Principal point x; pixel centres are at integer coordinates."""
        return (self.size - 1) / 2.0

    @property
    def cy(self) -> float:
        """Principal point y."""
        return (self.size - 1) / 2.0

    @property
    def r_ref(self) -> float:
        """Reference radius the FOV is measured at: ``S/2`` circular, ``S/sqrt(2)`` full-frame."""
        if self.circular:
            return self.size / 2.0
        return self.size / math.sqrt(2.0)

    @property
    def fov_deg(self) -> float:
        """Full-angle field of view at :attr:`r_ref`."""
        return ucm.fov_from_f(self.f, self.xi, self.r_ref)


def _pixel_grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(uu, vv)`` of shape ``(size, size)``; ``uu`` is the column index."""
    idx = np.arange(size, dtype=np.float64)
    return np.meshgrid(idx, idx, indexing="xy")


def render_fisheye(
    pano: np.ndarray, cam: FisheyeCamera, R: np.ndarray, supersample: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """Render a fisheye view of an equirectangular panorama.

    Parameters
    ----------
    pano
        ``(H, W, 3)`` uint8 equirectangular panorama.
    cam
        Output camera.
    R
        3x3 camera-to-world rotation, ``d_world = R @ d_cam``.
    supersample
        Render at ``supersample * cam.size`` and reduce with ``INTER_AREA``.

    Returns
    -------
    image
        ``(S, S, 3)`` uint8.  Invalid pixels, and pixels beyond ``r_ref`` for a
        circular fisheye, are 0.
    mask
        ``(S, S)`` bool of valid pixels.
    """
    pano = np.asarray(pano)
    if pano.ndim != 3 or pano.shape[2] != 3:
        raise ValueError(f"pano must be (H, W, 3), got {pano.shape}")
    if pano.dtype != np.uint8:
        raise ValueError(f"pano must be uint8, got {pano.dtype}")
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"R must be 3x3, got {R.shape}")
    s = int(supersample)
    if s < 1:
        raise ValueError(f"supersample must be >= 1, got {supersample}")

    H, W = pano.shape[:2]
    S = cam.size
    size_hi = s * S
    # Pixel-centre convention: a box-filter reduction by s maps the low
    # resolution coordinate u to u_hi = s*u + (s-1)/2 (SPEC.md 4.5).
    f_hi = s * cam.f
    c_hi = s * cam.cx + (s - 1) / 2.0

    uu, vv = _pixel_grid(size_hi)
    rays, valid = ucm.unproject(np.stack([uu, vv], axis=-1), f_hi, cam.xi, c_hi, c_hi)

    if cam.circular:
        valid &= np.hypot(uu - c_hi, vv - c_hi) <= s * cam.r_ref

    # Invalid rays are zero vectors; swap in the optical axis so the equirect
    # mapping never sees a zero-length direction.
    safe_rays = np.where(valid[..., None], rays, np.array([0.0, 0.0, 1.0]))
    dirs_world = safe_rays @ R.T

    px = equirect.dirs_to_equirect_px(dirs_world, W, H)
    img_hi = equirect.sample_equirect(pano, px[..., 0], px[..., 1])
    img_hi = np.where(valid[..., None], img_hi, 0).astype(np.uint8)

    if s > 1:
        image = cv2.resize(img_hi, (S, S), interpolation=cv2.INTER_AREA)
        coverage = cv2.resize(
            valid.astype(np.float32), (S, S), interpolation=cv2.INTER_AREA
        )
        mask = coverage >= 0.5
    else:
        image, mask = img_hi, valid

    image = np.where(mask[..., None], image, 0).astype(np.uint8)
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


def fisheye_to_perspective(
    img: np.ndarray, cam: FisheyeCamera, out_size: int, fov_deg: float
) -> np.ndarray:
    """Undistort a fisheye image to a square perspective view.

    ``fov_deg`` is the full angle across the output width (equivalently its
    height, the output being square).  Pixels whose ray falls outside the
    fisheye image are 0.
    """
    img = np.asarray(img)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"img must be (S, S, 3), got {img.shape}")
    out_size = int(out_size)
    if out_size < 1:
        raise ValueError(f"out_size must be >= 1, got {out_size}")
    half_angle = np.radians(float(fov_deg)) / 2.0
    if not 0.0 < half_angle < np.pi / 2.0:
        raise ValueError(f"fov_deg must be in (0, 180), got {fov_deg}")

    c_out = (out_size - 1) / 2.0
    f_out = c_out / np.tan(half_angle)

    uu, vv = _pixel_grid(out_size)
    rays = np.stack(
        [(uu - c_out) / f_out, (vv - c_out) / f_out, np.ones_like(uu)], axis=-1
    )
    px, valid = ucm.project(rays, cam.f, cam.xi, cam.cx, cam.cy)

    u, v = px[..., 0], px[..., 1]
    S = cam.size
    valid &= (u >= -0.5) & (u <= S - 0.5) & (v >= -0.5) & (v <= S - 0.5)
    if cam.circular:
        valid &= np.hypot(u - cam.cx, v - cam.cy) <= cam.r_ref

    # Push invalid samples far outside the image so BORDER_CONSTANT zeroes them.
    map_x = np.where(valid, u, -1.0e6).astype(np.float32)
    map_y = np.where(valid, v, -1.0e6).astype(np.float32)
    out = cv2.remap(
        np.ascontiguousarray(img),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return np.ascontiguousarray(np.where(valid[..., None], out, 0).astype(img.dtype))
