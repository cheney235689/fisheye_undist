"""Equirectangular panorama mapping and sampling.

SPEC.md 3.1 / 3.4 / 4.4.  For a ``W x H`` panorama with ``W = 2H``::

    phi    = atan2(x, z)                      in [-pi, pi), 0 at image centre
    lambda = arcsin(-y)                       positive upwards
    u_e    = (phi / 2pi + 0.5) * W - 0.5
    v_e    = (0.5 - lambda / pi) * H - 0.5
"""

import cv2
import numpy as np

__all__ = ["dirs_to_equirect_px", "equirect_px_to_dirs", "sample_equirect"]


def dirs_to_equirect_px(dirs: np.ndarray, W: int, H: int) -> np.ndarray:
    """World directions ``(..., 3)`` to panorama pixel coordinates ``(..., 2)``.

    ``dirs`` need not be unit length.
    """
    dirs = np.asarray(dirs, dtype=np.float64)
    if dirs.shape[-1] != 3:
        raise ValueError(f"dirs must have last dimension 3, got {dirs.shape}")

    norm = np.linalg.norm(dirs, axis=-1)
    if np.any(norm <= 0.0):
        raise ValueError("dirs contains a zero-length direction")
    x, y, z = dirs[..., 0] / norm, dirs[..., 1] / norm, dirs[..., 2] / norm

    phi = np.arctan2(x, z)
    lam = np.arcsin(np.clip(-y, -1.0, 1.0))

    u = (phi / (2.0 * np.pi) + 0.5) * W - 0.5
    v = (0.5 - lam / np.pi) * H - 0.5
    return np.stack([u, v], axis=-1)


def equirect_px_to_dirs(px: np.ndarray, W: int, H: int) -> np.ndarray:
    """Inverse of :func:`dirs_to_equirect_px`, returning unit vectors."""
    px = np.asarray(px, dtype=np.float64)
    if px.shape[-1] != 2:
        raise ValueError(f"px must have last dimension 2, got {px.shape}")

    phi = ((px[..., 0] + 0.5) / W - 0.5) * (2.0 * np.pi)
    lam = (0.5 - (px[..., 1] + 0.5) / H) * np.pi

    cos_lam = np.cos(lam)
    x = cos_lam * np.sin(phi)
    y = -np.sin(lam)
    z = cos_lam * np.cos(phi)
    return np.stack([x, y, z], axis=-1)


def sample_equirect(
    pano: np.ndarray, map_u: np.ndarray, map_v: np.ndarray
) -> np.ndarray:
    """Bilinear ``cv2.remap`` sampling with correct horizontal wrap-around.

    ``cv2.BORDER_WRAP`` cannot be used because it wraps vertically too, which
    corrupts the poles (SPEC.md 4.4).  Instead ``map_u`` is folded into
    ``[0, W)`` and the panorama is padded by one column on each side taken from
    the opposite edge; the vertical direction uses ``BORDER_REPLICATE``.
    """
    pano = np.asarray(pano)
    if pano.ndim not in (2, 3):
        raise ValueError(f"pano must be 2D or 3D, got shape {pano.shape}")
    map_u = np.asarray(map_u, dtype=np.float64)
    map_v = np.asarray(map_v, dtype=np.float64)
    if map_u.shape != map_v.shape:
        raise ValueError(
            f"map_u {map_u.shape} and map_v {map_v.shape} must have the same shape"
        )
    if map_u.ndim != 2:
        raise ValueError(f"maps must be 2D, got {map_u.shape}")

    W = pano.shape[1]

    # One padded column on each side is enough for bilinear interpolation once
    # map_u lies in [0, W): the sample never needs more than two neighbours.
    padded = np.concatenate([pano[:, W - 1 : W], pano, pano[:, 0:1]], axis=1)
    u_padded = np.mod(map_u, float(W)) + 1.0

    return cv2.remap(
        padded,
        u_padded.astype(np.float32),
        map_v.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
