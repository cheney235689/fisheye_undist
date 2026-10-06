"""Unified Camera Model (UCM) projection, unprojection and FOV conversion.

SPEC.md 4.1-4.3.  Camera coordinates follow OpenCV: x right, y down, z
forward.  Pixel centres sit on integer coordinates, so for an image of size
``S`` the principal point is ``cx = cy = (S - 1) / 2``.

Projection of a ray ``X = (x, y, z)`` with ``d = ||X||``::

    u = f * x / (xi * d + z) + cx
    v = f * y / (xi * d + z) + cy

which in terms of the angle ``theta`` to the optical axis is the radial
profile ``r(theta) = f * sin(theta) / (xi + cos(theta))``.
"""

import numpy as np

# Slack on the back-projection discriminant, in units where it is O(1).
_DISC_TOL = 1e-12

__all__ = [
    "project",
    "unproject",
    "radius_from_theta",
    "theta_max",
    "f_from_fov",
    "fov_from_f",
]


def theta_max(xi: float) -> float:
    """Largest ``theta`` for which the UCM projection is still monotone.

    ``theta_max(xi) = arccos(-min(xi, 1/xi))``, with ``pi/2`` at ``xi = 0``
    (SPEC.md 4.2).  For ``xi <= 1`` the radius diverges as ``theta`` reaches
    the bound; for ``xi > 1`` the projection folds back beyond it, so two
    directions would share one pixel and the region must be treated as
    invalid.
    """
    xi = float(xi)
    if xi < 0.0:
        raise ValueError(f"xi must be non-negative, got {xi}")
    if xi == 0.0:
        return float(np.pi / 2.0)
    return float(np.arccos(-min(xi, 1.0 / xi)))


def project(
    rays: np.ndarray, f: float, xi: float, cx: float, cy: float
) -> tuple[np.ndarray, np.ndarray]:
    """Project rays to pixels.

    Parameters
    ----------
    rays
        ``(..., 3)`` camera-frame directions.  They need not be unit length.
    f, xi, cx, cy
        UCM intrinsics in pixels (``xi`` is dimensionless).

    Returns
    -------
    pixels
        ``(..., 2)`` array of ``(u, v)``.  Entries where ``valid`` is False are
        filled with 0 rather than NaN (SPEC.md 4.2).
    valid
        ``(...)`` bool, True where ``xi * d + z > 0`` and ``theta < theta_max``.
    """
    rays = np.asarray(rays, dtype=np.float64)
    if rays.shape[-1] != 3:
        raise ValueError(f"rays must have last dimension 3, got {rays.shape}")
    f = float(f)
    if f <= 0.0:
        raise ValueError(f"f must be positive, got {f}")

    x, y, z = rays[..., 0], rays[..., 1], rays[..., 2]
    d = np.sqrt(x * x + y * y + z * z)
    nonzero = d > 0.0

    # cos(theta) only where the ray has length; elsewhere the value is unused.
    cos_theta = np.divide(z, d, out=np.zeros_like(d), where=nonzero)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    denom = xi * d + z
    valid = nonzero & (denom > 0.0) & (theta < theta_max(xi))

    # Substitute a harmless denominator where invalid so no NaN/inf is formed.
    safe_denom = np.where(valid, denom, 1.0)
    u = np.where(valid, f * x / safe_denom + cx, 0.0)
    v = np.where(valid, f * y / safe_denom + cy, 0.0)
    return np.stack([u, v], axis=-1), valid


def unproject(
    pixels: np.ndarray, f: float, xi: float, cx: float, cy: float
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project pixels to unit rays.

    With ``m = ((u - cx)/f, (v - cy)/f)`` and ``rho^2 = mx^2 + my^2``::

        eta = (xi + sqrt(1 + (1 - xi^2) * rho^2)) / (rho^2 + 1)
        ray = (eta * mx, eta * my, eta - xi)

    Returns
    -------
    rays
        ``(..., 3)`` unit vectors; invalid entries are filled with 0.
    valid
        ``(...)`` bool, True where ``1 + (1 - xi^2) * rho^2 >= 0`` — i.e. for
        ``xi > 1`` where ``rho <= 1 / sqrt(xi^2 - 1)`` (SPEC.md 4.2).
    """
    pixels = np.asarray(pixels, dtype=np.float64)
    if pixels.shape[-1] != 2:
        raise ValueError(f"pixels must have last dimension 2, got {pixels.shape}")
    f = float(f)
    if f <= 0.0:
        raise ValueError(f"f must be positive, got {f}")

    mx = (pixels[..., 0] - cx) / f
    my = (pixels[..., 1] - cy) / f
    rho2 = mx * mx + my * my

    disc = 1.0 + (1.0 - xi * xi) * rho2
    # At rho = rho_max the discriminant is analytically 0, so rounding can push
    # it a few ulps negative; a tiny tolerance keeps that boundary valid, which
    # matters because it is exactly theta_max (SPEC.md 4.2).
    valid = disc >= -_DISC_TOL
    # Feed 0 to sqrt where invalid, then overwrite via the mask (SPEC.md 4.2).
    eta = (xi + np.sqrt(np.where(valid, np.maximum(disc, 0.0), 0.0))) / (rho2 + 1.0)

    rays = np.stack([eta * mx, eta * my, eta - xi], axis=-1)
    rays = np.where(valid[..., None], rays, 0.0)
    return rays, valid


def radius_from_theta(
    theta: np.ndarray, f: float, xi: float, eps: float = 1e-6
) -> np.ndarray:
    """Radial profile ``r(theta) = f * sin(theta) / (xi + cos(theta))``.

    The denominator is clamped to ``eps`` so the function never divides by
    zero, matching the torch implementation in ``metrics/geometric.py``
    (SPEC.md 4.8).  Callers that need the validity of ``theta`` should consult
    :func:`theta_max`.
    """
    theta = np.asarray(theta, dtype=np.float64)
    denom = xi + np.cos(theta)
    denom = np.where(denom < eps, eps, denom)
    return float(f) * np.sin(theta) / denom


def f_from_fov(fov_deg: float, xi: float, r_ref: float) -> float:
    """Focal length that puts a full-angle FOV of ``fov_deg`` at radius ``r_ref``.

    ``f = r_ref * (xi + cos(theta)) / sin(theta)`` with ``theta = fov/2``
    (SPEC.md 4.3).
    """
    theta = np.radians(float(fov_deg)) / 2.0
    t_max = theta_max(xi)
    if not 0.0 < theta < t_max:
        raise ValueError(
            f"fov_deg/2 = {np.degrees(theta):.6f} deg is outside "
            f"(0, {np.degrees(t_max):.6f}) deg for xi = {xi}"
        )
    if r_ref <= 0.0:
        raise ValueError(f"r_ref must be positive, got {r_ref}")
    return float(r_ref * (xi + np.cos(theta)) / np.sin(theta))


def fov_from_f(f: float, xi: float, r_ref: float, iters: int = 100) -> float:
    """Inverse of :func:`f_from_fov`, solved numerically.

    ``r(theta)`` is monotone increasing on ``(0, theta_max)`` so the full-angle
    FOV is found by bisection (SPEC.md 4.3).
    """
    f = float(f)
    if f <= 0.0:
        raise ValueError(f"f must be positive, got {f}")
    if r_ref <= 0.0:
        raise ValueError(f"r_ref must be positive, got {r_ref}")

    t_max = theta_max(xi)
    if xi > 1.0:
        # r stays finite up to theta_max for xi > 1: r(theta_max) = f/sqrt(xi^2-1).
        r_at_max = f / np.sqrt(xi * xi - 1.0)
        if r_ref > r_at_max:
            raise ValueError(
                f"r_ref = {r_ref} exceeds the largest attainable radius "
                f"{r_at_max} for f = {f}, xi = {xi}"
            )

    lo, hi = 0.0, t_max
    for _ in range(int(iters)):
        mid = 0.5 * (lo + hi)
        if float(radius_from_theta(mid, f, xi)) < r_ref:
            lo = mid
        else:
            hi = mid
    return float(np.degrees(2.0 * (0.5 * (lo + hi))))
