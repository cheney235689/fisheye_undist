"""Camera parameter sampling with feasibility rejection.

SPEC.md 3.6 and 4.6.  Given an ``np.random.Generator`` this module draws one
camera: a full-angle FOV at ``r_ref``, a UCM ``xi``, whether the image is
circular or full-frame, and an orientation.  It performs no I/O and owns no
randomness of its own -- the caller supplies the generator, and
:func:`sample_seed` derives that generator's seed from stable inputs so the
result never depends on worker count or execution order (SPEC.md 4.11).

``(Theta, xi)`` is drawn uniformly from the rectangle and rejected unless it
lands in the feasible region, which yields a uniform distribution over that
region (SPEC.md 4.6).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from geometry import ucm

__all__ = [
    "CameraSample",
    "SamplingConfig",
    "edge_stretch",
    "sample_camera",
    "sample_camera_with_attempts",
    "sample_seed",
    "is_feasible",
]

MAX_ATTEMPTS = 10_000


@dataclass(frozen=True)
class CameraSample:
    """One drawn camera (SPEC.md 3.6)."""

    f: float
    xi: float
    fov_deg: float  # full angle at r_ref
    circular: bool
    r_ref: float
    yaw_deg: float
    pitch_deg: float
    roll_deg: float


@dataclass(frozen=True)
class SamplingConfig:
    """Ranges for :func:`sample_camera` (SPEC.md 4.6 defaults)."""

    image_size: int = 224
    circular_prob: float = 0.5
    fov_deg: tuple[float, float] = (100.0, 220.0)
    xi: tuple[float, float] = (0.0, 1.5)
    feasibility_margin: float = 0.05
    f_min_px: float = 30.0
    max_edge_stretch: float = 4.0
    yaw_deg: tuple[float, float] = (-180.0, 180.0)
    pitch_deg: tuple[float, float] = (-30.0, 30.0)
    roll_deg: tuple[float, float] = (-15.0, 15.0)

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> "SamplingConfig":
        """Build from a parsed ``configs/data.yaml``, ignoring unrelated keys."""
        def pair(name: str, default: tuple[float, float]) -> tuple[float, float]:
            value = cfg.get(name, default)
            return (float(value[0]), float(value[1]))

        return cls(
            image_size=int(cfg.get("image_size", 224)),
            circular_prob=float(cfg.get("circular_prob", 0.5)),
            fov_deg=pair("fov_deg", (100.0, 220.0)),
            xi=pair("xi", (0.0, 1.5)),
            feasibility_margin=float(cfg.get("feasibility_margin", 0.05)),
            f_min_px=float(cfg.get("f_min_px", 30.0)),
            max_edge_stretch=float(cfg.get("max_edge_stretch", 4.0)),
            yaw_deg=pair("yaw_deg", (-180.0, 180.0)),
            pitch_deg=pair("pitch_deg", (-30.0, 30.0)),
            roll_deg=pair("roll_deg", (-15.0, 15.0)),
        )

    def r_ref(self, circular: bool) -> float:
        """Reference radius: ``S/2`` circular, ``S/sqrt(2)`` full-frame (SPEC.md 4.3)."""
        size = float(self.image_size)
        return size / 2.0 if circular else size / np.sqrt(2.0)


def edge_stretch(fov_deg: float, xi: float) -> float:
    """Radial magnification at the edge of the view, relative to the centre.

    The radial scale of the projection is ``dr/dtheta = f (1 + xi cos t) /
    (xi + cos t)^2`` (SPEC.md 4.2), which at ``t = 0`` is ``f / (1 + xi)``.
    Their ratio at ``t = FOV/2`` is

        M = (1 + xi) (1 + xi cos t) / (xi + cos t)^2

    so M = 1 means the edge is sampled at the same rate as the centre and
    large M means the edge is smeared across many pixels while carrying little
    new information (SPEC.md 4.6, condition 4).
    """
    half = np.radians(float(fov_deg)) / 2.0
    cos_half = float(np.cos(half))
    denom = (xi + cos_half) ** 2
    if denom <= 0.0:
        return float("inf")
    return float((1.0 + xi) * (1.0 + xi * cos_half) / denom)


def is_feasible(
    fov_deg: float, xi: float, r_ref: float, cfg: SamplingConfig
) -> bool:
    """The four acceptance conditions of SPEC.md 4.6.

    1. ``xi + cos(Theta/2) >= feasibility_margin`` -- f is positive and not
       about to blow up.
    2. ``xi > 1`` also needs ``cos(Theta/2) >= -1/xi + feasibility_margin`` --
       stay inside the monotone range.
    3. ``f >= f_min_px``.
    4. ``edge_stretch(fov, xi) <= max_edge_stretch`` -- the edge of the frame
       is not smeared beyond a usable magnification.
    """
    half = np.radians(float(fov_deg)) / 2.0
    cos_half = float(np.cos(half))

    if xi + cos_half < cfg.feasibility_margin:
        return False
    if xi > 1.0 and cos_half < -1.0 / xi + cfg.feasibility_margin:
        return False

    f = r_ref * (xi + cos_half) / float(np.sin(half))
    if f < cfg.f_min_px:
        return False

    return edge_stretch(fov_deg, xi) <= cfg.max_edge_stretch


def sample_camera_with_attempts(
    rng: np.random.Generator, cfg: SamplingConfig
) -> tuple[CameraSample, int]:
    """Draw one camera; also report how many ``(Theta, xi)`` draws it took.

    ``circular`` is drawn once and held: it fixes ``r_ref``, and the rejection
    loop then produces a uniform draw over the feasible region for that
    ``r_ref``.  Orientation is drawn after acceptance so it cannot influence
    the rejection rate.
    """
    circular = bool(rng.random() < cfg.circular_prob)
    r_ref = cfg.r_ref(circular)

    attempts = 0
    for _ in range(MAX_ATTEMPTS):
        attempts += 1
        fov_deg = float(rng.uniform(*cfg.fov_deg))
        xi = float(rng.uniform(*cfg.xi))
        if is_feasible(fov_deg, xi, r_ref, cfg):
            break
    else:
        raise RuntimeError(
            f"no feasible (fov, xi) after {MAX_ATTEMPTS} draws; "
            f"check fov_deg={cfg.fov_deg}, xi={cfg.xi}, f_min_px={cfg.f_min_px}, "
            f"max_edge_stretch={cfg.max_edge_stretch}"
        )

    f = ucm.f_from_fov(fov_deg, xi, r_ref)
    sample = CameraSample(
        f=f,
        xi=xi,
        fov_deg=fov_deg,
        circular=circular,
        r_ref=r_ref,
        yaw_deg=float(rng.uniform(*cfg.yaw_deg)),
        pitch_deg=float(rng.uniform(*cfg.pitch_deg)),
        roll_deg=float(rng.uniform(*cfg.roll_deg)),
    )
    return sample, attempts


def sample_camera(rng: np.random.Generator, cfg: SamplingConfig) -> CameraSample:
    """Draw one camera (SPEC.md 3.6)."""
    return sample_camera_with_attempts(rng, cfg)[0]


def sample_seed(global_seed: int, split: str, asset_id: str, view_idx: int) -> int:
    """Per-sample seed from a stable hash.

    Python's built-in ``hash()`` is salted per process and must not be used
    (SPEC.md 6); SHA-256 over the joined fields gives the same seed on every
    run, for any worker count and any ordering (SPEC.md 4.11).
    """
    payload = f"{int(global_seed)}|{split}|{asset_id}|{int(view_idx)}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    # 63 bits keeps the value inside a positive int64, which numpy accepts.
    return int.from_bytes(digest[:8], "big") >> 1
