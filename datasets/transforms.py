"""Photometric and label-preserving geometric augmentation (SPEC.md 4.9).

Every op here leaves the camera intrinsics untouched, which is the whole
constraint: the regression target is ``(f, xi)``, so an augmentation that
changes the image geometry silently corrupts the label.

Allowed, and why
    Photometric ops (brightness, contrast, gamma, blur, noise, JPEG, vignette)
    never move a pixel.  Horizontal flip, vertical flip and 90 degree rotation
    do move pixels but map the pixel grid onto itself about its own centre; the
    principal point is at ``((S-1)/2, (S-1)/2)``, the image is square and the
    UCM is radially symmetric, so ``r(theta)`` -- and therefore ``f`` and
    ``xi`` -- is unchanged.

Forbidden
    Crop, resize, scale, translation, rotation by an arbitrary angle and any
    aspect-ratio change all rescale or shift the radial coordinate, so the
    recorded ``f`` no longer describes the image.  :data:`FORBIDDEN_OPS` names
    them so a test can assert none of them is wired in.

Transforms are callables of ``(image, rng)`` rather than ``(image)``: the
caller owns the randomness, so the same seed always produces the same output
and a DataLoader worker cannot silently reuse a parent process's global state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import cv2
import numpy as np

__all__ = [
    "ALLOWED_OPS",
    "FORBIDDEN_OPS",
    "Op",
    "Compose",
    "RandomBrightnessContrast",
    "RandomGamma",
    "RandomGaussianBlur",
    "RandomGaussianNoise",
    "RandomJPEG",
    "RandomVignette",
    "RandomDihedral",
    "build_train_transform",
    "build_eval_transform",
]

# Names a test can check against; see SPEC.md 4.9.
ALLOWED_OPS = frozenset(
    {
        "brightness_contrast",
        "gamma",
        "gaussian_blur",
        "gaussian_noise",
        "jpeg",
        "vignette",
        "dihedral",
    }
)

FORBIDDEN_OPS = frozenset(
    {
        "crop",
        "random_crop",
        "resize",
        "scale",
        "zoom",
        "translate",
        "shift",
        "rotate",
        "affine",
        "perspective",
        "aspect_ratio",
        "random_resized_crop",
        "pad",
        "elastic",
    }
)


class Op:
    """One augmentation step.

    ``name`` is part of the contract: :func:`Compose.op_names` exposes it so a
    test can verify the pipeline contains nothing from :data:`FORBIDDEN_OPS`.
    """

    name: str = "op"
    changes_geometry: bool = False

    def __call__(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError


@dataclass
class Compose:
    """Apply ops in order, each drawing from the caller's generator."""

    ops: Sequence[Op] = field(default_factory=tuple)

    def op_names(self) -> list[str]:
        names: list[str] = []
        for op in self.ops:
            names.extend(op.op_names() if isinstance(op, Compose) else [op.name])
        return names

    def __call__(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        out = np.asarray(image)
        for op in self.ops:
            out = op(out, rng)
        return out


def _as_uint8(values: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


@dataclass
class RandomBrightnessContrast(Op):
    """``out = (image - 128) * contrast + 128 + brightness``."""

    brightness: float = 32.0
    contrast: tuple[float, float] = (0.8, 1.25)
    p: float = 0.8
    name: str = "brightness_contrast"

    def __call__(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if rng.random() >= self.p:
            return image
        delta = rng.uniform(-self.brightness, self.brightness)
        gain = rng.uniform(*self.contrast)
        values = (image.astype(np.float32) - 128.0) * gain + 128.0 + delta
        return _as_uint8(values)


@dataclass
class RandomGamma(Op):
    """Power-law tone change, applied through a 256-entry lookup table."""

    gamma: tuple[float, float] = (0.7, 1.4)
    p: float = 0.5
    name: str = "gamma"

    def __call__(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if rng.random() >= self.p:
            return image
        gamma = float(rng.uniform(*self.gamma))
        table = _as_uint8(np.power(np.arange(256) / 255.0, gamma) * 255.0)
        return table[image]


@dataclass
class RandomGaussianBlur(Op):
    """Isotropic blur; a symmetric kernel cannot shift the image centre."""

    sigma: tuple[float, float] = (0.3, 1.2)
    p: float = 0.3
    name: str = "gaussian_blur"

    def __call__(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if rng.random() >= self.p:
            return image
        sigma = float(rng.uniform(*self.sigma))
        return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)


@dataclass
class RandomGaussianNoise(Op):
    sigma: tuple[float, float] = (2.0, 12.0)
    p: float = 0.4
    name: str = "gaussian_noise"

    def __call__(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if rng.random() >= self.p:
            return image
        sigma = float(rng.uniform(*self.sigma))
        noise = rng.normal(0.0, sigma, size=image.shape)
        return _as_uint8(image.astype(np.float32) + noise)


@dataclass
class RandomJPEG(Op):
    """Re-encode at a random quality, which is what real datasets look like.

    The datasets hand us RGB (SPEC.md 2.2) while OpenCV's codec assumes BGR.
    JPEG subsamples chroma, so feeding it swapped channels would compress the
    wrong ones and tint the result; the round trip through BGR keeps the
    degradation photometric, which is the only kind SPEC.md 4.9 allows.
    """

    quality: tuple[int, int] = (45, 95)
    p: float = 0.3
    name: str = "jpeg"

    def __call__(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if rng.random() >= self.p:
            return image
        quality = int(rng.integers(self.quality[0], self.quality[1] + 1))
        ok, buffer = cv2.imencode(
            ".jpg",
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), quality],
        )
        if not ok:
            return image
        decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        return image if decoded is None else cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


@dataclass
class RandomVignette(Op):
    """Radially symmetric darkening centred on the principal point.

    Centred and radial, so it darkens equally at equal ``r`` and cannot bias
    the apparent image centre or scale.
    """

    strength: tuple[float, float] = (0.1, 0.5)
    p: float = 0.3
    name: str = "vignette"

    def __call__(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if rng.random() >= self.p:
            return image
        strength = float(rng.uniform(*self.strength))
        height, width = image.shape[:2]
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
        radius = np.hypot(xx - cx, yy - cy)
        radius /= max(float(radius.max()), 1e-6)
        falloff = (1.0 - strength * radius**2)[:, :, None]
        return _as_uint8(image.astype(np.float32) * falloff)


@dataclass
class RandomDihedral(Op):
    """One of the eight square symmetries: flips and 90 degree rotations.

    These are the only geometric changes SPEC.md 4.9 allows.  The dihedral
    group maps the pixel grid onto itself about its own centre, so with a
    square image, a centred principal point and a radially symmetric model,
    every pixel keeps its radius and the intrinsics are untouched.
    """

    p: float = 1.0
    name: str = "dihedral"
    changes_geometry: bool = True

    def __call__(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if image.shape[0] != image.shape[1]:
            raise ValueError(
                f"dihedral augmentation needs a square image, got {image.shape[:2]}"
            )
        if rng.random() >= self.p:
            return image
        out = np.rot90(image, k=int(rng.integers(0, 4)))
        if rng.random() < 0.5:
            out = out[:, ::-1]
        if rng.random() < 0.5:
            out = out[::-1, :]
        return np.ascontiguousarray(out)


def build_train_transform(**overrides: Any) -> Compose:
    """The default train-time pipeline (SPEC.md 4.9); train only."""
    ops: list[Op] = [
        RandomDihedral(),
        RandomBrightnessContrast(),
        RandomGamma(),
        RandomGaussianBlur(),
        RandomGaussianNoise(),
        RandomJPEG(),
        RandomVignette(),
    ]
    if overrides:
        by_name = {op.name: op for op in ops}
        for name, value in overrides.items():
            if name not in by_name:
                raise KeyError(f"unknown op {name!r}; have {sorted(by_name)}")
            if value is None:
                ops = [op for op in ops if op.name != name]
            else:
                by_name[name].p = float(value)
    return Compose(tuple(ops))


def build_eval_transform() -> Compose:
    """No augmentation at all -- val, test and stress (SPEC.md 4.9)."""
    return Compose(())


def check_no_forbidden_ops(transform: Callable[..., Any] | None) -> list[str]:
    """Return the forbidden op names a transform exposes, ideally empty."""
    if transform is None:
        return []
    names: Iterable[str] = (
        transform.op_names() if isinstance(transform, Compose)
        else getattr(transform, "op_names", lambda: [])()
    )
    return sorted(set(names) & FORBIDDEN_OPS)
