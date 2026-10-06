"""Differentiable reprojection error and losses in torch (SPEC.md 3.12, 4.8).

Because the principal point is shared and the UCM is radially symmetric, the
reprojection error between two parameter sets collapses to a one-dimensional
comparison of radial profiles::

    theta_k = linspace(0, theta_ref_gt, n_theta)
    err     = mean_k |r_pred(theta_k) - r_gt(theta_k)|

This module is the torch half of the pair; ``geometry/ucm.py`` is the numpy
half, and the two must agree -- a test cross-checks them to 1e-4.

**Predictions are not ground truth.**  A network emits whatever it likes, so
every function here accepts ``xi < 0`` and ``xi > 1.5`` without raising.  In
particular nothing calls ``geometry.ucm.theta_max``, which rejects negative
``xi``: validity is decided inline by the conditions in SPEC.md 4.8, which are
written so that a non-positive ``xi`` simply fails the first one.

A clamped denominator is **not** a way to recover from an invalid prediction.
``torch.clamp(d, min=eps)`` is constant wherever it bites, so its gradient
there is exactly zero: the loss cannot feel which way is back.  The training
loss therefore masks the invalid angles out of the radial term entirely and
adds an explicit hinge barrier whose gradient does point inwards
(:func:`geometric_loss`).
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor

__all__ = [
    "radius_from_theta",
    "theta_grid",
    "is_valid_theta",
    "validity_terms",
    "denormalize",
    "normalize",
    "reprojection_error_px",
    "geometric_loss",
    "param_l1_loss",
]

DEFAULT_EPS = 1e-6
# Matches `feasibility_margin` in configs/data.yaml: the sampler never produces
# a camera closer than this to the degenerate boundary, so the loss treats
# anything nearer as out of bounds (SPEC.md 4.6, 4.8).
DEFAULT_MARGIN = 0.05
DEFAULT_BARRIER_WEIGHT = 1.0


def _as_tensor(value: Any, like: Tensor) -> Tensor:
    if isinstance(value, Tensor):
        return value.to(device=like.device, dtype=like.dtype)
    return torch.as_tensor(value, device=like.device, dtype=like.dtype)


def radius_from_theta(
    theta: Tensor, f: Tensor, xi: Tensor, eps: float = DEFAULT_EPS
) -> Tensor:
    """``r(theta) = f * sin(theta) / (xi + cos(theta))``, broadcasting.

    The denominator is clamped below at ``eps``, matching
    ``geometry.ucm.radius_from_theta``.  The clamp is what keeps this usable on
    raw predictions: when ``xi + cos(theta)`` goes to zero or negative -- which
    a wrong ``xi`` easily does -- the result saturates at a large finite radius
    instead of producing inf, NaN or a sign flip, and the gradient stays finite
    so the optimiser can still pull the prediction back.
    """
    denom = torch.clamp(xi + torch.cos(theta), min=eps)
    return f * torch.sin(theta) / denom


def theta_grid(theta_ref: Tensor, n_theta: int) -> Tensor:
    """``(B, n_theta)`` of angles from 0 to each sample's ``theta_ref``."""
    if n_theta < 2:
        raise ValueError(f"n_theta must be >= 2, got {n_theta}")
    theta_ref = torch.as_tensor(theta_ref)
    steps = torch.linspace(
        0.0, 1.0, n_theta, device=theta_ref.device, dtype=theta_ref.dtype
    )
    return theta_ref.reshape(-1, 1) * steps.reshape(1, -1)


def is_valid_theta(theta: Tensor, xi: Tensor, eps: float = DEFAULT_EPS) -> Tensor:
    """Where the projection is single-valued for these parameters (SPEC.md 4.8).

    Invalid when ``xi + cos(theta) <= eps``, or when ``xi > 1`` and
    ``cos(theta) < -1/xi`` (past the fold, where two directions share a pixel).

    Both tests are evaluated arithmetically, so ``xi <= 0`` needs no special
    case: it fails the first test exactly where it should, and the second test
    is gated off by ``xi > 1``.
    """
    cos_theta = torch.cos(theta)
    valid = (xi + cos_theta) > eps
    folded = (xi > 1.0) & (cos_theta < -1.0 / torch.clamp(xi, min=1.0 + eps))
    return valid & ~folded


def _stats(norm_stats: Mapping[str, Any], like: Tensor) -> tuple[Tensor, Tensor]:
    mean = _as_tensor(norm_stats["mean"], like).reshape(-1)
    std = _as_tensor(norm_stats["std"], like).reshape(-1)
    if mean.numel() != 2 or std.numel() != 2:
        raise ValueError("norm_stats mean/std must each have two entries")
    return mean, std


def denormalize(pred: Tensor, norm_stats: Mapping[str, Any]) -> tuple[Tensor, Tensor]:
    """``(B, 2)`` standardised output to ``(f, xi)`` (SPEC.md 3.12).

    The first channel is ``log f``, so ``f = exp(...)`` is positive whatever
    the network predicts.  ``xi`` is returned exactly as predicted -- it is not
    clamped, because the losses must see how wrong it is in order to correct
    it, and :func:`is_valid_theta` handles the out-of-range case.
    """
    if pred.ndim != 2 or pred.shape[-1] != 2:
        raise ValueError(f"pred must be (B, 2), got {tuple(pred.shape)}")
    mean, std = _stats(norm_stats, pred)
    raw = pred * std + mean
    return torch.exp(raw[:, 0]), raw[:, 1]


def normalize(f: Tensor, xi: Tensor, norm_stats: Mapping[str, Any]) -> Tensor:
    """Inverse of :func:`denormalize`; ``(f, xi)`` to a ``(B, 2)`` target."""
    mean, std = _stats(norm_stats, f if isinstance(f, Tensor) else xi)
    raw = torch.stack([torch.log(f), xi], dim=-1)
    return (raw - mean) / std


def reprojection_error_px(
    f_pred: Tensor,
    xi_pred: Tensor,
    f_gt: Tensor,
    xi_gt: Tensor,
    theta_ref: Tensor,
    n_theta: int = 64,
    r_cap: float = 224.0,
    eps: float = DEFAULT_EPS,
) -> tuple[Tensor, Tensor]:
    """Mean radial error in pixels, and the invalid fraction (SPEC.md 4.8).

    Angles where the *predicted* parameters are invalid are charged ``r_cap``
    rather than a meaningless radius, and their share is reported separately so
    a model that escapes the valid region cannot hide behind a small average.

    Valid angles are capped at ``r_cap`` as well.  Validity only requires the
    denominator to clear ``eps``, and a denominator of 1e-6 produces a radius
    in the millions, which would swamp the batch mean with one near-degenerate
    sample.  ``r_cap`` is an image diagonal: being wrong by more than that is
    simply "wrong", and the size of the number carries no further information.
    """
    theta = theta_grid(theta_ref, n_theta)
    xi_pred_b = xi_pred.reshape(-1, 1)
    xi_gt_b = xi_gt.reshape(-1, 1)

    r_pred = radius_from_theta(theta, f_pred.reshape(-1, 1), xi_pred_b, eps)
    r_gt = radius_from_theta(theta, f_gt.reshape(-1, 1), xi_gt_b, eps)

    valid = is_valid_theta(theta, xi_pred_b, eps)
    capped = torch.clamp(torch.abs(r_pred - r_gt), max=float(r_cap))
    error = torch.where(valid, capped, torch.full_like(r_pred, float(r_cap)))
    return error.mean(dim=1), (~valid).to(error.dtype).mean(dim=1)


def validity_terms(
    theta: Tensor, xi: Tensor, margin: float = DEFAULT_MARGIN
) -> tuple[Tensor, Tensor, Tensor]:
    """The two quantities that have to stay positive, and the resulting mask.

    ``d1 = xi + cos(theta)`` is the projection denominator: it must be positive
    for the point to be in front of the mirror at all.  ``d2 = 1 + xi*cos(theta)``
    is the numerator of ``dr/dtheta``: it must be positive for the radius to
    still be increasing, which is what keeps the projection single-valued.

    Together they are exactly SPEC.md 4.8's two conditions -- for ``xi > 1``,
    ``d2 > 0`` is the same as ``cos(theta) > -1/xi`` -- but written so that both
    are smooth functions of ``xi`` that a barrier can push on.
    """
    cos_theta = torch.cos(theta)
    d1 = xi + cos_theta
    d2 = 1.0 + xi * cos_theta
    return d1, d2, (d1 > margin) & (d2 > margin)


def geometric_loss(
    pred: Tensor,
    target_f: Tensor,
    target_xi: Tensor,
    theta_ref: Tensor,
    norm_stats: Mapping[str, Any],
    n_theta: int = 32,
    beta: float = 1.0,
    size: int = 224,
    eps: float = DEFAULT_EPS,
    margin: float = DEFAULT_MARGIN,
    barrier_weight: float = DEFAULT_BARRIER_WEIGHT,
) -> Tensor:
    """Differentiable reprojection loss: masked radial term plus a barrier.

    ``loss = radial + barrier_weight * barrier`` (SPEC.md 4.8), where

    ``radial``
        Smooth L1 (``beta = 1 px``) between the predicted and ground-truth
        radial profiles, averaged over the **valid** angles only and divided by
        ``size`` so the value sits around O(1).  Where the prediction is
        invalid the denominator is replaced by 1 rather than clamped, purely to
        keep the arithmetic finite; those entries are then masked out, so they
        contribute neither value nor gradient.  With no valid angle at all the
        term is 0.

    ``barrier``
        ``mean(relu(margin - d1) + relu(margin - d2))`` over *all* angles.  It
        is zero wherever the prediction is comfortably valid and grows linearly
        once either quantity falls below ``margin``.

    The split exists because the radial term alone cannot recover a runaway
    prediction.  Masking removes its gradient there, and clamping the
    denominator instead would be no better: a clamp is flat, so its gradient is
    zero exactly where a correcting signal is needed.  The hinge is the part
    with a non-zero slope pointing back into the valid region -- for a too-small
    ``xi``, ``d(relu(margin - d1))/d(xi) = -1``, so gradient descent raises it.
    """
    f_pred, xi_pred = denormalize(pred, norm_stats)
    theta = theta_grid(theta_ref, n_theta)

    xi_b = xi_pred.reshape(-1, 1)
    d1, d2, valid = validity_terms(theta, xi_b, margin)

    # Substitute 1 (not a clamp) so the invalid entries stay finite; the mask
    # below is what actually removes them.
    safe_d1 = torch.where(valid, d1, torch.ones_like(d1))
    r_pred = f_pred.reshape(-1, 1) * torch.sin(theta) / safe_d1
    r_gt = radius_from_theta(
        theta,
        _as_tensor(target_f, pred).reshape(-1, 1),
        _as_tensor(target_xi, pred).reshape(-1, 1),
        eps,
    )

    per_angle = torch.nn.functional.smooth_l1_loss(
        r_pred, r_gt, beta=float(beta), reduction="none"
    )
    weight = valid.to(per_angle.dtype)
    n_valid = weight.sum()
    radial = (per_angle * weight).sum() / torch.clamp(n_valid, min=1.0) / float(size)

    barrier = (torch.relu(margin - d1) + torch.relu(margin - d2)).mean()
    return radial + float(barrier_weight) * barrier


def param_l1_loss(pred: Tensor, target: Tensor) -> Tensor:
    """L1 in the standardised space (SPEC.md 3.12)."""
    if pred.shape != target.shape:
        raise ValueError(
            f"pred {tuple(pred.shape)} and target {tuple(target.shape)} must match"
        )
    return torch.nn.functional.l1_loss(pred, target, reduction="mean")
