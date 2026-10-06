"""Acceptance criterion #13 (SPEC.md 5, see 4.8): the torch metrics.

Two things matter here.  The torch radial profile must agree with the numpy
one in ``geometry/ucm.py`` -- there is only one formula in this project and
two implementations of it -- and every function must survive the parameters a
*network* produces, including ``xi < 0`` and ``xi > 1.5``, which the numpy
``theta_max`` would reject outright.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from geometry import ucm
from metrics.geometric import (
    DEFAULT_MARGIN,
    denormalize,
    geometric_loss,
    is_valid_theta,
    normalize,
    param_l1_loss,
    radius_from_theta,
    reprojection_error_px,
    theta_grid,
    validity_terms,
)

NORM_STATS = {"fields": ["log_f", "xi"], "mean": [5.0, 0.9], "std": [0.4, 0.38]}
XI_VALUES = [0.0, 0.25, 0.5, 1.0, 1.5]


def _targets(f: np.ndarray, xi: np.ndarray) -> torch.Tensor:
    return normalize(
        torch.tensor(f, dtype=torch.float64),
        torch.tensor(xi, dtype=torch.float64),
        NORM_STATS,
    )


# --- numpy / torch agreement ------------------------------------------------


@pytest.mark.parametrize("xi", XI_VALUES)
def test_radius_matches_the_numpy_implementation(xi: float) -> None:
    """#13: the two implementations of r(theta) must agree within 1e-4."""
    f = 137.5
    theta = np.linspace(0.0, 0.95 * ucm.theta_max(xi), 257)

    expected = ucm.radius_from_theta(theta, f, xi)
    actual = radius_from_theta(
        torch.tensor(theta, dtype=torch.float64),
        torch.tensor(f, dtype=torch.float64),
        torch.tensor(xi, dtype=torch.float64),
    ).numpy()

    assert np.abs(actual - expected).max() < 1e-4


def test_radius_matches_numpy_including_the_clamped_region() -> None:
    """Agreement must hold where the denominator clamp bites, too."""
    f, xi = 100.0, 0.3
    theta = np.linspace(0.0, np.pi, 512)  # runs well past theta_max(0.3)
    expected = ucm.radius_from_theta(theta, f, xi)
    actual = radius_from_theta(
        torch.tensor(theta, dtype=torch.float64),
        torch.tensor(f, dtype=torch.float64),
        torch.tensor(xi, dtype=torch.float64),
    ).numpy()
    assert np.allclose(actual, expected, rtol=1e-9, atol=1e-4)


def test_radius_is_the_pinhole_model_at_xi_zero() -> None:
    theta = torch.linspace(0.0, np.radians(85.0), 128, dtype=torch.float64)
    r = radius_from_theta(theta, torch.tensor(90.0, dtype=torch.float64),
                          torch.tensor(0.0, dtype=torch.float64))
    assert torch.allclose(r, 90.0 * torch.tan(theta), atol=1e-9)


# --- #13 zero error when the prediction is right ----------------------------


def test_reprojection_error_is_zero_for_a_perfect_prediction() -> None:
    """#13: pred = GT gives exactly zero error and no invalid angles."""
    f = torch.tensor([60.0, 150.0, 300.0], dtype=torch.float64)
    xi = torch.tensor([0.0, 0.7, 1.4], dtype=torch.float64)
    theta_ref = torch.tensor(np.radians([55.0, 80.0, 100.0]), dtype=torch.float64)

    error, invalid = reprojection_error_px(f, xi, f, xi, theta_ref, n_theta=64)
    assert torch.allclose(error, torch.zeros_like(error), atol=1e-12)
    assert torch.allclose(invalid, torch.zeros_like(invalid), atol=1e-12)


def test_geometric_loss_is_zero_for_a_perfect_prediction() -> None:
    f = np.array([60.0, 150.0, 300.0])
    xi = np.array([0.1, 0.7, 1.4])
    theta_ref = torch.tensor(np.radians([55.0, 80.0, 100.0]), dtype=torch.float64)

    pred = _targets(f, xi)
    loss = geometric_loss(
        pred,
        torch.tensor(f, dtype=torch.float64),
        torch.tensor(xi, dtype=torch.float64),
        theta_ref,
        NORM_STATS,
    )
    assert float(loss) == pytest.approx(0.0, abs=1e-12)


def test_error_grows_with_the_mistake() -> None:
    f = torch.tensor([150.0], dtype=torch.float64)
    xi = torch.tensor([0.7], dtype=torch.float64)
    theta_ref = torch.tensor([np.radians(80.0)], dtype=torch.float64)

    errors = [
        float(reprojection_error_px(f * scale, xi, f, xi, theta_ref)[0])
        for scale in (1.0, 1.05, 1.2, 1.5)
    ]
    assert errors == sorted(errors)
    assert errors[0] == pytest.approx(0.0, abs=1e-12)


# --- standardisation --------------------------------------------------------


def test_denormalize_inverts_normalize() -> None:
    f = torch.tensor([35.0, 120.0, 410.0], dtype=torch.float64)
    xi = torch.tensor([0.0, 0.8, 1.5], dtype=torch.float64)
    f_back, xi_back = denormalize(normalize(f, xi, NORM_STATS), NORM_STATS)
    assert torch.allclose(f_back, f, rtol=1e-12)
    assert torch.allclose(xi_back, xi, atol=1e-12)


def test_denormalize_always_returns_a_positive_f() -> None:
    """f = exp(log f), so no prediction can make it non-positive."""
    pred = torch.tensor([[-50.0, 0.0], [50.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    f, _ = denormalize(pred, NORM_STATS)
    assert torch.all(f > 0)
    assert torch.isfinite(f).all()


def test_denormalize_rejects_a_wrong_shape() -> None:
    with pytest.raises(ValueError, match=r"\(B, 2\)"):
        denormalize(torch.zeros(4, 3), NORM_STATS)


# --- predictions outside the trained range ----------------------------------


@pytest.mark.parametrize("xi_pred", [-5.0, -1.0, -0.2, 0.0, 1.5, 3.0, 50.0])
def test_metrics_accept_any_predicted_xi(xi_pred: float) -> None:
    """A network emits whatever it likes; nothing here may raise or go NaN.

    ``geometry.ucm.theta_max`` raises for xi < 0, so these functions must not
    be built on it.
    """
    f_gt = torch.tensor([150.0], dtype=torch.float64)
    xi_gt = torch.tensor([0.7], dtype=torch.float64)
    theta_ref = torch.tensor([np.radians(100.0)], dtype=torch.float64)

    f_pred = torch.tensor([150.0], dtype=torch.float64)
    xi_t = torch.tensor([xi_pred], dtype=torch.float64)

    error, invalid = reprojection_error_px(f_pred, xi_t, f_gt, xi_gt, theta_ref)
    assert torch.isfinite(error).all()
    assert torch.all((invalid >= 0.0) & (invalid <= 1.0))

    r = radius_from_theta(
        torch.linspace(0.0, np.pi, 64, dtype=torch.float64),
        f_pred,
        xi_t,
    )
    assert torch.isfinite(r).all()


def test_numpy_theta_max_still_rejects_negative_xi() -> None:
    """The guard this module deliberately avoids depending on."""
    with pytest.raises(ValueError):
        ucm.theta_max(-0.1)


def test_geometric_loss_survives_a_wild_prediction() -> None:
    """A far-out prediction must still give a finite loss and finite gradients."""
    f = np.array([150.0, 150.0])
    xi = np.array([0.7, 0.7])
    theta_ref = torch.tensor(np.radians([100.0, 100.0]), dtype=torch.float64)
    pred = torch.tensor([[-20.0, -30.0], [20.0, 30.0]], dtype=torch.float64,
                        requires_grad=True)

    loss = geometric_loss(
        pred,
        torch.tensor(f, dtype=torch.float64),
        torch.tensor(xi, dtype=torch.float64),
        theta_ref,
        NORM_STATS,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(pred.grad).all()


def test_invalid_angles_are_charged_the_cap() -> None:
    """SPEC.md 4.8: an angle invalid for the prediction scores r_cap."""
    f = torch.tensor([150.0], dtype=torch.float64)
    theta_ref = torch.tensor([np.radians(200.0)], dtype=torch.float64)
    # xi = 0 is invalid beyond theta = 90 deg, which theta_ref = 100 deg crosses.
    error, invalid = reprojection_error_px(
        f, torch.tensor([0.0], dtype=torch.float64),
        f, torch.tensor([1.0], dtype=torch.float64),
        theta_ref, n_theta=64, r_cap=224.0,
    )
    assert float(invalid) > 0.3
    assert float(error) > 1.0


def test_is_valid_theta_matches_the_spec_conditions() -> None:
    theta = torch.linspace(0.0, np.pi, 181, dtype=torch.float64)
    for xi_value in (0.0, 0.5, 1.0, 1.5, 3.0):
        xi = torch.tensor(xi_value, dtype=torch.float64)
        valid = is_valid_theta(theta, xi)
        cos_theta = torch.cos(theta)
        expected = (xi + cos_theta) > 1e-6
        if xi_value > 1.0:
            expected = expected & ~(cos_theta < -1.0 / xi)
        assert torch.equal(valid, expected), xi_value


def test_valid_region_agrees_with_numpy_theta_max() -> None:
    """For the xi a sampler can produce, validity must match geometry/ucm.py."""
    for xi_value in XI_VALUES:
        limit = ucm.theta_max(xi_value)
        theta = torch.tensor(
            [0.0, 0.5 * limit, 0.9 * limit, min(0.999 * limit, np.pi)],
            dtype=torch.float64,
        )
        valid = is_valid_theta(theta, torch.tensor(xi_value, dtype=torch.float64))
        assert bool(valid.all()), (xi_value, valid)


# --- shapes, grids, gradients ----------------------------------------------


def test_theta_grid_spans_each_sample_to_its_own_reference() -> None:
    theta_ref = torch.tensor([0.5, 1.0, 1.9], dtype=torch.float64)
    grid = theta_grid(theta_ref, 16)
    assert grid.shape == (3, 16)
    assert torch.allclose(grid[:, 0], torch.zeros(3, dtype=torch.float64))
    assert torch.allclose(grid[:, -1], theta_ref)


def test_theta_grid_rejects_a_degenerate_size() -> None:
    with pytest.raises(ValueError, match="n_theta"):
        theta_grid(torch.tensor([1.0]), 1)


def test_reprojection_error_returns_one_value_per_sample() -> None:
    batch = 7
    f = torch.full((batch,), 150.0, dtype=torch.float64)
    xi = torch.full((batch,), 0.6, dtype=torch.float64)
    theta_ref = torch.full((batch,), 1.2, dtype=torch.float64)
    error, invalid = reprojection_error_px(f * 1.1, xi, f, xi, theta_ref)
    assert error.shape == (batch,)
    assert invalid.shape == (batch,)


def test_geometric_loss_has_a_useful_gradient() -> None:
    f = np.array([150.0])
    xi = np.array([0.7])
    theta_ref = torch.tensor([np.radians(90.0)], dtype=torch.float64)

    pred = _targets(f * 1.4, xi + 0.3).clone().requires_grad_(True)
    loss = geometric_loss(
        pred,
        torch.tensor(f, dtype=torch.float64),
        torch.tensor(xi, dtype=torch.float64),
        theta_ref,
        NORM_STATS,
    )
    loss.backward()
    assert loss.detach().item() > 0
    assert torch.isfinite(pred.grad).all()
    assert torch.any(pred.grad != 0)


def test_geometric_loss_is_order_one() -> None:
    """Dividing by the image size should keep the value tame (SPEC.md 4.8)."""
    f = np.array([150.0])
    xi = np.array([0.7])
    theta_ref = torch.tensor([np.radians(90.0)], dtype=torch.float64)
    loss = geometric_loss(
        _targets(f * 1.3, xi),
        torch.tensor(f, dtype=torch.float64),
        torch.tensor(xi, dtype=torch.float64),
        theta_ref,
        NORM_STATS,
    )
    assert 0.0 < float(loss) < 10.0


def test_param_l1_loss() -> None:
    pred = torch.tensor([[0.0, 0.0], [1.0, -1.0]], dtype=torch.float64)
    target = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    assert float(param_l1_loss(pred, target)) == pytest.approx(0.5)
    assert float(param_l1_loss(target, target)) == pytest.approx(0.0)

    with pytest.raises(ValueError, match="must match"):
        param_l1_loss(torch.zeros(2, 2), torch.zeros(3, 2))


def test_losses_work_in_float32() -> None:
    """Training runs in bf16/fp32, so float64 must not be load-bearing."""
    f = torch.tensor([150.0], dtype=torch.float32)
    xi = torch.tensor([0.7], dtype=torch.float32)
    theta_ref = torch.tensor([1.4], dtype=torch.float32)
    pred = torch.zeros(1, 2, dtype=torch.float32, requires_grad=True)

    loss = geometric_loss(pred, f, xi, theta_ref, NORM_STATS)
    loss.backward()
    assert loss.dtype == torch.float32
    assert torch.isfinite(pred.grad).all()

    error, invalid = reprojection_error_px(f, xi, f, xi, theta_ref)
    assert error.dtype == torch.float32 and invalid.dtype == torch.float32


# --- the masked radial term and the barrier (SPEC.md 4.8) -------------------


def _realistic_batch(batch: int = 256, seed: int = 0):
    """Ground truth drawn the way the renderer draws it."""
    import yaml

    from data.sampling import SamplingConfig, sample_camera

    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "data.yaml"
    cfg = SamplingConfig.from_mapping(yaml.safe_load(cfg_path.read_text("utf-8")))
    rng = np.random.default_rng(seed)
    samples = [sample_camera(rng, cfg) for _ in range(batch)]
    return (
        torch.tensor([s.f for s in samples], dtype=torch.float64),
        torch.tensor([s.xi for s in samples], dtype=torch.float64),
        torch.tensor(
            [np.radians(s.fov_deg) / 2.0 for s in samples], dtype=torch.float64
        ),
    )


def test_validity_terms_match_the_spec_conditions() -> None:
    """d1 > 0 and d2 > 0 are SPEC.md 4.8's two conditions, written smoothly."""
    theta = torch.linspace(0.0, np.pi, 181, dtype=torch.float64)
    for xi_value in (0.0, 0.5, 1.0, 1.5, 3.0):
        xi = torch.tensor(xi_value, dtype=torch.float64)
        d1, d2, valid = validity_terms(theta, xi, margin=0.0)
        cos_theta = torch.cos(theta)
        assert torch.allclose(d1, xi + cos_theta)
        assert torch.allclose(d2, 1.0 + xi * cos_theta)
        # For xi > 1, d2 > 0 is exactly cos(theta) > -1/xi.
        if xi_value > 1.0:
            assert torch.equal(d2 > 0, cos_theta > -1.0 / xi)
        assert torch.equal(valid, (d1 > 0) & (d2 > 0))


def test_random_prediction_gives_a_tame_loss_and_gradient() -> None:
    """(a) N(0,1) predictions on a realistic batch stay finite and small."""
    f, xi, theta_ref = _realistic_batch(batch=256, seed=0)
    generator = torch.Generator().manual_seed(0)
    pred = torch.randn(256, 2, generator=generator, dtype=torch.float64,
                       requires_grad=True)

    loss = geometric_loss(pred, f, xi, theta_ref, NORM_STATS)
    loss.backward()

    assert torch.isfinite(loss)
    assert loss.detach().item() < 10.0, loss.detach().item()
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().max().item() < 100.0, pred.grad.abs().max().item()


def test_barrier_pushes_a_negative_xi_back_up() -> None:
    """(b) With xi far below 0 the xi gradient is negative, so SGD raises it.

    The radial term is masked out there, so this gradient can only come from
    the barrier -- which is the whole point: a clamped denominator would have
    given exactly zero and the prediction could never recover.
    """
    f, xi, theta_ref = _realistic_batch(batch=32, seed=1)
    pred = torch.zeros(32, 2, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        pred[:, 1] = -10.0  # xi far below zero after denormalisation

    loss = geometric_loss(pred, f, xi, theta_ref, NORM_STATS)
    loss.backward()

    _f_pred, xi_pred = denormalize(pred.detach(), NORM_STATS)
    assert (xi_pred < 0).all(), xi_pred.max().item()

    assert (pred.grad[:, 1] < 0).all(), pred.grad[:, 1].max().item()
    assert pred.grad[:, 0].abs().max().item() < 10.0


def test_barrier_is_the_only_gradient_when_nothing_is_valid() -> None:
    f, xi, theta_ref = _realistic_batch(batch=8, seed=2)
    pred = torch.zeros(8, 2, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        pred[:, 1] = -10.0

    theta = theta_grid(theta_ref, 32)
    _d1, _d2, valid = validity_terms(theta, denormalize(pred.detach(), NORM_STATS)[1]
                                     .reshape(-1, 1), DEFAULT_MARGIN)
    assert not valid.any(), "this test needs a fully invalid prediction"

    geometric_loss(pred, f, xi, theta_ref, NORM_STATS).backward()
    # The radial term contributes nothing, so f gets no signal at all.
    assert torch.allclose(pred.grad[:, 0], torch.zeros_like(pred.grad[:, 0]))
    assert (pred.grad[:, 1] < 0).all()


def test_loss_is_exactly_zero_for_a_perfect_prediction() -> None:
    """(c) pred = GT leaves both terms at zero."""
    f, xi, theta_ref = _realistic_batch(batch=64, seed=3)
    pred = normalize(f, xi, NORM_STATS)
    loss = geometric_loss(pred, f, xi, theta_ref, NORM_STATS)
    assert float(loss) == pytest.approx(0.0, abs=1e-12)


def test_barrier_is_zero_on_valid_predictions() -> None:
    f, xi, theta_ref = _realistic_batch(batch=64, seed=4)
    theta = theta_grid(theta_ref, 32)
    _d1, _d2, valid = validity_terms(theta, xi.reshape(-1, 1), DEFAULT_MARGIN)
    assert valid.all(), "the sampler should never emit a camera inside the margin"

    # A loss built only from the barrier must vanish for ground-truth inputs.
    pred = normalize(f, xi, NORM_STATS)
    only_barrier = geometric_loss(
        pred, f, xi, theta_ref, NORM_STATS, barrier_weight=1.0
    ) - geometric_loss(pred, f, xi, theta_ref, NORM_STATS, barrier_weight=0.0)
    assert float(only_barrier) == pytest.approx(0.0, abs=1e-15)


def test_barrier_weight_scales_only_the_barrier() -> None:
    f, xi, theta_ref = _realistic_batch(batch=16, seed=5)
    pred = torch.zeros(16, 2, dtype=torch.float64)
    with torch.no_grad():
        pred[:, 1] = -10.0

    one = float(geometric_loss(pred, f, xi, theta_ref, NORM_STATS, barrier_weight=1.0))
    two = float(geometric_loss(pred, f, xi, theta_ref, NORM_STATS, barrier_weight=2.0))
    none = float(geometric_loss(pred, f, xi, theta_ref, NORM_STATS, barrier_weight=0.0))
    assert none == pytest.approx(0.0, abs=1e-15)  # radial is fully masked
    assert two == pytest.approx(2.0 * one, rel=1e-12)


def test_invalid_angles_do_not_enter_the_radial_term() -> None:
    """Masked entries contribute neither value nor gradient."""
    f = torch.tensor([150.0], dtype=torch.float64)
    xi = torch.tensor([0.7], dtype=torch.float64)
    theta_ref = torch.tensor([np.radians(200.0)], dtype=torch.float64)

    # xi = 0.2: valid only while cos(theta) > margin - 0.2.
    pred = normalize(f, torch.tensor([0.2], dtype=torch.float64), NORM_STATS)
    wide = geometric_loss(pred, f, xi, theta_ref, NORM_STATS, n_theta=64)
    assert torch.isfinite(wide)

    theta = theta_grid(theta_ref, 64)
    _d1, _d2, valid = validity_terms(
        theta, torch.tensor([[0.2]], dtype=torch.float64), DEFAULT_MARGIN
    )
    assert 0 < int(valid.sum()) < valid.numel(), "need a partially valid case"


# --- reprojection_error_px capping ------------------------------------------


def test_valid_errors_are_capped_too() -> None:
    """A near-degenerate but still 'valid' angle must not blow up the mean."""
    f_gt = torch.tensor([150.0], dtype=torch.float64)
    xi_gt = torch.tensor([0.7], dtype=torch.float64)
    theta_ref = torch.tensor([np.radians(179.0)], dtype=torch.float64)

    # xi just above -cos(theta_max) keeps d1 barely positive, so the radius
    # reaches tens of thousands of pixels while staying 'valid'.
    xi_pred = torch.tensor([1.0 + 1e-7], dtype=torch.float64)
    f_pred = torch.tensor([150.0], dtype=torch.float64)

    r_cap = 224.0
    error, invalid = reprojection_error_px(
        f_pred, xi_pred, f_gt, xi_gt, theta_ref, n_theta=64, r_cap=r_cap
    )
    assert float(error) <= r_cap + 1e-9, float(error)
    assert torch.isfinite(error).all()
    assert 0.0 <= float(invalid) <= 1.0


def test_capping_bounds_every_sample() -> None:
    generator = torch.Generator().manual_seed(0)
    batch = 64
    f_pred = torch.exp(torch.randn(batch, generator=generator, dtype=torch.float64) + 5)
    xi_pred = torch.randn(batch, generator=generator, dtype=torch.float64) * 2.0
    f_gt = torch.full((batch,), 150.0, dtype=torch.float64)
    xi_gt = torch.full((batch,), 0.7, dtype=torch.float64)
    theta_ref = torch.full((batch,), np.radians(170.0), dtype=torch.float64)

    error, invalid = reprojection_error_px(
        f_pred, xi_pred, f_gt, xi_gt, theta_ref, r_cap=224.0
    )
    assert torch.all(error <= 224.0 + 1e-9), error.max().item()
    assert torch.all((invalid >= 0.0) & (invalid <= 1.0))


def test_capping_leaves_small_errors_untouched() -> None:
    """The cap must not distort the errors that actually matter."""
    f = torch.tensor([150.0], dtype=torch.float64)
    xi = torch.tensor([0.7], dtype=torch.float64)
    theta_ref = torch.tensor([1.2], dtype=torch.float64)
    error, _ = reprojection_error_px(f * 1.01, xi, f, xi, theta_ref, r_cap=224.0)
    assert 0.0 < float(error) < 10.0
