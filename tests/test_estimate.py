"""Tests for the simple autodiff parameter-estimation helpers."""

import jax
import jax.numpy as jnp
import numpy as np

from dqgs import (
    fit_parameters,
    phi_to_theta,
    simulate,
    theta_to_phi,
    trajectory_loss,
)


def linear_tendency(U, theta):
    """Small stable test system: dU/dt = theta[0] * U."""
    return theta[0] * U


def test_theta_phi_roundtrip():
    theta_ref = jnp.asarray([0.7, 2.0])
    theta = theta_ref * jnp.asarray([1.05, 0.9])

    phi = theta_to_phi(theta, theta_ref)

    np.testing.assert_allclose(phi_to_theta(phi, theta_ref), theta)


def test_trajectory_loss_gradient_matches_centered_difference():
    theta_ref = jnp.asarray([0.7])
    theta0 = theta_ref * 1.05
    U0 = jnp.asarray([1.0, -0.5])
    dt = 0.1
    n_steps = 4

    obs = simulate(linear_tendency, theta_ref, U0, dt, n_steps)
    phi0 = theta_to_phi(theta0, theta_ref)

    def loss(phi):
        return trajectory_loss(phi, theta_ref, linear_tendency, obs, dt, n_steps)

    grad_ad = jax.grad(loss)(phi0)
    eps = 1e-5
    direction = jnp.asarray([eps])
    grad_fd = (loss(phi0 + direction) - loss(phi0 - direction)) / (2.0 * eps)

    np.testing.assert_allclose(grad_ad[0], grad_fd, rtol=1e-6, atol=1e-10)


def test_fit_parameters_moves_wrong_guess_toward_truth():
    theta_ref = jnp.asarray([0.7])
    theta0 = theta_ref * 1.05
    U0 = jnp.asarray([1.0, -0.5])
    dt = 0.1
    n_steps = 8

    obs = simulate(linear_tendency, theta_ref, U0, dt, n_steps)
    initial_loss = trajectory_loss(
        theta_to_phi(theta0, theta_ref),
        theta_ref,
        linear_tendency,
        obs,
        dt,
        n_steps,
    )

    result = fit_parameters(
        f_theta=linear_tendency,
        theta0=theta0,
        theta_ref=theta_ref,
        obs=obs,
        dt=dt,
        n_steps=n_steps,
        learning_rate=5e-2,
        num_iterations=80,
    )
    final_loss = trajectory_loss(
        result.phi, theta_ref, linear_tendency, obs, dt, n_steps
    )

    assert final_loss < initial_loss
    assert abs(result.theta[0] - theta_ref[0]) < abs(theta0[0] - theta_ref[0])
    assert result.loss_history.shape == (80,)
    assert result.theta_history.shape == (80, 1)
    assert result.grad_norm_history.shape == (80,)
    assert np.all(np.isfinite(result.theta_history))
    assert np.all(np.isfinite(result.grad_norm_history))
