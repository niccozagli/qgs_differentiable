"""Simple autodiff parameter estimation helpers.

This module keeps parameter recovery small and explicit: observations are a
fixed trajectory window, ``obs[0]`` is the fixed initial condition, and only the
selected physical parameters are optimized.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax


@dataclass(frozen=True)
class FitResult:
    """Result returned by :func:`fit_parameters`."""

    theta: jax.Array
    phi: jax.Array
    loss_history: np.ndarray
    theta_history: np.ndarray
    grad_norm_history: np.ndarray
    predicted: jax.Array


def theta_to_phi(theta, theta_ref):
    """Convert positive physical parameters to log-relative optimizer values."""
    return jnp.log(jnp.asarray(theta) / jnp.asarray(theta_ref))


def phi_to_theta(phi, theta_ref):
    """Convert log-relative optimizer values back to physical parameters."""
    return jnp.asarray(theta_ref) * jnp.exp(jnp.asarray(phi))


def _rk4_step_theta(f_theta, U, theta, dt):
    """One RK4 step for a tendency with explicit parameter input."""
    k1 = f_theta(U, theta)
    k2 = f_theta(U + 0.5 * dt * k1, theta)
    k3 = f_theta(U + 0.5 * dt * k2, theta)
    k4 = f_theta(U + dt * k3, theta)
    return U + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@partial(jax.jit, static_argnums=(0, 4, 5))
def simulate(f_theta, theta, U0, dt, n_steps, write_steps=1):
    """Integrate ``dU/dt = f_theta(U, theta)`` and return sampled states.

    Args:
        f_theta: tendency returned by ``dqgs.make_tendency``.
        theta: free physical parameter vector.
        U0: fixed initial state.
        dt: RK4 timestep.
        n_steps: total number of RK4 steps.
        write_steps: record every ``write_steps`` steps.

    Returns:
        Trajectory with row 0 equal to ``U0``.
    """
    if n_steps % write_steps != 0:
        raise ValueError(
            f"n_steps ({n_steps}) must be a multiple of write_steps "
            f"({write_steps}) so the sample grid is uniform."
        )
    n_records = n_steps // write_steps

    def advance(U, _):
        U = jax.lax.fori_loop(
            0,
            write_steps,
            lambda _, u: _rk4_step_theta(f_theta, u, theta, dt),
            U,
        )
        return U, U

    _, sampled = jax.lax.scan(advance, U0, xs=None, length=n_records)
    return jnp.concatenate([U0[None, :], sampled], axis=0)


@partial(jax.jit, static_argnums=(2, 5, 6))
def trajectory_loss(phi, theta_ref, f_theta, obs, dt, n_steps, write_steps=1):
    """Mean squared future-trajectory error for fixed observations.

    ``obs[0]`` is used as the fixed initial condition. The first row is excluded
    from the loss because the predicted and observed trajectories start from the
    same state by construction.
    """
    theta = phi_to_theta(phi, theta_ref)
    pred = simulate(f_theta, theta, obs[0], dt, n_steps, write_steps)
    return jnp.mean((pred[1:] - obs[1:]) ** 2)


def fit_parameters(
    *,
    f_theta,
    theta0,
    theta_ref,
    obs,
    dt,
    n_steps,
    write_steps=1,
    learning_rate=1e-2,
    num_iterations=300,
    progress_every=None,
    parameter_names=None,
):
    """Recover free parameters by minimizing trajectory mismatch with Adam.

    Args:
        progress_every: if set, print loss / gradient norm / current parameters
            every ``progress_every`` iterations, plus the final iteration.
        parameter_names: optional labels used only for progress printing.
    """
    theta_ref = jnp.asarray(theta_ref)
    obs = jnp.asarray(obs)
    phi = theta_to_phi(theta0, theta_ref)

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(phi)

    def loss_for_phi(current_phi):
        return trajectory_loss(
            current_phi, theta_ref, f_theta, obs, dt, n_steps, write_steps
        )

    # Forward-mode differentiation: with only 1-3 free parameters and a scalar
    # loss, jacfwd propagates a handful of tangents through the RK4 scan, instead
    # of compiling a backward pass through it. Reverse-mode (jax.value_and_grad)
    # does not finish compiling that backward graph for the 36-D symbolic
    # tendency. Jitted so all iterations reuse one compiled program.
    value_and_grad = jax.jit(
        lambda p: (loss_for_phi(p), jax.jacfwd(loss_for_phi)(p))
    )
    loss_history = []
    theta_history = []
    grad_norm_history = []

    if parameter_names is None:
        parameter_names = tuple(f"theta[{i}]" for i in range(theta_ref.shape[0]))

    for iteration in range(num_iterations):
        value, grad = value_and_grad(phi)
        updates, opt_state = optimizer.update(grad, opt_state)
        phi = optax.apply_updates(phi, updates)
        theta = phi_to_theta(phi, theta_ref)
        grad_norm = jnp.linalg.norm(grad)

        loss_history.append(float(value))
        theta_history.append(np.asarray(theta))
        grad_norm_history.append(float(grad_norm))

        if progress_every and (
            iteration == 0
            or (iteration + 1) % progress_every == 0
            or iteration + 1 == num_iterations
        ):
            pieces = [
                f"{name}={float(param):.8g}"
                for name, param in zip(parameter_names, np.asarray(theta), strict=True)
            ]
            print(
                f"iter {iteration + 1:5d}/{num_iterations}: "
                f"loss={float(value):.6e}, "
                f"grad_norm={float(grad_norm):.6e}, "
                + ", ".join(pieces)
            )

    theta = phi_to_theta(phi, theta_ref)
    predicted = simulate(f_theta, theta, obs[0], dt, n_steps, write_steps)
    return FitResult(
        theta=theta,
        phi=phi,
        loss_history=np.asarray(loss_history, dtype=np.float64),
        theta_history=np.asarray(theta_history, dtype=np.float64),
        grad_norm_history=np.asarray(grad_norm_history, dtype=np.float64),
        predicted=predicted,
    )
