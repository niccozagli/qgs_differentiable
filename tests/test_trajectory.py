"""Trajectory validation: dqgs JAX model vs original qgs.

Distilled from ``notebooks/trajectory_validation.ipynb`` (layer 2 of
``docs/agent/validation.md``). Integrating the same initial condition, with the
same parameters / timestep / RK4 scheme through both codes must agree to float64
round-off on a short window.
"""

import jax.numpy as jnp
import numpy as np
import pytest
import subprocess
import sys

from dqgs import DEFAULT_PARAM_VECTOR, integrate, tendency
from qgs.integrators.integrate import integrate_runge_kutta

DT = 0.1


def f_dqgs(U):
    """Unary autonomous tendency closure at the default parameters."""
    return tendency(U, DEFAULT_PARAM_VECTOR)


def test_tendency_pointwise(ic, f_qgs):
    """Layer 1: the two right-hand sides agree to ~machine precision."""
    rng = np.random.default_rng(0)
    states = [ic] + [ic + 0.02 * rng.standard_normal(ic.shape) for _ in range(5)]

    worst = max(
        np.max(np.abs(np.asarray(tendency(jnp.asarray(U), DEFAULT_PARAM_VECTOR))
                      - f_qgs(0.0, U)))
        for U in states
    )
    assert worst < 1e-12, f"tendency mismatch: {worst:.3e}"


def test_short_window_trajectory(ic, f_qgs):
    """Layer 2: matched IC/dt/RK4 -> trajectory error near round-off."""
    N = 100
    T = N * DT

    _, traj_qgs = integrate_runge_kutta(f_qgs, 0.0, T, DT, ic, write_steps=1)
    traj_qgs = np.asarray(traj_qgs).T                       # (N+1, 36)
    traj_jax = np.asarray(integrate(f_dqgs, jnp.asarray(ic), DT, N))

    assert traj_jax.shape == traj_qgs.shape == (N + 1, 36)
    max_err = np.abs(traj_jax - traj_qgs).max()
    assert max_err < 1e-8, f"short-window trajectory error {max_err:.3e} exceeds 1e-8"


def test_write_steps_consistency(ic, f_qgs):
    """write_steps subsampling matches a full run sliced, and matches qgs."""
    N, k = 100, 10
    T = N * DT

    traj_full = np.asarray(integrate(f_dqgs, jnp.asarray(ic), DT, N))
    traj_sub = np.asarray(integrate(f_dqgs, jnp.asarray(ic), DT, N, write_steps=k))
    assert traj_sub.shape == (N // k + 1, 36)

    # (a) matches the full run sliced every k steps to round-off (the two cases
    # compile to differently-fused XLA kernels, so they differ at ~1e-18, not 0)
    assert np.max(np.abs(traj_sub - traj_full[::k])) < 1e-12

    # (b) matches qgs run with the same write_steps, to round-off
    _, traj_qgs_sub = integrate_runge_kutta(f_qgs, 0.0, T, DT, ic, write_steps=k)
    traj_qgs_sub = np.asarray(traj_qgs_sub).T
    assert np.max(np.abs(traj_sub - traj_qgs_sub)) < 1e-8


def test_integrate_rejects_nonmultiple_write_steps(ic):
    """n_steps must be a multiple of write_steps."""
    with pytest.raises(ValueError):
        integrate(f_dqgs, jnp.asarray(ic), DT, 100, write_steps=7)


def test_default_params_stay_float64_when_x64_enabled_after_import():
    """Importing dqgs before enabling x64 must not freeze defaults as float32."""
    code = """
import jax
import jax.numpy as jnp
import dqgs

assert str(dqgs.DEFAULT_PARAM_VECTOR.dtype) == "float64"
jax.config.update("jax_enable_x64", True)
assert jnp.asarray(dqgs.DEFAULT_PARAM_VECTOR).dtype == jnp.float64
"""
    subprocess.run([sys.executable, "-c", code], check=True)
