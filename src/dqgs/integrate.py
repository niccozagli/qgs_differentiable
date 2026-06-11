"""Pure-JAX RK4 integrator for the differentiable MAOOAM tendency.

Standalone (does **not** import qgs). Operates on a *unary* autonomous tendency
``f(U) -> dU/dt`` so the integrator is decoupled from the parameter signature:
callers pass a closure, e.g. ``lambda U: tendency(U, params)``.

The single step :func:`rk4_step` reproduces the classic RK4 tableau used by qgs
(``external/qgs/qgs/integrators/integrate.py``), which for an autonomous ``f`` is

    k1 = f(U)
    k2 = f(U + dt/2 * k1)
    k3 = f(U + dt/2 * k2)
    k4 = f(U + dt   * k3)
    U+ = U + dt/6 * (k1 + 2 k2 + 2 k3 + k4)

so that, given the same initial condition, timestep and tendency, a JAX
trajectory matches a qgs trajectory to float64 round-off until chaotic
divergence sets in.

No manual threading / multiprocessing: a jitted :func:`integrate` is one compiled
XLA program (XLA parallelizes the linear algebra itself); RK4 is inherently
sequential in time; ensemble parallelism, if ever needed, is ``jax.vmap`` over
the initial condition. Crucially, parameter recovery differentiates through the
whole trajectory, which requires a single traced computation -- multiprocessing
would break autodiff.

Float64 is mandatory (see :func:`dqgs.tendency_py2jax.require_x64`); this module
does not mutate the JAX config, so enable it before use::

    jax.config.update("jax_enable_x64", True)
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp


def rk4_step(f, U, dt):
    """One classic RK4 step of the autonomous system ``dU/dt = f(U)``.

    Args:
        f: unary tendency ``f(U) -> dU/dt``.
        U: state, shape (dim,).
        dt: timestep.

    Returns:
        State advanced by one step of size ``dt``, shape (dim,).
    """
    k1 = f(U)
    k2 = f(U + 0.5 * dt * k1)
    k3 = f(U + 0.5 * dt * k2)
    k4 = f(U + dt * k3)
    return U + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@partial(jax.jit, static_argnums=(0, 3, 4))
def integrate(f, U0, dt, n_steps, write_steps=1):
    """Integrate ``dU/dt = f(U)`` with RK4, sampling every ``write_steps`` steps.

    Mirrors the qgs ``write_steps`` convention: states are recorded at steps
    ``0, write_steps, 2*write_steps, ..., n_steps``. The dropped intermediate
    states are never stacked -- the inner advance carries the state only -- so
    memory scales with the number of *records*, not ``n_steps``.

    Args:
        f: unary tendency ``f(U) -> dU/dt``.
        U0: initial state, shape (dim,).
        dt: timestep.
        n_steps: total number of RK4 steps. Must be a multiple of ``write_steps``.
        write_steps: record every ``write_steps`` steps (default 1).

    Returns:
        Trajectory of shape ``(n_steps // write_steps + 1, dim)`` with row 0
        equal to ``U0`` and the final row equal to the state after ``n_steps``.

    Note:
        ``f``, ``n_steps`` and ``write_steps`` are static (they fix the compiled
        loop structure), so re-tracing happens only when they change.
    """
    if n_steps % write_steps != 0:
        raise ValueError(
            f"n_steps ({n_steps}) must be a multiple of write_steps "
            f"({write_steps}) so the sample grid is uniform."
        )
    n_records = n_steps // write_steps

    def advance(U, _):
        """Advance ``write_steps`` RK4 steps, carrying state only (no stacking)."""
        U = jax.lax.fori_loop(0, write_steps, lambda _, u: rk4_step(f, u, dt), U)
        return U, U

    _, sampled = jax.lax.scan(advance, U0, xs=None, length=n_records)
    # sampled holds states at steps write_steps, 2*write_steps, ..., n_steps;
    # prepend U0 so row k is the state at step k * write_steps.
    return jnp.concatenate([U0[None, :], sampled], axis=0)
