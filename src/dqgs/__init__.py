"""Differentiable JAX implementation of a fixed qgs MAOOAM configuration."""

from dqgs.estimate import (
    FitResult,
    fit_parameters,
    phi_to_theta,
    simulate,
    theta_to_phi,
    trajectory_loss,
)
from dqgs.integrate import integrate, rk4_step
from dqgs.tendency_py2jax import (
    DEFAULT_PARAMS,
    DEFAULT_PARAM_VECTOR,
    PARAMETER_NAMES,
    STATE_DIM,
    make_tendency,
    param_indices,
    require_x64,
    tendency,
)

__all__ = [
    "DEFAULT_PARAMS",
    "DEFAULT_PARAM_VECTOR",
    "FitResult",
    "PARAMETER_NAMES",
    "STATE_DIM",
    "fit_parameters",
    "integrate",
    "make_tendency",
    "param_indices",
    "phi_to_theta",
    "require_x64",
    "rk4_step",
    "simulate",
    "theta_to_phi",
    "tendency",
    "trajectory_loss",
]
