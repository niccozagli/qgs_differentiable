"""Differentiable JAX implementation of a fixed qgs MAOOAM configuration."""

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
    "PARAMETER_NAMES",
    "STATE_DIM",
    "integrate",
    "make_tendency",
    "param_indices",
    "require_x64",
    "rk4_step",
    "tendency",
]
