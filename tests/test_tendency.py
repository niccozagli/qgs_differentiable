"""Tendency validation tests for the generated JAX transcription."""

import importlib.util
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from numba import njit

from dqgs import DEFAULT_PARAM_VECTOR, tendency

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_generated_tendency(path):
    """Load qgs's generated source, injecting the globals it expects."""
    spec = importlib.util.spec_from_file_location("generated_tendency", path)
    module = importlib.util.module_from_spec(spec)
    module.__dict__.update({"np": np, "njit": njit})
    spec.loader.exec_module(module)
    return module.f


def test_tendency_matches_generated_source_for_perturbed_params(ic):
    """The JAX transcription must match the symbolic export beyond defaults."""
    generated = _load_generated_tendency(
        REPO_ROOT / "configs" / "tendencies_nonfixed_params.py"
    )

    rng = np.random.default_rng(123)
    base_params = np.asarray(DEFAULT_PARAM_VECTOR, dtype=np.float64)
    worst = 0.0

    for _ in range(10):
        state = ic + 0.02 * rng.standard_normal(ic.shape)
        params = base_params * (1.0 + 0.01 * rng.standard_normal(base_params.shape))

        expected = generated(0.0, state, *params)
        got = np.asarray(tendency(jnp.asarray(state), jnp.asarray(params)))
        worst = max(worst, float(np.max(np.abs(got - expected))))

    assert worst < 1e-12, f"generated-source tendency mismatch: {worst:.3e}"
