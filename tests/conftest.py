"""Shared pytest fixtures for the dqgs validation suite.

Enables JAX float64 at collection time (before any test creates a JAX array) and
provides the qgs reference model / tendency and the shared initial condition.

Run in the ``dqgs`` conda env, which has both jax and qgs:
``~/miniconda3/envs/dqgs/bin/python -m pytest``.
"""

from pathlib import Path

import jax

# MAOOAM accuracy requires float64; must run before any jax array is created.
jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
IC_PATH = REPO_ROOT / "configs" / "ic" / "de_cruz_active_ic.npy"


@pytest.fixture(scope="session")
def ic():
    """On-attractor initial condition (de Cruz active attractor), shape (36,)."""
    return np.load(IC_PATH)


@pytest.fixture(scope="session")
def qgs_model():
    """qgs ``QgParams`` built exactly as in the symbolic-export config script."""
    from qgs.params.params import QgParams

    mp = QgParams()
    mp.set_atmospheric_channel_fourier_modes(2, 2)
    mp.set_oceanic_basin_fourier_modes(2, 4)
    mp.set_params({'kd': 0.0290, 'kdp': 0.0290, 'n': 1.5, 'r': 1.e-7,
                   'h': 136.5, 'd': 1.1e-7})
    mp.atemperature_params.set_params({'eps': 0.7, 'T0': 289.3, 'hlambda': 15.06})
    mp.gotemperature_params.set_params({'gamma': 5.6e8, 'T0': 301.46})
    mp.atemperature_params.set_insolation(103.3333, 0)
    mp.gotemperature_params.set_insolation(310, 0)
    return mp


@pytest.fixture(scope="session")
def f_qgs(qgs_model):
    """qgs reference tendency ``f(t, x) -> dx/dt`` (numba-jitted)."""
    from qgs.functions.tendencies import create_tendencies

    f, _ = create_tendencies(qgs_model)
    return f
