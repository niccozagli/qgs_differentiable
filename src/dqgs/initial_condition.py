"""Initial-condition helpers for qgs reference runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_SPINUP_PATH = Path("configs/ic/de_cruz_active_ic.npy")


@dataclass(frozen=True)
class InitialCondition:
    """State vector plus provenance metadata."""

    values: np.ndarray
    metadata: dict[str, object]


def load_spinup_initial_condition(
    path: str | Path = DEFAULT_SPINUP_PATH,
    *,
    expected_dim: int | None = None,
) -> InitialCondition:
    """Load a spinup state from a NumPy binary file."""

    ic_path = Path(path)
    values = np.load(ic_path)
    values = _validate_state_vector(values, expected_dim=expected_dim)
    return InitialCondition(
        values=values,
        metadata={
            "kind": "spinup_file",
            "path": str(ic_path),
            "shape": list(values.shape),
            "dtype": str(values.dtype),
        },
    )


def random_initial_condition(
    ndim: int,
    *,
    seed: int = 21217,
    scale: float = 0.01,
) -> InitialCondition:
    """Create the small random initial condition used by upstream qgs examples."""

    rng = np.random.RandomState(seed)
    values = rng.rand(ndim).astype(np.float64) * scale
    return InitialCondition(
        values=values,
        metadata={
            "kind": "random",
            "seed": seed,
            "scale": scale,
            "shape": list(values.shape),
            "dtype": str(values.dtype),
        },
    )


def initial_condition_from_source(
    source: str,
    *,
    ndim: int,
    path: str | Path = DEFAULT_SPINUP_PATH,
    seed: int = 21217,
    random_scale: float = 0.01,
) -> InitialCondition:
    """Resolve a named initial-condition source."""

    if source == "spinup":
        return load_spinup_initial_condition(path, expected_dim=ndim)
    if source == "random":
        return random_initial_condition(ndim, seed=seed, scale=random_scale)
    raise ValueError(f"Unknown initial-condition source: {source!r}")


def _validate_state_vector(
    values: np.ndarray,
    *,
    expected_dim: int | None,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"Initial condition must be one-dimensional, got shape {values.shape}")
    if expected_dim is not None and values.shape != (expected_dim,):
        raise ValueError(
            f"Initial condition has shape {values.shape}; expected ({expected_dim},)"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("Initial condition contains non-finite values")
    return values
