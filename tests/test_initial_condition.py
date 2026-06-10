from pathlib import Path

import numpy as np
import pytest

from dqgs.initial_condition import (
    initial_condition_from_source,
    load_spinup_initial_condition,
    random_initial_condition,
)


def test_load_spinup_initial_condition_validates_dimension() -> None:
    ic = load_spinup_initial_condition(expected_dim=36)

    assert ic.values.shape == (36,)
    assert ic.values.dtype == np.float64
    assert ic.metadata["kind"] == "spinup_file"


def test_load_spinup_initial_condition_rejects_wrong_dimension() -> None:
    with pytest.raises(ValueError, match="expected"):
        load_spinup_initial_condition(expected_dim=35)


def test_random_initial_condition_is_reproducible() -> None:
    first = random_initial_condition(4, seed=7)
    second = random_initial_condition(4, seed=7)

    np.testing.assert_array_equal(first.values, second.values)
    assert first.metadata["kind"] == "random"


def test_initial_condition_from_source_rejects_unknown_source() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        initial_condition_from_source("missing", ndim=36, path=Path("unused.npy"))
