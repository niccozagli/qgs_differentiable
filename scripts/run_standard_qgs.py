#!/usr/bin/env python
"""Run the standard 36-dimensional qgs MAOOAM configuration and save output."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import sys
import time
from multiprocessing import freeze_support, get_start_method
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
LOCAL_QGS = ROOT / "external" / "qgs"
DEFAULT_OUTPUT_DIR = Path("data")
DEFAULT_OUTPUT_FILENAME = "qgs_standard_run.npz"
if SRC.exists():
    sys.path.insert(0, str(SRC))
if LOCAL_QGS.exists():
    sys.path.insert(0, str(LOCAL_QGS))

from dqgs.initial_condition import DEFAULT_SPINUP_PATH, initial_condition_from_source
from qgs.functions.tendencies import create_tendencies
from qgs.integrators.integrator import RungeKuttaIntegrator
from qgs.params.params import QgParams


STANDARD_MODEL_PARAMETERS = {
    "kd": 0.0290,
    "kdp": 0.0290,
    "n": 1.5,
    "r": 1.0e-7,
    "h": 136.5,
    "d": 1.1e-7,
}
STANDARD_ATEMPERATURE_PARAMETERS = {
    "eps": 0.7,
    "T0": 289.3,
    "hlambda": 15.06,
}
STANDARD_GOTEMPERATURE_PARAMETERS = {
    "gamma": 5.6e8,
    "T0": 301.46,
}
STANDARD_ATEMPERATURE_INSOLATION_VALUE = 103.3333
STANDARD_ATEMPERATURE_INSOLATION_POSITION = 0
STANDARD_GOTEMPERATURE_INSOLATION_VALUE = 310.0
STANDARD_GOTEMPERATURE_INSOLATION_POSITION = 0


class InitialConditionSource(str, Enum):
    SPINUP = "spinup"
    RANDOM = "random"


@dataclass(frozen=True)
class RunConfig:
    output_dir: Path
    filename: str
    dt: float
    transient_time: float
    integration_time: float
    write_steps: int
    num_threads: int
    initial_condition: InitialConditionSource
    initial_condition_path: Path
    seed: int
    random_scale: float
    print_params: bool


def build_standard_params() -> QgParams:
    """Build the qgs MAOOAM setup used by external/qgs/qgs_maooam.py."""

    params = QgParams()
    params.set_atmospheric_channel_fourier_modes(2, 2)
    params.set_oceanic_basin_fourier_modes(2, 4)
    params.set_params(STANDARD_MODEL_PARAMETERS)
    params.atemperature_params.set_params(STANDARD_ATEMPERATURE_PARAMETERS)
    params.gotemperature_params.set_params(STANDARD_GOTEMPERATURE_PARAMETERS)
    params.atemperature_params.set_insolation(
        STANDARD_ATEMPERATURE_INSOLATION_VALUE,
        STANDARD_ATEMPERATURE_INSOLATION_POSITION,
    )
    params.gotemperature_params.set_insolation(
        STANDARD_GOTEMPERATURE_INSOLATION_VALUE,
        STANDARD_GOTEMPERATURE_INSOLATION_POSITION,
    )
    return params


def run_model(config: RunConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    params = build_standard_params()
    if config.print_params:
        params.print_params()

    ic = initial_condition_from_source(
        config.initial_condition.value,
        ndim=params.ndim,
        path=resolve_repo_path(config.initial_condition_path),
        seed=config.seed,
        random_scale=config.random_scale,
    )
    initial_condition_metadata = dict(ic.metadata)
    if config.initial_condition == InitialConditionSource.SPINUP:
        initial_condition_metadata["path"] = str(config.initial_condition_path)

    f, _ = create_tendencies(params)
    integrator = RungeKuttaIntegrator(num_threads=config.num_threads)
    started_at = time.process_time()
    final_ic = ic.values

    try:
        integrator.set_func(f)
        if config.transient_time > 0:
            integrator.integrate(0.0, config.transient_time, config.dt, ic=final_ic, write_steps=0)
            _, final_ic = integrator.get_trajectories()
            final_ic = np.asarray(final_ic, dtype=np.float64)

        integrator.integrate(
            0.0,
            config.integration_time,
            config.dt,
            ic=final_ic,
            write_steps=config.write_steps,
        )
        run_time, trajectory = integrator.get_trajectories()
    finally:
        integrator.terminate()

    run_time = np.asarray(run_time, dtype=np.float64)
    trajectory = np.asarray(trajectory, dtype=np.float64)
    if trajectory.ndim == 1:
        trajectory = trajectory.reshape(1, -1)
    else:
        trajectory = np.moveaxis(trajectory, -1, 0)

    metadata = build_metadata(
        config=config,
        params=params,
        initial_condition_metadata=initial_condition_metadata,
        transient_initial_condition=ic.values,
        final_initial_condition=final_ic,
        n_records=trajectory.shape[0],
        process_time_seconds=time.process_time() - started_at,
    )
    return run_time, trajectory, final_ic, metadata


def build_metadata(
    *,
    config: RunConfig,
    params: QgParams,
    initial_condition_metadata: dict[str, object],
    transient_initial_condition: np.ndarray,
    final_initial_condition: np.ndarray,
    n_records: int,
    process_time_seconds: float,
) -> dict[str, object]:
    qgs_module = sys.modules.get("qgs")
    return {
        "model": "qgs_maooam_standard",
        "source_reference": "external/qgs/qgs_maooam.py",
        "output_path": str(output_path(config)),
        "qgs_version": getattr(qgs_module, "__version__", None),
        "state_dimension": int(params.ndim),
        "state_ordering": [
            "atmospheric streamfunction modes",
            "atmospheric temperature modes",
            "ocean streamfunction modes",
            "ocean temperature modes",
        ],
        "time_unit": "qgs nondimensional model time unit",
        "atmospheric_modes": {"kind": "channel_fourier", "nx": 2, "ny": 2},
        "oceanic_modes": {"kind": "basin_fourier", "nx": 2, "ny": 4},
        "parameters": {
            "model": STANDARD_MODEL_PARAMETERS,
            "atemperature": STANDARD_ATEMPERATURE_PARAMETERS,
            "gotemperature": STANDARD_GOTEMPERATURE_PARAMETERS,
            "atemperature_insolation": {
                "value": STANDARD_ATEMPERATURE_INSOLATION_VALUE,
                "position": STANDARD_ATEMPERATURE_INSOLATION_POSITION,
            },
            "gotemperature_insolation": {
                "value": STANDARD_GOTEMPERATURE_INSOLATION_VALUE,
                "position": STANDARD_GOTEMPERATURE_INSOLATION_POSITION,
            },
        },
        "integration": {
            "dt": config.dt,
            "transient_time": config.transient_time,
            "integration_time": config.integration_time,
            "write_steps": config.write_steps,
            "num_threads": config.num_threads,
            "n_records": n_records,
            "integrator": "qgs.integrators.integrator.RungeKuttaIntegrator",
            "scheme": "classic RK4 default",
        },
        "initial_condition": initial_condition_metadata,
        "transient_initial_condition_shape": list(transient_initial_condition.shape),
        "trajectory_initial_condition_shape": list(final_initial_condition.shape),
        "process_time_seconds": process_time_seconds,
    }


def save_npz(
    output: Path,
    *,
    time_values: np.ndarray,
    trajectory: np.ndarray,
    initial_condition: np.ndarray,
    metadata: dict[str, object],
) -> None:
    output = resolve_repo_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        time=time_values,
        trajectory=trajectory,
        initial_condition=initial_condition,
        metadata_json=np.array(json.dumps(metadata, indent=2, sort_keys=True)),
    )


def output_path(config: RunConfig) -> Path:
    return config.output_dir / config.filename


def resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return ROOT / path


def validate_config(config: RunConfig) -> None:
    if not config.filename:
        raise typer.BadParameter("must not be empty", param_hint="--filename")
    if Path(config.filename).name != config.filename:
        raise typer.BadParameter("must be a filename, not a path", param_hint="--filename")
    if Path(config.filename).suffix != ".npz":
        raise typer.BadParameter("must end with .npz", param_hint="--filename")
    if config.dt <= 0:
        raise typer.BadParameter("must be positive", param_hint="--dt")
    if config.integration_time < 0:
        raise typer.BadParameter("must be non-negative", param_hint="--integration-time")
    if config.transient_time < 0:
        raise typer.BadParameter("must be non-negative", param_hint="--transient-time")
    if config.write_steps < 1:
        raise typer.BadParameter("must be at least 1", param_hint="--write-steps")
    if config.num_threads < 1:
        raise typer.BadParameter("must be at least 1", param_hint="--num-threads")


def main(
    output_dir: Annotated[
        Path,
        typer.Option(help="Directory for the compressed NumPy output file."),
    ] = DEFAULT_OUTPUT_DIR,
    filename: Annotated[
        str,
        typer.Option(help="Output filename inside --output-dir."),
    ] = DEFAULT_OUTPUT_FILENAME,
    dt: Annotated[
        float,
        typer.Option(help="Fixed RK4 timestep in qgs nondimensional model time units."),
    ] = 0.1,
    transient_time: Annotated[
        float,
        typer.Option(help="Transient integration time before recording the saved trajectory."),
    ] = 0.0,
    integration_time: Annotated[
        float,
        typer.Option(help="Recorded integration time after the transient."),
    ] = 100.0,
    write_steps: Annotated[
        int,
        typer.Option(help="Record one state every this many RK4 steps."),
    ] = 10,
    num_threads: Annotated[
        int,
        typer.Option(help="Number of qgs integration worker processes."),
    ] = 1,
    initial_condition: Annotated[
        InitialConditionSource,
        typer.Option(help="Initial-condition source."),
    ] = InitialConditionSource.SPINUP,
    initial_condition_path: Annotated[
        Path,
        typer.Option(help="Path to the spinup .npy state used when initial-condition is spinup."),
    ] = DEFAULT_SPINUP_PATH,
    seed: Annotated[
        int,
        typer.Option(help="Random seed used when initial-condition is random."),
    ] = 21217,
    random_scale: Annotated[
        float,
        typer.Option(help="Random initial-condition amplitude used when initial-condition is random."),
    ] = 0.01,
    print_params: Annotated[
        bool,
        typer.Option(help="Print qgs parameter tables before integration."),
    ] = False,
) -> None:
    if get_start_method() == "spawn":
        freeze_support()

    config = RunConfig(
        output_dir=output_dir,
        filename=filename,
        dt=dt,
        transient_time=transient_time,
        integration_time=integration_time,
        write_steps=write_steps,
        num_threads=num_threads,
        initial_condition=initial_condition,
        initial_condition_path=initial_condition_path,
        seed=seed,
        random_scale=random_scale,
        print_params=print_params,
    )
    validate_config(config)
    time_values, trajectory, initial_condition_values, metadata = run_model(config)
    destination = output_path(config)
    save_npz(
        destination,
        time_values=time_values,
        trajectory=trajectory,
        initial_condition=initial_condition_values,
        metadata=metadata,
    )
    typer.echo(f"Wrote {destination}")
    typer.echo(f"time shape: {time_values.shape}")
    typer.echo(f"trajectory shape: {trajectory.shape}")


if __name__ == "__main__":
    typer.run(main)
