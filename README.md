# Differentiable qgs

Minimal differentiable JAX implementation of one fixed qgs MAOOAM configuration.

The project uses upstream `qgs` as the reference implementation for tensor
export, validation trajectories, and finite-difference checks. The JAX model
lives separately in `src/dqgs`.

## Setup

Clone upstream qgs into `external/qgs`:

```bash
git clone https://github.com/Climdyn/qgs.git external/qgs
```

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate dqgs
```

Install qgs from the local clone:

```bash
pip install -e external/qgs
```

Install this project:

```bash
pip install -e .
```

Verify the imports:

```bash
python -c "import qgs; print(qgs.__file__)"
python -c "import dqgs; print(dqgs.__file__)"
```

The qgs path should point inside `external/qgs`, and the dqgs path should point
inside `src/dqgs`.

Verify JAX is using 64-bit floats:

```bash
python - <<'PY'
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
print(jnp.ones(1).dtype)
PY
```

Expected output:

```text
float64
```

## Repository Layout

```text
external/qgs/   upstream qgs clone, treated as read-only reference code
configs/        exported tensors, metadata, initial conditions, trajectories
src/dqgs/       JAX implementation
scripts/        qgs export scripts, validation runners, experiments
tests/          pytest validation suite
notebooks/      reproducible analysis notebooks
docs/           project, model, and agent documentation
```

## Development Boundary

- `external/qgs` is the reference implementation. Do not edit it during normal
  project work.
- `src/dqgs` should not import `qgs`; it should remain a standalone JAX model.
- `scripts` and `tests` may import both `qgs` and `dqgs`.
- Validate every model layer against qgs or finite differences before using it
  in parameter recovery.

## Next Steps

1. Write a qgs export script for the fixed MAOOAM configuration.
2. Export tensor coordinates, tensor values, metadata, initial condition, and a
   short reference trajectory.
3. Implement the JAX tensor contraction in `src/dqgs`.
4. Implement fixed-step RK4 in JAX.
5. Add tests comparing JAX tendencies and trajectories against qgs.
6. Add parameter-dependent tensor export using qgs symbolic outputs for selected
   parameters.
7. Validate `df/dtheta` against qgs finite differences.
8. Build the parameter-recovery notebook.
