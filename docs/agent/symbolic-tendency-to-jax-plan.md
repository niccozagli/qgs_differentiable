# Symbolic qgs Tendency to JAX Converter Plan

## Summary

Implement a deterministic project-owned wrapper pipeline that converts qgs
symbolic tendency output into a pure JAX tendency function.

Do not modify `external/qgs`. Treat qgs as a read-only reference dependency.
The notebook `configs/symbolic_output_ocean_atmosphere.ipynb` remains
exploratory documentation, not the reproducible build path.

The desired workflow is:

```bash
python scripts/generate_symbolic_qgs_tendency.py
python scripts/convert_symbolic_tendency_to_jax.py configs/generated/tendency_symbolic.py --output src/dqgs/generated_tendency.py
pytest
```

## Key Changes

- Add `scripts/generate_symbolic_qgs_tendency.py`.
- Build the standard 36D MAOOAM qgs configuration.
- Select free parameters through qgs `continuation_variables`.
- Call `create_symbolic_tendencies(..., language="python")`.
- Write the generated symbolic Python tendency to
  `configs/generated/tendency_symbolic.py`.
- Write companion metadata with parameter names/default values.
- Harden `scripts/convert_symbolic_tendency_to_jax.py`.
- Parse qgs-generated Python with `ast`.
- Infer free parameter names from `def f(t, U, ...)`.
- Infer output dimension from `F[i] = ...` assignments.
- Emit mutation-free JAX code using `jnp.stack`.
- Preserve parameter order in `PARAMETER_NAMES`.

Generated JAX modules should expose:

```python
PARAMETER_NAMES = (...)

def f_jax(U, params):
    ...
```

where `params` is a 1D vector ordered exactly as `PARAMETER_NAMES`.

## Converter Rules

Accept only qgs-style straight-line generated functions:

```python
@njit
def f(t, U, p1, p2, ...):
    F = np.empty_like(U)
    F[0] = ...
    F[1] = ...
    return F
```

Reject unsupported inputs explicitly:

- missing `F[i]` assignments;
- duplicate `F[i]` assignments;
- non-contiguous output indices;
- wrong function name or signature;
- control flow inside the generated function;
- mutation other than `F[i] = ...`.

Convert expression namespaces:

- `np.*` to `jnp.*`;
- `math.*` to `jnp.*`;
- algebra, powers, state indexing `U[i]`, and scalar parameter names remain
  unchanged.

## Test Plan

- Unit-test converter on toy qgs-style functions:
  - zero free parameters;
  - one free parameter;
  - many free parameters;
  - different output dimensions;
  - expressions with powers, `np.sqrt`, `np.exp`, and `math.sqrt`.
- Unit-test failure modes:
  - missing output index;
  - duplicate output index;
  - wrong signature;
  - unsupported control flow.
- Integration validation:
  - generate symbolic MAOOAM tendency from qgs;
  - convert to JAX;
  - import generated JAX module;
  - evaluate at default parameters;
  - compare against qgs `create_tendencies(model_parameters)` on sampled states.
- Later scientific validation:
  - compare `jax.jacfwd(f_jax, argnums=0)` against qgs `Df`;
  - compare parameter derivatives against finite differences from qgs rebuilt at
    perturbed parameter values.

## Assumptions

- `external/qgs` is not edited.
- The qgs symbolic API remains the source of symbolic equations.
- The converter must work for arbitrary qgs truncation dimension and arbitrary
  number of free parameters.
- The parameter vector interface is preferred over a parameter dictionary for
  JAX/optimizer compatibility.
- Generated source may be committed initially for convenience, but the
  generation script plus converter are the reproducible source of truth.
