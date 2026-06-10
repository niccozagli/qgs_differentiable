# Implementation Notes

## Scope

- Reimplement only one fixed qgs MAOOAM configuration.
- Do not port qgs internals beyond what is needed to export model tensors,
  metadata, initial conditions, and validation trajectories.
- Keep qgs-facing export code separate from reusable JAX model code.

## JAX Model

- The default MAOOAM tendency is a sparse quadratic tensor contraction over the
  augmented state `[1, x]`.
- A dense tensor is acceptable for the recommended 36-dimensional model.
- Use `jax.config.update("jax_enable_x64", True)` before numerical work.
- Implement classic fixed-step RK4 and trajectory integration with
  `jax.lax.scan`.
- Keep JAX functions pure: pass tensors, parameters, state, and timestep
  explicitly.

## Tensor Representation

- For frozen validation, the model may use exported tensor coordinates and
  values from qgs.
- For parameter recovery, the tensor must be a function of the selected
  physical parameter values.
- Preferred approach: ask qgs for symbolic tensor entries involving the selected
  physical parameters, then build a JAX-compatible evaluator for those entries.
- Fallback approach: construct `T(theta) = T0 + theta * T1` from qgs tensors
  built at multiple values when the selected parameter is known to enter
  linearly.
