# Parameter Dependence

Do not freeze a single tensor when estimating physical parameters.

The qgs tendency tensor contains physical parameters in its entries. A JAX model
used for parameter recovery must represent those entries as functions of the
parameters being estimated.

## Preferred Approach: Symbolic qgs Tensor Entries

Ask qgs to write the tensor entries symbolically for the selected physical
parameters. Export enough information to evaluate the same tensor entries in
JAX:

- tensor coordinates;
- symbolic expression for each parameter-dependent value;
- numeric values for parameter-independent constants;
- mapping from symbolic parameter names to runtime JAX scalar arguments;
- reference numeric tensor values from qgs for validation.

The symbolic route is preferred because it preserves exact parameter dependence
for the selected parameters instead of inferring it from finite differences.

## Fallback Approach: Linear Tensor Decomposition

For parameters known to enter linearly, represent:

```python
T(theta) = T0 + theta * T1
```

Recover `T0` and `T1` by building qgs tensors at two parameter values:

```python
T1 = (T_b - T_a) / (theta_b - theta_a)
T0 = T_a - theta_a * T1
```

Validate linearity by rebuilding at a third parameter value.

## Required Validation

Before trusting gradients, compare `df/dtheta` from the JAX tensor function
against a finite difference of qgs tendencies rebuilt at `theta +/- delta`.

This validation is required for both symbolic tensor export and linear tensor
decomposition.
