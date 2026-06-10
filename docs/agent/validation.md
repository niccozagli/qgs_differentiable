# Validation Protocol

Validate in layers:

1. Compare JAX tendency against qgs tendency at sampled states.
2. Compare short JAX and qgs trajectories from the same initial condition and
   timestep.
3. Compare tensor parameter derivatives against qgs rebuilt at perturbed
   parameter values.
4. Compare AD gradients against centered finite differences of the full loss.
5. Compare long-run climate statistics after trajectories diverge.

Do not proceed to parameter recovery until tendency, parameter-dependence, and
gradient checks pass.

## Expected Checks

- Tendency error should be near float64 round-off for the frozen reference
  tensor.
- Short trajectory error should remain near round-off initially when qgs and JAX
  use the same initial state, timestep, and RK4 scheme.
- AD gradients should be checked only on short enough windows that finite
  differences remain meaningful.
- Long-window failures are scientific results only after short-window validation
  has passed.
