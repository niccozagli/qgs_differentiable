# Claude Project Instructions

## Project Goal

Build a minimal differentiable JAX implementation of one fixed qgs MAOOAM
configuration. Validate it against original qgs, then use gradients to recover
selected physical parameters from synthetic observations.

## Always Follow

- Do not rewrite the full qgs package.
- Use JAX with 64-bit precision.
- Keep reusable library code, export scripts, tests, and notebooks separate.
- Do not silently change physical parameter values.
- Validate generated code against qgs or finite differences.
- Record major agent prompts and human corrections in `docs/agent/agent_log.md`.

## Read When Relevant

- Project brief: `docs/project/FERS_projects_Agentic_Coding.pdf`
- Scientific/model context: `docs/model/maooam_qgs_context.md`
- JAX implementation details: `docs/agent/implementation.md`
- Validation requirements: `docs/agent/validation.md`
- Parameter-dependent tensors: `docs/agent/parameter-dependence.md`

## Current Design Choice

For parameter recovery, prefer asking qgs to produce symbolic tensor entries for
the selected physical parameters, then convert those symbolic expressions into a
JAX-compatible tensor function. Use finite-difference tensor decomposition as a
fallback and as an independent validation check.

## Current State

### Symbolic tendency export (done)

`configs/create_tendencies_nonfixed_params.py` builds the standard 36-D MAOOAM
config and calls qgs `create_symbolic_tendencies(..., language='python')` with
16 selected `continuation_variables`, writing the parameter-dependent tendency
to `configs/tendencies_nonfixed_params.py` (a Numba `@njit`
`f(t, U, k_d, ...)` of straight-line `F[i] = ...` assignments).

### JAX tendency module (done) — `src/dqgs/tendency_py2jax.py`

Pure-JAX, standalone (does **not** import qgs), a verbatim transcription of the
generated symbolic tendency:

- `tendency(U, params)` — all 36 `F[i]` expressions copied unchanged into local
  `f_i`, assembled mutation-free with `jnp.stack`; jitted; unused `t` dropped
  (autonomous system). State is 36-D; the `eta_0 == 1` dummy is already folded
  into additive constants.
- `PARAMETER_NAMES` — the 16 free params in signature order.
- `DEFAULT_PARAMS` / `DEFAULT_PARAM_VECTOR` — defaults **read from a qgs
  `model_parameters` object** built per the config script (derived
  nondimensional values like `sigma`, `g_p`, `sc`, `gamma_a` are exact, not
  hand-set).
- `make_tendency(free_names)` → `f_theta(U, theta)` — scatters `theta` (only the
  chosen params) into the default vector and calls `tendency`; fixed params are
  baked-in constants, so `jax.grad`/`jacfwd` w.r.t. `theta` differentiate only
  the selected 1-3 parameters (the parameter-recovery interface).
- `require_x64()` — asserts float64 is enabled; the module does not mutate JAX
  config, so callers/tests must run
  `jax.config.update("jax_enable_x64", True)` first.

Smoke check (in the `dqgs` conda env, which has jax + qgs): matches the
generated source to ~3e-15 in float64.

### JAX RK4 integrator (done) — `src/dqgs/integrate.py`

Pure-JAX, standalone (does **not** import qgs). Acts on a *unary* autonomous
tendency `f(U)` so it is decoupled from the parameter signature (callers pass a
closure `lambda U: tendency(U, params)`).

- `rk4_step(f, U, dt)` — one classic-RK4 step, matching the qgs tableau
  term-for-term (autonomous, so the time argument is dropped).
- `integrate(f, U0, dt, n_steps, write_steps=1)` → trajectory
  `(n_steps//write_steps + 1, dim)`, row 0 = `U0`. `lax.scan` over records, each
  advancing `write_steps` steps via an inner carry-only `fori_loop` (dropped
  states never stacked, mirroring qgs `write_steps`). Jitted with `f`, `n_steps`,
  `write_steps` static; requires `n_steps % write_steps == 0`.
- No manual threading/multiprocessing: one compiled XLA program; RK4 is
  sequential in time; ensembles are `jax.vmap`; autodiff needs a single traced
  graph (multiprocessing would break it).

### Trajectory validation (done) — `notebooks/trajectory_validation.ipynb`

Layer-2 check from `docs/agent/validation.md`: integrate the same IC
(`configs/ic/de_cruz_active_ic.npy`), params, `dt=0.1`, and RK4 through both qgs
(`create_tendencies` + `integrate_runge_kutta`) and dqgs, and compare. Results
(executed): pointwise tendency match ~1e-17, short-window (T=10) trajectory error
~9e-17 (round-off), expected chaotic divergence over T=1000, `write_steps`
cross-check matches qgs to ~1e-17. Includes the no-multiprocessing rationale.

### Trajectory test (done) — `tests/test_trajectory.py`

Pytest distilled from the notebook (4 tests, all passing in the `dqgs` env):
pointwise tendency vs qgs, short-window trajectory max error < 1e-8, `write_steps`
consistency (vs full run and vs qgs), and the non-multiple `write_steps`
guard. `tests/conftest.py` enables x64 at collection time and provides the qgs
model / tendency / IC fixtures. Run: `~/miniconda3/envs/dqgs/bin/python -m pytest`.

### Conventions / environment

- Run anything needing jax or qgs in the `dqgs` conda env
  (`~/miniconda3/envs/dqgs/bin/python`); a bare shell has neither.
- `src/dqgs` stays standalone; `scripts`/`tests` may import both qgs and dqgs.

### Not done yet

- `tests/` — open layers: state-Jacobian vs qgs `Df`, parameter-derivative
  AD-vs-finite-difference (incl. qgs rebuilt at perturbed params), long-run
  climate statistics. (Tendency match and trajectory validation are done — see
  `tests/test_trajectory.py`.)
- Parameter-recovery notebook.
