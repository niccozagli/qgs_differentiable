# Agent Log

Record major prompts, generated changes, and human corrections here.

## Entries

- 2026-06-10: Created initial Claude-first agent instruction scaffold.
- 2026-06-10: Added a standard qgs MAOOAM run script that saves trajectory
  arrays with JSON metadata in `.npz` format, plus a reusable initial-condition
  helper for loading the De Cruz spinup state or generating qgs-style random
  starts.
- 2026-06-10: Updated the standard qgs run script to use Typer for the CLI,
  keeping integration parameters as typed options and declaring `typer` in the
  project dependency files.
- 2026-06-10: Moved the standard qgs run output default to `data/` and split
  output naming into `--output-dir` and `--filename`.
- 2026-06-10: Converted the generated symbolic tendency
  (`configs/tendencies_nonfixed_params.py`) into a pure JAX module
  `src/dqgs/tendency.py`. Verbatim transcription of all 36 `F[i]` expressions
  into `tendency(U, params)` (mutation-free `jnp.stack`, dropped unused `t`,
  jitted). Added `PARAMETER_NAMES` / `DEFAULT_PARAMS` (16 defaults read from a
  qgs `model_parameters` object built per the config script — not hand-set) and
  `make_tendency(free_names)` so only a chosen 1-3 parameter subset is exposed to
  AD. Smoke check: matches the generated source to 2.7e-15 in float64.
  Human correction: implement the code first, defer the validation suite to a
  separate plan.
- 2026-06-11: Reviewed the dqgs JAX tendency/RK4 commit against qgs. Fixed an
  import-order precision bug by keeping `DEFAULT_PARAM_VECTOR` as a NumPy
  float64 array until callers convert it under JAX x64. Added regression tests
  for that import order and for matching the generated symbolic tendency at
  perturbed parameter values.
- 2026-06-12: Added a simple synthetic parameter-estimation layer. The workflow
  treats validated defaults as truth, fixes `U0 = obs[0]`, compares
  `pred[1:]` against `obs[1:]`, and optimizes log-relative parameter variables
  with Adam. Added a fast estimator test suite using a small linear JAX system
  plus an editable MAOOAM recovery notebook.
- 2026-06-12: Fixed the MAOOAM recovery notebook hanging before the first
  optimizer print. Staged diagnosis isolated the cause: forward `tendency`
  (~0.9s) and bare reverse-grad of `tendency` (~2.5s) compile fine, but
  reverse-mode `jax.value_and_grad` through `simulate` (the RK4 chain inside
  `lax.scan`/`fori_loop`) never finished compiling its backward graph for the
  36-D symbolic tendency. Switched `fit_parameters` to a jitted forward-mode
  gradient `(loss, jax.jacfwd(loss))` — correct for 1-3 free params and a scalar
  loss, since forward-mode propagates a few tangents through the existing scan
  with no backward pass. Verified: 36-D `k_d` recovery over a 20-tu window,
  60 iters in ~17s (mostly one-time compile, ~0.08 ms/iter steady state), loss
  3.5e-9 -> 1.1e-12; `tests/test_estimate.py` still passes (3). Note for the
  notebook: `N_STEPS=1` (0.1 tu) is too short to identify parameters — use a few
  hundred steps.
