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
