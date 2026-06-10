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
