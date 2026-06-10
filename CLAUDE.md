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
