# AGENTS.md

TOPS is a small dynamic power system simulator (RMS/phasor) for research and teaching.
Favour readable, simple code over performance. Quantities are per-unit; state the base (system or machine) when it matters.

## Commands (uv)
- Setup: `uv sync` (add `--group docs` for the docs site)
- Tests: `uv run pytest` (fast). If you change solvers or model equations, also run `uv run pytest -m slow` (PowerFactory validation).
- Lint: `uv run ruff check <files you changed>`. Run `uv run ruff format` on files you changed only, never repo-wide. Don't fix unrelated existing violations.
- Docs: `uv run mkdocs build --strict` when `docs/`, `mkdocs.yml` or docstrings change. Code blocks in pages are executed.

## Layout
- `src/tops/dynamic.py`: `PowerSystemModel`, built from a model dict (see `docs/guide/model-format.md`).
- `src/tops/dyn_models/`: dynamic models (gen, avr, gov, pss, vsc…). Subclass `DAEModel` (`utils.py`), compose `blocks.py`, wire via `connections()`.
- `src/tops/ps_models/`: test systems (k2a, ieee39, n44…). `examples/` holds runnable scripts and must keep working.
- `tests/test_dyn_mdls/`: a new dynamic model needs a test in both `test_init/` and `test_dyn_sim/`.

## Conventions
- Docstrings use reST/sphinx style (`:param x:`), rendered by mkdocstrings.
- Update `docs/guide` or `docs/api` in the same change as any API or behaviour change.
- Commits and PR titles follow Conventional Commits (`feat:`, `fix:`, `docs:`, `ci:`, `!` for breaking).
- Work on a branch. PRs target `main` or `dev`. Never commit to `main`.
- Commit only when asked. Do not add AI `Co-Authored-By` trailers.

## Ask first
- Changing the public API or the model dict format.
- Adding dependencies (core is numpy, scipy, pandas, matplotlib, progressbar2).
- Editing reference data in `tests/test_validation/`.
- Editing release-managed files: `CHANGELOG.md`, the `version` in `pyproject.toml`, `.release-please-manifest.json`.
