# Repository Guidelines

## Project Structure & Module Organization
Core library code lives in `slime/` (notably `backends/`, `rollout/`, `ray/`, `router/`, and `utils/`). Optional extensions and model bridges live in `slime_plugins/`. Use `examples/` for end-to-end usage patterns and `scripts/` for runnable launch scripts (for example, `scripts/run-qwen3-4B.sh`). Tests are in `tests/`, with CI helpers under `tests/ci/`. Documentation sources are in `docs/`; checkpoint/model conversion tools are in `tools/`; container definitions are in `docker/`.

## Build, Test, and Development Commands
- `pip install -e . --no-deps`: editable install used by CI jobs.
- `pytest`: run test suite with project defaults from `pyproject.toml`.
- `pytest tests/test_chunked_gae.py`: run a focused local test.
- `pre-commit run --all-files --show-diff-on-failure --color=always`: run lint/format checks before pushing.
- `bash docs/build.sh en && bash docs/serve.sh en`: build and serve docs locally (run from `docs/`).

For GPU-gated integration tests, follow CI style:
`python tests/ci/gpu_lock_exec.py --count 4 -- python tests/test_qwen2.5_0.5B_gsm8k_short.py`.

## Coding Style & Naming Conventions
Target Python is 3.10+. Use 4-space indentation and `snake_case` for modules, files, and functions. Formatting and import order are enforced by Black (line length 119) and isort (`profile=black`). Ruff checks `E/F/B/UP` rules (with `E402` and `E501` currently ignored). Keep new scripts/tests consistently named (`run-*.sh`, `test_*.py`).

## Testing Guidelines
Pytest is configured with `--verbose --pyargs --durations=0 --strict-markers`. Place tests under `tests/` and name them `test_*.py`. Use markers (`unit`, `integration`, `system`, `acceptance`) to classify intent. There is no explicit global coverage threshold in-repo; include targeted tests for each behavior change and note any hardware constraints.

## Commit & Pull Request Guidelines
Recent history favors bracketed prefixes plus concise imperative summaries, for example: `[Fix] ... (#1589)` or `[Feature] ... (#1588)`. Keep commits focused and reference issue/PR IDs when available. For PRs, include:
- what changed and why,
- exact validation commands run,
- required environment details (GPU count/model/data paths) for reproducibility.

Pre-commit runs on all PRs, while larger GPU suites are label-triggered (`run-ci-short`, `run-ci-fsdp`, `run-ci-megatron`, etc.).
