# Contributing to DriftGuard

Thanks for contributing to DriftGuard.

## Development Setup

Create or activate your virtual environment, then install the project with development dependencies:

```bash
pip install -e ".[dev]"
python -m spacy download en_core_web_sm
```

## Project Expectations

- Keep changes focused and easy to review.
- Add or update pytest-compatible tests alongside behavior changes.
- Prefer improving the shared runtime and documented public APIs over adding one-off entrypoints.
- Preserve backward compatibility where practical, especially around public package imports, storage formats, and demos.

## Running Checks

Run the full suite:

```bash
python -m pytest
```

Run collection only:

```bash
python -m pytest --collect-only
```

Run the benchmark CLI:

```bash
driftguard-benchmark
```

## Pull Request Guidance

- Describe the problem and the user-facing change.
- Mention any settings, storage, or API implications.
- Include relevant test coverage.
- Update `README.md` when the integration surface or documented behavior changes.

## Areas Where Contributions Are Especially Helpful

- storage backends and migration safety
- agent framework adapters
- benchmark datasets and evaluation quality
- observability and metrics export
- docs and integration examples
