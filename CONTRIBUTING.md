# Contributing

```bash
uv sync --locked
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

- Python 3.10+; line length 100.
- Conventional commits (`feat:`, `fix:`, `docs:`, `test:`, `chore:`).
- Every `feat:` / `fix:` commit adds a line under `## [Unreleased]` in `CHANGELOG.md`.
- Commit `uv.lock` together with any dependency change (`uv lock`).
- The default test suite makes no network calls and loads no models.
