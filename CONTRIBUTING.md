# Contributing to model-serving-stack

Thank you for your interest in contributing! This document outlines the development workflow and standards.

## Development Setup

```bash
git clone https://github.com/TylrDn/model-serving-stack.git
cd model-serving-stack
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
cp .env.template .env  # see docs/configuration.md
```

## Code Standards

CI (`.github/workflows/ci.yml`) enforces two gates on every push and pull request to `main`:

- **`ruff`** — lint rules `E`, `F`, `I` at line length 100 (`[tool.ruff]` in `pyproject.toml`)
- **`pytest`** — with `--cov-fail-under=80` over `api`, `vllm`, `triton`, `evals`

`mypy` is configured in `pyproject.toml` but is **not** run by CI. Running it locally is welcome; a clean `mypy` run is not a merge requirement today.

Run exactly what CI runs before submitting a PR:

```bash
ruff check .
pytest tests/ -v -m "not gpu" \
  --cov=api --cov=vllm --cov=triton --cov=evals \
  --cov-report=term-missing \
  --cov-fail-under=80
```

## Branch & PR Workflow

1. Branch from `main`: `git checkout -b feat/your-feature`
2. Make changes with focused commits using [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `docs:`, `test:`, `refactor:`
3. Open a PR against `main` — CI runs automatically
4. All CI checks must be green before merge

## Adding a New Backend

To add a new serving backend (e.g., TensorRT-LLM standalone):

1. Create a directory under the backend name: `tensorrt_llm/`
2. Implement a client with the same surface as `vllm/client.py` (`chat()` and `health_check()`), which is what `api/main.py`, `ray_serve/deployment.py` and `bentoml/service.py` consume
3. Add the backend key to `configs/models.yaml`
4. Add unit tests in `tests/test_<backend>.py`, mocking the client library the way `tests/test_triton_client.py` does, and mark anything needing real hardware with `@pytest.mark.gpu`
5. Document any new environment variables in `.env.template` and `docs/configuration.md`

## Testing

```bash
# All non-GPU tests
pytest tests/ -m "not gpu"

# Specific test file
pytest tests/test_api.py -v

# HTML coverage report
pytest --cov=. --cov-report=html
open htmlcov/index.html

# Load test against a running gateway on :8080
python -m evals.load_test --requests 500 --concurrency 32 --output-dir results/load_tests
```

See [docs/testing.md](docs/testing.md) for the coverage map and the `gpu` marker convention.

## Release Process

1. Bump `version` in `pyproject.toml` following [SemVer](https://semver.org/spec/v2.0.0.html).
2. Move the `## [Unreleased]` entries in `CHANGELOG.md` under a new
   `## [X.Y.Z] - YYYY-MM-DD` heading and leave a fresh empty `## [Unreleased]`
   section.
3. Commit with `chore(release): vX.Y.Z` and confirm CI is green on `main`.
4. Tag and push:
   ```bash
   git tag -a vX.Y.Z -m "vX.Y.Z"
   git push origin main --follow-tags
   ```
5. If the release changes benchmark numbers, commit the new artifact under
   `results/` using the schema in [results/README.md](results/README.md) and
   update the README table in the same commit.

## Questions

Open an issue on [this repository](https://github.com/TylrDn/model-serving-stack/issues).
