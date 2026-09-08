# Testing

## Running the suite

```bash
pip install -r requirements.txt
pip install -e ".[dev]"

ruff check .
pytest tests/ -v -m "not gpu" \
  --cov=api --cov=vllm --cov=triton --cov=evals \
  --cov-report=term-missing \
  --cov-fail-under=80
```

That is exactly what `.github/workflows/ci.yml` runs on every push and pull
request to `main`, on `ubuntu-latest` with Python 3.11 (no GPU runner). Pytest
settings come from `[tool.pytest.ini_options]` in `pyproject.toml`:
`testpaths = ["tests"]`, `pythonpath = ["."]`, `asyncio_mode = "auto"`.

## Coverage map

Six test modules, 16 tests, all mock-based — none require a GPU, a model, or a
running server.

| Test module | Covers | What it asserts |
|-------------|--------|-----------------|
| `tests/test_api.py` | `api/main.py`, `api/models.py` | `GET /health` returns `status: ok`; `POST /v1/chat/completions` returns an OpenAI-shaped `chat.completion` body with an `X-Request-ID` header (backend client patched); `GET /v1/models` returns a non-empty list from `configs/models.yaml`. |
| `tests/test_api_middleware.py` | `StructuredLoggingMiddleware` in `api/main.py` | `/health` is excluded from request-ID tagging; a chat request gets `X-Backend: vllm` and a 36-character UUID `X-Request-ID`. |
| `tests/test_benchmark.py` | `evals/load_test.py` (and the `BenchmarkResult` dataclass from `evals/benchmark.py`) | `build_report` maps benchmark fields; `save_report` writes JSON to a `tmp_path`; `run_load_test` delegates to a patched `run_benchmark`. |
| `tests/test_triton_client.py` | `triton/client.py` | `health_check` returns `True`/`False` against a fake `tritonclient` module tree injected into `sys.modules`, so the real GPU client library is never imported. |
| `tests/test_vllm_client.py` | `vllm/client.py` | `health_check` true/false paths and `chat()` returning the assistant content, with `openai.OpenAI` patched. |
| `tests/test_vllm_openai.py` | `vllm/openai_server.py`, `vllm/client.py`, `vllm/server.py` | Import-spec smoke checks only: each module file exists and is loadable by `importlib.util.spec_from_file_location`. These modules are never executed, because importing them requires the real `vllm` package and a GPU. |

Current measured coverage over the CI-scoped packages is **87%** (179
statements, 23 missed): `api/main.py` 92%, `api/models.py` 100%,
`vllm/client.py` 100%, `evals/load_test.py` 76%, `triton/client.py` 75%. The
gate is 80%, so there is roughly seven points of headroom.

## Not covered

**Excluded from measurement** via `[tool.coverage.run] omit` in `pyproject.toml`
— they need a GPU and the real `vllm` runtime to import:

- `evals/benchmark.py`
- `vllm/server.py`
- `vllm/openai_server.py`
- `vllm/client_test.py`

**Never exercised, and not in the `--cov` scope at all:**

- `ray_serve/deployment.py`, `ray_serve/multi_model_router.py` — no tests. The
  pure function `select_model()` in the router is testable without Ray and is
  the obvious first gap to close.
- `bentoml/service.py`, `bentoml/build_and_push.sh` — no tests.
- `triton/model_repository/llama3/1/model.py` — the Triton Python backend; also
  excluded from ruff via `[tool.ruff] exclude`.
- `api/__init__.py`, `triton/__init__.py`, `vllm/__init__.py` — empty.

**Not validated by any automated check:** the Kubernetes manifests in
`kubernetes/`, the Compose files, `monitoring/prometheus/prometheus.yml`,
`monitoring/grafana/gpu_serving_dashboard.json`, `configs/*.yaml`, and
`triton/model_repository/llama3/config.pbtxt`. Nothing in CI lints or
schema-checks them; they are reviewed by hand.

## The `gpu` marker

`pyproject.toml` registers one marker:

```toml
markers = ["gpu: marks tests that require a GPU"]
```

Any test that needs real CUDA, real model weights, or a live inference server
must be decorated:

```python
@pytest.mark.gpu
def test_real_engine_generates():
    ...
```

CI always runs `-m "not gpu"`, so marked tests are skipped there and only run
locally on a GPU host:

```bash
pytest tests/ -m gpu -v   # GPU host only
```

No test in the suite currently carries the marker; every existing test is a
mock-level unit test.
