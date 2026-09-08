# model-serving-stack

> **Reference stack for serving LLMs on NVIDIA GPUs** — an OpenAI-compatible FastAPI gateway in front of vLLM, alongside Triton Inference Server and Ray Serve deployments, DCGM-driven Kubernetes autoscaling, Prometheus/Grafana observability, and a BentoML packaging path.

[![CI](https://github.com/TylrDn/model-serving-stack/actions/workflows/ci.yml/badge.svg)](https://github.com/TylrDn/model-serving-stack/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

Serving an LLM well is mostly an infrastructure problem: which runtime you pick changes latency by tens of percent, and the autoscaling signal you pick decides whether you scale before or after users feel it. This repo puts three common runtimes — vLLM, Triton Inference Server, and Ray Serve — behind one OpenAI-compatible API surface, alongside the Kubernetes, monitoring, and benchmarking pieces needed to compare and operate them. It is a reference implementation and a benchmarking harness, not a hosted service; every number quoted here comes from a committed artifact in [`results/`](results/).

## Architecture

```mermaid
flowchart TD
    Client(["Client / App"])
    GW["FastAPI Gateway<br/>api/main.py<br/>(OpenAI-compatible, :8080)"]

    subgraph Backends ["Inference Backends"]
        VLLM["vLLM Server<br/>(PagedAttention)"]
        Triton["Triton Inference Server<br/>(Python backend)"]
        Ray["Ray Serve<br/>(autoscaling deployment)"]
    end

    subgraph Autoscaling ["GPU Autoscaling"]
        DCGM["DCGM Exporter<br/>(GPU metrics)"]
        Adapter["Prometheus Adapter"]
        HPA["Kubernetes HPA<br/>(DCGM_FI_DEV_GPU_UTIL)"]
    end

    subgraph Observability ["Observability"]
        Prom["Prometheus"]
        Graf["Grafana"]
    end

    Client --> GW
    GW -->|"vllm/client.py"| VLLM
    Client -->|"triton/client.py"| Triton
    Client -->|"serve run"| Ray
    Ray -->|"vllm/client.py"| VLLM
    DCGM -->|"GPU utilization metrics"| Prom
    Prom --> Adapter
    Adapter --> HPA
    HPA -->|"scale replicas"| VLLM
    Prom --> Graf
```

The gateway itself calls vLLM only; Triton and Ray Serve are deployed and addressed independently. See [docs/configuration.md](docs/configuration.md#there-is-no-backend-selector).

## Stack Components

| Component | Role | Key Config |
|-----------|------|------------|
| **FastAPI Gateway** | OpenAI-compatible `/v1/chat/completions`, `/v1/models`, `/health` on port `8080` | `api/main.py` |
| **vLLM** | High-throughput LLM inference with PagedAttention | `vllm/server.py`, `vllm/batching_config.yaml` |
| **Triton Inference Server** | Model repository whose Python backend wraps a vLLM engine | `triton/model_repository/llama3/` |
| **Ray Serve** | Autoscaling deployment and multi-model traffic split | `ray_serve/deployment.py`, `ray_serve/multi_model_router.py` |
| **BentoML** | Portable model packaging | `bentoml/service.py`, `bentoml/bentofile.yaml` |
| **DCGM Exporter + HPA** | GPU utilization metrics driving pod autoscaling | `kubernetes/dcgm-exporter.yaml`, `kubernetes/hpa-dcgm.yaml` |
| **Prometheus + Grafana** | Scrape config and GPU serving dashboard | `monitoring/prometheus/prometheus.yml`, `monitoring/grafana/gpu_serving_dashboard.json` |

---

## Quick Start

### 1. Prerequisites

- NVIDIA GPU and driver (the Compose services request `nvidia` GPU devices; `kubernetes/vllm-deployment.yaml` requests `nvidia.com/gpu: 1`)
- Docker + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Python 3.11+ (`requires-python = ">=3.11"`)

### 2. Environment

```bash
cp .env.template .env
```

The variables the code actually reads:

| Variable | Default | Read by |
|----------|---------|---------|
| `MODEL_NAME` | `meta/llama3-8b-instruct` | `vllm/client.py`, `vllm/server.py` |
| `VLLM_HOST` | `localhost` (client), `0.0.0.0` (server) | `vllm/client.py`, `vllm/server.py` |
| `VLLM_PORT` | `8000` | `vllm/client.py`, `vllm/server.py` |
| `TENSOR_PARALLEL_SIZE` | `1` | `vllm/server.py` |
| `MAX_MODEL_LEN` | `8192` | `vllm/server.py` |
| `VLLM_MODEL_NAME` | `meta-llama/Meta-Llama-3-8B-Instruct` | `vllm/openai_server.py`, root `docker-compose.yml` |
| `VLLM_TENSOR_PARALLEL_SIZE` | `1` | `vllm/openai_server.py` |
| `TRITON_HOST` / `TRITON_HTTP_PORT` / `TRITON_GRPC_PORT` | `localhost` / `8000` / `8001` | `triton/client.py` |
| `PROMETHEUS_PORT` / `GRAFANA_PORT` | `9090` / `3000` | root `docker-compose.yml` |
| `MODEL_PATH` / `HF_TOKEN` | — | `deploy/docker-compose.yml` |

`.env.template` also ships `NGC_API_KEY`, `NVIDIA_API_KEY`, `TRITON_METRICS_PORT`, `RAY_ADDRESS`, `RAY_SERVE_PORT` and `LANGCHAIN_*`, which no module in this repo reads — they exist for the surrounding tooling. Full details, including YAML file ownership, are in [docs/configuration.md](docs/configuration.md).

### 3. Run

```bash
# Gateway (:8080) + vLLM + Prometheus + Grafana
docker compose -f deploy/docker-compose.yml up

# Or backends only: Triton (:8000/:8001/:8002), vLLM (:8010), Prometheus, Grafana
docker compose up
```

Without Docker, run the pieces directly from the repository root:

```bash
python -m vllm.server                              # vLLM on :8000
uvicorn api.main:app --host 0.0.0.0 --port 8080    # gateway on :8080
```

### 4. Verify

```bash
# Gateway health (includes a vLLM readiness probe)
curl http://localhost:8080/health

# Models advertised from configs/models.yaml
curl http://localhost:8080/v1/models

# Run a completion
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3-8b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

---

## Configuration

Environment variables are read at process start (`python-dotenv` does not override anything already exported), and `configs/models.yaml` is the only YAML file loaded by application code — it populates `GET /v1/models`. There is no `SERVING_BACKEND` selector: the gateway always calls vLLM. See [docs/configuration.md](docs/configuration.md).

---

## Project Structure

```
model-serving-stack/
├── api/                          # OpenAI-compatible FastAPI gateway (port 8080)
│   ├── main.py                   # /v1/chat/completions, /v1/models, /health, StructuredLoggingMiddleware
│   └── models.py                 # Pydantic request/response schemas
├── vllm/                         # vLLM runtime
│   ├── server.py                 # Launches vllm.entrypoints.openai.api_server from env vars
│   ├── openai_server.py          # In-process FastAPI server over AsyncLLMEngine
│   ├── client.py                 # VLLMClient used by the gateway, Ray Serve and BentoML
│   ├── client_test.py            # Standalone load-test script (not a pytest module)
│   └── batching_config.yaml      # Continuous-batching reference values
├── triton/                       # Triton Inference Server
│   ├── client.py                 # HTTP/gRPC client wrapper
│   ├── perf_analyzer.sh          # perf_analyzer benchmark script
│   └── model_repository/llama3/  # config.pbtxt + 1/model.py (Python backend)
├── ray_serve/                    # Ray Serve
│   ├── deployment.py             # @serve.deployment with AutoscalingConfig
│   ├── multi_model_router.py     # Weighted traffic split across models
│   └── autoscaling_config.yaml   # Autoscaling reference values
├── bentoml/                      # BentoML packaging
│   ├── service.py                # Service + runner definition
│   ├── bentofile.yaml            # Build manifest
│   └── build_and_push.sh         # Build and push helper
├── kubernetes/                   # K8s manifests
│   ├── vllm-deployment.yaml      # Deployment + Service (nvidia.com/gpu: 1)
│   ├── triton-deployment.yaml    # Deployment + Service (HTTP/gRPC/metrics)
│   ├── dcgm-exporter.yaml        # DCGM Exporter DaemonSet
│   ├── hpa-dcgm.yaml             # HPA on DCGM_FI_DEV_GPU_UTIL (canonical)
│   └── ingress.yaml              # NGINX ingress for /vllm and /triton
├── monitoring/
│   ├── prometheus/prometheus.yml # Scrape configs (DCGM, vLLM, Triton)
│   └── grafana/gpu_serving_dashboard.json
├── configs/
│   ├── models.yaml               # Model registry read by the gateway
│   └── serving.yaml              # Reference backend settings
├── evals/
│   ├── benchmark.py              # Async latency/throughput harness
│   └── load_test.py              # CLI wrapper that writes a JSON report
├── results/                      # Committed benchmark artifacts + schema
│   ├── README.md
│   └── baseline_bench_2026-06.json
├── tests/                        # pytest suite (6 modules, mock-based)
├── docs/                         # architecture, configuration, benchmarks, autoscaling, testing
├── deploy/                       # Dockerfile + Compose stack with the gateway
├── notebooks/serving_demo.ipynb
├── docker-compose.yml            # Root: Triton + vLLM + Prometheus + Grafana
├── pyproject.toml                # ruff, mypy, pytest, coverage config
└── requirements.txt
```

---

## Benchmark Results (Baseline — 2026-06-09)

All benchmarks run against `meta/llama3-8b-instruct` on a single A10G GPU.

| Backend | TTFT p50 (ms) | TTFT p99 (ms) | Throughput (tok/s) | Concurrency |
|---------|--------------|--------------|-------------------|-------------|
| vLLM (PagedAttention) | 142 | 387 | 1,840 | 32 |
| Triton TRT-LLM | 98 | 241 | 2,310 | 32 |
| Ray Serve (vLLM backend) | 156 | 412 | 1,780 | 32 |

> **Key takeaway:** TRT-LLM via Triton delivers ~25% lower TTFT vs. native vLLM at equivalent concurrency. vLLM's PagedAttention advantage shows at higher concurrency (64+). See `results/baseline_bench_2026-06.json` for raw data.

---

## Kubernetes Deployment

```bash
# Apply DCGM exporter (DaemonSet)
kubectl apply -f autoscaling/dcgm-exporter.yaml

# Apply HPA (targets DCGM_FI_DEV_GPU_UTIL >= 70%)
kubectl apply -f autoscaling/hpa.yaml

# Deploy vLLM serving pods
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml

# Verify HPA status
kubectl get hpa -n model-serving
```

The HPA scales vLLM replicas when `DCGM_FI_DEV_GPU_UTIL` exceeds 70%, using the NVIDIA DCGM Exporter metric exposed to the Kubernetes metrics server via Prometheus Adapter.

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all checks (matches CI)
ruff check . && mypy . && pytest --cov --cov-fail-under=80

# Run load test
locust -f evals/load_test.py --headless -u 32 -r 4 --run-time 60s --host http://localhost:8000
```

---

## Related Repos

This repo is part of the [NVIDIA Solutions Architect portfolio](https://github.com/TylrDn):

| Repo | Relationship |
|------|--------------|
| [`nvidia-nim-agent-toolkit`](https://github.com/TylrDn/nvidia-nim-agent-toolkit) | NIM client that routes to this serving stack |
| [`inference-optimization-bench`](https://github.com/TylrDn/inference-optimization-bench) | Benchmark results inform backend selection here |
| [`llm-finetuning-lab`](https://github.com/TylrDn/llm-finetuning-lab) | Fine-tuned/exported model weights served here |

---

## License

MIT — see [LICENSE](LICENSE).
