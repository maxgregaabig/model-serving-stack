# Configuration

Two mechanisms configure this stack: environment variables (read at runtime by
Python and Compose) and YAML files. They are **not** layered — each setting is
owned by exactly one of them, and only one YAML file is read by application
code.

## Precedence

1. **Process environment wins.** Every entrypoint calls
   `load_dotenv()` from `python-dotenv`, which does not override variables
   already exported in the shell or injected by Compose/Kubernetes. `.env` only
   fills gaps.
2. **`.env` fills the rest.** Copy the template and edit:
   ```bash
   cp .env.template .env
   ```
3. **Hardcoded defaults apply last.** Every `os.getenv` call in this repo passes
   a default, so nothing is strictly required to start a process (see the table
   below).
4. **YAML files do not feed the environment.** `configs/models.yaml` is the only
   YAML read by Python (`api/main.py`). The others are reference configuration
   for the tools that consume them out of band.

## Environment variables actually read by code

| Variable | Read by | Default | Purpose |
|----------|---------|---------|---------|
| `MODEL_NAME` | `vllm/client.py`, `vllm/server.py`, `triton/perf_analyzer.sh`, `deploy/docker-compose.yml` | `meta/llama3-8b-instruct` (`llama3` in `perf_analyzer.sh`) | Model id sent on every gateway request and passed to `--model` on launch. |
| `VLLM_HOST` | `vllm/client.py`, `vllm/server.py` | `localhost` (client), `0.0.0.0` (server) | vLLM host the gateway connects to / the server binds. |
| `VLLM_PORT` | `vllm/client.py`, `vllm/server.py` | `8000` | vLLM port. |
| `TENSOR_PARALLEL_SIZE` | `vllm/server.py`, `deploy/docker-compose.yml` | `1` | GPUs for tensor parallelism (`--tensor-parallel-size`). |
| `MAX_MODEL_LEN` | `vllm/server.py` | `8192` | vLLM `--max-model-len`. |
| `VLLM_MODEL_NAME` | `vllm/openai_server.py`, root `docker-compose.yml` | `meta-llama/Meta-Llama-3-8B-Instruct` | Model for the in-process `AsyncLLMEngine` and the Compose `vllm` service. Distinct from `MODEL_NAME`. |
| `VLLM_TENSOR_PARALLEL_SIZE` | `vllm/openai_server.py` | `1` | Tensor parallelism for the in-process engine only. |
| `TRITON_HOST` | `triton/client.py` | `localhost` | Triton host. Not in `.env.template`. |
| `TRITON_HTTP_PORT` | `triton/client.py` | `8000` | Triton HTTP port. |
| `TRITON_GRPC_PORT` | `triton/client.py` | `8001` | Triton gRPC port (used when `protocol="grpc"`). |
| `TRITON_URL` | `triton/perf_analyzer.sh` | `localhost:8000` | `perf_analyzer -u` target. |
| `CONCURRENCY`, `REQUEST_COUNT` | `triton/perf_analyzer.sh` | `4`, `100` | `perf_analyzer` load shape. |
| `PROMETHEUS_PORT` | root `docker-compose.yml` | `9090` | Host port for Prometheus. |
| `GRAFANA_PORT` | root `docker-compose.yml` | `3000` | Host port for Grafana. |
| `HF_TOKEN` | `deploy/docker-compose.yml` | — | Passed to the vLLM container for gated model downloads. Not in `.env.template`. |
| `MODEL_PATH` | `deploy/docker-compose.yml` | — | Host path bind-mounted to `/models` in the vLLM container. |

In Kubernetes, `kubernetes/vllm-deployment.yaml` instead injects
`HUGGING_FACE_HUB_TOKEN` from the secret `hf-secret`, key `token`.

### Declared in `.env.template` but not read by repo code

`NGC_API_KEY`, `NVIDIA_API_KEY`, `TRITON_METRICS_PORT`, `RAY_ADDRESS`,
`RAY_SERVE_PORT`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT`.

They are placeholders for the surrounding tooling — `NGC_API_KEY` for
`docker login nvcr.io` when pulling the Triton image, `RAY_ADDRESS` for the
`ray` CLI and client libraries, `LANGCHAIN_*` for LangSmith — but no module in
this repo reads them. Setting them changes nothing about this codebase's
behaviour on its own.

### There is no backend selector

There is no `SERVING_BACKEND` variable. `api/main.py` instantiates `VLLMClient`
at import time and every `/v1/chat/completions` request goes to vLLM. Triton and
Ray Serve are deployed and called independently:

- Triton via `triton/client.py` (`TritonClient(...).infer(...)`) or its HTTP/gRPC
  ports directly.
- Ray Serve via `ray_serve/deployment.py` (`serve run ray_serve.deployment:entrypoint`),
  which exposes `/chat` and itself calls `VLLMClient`.

Switching the gateway's backend today requires a code change, not an env var.

## YAML configuration

| File | Read by | Notes |
|------|---------|-------|
| `configs/models.yaml` | `api/main.py` (`_load_model_registry`) | The only YAML loaded by application code. Read from the path `configs/models.yaml` **relative to the process working directory**, so run the gateway from the repository root. Only the `name` field (falling back to `path`) is used, to populate `GET /v1/models`; `backend`, `precision` and `tensor_parallel` are currently ignored. If the file is missing or empty, the registry falls back to a single card, `meta/llama3-8b-instruct`. |
| `configs/serving.yaml` | Nothing in this repo | Reference values for vLLM, Triton and Ray Serve (`gpu_memory_utilization: 0.90`, `max_model_len: 8192`, Ray `max_concurrent_queries: 64`, ...). Editing it does not change runtime behaviour. |
| `vllm/batching_config.yaml` | Nothing in this repo | Continuous-batching reference values (`max_num_seqs: 256`, `max_num_batched_tokens: 8192`, `swap_space: 4`). Apply them by passing the equivalent vLLM CLI flags. |
| `ray_serve/autoscaling_config.yaml` | Nothing in this repo | Mirrors the `AutoscalingConfig` hardcoded in `ray_serve/deployment.py` (min 1, max 4, 8 ongoing requests per replica). Change both together. |
| `bentoml/bentofile.yaml` | `bentoml build` | Build manifest for the Bento. |
| `triton/model_repository/llama3/config.pbtxt` | Triton Inference Server | Protobuf text, not YAML: `backend: "python"`, `max_batch_size: 0`, decoupled transaction policy, inputs `text_input` / optional `max_tokens` / optional `temperature`, output `text_output`, one GPU instance. |

## Examples

### Single model, vLLM through the gateway

```bash
cp .env.template .env
# .env
MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct
VLLM_HOST=localhost
VLLM_PORT=8000
TENSOR_PARALLEL_SIZE=1
MAX_MODEL_LEN=8192
```

```bash
python -m vllm.server                                   # launches vLLM on :8000
uvicorn api.main:app --host 0.0.0.0 --port 8080         # gateway on :8080
curl http://localhost:8080/health
```

### Multiple models advertised by the gateway

`GET /v1/models` reflects `configs/models.yaml`. The committed file lists
`llama3-8b`, `llama3-70b` and `mistral-7b`, so all three appear in the response.
Because the gateway forwards every request to the single vLLM instance
identified by `MODEL_NAME`, the `model` field in a chat request does not change
which backend serves it. Real multi-model routing lives in
`ray_serve/multi_model_router.py`, which splits traffic by the `TRAFFIC_SPLIT`
weights defined in that module (`llama3-8b` 0.8, `llama3-70b` 0.2).

### Multiple backends side by side

The root `docker-compose.yml` runs Triton (`8000`/`8001`/`8002`), vLLM
(host `8010` -> container `8001`), Prometheus and Grafana — but no gateway.
`deploy/docker-compose.yml` runs vLLM plus the gateway (`8080`), Prometheus and
Grafana, with `VLLM_HOST=vllm` so the gateway reaches vLLM over the Compose
network.

```bash
docker compose up                      # backends + monitoring
docker compose -f deploy/docker-compose.yml up   # gateway + vLLM + monitoring
```

Port `8000` is claimed by both Triton and a locally launched vLLM. When running
both on one host, move one of them (`VLLM_PORT`, or Triton's
`--http-port`) before starting.
