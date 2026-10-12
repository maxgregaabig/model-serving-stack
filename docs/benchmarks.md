# Benchmarks

**This repository publishes no benchmark numbers.** It has harnesses; it does
not yet have a result that can be reproduced.

## The retired baseline

`baseline_bench_2026-06.json` compared vLLM, Triton TRT-LLM and Ray Serve on a
single A10G. It has been moved to
[`results/unverified/`](../results/unverified/README.md) and is quoted nowhere
in this repository, because it fails rule 1 of
[`results/README.md`](../results/README.md) — "A result without `git_sha`,
`command`, and `software_versions` is not reproducible and should not be
committed". Specifically it recorded:

- **No `git_sha`** — the code state that produced the run is unknown.
- **No `command`** — the exact invocation, including flags, is unknown.
- **No `software_versions`** — vLLM, TensorRT-LLM, Triton, CUDA driver and
  container image tags are all unrecorded. `requirements.txt` pins only lower
  bounds (`vllm>=0.4.0`, `tritonclient[all]>=2.44.0`, `ray[serve]>=2.10.0`), so
  it does not identify the versions used.
- **No raw output** — only the aggregated summary, so nothing can be recomputed.
- **No tool that emits its schema** — `evals/benchmark.py` measured end-to-end
  request latency, not TTFT or tokens/sec.

The numbers are not restated here. Restating an unreproducible measurement in a
document that explains why it is unreproducible is how such numbers end up
quoted anyway.

## The replacement harness

`evals/bench_report.py` emits the schema in
[`results/README.md`](../results/README.md), with provenance:

```bash
python -m evals.bench_report \
    --backend vllm_paged_attention=http://localhost:8080/v1/chat/completions \
    --backend triton_trt_llm=http://localhost:8081/v1/chat/completions \
    --model meta/llama3-8b-instruct \
    --requests 500 --concurrency 32 \
    --output results/baseline_bench_2026-09.json
```

What it does that the earlier tooling did not:

| Property | `evals/benchmark.py` | `evals/bench_report.py` |
|---|---|---|
| Metric | total request latency | **time to first token**, from a streamed response |
| Backends per run | one endpoint | **one or more named backends**, compared under one workload |
| Provenance | none | commit SHA (with `-dirty` when the tree is), the exact quoted command, package versions, CUDA driver, detected GPU |
| Missing provenance | n/a | **refuses to write the report** |
| Raw output | not written | per-request JSONL under `raw_output_path` |
| `key_findings` | n/a | derived from the measurements; a hand-written percentage that disagrees with its own numbers is rejected |
| p99 | `sorted[int(0.99*n)]`, which returns the maximum for n=100 | linear-interpolated percentile |

Fields it will not invent: `gpu_util_avg_pct` is written as `null` unless read
from DCGM or `nvidia-smi` during the run, and `--prompt-tokens-avg` defaults to
0 rather than guessing a tokenizer's count. `hardware` is detected from
`nvidia-smi`; on a host without a GPU the run records that it "did not measure
GPU serving" and, absent `--hardware`, refuses to write at all.

`tests/test_bench_report.py` covers this, including a test that every JSON in
`results/` satisfies the schema — so an unreproducible result cannot be
committed there unnoticed.

## Status

No baseline has been run on GPU hardware since the harness was written. Until
one has, **no performance number from this repository is safe to quote.**

## Rerunning the harnesses

Both harnesses drive the FastAPI gateway (`api/main.py`), which listens on port
`8080` and proxies to vLLM via `vllm/client.py`. Start the gateway and a backend
first — see [configuration.md](configuration.md).

### `evals/benchmark.py`

No CLI flags. Defaults are hardcoded in `run_benchmark()`: endpoint
`http://localhost:8080/v1/chat/completions`, `num_requests=100`,
`concurrency=10`, prompt `"What is machine learning?"`.

```bash
python -m evals.benchmark
```

Prints throughput (req/s), average latency, p99 latency, and failure count to
stdout. **Writes no file.**

### `evals/load_test.py`

Wraps `run_benchmark()` and persists a JSON report. Real flags, from its
`argparse` definition:

| Flag | Default |
|------|---------|
| `--endpoint` | `http://localhost:8080/v1/chat/completions` |
| `--requests` | `20` |
| `--concurrency` | `4` |
| `--prompt` | `Explain tensor parallelism briefly.` |
| `--output-dir` | `results/load_tests` |

```bash
python -m evals.load_test \
  --endpoint http://localhost:8080/v1/chat/completions \
  --requests 500 \
  --concurrency 32 \
  --output-dir results/load_tests
```

Run it from the repository root (it imports `evals.benchmark`). Output lands at
`<output-dir>/load_test_<UTC timestamp>.json` containing `endpoint`,
`num_requests`, `concurrency`, `successful`, `failed`, `avg_latency_ms`,
`p99_latency_ms`, `throughput_rps`, and `timestamp`. Files under
`results/load_tests/` are git-ignored.

### `triton/perf_analyzer.sh`

Runs NVIDIA `perf_analyzer` against a deployed Triton model. Configured by
environment variables, not flags:

```bash
MODEL_NAME=llama3 TRITON_URL=localhost:8000 CONCURRENCY=32 REQUEST_COUNT=500 \
  bash triton/perf_analyzer.sh
```

Defaults: `MODEL_NAME=llama3`, `TRITON_URL=localhost:8000`, `CONCURRENCY=4`,
`REQUEST_COUNT=100`. The script reports the 95th percentile and requires
`perf_analyzer` on `PATH` (it ships in the Triton SDK container).

### `vllm/client_test.py`

A standalone script that hits `http://localhost:8000/v1/chat/completions`
directly (bypassing the gateway) with hardcoded `CONCURRENCY = 10` and
`NUM_REQUESTS = 100`, printing average/p95 latency. Excluded from coverage in
`pyproject.toml`.
