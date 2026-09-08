# Benchmark results

Committed artifacts in this directory back the numbers quoted in the README and
[`docs/benchmarks.md`](../docs/benchmarks.md). A number that is not backed by a
file here should not be quoted anywhere in the repo.

## Contents

| File | Description |
|------|-------------|
| _(none)_ | No reproducible benchmark has been committed yet. |

`results/unverified/` holds results that lack the provenance below. They are
**not** evidence and are quoted nowhere — see
[`unverified/README.md`](unverified/README.md).

`results/load_tests/` is git-ignored; it is the default `--output-dir` for
`python -m evals.load_test` and holds unreviewed local runs.

## Producing a result

`python -m evals.bench_report` emits this schema with provenance read from the
running machine, and refuses to write a report that is missing any required
field. See [`../docs/benchmarks.md`](../docs/benchmarks.md).

`tests/test_bench_report.py::TestCommittedResults` validates every `*.json` in
this directory against the rules below, so a result that cannot be reproduced
cannot land here unnoticed.

## Required schema for new runs

Every new committed result must include provenance, not just measurements.
Record the fields exactly as observed — never estimate or backfill a version.

```jsonc
{
  "benchmark_date": "YYYY-MM-DD",
  "model": "provider/model-id",
  "hardware": "GPU model + memory, vCPU count, host RAM",
  "git_sha": "full commit SHA of the repo state under test",
  "command": "the exact invocation, including every flag",
  "software_versions": {
    "vllm": "x.y.z",
    "tensorrt_llm": "x.y.z",
    "triton_server_image": "nvcr.io/nvidia/tritonserver:TAG",
    "ray": "x.y.z",
    "cuda_driver": "xxx.yy",
    "python": "3.11.x"
  },
  "workload": {
    "concurrency": 0,
    "request_count": 0,
    "prompt_tokens_avg": 0,
    "output_tokens_avg": 0
  },
  "raw_output_path": "results/raw/<run-id>/",
  "backends": {
    "<backend_key>": {
      "ttft_p50_ms": 0,
      "ttft_p99_ms": 0,
      "throughput_tokens_per_sec": 0,
      "requests_per_sec": 0,
      "error_rate_pct": 0,
      "gpu_util_avg_pct": 0,
      "notes": "configuration specifics for this backend"
    }
  },
  "key_findings": ["short, arithmetic-checkable statements"]
}
```

## Rules

1. **Provenance is mandatory.** A result without `git_sha`, `command`, and
   `software_versions` is not reproducible and should not be committed.
2. **Commit the raw output too**, under `raw_output_path`, so aggregates can be
   recomputed.
3. **Keep `key_findings` checkable.** A percentage must follow from the numbers
   in the same file (142 -> 98 ms is a 31% reduction, so write 31%).
4. **One file per run**, named `<name>_bench_<YYYY-MM>.json`. Never edit a
   published result in place; add a new file.
5. **Keep the README table in sync** with whatever the current baseline file
   says.
