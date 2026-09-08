# Unverified results

**Nothing in this directory is evidence.** These files record numbers whose
provenance was never captured, so they cannot be reproduced from this
repository and no claim anywhere should rest on them.

They are kept rather than deleted so the history is legible: the measurements
were presumably taken, and discarding them silently would be its own kind of
dishonesty. But they do not satisfy rule 1 in [`../README.md`](../README.md) —
"A result without `git_sha`, `command`, and `software_versions` is not
reproducible and should not be committed" — so they do not live in `results/`,
and `tests/test_bench_report.py` enforces that boundary.

## Contents

| File | Why it is here |
|------|----------------|
| `baseline_bench_2026-06.json` | Dated 2026-06-09, comparing vLLM, Triton TRT-LLM and Ray Serve on a single A10G. Records no `git_sha`, no `command`, no `software_versions`, no `workload` block and no `raw_output_path`. No committed tool emitted its schema at the time it was written. The code state, the invocation, and the vLLM / TensorRT-LLM / Triton / CUDA versions that produced it are all unknown and now unrecoverable. |

## What replaces it

`python -m evals.bench_report` emits the schema in [`../README.md`](../README.md)
with provenance captured from the running machine, and refuses to write a report
that is missing any of it. Producing a replacement baseline needs a GPU host:

```bash
python -m evals.bench_report \
    --backend vllm_paged_attention=http://localhost:8080/v1/chat/completions \
    --backend triton_trt_llm=http://localhost:8081/v1/chat/completions \
    --model meta/llama3-8b-instruct \
    --requests 500 --concurrency 32 \
    --output results/baseline_bench_YYYY-MM.json
```

Until that has been run on real hardware, this project has **no** benchmark
numbers that are safe to quote.
