# GPU Autoscaling

How this repo scales LLM serving pods on GPU utilization instead of CPU or memory.

## Canonical path

`kubernetes/hpa-dcgm.yaml` is the single supported HPA manifest.

An earlier duplicate, `autoscaling/hpa.yaml`, was removed. It targeted the same
`vllm-server` Deployment but referenced an external metric named
`dcgm_gpu_utilization`, which is not a metric the DCGM Exporter publishes
(`kubernetes/dcgm-exporter.yaml` runs the upstream exporter, whose GPU
utilization series is `DCGM_FI_DEV_GPU_UTIL`). It also pinned a different
threshold (`80`) and namespace (`serving`) and declared no scaling `behavior`,
so keeping both manifests meant two conflicting answers to the same question.

## Components

| File | Kind | What it does |
|------|------|--------------|
| `kubernetes/dcgm-exporter.yaml` | DaemonSet (`namespace: monitoring`) | Runs `nvcr.io/nvidia/k8s/dcgm-exporter:3.3.5-3.4.0-ubuntu22.04` on every GPU node, exposing metrics on container port `9400` with `prometheus.io/scrape: "true"`. Tolerates the `nvidia.com/gpu` taint. |
| `monitoring/prometheus/prometheus.yml` | Prometheus config | Scrapes job `dcgm_exporter` at `dcgm-exporter:9400`, plus `vllm-server:8000` and `triton-server:8002`. |
| `kubernetes/hpa-dcgm.yaml` | HorizontalPodAutoscaler (`autoscaling/v2`) | Scales the `vllm-server` Deployment between `minReplicas: 1` and `maxReplicas: 4` on the external metric `DCGM_FI_DEV_GPU_UTIL` with `target.type: AverageValue`, `averageValue: "70"`. |
| `kubernetes/vllm-deployment.yaml` | Deployment + Service | The `vllm-server` workload the HPA targets (`resources.limits.nvidia.com/gpu: 1`). |

## Why DCGM instead of CPU or memory

LLM inference is GPU-bound. A vLLM or Triton pod can sit at 90%+ GPU
utilization while its CPU request is nearly idle, because the Python process
mostly waits on CUDA kernels. Autoscaling on CPU therefore scales out late (the
GPU is already queuing requests before CPU crosses target) and can also scale
out spuriously when tokenization or HTTP handling spikes CPU while the GPU is
free. Memory is worse still: vLLM pre-allocates its KV cache to
`gpu_memory_utilization` (`0.90` in `configs/serving.yaml`), so GPU memory usage
is nearly constant regardless of load and carries no signal.

`DCGM_FI_DEV_GPU_UTIL` measures the fraction of time the GPU's execution engines
were busy, which tracks actual serving pressure and is the signal the HPA in
this repo consumes.

## Dependencies

1. **NVIDIA GPU Operator or device plugin** — required for `nvidia.com/gpu`
   resource limits to schedule.
2. **DCGM Exporter DaemonSet** — `kubectl apply -f kubernetes/dcgm-exporter.yaml`.
3. **Prometheus** — must scrape the exporter (see `monitoring/prometheus/prometheus.yml`).
4. **[Prometheus Adapter](https://github.com/kubernetes-sigs/prometheus-adapter)** —
   required. An `autoscaling/v2` HPA cannot read Prometheus directly; the adapter
   exposes `DCGM_FI_DEV_GPU_UTIL` through the `external.metrics.k8s.io` API. Without
   it the HPA reports `FailedGetExternalMetric` and never scales.

## Apply order

```bash
kubectl apply -f kubernetes/dcgm-exporter.yaml
kubectl apply -f kubernetes/vllm-deployment.yaml
kubectl apply -f kubernetes/hpa-dcgm.yaml

kubectl get hpa vllm-gpu-hpa
kubectl describe hpa vllm-gpu-hpa
```

Namespaces: `kubernetes/dcgm-exporter.yaml` pins `namespace: monitoring`, while
`hpa-dcgm.yaml` and `vllm-deployment.yaml` declare no namespace and land in
whatever namespace `kubectl` targets. The HPA and its `scaleTargetRef` Deployment
must be in the same namespace, so apply those two together.

## Tuning

- **`averageValue: "70"`** — headroom before saturation. Raising it toward 90
  packs GPUs more tightly but queues requests during the pod cold start; lowering
  it scales out earlier at higher idle cost. Cold start dominates: a vLLM pod
  loads weights and warms the KV cache before it can serve, which is why
  `kubernetes/vllm-deployment.yaml` sets `livenessProbe.initialDelaySeconds: 120`.
- **`scaleUp.stabilizationWindowSeconds: 60` / +1 pod per 60s** — deliberately
  conservative because each new replica claims a whole GPU.
- **`scaleDown.stabilizationWindowSeconds: 300` / -1 pod per 120s** — long, to
  avoid dropping a warm replica during a traffic lull and paying model load
  latency again.
- **`maxReplicas: 4`** — must not exceed the schedulable GPUs in the cluster;
  extra replicas stay `Pending` on `nvidia.com/gpu` and the HPA cannot recover
  from that on its own.
- **Metric selector `gpu: "0"`** — scopes the external metric to a single GPU
  index. On multi-GPU nodes, adjust or drop this selector so the average reflects
  every device the workload uses.

## Ray Serve autoscaling is separate

`ray_serve/deployment.py` and `ray_serve/autoscaling_config.yaml` scale replicas
inside Ray on `target_num_ongoing_requests_per_replica: 8` (min 1, max 4). That
is a request-queue signal owned by the Ray Serve controller, not the Kubernetes
HPA, and the two are not wired together in this repo.
