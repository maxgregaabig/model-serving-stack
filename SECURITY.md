# Security Policy

## Supported versions

This is a reference project, not a released product. Only the current `main`
branch receives fixes.

| Version | Supported |
|---------|-----------|
| `main` | Yes |
| Tagged releases prior to `main` | No |

## Reporting a vulnerability

Please do **not** open a public issue for a security problem.

- Preferred: open a [private security advisory](https://github.com/TylrDn/model-serving-stack/security/advisories/new)
  on this repository.
- Alternative: email taylorboonedean@gmail.com with `SECURITY` in the subject.

Include the affected file or endpoint, reproduction steps, and the impact you
believe it has. Expect an acknowledgement within seven days. Because this is a
personally maintained reference repository, there is no formal SLA for a fix.

## Scope

In scope: the code and manifests in this repository — the FastAPI gateway
(`api/`), the backend clients (`vllm/`, `triton/`, `ray_serve/`, `bentoml/`), the
Kubernetes and Compose manifests, and the CI workflow.

Out of scope: vulnerabilities in upstream projects (vLLM, Triton Inference
Server, Ray, BentoML, DCGM Exporter, Prometheus, Grafana) and in the container
images this repo references. Report those to their maintainers.

## Deployment notes

This repository is a reference architecture and its defaults are not hardened
for internet-facing use. Before deploying anything derived from it:

- The gateway (`api/main.py`) implements **no authentication, authorization, or
  rate limiting**. `vllm/client.py` sends the placeholder API key `"EMPTY"` to
  the vLLM server. Put an authenticating proxy in front of any exposed endpoint.
- `kubernetes/ingress.yaml` exposes `/vllm` and `/triton` with no auth
  annotations and no TLS configuration.
- The root `docker-compose.yml` sets `GF_SECURITY_ADMIN_PASSWORD=admin` for
  Grafana. Change it.
- `kubernetes/dcgm-exporter.yaml` runs the exporter with `hostNetwork: true`,
  `hostPID: true`, `runAsUser: 0` and the `SYS_ADMIN` capability, as the
  upstream DaemonSet requires. Restrict which nodes it schedules to.
- Never commit a filled-in `.env`. `.gitignore` excludes it; keep it that way,
  and supply Hugging Face and NGC tokens through Kubernetes secrets (see
  `hf-secret` in `kubernetes/vllm-deployment.yaml`).
- Container images are pinned loosely (`vllm/vllm-openai:latest`,
  `grafana/grafana:latest`). Pin digests before any real deployment.
