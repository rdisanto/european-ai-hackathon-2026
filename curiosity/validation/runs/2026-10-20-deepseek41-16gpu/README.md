# Historical DeepSeek 16-GPU evidence

Leonardo job `61481897` completed on 20 October 2026 with four nodes, sixteen
A100 GPUs, TP8/PP2, expert parallelism and GPU-resident Engram. It used the
earlier launcher, not this repository's unified CLI/router.

- `model-profile.json`: the collected profile, including its original ID, model
  name and pre-run validation wording. It is historical evidence, not an active
  selectable profile.
- `release-manifest.json`: hashes of the earlier launcher source files. These
  are provenance; they are not claims about the current `serve/` files.
- `startup-checks.json`: pass flags for all seven startup API checks. Response
  bodies and private checkpoint paths were removed from this export.
- `cpu-regression.json`: actual installed allocator results before/after the
  pinned fix: empty and partial projected groups omit remote layers, ordinary
  local allocation is unchanged, and group slots remain stable.
- `load-results.json`: numerical results for two finite bursts at each of
  1, 4, 8, 16, 32, 50 and 100 concurrent requests; 14 bursts, 422 successes,
  zero failures, 128 output tokens per request. No credentials, response text
  or private host paths are included.
- `benchmark_serving.py`: the benchmark source matching `source_sha256` in
  `load-results.json` (retained for reproducibility, not invoked by CI).
- `summary.json`: resource, patch-hash and validation-scope summary. The patch
  modules retain the exact bytes used in this historical runtime.

The load client was a CPU-only container on the inference head node, calling
localhost HTTP. This is not an SSH/router benchmark or a hardware/network-matched
model ranking. It is not a steady-state capacity measurement. The allocation
finished with Slurm `COMPLETED / 0:0`; stopping the inference step after the
benchmark was intentional cleanup. The endpoint and session key were removed.

The current integration still needs GPU validation through its new runtime
installer, profile/model name and unified CLI/router. Do not add this record as
an identical-current-code run in `validation/validated_code.json`.
