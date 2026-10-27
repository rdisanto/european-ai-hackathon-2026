# History, validation and what was consolidated

## Timeline

| Date | What happened |
| --- | --- |
| 7 Oct | Starter kit (`curiosity-vllm`, now `bringup/`): stock vLLM 0.31 with Qwen3-0.6B on 1 GPU and on 2 nodes × 4 GPUs (TP4/PP2, no Ray). NCCL used InfiniBand with GPUDirect RDMA. Reached from a laptop through `ssh -L` via the login node. |
| 8 Oct | Qwen3-235B-A22B-FP8 on stock vLLM 0.31: 2 nodes, TP4/PP2, Marlin FP8 weight-only kernels, 2,048-token context, 4 sequences. Eager safetensors loading after slow mmap reads on Lustre. |
| 13 Oct | GLM-5.3-Flash. Stock vLLM had no usable sparse-MLA backend on A100, so it switched to the community SM80 backport (vLLM 0.13.1) and repaired TileLang in a separate overlay. |
| 17 Oct | Junghwa published the work as [Jules0315/Guardrails-vllm](https://github.com/Jules0315/Guardrails-vllm), with personal paths and accounts replaced by placeholders (`team_vllm`, `YOUR_PROJECT`). |
| 18 Oct | Common framework: one launcher, model profiles, shared runtime. GLM (job 61299807) and DeepSeek-V4.1 (job 61300901) passed 7/7 API checks at startup and in 2 SSH-tunnel repeats. |
| 19 Oct | Qwen3-235B moved onto the common runtime (job 61389137, 7/7 ×3). The API port base moved from 21000 to 31000 after it was found that API and master ports could collide for job suffixes 8000–8999. |
| 20 Oct | Copied into this repository and consolidated (below). |
| 20 Oct | Unified: `bringup/` folded into `serve/` (stock vLLM runtime, one-GPU profile, check suites) and `diagnostics/`; team registry, router job, laptop CLI for everything from `init` to `down`, opt-in cluster link to GKE. |
| 20 Oct | First runs of the unified launcher on Curiosity, from a fresh team folder: install of the stock runtime and Qwen3-0.6B, then `qwen06b` (job 61559851) with 7/7 checks and a streamed reply through the router. The first two attempts failed only the exact-`42` checks: the model answered `17 + 25 = 42`, so those checks now accept phrased answers. |
| 27 Oct | Curiosity B300 adaptation (`curiosity/` folder): a `docker` runtime kind in the catalog (image pulled on the GPU node, GPU mode probed, every engine flag and parser name verified against the image before the 704 GiB cold load), `glm53` rewritten to the GLM-5.3 743B recipe (TP=EP=4 on one node, KV fp8_e4m3, MTP, 1M context / 3 seqs), `dsv4flashablit` moved onto one B300 node with parser candidates resolved from the image registry, router upstream timeout raised to 1 h for 1M-token prefills, and site defaults retargeted to partition `hackathon` (no QoS, optional account). |

Startup times to a healthy API (cold caches, excluding queue): Qwen 7.0 min, GLM 13.3 min, DeepSeek 25.8 min.

## Compatibility problems that were solved

1. No A100 sparse-MLA backend for GLM in stock vLLM: use the pinned SM80 backport image
   `lazymio/vllm-backport@sha256:0fc33cd7…`. The image names no source commit, so a source-to-binary match
   was not independently verified.
2. Incomplete TileLang in that image: install the official 0.1.12 wheel (SHA-256 pinned) into a separate
   Python overlay. mHC was checked numerically against PyTorch on an A100.
3. Read-only cache locations: FlashInfer, TileLang, Triton and other caches are redirected to writable storage.
4. Too little contained `/tmp` during installs: use work storage.
5. Unix socket path-length errors: the runtime temp dir is bound to a short container path.
6. Host driver 535 vs CUDA 13 images: `VLLM_ENABLE_CUDA_COMPATIBILITY=1`.

## Validation scope

Evidence is in [`validation/`](../validation/validated_code.json): per-run startup times, all three API suites
per model, and checkpoint metadata digests. Absolute checkpoint paths are redacted; keys and raw logs were
never published.

Validated: one operator account, two nodes and eight A100s per model, short functional API checks. Every
server was cancelled afterwards, so no endpoint or key from these runs is live.

Not validated yet:

- Other team accounts running servers or connecting as clients (read access was granted, but no run was made).
- Sustained or workshop-scale load, long context (all profiles cap at 2,048 tokens), multimodal input.
- Running moderation models next to a main model.
- Unattended operation: anything other than a person's own SSH login. CINECA usernames are personal;
  persistent tunnels or proxy routes need CINECA's agreement.

## What the consolidation changed

The upstream repository kept every release as an immutable snapshot folder. In this repository, Git is
the history, so only the current version stays in the tree:

| Upstream path | Here |
| --- | --- |
| `releases/inference/2026-10-19-docs1` | `serve/`, execution files byte-identical |
| `releases/inference/2026-10-18`, `-18-docs1`, `-19-r2` | Removed: same code (only the port line differs), evidence kept in `validation/` |
| `qwen/` | `bringup/`, without `05`/`06` (superseded by the checkpoint install and the `qwen235b` profile); then unified, below |
| `releases/glm53/operator-2026-10-13` `02`, `05`, `03` | `install/runtime.sbatch` and `install/checkpoint.sbatch`, generalised to any runtime and profile |
| `releases/glm53/operator-…`, `candidates/glm53/…`, `releases/glm53-tools/…` | Removed: GLM-only launcher, per-user candidate and helpers, all replaced by `serve/` |
| `tools/register_runtime.py` | `install/register_runtime.py` |
| `GPU_VALIDATION.json` (two files) | `validation/validated_code.json` |
| Docs (8 files) | `README.md`, `docs/how-it-works.md`, this page |

Nothing is lost: the original repository stays unchanged at
[Jules0315/Guardrails-vllm](https://github.com/Jules0315/Guardrails-vllm/tree/298432c789de33d28a4cb500352a0714d8bac2c4)
(this folder was copied from commit `298432c`). For example, the GLM GPU compatibility preflight
(`preflight.py`, which checks the sparse-MLA backend and mHC numerics without loading weights) is at
`candidates/glm53/per-user-2026-10-13/preflight.py` there.

The install scripts were rewritten for the generic layout and have not yet been re-run in this form.
Expect to debug paths on the first fresh install. The team's existing installation on Curiosity does not
need them.

## Unifying `bringup/` into `serve/`

The starter kit (`bringup/`) and the framework (`serve/`) each had their own launcher, key handling and
endpoint files. They are now one path:

| `bringup/` | Now |
| --- | --- |
| `00_prepare.sbatch` (image and model download) | `cli.py install <profile>` (`install/runtime.sbatch`, `install/checkpoint.sbatch`) |
| `02_vllm_single.sbatch`, `04_vllm_multinode.sbatch`, `run_vllm.sh` | profiles on the `vllm-0.31.0` runtime, such as `qwen06b`, through `serve/` |
| `smoke_test.sh` | `test_endpoint.py` with the `minimal` (or any other) check suite |
| `common.sh` endpoint files and mkdir locks | `serve/registry.py`: same atomic locks, JSON records, stale-lock takeover via `squeue` |
| per-user key in `~/.config/curiosity-vllm/` | a fresh key per run, as `serve/` always did; the router holds them |
| `tunnel.sh` (follows an endpoint over SSH) | `cli.py forward` to the router, or `client.py tunnel` to one server |
| `01_probe.sbatch`, `03_nccl_check.sbatch`, `Dockerfile.ib` | `diagnostics/`, now using a registered runtime instead of `common.sh` |

`install/02_install_tilelang.sbatch` became the overlay step of `install/runtime.sbatch`, which installs whatever pinned
packages `serve/catalog.json` lists for a runtime. The bring-up kit's regression tests covered its locking,
smoke test and tunnel; the equivalent behaviour is now tested in `tests/test_common.py` and
`tests/test_router.py`. The old files are in this repository's history.
