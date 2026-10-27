# The Slurm jobs

`cli.py` only orchestrates Slurm batch jobs. This page describes each job for people who prefer to run
them by hand: what it needs, which resources it asks for, what it executes and what it leaves behind.

## Before you start

Work on a Leonardo login node in a folder that contains `leonardo/serve`, `leonardo/install` and
`leonardo/diagnostics`: a clone of this repository, or a copy that `cli.py` uploaded
(`~/.leonardo-inference/releases/<hash>/`, which contains `serve/`, `install/` and `diagnostics/` directly;
there, drop the `leonardo/` prefix from the paths below). Then:

```bash
export SBATCH_ACCOUNT=<Slurm account>   # used by sbatch and by the prepare steps below
export BASE=$WORK/<team folder>         # the team folder; the install and NCCL jobs read it
```

Run `sbatch` from the repository root (or set `LEONARDO_RELEASE` to the `leonardo/` folder): the install
and diagnostic scripts find their code from the directory you submitted from. Logs go to
`<job name>-<job id>.out` in that directory unless you pass `--output`.

The team folder itself is not made by a job. `python3 leonardo/serve/team.py init --base "$BASE" --account
<Slurm account>` creates it (subfolders and `inference-site.json`), and `team.py share` grants access; see
`team.py --help`.

| Job name | Script | Queue and resources | Needs |
| --- | --- | --- | --- |
| `install-runtime` | `install/runtime.sbatch <runtime>` | `lrd_all_serial`, 4 CPUs, 30 GB, 2 h | `BASE` |
| `install-checkpoint` | `install/checkpoint.sbatch <profile>` | `lrd_all_serial`, 4 CPUs, 30 GB, 4 h | `BASE`, the profile's runtime installed |
| `install-link-client` | `install/link_client.sbatch` | `lrd_all_serial`, 1 CPU, 10 min | `BASE` |
| `<profile>-common` | `serve/serve_model.sbatch` | GPU partition; nodes, GPUs, CPUs, memory and time from the profile | a run folder made by `inference.py prepare` |
| `leonardo-router` | `serve/router.sbatch` | `lrd_all_serial`, 2 CPUs, 8 GB, 4 h, renews itself | a router folder made by `router.py prepare` |
| `probe` | `diagnostics/probe.sbatch` | `boost_usr_prod`, debug QoS, 1 GPU, 20 min | nothing |
| `reach` | `diagnostics/reach.sbatch` | `lrd_all_serial`, 1 CPU, 10 min | `TARGET` (optional `PROBE_PROXY`, `LINK_URL`) |
| `nccl-check` | `diagnostics/nccl_check.sbatch` | `boost_usr_prod`, debug QoS, 2 nodes × 4 GPUs, 15 min | `BASE`, optional `RUNTIME` |

`lrd_all_serial` is the CPU queue with internet access; GPU nodes have none. Its jobs have been seen running
on login nodes.

## Install a runtime: `install/runtime.sbatch`

```bash
sbatch leonardo/install/runtime.sbatch vllm-0.31.0      # or vllm-backport-sm80-0.13.1
```

Argument: a runtime id from `serve/catalog.json`. Steps:

1. Exits at once if `$BASE/runtimes/<runtime>/runtime.json` exists.
2. `singularity pull` of the catalog's image, pinned by digest, into `$BASE/staging/<runtime>/image.sif`
   (build caches in the same staging folder). This is the slow part: 20-60 minutes for an 8 GB image.
3. For each of the catalog's `overlay_packages`: `pip download` of the pinned wheel inside the image,
   SHA-256 check, `pip install --target` into a separate `python-overlay/` folder, and an import check.
   Stock vLLM has none; the backport image gets TileLang 0.1.12.
4. `install/register_runtime.py` hardlinks (or copies) the image and copies the overlay to
   `$BASE/runtimes/<runtime>/`, and writes `runtime.json` with their hashes. The staging folder is removed.

Last line on success: `RUNTIME_READY <path>`.

## Download model weights: `install/checkpoint.sbatch`

```bash
sbatch leonardo/install/checkpoint.sbatch qwen06b
sbatch --dependency=afterok:<runtime job id> leonardo/install/checkpoint.sbatch qwen06b   # right after the runtime job
```

Argument: a profile from `serve/profiles/`. It reads `model_id`, `revision`, `checkpoint` and `runtime` from
the profile, then runs Hugging Face's `snapshot_download` inside the runtime's image
(`$BASE/runtimes/<runtime>/image.sif`) into `$BASE/checkpoints/<checkpoint>/`. Before downloading it checks
that the shared file system has room for the weights plus 100 GiB; afterwards it checks every shard's
size and writes `download_manifest.json`. A profile without `revision` gets the current one, printed as
`PIN_REVISION <sha>`. `HF_TOKEN` from your environment is passed on for gated models. Last line on
success: `CHECKPOINT_READY <path> revision <sha> shards <n>`.

## Install the cluster link client: `install/link_client.sbatch`

```bash
sbatch leonardo/install/link_client.sbatch
```

Downloads the chisel release named in `serve/catalog.json`, checks both SHA-256 sums (compressed and
binary) and installs it as `$BASE/tools/chisel-<version>`. Only needed for the cluster link.

## Run a model server: `serve/serve_model.sbatch`

Two steps, because the private run folder and its API key must exist before the job is queued.

1. **Prepare, on the login node:**

   ```bash
   RUN=$(python3 leonardo/serve/inference.py prepare qwen06b --base "$BASE")
   ```

   This checks the release manifest, the profile, the site file, the runtime and every checkpoint file, and
   creates `$BASE/users/<login>/runs/<profile>-<random>/` (mode 700) with a copy of `serve/`,
   `RUN_CONFIG.json` (every resolved path) and a fresh `.runtime/api_key`. It prints only the folder.

2. **Submit, from that folder:**

   ```bash
   cd "$RUN"
   python3 inference.py submit --dry-run                    # prints the sbatch command
   python3 inference.py submit                              # or run that command yourself
   ```

   The command takes nodes, GPUs, CPUs, memory and time from the profile, and account, partition and
   QoS from the site file (`SBATCH_ACCOUNT`, `--qos` and `--time` override them). It always adds
   `--signal=B:TERM@90` and `--chdir=<run folder>`.

Inside the job (`serve_model.sbatch`):

1. `inference.py preflight` re-checks the assets.
2. Ports from the job id: distributed master `29000 + id % 1000`, API `31000 + id % 10000`.
3. `srun` starts `serve/node.py` once per node. It runs the profile's pre-start hooks, if any
   (`hooks/nccl_ep.py`, `hooks/warm_engram.py`), then `singularity exec --nv --cleanenv --no-home --contain`
   with the runtime and checkpoint bound read-only, the private cache and run folder writable, and
   `vllm serve <checkpoint> --served-model-name … --tensor-parallel-size … --pipeline-parallel-size …
   <profile engine_args>` plus `--host 0.0.0.0 --port <API port>` on the first node and `--headless` on the
   others. The API key reaches vLLM as `VLLM_API_KEY`.
4. In parallel, `serve/lifecycle.py` claims the served model name in `$BASE/endpoints/` (a second live
   server with that name makes the job fail here), waits for `/health`, and runs `serve/test_endpoint.py`
   with the profile's check suite. Only if all checks pass does it write `.runtime/endpoint.json`,
   `evidence/`, the registry record `$BASE/endpoints/<served model>.json` (address only), and print
   `INFERENCE_READY`. Then it checks `/health` every 5 seconds and ends the job after 3 failures.

On exit (time limit, `scancel` or failure) the endpoint file, registry record and claim are removed.
Log: `<run folder>/<profile>-common-<job id>.out`. Stop it with `scancel <job id>`.

## Run the router: `serve/router.sbatch`

Also two steps:

```bash
python3 leonardo/serve/router.py prepare --base "$BASE" --hours 12   # prints a "cd … && sbatch …" line
```

Prepare creates `$BASE/users/<login>/router/` with a copy of `serve/` and `ROUTER.json` (settings and the
end time), and `$BASE/users/<login>/router_token` if it does not exist. Run the printed line; it submits
`router.sbatch` from that folder with `--dependency=singleton`, so only one router job of yours runs at a
time.

Inside the job, on port `41000 + id % 10000`:

1. `router.py next` queues the successor job (the same command), unless the end time is near. It waits in
   the queue until this job ends.
2. `router.py serve` answers HTTP on all interfaces. It requires the router token, finds your ready
   servers in `$BASE/users/<login>/runs/` (only those whose job `squeue` still lists), and forwards each
   request to the server named in its `model` field with that server's key. With `cluster_link.enabled` in
   the site file it also starts the chisel client (`$BASE/tools/chisel-<version>`, verified and copied at
   prepare time).

Log: `$BASE/users/<login>/router/leonardo-router-<job id>.out`. Stop all of them, the queued successor
included, with `scancel --name=leonardo-router --user=$USER`.

## Checks: `diagnostics/`

```bash
sbatch leonardo/diagnostics/probe.sbatch                                  # prints lrdnXXXX:PORT
TARGET=lrdnXXXX:PORT PROBE_PROXY=http://… sbatch leonardo/diagnostics/reach.sbatch
RUNTIME=vllm-0.31.0 sbatch leonardo/diagnostics/nccl_check.sbatch
```

- **probe:** prints the GPU node's identity, network interfaces, routes, InfiniBand devices, GPU topology
  and outbound internet, then serves a test page with `python3 -m http.server` for 15 minutes.
- **reach:** from the CPU queue, `curl`s the probe's page (`TARGET`), the internet directly and through
  `PROBE_PROXY`, and the cluster link server's `/health` (`LINK_URL`).
- **nccl_check:** 8 tasks on 2 nodes, each running `diagnostics/nccl_check.py` in the runtime's image through
  `nccl_task.sh`: an all-reduce correctness check, then bandwidth within a node and across nodes, and a
  summary of the transport NCCL chose (`NET/IB` is InfiniBand; `NET/Socket` means TCP fallback).
