# Curiosity inference

## What is this for?

The team's app sends every model call through agentgateway, which today forwards it to Vertex AI (Gemini).
This folder lets a team run **open-weight models on CINECA's Curiosity supercomputer** and use them through
the same OpenAI-compatible API: large models for the main work, and small one-GPU models such as
moderation classifiers. It is a reusable template: everything below starts from an empty team folder.

| Profile | Model | Served as | Hardware | Cold start | Status |
| --- | --- | --- | --- | --- | --- |
| `qwen235b` | Qwen3-235B-A22B-FP8 | `qwen3-235b-a22b-fp8-curiosity` | 2 nodes, 8 A100 | 7 min | GPU-validated |
| `glm53` | GLM-5.3 | `glm-5.3` | 1 node, 4 B300 (docker, curiosity) | 20-40 min cold | B300 diagnostic recipe; GPU validation pending |
| `dsv4flashablit` | DeepSeek-V4-Flash-0731-Abliterated-FP8 | `deepseek-v4-flash-ablit` | 1 node, 4 B300 (docker, curiosity) | cold | B300 recipe; GPU validation pending |
| `deepseek41` | DeepSeek-V4.1-Flash-1017 | `deepseek-v4.1-flash-1017-curiosity` | 2 nodes, 8 A100 | 26 min | GPU-validated |
| `deepseek41-16gpu` | DeepSeek-V4.1-Flash-1017 | `deepseek-v4.1-flash-1017-16gpu-curiosity` | 4 nodes, 16 A100 | 15 min | Earlier launcher/load-tested; unified CLI rerun pending |
| `qwen06b` | Qwen3-0.6B | `qwen3-0.6b-curiosity` | 1 A100 | 1.5 min | GPU-validated, current launcher |

The large A100 profiles run with a 2,048-token context and 4 concurrent sequences (DeepSeek: 2) for up to two
hours per job. The curiosity B300 docker profiles (`glm53`, `dsv4flashablit`) run one node / 4 GPUs with the
full native context — GLM-5.3 at 1M tokens (3 concurrent sequences) and DeepSeek-V4-Flash-ablit at 524,288 —
through the B300 diagnostic recipe (rootless docker, image-by-digest, every flag verified before the cold load)
and are designed to be served together by a single router. The separate sixteen-GPU DeepSeek profile has 8
sequence slots and GPU-resident Engram; see [its guide and patch details](docs/deepseek16.md). The large models were validated with an earlier version of the launcher; see
[Validation status](#validation-status).

The serving framework was built by Junghwa Lee ([@Jules0315](https://github.com/Jules0315)) on top of the
team's starter kit, and copied from [Jules0315/Guardrails-vllm](https://github.com/Jules0315/Guardrails-vllm)
at commit `298432c`.

## How does it work?

Curiosity is a Slurm cluster: GPUs come as time-limited jobs, and GPU nodes have no internet. You run
`curiosity/cli.py` on your laptop; it logs in to Curiosity with your own account and submits everything as
Slurm jobs:

- **Install jobs** download container images and model weights into a **team folder** in `$WORK`, on a
  CPU queue that has internet.
- A **model server** job runs vLLM on GPU nodes. It gets a fresh API key and is announced to the team only
  after a set of API checks has passed.
- Your **router** job is a single entry point to all of your model servers. You reach it from your laptop
  with `cli.py forward` (an SSH port forward); later, the app in Kubernetes reaches it through the
  **cluster link**, an outbound connection from Curiosity to the cluster that stays off until approved.

Keys and runs stay in your private folder `$BASE/users/<login>/`; the rest of the team folder is shared with
the teammates you name. [docs/how-it-works.md](docs/how-it-works.md) has the full picture with a diagram,
and every step in detail.

## Installation

> Newcomer? Start with [docs/QUICKSTART.md](docs/QUICKSTART.md): shell setup,
> launching a model and reaching the router in a few commands.

### What you need

- **On your laptop:** an SSH client and Python 3.7 or newer. No packages, no virtual environment: the CLI
  and everything it runs on Curiosity use only the Python standard library.
- **On Curiosity:** your own CINECA login (logins are personal and must not be shared), membership in a
  project allocation (the Slurm *account*), and a place for the team folder in that project's `$WORK`.
  Ask your mentor or admin for the account name and where the team folder should go. Nothing needs to be
  installed there; the CLI uses the system `python3`, Slurm and Singularity.

### Configure the CLI

All commands in this README run from the repository root on your laptop. Set these once per shell (or put
them in an `.envrc` that stays out of Git):

```bash
export CURIOSITY_USER=<your login>
export CURIOSITY_BASE='$WORK/<team folder>'          # single quotes: $WORK is expanded on Curiosity
export CURIOSITY_HOST=login02-ext.curiosity.cineca.it # the login node you use or were assigned
```

`CURIOSITY_HOST` defaults to `login01-ext.curiosity.cineca.it`. The first command asks for your CINECA
authentication; later commands reuse that SSH connection for 30 minutes.

### Set up the team folder

Once per team:

```bash
./curiosity/cli.py init --account <Slurm account>   # creates the folder and its settings file
./curiosity/cli.py share <login> <login> ...       # give teammates access
./curiosity/cli.py status                          # "Installed: runtimes none; checkpoints none"
```

`share` can be repeated whenever someone joins, and `share --revoke <login>` takes access away. Your
teammates then set `CURIOSITY_BASE` to the same folder and can use every command below; each of them gets
their own personal folder in `users/`.

### Optional: connect the GKE cluster

Prepared, but **off until the organisers approve** the outbound connection, because the terms forbid
working around security controls. First find out what the router's CPU queue can reach (see
[Troubleshooting](#troubleshooting)): GPU-node ports, and the internet directly or through the hackathon
proxy. Then:

1. Create two secrets in GKE: the chisel server's key and the one user it accepts (which may open only
   port 9000), and your router token.

   ```bash
   SERVER_KEY=$(docker run --rm jpillora/chisel:1.13.0 server --keygen -)
   LINK_PASSWORD=$(openssl rand -hex 24)
   kubectl create secret generic curiosity-link --from-literal=server.key="$SERVER_KEY" \
     --from-literal=users.json="{\"curiosity:$LINK_PASSWORD\": [\"^R:0.0.0.0:9000\$\"]}"
   ./curiosity/cli.py router token | kubectl create secret generic curiosity-router --from-file=token=/dev/stdin
   ```

2. Add `components: [../curiosity]` to `infrastructure/gke/kustomization.yaml` and deploy. This adds the
   chisel server under `https://european-ai-hackathon-2026.codify.ch/curiosity-link`, the `curiosity-router`
   Service and one agentgateway route per model. Read the server's fingerprint with
   `kubectl logs deploy/curiosity-link | grep Fingerprint`.
3. In `$BASE/inference-site.json`, set `cluster_link.enabled` to `true`, `server` to that URL,
   `fingerprint`, and `proxy` if the route needs the hackathon proxy.
4. Install the chisel client and restart your router with the link password:

   ```bash
   ./curiosity/cli.py install --link-client
   printf 'curiosity:%s\n' "$LINK_PASSWORD" > /tmp/link_auth
   ./curiosity/cli.py router up --restart --link-auth-file /tmp/link_auth && rm /tmp/link_auth
   ```

The backend then uses the "Served as" names from the table above as model names in its agentgateway calls.

## Usage

### Commands

`./curiosity/cli.py --help` lists the commands, and `./curiosity/cli.py <command> --help` shows a command's
options and the scripts it runs on Curiosity. Each command uploads the code from your checkout first, if
Curiosity does not have that exact version yet.

| Command | Parameters | What it does |
| --- | --- | --- |
| `init` | `--account` (required); `--partition`, `--qos`, `--cpu-partition` to change the defaults; `--force` | Creates the team folder with its subfolders and the settings file `inference-site.json` (Slurm account, partitions, QoS, router and cluster link settings), plus your private personal folder. Run once per team; it never overwrites existing settings unless you add `--force`. |
| `share` | `LOGIN...`, `--revoke` | Gives the named Curiosity users read and write access to the team folder, including everything created later, so they can install models and start their own servers. Personal folders stay private. `--revoke` takes the access away again. |
| `install` | `PROFILE...`, `--link-client` | Submits CPU-queue jobs that download what the profiles need: the container image (runtime) and the model weights (checkpoint). Skips what is already installed; a weights job waits until its runtime job has finished. `--link-client` also downloads the chisel client for the cluster link. |
| `up` | `PROFILE`; `--qos`, `--time`, `--reservation` | Checks that the profile's runtime and weights are installed, creates a private run folder with a new API key and submits the model server job. The job starts vLLM, runs the profile's API checks and announces the server only if they pass. `--qos boost_qos_dbg --time 00:30:00` makes a short test job that usually starts sooner. |
| `status` | | Shows installed runtimes and checkpoints, running install jobs, your model servers (`starting`, or `READY` with node and port), your router, and every ready server in the team registry. |
| `logs` | `TARGET`; `--lines`, `--follow` | Prints the end of a job's log. `TARGET` is a job id (any job: install, server, router, check), a profile or served model name (your newest run), or `router`. |
| `down` | `TARGET` | Cancels your model servers for a profile, served model name or job id. `down router` stops your router, including its queued successor. |
| `router` | `up`, `down` or `token`; `--hours`, `--restart`, `--token-file`, `--link-auth-file` | `up` starts your router job, which renews itself every 4 hours until `--hours` (default 12) have passed; `--restart` replaces a running router. `token` prints the bearer token that callers of the router need; it is created on first use, and `--token-file` installs a given one instead. `--link-auth-file` stores the cluster link password. |
| `forward` | `--local-port` (default 18000) | Opens an SSH port forward from `127.0.0.1:<port>` on your laptop through the login node to your router, until you press Ctrl-C. |
| `diagnose` | `probe`, `reach` or `nccl`; `--target`, `--proxy`, `--link-url`, `--runtime` | `probe`: a short GPU job that prints the node's network, InfiniBand and GPU setup and serves a test page. `reach`: a CPU-queue job that tests whether it reaches a GPU node (`--target` from a probe), the internet, a proxy (`--proxy`) and the cluster link server (`--link-url`). `nccl`: a two-node job that measures GPU communication over InfiniBand. |
| `sync` | | Only uploads the code. |

### Install a model

Before a model can run, its runtime and weights must be in the team folder. This runs on the CPU queue and
can take a while:

```bash
./curiosity/cli.py install qwen06b       # submits two jobs; the weights job waits for the runtime job
./curiosity/cli.py status                # lists the install jobs while they run
./curiosity/cli.py logs <job id> -f      # job ids are printed by install and status
```

The runtime job pulls the container image and converts it (20-60 minutes for an 8 GB image; the
`xattr` warnings in its log are harmless), and its log ends with `RUNTIME_READY`. The weights job ends with
`CHECKPOINT_READY`; it takes minutes for `qwen06b` and hours for the large models (several hundred GB).
A profile without a pinned model `revision` downloads the current one and prints `PIN_REVISION <sha>` in
its log; put that sha into the profile. `install` accepts several profiles (`install qwen06b qwen235b`),
skips what is already there, and installs a runtime only once for all profiles that use it. For gated
models on Hugging Face, put `export HF_TOKEN=...` into your `~/.bashrc` on Curiosity before installing.

### Start and stop model servers

```bash
./curiosity/cli.py up qwen06b --qos boost_qos_dbg --time 00:30:00   # a quick test on the debug queue
./curiosity/cli.py up qwen235b                                      # the default queue: up to 2 hours
./curiosity/cli.py status                                           # wait for READY
./curiosity/cli.py logs qwen235b --follow                           # the server's job log
./curiosity/cli.py down qwen235b
```

A server shows `READY` once vLLM is up and its API checks have passed; until then it is `starting`. It
stops at its time limit (2 h on the default QoS `boost_qos_lprod`, 30 min on `boost_qos_dbg`); run `up`
again to replace it. Only one server per model name can run in a team folder at a time; a second one stops
within seconds. Coordinate large servers with the team: each takes two full nodes.

### Call the models

Start your router and forward a laptop port to it:

```bash
./curiosity/cli.py router up --hours 12                              # renews itself every 4 hours
./curiosity/cli.py router token > ~/.config/curiosity-router-token   # your bearer token
./curiosity/cli.py forward                                          # keep this running
```

In a second terminal:

```bash
TOKEN=$(cat ~/.config/curiosity-router-token)
curl http://127.0.0.1:18000/v1/models -H "Authorization: Bearer $TOKEN"
curl http://127.0.0.1:18000/v1/chat/completions -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-0.6b-curiosity","messages":[{"role":"user","content":"Hi"}],
       "chat_template_kwargs":{"enable_thinking":false}}'
```

Any OpenAI client works with base URL `http://127.0.0.1:18000/v1` and the token as API key. The router
reaches only servers started with your login. Expect a short gap every four hours, when one router job
hands over to the next. With the cluster link enabled, the app calls the same model names through
agentgateway instead.

### What runs on Curiosity

The CLI only orchestrates Slurm jobs; each is an ordinary batch script in this repository:

| Job name | Script | Queue and resources | Submitted by | What it runs |
| --- | --- | --- | --- | --- |
| `install-runtime-<runtime>` | `install/runtime.sbatch <runtime>` | CPU queue, 4 CPUs, 2 h | `install` | `singularity pull` of the image in `serve/catalog.json`, `pip install` of pinned extra packages into a side folder, `install/register_runtime.py` |
| `install-checkpoint-<checkpoint>` | `install/checkpoint.sbatch <profile>` | CPU queue, 4 CPUs, 4 h, after the runtime job | `install` | Hugging Face `snapshot_download` inside the runtime's container |
| `install-link-client` | `install/link_client.sbatch` | CPU queue, 10 min | `install --link-client` | download and checksum of the chisel client |
| `<profile>-common` | `serve/serve_model.sbatch` | GPU partition; nodes, GPUs, memory and time from the profile | `up` | `serve/node.py` once per node (vLLM in the container) and `serve/lifecycle.py` (API checks, registry) |
| `curiosity-router` | `serve/router.sbatch` | CPU queue, 2 CPUs, 4 h, queues its own successor | `router up` | `serve/router.py` (and the chisel client when the cluster link is on) |
| `probe`, `reach`, `nccl-check` | `diagnostics/*.sbatch` | GPU debug QoS, CPU queue, 2 GPU nodes on debug QoS | `diagnose` | shell checks; `diagnostics/nccl_check.py` on 8 GPUs |

`serve/team.py` (on Curiosity) builds the `sbatch` commands. [docs/slurm-jobs.md](docs/slurm-jobs.md)
describes each job in detail (inputs, resources, what it executes, what it writes) and how to submit it
with plain `sbatch`, without the CLI.

### Troubleshooting

- `./curiosity/cli.py logs <job id>` shows any job's log: servers, router, installs and diagnostics. A
  server that fails its API checks never becomes `READY`; its check results are in
  `$BASE/users/<login>/runs/<profile>-*/evidence/`.
- `./curiosity/cli.py diagnose probe` starts a GPU job that prints the node's network, InfiniBand and GPU
  setup and serves a test page for 15 minutes. Its log names `lrdnXXXX:PORT`.
- `./curiosity/cli.py diagnose reach --target lrdnXXXX:PORT [--proxy http://...] [--link-url https://.../curiosity-link]`
  checks from the router's CPU queue whether it reaches that GPU node, the internet, and the cluster link
  server. A router that answers `502 upstream_unreachable` means the first one failed.
- `./curiosity/cli.py diagnose nccl` checks NCCL bandwidth over InfiniBand between two GPU nodes.

## Further information and project structure

```text
curiosity/
  cli.py          the laptop CLI
  serve/          everything that runs on Curiosity:
    team.py         the commands cli.py calls (init, share, install, up, status, logs, down, router, diagnose)
    inference.py    prepare and submit one model server run; profile rules and preflight checks
    serve_model.sbatch, node.py, lifecycle.py, test_endpoint.py   the model server job
    router.py, router.sbatch      the router job
    registry.py     the team registry in $BASE/endpoints
    profiles/       one JSON file per model
    catalog.json    the runtimes (container images, pinned extra packages) and the chisel client
    site.example.json   template for $BASE/inference-site.json
    client.py       direct client for one server
  install/        install job scripts (runtime, checkpoint, link client)
  diagnostics/    check job scripts (probe, reach, NCCL)
  validation/     evidence from GPU runs, hashes of the code that passed them
  docs/           how-it-works.md, history.md
  tools/          generators and the repository check
  tests/          local tests with simulated Slurm, containers, SSH and HTTP
infrastructure/curiosity/   opt-in Kubernetes component for the cluster link
```

- [docs/slurm-jobs.md](docs/slurm-jobs.md): every Slurm job, and how to run it without the CLI.
- [docs/how-it-works.md](docs/how-it-works.md): a server run step by step, the router, the registry,
  ports, profile rules and the release manifest. Read it before changing `serve/`.
- [docs/history.md](docs/history.md): how the models were brought up, which compatibility problems were
  solved, and what was consolidated from Junghwa's repository and the starter kit.
- [docs/deepseek16.md](docs/deepseek16.md): optional four-node DeepSeek configuration, pinned PP KV
  cache patch, runtime installation, historical load evidence and remaining integration validation.
- [Cluster link / Kubernetes component](docs/how-it-works.md#the-router).

### Validation status

[validation/validated_code.json](validation/validated_code.json) records, for each GPU run, the exact
`serve/` files it ran with, and `python3 curiosity/tools/check_repository.py` lists what has changed since
each run. With the current launcher, `qwen06b` went through the whole path on Curiosity on 20 Oct: `init`,
`install` (stock vLLM runtime and weights), `up` with all seven checks, and a streamed reply through the
router and `cli.py forward`. The three large models passed with an earlier launcher: their profiles and the
container command are unchanged, but `inference.py`, `lifecycle.py`, `node.py` and `test_endpoint.py`
have changed since, so each needs one debug-queue run before the team relies on it. The backport runtime's
install job and the cluster link have not run on Curiosity or GKE yet (the cluster link only on a
workstation). All runs used one operator account; sustained load and unattended operation are untested.

## Contributing and extending

Changes go into `curiosity/` through pull requests to the Codify repository. CI runs these checks; run them
before pushing:

```bash
python3 curiosity/tools/check_repository.py      # syntax, manifest, agentgateway routes, links, drift
python3 -m unittest discover -s curiosity/tests  # simulated Slurm, containers, SSH and HTTP
```

Conventions:

- Code that runs on Curiosity (`serve/`, `install/`, `tools/`) uses only the standard library and Python
  3.6 syntax, because the login nodes run Python 3.6; CI checks this with `vermin`. `cli.py` needs 3.7.
- Never write keys or tokens outside `$BASE/users/<login>/`.
- After any change in `serve/`, run `python3 curiosity/tools/update_manifest.py`; Curiosity refuses code
  that does not match the manifest.
- A change to `serve/` is trusted only after a GPU run: run each affected profile once with
  `--qos boost_qos_dbg` and record the result and the new hashes in `validation/`.

### Add a model

1. Copy a profile in `curiosity/serve/profiles/` (`qwen06b.json` for one GPU, `qwen235b.json` for two
   nodes). Set `model_id`, `revision`, `checkpoint`, `served_model`, resources, parallelism and the vLLM
   `engine_args`. [The profile rules](docs/how-it-works.md#profiles) are checked before each run.
2. Pick the checks a server must pass before it is announced, with `"checks"`:

   | `checks` | Runs | For |
   | --- | --- | --- |
   | `chat_tools` (default) | 401, catalog, arithmetic, English, SSE, forced tool call, 4 concurrent | chat models |
   | `chat` | the same without the tool call | chat models without a tool parser |
   | `minimal` | 401, catalog, any completion | generative guard models such as Llama Guard |
   | `classify` | 401, catalog, `/classify` returns a label | `--runner pooling` classifiers |

   agentgateway handles chat-style APIs; classifiers on `/classify` are called through the router directly.
3. Run `python3 curiosity/tools/update_manifest.py` and `python3 curiosity/tools/render_agentgateway.py`
   (which adds the model's agentgateway route), then `./curiosity/cli.py install <profile>` and a debug run.

### Add a runtime

Add an entry to `curiosity/serve/catalog.json`: the image pinned by digest, any pinned extra wheels for its
Python overlay, the overlay files that must exist, and the modules the CPU check imports. Profiles can then
name it, and `install` installs it. `curiosity/diagnostics/Dockerfile.ib` is a starting point if NCCL
falls back to TCP because an image lacks InfiniBand libraries.
