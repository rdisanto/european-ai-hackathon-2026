# 🚀 QUICKSTART — leonardo/cli.py

Quick start for the Leonardo inference CLI in this repository. Full
documentation: leonardo/README.md.

## 📋 Prerequisites

- 💻 On your laptop: SSH client + Python 3.7+. Nothing to install (the CLI and
  everything it runs on Leonardo use only the standard library).
- 🏢 Project allocation: EUHPC_D30_031.

## ⚙️ Shell configuration (do this ONCE)

Add these lines to your local `~/.bashrc` or `~/.zshrc`. They set your account,
the team folder and the login node — every CLI command re-reads these
variables, so there is nothing else to set up.

```bash
export LEONARDO_USER=<your CINECA login>                 # e.g. a08trc22 — replace with YOUR ssh login
export LEONARDO_BASE='$WORK/cybertwin.ai/hackathon'      # single quotes: $WORK is expanded on Leonardo
export LEONARDO_HOST=login07-ext.leonardo.cineca.it
```

Then reload your shell:
```bash
source ~/.bashrc   # or: source ~/.zshrc
```

🔐 The first command will ask for your CINECA authentication; subsequent
commands reuse that SSH connection for 30 minutes.

📍 IMPORTANT — commands run from the repository root:
```bash
cd european-ai-hackathon-2026
```

## 🍴 Clone your own fork, on branch `cybertwin`

The working branch with all the Leonardo/cybertwin changes (qwen35 profile,
optimized dsv4flashablit, colored status, various server fixes) lives on the
`cybertwin` branch of the team fork. On a new machine, clone it with SSH (you
need your GitHub SSH key registered):

```bash
git clone --branch cybertwin git@github.com:rdisanto/european-ai-hackathon-2026.git
cd european-ai-hackathon-2026
```

The official repository is added as `upstream` so you can pull the latest
changes and open pull requests:

```bash
git remote add upstream git@github.com:CodifyAG/european-ai-hackathon-2026.git
git fetch upstream
```

To update your local branch with the latest official changes:
```bash
git pull upstream main
```

## 🔬 Curiosity cluster (B300, partition `hackathon`)

The `curiosity/` folder in this repository is the same stack retargeted to the
hackathon B300 cluster (8x NVIDIA B300 per node, driver 580.126.20 / CUDA 13.0,
docker through the `rootless-docker` module). On that deployment, replace
`./leonardo/cli.py` with `./curiosity/cli.py` in every command below. Everything
learned by `curiosity/diagnostics/glm53_b300_first_run.sh` is baked in:

- **Docker runtimes**: the `glm53` and `dsv4flashablit` profiles use the
  `vllm-openai-docker-0.32` runtime (vllm/vllm-openai v0.32.0, pinned by digest).
  No SIF, no overlay: the image is pulled on the GPU node, the GPU mode is probed
  (`--gpus all` / `--runtime=nvidia` / CDI) and **every engine flag plus the parser
  names are verified against the image itself before the 704 GiB cold load** — a
  bad flag fails in minutes, not after half an hour of loading weights.
- **GLM-5.3** (`up glm53`): TP=EP=4 on one node, KV fp8_e4m3, MTP speculative
  decoding, 1M-token context at 3 concurrent sequences (~12 GiB/card margin at
  util 0.95). Startup budget 90 min for the cold load.
- **DeepSeek-V4-Flash abliterated** (`up dsv4flashablit`): same image, one node,
  4 GPUs, 512K context; its parser names are resolved from the image's registry
  at startup instead of being guessed.
- **Router with both models**: `up glm53` and `up dsv4flashablit` are two Slurm
  jobs; the router serves both names at once (`glm-5.3`,
  `deepseek-v4-flash-ablit`) and routes by the `model` field. The upstream
  timeout is 1 h, enough for a 1M-token prefill.
- **The 756 GB GLM-5.3 checkpoint** already downloaded by the diagnostics script
  can be reused: map it in `inference-site.json` with
  `"checkpoints": {"glm53": "<MODEL_DIR of the diagnostics script>"}` instead of
  running `install glm53`.

Site defaults: partition `hackathon`, no QoS, no account (pass one to `init`
only if the cluster asks for it). Allocation shape per server: 1 node, 4 GPUs,
8 CPUs — the same as the manual
`srun --nodes=1 --ntasks-per-node=1 --gpus-per-task=4 --cpus-per-task=8 --partition=hackathon --pty bash`
used so far for the diagnostics.

## 🧑‍💻 Main commands

### 🩺 Cluster, server and team-registry status
```bash
./leonardo/cli.py status
```
Shows installed runtimes and checkpoints, your servers (starting / READY with
node and port), your router, and the ready servers of the whole team (sections
listed separately, colored output in an interactive terminal).

### ▶️ Start a model server
```bash
./leonardo/cli.py up qwen35             # Qwen3.5-122B (4 nodes × 1 GPU, TP4) — default QoS, up to 12 h
./leonardo/cli.py up dsv4flashablit     # DeepSeek-V4-Flash abliterated (4 nodes × 1 GPU, TP4)
```

Profiles available with `up` (full served model name, quantization, context,
and requested hardware — verified against `serve/profiles/*.json`):

| 🚀 Profile / command | 🧠 Full served model name | 🔢 Quantization | 📏 Context (tokens) | 🖥️ Nodes | 🎴 GPUs/node | ⏳ Lifetime |
| --- | --- | --- | --- | --- | --- | --- |
| `./leonardo/cli.py up qwen35` | `qwen3.5-122b-a10b-fp8-leonardo` | FP8 | 32,768 | 4 | 1 | 12 h |
| `./leonardo/cli.py up dsv4flashablit` | `deepseek-v4-flash-ablit` | FP8 weights | 524,288 | 1 | 4 | 12 h |
| `./leonardo/cli.py up deepseek41` | `deepseek-v4.1-flash-1017-leonardo` | bf16 (kv `fp8_ds_mla`) | 2,048 | 2 | 4 | 2 h |
| `./leonardo/cli.py up deepseek41-8gpu` | `deepseek-v4.1-flash-1017-16gpu-leonardo` | bf16 (kv `fp8_ds_mla`) | 32,768 | 8 | 1 | 2 h |
| `./leonardo/cli.py up glm53` | `glm-5.3` | FP8 weights + KV `fp8_e4m3`, MTP | 1,048,576 | 1 | 4 | 2 h |
| `./leonardo/cli.py up qwen06b` | `qwen3-0.6b-leonardo` | bf16 | 40,960 | 1 | 1 | 1 h |

Useful options:
```bash
./leonardo/cli.py up qwen35 --qos boost_qos_dbg --time 00:30:00   # debug queue: starts sooner, max 30 min
```
A server becomes READY only after its API checks have passed. Until then,
status shows `starting (not ready yet)`. Only one server per model can run at
a time in the team folder; run `up` again to replace it when it expires.

### 🌐 Access the models
```bash
./leonardo/cli.py router up --hours 12   # start your router (once; it renews itself)
./leonardo/cli.py forward                # forward 127.0.0.1:18000 to your router — keep it running
```
In a second terminal:
```bash
./leonardo/cli.py router token > ~/.config/leonardo-router-token
TOKEN=$(cat ~/.config/leonardo-router-token)
curl http://127.0.0.1:18000/v1/models -H "Authorization: Bearer ***"
```
→ OpenAI-compatible base URL: `http://127.0.0.1:18000/v1`, token as API key.
Served model names: `qwen3.5-122b-a10b-fp8-leonardo`,
`deepseek-v4-flash-0731-ablit-leonardo`, etc. (see the table in the README).

## 🛠️ Other useful commands

```bash
./leonardo/cli.py install <profile>      # ⬇️ download runtime + weights (CPU jobs, first time)
./leonardo/cli.py logs <profile|jobid> -f # 📜 logs of a server / job / router
./leonardo/cli.py down <profile|jobid>   # 🛑 stop a server; `down router` stops the router
./leonardo/cli.py router token           # 🔑 print the router token
./leonardo/cli.py --help                 # ❓ full command list
```

## 🚦 Typical startup sequence

```bash
# 1️⃣ Once per machine: variables in .bashrc/.zshrc (above) + source
# 2️⃣ Once per team, if the folder does not exist yet:
./leonardo/cli.py init --account EUHPC_D30_031
# 3️⃣ First-time model install (long — tens of minutes to hours):
./leonardo/cli.py install qwen35 dsv4flashablit
# 4️⃣ Launch:
./leonardo/cli.py up qwen35 --qos boost_qos_dbg --time 00:30:00
./leonardo/cli.py status                # wait for READY
# 5️⃣ Access:
./leonardo/cli.py router up --hours 12
./leonardo/cli.py forward               # in a terminal, keep it open
```

## 🆘 Troubleshooting

- ❌ Never READY? → `./leonardo/cli.py logs <profile>`; the check results are in
  `$BASE/users/<your login>/runs/<profile>-*/evidence/`.
- 🌩️ Router answers 502 upstream_unreachable? → your server is not reachable from
  the CPU queue; see `./leonardo/cli.py diagnose`.
- 🔌 SSH connection stale after 30 min of inactivity: the first command asks for
  authentication again — this is normal.
