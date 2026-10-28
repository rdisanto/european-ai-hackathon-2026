# 🚀 QUICKSTART — curiosity/cli.py

Run the team's shared 8× B300 **stage** from your laptop: reserve, load/unload
models, forward the router port. Nothing to install on the laptop or on
Curiosity beyond an SSH client and Python 3.7+ (stdlib only). Full
documentation: `curiosity/README.md`, `docs/slurm-jobs.md`, `docs/how-it-works.md`.

## ⚙️ Shell configuration (do this ONCE)

Add to your local `~/.bashrc` or `~/.zshrc` — every command re-reads them:

```bash
export CURIOSITY_USER=<your curiosity login>           
export CURIOSITY_BASE='/storage/hackathon_teams/eai-team05/'
export CURIOSITY_HOST=ssh.axisapps.io
```

```bash
source ~/.bashrc   # or: source ~/.zshrc
```

🔐 The first command asks for your CINECA authentication; later commands reuse
that SSH connection for 30 minutes.
📍 Run from the **repository root** (`cd european-ai-hackathon-2026`); the CLI
uploads the matching `serve/` release to the cluster itself when needed.

## 🍴 Clone your own fork, on branch `curiosity`

```bash
git clone --branch curiosity git@github.com:rdisanto/european-ai-hackathon-2026.git
cd european-ai-hackathon-2026
```

## 🏟️ The stage in one paragraph

The team shares **one** long Slurm reservation on one node with **8 B300 GPUs**
(`stage reserve`). Inside it, a headless supervisor owns rootless docker and a
router; `stage load <profile>` puts a model on its GPU slice (sized by the
profile's parallelism) and `stage unload` takes it back — no new job, no queue,
models swap in minutes. Every loaded model is served by the same
OpenAI-compatible router under one token. Team-shared: any member can
load/unload/read status; only the **owner** can `stage down` or `--restart`.

## 🧑‍💻 Main commands

```bash
./curiosity/cli.py stage status          # 🩺 reservation state, loaded models, GPU slices, API ports
./curiosity/cli.py stage reserve         # 🏟️ submit the stage (or REATTACH to a teammate's — never a 2nd job)
./curiosity/cli.py stage load gemma4     # ⬆️ load a model into the stage
./curiosity/cli.py stage unload gemma4   # ⬇️ free its GPU slice
./curiosity/cli.py stage down            # 🛑 cancel the reservation (owner only)
./curiosity/cli.py stage forward         # 🔌 forward 127.0.0.1:18000 → the stage router (keep running)
```

### 🚀 Profiles (verified against `serve/profiles/*.json`)

| Profile | 🧠 Served name | 🎴 GPUs | 🔢 Quantization | 📏 Context | ⏳ Cold load |
| --- | --- | --- | --- | --- | --- |
| `glm53` | `glm-5.3` | 4 (TP4+EP) | FP8 weights + KV `fp8_e4m3`, MTP | 1,048,576 | 20-40 min (704 GiB) |
| `qwen38flash` | `qwen-3.8-flash` | 1 (TP1) | FP8 weights + KV, MTP k=3 | 262,144 | ~8 min (measured) |
| `gemma4` | `gemma-4` | 1 (TP1) | bf16 | 32,768 | ~2 min (58 GiB) |

Slices are allocated from the first aligned free block: glm53 takes GPUs 0-3,
the TP1 profiles take 4 and 5 — one TP4 + up to two TP1 models fit at once.
Every engine flag and tool/reasoning parser name is verified against the
imported docker image **before** the cold load, so a bad profile fails in
seconds, not after 40 minutes of weight loading.

## 🌐 Access the models (second terminal)

```bash
./curiosity/cli.py stage forward --local-port 18000   # 18000 is the default
./curiosity/cli.py router token                        # 🔑 the stage router's shared token
curl http://127.0.0.1:18000/v1/models -H "Authorization: Bearer ***"
```

→ OpenAI-compatible base URL: `http://127.0.0.1:18000/v1`, token as API key
(point Hermes/o at it; model names per the table above).

## 🛠️ One-time setup (first machine / first model)

```bash
./curiosity/cli.py init --account EUHPC_D30_031   # 🏗️ team folder + inference-site.json (skip if it exists)
./curiosity/cli.py install <profile>              # ⬇️ download the checkpoint (CPU job, resumable, verified shard by shard)
```

## 🚦 Typical day

```bash
# 1️⃣ morning: is the stage alive?
./curiosity/cli.py stage status
# 2️⃣ not reserved? (or reattaches to a teammate's)
./curiosity/cli.py stage reserve
# 3️⃣ load what you need, then forward the router in another terminal
./curiosity/cli.py stage load qwen38flash
./curiosity/cli.py stage forward
# 4️⃣ done for the day: free GPUs for teammates (keep the reservation running)
./curiosity/cli.py stage unload qwen38flash
```

## 🆘 Troubleshooting

- ⏳ `stage status` says *no model loaded* while a load runs → normal: the
  model appears in `models.json` only when READY; the loading claim shows the
  GPU slice it already pins.
- 🧊 Load stuck at *preflight* → the docker image is missing on the node and
  the pull is synchronous; check `status.json.updated` age, then pre-pull the
  image on that node. `stage down` also empties pending `cmd/` files, so a
  wedged command replays never.
- 🧨 `docker introuvable` inside a job → the `rootless-docker/1.75` module
  failed to load; the reason is in `$BASE/stage/reserve-stage-<jobid>.out`.
  The supervisor log is `$BASE/stage/supervisor.out`.
- 🔌 `stage forward` port busy → `--local-port <N>`, or close the shared SSH
  connection (it drops its forwards).
- 🚫 `stage down` refused → only the reservation owner can cancel it (Slurm
  rejects cross-account scancel); `stage status` shows the owner.
- 📜 Anything else → `./curiosity/cli.py logs <profile|jobid|router>` and
  `./curiosity/cli.py diagnose probe`.
