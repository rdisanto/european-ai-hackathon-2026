# QUICKSTART — leonardo/cli.py

Quick start for the Leonardo inference CLI in this repository. Full
documentation: leonardo/README.md.

## Prerequisites

On your laptop: SSH client + Python 3.7+. Nothing to install (the CLI and
everything it runs on Leonardo use only the standard library). Project
allocation: EUHPC_D30_031.

## Shell configuration (do this ONCE)

Add these lines to your local ~/.bashrc or ~/.zshrc. They set your account,
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

The first command will ask for your CINECA authentication; subsequent
commands reuse that SSH connection for 30 minutes.

IMPORTANT — commands run from the repository root:
```bash
cd european-ai-hackathon-2026
```

## Main commands

### Cluster, server and team-registry status
```bash
./leonardo/cli.py status
```
Shows installed runtimes and checkpoints, your servers (starting / READY with
node and port), your router, and the ready servers of the whole team (sections
listed separately, colored output in an interactive terminal).

### Start a model server
```bash
./leonardo/cli.py up qwen35             # Qwen3.5-122B (4 nodes × 1 GPU, TP4) — default QoS, up to 12 h
./leonardo/cli.py up dsv4flashablit     # DeepSeek-V4-Flash abliterated (4 nodes × 1 GPU, TP4)
```
Useful options:
```bash
./leonardo/cli.py up qwen35 --qos boost_qos_dbg --time 00:30:00   # debug queue: starts sooner, max 30 min
```
A server becomes READY only after its API checks have passed. Until then,
status shows `starting (not ready yet)`. Only one server per model can run at
a time in the team folder; run `up` again to replace it when it expires.

### Access the models
```bash
./leonardo/cli.py router up --hours 12   # start your router (once; it renews itself)
./leonardo/cli.py forward                # forward 127.0.0.1:18000 to your router — keep it running
```
In a second terminal:
```bash
./leonardo/cli.py router token > ~/.config/leonardo-router-token
TOKEN=$(cat ~/.config/leonardo-router-token)
curl http://127.0.0.1:18000/v1/models -H "Authorization: Bearer $TOKEN"
```
→ OpenAI-compatible base URL: http://127.0.0.1:18000/v1, token as API key.
Served model names: `qwen3.5-122b-a10b-fp8-leonardo`,
`deepseek-v4-flash-0731-ablit-leonardo`, etc. (see the table in the README).

## Other useful commands

```bash
./leonardo/cli.py install <profile>      # download runtime + weights (CPU jobs, first time)
./leonardo/cli.py logs <profile|jobid> -f # logs of a server / job / router
./leonardo/cli.py down <profile|jobid>   # stop a server; `down router` stops the router
./leonardo/cli.py router token           # print the router token
./leonardo/cli.py --help                 # full command list
```

## Typical startup sequence

```bash
# 1. Once per machine: variables in .bashrc/.zshrc (above) + source
# 2. Once per team, if the folder does not exist yet:
./leonardo/cli.py init --account EUHPC_D30_031
# 3. First-time model install (long — tens of minutes to hours):
./leonardo/cli.py install qwen35 dsv4flashablit
# 4. Launch:
./leonardo/cli.py up qwen35 --qos boost_qos_dbg --time 00:30:00
./leonardo/cli.py status                # wait for READY
# 5. Access:
./leonardo/cli.py router up --hours 12
./leonardo/cli.py forward               # in a terminal, keep it open
```

## Troubleshooting

- Never READY? → `./leonardo/cli.py logs <profile>`; the check results are in
  `$BASE/users/<your login>/runs/<profile>-*/evidence/`.
- Router answers 502 upstream_unreachable? → your server is not reachable from
  the CPU queue; see `./leonardo/cli.py diagnose`.
- SSH connection stale after 30 min of inactivity: the first command asks for
  authentication again — this is normal.
