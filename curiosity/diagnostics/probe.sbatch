#!/bin/bash
# Find out what a GPU node looks like from the inside, then serve a static page for ~15 minutes so
# you can test whether other nodes reach a port on a GPU node, without vLLM in the way.
# Submitted by `cli.py diagnose probe`. By hand:  sbatch -A <account> leonardo/diagnostics/probe.sbatch
#SBATCH --job-name=probe
#SBATCH --partition=boost_usr_prod
#SBATCH --output=%x-%j.out
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120GB
#SBATCH --time=00:20:00
set -uo pipefail

section() { printf '\n===== %s =====\n' "$1"; }

section "identity"
echo "hostname: $(hostname) / short: $(hostname -s)"
echo "resolved IP of own hostname: $(getent hosts "$(hostname -s)" | awk '{print $1; exit}')"
grep PRETTY_NAME /etc/os-release
uname -r

section "network interfaces"
ip -br addr
section "routes"
ip route
section "InfiniBand devices"
ls /sys/class/infiniband 2>/dev/null || echo "no /sys/class/infiniband"
for p in /sys/class/infiniband/*/ports/*; do
  [[ -e "$p/state" ]] && echo "$p: $(cat "$p/state") $(cat "$p/rate" 2>/dev/null)"
done
command -v ibv_devinfo >/dev/null && ibv_devinfo -l
ls -l /dev/infiniband 2>/dev/null | head
echo "locked-memory limit (RDMA wants unlimited): $(ulimit -l)"
lsmod 2>/dev/null | grep -E 'nvidia_peermem|nv_peer_mem' || echo "nvidia_peermem not listed (GPUDirect RDMA may be off)"

section "GPU"
nvidia-smi
nvidia-smi topo -m

section "container runtime"
(singularity --version || apptainer --version) 2>/dev/null
df -h /dev/shm /tmp | sed 1d

section "outbound internet (expect failure on GPU nodes)"
curl -sS -m 8 -o /dev/null -w 'direct https://pypi.org -> %{http_code}\n' https://pypi.org || true
if [[ -n "${PROBE_PROXY:-}" ]]; then
  curl -sS -m 8 -x "$PROBE_PROXY" -o /dev/null -w 'via proxy https://pypi.org -> %{http_code}\n' https://pypi.org || true
fi

section "static web server"
PORT=$(( 31000 + SLURM_JOB_ID % 10000 ))
WWW=$(mktemp -d)
trap 'rm -rf "$WWW"' EXIT
printf '<!doctype html><title>Leonardo probe</title><h1>Hello from %s</h1><p>job %s, port %s</p>\n' \
  "$(hostname -s)" "$SLURM_JOB_ID" "$PORT" > "$WWW/index.html"
cd "$WWW" || exit 1
python3 -m http.server "$PORT" --bind 0.0.0.0 &
SRV=$!
sleep 2
curl -sS -m 5 "http://127.0.0.1:${PORT}/" >/dev/null && echo "local check OK"
cat <<EOF

Test it now (the page stays up for ~15 minutes):
  from the router's CPU queue: cli.py diagnose reach --target $(hostname -s):${PORT}
  from a login node:           curl http://$(hostname -s):${PORT}/
EOF
sleep 900
kill "$SRV" 2>/dev/null
