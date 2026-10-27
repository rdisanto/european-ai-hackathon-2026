#!/bin/bash
# Answers the two questions the router depends on, from the CPU queue it runs in:
#   1. Can this node reach a port on a GPU node?  TARGET=lrdnXXXX:PORT (start a probe first)
#   2. Can it reach the internet, directly or via a proxy?  PROBE_PROXY=http://host:port (optional)
#      LINK_URL=https://.../leonardo-link (optional): /health of the cluster link server in Kubernetes
# Submitted by `cli.py diagnose reach --target ... [--proxy ...] [--link-url ...]`. By hand:
#   TARGET=lrdnXXXX:PORT sbatch -A <account> leonardo/diagnostics/reach.sbatch
#SBATCH --job-name=reach
#SBATCH --partition=lrd_all_serial
#SBATCH --output=%x-%j.out
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000M
#SBATCH --time=00:10:00
set -uo pipefail
echo "node: $(hostname -s)"
if [[ -n "${TARGET:-}" ]]; then
  curl -sS -m 10 -o /dev/null -w "GPU node http://${TARGET}/ -> %{http_code}\n" "http://${TARGET}/" || echo "GPU node ${TARGET}: UNREACHABLE"
fi
curl -sS -m 10 -o /dev/null -w 'direct https://pypi.org -> %{http_code}\n' https://pypi.org || echo 'direct internet: NO'
if [[ -n "${PROBE_PROXY:-}" ]]; then
  curl -sS -m 10 -x "$PROBE_PROXY" -o /dev/null -w 'via proxy https://pypi.org -> %{http_code}\n' https://pypi.org || echo 'via proxy: NO'
fi
if [[ -n "${LINK_URL:-}" ]]; then
  curl -sS -m 10 ${PROBE_PROXY:+-x "$PROBE_PROXY"} -w ' <- cluster link server /health %{http_code}\n' "${LINK_URL%/}/health" || echo 'cluster link server: UNREACHABLE'
fi
