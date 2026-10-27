#!/bin/bash
# The router job. Submitted by `team.py router up` (cli.py router up), which supplies the resources
# from the site file and runs it from $BASE/users/<login>/router.
set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?}"
export ROUTER_PORT=$((41000 + SLURM_JOB_ID % 10000))
printf 'START %s job=%s node=%s port=%s\n' "$(date -Is)" "$SLURM_JOB_ID" "$(hostname -s)" "$ROUTER_PORT"
# Queue the next job first: --dependency=singleton holds it until this one ends.
python3 router.py next || echo 'WARNING: successor not queued; the router stops at this job end' >&2
ROUTER_PID=''
cleanup() {
    if [[ -n "$ROUTER_PID" ]]; then kill "$ROUTER_PID" 2>/dev/null || true; wait "$ROUTER_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
python3 router.py serve &
ROUTER_PID=$!
wait "$ROUTER_PID"
