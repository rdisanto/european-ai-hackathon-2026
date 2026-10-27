#!/bin/bash
# Resource requests are supplied by inference.py from the selected profile.
set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?}"
python3 inference.py preflight
export INFERENCE_MASTER_ADDR="$(getent hosts "$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)" | awk '{print $1; exit}')"
export INFERENCE_MASTER_PORT=$((29000 + SLURM_JOB_ID % 1000))
export INFERENCE_API_PORT=$((31000 + SLURM_JOB_ID % 10000))
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:?}"
export PMIX_MCA_gds=hash
SRUN_PID=''
WATCH_PID=''
cleanup() {
    rm -f .runtime/endpoint.json
    if [[ -n "$WATCH_PID" ]]; then kill "$WATCH_PID" 2>/dev/null || true; wait "$WATCH_PID" 2>/dev/null || true; fi
    if [[ -n "$SRUN_PID" ]]; then kill "$SRUN_PID" 2>/dev/null || true; wait "$SRUN_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
printf 'START %s job=%s nodes=%s\n' "$(date -Is)" "$SLURM_JOB_ID" "$SLURM_JOB_NODELIST"
srun --kill-on-bad-exit=1 --ntasks-per-node=1 python3 node.py &
SRUN_PID=$!
# All long-running supervisors use interruptible waits so batch-only TERM works.
python3 lifecycle.py --launcher-pid "$SRUN_PID" &
WATCH_PID=$!
wait "$WATCH_PID"
