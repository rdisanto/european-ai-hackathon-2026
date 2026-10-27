#!/bin/bash
# Before a new multi-node layout, prove that GPUs on two nodes talk over InfiniBand from inside the
# container. Prints bandwidth numbers and the transport NCCL picked.
# Submitted by `cli.py diagnose nccl [--runtime vllm-0.31.0]`. By hand, from the repository root:
#   BASE=$WORK/<team folder> RUNTIME=vllm-0.31.0 sbatch -A <account> leonardo/diagnostics/nccl_check.sbatch
#SBATCH --job-name=nccl-check
#SBATCH --partition=boost_usr_prod
#SBATCH --output=%x-%j.out
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=480GB
#SBATCH --time=00:15:00
set -euo pipefail
BASE="${BASE:?}"
RUNTIME="${RUNTIME:-vllm-0.31.0}"
# The registered image of RUNTIME ($BASE/runtimes/<id>, unless the site file maps it elsewhere).
SIF=$(python3 -c 'import json, sys, os
site = json.load(open(os.path.join(sys.argv[1], "inference-site.json")))
path = os.path.join(sys.argv[1], site.get("runtime_paths", {}).get(sys.argv[2], "runtimes/" + sys.argv[2]))
print(os.path.join(path, json.load(open(os.path.join(path, "runtime.json")))["sif"]))' "$BASE" "$RUNTIME")
# The leonardo/ folder with serve/: set by cli.py, or found from where you ran sbatch.
for RELEASE in "${LEONARDO_RELEASE:-}" "${SLURM_SUBMIT_DIR:-.}/leonardo" "${SLURM_SUBMIT_DIR:-.}"; do
    [[ -n "$RELEASE" && -f "$RELEASE/serve/catalog.json" ]] && break
done
[[ -f "$RELEASE/serve/catalog.json" ]] || { echo 'Run sbatch from the repository root, or set LEONARDO_RELEASE to its leonardo/ folder' >&2; exit 1; }
export SIF SCRIPT_DIR="$RELEASE/diagnostics"

HEAD_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
MASTER_ADDR=$(getent hosts "$HEAD_HOST" | awk '{print $1; exit}')
MASTER_PORT=$(( 29000 + SLURM_JOB_ID % 1000 ))
export MASTER_ADDR MASTER_PORT
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
export PMIX_MCA_gds=hash          # silences a PMIx warning on Leonardo

# Verbose NCCL init so the log shows which transport was chosen.
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET}"
# Optional knobs, once the probe told you the names:
# export NCCL_SOCKET_IFNAME=ib0 GLOO_SOCKET_IFNAME=ib0 NCCL_IB_HCA=mlx5

echo "runtime: $RUNTIME ($SIF)  nodes: $SLURM_JOB_NODELIST  master: $HEAD_HOST ($MASTER_ADDR:$MASTER_PORT)"
srun --kill-on-bad-exit=1 bash "$SCRIPT_DIR/nccl_task.sh"

echo
echo "===== transport summary ====="
LOG=$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/^ *StdOut=//p')
grep -hoE 'NET/(IB|Socket)[^ ]*|via NET/[A-Za-z/]+|GDRDMA' "$LOG" \
  | sort | uniq -c || echo "no NET lines found - check NCCL_DEBUG output above"
