#!/bin/bash
# One Slurm task = one GPU. Called by nccl_check.sbatch through srun; needs SIF, SCRIPT_DIR,
# MASTER_ADDR and MASTER_PORT from it.
set -euo pipefail
CTR=$(command -v singularity || command -v apptainer)
ENVPFX=$([[ "$(basename "$CTR")" == apptainer ]] && echo APPTAINERENV_ || echo SINGULARITYENV_)
cenv() { export "${ENVPFX}$1=$2"; }

cenv HF_HUB_OFFLINE 1
cenv VLLM_NO_USAGE_STATS 1
# Leonardo's host driver (535) is older than the images' CUDA: load their forward-compat libraries.
cenv VLLM_ENABLE_CUDA_COMPATIBILITY 1
for v in $(compgen -e | grep -E '^(NCCL_|GLOO_|UCX_|TORCH_NCCL_|TORCH_DISTRIBUTED_)' || true); do
  cenv "$v" "${!v}"
done
cenv RANK "$SLURM_PROCID"
cenv LOCAL_RANK "$SLURM_LOCALID"
cenv WORLD_SIZE "$SLURM_NTASKS"
cenv MASTER_ADDR "${MASTER_ADDR:?}"
cenv MASTER_PORT "${MASTER_PORT:?}"

# vLLM sets LD_LIBRARY_PATH for CUDA compatibility. A fresh interpreter is
# required for the dynamic linker to honor that path; importing vLLM inside
# the benchmark's existing interpreter alone does not update linker search.
exec "$CTR" exec --nv --cleanenv --bind "$SCRIPT_DIR" "${SIF:?}" \
  python3 -c 'import os, sys; import vllm; os.execv(sys.executable, [sys.executable, *sys.argv[1:]])' \
  "$SCRIPT_DIR/nccl_check.py"
