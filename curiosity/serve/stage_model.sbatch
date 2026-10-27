#!/bin/bash
# The single long reservation: 1 node / 8x B300, for as long as the site allows.
# Submit it with `cli.py stage reserve` (or by hand, from the repository root):
#   BASE=$WORK/<team> sbatch [--account=...] [--time=11:00:00] serve/stage_model.sbatch
# Inside it runs the stage supervisor (serve/stage.py serve), which loads/unloads models
# through rootless docker and runs the router. PORT/API ports are allocated by the supervisor.
#SBATCH --job-name=stage
#SBATCH --partition=hackathon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=480GB
#SBATCH --time=11:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=%x-%j.out
set -euo pipefail
umask 077
for RELEASE in "${LEONARDO_RELEASE:-}" "${SLURM_SUBMIT_DIR:-.}/curiosity" "${SLURM_SUBMIT_DIR:-.}"; do
    [[ -n "$RELEASE" && -f "$RELEASE/serve/stage.py" ]] && break
done
[[ -f "$RELEASE/serve/stage.py" ]] || { echo 'set LEONARDO_RELEASE to the curiosity/ folder (or run from the repository root)' >&2; exit 1; }
export STAGE_GPUS=8
cd "$RELEASE"
exec python3 serve/stage.py serve --base "${BASE:?}"
