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
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=32
#SBATCH --mem=480GB
#SBATCH --time=11:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=%x-%j.out
set -euo pipefail
umask 077
for RELEASE in "${CURIOSITY_RELEASE:-}" "${SLURM_SUBMIT_DIR:-.}/curiosity" "${SLURM_SUBMIT_DIR:-.}"; do
    [[ -n "$RELEASE" && -f "$RELEASE/serve/stage.py" ]] && break
done
[[ -f "$RELEASE/serve/stage.py" ]] || { echo 'set CURIOSITY_RELEASE to the curiosity/ folder (or run from the repository root)' >&2; exit 1; }
# curiosity has no docker in the PATH: load the rootless-docker module HERE so the supervisor
# starts with docker and its daemon ready (module load auto-starts it). This also keeps the
# module's own env (DOCKER_HOST, XDG_RUNTIME_DIR, ...) intact, which the node.py fallback loop
# cannot (it only copies $PATH back). Fail loudly here, not deep inside stage.py.
if ! command -v docker >/dev/null; then
    for init in /usr/share/lmod/lmod/init/bash /etc/profile.d/modules.sh /usr/share/Modules/init/bash; do
        [[ -r "$init" ]] && . "$init" 2>/dev/null && break
    done
    if command -v module >/dev/null; then
        module load rootless-docker/1.75 2>/dev/null || module load rootless-docker 2>/dev/null || true
    fi
fi
command -v docker >/dev/null || { echo 'docker introuvable : module load rootless-docker/1.75, puis relancer (cf. glm53_b300_first_run.sh preflight)' >&2; exit 1; }
export STAGE_GPUS=6
cd "$RELEASE"
exec python3 serve/stage.py serve --base "${BASE:?}"
