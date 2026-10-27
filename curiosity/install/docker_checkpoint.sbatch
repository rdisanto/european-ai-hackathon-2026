#!/bin/bash
# Download a docker-runtime profile's model weights (model_id and revision from the profile) to
# $BASE/checkpoints/<checkpoint>, using huggingface_hub inside the profile's own docker image.
# Submitted by `cli.py install <profile>` on a partition with internet. For docker runtimes there is
# no SIF to convert: the image is pulled on this node and the download runs through `docker run`,
# following the glm53_b300_first_run.sh `download` recipe (no GPU needed). A profile without a pinned
# revision gets the current one, printed as PIN_REVISION.
# By hand, from the repository root, after docker is available:
#   BASE=$BASE sbatch curiosit/install/docker_checkpoint.sbatch <profile>
#SBATCH --job-name=install-checkpoint
#SBATCH --partition=hackathon
#SBATCH --output=%x-%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=30000M
#SBATCH --time=08:00:00
set -euo pipefail
NAME="${1:?usage: docker_checkpoint.sbatch <profile name>}"
BASE="${BASE:?}"
# The curiosity/ folder with serve/: set by cli.py, or found from where you ran sbatch.
for RELEASE in "${CURIOSITY_RELEASE:-}" "${SLURM_SUBMIT_DIR:-.}/curiosity" "${SLURM_SUBMIT_DIR:-.}"; do
    [[ -n "$RELEASE" && -f "$RELEASE/serve/catalog.json" ]] && break
done
[[ -f "$RELEASE/serve/catalog.json" ]] || { echo 'Run sbatch from the repository root, or set CURIOSITY_RELEASE to its curiosity/ folder' >&2; exit 1; }

# curiosity has no docker in the PATH: load the rootless-docker module (cf. glm53_b300_first_run.sh).
if ! command -v docker >/dev/null; then
    for init in /usr/share/lmod/lmod/init/bash /etc/profile.d/modules.sh /usr/share/Modules/init/bash; do
        [[ -r "$init" ]] && . "$init" 2>/dev/null && break
    done
    command -v module >/dev/null && { module load rootless-docker 2>/dev/null || module load rootless-docker/1.75 2>/dev/null || true; }
fi
command -v docker >/dev/null || { echo 'docker introuvable : module load rootless-docker' >&2; exit 1; }

read -r MODEL_ID PROFILE_REVISION CHECKPOINT RUNTIME < <(python3 -c 'import json, sys
p = json.load(open(sys.argv[1]))
print(p["model_id"], p.get("revision") or "-", p["checkpoint"], p["runtime"])' "$RELEASE/serve/profiles/$NAME.json")
MODEL_REVISION=""
if [[ "$PROFILE_REVISION" != - ]]; then MODEL_REVISION="$PROFILE_REVISION"; fi
IMAGE=$(python3 -c 'import json, sys
print(json.load(open(sys.argv[1]))["runtimes"][sys.argv[2]]["image"])' "$RELEASE/serve/catalog.json" "$RUNTIME")

USER_ARGS=""
if ! docker info --format '{{json .SecurityOptions}}' 2>/dev/null | grep -qi rootless; then
    USER_ARGS="--user $(id -u):$(id -g)"
fi
mkdir -p "$BASE/checkpoints" "$BASE/staging/hf-cache"
echo "Pull $IMAGE" >&2
docker pull "$IMAGE"

AUTH_ENV=()
if [[ -n "${HF_TOKEN:-}" ]]; then AUTH_ENV=(-e HF_TOKEN); fi
docker run --rm $USER_ARGS \
    -v "$BASE/checkpoints/$CHECKPOINT:/models" \
    -v "$BASE/staging/hf-cache:/cache" \
    -e HOME=/tmp -e HF_HOME=/cache -e HF_HUB_DOWNLOAD_TIMEOUT=120 -e HF_HUB_DISABLE_TELEMETRY=1 \
    -e MODEL_ID -e MODEL_DIR=/models -e MODEL_REVISION -e IMAGE \
    ${AUTH_ENV[@]+"${AUTH_ENV[@]}"} \
    --entrypoint python3 "$IMAGE" -u - <<'PY'
import json, os, shutil
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

repo = os.environ["MODEL_ID"]
revision = os.environ.get("MODEL_REVISION") or ""
directory = Path(os.environ["MODEL_DIR"])
if not revision:
    revision = HfApi().model_info(repo).sha
    print("PIN_REVISION", repo, revision, "- set this as \"revision\" in the profile", flush=True)
metadata = HfApi().model_info(repo, revision=revision, files_metadata=True)
weights = {f.rfilename: f for f in metadata.siblings if f.rfilename.endswith(".safetensors")}
needed = sum(f.size for f in weights.values())
existing = sum((directory / name).stat().st_size for name, f in weights.items() if (directory / name).is_file() and (directory / name).stat().st_size == f.size)
free = shutil.disk_usage(directory.parent).free
if free < needed - existing + 100 * 2**30:
    raise RuntimeError("Insufficient free shared storage for checkpoint plus 100 GiB reserve")
print("Downloading", repo, revision, "weight_bytes", needed, flush=True)
snapshot_download(repo, revision=revision, local_dir=directory, max_workers=16,
                  allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken", "*.jinja", "*.py", "LICENSE"])
if (directory / "model.safetensors.index.json").exists():
    shards = set(json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"].values())
else:
    shards = {"model.safetensors"}
required = shards | {"config.json", "tokenizer.json", "tokenizer_config.json"}
remote = {f.rfilename: f for f in metadata.siblings}
for name in sorted(required):
    path = directory / name
    if not path.is_file() or path.stat().st_size != remote[name].size:
        raise RuntimeError("Missing or incomplete checkpoint file: " + name)
manifest = {"repo": repo, "revision": revision, "weight_bytes": needed,
            "verified_files": {name: remote[name].size for name in sorted(required)},
            "runtime": "docker:" + os.environ.get("IMAGE", ""),
            "validation": "HF download integrity and required-file size checks via docker image"}
(directory / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("CHECKPOINT_READY", directory, "revision", revision, "shards", len(shards), flush=True)
PY
du -sh "$BASE/checkpoints/$CHECKPOINT"
