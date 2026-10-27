#!/bin/bash
# Download a profile's model weights (model_id and revision from serve/profiles/<profile>.json) to
# $BASE/checkpoints/<checkpoint>, using huggingface_hub inside the profile's own runtime image.
# Submitted by `cli.py install <profile>` on the CPU queue (internet, no GPU budget), after the
# runtime job. A profile without a pinned revision gets the current one, printed as PIN_REVISION.
# Sizes: qwen235b ~239 GB, glm53 ~328 GB, deepseek41 48 shards, qwen06b ~1.5 GB. By hand, from the
# repository root, after the runtime is installed:
#   BASE=$WORK/<team folder> sbatch -A <account> curiosity/install/checkpoint.sbatch <profile>
#SBATCH --job-name=install-checkpoint
#SBATCH --partition=lrd_all_serial
#SBATCH --output=%x-%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=30000M
#SBATCH --time=04:00:00
set -euo pipefail
NAME="${1:?usage: checkpoint.sbatch <profile name>}"
BASE="${BASE:?}"
# The curiosity/ folder with serve/: set by cli.py, or found from where you ran sbatch.
for RELEASE in "${CURIOSITY_RELEASE:-}" "${SLURM_SUBMIT_DIR:-.}/curiosity" "${SLURM_SUBMIT_DIR:-.}/curiosity" "${SLURM_SUBMIT_DIR:-.}"; do
    [[ -n "$RELEASE" && -f "$RELEASE/serve/catalog.json" ]] && break
done
[[ -f "$RELEASE/serve/catalog.json" ]] || { echo 'Run sbatch from the repository root, or set CURIOSITY_RELEASE to its curiosity/ folder' >&2; exit 1; }
read -r MODEL_ID PROFILE_REVISION CHECKPOINT RUNTIME < <(python3 -c 'import json, sys
p = json.load(open(sys.argv[1]))
print(p["model_id"], p.get("revision") or "-", p["checkpoint"], p["runtime"])' "$RELEASE/serve/profiles/$NAME.json")
if [[ "$PROFILE_REVISION" != - ]]; then MODEL_REVISION="$PROFILE_REVISION"; fi
SIF="$BASE/runtimes/$RUNTIME/image.sif"
mkdir -p "$BASE/checkpoints" "$BASE/staging/hf-cache"
export SINGULARITYENV_MODEL_ID="$MODEL_ID"
export SINGULARITYENV_MODEL_REVISION="${MODEL_REVISION:-}"
export SINGULARITYENV_MODEL_DIR="$BASE/checkpoints/$CHECKPOINT"
export SINGULARITYENV_HF_HOME="$BASE/staging/hf-cache"
export SINGULARITYENV_HF_HUB_DOWNLOAD_TIMEOUT=120
export SINGULARITYENV_HF_HUB_DISABLE_TELEMETRY=1
for v in HF_TOKEN HTTPS_PROXY HTTP_PROXY https_proxy http_proxy NO_PROXY no_proxy; do
    if [[ -n "${!v:-}" ]]; then export "SINGULARITYENV_$v=${!v}"; fi
done
singularity exec --cleanenv --no-home --contain --bind "$BASE" "$SIF" python3 -u - <<'PY'
import json, os, shutil
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

repo = os.environ["MODEL_ID"]
revision = os.environ["MODEL_REVISION"]
directory = Path(os.environ["MODEL_DIR"])
if not revision:
    # Profile not pinned yet: take the current main and say so, so the profile can be pinned.
    revision = HfApi().model_info(repo).sha
    print("PIN_REVISION", repo, revision, "- set this as \"revision\" in the profile", flush=True)
metadata = HfApi().model_info(repo, revision=revision, files_metadata=True)
assert metadata.sha == revision
weights = {f.rfilename: f for f in metadata.siblings if f.rfilename.endswith(".safetensors")}
needed = sum(f.size for f in weights.values())
existing = sum((directory / name).stat().st_size for name, f in weights.items() if (directory / name).is_file() and (directory / name).stat().st_size == f.size)
free = shutil.disk_usage(directory.parent).free
if free < needed - existing + 100 * 2**30:
    raise RuntimeError("Insufficient free shared storage for checkpoint plus 100 GiB reserve")
print("Downloading", repo, revision, "weight_bytes", needed, flush=True)
snapshot_download(repo, revision=revision, local_dir=directory, max_workers=4,
                  allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken", "*.jinja", "*.py", "LICENSE"])
if (directory / "model.safetensors.index.json").exists():
    shards = set(json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"].values())
else:
    shards = {"model.safetensors"}  # small models ship one unindexed file
required = shards | {"config.json", "tokenizer.json", "tokenizer_config.json"}
remote = {f.rfilename: f for f in metadata.siblings}
for name in sorted(required):
    path = directory / name
    if not path.is_file() or path.stat().st_size != remote[name].size:
        raise RuntimeError("Missing or incomplete checkpoint file: " + name)
manifest = {"repo": repo, "revision": revision, "weight_bytes": needed,
            "verified_files": {name: remote[name].size for name in sorted(required)},
            "validation": "HF download integrity and required-file size checks"}
(directory / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("CHECKPOINT_READY", directory, "revision", revision, "shards", len(shards), flush=True)
PY
du -sh "$BASE/checkpoints/$CHECKPOINT"
