#!/bin/bash
# Install one runtime from serve/catalog.json into $BASE/runtimes/<id>: pull the container image
# (pinned by digest), add the runtime's pinned extra Python packages in a separate overlay, and
# register both. The backport image needs the official TileLang wheel; stock vLLM needs nothing.
# Submitted by `cli.py install <profile>` on the CPU queue, which has internet. By hand, from the
# repository root:  BASE=$WORK/<team folder> sbatch -A <account> leonardo/install/runtime.sbatch <runtime id>
#SBATCH --job-name=install-runtime
#SBATCH --partition=lrd_all_serial
#SBATCH --output=%x-%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=30000M
#SBATCH --time=02:00:00
set -euo pipefail
RUNTIME="${1:?usage: runtime.sbatch <runtime id from serve/catalog.json>}"
BASE="${BASE:?}"
# The leonardo/ folder with serve/: set by cli.py, or found from where you ran sbatch.
for RELEASE in "${LEONARDO_RELEASE:-}" "${SLURM_SUBMIT_DIR:-.}/leonardo" "${SLURM_SUBMIT_DIR:-.}/curiosity" "${SLURM_SUBMIT_DIR:-.}"; do
    [[ -n "$RELEASE" && -f "$RELEASE/serve/catalog.json" ]] && break
done
[[ -f "$RELEASE/serve/catalog.json" ]] || { echo 'Run sbatch from the repository root, or set LEONARDO_RELEASE to its curiosity/ folder' >&2; exit 1; }
CATALOG="$RELEASE/serve/catalog.json"
DEST="$BASE/runtimes/$RUNTIME"
if [[ -e "$DEST/runtime.json" ]]; then echo "RUNTIME_READY $DEST (already installed)"; exit 0; fi
IMAGE=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["runtimes"][sys.argv[2]]["image"])' "$CATALOG" "$RUNTIME")
STAGING="$BASE/staging/$RUNTIME"
mkdir -p "$STAGING/python-overlay" "$STAGING/tmp/pip" "$STAGING/oci-cache" "$STAGING/wheels" "$STAGING/cache"
export SINGULARITY_TMPDIR="$STAGING/tmp" SINGULARITY_CACHEDIR="$STAGING/oci-cache"

echo "== 1/3 pulling $IMAGE"
SIF="$STAGING/image.sif"
REUSED=$(python3 "$RELEASE/install/overlay_files.py" reuse "$BASE" "$RELEASE/serve" "$RUNTIME" "$STAGING")
if [[ ! -s "$SIF" ]]; then
    singularity pull "$SIF.partial" "$IMAGE"
    mv "$SIF.partial" "$SIF"
fi
# A re-pulled SIF may hash differently from an earlier one: squashfs builds are not reproducible.
sha256sum "$SIF"

echo "== 2/3 overlay packages"
export SINGULARITYENV_PYTHONPATH="$STAGING/python-overlay"
export SINGULARITYENV_XDG_CACHE_HOME="$STAGING/cache" SINGULARITYENV_TILELANG_CACHE_DIR="$STAGING/cache/tilelang"
export SINGULARITYENV_PIP_DISABLE_PIP_VERSION_CHECK=1 SINGULARITYENV_TMPDIR="$STAGING/tmp/pip"
in_image() { singularity exec --cleanenv --no-home --contain --bind "$BASE" "$SIF" "$@" </dev/null; }
# One line per package: name version wheel sha256 modules-to-import
while read -r NAME VERSION WHEEL SHA MODULES; do
    [[ "$REUSED" == yes ]] && continue
    if [[ ! -s "$STAGING/wheels/$WHEEL" ]]; then
        in_image python3 -m pip download --no-deps --only-binary=:all: --dest "$STAGING/wheels" "$NAME==$VERSION"
    fi
    printf '%s  %s\n' "$SHA" "$STAGING/wheels/$WHEEL" | sha256sum -c -
    in_image python3 -m pip install --no-index --no-deps --no-compile --upgrade \
        --target "$STAGING/python-overlay" "$STAGING/wheels/$WHEEL"
    # The backport image's own TileLang was incomplete (no tilelang.webgpu): import every listed module.
    in_image python3 -c 'import importlib, importlib.metadata, sys
name, version, modules = sys.argv[1:]
assert importlib.metadata.version(name) == version, importlib.metadata.version(name)
for module in modules.split(","):
    print("OVERLAY_IMPORT_OK", module, importlib.import_module(module).__file__)' "$NAME" "$VERSION" "$MODULES"
done < <(python3 -c 'import json, sys
for p in json.load(open(sys.argv[1]))["runtimes"][sys.argv[2]]["overlay_packages"]:
    print(p["name"], p["version"], p["wheel"], p["sha256"], ",".join(p.get("import_check", [p["name"]])))' "$CATALOG" "$RUNTIME")
python3 "$RELEASE/install/overlay_files.py" copy "$RELEASE/serve" "$RUNTIME" "$STAGING/python-overlay"
python3 -c 'import json, os, sys
for name in json.load(open(sys.argv[1]))["runtimes"][sys.argv[2]]["overlay_required"]:
    assert os.path.getsize(os.path.join(sys.argv[3], name)) > 0, name' "$CATALOG" "$RUNTIME" "$STAGING/python-overlay"
in_image python3 -c 'import vllm, torch; print("versions: vllm", vllm.__version__, "torch", torch.__version__, "cuda", torch.version.cuda)'
# Import the patch target too: a different vLLM source hash must fail before registration.
IMPORTS=$(python3 -c 'import json, sys; print(",".join(json.load(open(sys.argv[1]))["runtimes"][sys.argv[2]]["imports"]))' "$CATALOG" "$RUNTIME")
in_image python3 -c 'import importlib, sys
for name in sys.argv[1].split(","):
    importlib.import_module(name)
    print("RUNTIME_IMPORT_OK", name)' "$IMPORTS"

echo "== 3/3 registering"
python3 "$RELEASE/install/register_runtime.py" --id "$RUNTIME" --sif "$SIF" \
    --overlay "$STAGING/python-overlay" --destination "$DEST"
# The registered SIF is a hardlink (or a copy), so the staging area can go.
rm -rf "$STAGING"
echo "RUNTIME_READY $DEST"
