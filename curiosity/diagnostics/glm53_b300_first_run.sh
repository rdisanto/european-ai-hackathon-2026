#!/bin/bash
# Premier run GLM-5.3 (743B total / 39B actifs, checkpoint FP8 703.7 GiB) sur UN noeud curiosity
#   8x NVIDIA B300 SXM6 AC, 275040 MiB par carte, pilote 580.126.20 / CUDA 13.0.
#
# Arithmetique verifiee (util 0.90 -> 241.7 GiB utilisables/carte) :
#   poids 703.7 GiB / 8 = 88.0 GiB par carte (36 %)  -> 153.8 GiB/carte pour le KV, 1230 GiB agreges
#   KV fp8_e4m3 = 46.4 KiB/token -> 46.4 GiB pour 1M tokens, 371 GiB pour 8 sequences de 1M.
#
# Sous-commandes :
#   preflight    module rootless-docker, daemon joignable, 8 GPU visibles dans le conteneur, disque
#   login        authentifie Hugging Face une fois ; le token va dans $CACHE_DIR/hf/token (600),
#                jamais dans ce fichier ni dans un argv
#   pull         image vllm/vllm-openai:latest (9.2 GB compresses, ~25 GB sur disque), digest resolu note
#   capabilities vllm/torch du conteneur, et verification que CHAQUE flag utilise existe dans
#                `vllm serve --help` (a faire avant un chargement a froid de 704 GiB)
#   download     checkpoint 756 GB dans $MODEL_DIR ; relancable, reprend ou il s'est arrete
#   serve        lance le serveur, attend le demarrage, imprime le temps de chargement
#   smoke        une requete chat + debit mesure
#   args         les flags exacts du conteneur en cours (verifier tool-call-parser / auto-tool-choice)
#   stop         arrete le conteneur
#
# curiosity n'a pas de docker dans le PATH : c'est le module rootless-docker/1.75 (Bright, Slurm 25.05).
# Le script le charge tout seul ; sinon : module load rootless-docker
# TRACE=0 pour desactiver l'echo de chaque commande docker (par defaut TRACE=1, sur stderr).
#
# Usage dans une allocation (interactive ou via sbatch) :
#   export MODEL_DIR=/chemin/partage/models/GLM-5.3 CACHE_DIR=/chemin/partage/glm53-cache
#   ./glm53_b300_first_run.sh preflight capabilities
#   ./glm53_b300_first_run.sh download          # une seule fois, long
#   ./glm53_b300_first_run.sh serve             # garde le shell occupe
#   ./glm53_b300_first_run.sh smoke             # depuis un autre shell du meme noeud
set -euo pipefail

# Tag flottant volontaire (demande utilisateur). Au 27/10/2026, latest == v0.32.0, meme digest
# sha256:6ee133356d49ed24cee03dd0cda3d3980279a11c107ce3e752c89e76540f41c9 (CUDA 13.0, sm_103 inclus).
# Pour figer : IMAGE=vllm/vllm-openai@sha256:6ee1333... ./glm53_b300_first_run.sh ...
# Garde-fou : si latest passe un jour en CUDA 13.4, il exigera un pilote plus recent que 580.126 ;
# `capabilities` affiche torch.version.cuda, c'est la que ca se voit.
IMAGE=${IMAGE:-vllm/vllm-openai:latest}
HF_REPO=${HF_REPO:-zai-org/GLM-5.3}
MODEL_DIR=${MODEL_DIR:-$PWD/models/GLM-5.3}
CACHE_DIR=${CACHE_DIR:-$PWD/glm53-cache}
CONTAINER=${CONTAINER:-glm53}
SERVED=${SERVED:-glm-5.3}
PORT=${PORT:-8000}
GPUS=${GPUS:-4}
MAX_LEN=${MAX_LEN:-}
MAX_SEQS=${MAX_SEQS:-}
MAX_BATCHED=${MAX_BATCHED:-8192}
GPU_UTIL=${GPU_UTIL:-}
MTP_TOKENS=${MTP_TOKENS:-5}
# Calibrage automatique selon l'allocation (KV fp8 = 46.4 KiB/token ; buffer du indexer = batché x ctx x 2 B) :
#   GPUS >= 8 : 88.0 GiB de poids/carte (36 % du budget a util 0.90) -> 1M de contexte, 8 seqs, confortable.
#   GPUS = 4  : 175.9 GiB de poids/carte (69 % du budget a util 0.95) -> ~79 GiB/carte libres, contexte 1M
#               a 3 seqs (KV 34.8 + indexer 32 = 66.8 GiB/carte, ~12 GiB de marge ; a 4 seqs, ~0.8 GiB).
#               util montee a 0.95 pour recuperer du matelas KV sur des cartes deja a 65 %. Repli plus
#               confortable : MAX_LEN=524288 -> 4 seqs (KV 23.2 + indexer 16 = 39.2 GiB/carte).
if [[ -z "$MAX_LEN" ]]; then
    MAX_LEN=1048576   # 1M sur 4 et 8 cartes ; le calibrage MAX_SEQS ci-dessous absorbe la difference
fi
if [[ -z "$GPU_UTIL" ]]; then
    if (( GPUS >= 8 )); then GPU_UTIL=0.90; else GPU_UTIL=0.95; fi
fi
if [[ -z "$MAX_SEQS" ]]; then
    if (( GPUS >= 8 )); then MAX_SEQS=8
    elif (( MAX_LEN >= 1048576 )); then MAX_SEQS=3
    elif (( MAX_LEN >= 512000 )); then MAX_SEQS=4
    else MAX_SEQS=8; fi
fi
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-5400}
HELP_FILE=$CACHE_DIR/vllm-serve-help.txt
PARSERS_FILE=$CACHE_DIR/vllm-parsers.json
# Surcharges explicites si la resolution automatique se trompe :
TOOL_PARSER=${TOOL_PARSER:-}
REASONING_PARSER=${REASONING_PARSER:-}

# GPUS est declare plus haut (defaut 4) ; TP = EP = GPUS. Tresors du choix GPUS :
#   4 = TP4 : 175.9 GiB de poids/carte (73 % du budget), ~66 GiB/carte libres -> contexte auto 1M, 3 seqs
#   8 = TP8 : 88.0 GiB de poids/carte (36 % du budget) la forme verifyue par la recette constructeur
TP=${TP:-$GPUS}

# Flags valides pour GLM-5.3 (recette constructeur zai-org/GLM-5.3.yaml, b300 = verified, TEP8).
BASE_FLAGS=(--tensor-parallel-size "$TP" --enable-expert-parallel --kv-cache-dtype fp8_e4m3
            --max-model-len "$MAX_LEN" --max-num-seqs "$MAX_SEQS"
            --max-num-batched-tokens "$MAX_BATCHED" --enable-chunked-prefill
            --gpu-memory-utilization "$GPU_UTIL"
            --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_TOKENS}")

say() { printf '\n== %s\n' "$*"; }
die() { printf '\nERREUR: %s\n' "$*" >&2; exit 1; }

# Trace des commandes lancees : TRACE=1 par defaut. Toute commande docker passe par drun(), qui
# imprime la ligne exacte sur stderr (jamais sur stdout, pour ne pas polluer les pipelines) puis
# l'execute. Mettre TRACE=0 pour un mode silencieux.
TRACE=${TRACE:-1}
xtrace() { [[ "${TRACE:-1}" == 1 ]] && printf '+ %s\n' "$*" >&2; }
drun() { xtrace docker "$@"; docker "$@"; }

# curiosity : pas de docker dans le PATH par defaut, c'est le module rootless-docker/1.75.
ensure_docker() {
    command -v docker >/dev/null && return 0
    local init
    for init in /usr/share/lmod/lmod/init/bash /etc/profile.d/modules.sh /usr/share/Modules/init/bash; do
        [[ -r "$init" ]] && . "$init" && break
    done
    if command -v module >/dev/null; then
        module load rootless-docker 2>/dev/null || module load rootless-docker/1.75 2>/dev/null || true
    fi
    command -v docker >/dev/null || die 'docker introuvable : module load rootless-docker, puis relancer'
}

# Demon rootful : tourner avec notre uid garde les fichiers des bind mounts a nous. Rootless mappe deja.
user_args() {
    if docker info --format '{{json .SecurityOptions}}' 2>/dev/null | grep -qi rootless; then return 0; fi
    printf -- '--user %s:%s' "$(id -u)" "$(id -g)"
}
mounts() { printf -- '-v %s:/models -v %s:/cache' "$MODEL_DIR" "$CACHE_DIR"; }
cache_env() {
    printf -- '-e HOME=/tmp -e HF_HOME=/cache/hf -e VLLM_CACHE_ROOT=/cache/vllm -e TRITON_CACHE_DIR=/cache/triton'
    printf -- ' -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor -e XDG_CACHE_HOME=/cache/xdg'
    printf -- ' -e FLASHINFER_WORKSPACE_BASE=/cache/flashinfer -e HF_XET_HIGH_PERFORMANCE=1'
}
# shellcheck disable=SC2046
# vLLM 0.32 inferce le device (cuda) au moment du PARSING des args : toute commande conteneur qui
# touche a vllm doit voir les GPU, sinon « Failed to infer device type ». GPU_RUN_ARGS est resolu
# par capabilities/serve avant le premier appel.
in_image() { drun run --rm $(user_args) $(gpu_env) ${GPU_RUN_ARGS:+$GPU_RUN_ARGS} $(mounts) $(cache_env) --entrypoint "$1" "$IMAGE" "${@:2}"; }

# Restreindre le conteneur aux $GPUS premiers GPU si l'hote en expose davantage. Slurm met deja
# CUDA_VISIBLE_DEVICES (cas --gres=gpu:4) : on le transmet tel quel, ne rien deviner.
gpu_env() {
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        printf -- '-e CUDA_VISIBLE_DEVICES=%s' "$CUDA_VISIBLE_DEVICES"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        local vis
        vis=$(nvidia-smi -L 2>/dev/null | wc -l)
        if (( vis > GPUS )); then
            local ids
            ids=$(seq -s, 0 $((GPUS - 1)))
            printf 'GPU hotes %s > GPUS=%s : restreint a CUDA_VISIBLE_DEVICES=%s\n' "$vis" "$GPUS" "$ids" >&2
            printf -- '-e CUDA_VISIBLE_DEVICES=%s' "$ids"
        fi
    fi
}

# Resolution du mode GPU du demon, une seule fois : --gpus all (rootful, toolkit), --runtime=nvidia,
# ou CDI --device nvidia.com/gpu=all (rootless). Appel DANS le shell principal (jamais dans $( ) :
# un echec doit remonter via `gpu_args || die`). Sur echec total : diagnostic sur STDERR (+ tentative
# d'auto-config du runtime nvidia dans le daemon.json rootless si nvidia-ctk existe), retour 1.
GPU_RUN_ARGS=''
GPU_FIX_TRIED=''
gpu_args() {
    [[ -n "$GPU_RUN_ARGS" ]] && return 0
    local cand
    for cand in '--gpus all' '--runtime=nvidia' '--device nvidia.com/gpu=all'; do
        # shellcheck disable=SC2086
        if docker run --rm $(user_args) $cand --entrypoint nvidia-smi "$IMAGE" -L >/dev/null 2>&1; then
            GPU_RUN_ARGS=$cand
            printf 'GPU_OK mode conteneur : %s\n' "$cand" >&2
            return 0
        fi
    done
    printf 'GPU_EVAL echec : runtimes enregistres %s\n' \
        "$(docker info --format 'runtimes : {{json .Runtimes}}' 2>/dev/null || docker info --format 'runtimes inconnus')" >&2
    if command -v nvidia-ctk >/dev/null 2>&1; then
        printf 'GPU_EVAL CDI trouves (nvidia-ctk cdi list) :\n' >&2
        nvidia-ctk cdi list 2>&1 | head -5 >&2
    else
        printf 'GPU_EVAL nvidia-ctk absent : toolkit NVIDIA non installe\n' >&2
    fi
    # Tentative d'auto-config, UNE fois : enregistre le runtime nvidia pour CE daemon rootless
    # (n'ecrit que ~/.config/docker/daemon.json, aucun droit admin requis). Un daemon rootless
    # relu daemon.json seulement au redemarrage : si on vient d'ecrire, le caller doit le dire.
    if [[ -z "$GPU_FIX_TRIED" ]] && command -v nvidia-ctk >/dev/null 2>&1 && [[ -n "${HOME:-}" ]]; then
        GPU_FIX_TRIED=1
        local cfg="${XDG_CONFIG_HOME:-$HOME/.config}/docker/daemon.json"
        printf 'GPU_FIX tentative d''auto-config dans %s\n' "$cfg" >&2
        mkdir -p "$(dirname "$cfg")"
        if grep -qs 'nvidia-container-runtime\|nvidia-ctk' "$cfg" 2>/dev/null; then
            printf 'GPU_FIX daemon.json deja configure, mais le daemon ne l''utilise pas encore : relancez le daemon rootless.\n' >&2
        else
            if nvidia-ctk runtime configure --runtime=docker --config="$cfg" >&2 2>&1; then
                printf 'GPU_FIX daemon.json ecrit. RELANCEZ LE DAEMON ROOTLESS puis relancez ce script :\n' >&2
                printf '  module load rootless-docker\n  dockerd-rootless-setuptool.sh stop && dockerd-rootless-setuptool.sh start\n' >&2
                printf '  docker run --rm --gpus all %s nvidia-smi -L   # verif, doit lister 4 cartes\n' "$IMAGE" >&2
            else
                printf 'GPU_FIX echec de configuration (voir ligne precedente)\n' >&2
            fi
        fi
    fi
    return 1
}

preflight() {
    say 'runtime conteneur'
    ensure_docker
    docker info >/dev/null 2>&1 || die "demon docker injoignable pour l'uid $(id -u) : module charge mais daemon arrete (rootlesskit), ou DOCKER_HOST incorrect"
    docker info --format 'Server {{.ServerVersion}} / driver {{.Driver}} / {{json .SecurityOptions}}'
    local root
    root=$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || true)
    printf 'DockerRootDir: %s\n' "${root:-inconnu}"
    if [[ -n "$root" ]]; then
        df -h "$root" 2>/dev/null | tail -1 || true
    fi

    say 'GPU hote'
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
    local count
    count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
    [[ "$count" == "$GPUS" ]] || printf 'ATTENTION: %s GPU visibles, GPUS=%s (CUDA_VISIBLE_DEVICES ?)\n' "$count" "$GPUS"

    say 'GPU dans le conteneur'
    gpu_args || die 'aucune methode GPU ne fonctionne (diagnostic ci-dessus) : si le daemon.json vient d etre ecrit, relance le daemon rootless puis relance ce script'
    local gpu_mode=$GPU_RUN_ARGS
    drun run --rm $(user_args) $gpu_mode --entrypoint nvidia-smi "$IMAGE" -L 2>&1 | tail -3
    printf 'mode GPU retenu pour ce run : %s\n' "$gpu_mode"

    say 'disque'
    df -h "$(dirname "$MODEL_DIR")" 2>/dev/null | tail -2 || true
    printf 'checkpoint attendu : 755.6 GB (703.7 GiB) + image ~25 GB\n'
}

pull() {
    mkdir -p "$CACHE_DIR"
    say "image $IMAGE"
    if [[ "$IMAGE" == *"@sha256:"* ]]; then
        drun image inspect "$IMAGE" >/dev/null 2>&1 || drun pull "$IMAGE"
    else
        drun pull "$IMAGE"   # tag flottant : on veut la version du jour, pas celle du cache local
    fi
    local id
    id=$(drun image inspect -f '{{index .RepoDigests 0}}' "$IMAGE" 2>/dev/null || drun image inspect -f '{{.Id}}' "$IMAGE")
    printf '%s\n' "$id" | tee "$CACHE_DIR/image-digest.txt"
}

capabilities() {
    mkdir -p "$CACHE_DIR"
    pull
    say 'versions dans le conteneur'
    gpu_args || die 'aucune methode GPU ne fonctionne (diagnostic ci-dessus) : si le daemon.json vient d etre ecrit, relance le daemon rootless puis relance ce script'
    drun run --rm $(user_args) $(gpu_env) $GPU_RUN_ARGS $(mounts) $(cache_env) --entrypoint python3 "$IMAGE" -c \
        'import torch, vllm; print("vllm", vllm.__version__, "torch", torch.__version__, "cuda", torch.version.cuda, "cc", torch.cuda.get_device_capability(), "devices", torch.cuda.device_count())'
    say "vllm serve --help=all -> $HELP_FILE"
    # vLLM 0.32 : `--help` n'affiche qu'un menu de sections (Config Groups). La liste complete des
    # flags est derriere `--help=all` — sans quoi aucun grep de flag ne retombe sur quoi que ce soit.
    # Les GPU sont passes (device inference au parsing, cf. in_image) ; 2>&1 pour ne rien perdre.
    drun run --rm $(user_args) $(gpu_env) $GPU_RUN_ARGS $(mounts) $(cache_env) --entrypoint sh "$IMAGE" -c \
        'vllm serve --help=all > /cache/vllm-serve-help.txt 2>&1' \
        || die 'vllm serve --help=all a echoue (voir la commande tracee et l erreur ci-dessus)'
    [[ -s "$HELP_FILE" ]] || die 'vllm serve --help a produit un fichier vide'
    wc -l <"$HELP_FILE" | xargs printf 'help: %s lignes\n'
    # Un flag inconnu fait tomber le serveur APRES le chargement des 704 GiB : on verifie avant.
    # BASE_FLAGS + les flags de parseurs (qui ne vivent pas dans arg_utils mais dans le CLI).
    local missing=()
    local flag
    for flag in "${BASE_FLAGS[@]}" --tool-call-parser --enable-auto-tool-choice --reasoning-parser; do
        [[ "$flag" == --* ]] || continue
        grep -q -- "$flag" "$HELP_FILE" || missing+=("$flag")
    done
    if (( ${#missing[@]} )); then
        printf 'FLAGS ABSENTS de cette image : %s\n' "${missing[*]}"
        printf -- '--- %s : %s lignes, debut du fichier :\n' "$HELP_FILE" "$(wc -l <"$HELP_FILE")"
        head -12 "$HELP_FILE"
        local f hit
        for f in "${missing[@]}"; do
            [[ "$f" == --* ]] || continue
            hit=$(grep -m1 -h -- "$f" "$HELP_FILE")
            printf '  %-32s : %s\n' "$f" "${hit:-INTOUVABLE dans le help}"
        done
        printf -- '--- fin du dump (collez ce pavé si un flag reel semble absent de l image)\n'
        return 1
    fi
    printf 'tous les flags du profil existent dans cette image\n'
    say 'parseurs reellement enregistres dans l image'
    in_image python3 -c 'import json
from vllm.tool_parsers import ToolParserManager as T
from vllm.reasoning import ReasoningParserManager as R
def list_reg(M, *attrs):
    # Canonique : list_registered() (union tool_parsers + lazy_parsers). Repli : lecture des dicts.
    try:
        v = M.list_registered()
        if v:
            return sorted(v)
    except Exception:
        pass
    out = set()
    for a in attrs:
        d = getattr(M, a, None)
        if isinstance(d, dict):
            out |= set(d.keys())
    return sorted(out)
print(json.dumps({"tool": list_reg(T, "tool_parsers", "lazy_parsers"),
                  "reasoning": list_reg(R, "reasoning_parsers", "tool_parsers", "lazy_parsers")}))' >"$PARSERS_FILE" \
        || { rm -f "$PARSERS_FILE"; printf 'ATTENTION: registre des parseurs illisible\n'; }
    if [[ -s "$PARSERS_FILE" ]]; then
        python3 -c 'import json,sys
d=json.load(open(sys.argv[1]))
print("tool     :", ",".join(d.get("tool") or []) or "(vide)")
print("reasoning:", ",".join(d.get("reasoning") or []) or "(vide)")' "$PARSERS_FILE" \
            || { rm -f "$PARSERS_FILE"; printf 'ATTENTION: registre des parseurs illisible, bascule sur le grep du help\n'; }
    fi
}

# Resolution du nom de parseur : registre du conteneur ($PARSERS_FILE) PUIS grep du help en secours
# (un registre vide n'est pas une preuve d'absence). 
pick_parser() {
    local kind=$1; shift
    local candidate ok
    for candidate in "$@"; do
        ok=''
        if [[ -s "$PARSERS_FILE" ]]; then
            if python3 -c 'import json,sys
sys.exit(0 if sys.argv[3] in (json.load(open(sys.argv[1])).get(sys.argv[2]) or []) else 1)' \
                "$PARSERS_FILE" "$kind" "$candidate" 2>/dev/null; then
                ok=1
            fi
        fi
        if [[ -n "$ok" ]] || grep -q -- "$candidate" "$HELP_FILE" 2>/dev/null; then
            printf '%s' "$candidate"; return 0
        fi
    done
    return 1
}

# Vrai si le fichier de registre est absent, illisible, ou declare une liste VIDE dans les deux
# categories (registre perime de l'ancienne sonde) : il faut le regenerer, pas lui faire confiance.
registry_empty() {
    [[ ! -s "$PARSERS_FILE" ]] && return 0
    python3 -c 'import json,sys
try:
    d = json.load(open(sys.argv[1]))
    sys.exit(0 if not (d.get("tool") or []) and not (d.get("reasoning") or []) else 1)
except Exception:
    sys.exit(0)' "$PARSERS_FILE"
}

# Authentification HF sans faire passer le secret par un fichier du kit : le token est ecrit
# par `hf auth login` DANS le conteneur, donc dans $CACHE_DIR/hf/token (HF_HOME), persistant.
# A faire depuis un shell interactif (le prompt demande le token, il ne s'affiche pas).
login() {
    mkdir -p "$CACHE_DIR/hf" && chmod 700 "$CACHE_DIR/hf"
    if [[ ! -s "$CACHE_DIR/hf/token" && -s "$HOME/.cache/huggingface/token" ]]; then
        install -m 600 "$HOME/.cache/huggingface/token" "$CACHE_DIR/hf/token"
        printf 'token deja present sur le noeud : repris depuis ~/.cache/huggingface/token\n'
    fi
    if [[ ! -s "$CACHE_DIR/hf/token" ]]; then
        printf 'coller le token quand il est demande (rien ne s affiche pendant la saisie)\n'
        drun run --rm -it $(user_args) $(mounts) $(cache_env) --entrypoint hf "$IMAGE" auth login
    fi
    chmod 600 "$CACHE_DIR/hf/token" 2>/dev/null || printf 'note: chmod impossible (fichier ecrit par root dans le conteneur)\n'
    say 'verification'
    drun run --rm $(user_args) $(mounts) $(cache_env) --entrypoint hf "$IMAGE" auth whoami || true
}

download() {
    mkdir -p "$MODEL_DIR" "$CACHE_DIR"
    if compgen -G "$MODEL_DIR/model-*.safetensors" >/dev/null; then
        printf 'checkpoint deja present : %s shards\n' "$(compgen -G "$MODEL_DIR/model-*.safetensors" | wc -l)"
        return 0
    fi
    # zai-org/GLM-5.3 est public (gated=false) : un token n'est pas requis. Il augmente les quotas
    # et le debit. Deux chemins, aucun ne fait passer le secret par un fichier de ce kit :
    #   ./glm53_b300_first_run.sh login     (une fois : ecrit $CACHE_DIR/hf/token, mode 600)
    #   export HF_TOKEN=...                 (transmis au conteneur par nom, -e HF_TOKEN)
    local auth=()
    if [[ -n "${HF_TOKEN:-}" ]]; then
        auth=(-e HF_TOKEN)
        printf 'auth : HF_TOKEN transmis par nom\n'
    elif [[ -s "$CACHE_DIR/hf/token" ]]; then
        printf 'auth : token persistant present dans %s/hf/token\n' "$CACHE_DIR"
    else
        printf 'auth : anonyme (quotas publics). Pour un debit maximal : ./glm53_b300_first_run.sh login\n'
    fi
    say "telechargement $HF_REPO -> $MODEL_DIR (~756 GB, relancable)"
    drun run --rm $(user_args) $(mounts) $(cache_env) "${auth[@]}" \
        --entrypoint python3 "$IMAGE" -c 'import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], local_dir="/models", max_workers=16)
print("DOWNLOAD_OK")' "$HF_REPO"
}

serve() {
    # NB : registry_empty est une COMMANDE, donc hors du [[ ]] (sinon bash la traite comme une
    # chaine toujours-vraie et refuse toute mise en cache).
    if [[ ! -s "$HELP_FILE" || ! -s "$PARSERS_FILE" ]] || registry_empty; then
        say 'capabilities manquantes ou registre des parseurs perime, je les refais'
        capabilities || die 'capabilities a echoue (voir le pavé ci-dessus) : rien n est lance'
    fi
    local flags=("${BASE_FLAGS[@]}")
    local tool reasoning
    if [[ -n "$TOOL_PARSER" ]]; then
        tool=$TOOL_PARSER
    else
        # Le registre 0.32 declare "glm47" et "glm45" (plus de "glm47_moe", legacy 0.13).
        tool=$(pick_parser tool glm47 glm45 glm47_moe glm4 glm4_moe) || {
            printf 'registre : %s\n' "$(cat "$PARSERS_FILE" 2>/dev/null || echo inconnu)"
            die "aucun parseur d'outils glm dans cette image ; relancer avec TOOL_PARSER=<nom du registre>"
        }
    fi
    # Non negociable : sans ces deux flags, toute requete du client avec tool_choice "auto" est
    # rejetee (« auto tool choice requires --enable-auto-tool-choice and --tool-call-parser »).
    flags+=(--tool-call-parser "$tool" --enable-auto-tool-choice)
    printf 'tool-call-parser: %s\n' "$tool"

    if [[ -n "$REASONING_PARSER" ]]; then
        reasoning=$REASONING_PARSER
    elif reasoning=$(pick_parser reasoning glm47 glm45 glm47_moe_reasoning_parser); then
        :
    else
        reasoning=''
        printf 'ATTENTION: aucun parseur de raisonnement glm resolu, passee sans\n'
    fi
    if [[ -n "$reasoning" ]]; then
        flags+=(--reasoning-parser "$reasoning")
        printf 'reasoning-parser: %s\n' "$reasoning"
    fi
    printf 'flags: %s\n' "${flags[*]}"

    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    say "TP $TP + EP, ctx $MAX_LEN, $MAX_SEQS seqs, util $GPU_UTIL, KV fp8"
    # Pas de >/dev/null ici : l'id du conteneur ET toute erreur du docker run lui-meme restent visibles.
    gpu_args || die 'aucune methode GPU ne fonctionne (diagnostic ci-dessus) : si le daemon.json vient d etre ecrit, relance le daemon rootless puis relance ce script'
    drun run -d --name "$CONTAINER" $(user_args) $(gpu_env) $GPU_RUN_ARGS \
        --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
        $(mounts) $(cache_env) \
        -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e VLLM_NO_USAGE_STATS=1 \
        -e NCCL_ALGO=Ring -e NCCL_PROTO=Simple \
        -p "$PORT:$PORT" --entrypoint vllm "$IMAGE" serve /models \
        --served-model-name "$SERVED" "${flags[@]}" --host 0.0.0.0 --port "$PORT" \
        || die 'docker run a echoue a l instant (commande tracee et erreur ci-dessus)'
    printf 'conteneur %s lance ; chargement a froid de 704 GiB, compter 20-40 min\n' "$CONTAINER"
    local start try=0
    start=$(date +%s)
    while :; do
        if (( $(date +%s) - start > STARTUP_TIMEOUT )); then
            docker logs --tail 200 "$CONTAINER" >&2
            die "moteur pas pret apres ${STARTUP_TIMEOUT}s (fin des logs ci-dessus)"
        fi
        if ! docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -q true; then
            docker logs --tail 200 "$CONTAINER" >&2
            die 'conteneur termine avant de servir (fin des logs ci-dessus)'
        fi
        docker logs "$CONTAINER" 2>&1 | grep -q 'Application startup complete' && break
        if (( try % 6 == 0 )); then
            printf '  [%ss] %s\n' "$(( $(date +%s) - start ))" \
                "$(docker logs "$CONTAINER" 2>&1 | grep -v '^$' | tail -1)"
        fi
        (( try += 1 )) || true
        sleep 15
    done
    printf 'DEMARRE en %ss\n' "$(( $(date +%s) - start ))"
    docker logs "$CONTAINER" 2>&1 | grep -E 'KV cache size|maximum concurrency|Available KV cache memory|graph' | tail -6 || true
    curl -fsS "http://127.0.0.1:$PORT/v1/models"; printf '\n'
}

smoke() {
    say 'requete chat (debit decode, borne haute incluant le prefill)'
    local body resp t0 t1
    body=$(printf '{"model":"%s","messages":[{"role":"user","content":"Explique en trois phrases pourquoi le ciel est bleu."}],"max_tokens":256,"temperature":0}' "$SERVED")
    t0=$(date +%s%N)
    resp=$(curl -fsS -m 900 -H 'Content-Type: application/json' -d "$body" "http://127.0.0.1:$PORT/v1/chat/completions")
    t1=$(date +%s%N)
    python3 - "$resp" "$t0" "$t1" <<'PY'
import json, sys
r = json.loads(sys.argv[1]); u = r.get('usage', {})
dt = (int(sys.argv[3]) - int(sys.argv[2])) / 1e9
ct = u.get('completion_tokens') or 0
print('reponse:', (r['choices'][0]['message'].get('content') or '')[:300].replace('\n', ' '))
print('tokens: prompt %s / completion %s en %.1fs -> %.1f tok/s' %
      (u.get('prompt_tokens'), ct, dt, ct / dt if dt else 0))
PY
    say 'tool-call auto (valide --enable-auto-tool-choice et --tool-call-parser)'
    local tbody tresp
    tbody=$(printf '%s' "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Quel temps fait-il a Lyon ?\"}],\"max_tokens\":128,\"temperature\":0,\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"weather\",\"description\":\"meteo d une ville\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}],\"tool_choice\":\"auto\"}")
    tresp=$(curl -fsS -m 900 -H 'Content-Type: application/json' -d "$tbody" "http://127.0.0.1:$PORT/v1/chat/completions")
    printf '%s' "$tresp" | python3 -c '
import json, sys
r = json.load(sys.stdin); m = r["choices"][0]["message"]
print("tool_calls:", json.dumps(m.get("tool_calls")))
print("content   :", (m.get("content") or "").strip()[:200].replace("\n", " "))'
}

case "${1:-}" in
    preflight) preflight ;;
    pull) ensure_docker; pull ;;
    capabilities) ensure_docker; capabilities ;;
    login) ensure_docker; login ;;
    download) ensure_docker; download ;;
    serve) ensure_docker; serve ;;
    smoke) smoke ;;
    args) ensure_docker; drun inspect -f '{{json .Args}}' "$CONTAINER" 2>/dev/null | python3 -m json.tool \
            || die "conteneur $CONTAINER introuvable" ;;
    stop) ensure_docker; drun rm -f "$CONTAINER" ;;
    *) sed -n '2,31p' "$0"; exit 1 ;;
esac
