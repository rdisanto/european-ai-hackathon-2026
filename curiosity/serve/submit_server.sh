#!/bin/bash
set -euo pipefail
umask 077
SOURCE="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
exec python3 "$SOURCE/inference.py" submit "$@"
