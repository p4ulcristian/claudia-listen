#!/usr/bin/env bash
# Install iris-listen (Canary-1B-v2) as a user service. Safe to run again.
#
#   ./install.sh
#   LISTEN_HOSTS="127.0.0.1 <vpn address>" ./install.sh   # also listen on the VPN
#   LISTEN_LANGUAGES="hu en" ./install.sh                  # clips without a language: Hungarian or English
#
# Needs Linux, an NVIDIA GPU with ~6.5 GB free, uv and python3. Takes ~8 GB of
# disk: the venv here (.venv, onnxruntime with its CUDA) and the models (4 GB, in
# LISTEN_MODELS). Every request needs the bearer token in LISTEN_TOKEN_FILE;
# the first install makes one.
set -euo pipefail
PORT=${LISTEN_PORT:-7861}
HOST_ARGS=$(printf -- '--host %s ' ${LISTEN_HOSTS:-127.0.0.1})
MODELS=${LISTEN_MODELS:-$HOME/.local/share/iris-listen/models}
LANGUAGES=${LISTEN_LANGUAGES:-en}
TOKEN_FILE=${LISTEN_TOKEN_FILE:-$HOME/.config/iris-listen/token}
HERE="$(dirname "$(readlink -f "$0")")"
cd "$HERE"

if [ ! -s "$TOKEN_FILE" ]; then
  mkdir -p "$(dirname "$TOKEN_FILE")"
  (umask 077 && python3 -c 'import secrets; print(secrets.token_urlsafe(32))' > "$TOKEN_FILE")
  echo "Made a token in $TOKEN_FILE"
fi

echo "Setting up the Python environment ..."
# A fresh venv each time, so nothing requirements.txt no longer lists stays (uv's cache makes it quick).
uv venv -q --clear --python 3.12 .venv
uv pip install -q --python .venv/bin/python -r requirements.txt

echo "Fetching Canary-1B-v2 (ONNX, fp32) and Silero VAD into $MODELS (4 GB the first time) ..."
mkdir -p "$MODELS"
.venv/bin/python - "$MODELS" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download("istupakov/canary-1b-v2-onnx", local_dir=f"{sys.argv[1]}/canary-1b-v2",
                  allow_patterns=["config.json", "vocab.txt", "encoder-model.onnx",
                                  "encoder-model.onnx.data", "decoder-model.onnx"])
snapshot_download("istupakov/silero-vad-onnx", local_dir=f"{sys.argv[1]}/silero-vad",
                  allow_patterns=["config.json", "silero_vad.onnx"])
PY

mkdir -p ~/.config/systemd/user
sed -e "s|HERE_DIR|$HERE|g" -e "s|LISTEN_PORT|$PORT|g" -e "s|HOST_ARGS|${HOST_ARGS% }|g" \
    -e "s|MODELS_DIR|$MODELS|g" -e "s|LISTEN_LANGUAGES|$LANGUAGES|g" -e "s|TOKEN_PATH|$TOKEN_FILE|g" \
    "$HERE/iris-listen.service" > ~/.config/systemd/user/iris-listen.service
systemctl --user daemon-reload
systemctl --user enable --now iris-listen.service
systemctl --user restart iris-listen.service
echo "Starting: ~20 s (loading the model). Logs: journalctl --user -u iris-listen -f"
echo "Callers send: Authorization: Bearer \$(cat $TOKEN_FILE)  to http://127.0.0.1:$PORT"
