#!/usr/bin/env bash
# Install iris-listen (Musci-ASR-qwen-1019 + word lists + cleanup) as two user services. Safe to run again.
#
#   ./install.sh
#   LISTEN_HOSTS="127.0.0.1 <vpn address>" ./install.sh   # also listen on the VPN
#   LISTEN_UNIT=iris-listen-v2 LISTEN_PORT=7862 ./install.sh   # beside an older iris-listen
#
# Needs Linux, an NVIDIA GPU (about 4.5 GB for the speech model, 3.7 GB for the cleanup model),
# uv, curl and python3. Takes about 15 GB of disk: the venv here (.venv, torch with its CUDA),
# llama.cpp's CUDA build and the models (LISTEN_MODELS). Every request needs the bearer token in
# LISTEN_TOKEN_FILE; the first install makes one, and a key for the cleanup server beside it.
set -euo pipefail
UNIT=${LISTEN_UNIT:-iris-listen}
CLEANUP_UNIT=${LISTEN_CLEANUP_UNIT:-$UNIT-cleanup}
PORT=${LISTEN_PORT:-7861}
CLEANUP_PORT=${LISTEN_CLEANUP_PORT:-$((PORT + 100))}
HOST_ARGS=$(printf -- '--host %s ' ${LISTEN_HOSTS:-127.0.0.1})
MODELS=${LISTEN_MODELS:-$HOME/.local/share/iris-listen/models}
DATA=${LISTEN_DATA:-$HOME/.local/share/$UNIT}
TOKEN_FILE=${LISTEN_TOKEN_FILE:-$HOME/.config/$UNIT/token}
CLEANUP_KEY=${LISTEN_CLEANUP_KEY_FILE:-$(dirname "$TOKEN_FILE")/cleanup-key}
GPU_FRACTION=${LISTEN_GPU_FRACTION:-0.4}
IDLE_MINUTES=${LISTEN_IDLE_MINUTES:-15}
LLAMA_BUILD=b12451   # llama.cpp release with Gemma 4 support; its CUDA 12.8 build
HERE="$(dirname "$(readlink -f "$0")")"
LLAMA="$HERE/llama/llama-$LLAMA_BUILD"
cd "$HERE"

for f in "$TOKEN_FILE" "$CLEANUP_KEY"; do
  if [ ! -s "$f" ]; then
    mkdir -p "$(dirname "$f")"
    (umask 077 && python3 -c 'import secrets; print(secrets.token_urlsafe(32))' > "$f")
    echo "Made $f"
  fi
done

echo "Setting up the Python environment ..."
uv venv -q --clear --python 3.12 .venv
uv pip install -q --python .venv/bin/python --index-strategy unsafe-best-match -r requirements.txt

if [ ! -x "$LLAMA/llama-server" ]; then
  echo "Fetching llama.cpp $LLAMA_BUILD (CUDA 12.8) ..."
  mkdir -p "$HERE/llama" && cd "$HERE/llama"
  base=https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA_BUILD
  curl -fsSL "$base/llama-$LLAMA_BUILD-bin-ubuntu-cuda-12.8-x64.tar.gz" | tar xz
  curl -fsSL "$base/cudart-llama-$LLAMA_BUILD-bin-ubuntu-cuda-12.8-x64.tar.gz" | tar xz --strip-components=1 -C "llama-$LLAMA_BUILD"
  cd "$HERE"
fi

echo "Fetching Musci-ASR-qwen-1019 and Gemma 4 E4B (Q4_K_M) into $MODELS (about 9 GB the first time) ..."
mkdir -p "$MODELS"
.venv/bin/python - "$MODELS" <<'PY'
import sys
from huggingface_hub import hf_hub_download, snapshot_download
snapshot_download("Musci-research/Musci-ASR-qwen-1019-hf", revision="90caacb67144e9f105dc1b66964789ae803b6b29",
                  local_dir=f"{sys.argv[1]}/musci-asr-qwen-1019")
hf_hub_download("unsloth/gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-Q4_K_M.gguf", local_dir=f"{sys.argv[1]}/gemma")
PY

mkdir -p ~/.config/systemd/user "$DATA"
fill() {
  sed -e "s|HERE_DIR|$HERE|g" -e "s|LISTEN_PORT|$PORT|g" -e "s|HOST_ARGS|${HOST_ARGS% }|g" \
      -e "s|MODELS_DIR|$MODELS|g" -e "s|TOKEN_PATH|$TOKEN_FILE|g" \
      -e "s|DATA_DIR|$DATA|g" -e "s|GPU_FRACTION|$GPU_FRACTION|g" -e "s|IDLE_MINUTES|$IDLE_MINUTES|g" \
      -e "s|IDLE_SECONDS|$((IDLE_MINUTES * 60))|g" -e "s|CLEANUP_PORT|$CLEANUP_PORT|g" \
      -e "s|CLEANUP_KEY_PATH|$CLEANUP_KEY|g" -e "s|CLEANUP_UNIT|$CLEANUP_UNIT|g" -e "s|LLAMA_DIR|$LLAMA|g" "$1"
}
fill "$HERE/iris-listen-cleanup.service" > ~/.config/systemd/user/"$CLEANUP_UNIT".service
fill "$HERE/iris-listen.service" > ~/.config/systemd/user/"$UNIT".service
systemctl --user daemon-reload
systemctl --user enable --now "$CLEANUP_UNIT".service "$UNIT".service
systemctl --user restart "$CLEANUP_UNIT".service "$UNIT".service
echo "Starting: ~30 s (loading the models). Logs: journalctl --user -u $UNIT -u $CLEANUP_UNIT -f"
echo "Callers send: Authorization: Bearer \$(cat $TOKEN_FILE)  to http://127.0.0.1:$PORT"
