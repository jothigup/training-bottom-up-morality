#!/usr/bin/env bash
# Download the large files that are deliberately kept out of git.
# Safe to re-run: anything already present is skipped.
#
# Usage:
#   bash scripts/fetch_large_files.sh            # fetch everything
#   bash scripts/fetch_large_files.sh gguf       # only the quantized Qwen model
#   bash scripts/fetch_large_files.sh llamacpp   # only llama.cpp source
#   bash scripts/fetch_large_files.sh resnet     # only the ResNet-34 checkpoint
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GGUF_PATH="artifacts/playground/qwen2.5-1.5b-instruct-q4_k_m.gguf"
GGUF_URL="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"

LLAMA_COMMIT="4227c9be4268ac844921b90f31595f81236bd317"
LLAMA_DIR="artifacts/playground/llama.cpp-${LLAMA_COMMIT}"
LLAMA_URL="https://github.com/ggml-org/llama.cpp/archive/${LLAMA_COMMIT}.tar.gz"

RESNET_PATH="saved_models/tutorial10/hub/checkpoints/resnet34-b627a593.pth"
RESNET_URL="https://download.pytorch.org/models/resnet34-b627a593.pth"

download() {  # url dest
  mkdir -p "$(dirname "$2")"
  echo "  downloading $1"
  curl -fL --retry 3 -o "$2.part" "$1"
  mv "$2.part" "$2"
}

fetch_gguf() {
  echo "Qwen2.5-1.5B-Instruct GGUF (q4_k_m, about 1 GB)"
  if [[ -f "$GGUF_PATH" ]]; then echo "  already present: $GGUF_PATH"; return; fi
  download "$GGUF_URL" "$GGUF_PATH"
}

fetch_llamacpp() {
  echo "llama.cpp source at commit ${LLAMA_COMMIT:0:10}"
  if [[ -d "$LLAMA_DIR" ]]; then echo "  already present: $LLAMA_DIR"; return; fi
  mkdir -p artifacts/playground
  if command -v git >/dev/null 2>&1; then
    git init -q "$LLAMA_DIR"
    git -C "$LLAMA_DIR" fetch -q --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_COMMIT"
    git -C "$LLAMA_DIR" checkout -q FETCH_HEAD
  else
    curl -fL --retry 3 "$LLAMA_URL" | tar -xz -C artifacts/playground
  fi
  echo "  source in $LLAMA_DIR (build it with: cmake -B build && cmake --build build -j)"
}

fetch_resnet() {
  echo "torchvision ResNet-34 checkpoint (saved_models/tutorial10)"
  if [[ -f "$RESNET_PATH" ]]; then echo "  already present: $RESNET_PATH"; return; fi
  download "$RESNET_URL" "$RESNET_PATH"
}

targets=("$@")
[[ ${#targets[@]} -eq 0 ]] && targets=(gguf llamacpp resnet)
for t in "${targets[@]}"; do
  case "$t" in
    gguf) fetch_gguf ;;
    llamacpp) fetch_llamacpp ;;
    resnet) fetch_resnet ;;
    *) echo "Unknown target: $t (use gguf, llamacpp or resnet)" >&2; exit 1 ;;
  esac
done
echo "Done."
