#!/usr/bin/env bash
# Downloads Qwen3-14B-Q4_K_M GGUF from bartowski's HuggingFace repo.

set -e

DEST="$HOME/.models"
FILE="Qwen_Qwen3-14B-Q4_K_M.gguf"
REPO="bartowski/Qwen_Qwen3-14B-GGUF"

mkdir -p "$DEST"

if [ -f "$DEST/$FILE" ]; then
    echo "Already exists: $DEST/$FILE"
    exit 0
fi

echo "Downloading $FILE (~8GB) ..."
huggingface-cli download "$REPO" "$FILE" --local-dir "$DEST"
echo "Done: $DEST/$FILE"
