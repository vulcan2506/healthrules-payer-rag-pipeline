#!/usr/bin/env bash
# Starts llama.cpp server with Qwen3.5-9B Q4_K_M, 4 parallel slots.
# Same model file as MTP build — MTP head unused.
# 4 parallel slots: n_ctx_slot = 49152/4 = 12288 tokens/slot, covering the
# largest observed prompt (8885 tokens) + response budget (700) with margin.
# VRAM: measured live — 6525 MiB used @ -c 49152 -np 4, 1318 MiB free on 8GB card.

MODEL="$HOME/.models/Qwen3.5-9B-Q4_K_M.gguf"
SERVER="$HOME/llama.cpp/build/bin/llama-server"

if [ ! -f "$SERVER" ]; then
    echo "llama-server not found at $SERVER"
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "Model not found at $MODEL"
    echo "Run: python3 -c \"from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Qwen3.5-9B-MTP-GGUF', 'Qwen3.5-9B-Q4_K_M.gguf', local_dir='\$HOME/.models/')\""
    exit 1
fi

echo "Starting Qwen3.5-9B (Q4_K_M) — 4 parallel slots, 12288 tokens/slot — RTX 3070 Ti (8GB VRAM)"

exec "$SERVER" \
    --model          "$MODEL" \
    --alias          qwen35-9b \
    -ngl             99 \
    -c               49152 \
    --flash-attn     on \
    --cache-type-k   q8_0 \
    --cache-type-v   q8_0 \
    -b               1024 \
    -ub              512 \
    --cont-batching \
    -np              4 \
    --host           127.0.0.1 \
    --port           8080
