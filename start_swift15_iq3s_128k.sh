#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/ai/llama-swift15"
MODEL="$ROOT/models/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"
SERVER="$ROOT/llama.cpp-runtime/llama-server"
CONTEXT="${SWIFT_CONTEXT:-131072}"
# Direct I/O avoids populating the filesystem page cache where supported.
LOAD_MODE="${SWIFT_LOAD_MODE:-dio}"
[[ -x "$SERVER" ]] || { echo "Missing executable: $SERVER" >&2; exit 1; }
[[ -f "$MODEL" ]] || { echo "Missing verified model: $MODEL" >&2; exit 1; }
export LD_LIBRARY_PATH="$ROOT/llama.cpp-runtime${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
nvidia-smi
printf 'Model: %s\nContext: %s; KV: q8_0/q8_0; MTP drafts: 3; load mode: %s\n' "$MODEL" "$CONTEXT" "$LOAD_MODE"
exec "$SERVER" \
  -m "$MODEL" --load-mode "$LOAD_MODE" -ngl 999 -fa on -c "$CONTEXT" -np 1 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  --host 127.0.0.1 --port 1235 --jinja \
  --alias swift-1.5-qwen3.8-27b-iq3s-mtp --metrics --perf --log-verbosity 4
