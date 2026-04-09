#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR/sam2"

PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH="$SCRIPT_DIR" SAMURAI_APP_ROOT="$SCRIPT_DIR/sam2" exec uvicorn api.main:app --host 0.0.0.0 --port 8000 "$@"
