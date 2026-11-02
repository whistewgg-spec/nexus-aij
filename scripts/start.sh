#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -x .venv/bin/python || ! -x .llama.cpp/build/bin/llama-server || ! -d frontend/node_modules ]]; then
  echo "Dependencies are missing. Run: bash scripts/setup.sh" >&2
  exit 1
fi

if [[ -f .env ]]; then
  set -a
  # .env is a trusted local shell configuration file. Never commit it.
  source .env
  set +a
fi

MODEL_PATH="${NEXUS_MODEL_PATH:-$ROOT/data/models/Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf}"
if [[ ! -s "$MODEL_PATH" ]]; then
  echo "Downloading the public Qwen2.5-Coder 3B GGUF model (~1.8 GiB)..."
  .venv/bin/python scripts/download-model.py
fi
export NEXUS_MODEL_PATH="$MODEL_PATH"

DETECTED_RESOURCES="$(PYTHONPATH="$ROOT/backend" "$ROOT/.venv/bin/python" -c 'from app.main import resource_info; r = resource_info(); print(r["context_tokens"], r["cpu_count"])')"
read -r DETECTED_CONTEXT CPU_COUNT <<< "$DETECTED_RESOURCES"
CONTEXT_TOKENS="${NEXUS_CONTEXT_TOKENS:-$DETECTED_CONTEXT}"
if [[ ! "$CONTEXT_TOKENS" =~ ^[0-9]+$ ]] || [[ "$CONTEXT_TOKENS" -lt 512 ]] || [[ "$CONTEXT_TOKENS" -gt 32768 ]]; then
  echo "NEXUS_CONTEXT_TOKENS must be an integer from 512 to 32768 (the selected model limit)." >&2
  exit 1
fi
THREADS="$CPU_COUNT"
if [[ "$THREADS" -gt 8 ]]; then THREADS=8; fi
export NEXUS_CONTEXT_TOKENS="$CONTEXT_TOKENS"

cleanup() {
  if [[ -n "${LLAMA_PID:-}" ]] && kill -0 "$LLAMA_PID" 2>/dev/null; then
    kill "$LLAMA_PID" 2>/dev/null || true
    wait "$LLAMA_PID" 2>/dev/null || true
  fi
  if [[ -n "${API_PID:-}" ]] && kill -0 "$API_PID" 2>/dev/null; then
    kill "$API_PID" 2>/dev/null || true
    wait "$API_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

LLAMA_PID=""
if curl -fsS http://127.0.0.1:8081/health >/dev/null 2>&1; then
  echo "Using the llama.cpp server already listening on port 8081."
else
  .llama.cpp/build/bin/llama-server \
    --model "$MODEL_PATH" \
    --host 127.0.0.1 \
    --port 8081 \
    --ctx-size "$CONTEXT_TOKENS" \
    --threads "$THREADS" \
    --threads-batch "$THREADS" \
    --parallel 1 &
  LLAMA_PID=$!

  READY=0
  for _ in $(seq 1 180); do
    if curl -fsS http://127.0.0.1:8081/health >/dev/null 2>&1; then
      READY=1
      break
    fi
    if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
      echo "llama.cpp exited before becoming ready. Check model path, RAM, and model download." >&2
      wait "$LLAMA_PID" || true
      exit 1
    fi
    sleep 1
  done
  if [[ "$READY" -ne 1 ]]; then
    echo "llama.cpp did not become ready within 180 seconds." >&2
    exit 1
  fi
fi

export NEXUS_API_TOKEN="${NEXUS_API_TOKEN:-}"
.venv/bin/python -m uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000 &
API_PID=$!
npm --prefix frontend run dev
