#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

command -v git >/dev/null || { echo "git is required to build llama.cpp." >&2; exit 1; }
command -v cmake >/dev/null || { echo "cmake is required. Install build-essential and cmake, then retry." >&2; exit 1; }
command -v python3 >/dev/null || { echo "Python 3 is required." >&2; exit 1; }
command -v npm >/dev/null || { echo "Node.js and npm are required." >&2; exit 1; }
command -v curl >/dev/null || { echo "curl is required to check the local model server." >&2; exit 1; }

if ! command -v c++ >/dev/null; then
  echo "A C++ compiler is required to build llama.cpp. Install build-essential, then retry." >&2
  exit 1
fi

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r backend/requirements.txt
npm --prefix frontend install

if [[ ! -x .llama.cpp/build/bin/llama-server ]]; then
  if [[ ! -d .llama.cpp/.git ]]; then
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git .llama.cpp
  fi
  cmake -S .llama.cpp -B .llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_CUDA=OFF
  cmake --build .llama.cpp/build --config Release --target llama-server -j 2
fi

echo "NEXUS dependencies are ready. Run: bash scripts/start.sh"
