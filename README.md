# NEXUS AI — Codespaces Edition

NEXUS runs inference on the CPU in your Codespace with llama.cpp. The web app and FastAPI backend talk to that local model; GitHub Actions is not used for inference. This first delivery is intentionally small: one local GGUF model, project-scoped conversation history in SQLite, and streamed chat.

## Start in GitHub Codespaces

1. Create a Codespace for this repository. The dev container forwards the web app on **5173** and API on **8000**, both private by default.
2. The dev container runs `bash scripts/setup.sh` automatically after creation. If setup did not run, start it from the Codespace terminal. It installs Python and frontend dependencies and builds llama.cpp for CPU. It requires network access, a C++ compiler, CMake, Node.js, Python 3, and Git.
3. Optionally configure a bearer token before exposing anything beyond private forwarded ports:

   ```bash
   cp .env.example .env
   openssl rand -hex 32
   ```

   Put the generated value in `NEXUS_API_TOKEN` in `.env`. The frontend asks for this token in the browser; it stays in that tab's `sessionStorage`. Keep forwarded ports private and never commit `.env`.
4. Run `bash scripts/start.sh`. On first start it downloads the public Qwen2.5 0.5B Q4_K_M GGUF (about 400 MiB) into `data/models/`, checks the file size and verifies its SHA-256 against the pinned checksum. It then starts llama.cpp, FastAPI, and Vite.
5. Open the **NEXUS AI web app** forwarded port. Send a message; the status indicator reports whether the local model is actually reachable. Conversations and messages persist in `data/nexus.sqlite3`, separated by project ID.

`data/` and `.llama.cpp/` are local/Codespace data and are ignored by Git. Codespace storage persists while the Codespace is retained; deleting the Codespace also deletes its local database and downloaded model unless you back them up.

## Local development and tests

On Debian/Ubuntu, install the system tools with `sudo apt-get update && sudo apt-get install -y build-essential cmake git curl python3-venv`; ensure Node.js 22+ is installed. Then run:

```bash
bash scripts/setup.sh
bash scripts/test.sh
bash scripts/start.sh
```

The test script runs backend API tests and a TypeScript production build; neither test nor build downloads a model. Local non-Linux inference is not configured by these scripts; the supported runtime is a Linux Codespace.

## Resources, configuration, and limits

- The default model is the 0.5B Q4_K_M GGUF from `bartowski/Qwen2.5-0.5B-Instruct-GGUF` on Hugging Face, pinned to revision `41ba88dbac95fed2528c92514c131d73eb5a174b`. It is downloaded on first launch, not stored in Git. The downloader checks the exact 397,808,192-byte size and a pinned SHA-256 (`6eb923e7d26e9cea28811e1a8e852009b21242fb157b26149d3b188f3a8c8653`) before installing the file.
- The selected Qwen2.5-0.5B model supports a maximum context of 32,768 tokens (`max_position_embeddings`). There is no truly unlimited context: the prompt, conversation history, and generated reply share that model limit and available RAM. Startup detects available memory and CPU limits (including Linux cgroups), choosing a conservative context up to 32,768; override it with `NEXUS_CONTEXT_TOKENS=32768` to use the model's full context (requires more RAM). Supported values are 512–32,768. CPU inference uses at most eight threads and one parallel request. Chat generation no longer has the previous 512-token answer cap; it continues until the model ends its answer or the context limit is reached.
- `NEXUS_MODEL_PATH`, `NEXUS_DATA_DIR`, `NEXUS_INFERENCE_URL`, `NEXUS_CONTEXT_TOKENS`, and `NEXUS_API_TOKEN` can be set in the shell or `.env`. The backend exposes `/api/health`, `/api/auth/status`, and `/api/capabilities`. Authentication is optional for localhost/private development; set a strong random token before changing forwarded-port visibility. The inference server listens only on loopback.
- A 0.5B model is intentionally modest and may be slow or less capable than hosted services. Real inference requires a successful llama.cpp build and model download plus enough Codespace memory. If startup cannot launch the model, NEXUS reports the model as unavailable instead of substituting a fake response.
- Project IDs separate conversations in SQLite. This delivery does not yet include file indexing, semantic memory, project files, WebSocket streaming, desktop/Tauri clients, or identity-provider authentication.

## API

- `GET /api/health` and `GET /api/auth/status` are available without a token.
- `GET /api/capabilities` reports detected resources and actual local inference/model availability.
- `GET/POST /api/conversations?project_id=...` lists or creates project-scoped conversations.
- `GET /api/conversations/{id}/messages?project_id=...` loads persistent history.
- `POST /api/conversations/{id}/messages/stream` streams SSE events (`token`, `done`, or `error`) while storing successful turns.
