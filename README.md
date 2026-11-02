# NEXUS AI — Codespaces Edition

NEXUS runs inference on the CPU in your Codespace with llama.cpp. The web app and FastAPI backend talk to that local model; GitHub Actions is not used for inference. This first delivery is intentionally small: one local GGUF model, project-scoped conversation history in SQLite, streamed chat, text/code file attachments for local-model analysis, and AI-drafted project files that require review before creation.

## Start in GitHub Codespaces

1. Create a Codespace for this repository. The dev container forwards the web app on **5173** and API on **8000**, both private by default. The Vite web server proxies `/api` requests to the local FastAPI server, so the browser only needs the web-app port.
2. The dev container runs `bash scripts/setup.sh` automatically after creation. If setup did not run, start it from the Codespace terminal. It installs Python and frontend dependencies and builds llama.cpp for CPU. It requires network access, a C++ compiler, CMake, Node.js, Python 3, and Git.
3. Optionally configure a bearer token before exposing anything beyond private forwarded ports:

   ```bash
   cp .env.example .env
   openssl rand -hex 32
   ```

   Put the generated value in `NEXUS_API_TOKEN` in `.env`. The frontend asks for this token in the browser; it stays in that tab's `sessionStorage`. Keep forwarded ports private and never commit `.env`.
4. Run `bash scripts/start.sh`. On first start it downloads the public Qwen2.5-Coder 3B Q4_K_M GGUF (about 1.8 GiB) into `data/models/`, checks the file size and verifies its SHA-256 against the pinned checksum. It then starts llama.cpp, FastAPI, and Vite.
5. Open the **NEXUS AI web app** forwarded port. Send a message; the status indicator reports whether the local model is actually reachable. Conversations and messages persist in `data/nexus.sqlite3`, separated by project ID. Use the paperclip button to attach up to three text or code files for the local model to review. Choose **Create file**, enter a relative path, describe what you want, review/edit the model's draft, then explicitly choose **Create file** to save it.

`data/` and `.llama.cpp/` are local/Codespace data and are ignored by Git. User-created project files are saved under `data/projects/<project-id>/`, not in the repository checkout. Codespace storage persists while the Codespace is retained; deleting the Codespace also deletes its local database, project files, and downloaded model unless you back them up.

## Local development and tests

On Debian/Ubuntu, install the system tools with `sudo apt-get update && sudo apt-get install -y build-essential cmake git curl python3-venv`; ensure Node.js 22+ is installed. Then run:

```bash
bash scripts/setup.sh
bash scripts/test.sh
bash scripts/start.sh
```

The test script runs backend API tests and a TypeScript production build; neither test nor build downloads a model. Local non-Linux inference is not configured by these scripts; the supported runtime is a Linux Codespace.

## Resources, configuration, and limits

- The default is the code-specialized Qwen2.5-Coder-3B-Instruct Q4_K_M GGUF from `bartowski/Qwen2.5-Coder-3B-Instruct-GGUF` on Hugging Face, pinned to revision `7c137640ef0332dfedb229f2504c58d83ed4307a`. It is downloaded on first launch, not stored in Git. The downloader checks the exact 1,929,903,360-byte size and pinned SHA-256 (`3da3afe6cf5c674ac195803ea0dd6fee7e1c228c2105c1ce8c66890d1d4ab460`) before installing it.
- The selected Qwen2.5-Coder-3B model supports a maximum context of 32,768 tokens. There is no truly unlimited context: the prompt, conversation history, and generated reply share that model limit and available RAM. Startup detects available memory and CPU limits (including Linux cgroups), choosing a conservative context up to 32,768; override it with `NEXUS_CONTEXT_TOKENS=32768` to use the model's full context (requires substantially more RAM). Supported values are 512–32,768. CPU inference uses at most eight threads and one parallel request. Chat generation continues until the model ends its answer or the context limit is reached.
- Attachments support text/code formats such as `.txt`, `.md`, `.json`, `.py`, `.js`, `.ts`, `.html`, `.css`, `.csv`, and `.yaml`. Up to three files can be attached per message, limited to 32 KiB each and 64 KiB total. Binary formats and common secret/key files (including `.env`, private keys, and certificates) are rejected. File text is sent only to the local model, stored in the project's SQLite conversation history, and treated as untrusted input; attachments are not executed.
- **Create file** asks the model for one new file through a tool call. NEXUS previews the path and contents and does not write until you approve; the preview can be edited. The file API is confined to the active project's `data/projects/<project-id>/` directory, rejects traversal, absolute paths and symlink directories, caps files at 256 KiB, and never overwrites an existing file. Approved files appear in the sidebar. This first version creates files only; editing or deleting files is not enabled. The compact model may fail to produce a valid tool call for complex requests.
- `NEXUS_MODEL_PATH`, `NEXUS_DATA_DIR`, `NEXUS_INFERENCE_URL`, `NEXUS_CONTEXT_TOKENS`, and `NEXUS_API_TOKEN` can be set in the shell or `.env`. The backend exposes `/api/health`, `/api/auth/status`, and `/api/capabilities`. Authentication is optional for localhost/private development; set a strong random token before changing forwarded-port visibility. The inference server listens only on loopback.
- A 3B code model is more capable at programming and structured file drafts than the previous 0.5B general model, but it is not guaranteed to complete large game mods in one request. It needs about 1.8 GiB just for the quantized weights, plus runtime/context memory, and CPU inference may be slow. Real inference requires a successful llama.cpp build and model download plus enough Codespace memory. If startup cannot launch the model, NEXUS reports the model as unavailable instead of substituting a fake response. Set `NEXUS_MODEL_PATH` to an existing smaller GGUF if your Codespace cannot run it.
- Project IDs separate conversations in SQLite. This delivery does not yet include file indexing, semantic memory, project files, WebSocket streaming, desktop/Tauri clients, or identity-provider authentication.

## API

- `GET /api/health` and `GET /api/auth/status` are available without a token.
- `GET /api/capabilities` reports detected resources and actual local inference/model availability.
- `GET /api/projects/{project_id}/files` lists files in that project's NEXUS workspace.
- `POST /api/projects/{project_id}/files` creates a new file without overwriting existing files.
- `GET/POST /api/conversations?project_id=...` lists or creates project-scoped conversations.
- `GET /api/conversations/{id}/messages?project_id=...` loads persistent history.
- `POST /api/conversations/{id}/messages/stream` streams SSE events (`token`, `done`, or `error`) while storing successful turns.
