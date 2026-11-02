import asyncio
import hmac
import json
import os
import re
import sqlite3
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "data"
INFERENCE_URL = os.getenv("NEXUS_INFERENCE_URL", "http://127.0.0.1:8081")
PROJECT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
MODEL_MAX_CONTEXT_TOKENS = 32_768


def database_path() -> Path:
    return Path(os.getenv("NEXUS_DATA_DIR", str(DEFAULT_DATA_DIR))) / "nexus.sqlite3"


def connect_db() -> sqlite3.Connection:
    path = database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=10)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    return db


def initialize_db() -> None:
    with connect_db() as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS conversations_project_updated
                ON conversations(project_id, updated_at DESC);
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS messages_conversation_created
                ON messages(conversation_id, created_at);
            """
        )


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_db()
    yield


app = FastAPI(title="NEXUS AI", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_origin_regex=r"https://[a-z0-9-]+-5173\.app\.github\.dev",
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


class ConversationCreate(BaseModel):
    project_id: str = Field(default="default", min_length=1, max_length=64)
    title: str = Field(default="New conversation", min_length=1, max_length=120)


class MessageCreate(BaseModel):
    project_id: str = Field(default="default", min_length=1, max_length=64)
    content: str = Field(min_length=1, max_length=20_000)


def validate_project_id(project_id: str) -> str:
    if not PROJECT_ID_PATTERN.fullmatch(project_id):
        raise HTTPException(
            status_code=422,
            detail="Project IDs must be 1-64 letters, digits, dots, underscores, or hyphens.",
        )
    return project_id


def require_auth(authorization: str | None) -> None:
    expected = os.getenv("NEXUS_API_TOKEN", "").strip()
    if not expected:
        return
    supplied = authorization.removeprefix("Bearer ").strip() if authorization else ""
    if not supplied or not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="A valid NEXUS API token is required.")


@app.middleware("http")
async def authenticate_api(request: Request, call_next):
    if (
        request.method != "OPTIONS"
        and request.url.path.startswith("/api/")
        and request.url.path not in {"/api/health", "/api/auth/status"}
    ):
        try:
            require_auth(request.headers.get("authorization"))
        except HTTPException as error:
            return JSONResponse(
                status_code=error.status_code,
                content={"detail": error.detail},
                headers=error.headers,
            )
    return await call_next(request)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def model_path() -> str:
    return os.getenv(
        "NEXUS_MODEL_PATH",
        str(DEFAULT_DATA_DIR / "models" / "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"),
    )


async def inference_health() -> bool:
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(2.0)) as client:
            response = await client.get(f"{INFERENCE_URL}/health")
            return response.is_success
    except (httpx.HTTPError, asyncio.TimeoutError):
        return False


def _read_integer(path: Path) -> int | None:
    try:
        value = path.read_text().strip()
        return int(value) if value != "max" else None
    except (OSError, ValueError):
        return None


def _cpuset_count(value: str) -> int:
    count = 0
    for group in value.split(","):
        start, separator, end = group.partition("-")
        try:
            count += int(end) - int(start) + 1 if separator else 1
        except ValueError:
            return 0
    return count


def resource_info() -> dict:
    cpu_count = os.cpu_count() or 1
    cpuset = Path("/sys/fs/cgroup/cpuset.cpus.effective")
    try:
        allowed_cpus = _cpuset_count(cpuset.read_text().strip())
        if allowed_cpus:
            cpu_count = min(cpu_count, allowed_cpus)
    except OSError:
        pass
    cpu_quota = Path("/sys/fs/cgroup/cpu.max")
    try:
        quota, period = cpu_quota.read_text().split()
        if quota != "max":
            cpu_count = min(cpu_count, max(1, int(quota) // int(period)))
    except (OSError, ValueError, ZeroDivisionError):
        pass

    memory_bytes = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                memory_bytes = int(line.split()[1]) * 1024
                break
    except (OSError, ValueError, IndexError):
        try:
            memory_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (AttributeError, OSError, ValueError):
            pass

    cgroup_limit = _read_integer(Path("/sys/fs/cgroup/memory.max"))
    cgroup_usage = _read_integer(Path("/sys/fs/cgroup/memory.current"))
    if cgroup_limit is not None and cgroup_usage is not None:
        available_cgroup = max(0, cgroup_limit - cgroup_usage)
        memory_bytes = (
            min(memory_bytes, available_cgroup)
            if memory_bytes is not None
            else available_cgroup
        )

    gib = memory_bytes / (1024**3) if memory_bytes else None
    if gib is not None and gib >= 16:
        context_tokens = MODEL_MAX_CONTEXT_TOKENS
    elif gib is not None and gib >= 8:
        context_tokens = 8192
    elif gib is not None and gib >= 4:
        context_tokens = 2048
    else:
        context_tokens = 1024
    configured_context = os.getenv("NEXUS_CONTEXT_TOKENS", "").strip()
    if configured_context:
        try:
            context_tokens = int(configured_context)
        except ValueError:
            raise ValueError("NEXUS_CONTEXT_TOKENS must be an integer from 512 to 32768.") from None
        if not 512 <= context_tokens <= MODEL_MAX_CONTEXT_TOKENS:
            raise ValueError("NEXUS_CONTEXT_TOKENS must be between 512 and 32768.")
    return {
        "cpu_count": cpu_count,
        "memory_bytes": memory_bytes,
        "memory_gib": round(gib, 1) if gib else None,
        "context_tokens": context_tokens,
    }


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/auth/status")
async def auth_status():
    return {"required": bool(os.getenv("NEXUS_API_TOKEN", "").strip())}


@app.get("/api/capabilities")
async def capabilities():
    current_model = Path(model_path())
    return {
        "inference_available": await inference_health(),
        "model_available": current_model.is_file(),
        "model_path": current_model.name,
        "resources": resource_info(),
    }


@app.get("/api/conversations")
async def list_conversations(
    project_id: str = Query(default="default", min_length=1, max_length=64),
    authorization: str | None = Header(default=None),
):
    validate_project_id(project_id)
    require_auth(authorization)
    with connect_db() as db:
        rows = db.execute(
            """SELECT id, project_id, title, created_at, updated_at
               FROM conversations WHERE project_id = ? ORDER BY updated_at DESC""",
            (project_id,),
        ).fetchall()
    return [dict(row) for row in rows]


@app.post("/api/conversations", status_code=201)
async def create_conversation(
    body: ConversationCreate,
    authorization: str | None = Header(default=None),
):
    require_auth(authorization)
    project_id = validate_project_id(body.project_id)
    conversation_id = str(uuid.uuid4())
    timestamp = now_iso()
    with connect_db() as db:
        db.execute(
            """INSERT INTO conversations (id, project_id, title, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?)""",
            (conversation_id, project_id, body.title, timestamp, timestamp),
        )
    return {
        "id": conversation_id,
        "project_id": project_id,
        "title": body.title,
        "created_at": timestamp,
        "updated_at": timestamp,
    }


@app.get("/api/conversations/{conversation_id}/messages")
async def list_messages(
    conversation_id: str,
    project_id: str = Query(default="default", min_length=1, max_length=64),
    authorization: str | None = Header(default=None),
):
    require_auth(authorization)
    validate_project_id(project_id)
    with connect_db() as db:
        conversation = db.execute(
            "SELECT id FROM conversations WHERE id = ? AND project_id = ?",
            (conversation_id, project_id),
        ).fetchone()
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        rows = db.execute(
            """SELECT id, role, content, created_at FROM messages
               WHERE conversation_id = ? ORDER BY created_at, rowid""",
            (conversation_id,),
        ).fetchall()
    return [dict(row) for row in rows]


async def inference_tokens(messages: list[dict]) -> AsyncIterator[str]:
    timeout = httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST",
            f"{INFERENCE_URL}/v1/chat/completions",
            json={
                "model": "local",
                "messages": messages,
                "stream": True,
                "temperature": 0.7,
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    token = json.loads(payload)["choices"][0]["delta"].get("content", "")
                except (KeyError, IndexError, TypeError, json.JSONDecodeError):
                    continue
                if token:
                    yield token


def event(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/api/conversations/{conversation_id}/messages/stream")
async def stream_message(
    conversation_id: str,
    body: MessageCreate,
    authorization: str | None = Header(default=None),
):
    require_auth(authorization)
    project_id = validate_project_id(body.project_id)
    with connect_db() as db:
        conversation = db.execute(
            "SELECT id FROM conversations WHERE id = ? AND project_id = ?",
            (conversation_id, project_id),
        ).fetchone()
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        db.execute(
            "INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), conversation_id, "user", body.content, now_iso()),
        )
        db.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now_iso(), conversation_id),
        )
        history = db.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY created_at, rowid",
            (conversation_id,),
        ).fetchall()
        messages = [dict(row) for row in history]

    async def generate():
        response_text = ""
        try:
            if not await inference_health():
                raise RuntimeError("Local model server is unavailable. Run scripts/start.sh.")
            async for token in inference_tokens(messages):
                response_text += token
                yield event({"type": "token", "token": token})
            if not response_text:
                raise RuntimeError("The local model returned an empty response.")
            with connect_db() as db:
                db.execute(
                    "INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
                    (str(uuid.uuid4()), conversation_id, "assistant", response_text, now_iso()),
                )
                db.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?",
                    (now_iso(), conversation_id),
                )
            yield event({"type": "done"})
        except (httpx.HTTPError, asyncio.TimeoutError, RuntimeError) as exc:
            yield event({"type": "error", "message": str(exc)})

    return StreamingResponse(generate(), media_type="text/event-stream")
