import json

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.app import main


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("NEXUS_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("NEXUS_API_TOKEN", raising=False)
    main.initialize_db()
    with TestClient(main.app) as test_client:
        yield test_client


def test_health_and_capabilities_report_unavailable_inference(client, monkeypatch):
    async def unavailable():
        return False

    monkeypatch.setattr(main, "inference_health", unavailable)
    assert client.get("/api/health").json() == {"status": "ok"}
    result = client.get("/api/capabilities").json()
    assert result["inference_available"] is False
    assert result["resources"]["cpu_count"] >= 1
    assert result["resources"]["context_tokens"] > 0


def test_context_can_be_set_to_the_model_limit(monkeypatch):
    monkeypatch.setenv("NEXUS_CONTEXT_TOKENS", "32768")

    assert main.resource_info()["context_tokens"] == 32768


@pytest.mark.parametrize("value", ["not-a-number", "32769", "511"])
def test_invalid_context_setting_is_reported(monkeypatch, value):
    monkeypatch.setenv("NEXUS_CONTEXT_TOKENS", value)

    with pytest.raises(ValueError, match="NEXUS_CONTEXT_TOKENS"):
        main.resource_info()


def test_conversations_are_isolated_by_project(client):
    alpha = client.post(
        "/api/conversations", json={"project_id": "alpha", "title": "Alpha"}
    ).json()
    client.post("/api/conversations", json={"project_id": "beta", "title": "Beta"})

    assert [
        item["id"]
        for item in client.get(
            "/api/conversations", params={"project_id": "alpha"}
        ).json()
    ] == [alpha["id"]]
    assert (
        client.get(
            f"/api/conversations/{alpha['id']}/messages",
            params={"project_id": "beta"},
        ).status_code
        == 404
    )


def test_message_stream_persists_user_and_assistant_messages(client, monkeypatch):
    conversation = client.post(
        "/api/conversations", json={"project_id": "demo"}
    ).json()

    async def available():
        return True

    async def tokens(messages):
        assert messages[-1] == {"role": "user", "content": "Hello"}
        yield "Hello"
        yield " from NEXUS"

    monkeypatch.setattr(main, "inference_health", available)
    monkeypatch.setattr(main, "inference_tokens", tokens)
    response = client.post(
        f"/api/conversations/{conversation['id']}/messages/stream",
        json={"project_id": "demo", "content": "Hello"},
    )
    events = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]

    assert events == [
        {"type": "token", "token": "Hello"},
        {"type": "token", "token": " from NEXUS"},
        {"type": "done"},
    ]
    history = client.get(
        f"/api/conversations/{conversation['id']}/messages",
        params={"project_id": "demo"},
    ).json()
    assert [(item["role"], item["content"]) for item in history] == [
        ("user", "Hello"),
        ("assistant", "Hello from NEXUS"),
    ]


def test_unavailable_inference_returns_error_without_fake_assistant_message(
    client, monkeypatch
):
    conversation = client.post(
        "/api/conversations", json={"project_id": "demo"}
    ).json()

    async def unavailable():
        return False

    monkeypatch.setattr(main, "inference_health", unavailable)
    response = client.post(
        f"/api/conversations/{conversation['id']}/messages/stream",
        json={"project_id": "demo", "content": "Hello"},
    )
    events = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]

    assert events == [
        {
            "type": "error",
            "message": "Local model server is unavailable. Run scripts/start.sh.",
        }
    ]
    history = client.get(
        f"/api/conversations/{conversation['id']}/messages",
        params={"project_id": "demo"},
    ).json()
    assert [(item["role"], item["content"]) for item in history] == [
        ("user", "Hello")
    ]


@pytest.mark.anyio
async def test_inference_client_parses_openai_compatible_sse(monkeypatch):
    requests = []
    def handle_request(request):
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                b'data: {"choices":[{"delta":{"content":"Local "}}]}\n\n'
                b'data: {"choices":[{"delta":{"content":"reply"}}]}\n\n'
                b"data: [DONE]\n\n"
            ),
        )

    transport = httpx.MockTransport(handle_request)
    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(
        main.httpx,
        "AsyncClient",
        lambda **kwargs: real_async_client(transport=transport, **kwargs),
    )

    tokens = [
        token
        async for token in main.inference_tokens(
            [{"role": "user", "content": "Hello"}]
        )
    ]

    assert tokens == ["Local ", "reply"]
    request_payload = json.loads(requests[0].content)
    assert "max_tokens" not in request_payload


def test_api_token_is_enforced(client, monkeypatch):
    monkeypatch.setenv("NEXUS_API_TOKEN", "test-token")
    assert client.get("/api/auth/status").json() == {"required": True}
    assert client.get("/api/conversations").status_code == 401
    assert (
        client.get(
            "/api/conversations",
            headers={"Authorization": "Bearer wrong-token"},
        ).status_code
        == 401
    )
    assert (
        client.get(
            "/api/conversations",
            headers={"Authorization": "Bearer test-token"},
        ).status_code
        == 200
    )


def test_codespaces_cors_preflight_is_allowed_with_auth_enabled(client, monkeypatch):
    monkeypatch.setenv("NEXUS_API_TOKEN", "test-token")
    response = client.options(
        "/api/conversations",
        headers={
            "Origin": "https://my-codespace-5173.app.github.dev",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )

    assert response.status_code == 200
    assert (
        response.headers["access-control-allow-origin"]
        == "https://my-codespace-5173.app.github.dev"
    )
