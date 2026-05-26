import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    # Patch engine.start so camera is never opened during tests
    with patch("app.engine.RecognitionEngine.start"):
        from app.main import app
        with TestClient(app) as c:
            yield c


def test_root_serves_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_history_empty_initially(client):
    r = client.get("/api/history")
    assert r.status_code == 200
    assert r.json() == []


def test_config_update_min_prob(client):
    r = client.post("/api/config", json={"min_prob": 0.7})
    assert r.status_code == 200
    data = r.json()
    assert data["min_prob"] == 0.7


def test_config_update_backend(client):
    r = client.post("/api/config", json={"backend": "hog"})
    assert r.status_code == 200
    assert r.json()["backend"] == "hog"


def test_tts_speak_endpoint(client):
    with patch("app.main.tts_module.speak", new_callable=AsyncMock) as mock_speak:
        r = client.post("/api/tts/speak", json={"text": "hello world"})
        assert r.status_code == 200
        assert r.json() == {"ok": True}


def test_tts_config_endpoint(client):
    r = client.post("/api/tts/config", json={"backend": "offline"})
    assert r.status_code == 200


def test_stt_endpoint_without_key(client):
    with patch("app.stt._groq_client", None):
        content = b"fake_audio"
        r = client.post(
            "/api/stt",
            files={"audio": ("audio.webm", content, "audio/webm")},
        )
        assert r.status_code == 200
        assert "text" in r.json()
