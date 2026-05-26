import pytest
from unittest.mock import MagicMock


@pytest.mark.asyncio
async def test_transcribe_returns_text(monkeypatch):
    import app.stt as stt_mod

    mock_response = MagicMock()
    mock_response.text = "hello world"
    mock_client = MagicMock()
    mock_client.audio.transcriptions.create.return_value = mock_response
    monkeypatch.setattr(stt_mod, "_groq_client", mock_client)

    result = await stt_mod.transcribe(b"fake_audio", "audio.webm")
    assert result == "hello world"


@pytest.mark.asyncio
async def test_transcribe_passes_correct_model(monkeypatch):
    import app.stt as stt_mod

    mock_response = MagicMock()
    mock_response.text = "test"
    mock_client = MagicMock()
    mock_client.audio.transcriptions.create.return_value = mock_response
    monkeypatch.setattr(stt_mod, "_groq_client", mock_client)

    await stt_mod.transcribe(b"audio", "audio.webm")
    call_kwargs = mock_client.audio.transcriptions.create.call_args
    assert call_kwargs.kwargs.get("model") == "whisper-large-v3-turbo"


@pytest.mark.asyncio
async def test_transcribe_returns_empty_when_no_client(monkeypatch):
    import app.stt as stt_mod
    monkeypatch.setattr(stt_mod, "_groq_client", None)
    result = await stt_mod.transcribe(b"audio", "audio.webm")
    assert result == ""
