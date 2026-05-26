import pytest
from unittest.mock import MagicMock, AsyncMock


def test_offline_backend_is_default():
    from app.tts import TTSModule
    t = TTSModule()
    assert t.backend == "offline"


def test_set_backend_online():
    from app.tts import TTSModule
    t = TTSModule()
    t.set_backend("online")
    assert t.backend == "online"


def test_set_backend_invalid_raises():
    from app.tts import TTSModule
    t = TTSModule()
    with pytest.raises(ValueError):
        t.set_backend("banana")


def test_offline_engine_initializes():
    from app.tts import TTSModule
    t = TTSModule()
    assert t._offline_engine is not None


@pytest.mark.asyncio
async def test_speak_offline_calls_pyttsx3(monkeypatch):
    from app import tts as tts_mod
    spoken = []
    mock_engine = MagicMock()
    mock_engine.say = lambda text: spoken.append(text)
    mock_engine.runAndWait = MagicMock()

    t = tts_mod.TTSModule()
    t._offline_engine = mock_engine
    t.set_backend("offline")
    await t.speak("hello world")
    assert "hello world" in spoken


@pytest.mark.asyncio
async def test_speak_online_calls_groq(monkeypatch):
    from app import tts as tts_mod
    import app.tts

    mock_response = MagicMock()
    mock_response.content = b"fake_audio_bytes"
    mock_groq = MagicMock()
    mock_groq.audio.speech.create.return_value = mock_response

    monkeypatch.setattr(app.tts, "_groq_client", mock_groq)

    t = tts_mod.TTSModule()
    t.set_backend("online")
    # Should not raise even with fake audio bytes (sounddevice/soundfile will be mocked)
    # We just verify the Groq client was called
    import unittest.mock
    with unittest.mock.patch("app.tts.sf") as mock_sf, \
         unittest.mock.patch("app.tts.sd") as mock_sd:
        mock_sf.read.return_value = ([0.0, 0.1], 44100)
        mock_sd.play = MagicMock()
        mock_sd.wait = MagicMock()
        await t.speak("hello")
    mock_groq.audio.speech.create.assert_called_once()
