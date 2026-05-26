import asyncio
import io
import os

import pyttsx3
from dotenv import load_dotenv

load_dotenv()

_VALID_BACKENDS = ("offline", "online")


def _make_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    from groq import Groq
    return Groq(api_key=api_key)


_groq_client = _make_groq_client()

try:
    import sounddevice as sd
    import soundfile as sf
except ImportError:
    sd = None
    sf = None


class TTSModule:
    def __init__(self):
        self.backend = "offline"
        try:
            self._offline_engine = pyttsx3.init()
            self._offline_engine.setProperty("rate", 160)
        except Exception:
            self._offline_engine = None

    def set_backend(self, backend: str):
        if backend not in _VALID_BACKENDS:
            raise ValueError(f"backend must be one of {_VALID_BACKENDS}, got {backend!r}")
        self.backend = backend

    async def speak(self, text: str):
        if not text.strip():
            return
        if self.backend == "online" and _groq_client:
            await self._speak_online(text)
        else:
            await self._speak_offline(text)

    async def _speak_offline(self, text: str):
        loop = asyncio.get_event_loop()

        def _run():
            self._offline_engine.say(text)
            self._offline_engine.runAndWait()

        await loop.run_in_executor(None, _run)

    async def _speak_online(self, text: str):
        loop = asyncio.get_event_loop()

        def _run():
            response = _groq_client.audio.speech.create(
                model="playai-tts",
                voice="Fritz-PlayAI",
                input=text,
                response_format="wav",
            )
            audio_bytes = response.content
            with io.BytesIO(audio_bytes) as buf:
                data, samplerate = sf.read(buf)
            sd.play(data, samplerate)
            sd.wait()

        await loop.run_in_executor(None, _run)
