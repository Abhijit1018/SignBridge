import asyncio
import os

from dotenv import load_dotenv

load_dotenv()


def _make_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    from groq import Groq
    return Groq(api_key=api_key)


_groq_client = _make_groq_client()


async def transcribe(audio_bytes: bytes, filename: str) -> str:
    """Transcribe audio bytes using Groq Whisper. Returns empty string if no client."""
    if not _groq_client:
        return ""

    loop = asyncio.get_event_loop()

    def _run():
        response = _groq_client.audio.transcriptions.create(
            file=(filename, audio_bytes),
            model="whisper-large-v3-turbo",
            language="en",
        )
        return response.text

    return await loop.run_in_executor(None, _run)
