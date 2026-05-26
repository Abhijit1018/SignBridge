import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.engine import RecognitionEngine
from app.tts import TTSModule
from app import stt

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="SignBridge")
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

engine = RecognitionEngine()
tts_module = TTSModule()
_event_q: asyncio.Queue = asyncio.Queue()
_listeners: list[asyncio.Queue] = []   # fan-out: one queue per consumer
_history: list[dict] = []


# ── WebSocket hub ─────────────────────────────────────────────────────────────

class _Hub:
    def __init__(self):
        self._sockets: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self._sockets.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self._sockets:
            self._sockets.remove(ws)

    async def broadcast(self, msg: str):
        dead = []
        for ws in self._sockets:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


hub = _Hub()


# ── Fan-out: one engine event → every consumer queue ─────────────────────────

async def _fanout():
    while True:
        event = await _event_q.get()
        for q in _listeners:
            await q.put(event)


async def _broadcaster():
    """Copy every engine event to connected WebSocket clients."""
    q: asyncio.Queue = asyncio.Queue()
    _listeners.append(q)
    while True:
        event = await q.get()
        await hub.broadcast(json.dumps(event))


async def _tts_handler():
    """Speak sentence_ready events automatically."""
    q: asyncio.Queue = asyncio.Queue()
    _listeners.append(q)
    while True:
        event = await q.get()
        if event.get("type") == "sentence_ready":
            sentence = event["sentence"]
            _history.append({"role": "signer", "text": sentence})
            await tts_module.speak(sentence)


# ── Lifecycle ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def _startup():
    loop = asyncio.get_event_loop()
    engine.start(loop, _event_q)
    asyncio.create_task(_fanout())
    asyncio.create_task(_broadcaster())
    asyncio.create_task(_tts_handler())


@app.on_event("shutdown")
async def _shutdown():
    engine.stop()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/video_feed")
async def video_feed():
    async def _generate():
        while True:
            frame = engine.latest_frame()
            if frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame
                    + b"\r\n"
                )
            else:
                await asyncio.sleep(0.033)

    return StreamingResponse(
        _generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await hub.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # keep-alive
    except WebSocketDisconnect:
        hub.disconnect(websocket)


@app.get("/api/history")
async def get_history():
    return _history


class SpeakRequest(BaseModel):
    text: str


@app.post("/api/tts/speak")
async def tts_speak(req: SpeakRequest):
    _history.append({"role": "signer", "text": req.text})
    asyncio.create_task(tts_module.speak(req.text))
    return {"ok": True}


class TTSConfigRequest(BaseModel):
    backend: str


@app.post("/api/tts/config")
async def tts_config(req: TTSConfigRequest):
    tts_module.set_backend(req.backend)
    return {"backend": tts_module.backend}


@app.post("/api/stt")
async def stt_endpoint(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    text = await stt.transcribe(audio_bytes, audio.filename or "audio.webm")
    if text:
        _history.append({"role": "speaker", "text": text})
        await hub.broadcast(json.dumps({"type": "stt_result", "text": text}))
    return {"text": text}


class ConfigRequest(BaseModel):
    backend: str | None = None
    min_prob: float | None = None
    camera: int | None = None
    flip: bool | None = None


@app.post("/api/config")
async def update_config(req: ConfigRequest):
    loop = asyncio.get_event_loop()
    changed = False
    if req.backend is not None:
        engine.config["backend"] = req.backend
        changed = True
    if req.min_prob is not None:
        engine.config["min_prob"] = req.min_prob
        changed = True
    if req.camera is not None:
        engine.config["camera"] = req.camera
        changed = True
    if req.flip is not None:
        engine.config["flip"] = req.flip
        changed = True
    if changed:
        engine.restart(loop, _event_q)
    return engine.config
