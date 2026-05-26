# SignBridge

A real-time communication bridge between a deaf person and a hearing person in the same room.

- **Sign → Text → Speech**: Deaf person signs ASL letters → app recognizes them → builds a sentence → speaks it aloud automatically
- **Speech → Sign**: Hearing person speaks → Groq Whisper transcribes → app animates ASL fingerspelling letter-by-letter on screen

---

## Demo

| Feature | Description |
|---------|-------------|
| Live webcam feed | Stream with ROI overlay, confidence bar, stability progress |
| Sign recognition | CNN (HSV), HOG+SVM, or MediaPipe landmarks — switchable at runtime |
| Auto TTS | Sentence spoken aloud after 3s of no hand, or on `Enter` key |
| TTS voice | Toggle between offline (pyttsx3, zero latency) and online (Groq PlayAI) |
| Speech → Fingerspell | Hold the mic button → speak → watch letters animate on screen |
| Conversation history | Scrollable log of what was signed and what was spoken |

---

## Architecture

```
d:\handgesture\
├── app\
│   ├── main.py          FastAPI app — routes, WebSocket hub, fan-out event system
│   ├── engine.py        Sign recognition loop (background thread, MJPEG frames)
│   ├── word_builder.py  State machine — letter → word → sentence (two gap timers)
│   ├── tts.py           TTS — pyttsx3 offline + Groq PlayAI online, togglable
│   ├── stt.py           STT — Groq Whisper via REST
│   └── static\
│       ├── index.html   Dashboard UI
│       ├── style.css
│       ├── app.js
│       └── signs\       26 ASL letter PNGs (A–Z)
├── Sign-Language-Recognition\   CNN model + training scripts (original)
├── tests\               28 unit tests (pytest)
├── scripts\
│   └── generate_signs.py  Generate placeholder sign images with Pillow
├── run.py               Uvicorn entrypoint
├── requirements.txt
└── .env.example
```

---

## Quick Start

### 1. Prerequisites

- Python 3.9–3.11 (TensorFlow does not support 3.13)
- A trained CNN model at `Sign-Language-Recognition/CNNmodel.h5` ([train instructions below](#training-the-cnn-model))
- A [Groq API key](https://console.groq.com) (free tier)

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment

```bash
cp .env.example .env
# Edit .env and set GROQ_API_KEY=gsk_...
```

### 4. Run

```bash
python run.py
```

Open **http://localhost:8000** in your browser.

---

## Usage

### Deaf person (signing)

1. Put your hand inside the yellow ROI box on the webcam feed
2. Hold each letter steady — it auto-appends after ~0.7s of stability
3. Remove your hand for 1.5s → space is added
4. Remove your hand for 3.0s → sentence is spoken aloud automatically
5. Or press **Enter** to speak the current sentence immediately
6. **C** clears the word buffer, **Backspace** deletes the last character

### Hearing person (speaking)

1. Click and hold **🎤 Record** in the right panel
2. Speak your message
3. Release → Groq Whisper transcribes → letters animate as ASL fingerspelling on screen

### Controls

| Key / Button | Action |
|---|---|
| `Enter` | Speak current sentence via TTS |
| `C` | Clear word buffer |
| `Backspace` | Delete last character |
| Hold 🎤 | Record audio for STT |
| ▶ Speak | Manual TTS trigger |
| Offline / Online | Switch TTS voice backend |
| Model selector | Switch recognition backend (CNN / HOG+SVM / MediaPipe) |

---

## Recognition Backends

| Backend | How it works | When to use |
|---------|-------------|-------------|
| **CNN (HSV)** | HSV skin segmentation → 28×28 grayscale → CNN | Default; best accuracy with good lighting |
| **HOG+SVM** | HOG features → SVM classifier | Faster on CPU; train with `train_hog_model.py` |
| **MediaPipe** | 21 hand landmarks → SVM on normalized coords | Lighting-independent; requires hand detection |

---

## Training the CNN Model

1. Download `sign_mnist_train.csv` and `sign_mnist_test.csv` from [Kaggle](https://www.kaggle.com/datamunge/sign-language-mnist)
2. Place them in `Sign-Language-Recognition/data/`
3. Train:

```bash
cd Sign-Language-Recognition
python cnn.py --train-csv data/sign_mnist_train.csv --test-csv data/sign_mnist_test.csv --epochs 15 --model-out CNNmodel.h5 --labels-out labels.txt
```

> **Note:** J and Z are excluded from the dataset — they require motion gestures.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Dashboard HTML |
| `GET` | `/video_feed` | MJPEG webcam stream |
| `WS` | `/ws` | Engine event stream (JSON) |
| `POST` | `/api/stt` | Upload audio → transcribed text |
| `POST` | `/api/tts/speak` | `{text}` → speak immediately |
| `POST` | `/api/tts/config` | `{backend: "offline"\|"online"}` |
| `POST` | `/api/config` | `{model, camera, roi, min_prob, flip}` |
| `GET` | `/api/history` | Conversation history list |

### WebSocket Events (server → client)

```json
{ "type": "letter_detected",  "letter": "A", "confidence": 0.92, "stable_progress": 0.6 }
{ "type": "word_updated",     "word": "HELLO" }
{ "type": "sentence_ready",   "sentence": "HELLO WORLD" }
{ "type": "stt_result",       "text": "How are you?" }
{ "type": "engine_error",     "message": "Camera not found" }
```

---

## Running Tests

```bash
pytest tests/ -v
```

28 tests across word builder, TTS, STT, and API modules.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` + `uvicorn` | Web server and WebSocket hub |
| `opencv-python` | Camera capture, HSV segmentation, MJPEG encoding |
| `tensorflow` | CNN sign classification |
| `mediapipe` | Hand landmark detection |
| `scikit-learn` | HOG+SVM backend |
| `pyttsx3` | Offline TTS (Windows SAPI) |
| `groq` | Groq Whisper STT + PlayAI TTS |
| `python-dotenv` | `.env` API key loading |
| `Pillow` | Generating placeholder sign images |

---

## Troubleshooting

**Camera not opening**
Try `POST /api/config` with `{"camera": 1}` or check that no other app is using the webcam.

**Low sign recognition accuracy**
- Ensure good lighting on your hand
- Use `--roi` to reposition the detection box over your hand
- Retrain the CNN with more data or more epochs

**No TTS audio**
- Offline: ensure Windows SAPI voices are installed (Control Panel → Speech)
- Online: check `GROQ_API_KEY` in `.env` and that `sounddevice` is installed

**Groq API errors**
Verify your key at [console.groq.com](https://console.groq.com). The free tier supports Whisper + PlayAI TTS.

---

## License

Apache 2.0 — see original `Sign-Language-Recognition/` for the base project license.
