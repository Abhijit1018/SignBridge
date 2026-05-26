<div align="center">

# 🤟 SignBridge

**Real-time communication between a deaf person and a hearing person — no interpreter needed.**

[![Python](https://img.shields.io/badge/Python-3.9--3.11-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Groq](https://img.shields.io/badge/Powered%20by-Groq-F55036?style=flat)](https://console.groq.com)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue?style=flat)](https://opensource.org/licenses/Apache-2.0)

</div>

---

## What is SignBridge?

SignBridge is a web-based communication bridge that lets a **deaf person** and a **hearing person** talk to each other in the same room using just a laptop.

```
Deaf person signs ASL  →  webcam recognizes letters  →  builds sentence  →  speaks it aloud
Hearing person speaks  →  Groq Whisper transcribes   →  animates ASL fingerspelling on screen
```

No interpreter. No typing. Just sign and speak.

---

## Features

| | Feature | Detail |
|--|---------|--------|
| 🎥 | **Live webcam feed** | MJPEG stream with ROI overlay, confidence bar, stability progress |
| 🧠 | **3 recognition backends** | CNN (HSV segmentation), HOG+SVM, MediaPipe landmarks — switchable at runtime |
| 🔊 | **Auto Text-to-Speech** | Sentence auto-spoken after 3s pause, or manually with `Enter` |
| 🎙 | **Dual TTS voice** | Offline (pyttsx3, zero latency) or Online (Groq PlayAI, natural voice) |
| 👋 | **Speech → Fingerspelling** | Hold-to-record mic → Groq Whisper → ASL letter animation |
| 📜 | **Conversation history** | Live scrollable log with signer (green) and speaker (cyan) messages |
| ⚙️ | **Runtime config** | Switch model, camera, ROI, confidence threshold from the dashboard |

---

## How It Works

```
┌─────────────────────────────────────────────────────┐
│                   Browser Dashboard                 │
│  ┌──────────────────┐    ┌─────────────────────┐   │
│  │  Webcam + ROI    │    │  Fingerspelling      │   │
│  │  Letter: A 0.92  │    │  H E [L] L O        │   │
│  │  ████████░░ 80%  │    │  🎤 Hold to Record  │   │
│  │                  │    │                     │   │
│  │  Word:  HELLO    │    │  TTS: ○ Offline     │   │
│  │  Sentence: ...   │    │       ● Online      │   │
│  └──────────────────┘    └─────────────────────┘   │
└────────────────┬───────────────────┬────────────────┘
                 │ WebSocket         │ POST /api/stt
        ┌────────▼────────┐  ┌───────▼────────┐
        │   engine.py     │  │    stt.py       │
        │  Camera thread  │  │  Groq Whisper   │
        │  CNN/HOG/MP     │  └───────┬────────┘
        │  WordBuilder    │          │ text
        └────────┬────────┘   ┌──────▼──────┐
                 │ events     │   main.py   │
                 └────────────►  Fan-out    │
                              │  hub        │
                              └──────┬──────┘
                                     │
                          ┌──────────▼──────────┐
                          │      tts.py          │
                          │  sentence_ready →    │
                          │  pyttsx3 / Groq TTS  │
                          └─────────────────────┘
```

### Word Builder — Two Gap Timers

The `WordBuilder` state machine converts raw predictions into words:

| Event | Trigger | Action |
|-------|---------|--------|
| Letter accepted | Sign held steady for **~0.7s** (20 stable frames) | Appended to word buffer |
| Space added | No confident hand for **1.5s** | Space inserted |
| Sentence spoken | No confident hand for **3.0s** + buffer non-empty | `sentence_ready` event → TTS |

---

## Project Structure

```
SignBridge/
├── app/
│   ├── main.py           FastAPI app — routes, WebSocket hub, fan-out event system
│   ├── engine.py         Recognition engine — background camera thread, MJPEG frames
│   ├── word_builder.py   State machine — letter → word → sentence (two gap timers)
│   ├── tts.py            TTS — pyttsx3 offline + Groq PlayAI online, togglable
│   ├── stt.py            STT — Groq Whisper transcription via REST
│   └── static/
│       ├── index.html    Dashboard UI
│       ├── style.css     Dark theme
│       ├── app.js        WebSocket client, fingerspell animation, MediaRecorder
│       └── signs/        26 ASL letter PNGs (A–Z)
├── Sign-Language-Recognition/   CNN model, training scripts, HOG+SVM trainer
├── tests/                28 unit + integration tests (pytest)
├── scripts/
│   └── generate_signs.py Generate placeholder sign images with Pillow
├── run.py                Uvicorn entrypoint
├── requirements.txt
└── .env.example
```

---

## Quick Start

### Prerequisites

- **Python 3.9–3.11** (TensorFlow doesn't support 3.13 yet)
- A trained CNN model at `Sign-Language-Recognition/CNNmodel.h5` → [train it here](#training-the-cnn-model)
- A **Groq API key** → [get one free at console.groq.com](https://console.groq.com)

### Install

```bash
git clone https://github.com/Abhijit1018/SignBridge.git
cd SignBridge
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
```

Edit `.env`:

```env
GROQ_API_KEY=gsk_your_key_here
```

### Run

```bash
python run.py
```

Open **[http://localhost:8000](http://localhost:8000)**

---

## Usage

### For the deaf person (signing)

1. Position your hand inside the **yellow ROI box** on the webcam feed
2. Hold each letter steady — it auto-appends after ~0.7s
3. Remove hand for **1.5s** → space is added automatically
4. Remove hand for **3.0s** → sentence is **spoken aloud** for the hearing person
5. Press `Enter` to speak immediately without waiting

### For the hearing person (speaking)

1. Click and **hold** the **🎤 Record** button in the right panel
2. Speak your message clearly
3. Release → Groq Whisper transcribes → ASL letters **animate on screen** for the deaf person

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Enter` | Speak current sentence via TTS |
| `C` | Clear word buffer |
| `Backspace` | Delete last character |

---

## Recognition Backends

Switch between backends from the **navbar dropdown** — no restart needed.

| Backend | Method | Best for |
|---------|--------|---------|
| **CNN (HSV)** | HSV skin segmentation → 28×28 CNN | Default — best accuracy with good lighting |
| **HOG+SVM** | Histogram of Gradients → SVM | Faster on CPU, good for low-end hardware |
| **MediaPipe** | 21 hand landmarks → normalized SVM | Lighting-independent, most robust |

---

## Training the CNN Model

1. Download from [Kaggle — Sign Language MNIST](https://www.kaggle.com/datamunge/sign-language-mnist)
2. Place CSVs in `Sign-Language-Recognition/data/`
3. Train:

```bash
cd Sign-Language-Recognition
python cnn.py \
  --train-csv data/sign_mnist_train.csv \
  --test-csv  data/sign_mnist_test.csv \
  --epochs 15 \
  --model-out CNNmodel.h5 \
  --labels-out labels.txt
```

> **Note:** Letters J and Z are excluded — they require motion and can't be captured as static frames.

---

## API Reference

### REST Endpoints

| Method | Endpoint | Body | Response |
|--------|----------|------|---------|
| `GET` | `/` | — | Dashboard HTML |
| `GET` | `/video_feed` | — | MJPEG stream |
| `GET` | `/api/history` | — | `[{role, text, timestamp}]` |
| `POST` | `/api/stt` | `multipart/form-data` audio | `{text}` |
| `POST` | `/api/tts/speak` | `{text}` | `204` |
| `POST` | `/api/tts/config` | `{backend: "offline"\|"online"}` | `204` |
| `POST` | `/api/config` | `{model, camera, roi, min_prob, flip}` | `204` |

### WebSocket Events — `ws://localhost:8000/ws`

```json
{ "type": "letter_detected",  "letter": "A", "confidence": 0.92, "stable_progress": 0.6 }
{ "type": "word_updated",     "word": "HELLO" }
{ "type": "sentence_ready",   "sentence": "HELLO WORLD" }
{ "type": "stt_result",       "text": "How are you today?" }
{ "type": "engine_error",     "message": "Camera not found at index 0" }
```

---

## Tests

```bash
pytest tests/ -v
```

```
tests/test_word_builder.py   12 tests  — state machine, timers, edge cases
tests/test_tts.py             6 tests  — backend switching, speak routing
tests/test_stt.py             3 tests  — transcription, empty audio, no key
tests/test_api.py             7 tests  — routes, WebSocket, history endpoint
─────────────────────────────────────────
28 passed in 1.78s
```

---

## Stack

| Layer | Technology |
|-------|-----------|
| Web server | FastAPI + Uvicorn |
| Real-time | WebSocket (fan-out queue) + MJPEG |
| Computer vision | OpenCV (capture, segmentation, encoding) |
| Sign classification | TensorFlow/Keras CNN, scikit-learn SVM |
| Hand landmarks | MediaPipe |
| Speech-to-text | Groq Whisper (`whisper-large-v3-turbo`) |
| Text-to-speech | pyttsx3 (offline) + Groq PlayAI (`playai-tts`) |
| Frontend | Vanilla JS + CSS (no framework) |

---

## Troubleshooting

<details>
<summary><strong>Camera not opening</strong></summary>

Try switching camera index from the dashboard or send:
```bash
curl -X POST http://localhost:8000/api/config -H "Content-Type: application/json" -d '{"camera": 1}'
```
Close any other app using the webcam (Teams, Zoom, etc.).
</details>

<details>
<summary><strong>Low sign recognition accuracy</strong></summary>

- Ensure your hand is well-lit and fully inside the yellow ROI box
- Try switching to the **MediaPipe** backend — it's lighting-independent
- Retrain the CNN on your own hands with more epochs
</details>

<details>
<summary><strong>No TTS audio (offline)</strong></summary>

Ensure Windows SAPI voices are installed:
**Settings → Time & Language → Speech → Manage voices**
</details>

<details>
<summary><strong>No TTS audio (online) / Groq errors</strong></summary>

- Verify `GROQ_API_KEY` is set in `.env`
- Check your key at [console.groq.com](https://console.groq.com)
- Ensure `sounddevice` is installed: `pip install sounddevice soundfile`
</details>

<details>
<summary><strong>MediaRecorder not working in browser</strong></summary>

Chrome and Edge require HTTPS or localhost for microphone access. If you're accessing from another machine on the network, set up a reverse proxy with TLS or use the host machine's browser directly.
</details>

---

## License

Apache 2.0 — see [`Sign-Language-Recognition/`](./Sign-Language-Recognition/) for the base project license.

---

<div align="center">

Built with FastAPI · OpenCV · TensorFlow · MediaPipe · Groq

</div>
