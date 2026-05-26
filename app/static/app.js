// ── State ────────────────────────────────────────────────────────────────────
let ws = null;
let recorder = null;
let audioChunks = [];
let spellingTimer = null;

// ── WebSocket ────────────────────────────────────────────────────────────────
function connectWS() {
  ws = new WebSocket(`ws://${location.host}/ws`);

  ws.onopen = () => {
    document.getElementById("statusDot").classList.add("connected");
  };

  ws.onclose = () => {
    document.getElementById("statusDot").classList.remove("connected");
    setTimeout(connectWS, 2000); // auto-reconnect
  };

  ws.onmessage = (evt) => {
    const event = JSON.parse(evt.data);
    handleEvent(event);
  };
}

function handleEvent(event) {
  switch (event.type) {
    case "letter_detected":
      document.getElementById("letterOverlay").textContent =
        `${event.letter}  ${(event.confidence * 100).toFixed(0)}%`;
      break;

    case "word_updated":
      document.getElementById("wordText").textContent = event.word || "(hold a sign)";
      break;

    case "sentence_ready":
      addHistory("signer", event.sentence);
      document.getElementById("sentenceText").textContent = "";
      document.getElementById("wordText").textContent = "(hold a sign)";
      break;

    case "stt_result":
      addHistory("speaker", event.text);
      fingerspell(event.text);
      break;

    case "engine_error":
      console.error("Engine error:", event.message);
      document.getElementById("letterOverlay").textContent = "⚠ " + event.message;
      break;
  }
}

// ── Keyboard shortcuts ────────────────────────────────────────────────────────
document.addEventListener("keydown", (e) => {
  if (e.key === "Enter") speakNow();
  if (e.key === "c" || e.key === "C") clearWord();
  if (e.key === "Backspace") deleteLastLetter();
});

// ── TTS controls ──────────────────────────────────────────────────────────────
async function speakNow() {
  const text = document.getElementById("sentenceText").textContent.trim()
    || document.getElementById("wordText").textContent.trim();
  if (!text || text === "(hold a sign)") return;
  await fetch("/api/tts/speak", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  addHistory("signer", text);
  document.getElementById("sentenceText").textContent = "";
}

async function setTTSBackend(backend) {
  await fetch("/api/tts/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ backend }),
  });
  document.getElementById("offlineBtn").classList.toggle("active", backend === "offline");
  document.getElementById("onlineBtn").classList.toggle("active", backend === "online");
  document.getElementById("ttsBadge").textContent = `🔊 TTS: ${backend === "online" ? "Online" : "Offline"}`;
}

function clearWord() {
  document.getElementById("wordText").textContent = "(hold a sign)";
  document.getElementById("sentenceText").textContent = "";
}

function deleteLastLetter() {
  const el = document.getElementById("wordText");
  const t = el.textContent;
  if (t && t !== "(hold a sign)") {
    el.textContent = t.slice(0, -1) || "(hold a sign)";
  }
}

async function setModel(backend) {
  await fetch("/api/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ backend }),
  });
}

// ── STT + Fingerspelling ──────────────────────────────────────────────────────
async function startRecording() {
  if (recorder && recorder.state === "recording") return;
  audioChunks = [];
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  recorder = new MediaRecorder(stream);
  recorder.ondataavailable = (e) => audioChunks.push(e.data);
  recorder.onstop = sendAudio;
  recorder.start();
  document.getElementById("recordBtn").classList.add("recording");
  document.getElementById("recordBtn").textContent = "Recording...";
  document.getElementById("sttBadge").classList.add("recording");
  document.getElementById("sttBadge").textContent = "🎤 Recording";
}

function stopRecording() {
  if (recorder && recorder.state === "recording") {
    recorder.stop();
    recorder.stream.getTracks().forEach((t) => t.stop());
  }
  document.getElementById("recordBtn").classList.remove("recording");
  document.getElementById("recordBtn").textContent = "Hold to Record";
  document.getElementById("sttBadge").classList.remove("recording");
  document.getElementById("sttBadge").textContent = "🎤 Processing...";
}

async function sendAudio() {
  const blob = new Blob(audioChunks, { type: "audio/webm" });
  const fd = new FormData();
  fd.append("audio", blob, "audio.webm");
  const res = await fetch("/api/stt", { method: "POST", body: fd });
  const data = await res.json();
  document.getElementById("sttBadge").textContent = "🎤 STT: Ready";
  if (data.text) fingerspell(data.text);
}

async function fingerspell(text) {
  const letters = text.toUpperCase().replace(/[^A-Z ]/g, "").split("");
  const img = document.getElementById("signImg");
  const placeholder = document.getElementById("signPlaceholder");
  const wordEl = document.getElementById("spellWord");

  clearTimeout(spellingTimer);
  placeholder.style.display = "none";
  img.style.display = "block";

  const renderWord = (idx) => {
    return letters.map((l, i) => {
      if (l === " ") return " &nbsp; ";
      return i === idx
        ? `<span class="current">${l}</span>`
        : l;
    }).join("");
  };

  for (let i = 0; i < letters.length; i++) {
    const letter = letters[i];
    wordEl.innerHTML = renderWord(i);
    if (letter === " ") {
      img.src = "";
      img.style.display = "none";
      placeholder.style.display = "flex";
      placeholder.textContent = "·";
    } else {
      img.style.display = "block";
      placeholder.style.display = "none";
      img.src = `/static/signs/${letter}.png`;
    }
    await new Promise((r) => setTimeout(r, 300));
  }

  wordEl.innerHTML = text.toUpperCase();
  placeholder.style.display = "flex";
  placeholder.textContent = "🎤";
  img.style.display = "none";
}

// ── History ───────────────────────────────────────────────────────────────────
function addHistory(role, text) {
  const list = document.getElementById("historyList");
  const div = document.createElement("div");
  div.className = `history-msg ${role}`;
  const who = role === "signer" ? "🤟 Signer" : "🎤 Speaker";
  const action = role === "signer" ? "spoken aloud" : "fingerspelled";
  div.innerHTML = `
    <div class="who">${who}</div>
    <div class="text">${text}</div>
    <div class="history-ts">just now · ${action}</div>
  `;
  list.prepend(div);
}

function toggleTTSPanel() {
  const p = document.getElementById("ttsPanel");
  p.style.display = p.style.display === "none" ? "block" : "none";
}

// ── Init ──────────────────────────────────────────────────────────────────────
connectWS();
