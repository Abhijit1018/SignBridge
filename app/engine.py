import asyncio
import os
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

SLR = Path(__file__).parent.parent / "Sign-Language-Recognition"
sys.path.insert(0, str(SLR))

from app.word_builder import WordBuilder  # noqa: E402


class RecognitionEngine:
    def __init__(self):
        self._thread: threading.Thread | None = None
        self._stop_evt = threading.Event()
        self._frame_q: queue.Queue[bytes] = queue.Queue(maxsize=2)
        self._event_q: asyncio.Queue | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self.config = {
            "backend": "cnn",           # "cnn" | "hog" | "mediapipe"
            "model_path": str(SLR / "CNNmodel.h5"),
            "labels_path": str(SLR / "labels.txt"),
            "camera": 0,
            "roi": (350, 50, 250, 260),
            "flip": False,
            "min_prob": 0.5,
            "hist_path": None,
        }

    # ------------------------------------------------------------------ public

    def start(self, loop: asyncio.AbstractEventLoop, event_q: asyncio.Queue):
        self._loop = loop
        self._event_q = event_q
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="engine")
        self._thread.start()

    def stop(self):
        self._stop_evt.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def restart(self, loop: asyncio.AbstractEventLoop, event_q: asyncio.Queue):
        self.stop()
        self.start(loop, event_q)

    def latest_frame(self) -> bytes | None:
        try:
            return self._frame_q.get_nowait()
        except queue.Empty:
            return None

    # ----------------------------------------------------------------- private

    def _emit(self, event: dict):
        if self._loop and self._event_q and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(self._event_q.put(event), self._loop)

    def _push_frame(self, frame):
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            return
        jpeg = buf.tobytes()
        if self._frame_q.full():
            try:
                self._frame_q.get_nowait()
            except queue.Empty:
                pass
        self._frame_q.put(jpeg)

    def _open_camera(self, index: int):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera {index}. Check it is connected and not in use."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        for _ in range(20):
            cap.read()
        return cap

    def _handle_events(self, events: list[dict], wb: WordBuilder, label, score):
        word = wb.current_word
        for evt in events:
            t = evt["type"]
            if t == "letter_added":
                self._emit({"type": "word_updated", "word": word, "progress": 0.0})
            elif t == "space_added":
                self._emit({"type": "word_updated", "word": word, "progress": 0.0})
            elif t == "sentence_ready":
                self._emit({"type": "sentence_ready", "sentence": evt["sentence"]})
        if label:
            self._emit({
                "type": "letter_detected",
                "letter": label,
                "confidence": round(float(score), 3),
                "progress": round(wb.stable_progress, 3),
                "word": word,
            })

    def _has_hand_blob(self, mask: np.ndarray) -> bool:
        """Return True only if the mask contains a plausible hand-sized skin blob.

        Rejects: no skin at all, giant face/body blobs, tiny noise specks.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False
        largest_area = max(cv2.contourArea(c) for c in contours)
        total_pixels = mask.shape[0] * mask.shape[1]
        min_area = total_pixels * 0.04   # hand must cover at least 4% of ROI
        max_area = total_pixels * 0.75   # blobs larger than 75% = face/body, reject
        return min_area <= largest_area <= max_area

    def _draw_overlay(self, frame, wb: WordBuilder, roi, label, score):
        x, y, w, h = roi
        fh, fw = frame.shape[:2]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        if label:
            cv2.putText(frame, f"{label}  {score:.2f}", (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        if wb.stable_progress > 0:
            bar_w = int(w * wb.stable_progress)
            cv2.rectangle(frame, (x, y + h + 2), (x + bar_w, y + h + 8), (0, 220, 80), -1)
        bar_h = 48
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, fh - bar_h), (fw, fh), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        word_text = wb.current_word or "(hold a sign)"
        cv2.putText(frame, word_text, (10, fh - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 128), 2, cv2.LINE_AA)

    def _run(self):
        try:
            backend = self.config["backend"]
            if backend == "mediapipe":
                self._run_mediapipe()
            elif backend == "hog":
                self._run_hog()
            else:
                self._run_cnn()
        except Exception as exc:
            self._emit({"type": "engine_error", "message": str(exc)})

    def _run_cnn(self):
        from tensorflow import keras
        from collections import deque
        from predict_webcam import load_labels, segment_hand, prepare_tensor
        import pickle

        cfg = self.config
        model_path = cfg["model_path"]
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"CNN model not found: {model_path}\n"
                "Run python cnn.py inside Sign-Language-Recognition/ first."
            )
        model = keras.models.load_model(model_path)
        input_shape = model.input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        input_shape = tuple(input_shape)
        labels = load_labels(cfg["labels_path"])

        hist = None
        if cfg["hist_path"] and os.path.exists(cfg["hist_path"]):
            with open(cfg["hist_path"], "rb") as f:
                hist = pickle.load(f)

        cap = self._open_camera(cfg["camera"])
        wb = WordBuilder()
        history = deque(maxlen=10)

        try:
            while not self._stop_evt.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue
                if cfg["flip"]:
                    frame = cv2.flip(frame, 1)

                x, y, w, h = cfg["roi"]
                fh, fw = frame.shape[:2]
                x, y = max(0, x), max(0, y)
                w, h = min(w, fw - x), min(h, fh - y)
                roi_frame = frame[y:y+h, x:x+w]
                if roi_frame.size == 0:
                    continue

                segmented, mask = segment_hand(roi_frame, hist)
                tensor = prepare_tensor(segmented, input_shape)
                probs = model.predict(tensor, verbose=0)[0]
                pred_class = int(np.argmax(probs))
                pred_score = float(np.max(probs))
                history.append(pred_class)

                counts: dict = {}
                for v in history:
                    counts[v] = counts.get(v, 0) + 1
                smooth = max(counts, key=counts.get)
                hand_present = self._has_hand_blob(mask)
                confident = hand_present and pred_score >= cfg["min_prob"] and smooth < len(labels)
                label = labels[smooth] if confident else None

                events = wb.process(label, confident)
                self._handle_events(events, wb, label, pred_score)
                self._draw_overlay(frame, wb, (x, y, w, h), label, pred_score)
                self._push_frame(frame)
        finally:
            cap.release()

    def _run_hog(self):
        from collections import deque
        from predict_webcam import segment_hand, load_hist
        import pickle

        cfg = self.config
        hog_path = str(SLR / "hog_model.pkl")
        if not os.path.exists(hog_path):
            raise FileNotFoundError(f"HOG model not found: {hog_path}")

        with open(hog_path, "rb") as f:
            payload = pickle.load(f)
        hist = load_hist(cfg["hist_path"])
        cap = self._open_camera(cfg["camera"])
        wb = WordBuilder()

        def _predict(seg):
            img_size = payload.get("img_size", (64, 64))
            resized = cv2.resize(seg, img_size)
            hog_desc = cv2.HOGDescriptor(img_size, (16, 16), (8, 8), (8, 8), 9)
            feat = hog_desc.compute(resized).flatten().reshape(1, -1)
            clf = payload["model"]
            le = payload["label_encoder"]
            probs = clf.predict_proba(feat)[0]
            idx = int(np.argmax(probs))
            return le.inverse_transform([idx])[0], float(np.max(probs))

        try:
            while not self._stop_evt.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue
                if cfg["flip"]:
                    frame = cv2.flip(frame, 1)
                x, y, w, h = cfg["roi"]
                fh, fw = frame.shape[:2]
                x, y = max(0, x), max(0, y)
                w, h = min(w, fw - x), min(h, fh - y)
                roi_frame = frame[y:y+h, x:x+w]
                if roi_frame.size == 0:
                    continue
                segmented, mask = segment_hand(roi_frame, hist)
                label, score = _predict(segmented)
                hand_present = self._has_hand_blob(mask)
                confident = hand_present and score >= cfg["min_prob"]
                label = label if confident else None
                events = wb.process(label, confident)
                self._handle_events(events, wb, label, score)
                self._draw_overlay(frame, wb, (x, y, w, h), label, score)
                self._push_frame(frame)
        finally:
            cap.release()

    def _run_mediapipe(self):
        import mediapipe as _mp
        mp_hands = _mp.solutions.hands
        mp_draw = _mp.solutions.drawing_utils
        mp_styles = _mp.solutions.drawing_styles
        import pickle
        from collections import deque

        cfg = self.config
        model_path = str(SLR / "landmark_model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Landmark model not found: {model_path}")

        with open(model_path, "rb") as f:
            payload = pickle.load(f)
        classifier = payload["model"]
        le = payload["label_encoder"]

        pred_history: deque = deque(maxlen=10)
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=cfg["min_prob"],
            min_tracking_confidence=0.5,
        )
        cap = self._open_camera(cfg["camera"])
        wb = WordBuilder()

        def _normalize(landmarks):
            pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
            pts -= pts[0].copy()
            scale = np.max(np.linalg.norm(pts, axis=1)) + 1e-6
            pts /= scale
            return pts.flatten().reshape(1, -1)

        try:
            while not self._stop_evt.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue
                frame = cv2.flip(frame, 1)
                fh, fw = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)
                label, score = None, 0.0

                if result.multi_hand_landmarks:
                    for hand_lm in result.multi_hand_landmarks:
                        mp_draw.draw_landmarks(
                            frame, hand_lm,
                            mp_hands.HAND_CONNECTIONS,
                            mp_styles.get_default_hand_landmarks_style(),
                            mp_styles.get_default_hand_connections_style(),
                        )
                        feat = _normalize(hand_lm.landmark)
                        probs = classifier.predict_proba(feat)[0]
                        idx = int(np.argmax(probs))
                        score = float(np.max(probs))
                        pred_history.append(le.inverse_transform([idx])[0])
                        counts: dict = {}
                        for v in pred_history:
                            counts[v] = counts.get(v, 0) + 1
                        label = max(counts, key=counts.get)

                confident = label is not None and score >= cfg["min_prob"]
                label = label if confident else None
                events = wb.process(label, confident)
                self._handle_events(events, wb, label, score)
                roi = (0, 0, fw, fh)
                self._draw_overlay(frame, wb, roi, label, score)
                self._push_frame(frame)
        finally:
            cap.release()
            hands.close()
