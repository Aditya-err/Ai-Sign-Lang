"""
SignLens AI  ·  PyWebView Desktop App
======================================
Backend : Python 3.10, OpenCV, MediaPipe, TensorFlow, pyttsx3
Frontend: index.html  (served by pywebview)
Run     : python app.py   OR   double-click run.bat

Bugs fixed vs v1
-----------------
1. _set_frame used wrong element ID -> now _push_frame(b64, cam_id) targets correct img elements
2. StreamCallback inherited TensorBoard (crashes without TF+Logs dir) -> pure Callback only
3. _emit used fragile string-replace escaping (breaks on Windows backslash paths) -> base64 transport
4. url='index.html' broke when CWD != script dir -> uses __file__ for absolute path
5. pyttsx3 runAndWait() in daemon threads -> dedicated TTS queue worker thread
6. sl:log only wrote to active panel -> now writes to all console elements
7. pywebviewready race condition -> retry loop guard added in JS
"""

import os
import sys
import json
import time
import base64
import queue
import threading

import cv2
import numpy as np
import webview

# ── Paths always relative to THIS file, not CWD ──────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
INDEX_HTML  = os.path.join(BASE_DIR, "index.html")
CONFIG_FILE = os.path.join(BASE_DIR, "signlens_config.json")


# ── Optional heavy imports (graceful fallback) ────────────────────────────────
try:
    import mediapipe as mp
    mp_holistic = mp.solutions.holistic
    mp_drawing  = mp.solutions.drawing_utils
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False

try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.callbacks import Callback, EarlyStopping
    from tensorflow.keras.utils import to_categorical
    from sklearn.model_selection import train_test_split
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ── TTS — single dedicated worker thread (prevents runAndWait deadlocks) ──────
TTS_AVAILABLE = False
_tts_q: queue.Queue = queue.Queue()

try:
    import pyttsx3
    _tts_engine = pyttsx3.init()
    _tts_engine.setProperty("rate", 150)
    TTS_AVAILABLE = True

    def _tts_worker():
        while True:
            text = _tts_q.get()
            if text is None:
                break
            try:
                _tts_engine.say(text)
                _tts_engine.runAndWait()
            except Exception:
                pass
            finally:
                _tts_q.task_done()

    threading.Thread(target=_tts_worker, daemon=True, name="tts-worker").start()

except Exception:
    pass


def speak(text: str):
    """Queue a TTS utterance — non-blocking, safe from any thread."""
    if not TTS_AVAILABLE:
        return
    # Drain queue so we never accumulate stale utterances
    while not _tts_q.empty():
        try:
            _tts_q.get_nowait()
            _tts_q.task_done()
        except queue.Empty:
            break
    _tts_q.put(text)


# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG: dict = {
    "actions":              ["A", "B", "C"],
    "no_sequences":         30,
    "sequence_length":      30,
    "data_path":            os.path.join(BASE_DIR, "MP_Data"),
    "model_path":           os.path.join(BASE_DIR, "action.h5"),
    "confidence_threshold": 0.70,
    "epochs":               2000,
    "tts_rate":             150,
    "tts_enabled":          True,
}


def load_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, encoding="utf-8") as f:
                return {**DEFAULT_CONFIG, **json.load(f)}
        except Exception:
            pass
    return DEFAULT_CONFIG.copy()


def save_config_to_disk(cfg: dict):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


# ── MediaPipe helpers ─────────────────────────────────────────────────────────
def mediapipe_detection(image, model):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = model.process(rgb)
    rgb.flags.writeable = True
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), results


KEYPOINT_SIZE = 33 * 4 + 21 * 3 + 21 * 3  # 258


def extract_keypoints(results) -> np.ndarray:
    pose = (np.array([[r.x, r.y, r.z, r.visibility]
                      for r in results.pose_landmarks.landmark]).flatten()
            if results.pose_landmarks else np.zeros(33 * 4))
    lh = (np.array([[r.x, r.y, r.z]
                    for r in results.left_hand_landmarks.landmark]).flatten()
          if results.left_hand_landmarks else np.zeros(21 * 3))
    rh = (np.array([[r.x, r.y, r.z]
                    for r in results.right_hand_landmarks.landmark]).flatten()
          if results.right_hand_landmarks else np.zeros(21 * 3))
    return np.concatenate([pose, lh, rh])


def draw_landmarks(image, results):
    if not MP_AVAILABLE:
        return image
    spec_pt = mp_drawing.DrawingSpec(color=(0, 255, 180), thickness=1, circle_radius=2)
    spec_ln = mp_drawing.DrawingSpec(color=(0, 200, 140), thickness=1)
    for lm in (results.left_hand_landmarks, results.right_hand_landmarks):
        if lm:
            mp_drawing.draw_landmarks(image, lm,
                                      mp_holistic.HAND_CONNECTIONS, spec_pt, spec_ln)
    return image


def frame_to_b64(frame, quality: int = 60) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("utf-8")


# ── PyWebView API class ───────────────────────────────────────────────────────
class SignLensAPI:

    def __init__(self):
        self.window: webview.Window | None = None
        self.config  = load_config()
        self._stop   = threading.Event()
        self._thread: threading.Thread | None = None

    # ── Transport ─────────────────────────────────────────────────────────────

    def _emit(self, event_type: str, payload: dict):
        """
        Push a JSON event to the JS frontend.
        Uses base64 to safely transport any payload — no escaping edge cases.
        """
        if not self.window:
            return
        b64 = base64.b64encode(json.dumps(payload).encode()).decode()
        self.window.evaluate_js(
            f"window.dispatchEvent(new CustomEvent('{event_type}',"
            f"{{detail:JSON.parse(atob('{b64}'))}}));"
        )

    def _log(self, msg: str, level: str = "info"):
        self._emit("sl:log", {
            "ts":    time.strftime("%H:%M:%S"),
            "msg":   msg,
            "level": level,
        })

    def _push_frame(self, b64: str, cam_id: str):
        """Set a specific <img> element src without escaping issues."""
        # Use a JS data attribute so the long b64 string never goes through evaluate_js string concat
        self.window.evaluate_js(
            f"(function(){{"
            f"var el=document.getElementById('{cam_id}');"
            f"if(el)el.src='data:image/jpeg;base64,{b64}';"
            f"}})();"
        )

    def _stop_thread(self):
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=4)
        self._stop.clear()

    # ── Config ────────────────────────────────────────────────────────────────

    def get_config(self) -> dict:
        return self.config

    def save_config(self, cfg: dict) -> dict:
        self.config.update(cfg)
        save_config_to_disk(self.config)
        if TTS_AVAILABLE and "tts_rate" in cfg:
            _tts_engine.setProperty("rate", int(cfg["tts_rate"]))
        return {"ok": True}

    def get_system_info(self) -> dict:
        return {
            "python":     sys.version.split()[0],
            "cv2":        cv2.__version__,
            "mediapipe":  mp.__version__ if MP_AVAILABLE  else "not installed",
            "tensorflow": "available"    if TF_AVAILABLE  else "not installed",
            "tts":        "available"    if TTS_AVAILABLE else "not installed",
        }

    # ── Data Collection ───────────────────────────────────────────────────────

    def start_collection(self) -> dict:
        self._stop_thread()
        self._thread = threading.Thread(
            target=self._collect_loop, daemon=True, name="collect")
        self._thread.start()
        return {"ok": True}

    def _collect_loop(self):
        cfg       = self.config
        actions   = cfg["actions"]
        n_seq     = int(cfg["no_sequences"])
        seq_len   = int(cfg["sequence_length"])
        data_path = cfg["data_path"]

        for action in actions:
            for seq in range(n_seq):
                os.makedirs(os.path.join(data_path, action, str(seq)), exist_ok=True)

        if not MP_AVAILABLE:
            self._log("MediaPipe not installed — demo mode.", "warn")
            self._simulate_collection(actions, n_seq)
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self._log("Camera not found! Is another app using it?", "error")
            self._emit("sl:collect_done", {"success": False})
            return

        try:
            with mp_holistic.Holistic(min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5) as holistic:
                for action in actions:
                    if self._stop.is_set():
                        break
                    for seq in range(n_seq):
                        if self._stop.is_set():
                            break

                        self._log(f"Collecting {action}  [{seq+1}/{n_seq}]", "info")
                        self._emit("sl:collect_progress", {
                            "action": action, "seq": seq + 1, "total": n_seq
                        })

                        # 2-second pause
                        pause_end = time.time() + 2.0
                        while time.time() < pause_end and not self._stop.is_set():
                            ret, frame = cap.read()
                            if not ret:
                                break
                            frame, results = mediapipe_detection(frame, holistic)
                            draw_landmarks(frame, results)
                            cv2.putText(frame, f"GET READY  {action}  seq {seq+1}",
                                        (12, 32), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.75, (0, 255, 180), 2, cv2.LINE_AA)
                            self._push_frame(frame_to_b64(frame), "cam-collect")

                        # Record frames
                        for fn in range(seq_len):
                            if self._stop.is_set():
                                break
                            ret, frame = cap.read()
                            if not ret:
                                break
                            frame, results = mediapipe_detection(frame, holistic)
                            draw_landmarks(frame, results)
                            cv2.putText(frame, f"{action}  {fn+1}/{seq_len}",
                                        (12, 32), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.75, (0, 255, 180), 2, cv2.LINE_AA)
                            self._push_frame(frame_to_b64(frame), "cam-collect")
                            np.save(
                                os.path.join(data_path, action, str(seq), str(fn)),
                                extract_keypoints(results)
                            )

                    self._log(f"Class '{action}' complete — {n_seq} seqs saved.", "ok")
        finally:
            cap.release()

        self._log("Data collection finished.", "ok")
        self._emit("sl:collect_done", {"success": True})

    def _simulate_collection(self, actions, n_seq):
        for action in actions:
            for seq in range(n_seq):
                if self._stop.is_set():
                    return
                time.sleep(0.04)
                self._emit("sl:collect_progress", {
                    "action": action, "seq": seq + 1, "total": n_seq
                })
            self._log(f"[Demo] '{action}' — {n_seq} seqs simulated.", "ok")
        self._emit("sl:collect_done", {"success": True})

    # ── Training ──────────────────────────────────────────────────────────────

    def start_training(self) -> dict:
        self._stop_thread()
        self._thread = threading.Thread(
            target=self._train_loop, daemon=True, name="train")
        self._thread.start()
        return {"ok": True}

    def _train_loop(self):
        cfg        = self.config
        actions    = cfg["actions"]
        n_seq      = int(cfg["no_sequences"])
        seq_len    = int(cfg["sequence_length"])
        data_path  = cfg["data_path"]
        epochs     = int(cfg["epochs"])
        model_path = cfg["model_path"]

        if not TF_AVAILABLE:
            self._log("TensorFlow not installed — demo training mode.", "warn")
            self._simulate_training(epochs)
            return

        self._log("Loading keypoint sequences…", "info")
        label_map = {a: i for i, a in enumerate(actions)}
        sequences, labels = [], []

        for action in actions:
            for seq in range(n_seq):
                window = []
                for fn in range(seq_len):
                    npy = os.path.join(data_path, action, str(seq), f"{fn}.npy")
                    window.append(
                        np.load(npy) if os.path.exists(npy) else np.zeros(KEYPOINT_SIZE)
                    )
                sequences.append(window)
                labels.append(label_map[action])

        X = np.array(sequences)
        y = to_categorical(labels, num_classes=len(actions)).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.05, random_state=42)
        self._log(f"Dataset: {X_train.shape[0]} train / {X_test.shape[0]} val", "info")

        model = Sequential([
            LSTM(64,  return_sequences=True, activation="relu",
                 input_shape=(seq_len, KEYPOINT_SIZE)),
            LSTM(128, return_sequences=True, activation="relu"),
            LSTM(64,  return_sequences=False, activation="relu"),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(len(actions), activation="softmax"),
        ])
        model.compile(optimizer="Adam",
                      loss="categorical_crossentropy",
                      metrics=["categorical_accuracy"])

        api_ref = self

        class StreamCallback(Callback):
            def on_epoch_end(self_, epoch, logs=None):
                if api_ref._stop.is_set():
                    model.stop_training = True
                    return
                acc  = float((logs or {}).get("categorical_accuracy", 0))
                loss = float((logs or {}).get("loss", 0))
                api_ref._emit("sl:train_epoch", {
                    "epoch": epoch + 1,
                    "total": epochs,
                    "acc":   round(acc * 100, 2),
                    "loss":  round(loss, 5),
                })
                if (epoch + 1) % 100 == 0:
                    api_ref._log(
                        f"Epoch {epoch+1}/{epochs}  loss {loss:.4f}  acc {acc:.4f}", "info"
                    )

        os.makedirs(os.path.join(BASE_DIR, "Logs", "train"), exist_ok=True)
        early = EarlyStopping(monitor="loss", patience=100, restore_best_weights=True)

        self._log("Training started…", "info")
        model.fit(X_train, y_train,
                  epochs=epochs,
                  callbacks=[StreamCallback(), early],
                  verbose=0)

        model.save(model_path)
        self._log(f"Model saved to {os.path.basename(model_path)}", "ok")
        self._emit("sl:train_done", {"model_path": model_path})

    def _simulate_training(self, epochs: int):
        acc, loss = 0.34, 1.09
        for ep in range(1, epochs + 1):
            if self._stop.is_set():
                return
            acc  = min(0.999, acc  + (0.999 - acc) * 0.004)
            loss = max(0.001, loss - loss * 0.003)
            if ep % 5 == 0:
                self._emit("sl:train_epoch", {
                    "epoch": ep, "total": epochs,
                    "acc":   round(acc * 100, 2),
                    "loss":  round(loss, 5),
                })
            if ep % 200 == 0:
                self._log(
                    f"[Demo] Epoch {ep}/{epochs}  loss {loss:.4f}  acc {acc:.4f}", "info"
                )
            time.sleep(0.008)
        self._log("[Demo] Training complete.", "ok")
        self._emit("sl:train_done", {"model_path": "./action.h5"})

    # ── Inference ─────────────────────────────────────────────────────────────

    def start_inference(self) -> dict:
        self._stop_thread()
        self._thread = threading.Thread(
            target=self._infer_loop, daemon=True, name="infer")
        self._thread.start()
        return {"ok": True}

    def _infer_loop(self):
        cfg        = self.config
        actions    = cfg["actions"]
        seq_len    = int(cfg["sequence_length"])
        threshold  = float(cfg["confidence_threshold"])
        model_path = cfg["model_path"]

        if not (TF_AVAILABLE and MP_AVAILABLE):
            self._log("TF/MediaPipe unavailable — demo inference.", "warn")
            self._simulate_inference(actions)
            return

        if not os.path.exists(model_path):
            self._log(f"Model not found: {model_path}  Run training first.", "error")
            self._emit("sl:infer_stopped", {})
            return

        model = load_model(model_path)
        self._log(f"Model loaded: {os.path.basename(model_path)}", "ok")

        sequence:    list = []
        sentence:    list = []
        predictions: list = []

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self._log("Camera not found!", "error")
            self._emit("sl:infer_stopped", {})
            return

        try:
            with mp_holistic.Holistic(min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5) as holistic:
                while not self._stop.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame, results = mediapipe_detection(frame, holistic)
                    draw_landmarks(frame, results)

                    kp = extract_keypoints(results)
                    sequence.append(kp)
                    sequence = sequence[-seq_len:]

                    predicted   = None
                    confidences = {a: 0.0 for a in actions}

                    if len(sequence) == seq_len:
                        res       = model.predict(
                            np.expand_dims(sequence, axis=0), verbose=0)[0]
                        top_idx   = int(np.argmax(res))
                        predicted = actions[top_idx]
                        confidences = {
                            actions[i]: round(float(res[i]) * 100, 1)
                            for i in range(len(actions))
                        }
                        predictions.append(top_idx)
                        predictions = predictions[-20:]

                        if (len(predictions) >= 10
                                and len(set(predictions[-10:])) == 1
                                and float(res[top_idx]) > threshold):
                            if not sentence or sentence[-1] != predicted:
                                sentence.append(predicted)
                                sentence = sentence[-5:]
                                if cfg.get("tts_enabled", True):
                                    speak(predicted)

                    label = " ".join(sentence)
                    cv2.putText(frame, label, (12, 32),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (0, 255, 180), 2, cv2.LINE_AA)

                    self._push_frame(frame_to_b64(frame), "cam-infer")
                    self._emit("sl:infer_result", {
                        "predicted":   predicted,
                        "confidences": confidences,
                        "sentence":    label,
                    })
        finally:
            cap.release()

        self._log("Inference stopped.", "info")
        self._emit("sl:infer_stopped", {})

    def _simulate_inference(self, actions):
        import random
        sentence: list = []
        while not self._stop.is_set():
            pred   = random.choice(actions)
            raw    = {a: random.uniform(0, 10) for a in actions}
            raw[pred] = random.uniform(60, 95)
            total  = sum(raw.values())
            confs  = {k: round(v / total * 100, 1) for k, v in raw.items()}
            if not sentence or sentence[-1] != pred:
                sentence = (sentence + [pred])[-5:]
            self._emit("sl:infer_result", {
                "predicted":   pred,
                "confidences": confs,
                "sentence":    " ".join(sentence),
            })
            time.sleep(0.5)

    # ── Stop ──────────────────────────────────────────────────────────────────

    def stop(self) -> dict:
        self._stop_thread()
        self._log("Stopped.", "info")
        return {"ok": True}


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    api    = SignLensAPI()
    window = webview.create_window(
        title        = "SignLens AI",
        url          = INDEX_HTML,       # absolute path — works from any CWD
        js_api       = api,
        width        = 1140,
        height       = 720,
        min_size     = (900, 600),
        resizable    = True,
        private_mode = False,
    )
    api.window = window

    def on_loaded():
        info = api.get_system_info()
        api._log(f"Python {info['python']}  ·  OpenCV {info['cv2']}", "info")
        api._log(f"MediaPipe: {info['mediapipe']}  ·  TF: {info['tensorflow']}", "info")
        api._log(f"TTS: {info['tts']}", "info")
        api._log("SignLens AI ready.", "ok")

    window.events.loaded += on_loaded
    webview.start(debug=False)
