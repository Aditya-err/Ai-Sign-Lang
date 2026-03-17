# SignLens AI — Real-Time Sign Language Detector

A desktop application for real-time hand sign language recognition using MediaPipe, TensorFlow LSTM, and a PyWebView dark-theme UI.

---

## Features

- **3-Stage Pipeline**: Data Collection → Model Training → Live Translation
- **Live Camera Feed** with MediaPipe hand landmark overlay
- **LSTM Sequence Model** (3 LSTM layers + Dense) trained on keypoint sequences
- **Real-time Confidence Bars** per gesture class
- **Text-to-Speech** output via pyttsx3
- **Live Training Charts** (loss + accuracy curves streamed epoch by epoch)
- **Dark UI** — IBM Plex Mono + Syne fonts, Teal (#1D9E75) accent

---

## Requirements

- **Python 3.10** (MediaPipe is not yet compatible with 3.13)
- Webcam

```
pip install -r requirements.txt
```

---

## Quick Start

**Windows:**
```
run.bat
```

**All platforms:**
```bash
python app.py
```

---

## Pipeline Usage

### Step 1 — Collect Data
1. Launch app → click **Data Collect** in sidebar
2. Click **Start Collection**
3. Perform each gesture (A, B, C by default) in front of the camera
4. 30 sequences × 30 frames captured per class with a 2-second pause between sequences

### Step 2 — Train Model
1. Click **Train Model** → **Start Training**
2. Watch the live epoch / accuracy / loss dashboard
3. Training auto-stops early if loss plateaus (patience = 100 epochs)
4. Model saved as `action.h5`

### Step 3 — Live Translation
1. Click **Live Translate** → **Start Camera**
2. Perform trained gestures; prediction appears at top of feed
3. Stable predictions (10 consistent frames) trigger TTS speech output

---

## Configuration

Open **Settings** in the sidebar to change:

| Setting | Default | Description |
|---|---|---|
| Actions | A, B, C | Gesture class labels |
| Sequences per class | 30 | Videos to record |
| Frames per sequence | 30 | Frames per video |
| Training epochs | 2000 | Max epochs (early stopping enabled) |
| Confidence threshold | 70% | Minimum confidence to accept prediction |
| TTS rate | 150 wpm | Speech speed |

---

## Project Structure

```
├── app.py                  # Python backend (PyWebView + CV + TF)
├── index.html              # Frontend UI
├── requirements.txt        # Python dependencies
├── run.bat                 # Windows launcher
├── signlens_config.json    # Auto-generated config (after first run)
├── MP_Data/                # Auto-generated keypoint data
│   ├── A/
│   ├── B/
│   └── C/
├── action.h5               # Trained model (after training)
└── Logs/                   # TensorBoard logs
```

---

## Model Architecture

```
Input: (30 frames, 258 keypoints)
  → LSTM(64,  return_sequences=True)
  → LSTM(128, return_sequences=True)
  → LSTM(64,  return_sequences=False)
  → Dense(64, relu)
  → Dense(32, relu)
  → Dense(num_classes, softmax)
```

Keypoints = 33 pose × 4 + 21 left hand × 3 + 21 right hand × 3 = **258 features per frame**

---

## Troubleshooting

| Problem | Solution |
|---|---|
| Camera not opening | Close other apps using webcam; check device index (default: 0) |
| Low accuracy | Collect more data; ensure consistent lighting; keep background neutral |
| MediaPipe import error | Ensure Python 3.10 is active (`python --version`) |
| `action.h5` not found | Complete training before running inference |
| TTS not speaking | `pyttsx3` may need OS voice drivers installed |

---

## Tech Stack

- [OpenCV](https://opencv.org/) — camera capture & frame processing  
- [MediaPipe](https://mediapipe.dev/) — holistic hand/pose landmark detection  
- [TensorFlow / Keras](https://keras.io/) — LSTM sequence model  
- [PyWebView](https://pywebview.flowrl.com/) — desktop window serving HTML/JS UI  
- [pyttsx3](https://pyttsx3.readthedocs.io/) — offline text-to-speech  
- [scikit-learn](https://scikit-learn.org/) — train/test split  

---

## License

MIT
