# 🧠 Cortexium — Social Intelligence Agent

![Cortexium Banner](assets/images/banner.png)

A full-stack AI agent that performs **real-time face recognition, social interaction logging, HUD rendering, multimodal behavior understanding**, and builds a **behavioral knowledge graph to train robots** — all running 100% locally.

![Cortexium HUD Mockup](assets/images/hud_mockup.png)

---

## 🗂 Project Structure

```
cortexium-social/
├── main.py                        # 🚀 Main orchestration loop
├── config.py                      # ⚙️  Central config (loaded from .env)
├── requirements.txt
├── .env.example                   # Copy to .env and fill in
│
├── core/
│   ├── vision/
│   │   ├── face_pipeline.py       # InsightFace detection + ArcFace recognition
│   │   ├── body_pose.py           # MediaPipe Holistic (gestures, proximity)
│   │   └── emotion_detector.py    # DeepFace per-face emotion
│   ├── audio/
│   │   ├── speech_engine.py       # Whisper STT (mic streaming)
│   │   └── diarization.py         # PyAnnote speaker diarization
│   ├── fusion/
│   │   └── interaction_analyzer.py# Multimodal fusion → InteractionEvent
│   ├── enrollment/
│   │   └── enrollment_flow.py     # Voice-first face enrollment
│   ├── hud/
│   │   └── hud_renderer.py        # OpenCV + PyGame HUD overlay
│   ├── knowledge/
│   │   └── behavior_extractor.py  # Robot knowledge → YAML/JSON-LD/CSV
│   └── storage/
│       ├── db.py                  # SQLite + ChromaDB setup
│       ├── person_registry.py     # Enroll, identify, log interactions
│       └── graph_db.py            # Neo4j social knowledge graph
│
├── api/
│   └── main.py                    # FastAPI REST + WebSocket server
│
├── dashboard/                     # React + Vite web dashboard
│   ├── src/
│   │   ├── App.jsx                # Main dashboard app
│   │   └── index.css
│   ├── index.html
│   └── package.json
│
├── scripts/
│   ├── setup.py                   # One-time setup checker
│   ├── test_recognition.py        # Face recognition validation
│   └── test_knowledge_export.py   # Robot export validation
│
├── data/                          # Auto-created
│   ├── faces/
│   ├── logs/                      # SQLite DB
│   └── knowledge/
└── exports/robot_training/        # YAML + JSON-LD + CSV exports
```

---

## ⚡ Quick Start

### 1. Prerequisites

```bash
# Python 3.11+
pip install -r requirements.txt
```

| Service | Link | Cost |
|---|---|---|
| Ollama (LLaMA 3.1 8B) | https://ollama.com/download | Free |
| Neo4j Community | https://neo4j.com/download-center/#community | Free |
| HuggingFace token | https://huggingface.co/settings/tokens | Free |

### 2. Setup

```bash
# Check everything is ready
python scripts/setup.py

# Copy and edit .env
cp .env.example .env
# Edit .env: set HF_TOKEN, CAMERA_SOURCE, etc.

# Pull LLaMA model
ollama pull llama3.1:8b
```

### 3. Run

**Terminal 1 — Main Vision Agent:**
```bash
# Webcam
python main.py

# Video file
python main.py --source "path/to/video.mp4"

# Vision only (no mic/audio)
python main.py --no-audio
```

**Terminal 2 — API Server:**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8765 --reload
```

**Terminal 3 — Web Dashboard:**
```bash
cd dashboard
npm install
npm run dev
# Open: http://localhost:3000
```

---

## 🎤 Voice Enrollment

When a new face is detected, the HUD shows:
```
👤 New person detected — say: "This is [Name]"
```

Just say `"This is Alice"` or `"Meet Bob"` while looking at the camera.
The system will:
1. Transcribe via Whisper
2. Extract the name
3. Save face embedding to ChromaDB
4. Create profile in SQLite
5. Add node in Neo4j graph
6. Confirm: `"✓ Enrolled: Alice"`

---

## 🤖 Robot Knowledge Export

```bash
# Via API
curl http://localhost:8765/knowledge/export

# Via script
python scripts/test_knowledge_export.py
```

Exports to `exports/robot_training/`:
- `behavior_patterns.yaml` — symbolic AI / rule-based robots
- `knowledge_graph.jsonld` — semantic reasoning (RDF)
- `interaction_dataset.csv` — ML training data

---

## 🔌 API Reference

| Method | Path | Description |
|---|---|---|
| POST | `/enroll` | Enroll new face (name + embedding) |
| GET | `/persons` | List all known people |
| POST | `/identify` | Identify face by embedding |
| GET | `/interactions` | Recent interaction events |
| GET | `/knowledge/patterns` | Current behavior patterns |
| GET | `/knowledge/export` | Trigger full robot export |
| WS | `/stream` | Live event WebSocket stream |

---

## ⚠️ Privacy Notice

This system processes **biometric data** (faces, voice). You are responsible for:
- Obtaining consent from people being recorded
- Complying with applicable laws (GDPR, CCPA, etc.)
- Keeping all data local (no cloud by default)

---

## 🛠 Hardware Recommendations

| Component | Minimum | Recommended |
|---|---|---|
| GPU | GTX 1060 6GB | RTX 3070+ |
| RAM | 16GB | 32GB |
| Camera | 1080p 30fps | 4K 60fps |
| Mic | Any USB mic | Directional lapel mic |
