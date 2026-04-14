# Cortexium Social Intelligence Agent — Implementation Plan

A full-stack AI agent for smart glasses that performs real-time face recognition, social interaction logging, HUD rendering, multi-modal understanding of human behavior, and builds a behavioral knowledge graph to train robots.

---

## User Review Required

> [!IMPORTANT]
> This system processes personal biometric data (faces, voice). You are responsible for obtaining consent from people being recorded. Always follow applicable privacy laws (GDPR, CCPA, etc.).

> [!WARNING]
> Face recognition from far distances (long-range) requires high-quality camera hardware. The software stack will be designed for this, but results depend on the actual glasses hardware specs.

> [!CAUTION]
> Some APIs require registration. All API links are provided in each section. For the initial build, **everything runs locally** — no cloud calls. Cloud upgrade paths are described at the end.

---

## Architecture Overview

```
[Smart Glasses Camera] → [Video Pipeline]─┐
                                           ├→ [Multi-Modal AI Core] → [HUD Renderer]
[Smart Glasses Mic]   → [Audio Pipeline]─┘          │
                                                     ↓
                                        [Interaction Logger & Graph DB]
                                                     │
                                                     ↓
                                        [Robot Behavioral Knowledge Exporter]
```

---

## Tech Stack

| Layer | Technology | Reason |
|---|---|---|
| Core Language | **Python 3.11+** | Best ML ecosystem |
| Face Detection | **InsightFace / DeepFace + RetinaFace** | Long-range, accurate |
| Face Recognition | **ArcFace (via InsightFace)** | State-of-art, local |
| Object Tracking | **ByteTrack + YOLO v8** | Real-time multi-person |
| Pose Estimation | **MediaPipe Holistic** | Body language analysis |
| Audio Processing | **Whisper (local)** | Real-time speech-to-text |
| Speaker Diarization | **PyAnnote Audio** | Who is speaking |
| Emotion Detection | **DeepFace (local)** | Facial emotion from far |
| NLP / Behavior Understanding | **Ollama + LLaMA 3.1 (local LLM)** | Interaction summarization |
| Video Capture | **OpenCV + FFmpeg** | Camera input |
| HUD Rendering | **PyGame / Electron overlay** | Glasses HUD display |
| Local Database | **SQLite + ChromaDB (vector)** | Face profiles + embeddings |
| Knowledge Graph | **Neo4j Community (local)** | Social graph of interactions |
| Robot Knowledge Export | **JSON-LD / RDF / YAML** | Structured behavior data |
| Dashboard UI | **React + Vite** | Web dashboard |
| API Server | **FastAPI** | Internal service bus |

---

## Proposed Changes

### Component 1: Project Scaffold & Core Infrastructure

#### [NEW] Project Root Structure
```
cortexium-social/
├── core/                    # Main AI pipeline
│   ├── vision/              # Face, body, emotion detection
│   ├── audio/               # Speech, diarization
│   ├── fusion/              # Multimodal fusion
│   └── hud/                 # HUD rendering engine
├── data/
│   ├── faces/               # Local face profile storage
│   ├── logs/                # Interaction logs (SQLite)
│   └── knowledge/           # Robot behavior knowledge base
├── api/                     # FastAPI service
├── dashboard/               # React HUD dashboard
├── models/                  # Downloaded ML model weights
├── scripts/                 # Setup & training scripts
└── exports/                 # Robot training data exports
```

---

### Component 2: Vision Pipeline

#### [NEW] `core/vision/face_pipeline.py`
- **Long-range face detection**: RetinaFace with ResNet-50 backbone
- **Face recognition**: ArcFace embeddings (512-dim vectors) stored in ChromaDB
- **Multi-face tracking**: ByteTrack ID persistence across frames
- **New person enrollment**: Triggered when no match > threshold; prompts voice "Who is this?"
- **Known person overlay**: Shows name, relationship, last seen info on HUD

#### [NEW] `core/vision/body_pose.py`
- MediaPipe Holistic — skeleton landmarks, hand gestures, head orientation
- Extracts social signals: nodding, pointing, handshaking, waving
- Emotion from body language (crossed arms, leaning in, etc.)

#### [NEW] `core/vision/emotion_detector.py`
- DeepFace with FER+ model — runs per-face crop
- Maps to: happy, sad, angry, surprised, neutral, disgust, fear
- Outputs confidence scores → fed to interaction logger

---

### Component 3: Audio Pipeline

#### [NEW] `core/audio/speech_engine.py`
- **Whisper (small/medium model, local)** — real-time transcription
- Runs on GPU if available, falls back to CPU
- Chunked streaming: 3-second windows with overlap

#### [NEW] `core/audio/diarization.py`
- **PyAnnote Audio 3.1** — identifies who is speaking
- Links speaker segments to face-tracked IDs via fusion module
- Requires HuggingFace token (free): https://huggingface.co/settings/tokens
- Model: `pyannote/speaker-diarization-3.1` — free for local use

---

### Component 4: Multimodal Fusion & Behavior Understanding

#### [NEW] `core/fusion/interaction_analyzer.py`
- Fuses: face ID + speaker ID + emotion + pose + transcribed speech
- Every 10 seconds → creates an "Interaction Event":
  ```json
  {
    "timestamp": "...",
    "persons": ["Alice", "Unknown#3"],
    "speaker": "Alice",
    "emotion": "happy",
    "gesture": "nodding",
    "speech": "Nice to meet you!",
    "social_context": "introduction",
    "proximity": "close"
  }
  ```
- **Local LLM (Ollama + LLaMA 3.1 8B)**: Summarizes and tags social context
  - Install: https://ollama.com/download
  - Model: `ollama pull llama3.1:8b`

---

### Component 5: Face Profile & Social Graph Storage

#### [NEW] `data/` — Local Databases
- **ChromaDB**: Vector store for ArcFace embeddings (face search in <5ms)
- **SQLite**: Person profiles, interaction history, timestamps
- **Neo4j Community Edition (local)**: Social knowledge graph
  - Download: https://neo4j.com/download-center/#community
  - Nodes: Person, Interaction, Location, Topic
  - Edges: MET, KNOWS, SPOKE_WITH, INTRODUCED_BY

#### [NEW] `core/storage/person_registry.py`
- `enroll_person(face_crop, name, metadata)` — adds new person
- `identify_person(face_crop)` → returns best match + confidence
- `log_interaction(person_a, person_b, event)` — writes to graph + SQLite
- `get_relationship(person_a, person_b)` — retrieves social context

---

### Component 6: HUD Rendering Engine

#### [NEW] `core/hud/hud_renderer.py`
- Runs as a **transparent overlay** (PyGame or system overlay)
- Designed for a **1920×1080 or lower-res glasses display** (configurable)
- HUD Elements:
  - 🔲 Bounding boxes around faces with name + emotion badge
  - 📊 Relationship info panel (right side): "Known: 3 months | Met at: Work"
  - 🎙️ Live transcription ticker (bottom)
  - 🧠 AI insight bubble: "Alice seems excited. She usually is when talking about tech."
  - 🆕 "New person detected" enrollment prompt
  - ⏱️ Session timer + interaction count

#### [NEW] `dashboard/` — React Web Dashboard
- Vite + React
- Live view of: current session, all known faces, interaction timeline
- Knowledge graph visualizer (using vis.js or D3)
- Export robot training data

---

### Component 7: New Person Introduction Flow

#### [NEW] `core/enrollment/enrollment_flow.py`

**Automatic Flow:**
1. Face detected → no match (confidence < 0.6)
2. HUD shows: `"👤 New person detected — Who is this?"`
3. System listens for voice: `"This is [Name]"` or `"Meet [Name]"`
4. Whisper transcribes → name extracted via regex/LLM NER
5. Face crops saved → ArcFace embedding computed → stored in ChromaDB
6. Profile created in SQLite with: name, first_seen, introduced_by, context
7. Neo4j: `(You)-[:MET]->(NewPerson)` edge created
8. HUD confirms: `"✓ [Name] saved"`

**Manual Flow (via companion app):**
- Web dashboard shows "Unidentified faces from today"
- User can label, merge duplicates, add notes

---

### Component 8: Robot Behavioral Knowledge System

#### [NEW] `data/knowledge/` — Behavior Knowledge Base

This is the robot training knowledge system. It converts observed human interactions into structured, machine-readable behavioral patterns.

**Schema:**
```yaml
behavior_pattern:
  id: "greet_introduction_001"
  trigger: "new_person_encountered"
  context: "professional_setting"
  actions:
    - make_eye_contact: true
    - extend_hand: true
    - say: "Nice to meet you, [name]"
    - smile: true
  outcome: "positive_reception"
  frequency: 47
  confidence: 0.89
  observed_from: ["Alice", "Bob", "Charlie"]
```

#### [NEW] `core/knowledge/behavior_extractor.py`
- Reads interaction events from the DB
- Clusters similar events using embedding similarity
- Generates behavior patterns via LLM:
  - Prompt: "Given these observations, what is the general social rule being followed?"
- Exports to JSON-LD (robot-readable RDF triples)

#### [NEW] `exports/robot_training/`
- **YAML behavior patterns** — for symbolic AI / rule-based robots
- **JSON-LD knowledge graph** — for semantic reasoning
- **CSV datasets** — for ML training (emotion, gesture, speech → action)
- **Video clip index** — labeled video segments for robot vision training

---

### Component 9: API Server

#### [NEW] `api/main.py` — FastAPI
- `POST /enroll` — enroll new face
- `GET /identify/{face_id}` — get person info
- `GET /interactions` — recent interaction log
- `GET /knowledge/patterns` — export behavior patterns
- `POST /hud/config` — update HUD layout
- `WS /stream` — WebSocket for live HUD data

---

## API Keys & Downloads Required

| Service | Purpose | Link | Cost |
|---|---|---|---|
| HuggingFace Token | PyAnnote diarization model download | https://huggingface.co/settings/tokens | **Free** |
| Ollama | Local LLM runtime | https://ollama.com/download | **Free** |
| Neo4j Community | Graph database | https://neo4j.com/download-center/#community | **Free** |
| InsightFace | ArcFace model (auto-downloads) | Installed via pip | **Free** |
| Whisper | STT model (auto-downloads) | Installed via pip | **Free** |
| PyAnnote Audio | Speaker diarization | https://hf.co/pyannote/speaker-diarization-3.1 | **Free** (needs HF token) |

> [!NOTE]
> All processing is **100% local**. No data leaves your machine. Cloud upgrade paths (AWS Rekognition, Google Speech, etc.) can be added later.

---

## Hardware Recommendations

| Component | Minimum | Recommended |
|---|---|---|
| GPU | GTX 1060 6GB | RTX 3070+ (for real-time) |
| RAM | 16GB | 32GB |
| CPU | i5 10th gen | i7/i9 or Ryzen 9 |
| Camera | 1080p 30fps | 4K 60fps (for long-range) |
| Microphone | Any USB mic | Directional lapel mic |

---

## Verification Plan

### Automated Tests
- `pytest tests/` — unit tests for face pipeline, audio pipeline, storage
- `python scripts/test_recognition.py` — accuracy test with sample faces
- `python scripts/test_hud.py` — HUD rendering smoke test

### Manual Verification
- Feed a test video → verify faces are detected and tracked
- Introduce "test person" via voice → verify enrollment flow
- Check Neo4j browser (localhost:7474) → verify social graph is built
- Open React dashboard → verify live HUD stream works
- Export robot knowledge → verify YAML/JSON-LD structure is valid

---

## Open Questions

> [!IMPORTANT]
> **What smart glasses hardware are you targeting?**
> - Vuzix, RayBan Meta, custom Raspberry Pi glasses, or a phone-as-glasses setup?
> - This affects how the camera feed is captured and how the HUD is rendered.

> [!IMPORTANT]
> **Do you want the HUD on the glasses display itself, or mirrored on a phone/PC screen?**
> - For development, we can render on PC screen first and port to glasses later.

> [!IMPORTANT]
> **GPU available?**  
> - Real-time face recognition + Whisper + LLM all benefit heavily from GPU. What GPU do you have?

> [!NOTE]
> Should I implement the **voice enrollment** ("This is [Name]") immediately, or start with **manual dashboard enrollment** and add voice later?

> [!NOTE]
> For the **robot knowledge export**, do you have a specific robot platform in mind (ROS2, Boston Dynamics, custom), or should we use a generic format?
