# Cortexium Tech Stack

Cortexium is a production-grade, AI-native social intelligence platform designed for real-time human analysis and relationship mapping. It employs a multi-agent architecture with a high-performance Python backend and a React-based neural dashboard.

## 🧠 Backend (Intelligence Core)
The backend is a real-time multi-agent system built with **FastAPI** and **Python 3.10+**.

| Layer | Tools & Technologies |
| :--- | :--- |
| **API Framework** | FastAPI, Uvicorn, WebSockets (Real-time streams) |
| **Vision (AI)** | YOLOv8 (Ultralytics), InsightFace, DeepFace (Emotion), MediaPipe |
| **Audio (AI)** | Pyannote.audio (Diarization), PyAudio, Torch/Torchaudio |
| **LLM Interface** | Ollama (Local Model Hosting) |
| **Vector Memory** | ChromaDB (High-performance Vector Database) |
| **Graph DB** | Neo4j (Entity Relationship Mapping) |
| **Persistence** | SQLAlchemy + AioSQLite (Async Structured Storage) |
| **Environment** | python-dotenv, config.py (Unified Configuration) |

## 🖥️ Frontend (Neural Dashboard)
A high-fidelity dashboard built for real-time visualization of AI agents' findings.

| Component | Technology |
| :--- | :--- |
| **Framework** | React 18+ powered by Vite |
| **Neural Maps** | Vis-network (Interactive Knowledge Graph) |
| **Analytics** | Recharts (Data Visualization) |
| **Icons** | Lucide React |
| **Styling** | Premium Vanilla CSS (Custom tokens, Glassmorphism, Animations) |
| **Data Fetching** | WebSockets + HTTP (FastAPI Integration) |

## ⚙️ Infrastructure & Operations
Designed for local deployment with maximum performance.

- **Operating System:** Windows Optimized
- **Orchestration:** `run_all.bat` (Integrated process management)
- **Logging:** Loguru + Rich (High-speed structured logging)
- **Task Scheduling:** Schedule (Background cleanup/updates)
- **Data Exchange:** RDFLib (JSON-LD & RDF Semantic Exports)

## 📁 Repository Structure
- `/api`: FastAPI endpoints and WebSocket controllers.
- `/core`: Intelligence logic (storage, vision, audio, LLM modules).
- `/dashboard`: React/Vite frontend application.
- `/models`: Pre-trained YOLO/Face weights and configuration.
- `/data`: Local persistent storage (SQLite, Chroma, Neo4j).
