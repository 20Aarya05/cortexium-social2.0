"""
Cortexium Social Intelligence Agent
====================================
Shared configuration loaded from .env
"""

import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# ── GPU / CUDA ───────────────────────────────────────
# Inject CUDA bin paths to resolve ONNXRuntime-GPU "DLL Error 126"
import sys
if sys.platform == "win32":
    cuda_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin",
        r"C:\Program Files\NVIDIA\CUDNN\v9.15\bin\12.9",
        r"C:\Program Files\NVIDIA\CUDNN\v9.15\bin\13.0",
    ]
    current_path = os.environ.get("PATH", "")
    for cp in cuda_paths:
        if os.path.exists(cp) and cp not in current_path:
            os.environ["PATH"] = cp + os.pathsep + os.environ["PATH"]
            # Also use add_dll_directory for Python 3.8+
            try:
                os.add_dll_directory(cp)
            except Exception:
                pass


# ── Camera ────────────────────────────────────────────
CAMERA_SOURCE: str = os.getenv("CAMERA_SOURCE", "0")
# Normalise: if it's a digit string, convert to int (webcam index)
if CAMERA_SOURCE.isdigit():
    CAMERA_SOURCE = int(CAMERA_SOURCE)

# ── Audio ─────────────────────────────────────────────
MIC_SOURCE: str = os.getenv("MIC_SOURCE", None)
# Normalise: if it's a digit string, convert to int (device index)
if MIC_SOURCE and MIC_SOURCE.isdigit():
    MIC_SOURCE = int(MIC_SOURCE)

# ── Face recognition ──────────────────────────────────
FACE_CONFIDENCE_THRESHOLD: float = float(
    os.getenv("FACE_CONFIDENCE_THRESHOLD", "0.6")
)
FACE_RECOGNITION_MODEL: str = os.getenv(
    "FACE_RECOGNITION_MODEL", "buffalo_l"
)

# ── Paths ─────────────────────────────────────────────
DATA_DIR        = ROOT / "data"
FACES_DIR       = DATA_DIR / "faces"
LOGS_DIR        = DATA_DIR / "logs"
KNOWLEDGE_DIR   = DATA_DIR / "knowledge"
MODELS_DIR      = ROOT / "models"
EXPORTS_DIR     = ROOT / "exports" / "robot_training"

for _d in [FACES_DIR, LOGS_DIR, KNOWLEDGE_DIR, MODELS_DIR, EXPORTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Database ──────────────────────────────────────────
SQLITE_DB       = LOGS_DIR / "cortexium.db"
CHROMA_DIR      = DATA_DIR / "chroma"

# ── Neo4j ─────────────────────────────────────────────
NEO4J_URI       = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER      = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD  = os.getenv("NEO4J_PASSWORD", "cortexium")

# ── Ollama / LLM ─────────────────────────────────────
OLLAMA_HOST     = os.getenv("OLLAMA_HOST",  "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# ── HuggingFace ──────────────────────────────────────
HF_TOKEN        = os.getenv("HF_TOKEN", "")

# ── API ───────────────────────────────────────────────
API_HOST        = os.getenv("API_HOST", "0.0.0.0")
API_PORT        = int(os.getenv("API_PORT", "8765"))

# ── HUD ───────────────────────────────────────────────
HUD_WIDTH       = int(os.getenv("HUD_WIDTH",  "1280"))
HUD_HEIGHT      = int(os.getenv("HUD_HEIGHT", "720"))
HUD_FULLSCREEN  = os.getenv("HUD_FULLSCREEN", "false").lower() == "true"
