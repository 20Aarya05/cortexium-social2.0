"""
Speaker Diarization — PyAnnote Audio 3.1.
Identifies WHO is speaking in each audio segment.
"""

from __future__ import annotations

import io
import threading
import time
from typing import Optional

import numpy as np
from loguru import logger

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import config as cfg

try:
    from pyannote.audio import Pipeline as PyannotePipeline
    _PYANNOTE_OK = True
except ImportError:
    _PYANNOTE_OK = False
    logger.warning("[Diarize] pyannote.audio not installed — diarization disabled")

try:
    import soundfile as sf
    _SF_OK = True
except ImportError:
    _SF_OK = False


class DiarizationEngine:
    def __init__(self):
        self._pipeline: Optional[PyannotePipeline] = None
        self._lock = threading.Lock()
        self._last_speaker: Optional[str] = None

        if _PYANNOTE_OK and cfg.HF_TOKEN:
            self._load()
        elif _PYANNOTE_OK and not cfg.HF_TOKEN:
            logger.warning(
                "[Diarize] HF_TOKEN not set. "
                "Set HF_TOKEN in .env to enable speaker diarization."
            )

    def _load(self):
        try:
            logger.info("[Diarize] Loading pyannote/speaker-diarization-3.1 …")
            self._pipeline = PyannotePipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=cfg.HF_TOKEN,
            )
            logger.info("[Diarize] Diarization pipeline loaded ✓")
        except Exception as e:
            logger.error(f"[Diarize] Load failed: {e}")

    def diarize(self, audio_np: np.ndarray, sample_rate: int = 16_000) -> dict[str, str]:
        """
        Returns {speaker_label: segment_description}.
        Falls back to {"SPEAKER_00": "unknown"} if unavailable.
        """
        if self._pipeline is None or not _SF_OK:
            return {"SPEAKER_00": "unknown"}

        with self._lock:
            try:
                buf = io.BytesIO()
                sf.write(buf, audio_np, sample_rate, format="WAV")
                buf.seek(0)
                diarization = self._pipeline({"uri": "stream", "audio": buf})
                segments: dict[str, str] = {}
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    segments[speaker] = f"{turn.start:.1f}s–{turn.end:.1f}s"
                self._last_speaker = list(segments.keys())[0] if segments else None
                return segments
            except Exception as e:
                logger.debug(f"[Diarize] Error: {e}")
                return {"SPEAKER_00": "unknown"}

    @property
    def last_speaker(self) -> Optional[str]:
        return self._last_speaker
