"""
Emotion Detector — DeepFace per-face crop.
Outputs one of 7 emotions with confidence score.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

try:
    from deepface import DeepFace
    _DF_OK = True
except ImportError:
    _DF_OK = False
    logger.warning("[Emotion] deepface not installed — emotion detection disabled")

_EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprised"]


class EmotionDetector:
    def __init__(self):
        # Warm-up build (downloads model on first run)
        if _DF_OK:
            logger.info("[Emotion] DeepFace emotion model ready (lazy load)")

    def detect(self, face_crop_bgr: np.ndarray) -> tuple[str, float]:
        """
        Returns (emotion_label, confidence 0-1).
        Falls back to ('neutral', 0.0) on error.
        """
        if not _DF_OK or face_crop_bgr is None or face_crop_bgr.size == 0:
            return "neutral", 0.0

        # face crop must be at least 48×48
        h, w = face_crop_bgr.shape[:2]
        if h < 48 or w < 48:
            return "neutral", 0.0

        try:
            result = DeepFace.analyze(
                face_crop_bgr,
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
            )
            if isinstance(result, list):
                result = result[0]
            dominant = result["dominant_emotion"]
            scores   = result["emotion"]            # dict label→score
            conf     = scores.get(dominant, 0.0) / 100.0
            return dominant, round(conf, 3)
        except Exception as e:
            logger.debug(f"[Emotion] Detection failed: {e}")
            return "neutral", 0.0
