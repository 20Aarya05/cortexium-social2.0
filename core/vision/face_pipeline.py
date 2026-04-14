"""
Vision Pipeline — Face Detection, Recognition, and Multi-Person Tracking.

Uses:
  - InsightFace (ArcFace / RetinaFace) for detection + embedding
  - ByteTrack (via Ultralytics) for persistent cross-frame IDs
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from loguru import logger

try:
    import insightface
    from insightface.app import FaceAnalysis
    _INSIGHTFACE_OK = True
except ImportError:
    _INSIGHTFACE_OK = False
    logger.warning("[Face] insightface not installed — face recognition disabled")

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import config as cfg
from core.storage.person_registry import identify_person


@dataclass
class TrackedFace:
    track_id:    int
    bbox:        tuple[int, int, int, int]   # x1, y1, x2, y2
    person_id:   Optional[str]   = None
    person_name: str             = "Unknown"
    confidence:  float           = 0.0
    emotion:     str             = "neutral"
    embedding:   Optional[list]  = field(default=None, repr=False)
    last_updated: float          = field(default_factory=time.time)


class FacePipeline:
    """Initialise once; call process_frame(frame) each iteration."""

    def __init__(self):
        self._app: Optional[FaceAnalysis] = None
        self._track_map: dict[int, TrackedFace] = {}   # track_id → TrackedFace
        self._reidentify_interval = 30   # re-run recognition every N frames
        self._frame_count = 0
        self._init_model()

    def _init_model(self):
        if not _INSIGHTFACE_OK:
            return
        try:
            self._app = FaceAnalysis(
                name=cfg.FACE_RECOGNITION_MODEL,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            # Multi-user tuning: increase det_size for better distance detection
            # but keep it balanced to avoid lag. 480x480 is a good sweet spot.
            self._app.prepare(ctx_id=0, det_size=(480, 480))
            logger.info(f"[Face] InsightFace model '{cfg.FACE_RECOGNITION_MODEL}' loaded (Multi-user optimized)")
        except Exception as e:
            logger.error(f"[Face] Failed to load InsightFace: {e}")
            self._app = None

    # ── Main entry ────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> list[TrackedFace]:
        """Run detection + recognition on a BGR frame. Returns tracked faces."""
        self._frame_count += 1

        if self._app is None:
            return []

        try:
            faces = self._app.get(frame)
        except Exception as e:
            logger.debug(f"[Face] Detection error: {e}")
            return []

        results: list[TrackedFace] = []
        seen_ids: set[int] = set()

        for face in faces:
            bbox = self._to_bbox(face.bbox)
            # Use det_score as a proxy track_id seed (real ByteTrack hooks in main loop)
            track_id = self._assign_track_id(face.embedding)
            seen_ids.add(track_id)

            tf = self._track_map.get(track_id)
            should_identify = (
                tf is None
                or tf.person_id is None
                or self._frame_count % self._reidentify_interval == 0
            )

            if tf is None:
                tf = TrackedFace(track_id=track_id, bbox=bbox)
                self._track_map[track_id] = tf

            tf.bbox = bbox
            tf.embedding = face.embedding.tolist()
            tf.last_updated = time.time()

            if should_identify and face.embedding is not None:
                person, conf = identify_person(face.embedding.tolist())
                if person:
                    tf.person_id   = person.id
                    tf.person_name = person.name
                    tf.confidence  = conf
                else:
                    tf.person_id   = None
                    tf.person_name = "Unknown"
                    tf.confidence  = conf

            results.append(tf)

        # Prune stale tracks (not seen for > 5 seconds)
        now = time.time()
        stale = [tid for tid, tf in self._track_map.items()
                 if now - tf.last_updated > 5.0 and tid not in seen_ids]
        for tid in stale:
            del self._track_map[tid]

        return results

    def get_face_crop(self, frame: np.ndarray, bbox: tuple) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        return frame[max(0, y1):y2, max(0, x1):x2]

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _to_bbox(self, raw) -> tuple[int, int, int, int]:
        return tuple(int(v) for v in raw[:4])

    def _assign_track_id(self, embedding: np.ndarray) -> int:
        """
        Simple cosine-nearest-neighbour track from existing embeddings.
        Falls back to new ID if no close match (threshold 0.25 distance).
        """
        if not self._track_map or embedding is None:
            return self._new_track_id()

        emb = embedding / (np.linalg.norm(embedding) + 1e-6)
        best_id, best_dist = None, float("inf")

        for tid, tf in self._track_map.items():
            if tf.embedding is None:
                continue
            stored = np.array(tf.embedding)
            stored = stored / (np.linalg.norm(stored) + 1e-6)
            dist = float(np.dot(emb, stored))   # cosine similarity (higher = closer)
            dist = 1.0 - dist                   # convert to distance
            if dist < best_dist:
                best_dist = dist
                best_id = tid

        if best_dist < 0.25:
            return best_id
        return self._new_track_id()

    def _new_track_id(self) -> int:
        existing = set(self._track_map.keys())
        tid = 1
        while tid in existing:
            tid += 1
        return tid
