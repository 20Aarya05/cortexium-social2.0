"""
Voiceprint Store — speaker embedding enrollment and identification.

Uses Resemblyzer (already installed) to extract 256-d d-vector embeddings
from raw audio. Falls back to MFCC-mean cosine if Resemblyzer unavailable.

Embeddings stored in ChromaDB collection "voice_embeddings" (separate from
the existing "face_embeddings" collection).
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np
from loguru import logger

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import config as cfg

# ── Resemblyzer ───────────────────────────────────────────────────────────────
try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    _RESEMBLYZER_OK = True
    logger.info("[Voiceprint] Resemblyzer available ✓")
except ImportError:
    _RESEMBLYZER_OK = False
    logger.warning("[Voiceprint] Resemblyzer not found — using MFCC fallback")

# ── LibROSA (MFCC fallback) ───────────────────────────────────────────────────
try:
    import librosa
    _LIBROSA_OK = True
except ImportError:
    _LIBROSA_OK = False

SAMPLE_RATE = 16_000
VOICE_COLLECTION = "voice_embeddings"
DEFAULT_THRESHOLD = 0.80   # cosine similarity — higher = stricter


class VoiceprintStore:
    """
    Enroll a person's voice and identify speakers from audio chunks.

    Usage:
        store = VoiceprintStore()
        store.enroll(person_id="uuid", person_name="Alice", audio_np=arr, sr=16000)
        name, conf = store.identify(audio_np=arr, sr=16000)
    """

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self._threshold = threshold
        self._lock = threading.Lock()
        self._encoder: Optional[VoiceEncoder] = None
        self._collection = None
        self._init_encoder()
        self._init_collection()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_encoder(self):
        if _RESEMBLYZER_OK:
            try:
                self._encoder = VoiceEncoder()
                logger.info("[Voiceprint] VoiceEncoder loaded ✓")
            except Exception as e:
                logger.error(f"[Voiceprint] VoiceEncoder init failed: {e}")
                self._encoder = None

    def _init_collection(self):
        try:
            from core.storage.db import get_chroma
            client, _ = get_chroma()
            self._collection = client.get_or_create_collection(
                name=VOICE_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"[Voiceprint] ChromaDB collection '{VOICE_COLLECTION}' ready ✓")
        except Exception as e:
            logger.error(f"[Voiceprint] ChromaDB init failed: {e}")

    # ── Embedding extraction ──────────────────────────────────────────────────

    def _extract_embedding(self, audio_np: np.ndarray, sample_rate: int) -> Optional[np.ndarray]:
        """Returns a 1-D numpy embedding or None on failure."""
        # Resample to 16 kHz if needed
        audio = audio_np.astype(np.float32)
        if sample_rate != SAMPLE_RATE and _LIBROSA_OK:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)

        if self._encoder is not None and _RESEMBLYZER_OK:
            try:
                wav = preprocess_wav(audio, source_sr=SAMPLE_RATE)
                embedding = self._encoder.embed_utterance(wav)
                return embedding.astype(np.float32)
            except Exception as e:
                logger.debug(f"[Voiceprint] Resemblyzer embed failed: {e}")

        # ── MFCC fallback ─────────────────────────────────────────────────────
        if _LIBROSA_OK:
            try:
                mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=40)
                embedding = np.mean(mfcc, axis=1).astype(np.float32)
                return embedding
            except Exception as e:
                logger.debug(f"[Voiceprint] MFCC fallback failed: {e}")

        return None

    # ── Public API ────────────────────────────────────────────────────────────

    def enroll(
        self,
        person_id: str,
        person_name: str,
        audio_np: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
    ) -> bool:
        """
        Store a voiceprint for person_id.
        Returns True on success.
        """
        if self._collection is None:
            logger.warning("[Voiceprint] No ChromaDB collection — skipping enrollment")
            return False

        if len(audio_np) < sample_rate:   # need at least 1 second
            logger.warning(f"[Voiceprint] Audio too short for enrollment ({len(audio_np)/sample_rate:.1f}s)")
            return False

        embedding = self._extract_embedding(audio_np, sample_rate)
        if embedding is None:
            logger.warning(f"[Voiceprint] Could not extract embedding for {person_name}")
            return False

        with self._lock:
            try:
                self._collection.upsert(
                    ids=[person_id],
                    embeddings=[embedding.tolist()],
                    metadatas=[{"name": person_name, "person_id": person_id}],
                )
                logger.success(f"[Voiceprint] ✓ Enrolled voiceprint for '{person_name}' ({person_id})")
                return True
            except Exception as e:
                logger.error(f"[Voiceprint] Enroll DB error: {e}")
                return False

    def identify(
        self,
        audio_np: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
        threshold: Optional[float] = None,
    ) -> tuple[Optional[str], Optional[str], float]:
        """
        Identify speaker from audio.
        Returns (person_id, person_name, confidence) or (None, None, 0.0).
        Confidence = cosine similarity (0–1, higher = better match).
        """
        if self._collection is None or self._collection.count() == 0:
            return None, None, 0.0

        embedding = self._extract_embedding(audio_np, sample_rate)
        if embedding is None:
            return None, None, 0.0

        thr = threshold if threshold is not None else self._threshold

        with self._lock:
            try:
                results = self._collection.query(
                    query_embeddings=[embedding.tolist()],
                    n_results=1,
                )
                if not results["ids"] or not results["ids"][0]:
                    return None, None, 0.0

                person_id = results["ids"][0][0]
                distance  = results["distances"][0][0]
                confidence = 1.0 - float(distance)   # cosine distance → similarity
                meta = results["metadatas"][0][0] if results.get("metadatas") else {}
                name = meta.get("name", "Unknown")

                if confidence >= thr:
                    logger.debug(f"[Voiceprint] Identified '{name}' (conf={confidence:.2f})")
                    return person_id, name, confidence
                else:
                    logger.debug(f"[Voiceprint] No match (best={name}, conf={confidence:.2f} < thr={thr})")
                    return None, None, confidence

            except Exception as e:
                logger.debug(f"[Voiceprint] Identify error: {e}")
                return None, None, 0.0

    def is_enrolled(self, person_id: str) -> bool:
        """Check if a voiceprint exists for person_id."""
        if self._collection is None:
            return False
        try:
            res = self._collection.get(ids=[person_id])
            return bool(res["ids"])
        except Exception:
            return False

    @property
    def enrolled_count(self) -> int:
        if self._collection is None:
            return 0
        try:
            return self._collection.count()
        except Exception:
            return 0
