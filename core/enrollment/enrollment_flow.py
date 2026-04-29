"""
Voice-First Enrollment Flow.

Listens for voice commands like:
  "This is Alice"
  "Meet Bob"
  "His name is Charlie"

Uses Whisper to transcribe → regex/LLM NER to extract name →
then enrolls the person in ChromaDB + SQLite + Neo4j.
"""

from __future__ import annotations

import re
import time
from typing import Optional

from loguru import logger

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from core.storage.person_registry import enroll_person, update_person_name
from core.storage.graph_db import ensure_person, record_meeting


# Lazy import to avoid circular dependency
_voiceprint_store = None

def _get_voiceprint_store():
    global _voiceprint_store
    if _voiceprint_store is None:
        try:
            from core.audio.voiceprint import VoiceprintStore
            _voiceprint_store = VoiceprintStore()
        except Exception as e:
            from loguru import logger
            logger.warning(f"[Enroll] VoiceprintStore unavailable: {e}")
    return _voiceprint_store


# ── Name extraction patterns ──────────────────────────────────────────────────

_PATTERNS = [
    r"(?:this is|meet|his name is|her name is|their name is|call (?:him|her|them))\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
    r"(?:introduce you to|introducing)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
    # Bare names are risky (noise triggers them), so we only allow them if specifically structured
]

_NAME_BLACKLIST = {
    "You", "Me", "Him", "Her", "Them", "They", "Us", "Everyone", "Someone",
    "Hello", "Hi", "Hey", "Yes", "No", "Thanks", "Thank", "Okay", "Oh", "And",
    "But", "The", "This", "That", "It", "A", "An", "Please", "Sure",
}

def extract_name_from_text(text: str) -> Optional[str]:
    """Return extracted proper name or None."""
    # 1. Try Regex Patterns
    for pattern in _PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            name = m.group(1).strip().title()
            if len(name) >= 2 and name not in _NAME_BLACKLIST:
                return name
    
    # 2. Heuristic for short, single-word transcripts (only if highly confident)
    # But for now, we'll rely on the specific "This is [Name]" patterns to avoid noise-triggering.
    return None


async def extract_name_via_llm(text: str) -> Optional[str]:
    """
    Fallback: ask local LLM to extract the person's name.
    Returns None if Ollama is unavailable.
    """
    try:
        import ollama
        response = ollama.chat(
            model="llama3.2:3b",
            messages=[{
                "role": "user",
                "content": (
                    f"Extract only the person's name from this sentence. "
                    f"Reply with ONLY the name, nothing else: \"{text}\""
                ),
            }],
        )
        name = response["message"]["content"].strip().title()
        if 2 <= len(name) <= 40 and " " not in name or name.count(" ") <= 1:
            return name
    except Exception as e:
        logger.debug(f"[Enroll] LLM name extraction failed: {e}")
    return None


class EnrollmentFlow:
    """
    Manages the voice enrollment lifecycle for unknown persons.

    Usage:
        flow = EnrollmentFlow()
        # When a new face is detected with no match:
        flow.trigger(track_id=42, embedding=[...])
        # Each frame, feed latest transcript:
        enrolled = flow.feed_transcript("This is Alice", track_id=42)
        if enrolled:
            print(enrolled.name)
    """

    def __init__(self):
        self._pending: dict[int, dict] = {}   # track_id → {embedding, triggered_at}
        self._timeout = 15.0   # seconds to wait for voice response

    def trigger(self, track_id: int, embedding: list[float], audio_chunk=None):
        """Mark a track as waiting for voice enrollment.
        
        audio_chunk: optional np.ndarray — raw audio captured at trigger time,
                     stored for voiceprint enrollment once name is resolved.
        """
        if track_id not in self._pending:
            logger.info(f"[Enroll] Waiting for voice to identify track #{track_id}")
            self._pending[track_id] = {
                "embedding":    embedding,
                "triggered_at": time.time(),
                "audio_chunk":  audio_chunk,   # may be None — that's fine
            }

    def feed_transcript(self, text: str, track_id: Optional[int] = None):
        """
        Try to extract a name from the transcript and enroll the pending face.
        Returns the enrolled Person object, or None.
        """
        if not self._pending or not text:
            return None

        name = extract_name_from_text(text)
        if not name:
            return None

        # Match to oldest pending track if no track_id specified
        if track_id is None or track_id not in self._pending:
            track_id = min(self._pending, key=lambda t: self._pending[t]["triggered_at"])

        pending = self._pending.pop(track_id, None)
        if pending is None:
            return None

        embedding = pending["embedding"]
        person = enroll_person(embedding=embedding, name=name)
        ensure_person(person.id, name)
        logger.success(f"[Enroll] ✓ Enrolled '{name}' for track #{track_id}")

        # ── Voiceprint enrollment ─────────────────────────────────────────────
        audio_chunk = pending.get("audio_chunk")
        if audio_chunk is not None:
            vp = _get_voiceprint_store()
            if vp:
                import threading
                threading.Thread(
                    target=vp.enroll,
                    args=(person.id, name, audio_chunk),
                    daemon=True,
                ).start()

        return person

    def is_pending(self, track_id: int) -> bool:
        return track_id in self._pending

    def get_pending_ids(self) -> list[int]:
        # Expire old pending entries
        now = time.time()
        expired = [
            tid for tid, v in self._pending.items()
            if now - v["triggered_at"] > self._timeout
        ]
        for tid in expired:
            logger.debug(f"[Enroll] Track #{tid} enrollment timed out")
            del self._pending[tid]
        return list(self._pending.keys())
