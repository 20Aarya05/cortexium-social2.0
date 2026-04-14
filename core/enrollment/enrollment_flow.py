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


# ── Name extraction patterns ──────────────────────────────────────────────────

_PATTERNS = [
    r"(?:this is|meet|his name is|her name is|their name is|call (?:him|her|them))\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
    r"(?:introduce you to|introducing)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
    r"^([A-Z][a-z]+)$",   # bare name fallback
]


def extract_name_from_text(text: str) -> Optional[str]:
    """Return extracted proper name or None."""
    for pattern in _PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            name = m.group(1).strip().title()
            if len(name) >= 2:
                return name
    return None


async def extract_name_via_llm(text: str) -> Optional[str]:
    """
    Fallback: ask local LLM to extract the person's name.
    Returns None if Ollama is unavailable.
    """
    try:
        import ollama
        response = ollama.chat(
            model="llama3.1:8b",
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

    def trigger(self, track_id: int, embedding: list[float]):
        """Mark a track as waiting for voice enrollment."""
        if track_id not in self._pending:
            logger.info(f"[Enroll] Waiting for voice to identify track #{track_id}")
            self._pending[track_id] = {
                "embedding": embedding,
                "triggered_at": time.time(),
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
