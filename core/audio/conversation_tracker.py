"""
Conversation Tracker + LLM Summarizer.

ConversationTracker:
  - Collects (speaker_name, text) turns as people talk.
  - Detects natural pauses and decides when to summarize.

ConversationSummarizer:
  - Sends the collected dialogue to local Ollama (llama3.2:3b).
  - Returns 2-3 bullet points: key topics, decisions, sentiment.
  - Runs fire-and-forget in a daemon thread — never blocks main loop.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Callable, Optional

from loguru import logger

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import config as cfg


# ─────────────────────────────────────────────────────────────────────────────
# Conversation Tracker
# ─────────────────────────────────────────────────────────────────────────────

class ConversationTurn:
    __slots__ = ("speaker", "text", "timestamp")

    def __init__(self, speaker: str, text: str):
        self.speaker   = speaker
        self.text      = text
        self.timestamp = datetime.now(timezone.utc)

    def __str__(self) -> str:
        return f"{self.speaker}: {self.text}"


class ConversationTracker:
    """
    Accumulates dialogue turns and detects when to summarize.

    Triggers summarization when:
      - At least MIN_TURNS turns have been collected AND
      - No new speech for PAUSE_SECONDS (natural pause), OR
      - MAX_TURNS turns have been collected regardless.
    """

    MIN_TURNS    = 4
    MAX_TURNS    = 20
    PAUSE_SECONDS = 8.0    # silence gap that signals end of exchange

    def __init__(self):
        self._turns: list[ConversationTurn] = []
        self._last_turn_time: float = 0.0
        self._lock = threading.Lock()
        # Track participants seen in this conversation window
        self._participants: set[str] = set()

    def add_turn(self, speaker: str, text: str):
        """Add a new dialogue turn. speaker is a resolved name (not SPEAKER_XX)."""
        if not text or not text.strip():
            return
        with self._lock:
            self._turns.append(ConversationTurn(speaker, text.strip()))
            self._last_turn_time = time.time()
            self._participants.add(speaker)
            logger.debug(f"[Conversation] Turn #{len(self._turns)} — {speaker}: {text[:60]}")

    def should_summarize(self) -> bool:
        """Return True when a summarization should be triggered."""
        with self._lock:
            n = len(self._turns)
            if n == 0:
                return False
            if n >= self.MAX_TURNS:
                return True
            if n >= self.MIN_TURNS:
                elapsed = time.time() - self._last_turn_time
                return elapsed >= self.PAUSE_SECONDS
        return False

    def get_transcript_text(self) -> str:
        """Return formatted dialogue string for the LLM prompt."""
        with self._lock:
            return "\n".join(str(t) for t in self._turns)

    def get_participants(self) -> list[str]:
        with self._lock:
            return sorted(self._participants)

    def flush(self) -> list[ConversationTurn]:
        """Clear buffer and return the flushed turns."""
        with self._lock:
            turns = list(self._turns)
            self._turns.clear()
            self._participants.clear()
            self._last_turn_time = 0.0
            return turns

    @property
    def turn_count(self) -> int:
        with self._lock:
            return len(self._turns)


# ─────────────────────────────────────────────────────────────────────────────
# LLM Summarizer
# ─────────────────────────────────────────────────────────────────────────────

_SUMMARIZE_PROMPT = """\
You are a social intelligence assistant. Below is a conversation transcript \
between {participants}.

TRANSCRIPT:
{transcript}

Summarize this conversation in exactly 2-3 bullet points. Focus on:
- What was discussed or decided
- The tone/sentiment of the exchange
- Any notable information exchanged

Reply ONLY with the bullet points, starting each with "•". Be concise."""


class ConversationSummarizer:
    """
    Calls local Ollama to summarize a conversation transcript.
    Invokes on_summary callback (in a daemon thread) when done.
    """

    def __init__(self, on_summary: Optional[Callable[[str, str], None]] = None):
        """
        on_summary(transcript, summary) — called when summarization completes.
        """
        self._on_summary = on_summary

    def summarize(
        self,
        transcript: str,
        participants: list[str],
        on_done: Optional[Callable[[str], None]] = None,
    ):
        """
        Fire-and-forget summarization.
        Calls on_done(summary_text) or self._on_summary(transcript, summary_text).
        """
        if not transcript.strip():
            return

        def _run():
            summary = self._call_llm(transcript, participants)
            if summary:
                if on_done:
                    on_done(summary)
                elif self._on_summary:
                    self._on_summary(transcript, summary)
                logger.info(f"[Conversation] Summary ready:\n{summary}")

        threading.Thread(target=_run, daemon=True).start()

    def _call_llm(self, transcript: str, participants: list[str]) -> Optional[str]:
        try:
            import ollama
            participant_str = " and ".join(participants) if participants else "participants"
            prompt = _SUMMARIZE_PROMPT.format(
                participants=participant_str,
                transcript=transcript[:2000],   # cap to avoid context overflow on 3B
            )
            resp = ollama.chat(
                model=cfg.OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"num_predict": 150, "temperature": 0.3},
            )
            return resp["message"]["content"].strip()
        except Exception as e:
            logger.debug(f"[Conversation] LLM summarize failed: {e}")
            return None

    def summarize_sync(self, transcript: str, participants: list[str]) -> Optional[str]:
        """Blocking version — use only in non-critical paths."""
        return self._call_llm(transcript, participants)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function to persist a conversation to SQLite
# ─────────────────────────────────────────────────────────────────────────────

def save_conversation_to_db(
    person_a: Optional[str],
    person_b: Optional[str],
    transcript: str,
    summary: str,
    started_at: Optional[datetime] = None,
    ended_at: Optional[datetime] = None,
):
    """Persist conversation + summary to SQLite Conversation table."""
    try:
        from core.storage.db import SessionLocal, Conversation
        with SessionLocal() as session:
            conv = Conversation(
                person_a      = person_a,
                person_b      = person_b,
                full_transcript = transcript,
                summary       = summary,
                started_at    = started_at or datetime.now(timezone.utc),
                ended_at      = ended_at   or datetime.now(timezone.utc),
            )
            session.add(conv)
            session.commit()
            logger.info(f"[Conversation] Saved conversation between {person_a} & {person_b}")
    except Exception as e:
        logger.error(f"[Conversation] DB save failed: {e}")
