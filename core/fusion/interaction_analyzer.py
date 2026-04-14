"""
Multimodal Fusion — Interaction Analyzer.

Every 10 seconds, fuses:
  - face IDs + names (from vision pipeline)
  - speaker ID (from diarization)
  - emotion (from emotion detector)
  - gesture / head orientation (from body pose)
  - transcribed speech (from Whisper)

→ Creates a structured InteractionEvent dict
→ Sends to LLM for social context tagging
→ Persists to SQLite + Neo4j
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import config as cfg
from core.storage.person_registry import log_interaction
from core.storage.graph_db import ensure_person, record_meeting, record_spoke_with


class InteractionAnalyzer:
    def __init__(self, interval_seconds: float = 5.0):
        self._interval = interval_seconds
        self._last_flush = time.time()

        # Accumulators (reset each interval)
        self._faces:    list[dict] = []   # {person_id, name, emotion}
        self._speech:   list[str]  = []
        self._gestures: list[str]  = []
        self._speaker:  Optional[str] = None
        self._proximity: str = "medium"

    # ── Feed methods (called from main loop) ─────────────────────────────────

    def feed_faces(self, faces: list[dict]):
        self._faces = faces

    def feed_speech(self, text: str):
        if text:
            self._speech.append(text)

    def feed_gesture(self, gesture: str):
        if gesture and gesture != "neutral":
            self._gestures.append(gesture)

    def feed_speaker(self, speaker: Optional[str]):
        self._speaker = speaker

    def feed_proximity(self, proximity: str):
        self._proximity = proximity

    # ── Tick — called each frame ─────────────────────────────────────────────

    def tick(self) -> Optional[dict]:
        """Returns an event dict when interval has elapsed, else None."""
        now = time.time()
        if now - self._last_flush < self._interval:
            return None
        self._last_flush = now
        return self._flush()

    def _flush(self) -> Optional[dict]:
        if not self._faces:
            self._reset()
            return None

        persons   = [f.get("name", "Unknown") for f in self._faces]
        person_ids= [f.get("person_id") for f in self._faces if f.get("person_id")]
        emotions  = [f.get("emotion", "neutral") for f in self._faces]
        dominant_emo = max(set(emotions), key=emotions.count) if emotions else "neutral"
        speech_blob  = " ".join(self._speech)
        gesture      = max(set(self._gestures), key=self._gestures.count) if self._gestures else "neutral"

        event = {
            "timestamp":     datetime.now(timezone.utc).isoformat(),
            "persons":       persons,
            "person_ids":    person_ids,
            "participants":  persons,
            "person_a":      persons[0] if len(persons) > 0 else None,
            "person_b":      persons[1] if len(persons) > 1 else None,
            "speaker":       self._speaker,
            "emotion":       dominant_emo,
            "gesture":       gesture,
            "speech":        speech_blob[:500],
            "social_context": self._classify_context(speech_blob, gesture, dominant_emo),
            "proximity":     self._proximity,
        }

        # ── Persist ──────────────────────────────────────────────────────────
        try:
            event_id = log_interaction(event)
            event["id"] = event_id
            logger.info(
                f"[Fusion] Event: {persons} | {dominant_emo} | '{speech_blob[:60]}…'"
            )
        except Exception as e:
            logger.error(f"[Fusion] Persist error: {e}")

        # ── Neo4j graph edges ─────────────────────────────────────────────────
        for pid, name in zip(
            [f.get("person_id") for f in self._faces],
            [f.get("name", "Unknown") for f in self._faces],
        ):
            if pid:
                ensure_person(pid, name)

        if len(person_ids) >= 2:
            record_meeting(person_ids[0], person_ids[1], event["social_context"])

        if self._speaker and speech_blob and len(person_ids) >= 2:
            listener = next((p for p in person_ids if p != self._speaker), None)
            if listener:
                record_spoke_with(self._speaker, listener, speech_blob)

        # ── LLM enrichment (async / fire-and-forget) ─────────────────────────
        self._enrich_via_llm(event)

        self._reset()
        return event

    def _classify_context(self, speech: str, gesture: str, emotion: str) -> str:
        """Rule-based quick classification before LLM enrichment."""
        speech_l = speech.lower()
        if any(w in speech_l for w in ["nice to meet", "hello", "hi", "hey"]):
            return "introduction"
        if any(w in speech_l for w in ["bye", "goodbye", "see you", "later"]):
            return "farewell"
        if "?" in speech:
            return "question"
        if gesture == "waving":
            return "greeting"
        if gesture == "crossed_arms":
            return "defensive_posture"
        if emotion == "happy":
            return "positive_exchange"
        if emotion in ("angry", "disgust"):
            return "conflict"
        return "conversation"

    def _enrich_via_llm(self, event: dict):
        """Fire-and-forget: ask Ollama to add a one-line AI insight."""
        import threading

        def _run():
            try:
                import ollama
                prompt = (
                    f"Persons: {event['persons']}\n"
                    f"Emotion: {event['emotion']}\n"
                    f"Gesture: {event['gesture']}\n"
                    f"Speech: \"{event['speech'][:300]}\"\n\n"
                    "In one sentence, summarize the social interaction happening."
                )
                resp = ollama.chat(
                    model=cfg.OLLAMA_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    options={"num_predict": 80},
                )
                insight = resp["message"]["content"].strip()
                event["ai_insight"] = insight
                logger.debug(f"[LLM] Insight: {insight}")
            except Exception as e:
                logger.debug(f"[LLM] Enrichment skipped: {e}")

        threading.Thread(target=_run, daemon=True).start()

    def _reset(self):
        self._speech   = []
        self._gestures = []
        self._speaker  = None
