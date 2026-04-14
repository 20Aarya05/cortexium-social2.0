"""
Person Registry — enroll, identify, and update face profiles.
Combines ChromaDB (vector search) + SQLite (metadata).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

import numpy as np
from loguru import logger

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import config as cfg
from core.storage.db import (
    Person,
    InteractionEvent,
    SessionLocal,
    add_face_embedding,
    search_face,
)


# ── Public API ────────────────────────────────────────────────────────────────

def enroll_person(
    embedding: list[float],
    name: str = "Unknown",
    introduced_by: Optional[str] = None,
    context: Optional[str] = None,
) -> Person:
    """Create a new person record and store their face embedding."""
    person_id = str(uuid.uuid4())
    now = datetime.utcnow()

    with SessionLocal() as session:
        person = Person(
            id=person_id,
            name=name,
            first_seen=now,
            last_seen=now,
            introduced_by=introduced_by,
            context=context,
            face_count=1,
        )
        session.add(person)
        session.commit()
        session.refresh(person)

    add_face_embedding(
        person_id=person_id,
        embedding=embedding,
        metadata={"name": name, "enrolled_at": now.isoformat()},
    )
    logger.info(f"[Registry] Enrolled new person: {name} ({person_id})")
    return person


def identify_person(
    embedding: list[float],
) -> tuple[Optional[Person], float]:
    """
    Returns (Person, confidence) or (None, 0.0) if no match above threshold.
    Confidence = 1 - cosine_distance (higher is better).
    """
    results = search_face(embedding, top_k=1)
    if not results:
        return None, 0.0

    person_id, distance = results[0]
    confidence = 1.0 - float(distance)

    if confidence < cfg.FACE_CONFIDENCE_THRESHOLD:
        return None, confidence

    with SessionLocal() as session:
        person = session.get(Person, person_id)
        if person:
            person.last_seen = datetime.utcnow()
            person.face_count += 1
            session.commit()
            session.refresh(person)
        return person, confidence


def get_person(person_id: str) -> Optional[Person]:
    with SessionLocal() as session:
        return session.get(Person, person_id)


def update_person_name(person_id: str, name: str):
    with SessionLocal() as session:
        person = session.get(Person, person_id)
        if person:
            person.name = name
            session.commit()
    # Also update chroma metadata
    from core.storage.db import get_chroma
    _, col = get_chroma()
    col.update(ids=[person_id], metadatas=[{"name": name}])
    logger.info(f"[Registry] Updated name for {person_id} → {name}")


def delete_person_data(person_id: str):
    """Permanently wipe a person from SQLite and ChromaDB embeddings."""
    from core.storage.db import get_chroma
    
    # 1. Remove from SQLite metadata
    with SessionLocal() as session:
        person = session.get(Person, person_id)
        if person:
            logger.info(f"[Registry] Deleting person record: {person.name} ({person_id})")
            session.delete(person)
            session.commit()
        else:
            logger.warning(f"[Registry] Person {person_id} not found in SQLite for deletion")

    # 2. Remove from ChromaDB vector store
    try:
        client, col = get_chroma()
        col.delete(ids=[person_id])
        logger.info(f"[Registry] Removed embeddings for {person_id} from ChromaDB")
    except Exception as e:
        logger.error(f"[Registry] Failed to delete ChromaDB embeddings for {person_id}: {e}")


def log_interaction(event: dict) -> str:
    """Persist an interaction event dict to SQLite. Returns event id."""
    import json

    event_id = str(uuid.uuid4())
    with SessionLocal() as session:
        rec = InteractionEvent(
            id=event_id,
            timestamp=datetime.fromisoformat(event.get("timestamp", datetime.utcnow().isoformat())),
            person_a=event.get("person_a"),
            person_b=event.get("person_b"),
            speaker_id=event.get("speaker"),
            emotion=event.get("emotion"),
            gesture=event.get("gesture"),
            speech=event.get("speech"),
            social_context=event.get("social_context"),
            proximity=event.get("proximity"),
            raw_json=json.dumps(event),
        )
        session.add(rec)
        session.commit()
    return event_id


def get_all_persons() -> list[Person]:
    with SessionLocal() as session:
        return session.query(Person).order_by(Person.last_seen.desc()).all()


def get_recent_interactions(limit: int = 50) -> list[InteractionEvent]:
    with SessionLocal() as session:
        return (
            session.query(InteractionEvent)
            .order_by(InteractionEvent.timestamp.desc())
            .limit(limit)
            .all()
        )
