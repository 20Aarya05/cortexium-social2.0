"""
Storage layer — SQLite (via SQLAlchemy async) + ChromaDB
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Optional

import chromadb
from chromadb.config import Settings
from loguru import logger
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import config as cfg


# ── SQLAlchemy setup ──────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class Person(Base):
    __tablename__ = "persons"
    id            = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name          = Column(String(120), nullable=False, default="Unknown")
    first_seen    = Column(DateTime, default=datetime.utcnow)
    last_seen     = Column(DateTime, default=datetime.utcnow)
    introduced_by = Column(String(36), nullable=True)
    context       = Column(Text, nullable=True)
    face_count    = Column(Integer, default=0)


class InteractionEvent(Base):
    __tablename__ = "interactions"
    id            = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp     = Column(DateTime, default=datetime.utcnow)
    person_a      = Column(String(36), nullable=True)
    person_b      = Column(String(36), nullable=True)
    speaker_id    = Column(String(36), nullable=True)
    emotion       = Column(String(40), nullable=True)
    gesture       = Column(String(80), nullable=True)
    speech        = Column(Text, nullable=True)
    social_context= Column(String(120), nullable=True)
    proximity     = Column(String(40), nullable=True)
    raw_json      = Column(Text, nullable=True)


ENGINE = create_engine(
    f"sqlite:///{cfg.SQLITE_DB}",
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=ENGINE)


def init_db():
    Base.metadata.create_all(bind=ENGINE)
    logger.info(f"SQLite DB initialised at {cfg.SQLITE_DB}")


# ── ChromaDB setup ────────────────────────────────────────────────────────────

_chroma_client: Optional[chromadb.ClientAPI] = None
_face_collection = None


def get_chroma():
    global _chroma_client, _face_collection
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=str(cfg.CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        _face_collection = _chroma_client.get_or_create_collection(
            name="face_embeddings",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"ChromaDB initialised at {cfg.CHROMA_DIR}")
    return _chroma_client, _face_collection


def add_face_embedding(person_id: str, embedding: list[float], metadata: dict):
    _, col = get_chroma()
    col.upsert(
        ids=[person_id],
        embeddings=[embedding],
        metadatas=[metadata],
    )


def search_face(embedding: list[float], top_k: int = 1):
    """Returns list of (person_id, distance) sorted by distance ascending."""
    _, col = get_chroma()
    if col.count() == 0:
        return []
    results = col.query(
        query_embeddings=[embedding],
        n_results=min(top_k, col.count()),
    )
    ids       = results["ids"][0]
    distances = results["distances"][0]
    return list(zip(ids, distances))
