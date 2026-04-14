"""
Neo4j Social Knowledge Graph driver.
Nodes: Person, Interaction, Location, Topic
Edges: MET, KNOWS, SPOKE_WITH, INTRODUCED_BY
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from loguru import logger
from neo4j import GraphDatabase, Driver

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import config as cfg

_driver: Optional[Driver] = None


def get_driver() -> Driver:
    global _driver
    if _driver is None:
        try:
            _driver = GraphDatabase.driver(
                cfg.NEO4J_URI,
                auth=(cfg.NEO4J_USER, cfg.NEO4J_PASSWORD),
            )
            _driver.verify_connectivity()
            logger.info(f"[Graph] Connected to Neo4j at {cfg.NEO4J_URI}")
        except Exception as e:
            logger.warning(f"[Graph] Neo4j unavailable: {e} — graph features disabled")
            _driver = None
    return _driver


def ensure_person(person_id: str, name: str):
    driver = get_driver()
    if not driver:
        return
    with driver.session() as s:
        s.run(
            """
            MERGE (p:Person {id: $id})
            SET p.name = $name, p.updated = $ts
            """,
            id=person_id,
            name=name,
            ts=datetime.utcnow().isoformat(),
        )


def record_meeting(person_a_id: str, person_b_id: str, ctx: str = ""):
    driver = get_driver()
    if not driver:
        return
    with driver.session() as s:
        s.run(
            """
            MATCH (a:Person {id: $aid}), (b:Person {id: $bid})
            MERGE (a)-[r:MET]->(b)
            SET r.last = $ts, r.context = $ctx
            ON CREATE SET r.first = $ts, r.count = 1
            ON MATCH SET r.count = r.count + 1
            """,
            aid=person_a_id,
            bid=person_b_id,
            ts=datetime.utcnow().isoformat(),
            ctx=ctx,
        )


def record_spoke_with(speaker_id: str, listener_id: str, speech: str):
    driver = get_driver()
    if not driver:
        return
    with driver.session() as s:
        s.run(
            """
            MATCH (a:Person {id: $sid}), (b:Person {id: $lid})
            MERGE (a)-[r:SPOKE_WITH]->(b)
            SET r.last_speech = $speech, r.last = $ts
            ON CREATE SET r.count = 1
            ON MATCH SET r.count = r.count + 1
            """,
            sid=speaker_id,
            lid=listener_id,
            speech=speech[:200],
            ts=datetime.utcnow().isoformat(),
        )


def get_relationship(person_a_id: str, person_b_id: str) -> dict:
    driver = get_driver()
    if not driver:
        return {}
    with driver.session() as s:
        result = s.run(
            """
            MATCH (a:Person {id: $aid})-[r]-(b:Person {id: $bid})
            RETURN type(r) AS rel, r.count AS count, r.first AS first, r.last AS last
            ORDER BY r.count DESC LIMIT 5
            """,
            aid=person_a_id,
            bid=person_b_id,
        )
        rows = [dict(record) for record in result]
        return {"relationships": rows}


def close():
    global _driver
    if _driver:
        _driver.close()
        _driver = None
