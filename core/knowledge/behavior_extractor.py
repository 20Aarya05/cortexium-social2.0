"""
Robot Behavioral Knowledge Extractor.

Reads interaction events from SQLite → clusters → generates behavior patterns →
exports to YAML + JSON-LD + CSV.
"""

from __future__ import annotations

import csv
import json
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger
from rdflib import RDF, XSD, Graph, Literal, Namespace, URIRef
from sqlalchemy.orm import Session

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import config as cfg
from core.storage.db import InteractionEvent, SessionLocal


CORTEX = Namespace("https://cortexium.ai/ontology/")


class BehaviorExtractor:
    def __init__(self):
        cfg.EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load interactions ──────────────────────────────────────────────────────

    def _load_events(self) -> list[dict]:
        with SessionLocal() as session:
            rows = session.query(InteractionEvent).all()
            events = []
            for r in rows:
                try:
                    raw = json.loads(r.raw_json) if r.raw_json else {}
                except Exception:
                    raw = {}
                raw.update({
                    "id":            r.id,
                    "timestamp":     r.timestamp.isoformat() if r.timestamp else "",
                    "emotion":       r.emotion or "neutral",
                    "gesture":       r.gesture or "neutral",
                    "speech":        r.speech or "",
                    "social_context":r.social_context or "conversation",
                    "proximity":     r.proximity or "medium",
                })
                events.append(raw)
        return events

    # ── Pattern clustering ───────────────────────────────────────────────────

    def extract_patterns(self) -> list[dict]:
        events = self._load_events()
        if not events:
            logger.warning("[Knowledge] No events to extract patterns from")
            return []

        # Group by (context, emotion, gesture)
        clusters: dict[tuple, list] = defaultdict(list)
        for ev in events:
            key = (ev["social_context"], ev["emotion"], ev["gesture"])
            clusters[key].append(ev)

        patterns = []
        for (context, emotion, gesture), evs in clusters.items():
            speeches = [e["speech"] for e in evs if e.get("speech")]
            observed = list({p for e in evs for p in e.get("persons", []) if p != "Unknown"})

            pattern = {
                "id":          f"{context}_{emotion}_{gesture}_{uuid.uuid4().hex[:6]}",
                "trigger":     self._infer_trigger(context, gesture),
                "context":     context,
                "actions":     self._infer_actions(emotion, gesture, speeches),
                "outcome":     self._infer_outcome(emotion, context),
                "frequency":   len(evs),
                "confidence":  round(min(0.99, 0.5 + len(evs) * 0.05), 2),
                "observed_from": observed[:10],
                "sample_speech": speeches[0][:200] if speeches else "",
            }
            patterns.append(pattern)
            logger.debug(f"[Knowledge] Pattern: {context}/{emotion} × {len(evs)} obs")

        return patterns

    # ── Export methods ────────────────────────────────────────────────────────

    def export_yaml(self, patterns: list[dict]) -> Path:
        out = cfg.EXPORTS_DIR / "behavior_patterns.yaml"
        with open(out, "w", encoding="utf-8") as f:
            yaml.dump({"behavior_patterns": patterns}, f, allow_unicode=True, sort_keys=False)
        logger.success(f"[Knowledge] YAML exported → {out}")
        return out

    def export_jsonld(self, patterns: list[dict]) -> Path:
        g = Graph()
        g.bind("cortex", CORTEX)

        for p in patterns:
            subj = CORTEX[p["id"]]
            g.add((subj, RDF.type,          CORTEX.BehaviorPattern))
            g.add((subj, CORTEX.trigger,    Literal(p["trigger"])))
            g.add((subj, CORTEX.context,    Literal(p["context"])))
            g.add((subj, CORTEX.outcome,    Literal(p["outcome"])))
            g.add((subj, CORTEX.frequency,  Literal(p["frequency"],  datatype=XSD.integer)))
            g.add((subj, CORTEX.confidence, Literal(p["confidence"], datatype=XSD.float)))
            for obs in p.get("observed_from", []):
                g.add((subj, CORTEX.observedFrom, Literal(obs)))

        out = cfg.EXPORTS_DIR / "knowledge_graph.jsonld"
        g.serialize(destination=str(out), format="json-ld", indent=2)
        logger.success(f"[Knowledge] JSON-LD exported → {out}")
        return out

    def export_csv(self, events: Optional[list[dict]] = None) -> Path:
        if events is None:
            events = self._load_events()
        out = cfg.EXPORTS_DIR / "interaction_dataset.csv"
        fields = ["timestamp", "persons", "emotion", "gesture", "speech",
                  "social_context", "proximity"]
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for ev in events:
                row = {k: ev.get(k, "") for k in fields}
                if isinstance(row["persons"], list):
                    row["persons"] = ", ".join(row["persons"])
                writer.writerow(row)
        logger.success(f"[Knowledge] CSV exported → {out}")
        return out

    def run_full_export(self):
        logger.info("[Knowledge] Running full export …")
        patterns = self.extract_patterns()
        events   = self._load_events()
        yaml_p   = self.export_yaml(patterns)
        jsonld_p = self.export_jsonld(patterns)
        csv_p    = self.export_csv(events)
        return {"yaml": str(yaml_p), "jsonld": str(jsonld_p), "csv": str(csv_p)}

    # ── Classification helpers ────────────────────────────────────────────────

    def _infer_trigger(self, context: str, gesture: str) -> str:
        mapping = {
            "introduction": "new_person_encountered",
            "greeting":     "familiar_person_encountered",
            "farewell":     "departure",
            "conflict":     "disagreement_detected",
            "question":     "query_directed",
            "conversation": "ongoing_interaction",
        }
        return mapping.get(context, "interaction_detected")

    def _infer_actions(self, emotion: str, gesture: str, speeches: list[str]) -> list[dict]:
        actions = []
        if emotion == "happy":
            actions.append({"smile": True})
        if gesture == "waving":
            actions.append({"wave": True})
        if gesture == "pointing":
            actions.append({"point": True})
        if speeches:
            actions.append({"say": speeches[0][:120]})
        return actions or [{"observe": True}]

    def _infer_outcome(self, emotion: str, context: str) -> str:
        if emotion in ("happy", "surprised") or context in ("introduction", "greeting"):
            return "positive_reception"
        if emotion in ("angry", "disgust") or context == "conflict":
            return "negative_reception"
        return "neutral_reception"
