"""
Test: Knowledge extractor — seeds dummy events and runs full export.
Run: python scripts/test_knowledge_export.py
"""

import sys
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.storage.db import init_db, InteractionEvent, SessionLocal
from core.knowledge.behavior_extractor import BehaviorExtractor
from loguru import logger


SEED_EVENTS = [
    {"social_context":"introduction","emotion":"happy","gesture":"waving",
     "speech":"Nice to meet you!","persons":["Alice","Bob"],"proximity":"close"},
    {"social_context":"introduction","emotion":"happy","gesture":"neutral",
     "speech":"Hello, I am Bob.","persons":["Bob","Charlie"],"proximity":"medium"},
    {"social_context":"conversation","emotion":"neutral","gesture":"neutral",
     "speech":"What do you think about the project?","persons":["Alice"],"proximity":"medium"},
    {"social_context":"conflict","emotion":"angry","gesture":"crossed_arms",
     "speech":"I disagree with this approach.","persons":["Dave"],"proximity":"far"},
    {"social_context":"farewell","emotion":"happy","gesture":"waving",
     "speech":"Goodbye, see you tomorrow!","persons":["Alice","Bob"],"proximity":"medium"},
]


def seed_events():
    with SessionLocal() as session:
        for ev in SEED_EVENTS:
            rec = InteractionEvent(
                id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc),
                emotion=ev["emotion"],
                gesture=ev["gesture"],
                speech=ev["speech"],
                social_context=ev["social_context"],
                proximity=ev["proximity"],
                raw_json=json.dumps(ev),
            )
            session.add(rec)
        session.commit()
    logger.info(f"Seeded {len(SEED_EVENTS)} dummy events")


def main():
    init_db()
    seed_events()
    extractor = BehaviorExtractor()
    result = extractor.run_full_export()
    logger.success(f"Export complete: {result}")


if __name__ == "__main__":
    main()
