"""
Test: Face recognition pipeline with a synthetic face embedding.
Run: python scripts/test_recognition.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from loguru import logger
from config import FACE_CONFIDENCE_THRESHOLD
from core.storage.db import init_db
from core.storage.person_registry import enroll_person, identify_person, get_all_persons


def random_embedding(dim=512) -> list[float]:
    v = np.random.randn(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


def test_enroll_and_identify():
    init_db()

    # Enroll two synthetic people
    emb_alice = random_embedding()
    emb_bob   = random_embedding()

    alice = enroll_person(emb_alice, name="Alice_Test", context="unit-test")
    bob   = enroll_person(emb_bob,   name="Bob_Test",   context="unit-test")
    logger.info(f"Enrolled: {alice.name} ({alice.id[:8]}…), {bob.name} ({bob.id[:8]}…)")

    # Identify with same embedding
    person, conf = identify_person(emb_alice)
    assert person is not None, "Alice not found!"
    assert person.name == "Alice_Test", f"Wrong match: {person.name}"
    logger.success(f"✓ Alice identified with confidence {conf:.3f}")

    # Identify with slightly noisy embedding
    noise = np.array(emb_alice) + np.random.randn(512) * 0.05
    noise /= np.linalg.norm(noise)
    person2, conf2 = identify_person(noise.tolist())
    if person2 and person2.name == "Alice_Test":
        logger.success(f"✓ Alice identified (noisy) with confidence {conf2:.3f}")
    else:
        logger.warning(f"⚠ Noisy embedding matched {person2.name if person2 else 'nobody'} (conf {conf2:.3f})")

    # Unknown embedding should not match
    unknown = random_embedding()
    person3, conf3 = identify_person(unknown)
    if person3 is None:
        logger.success(f"✓ Unknown embedding correctly returned None (conf {conf3:.3f})")
    else:
        logger.warning(f"⚠ Unknown matched {person3.name} (conf {conf3:.3f}) — check threshold {FACE_CONFIDENCE_THRESHOLD}")

    persons = get_all_persons()
    logger.info(f"Total enrolled: {len(persons)}")
    logger.success("✅ Recognition test complete")


if __name__ == "__main__":
    test_enroll_and_identify()
