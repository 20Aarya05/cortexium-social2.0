"""
Cleanup Script — Deletes all person records named 'You' (case-insensitive)
to fix noise-triggered duplicate enrollments.
"""

import sys
import pathlib

# Add root to sys.path
root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from core.storage.person_registry import get_all_persons, delete_person_data
from loguru import logger

def cleanup_you():
    logger.info("Starting cleanup of 'You' duplicates...")
    
    persons = get_all_persons()
    target_name = "You"
    
    count = 0
    for p in persons:
        if p.name.strip().lower() == target_name.lower():
            logger.info(f"Deleting duplicate: {p.name} (Sightings: {p.face_count}, ID: {p.id})")
            try:
                delete_person_data(p.id)
                count += 1
            except Exception as e:
                logger.error(f"Failed to delete {p.id}: {e}")
                
    logger.success(f"Cleanup complete! Removed {count} duplicate 'You' records.")

if __name__ == "__main__":
    cleanup_you()
