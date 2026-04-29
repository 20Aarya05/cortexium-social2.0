"""
Factory Reset — Deep Clean of all Cortexium data.
Wipes SQLite tables (persons, interactions, conversations) and ChromaDB collections.
"""

import sys
import pathlib

# Add root to sys.path
root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from core.storage.db import ENGINE, Base, get_chroma
from sqlalchemy import text
from loguru import logger

def factory_reset():
    logger.warning("!!! STARTING FACTORY RESET !!!")
    
    # 1. Wipe SQLite
    logger.info("Wiping SQLite tables...")
    try:
        with ENGINE.connect() as conn:
            # Table names to truncate
            tables = ["persons", "interactions", "conversations"]
            for table in tables:
                logger.info(f"  Truncating table: {table}")
                conn.execute(text(f"DELETE FROM {table}"))
            conn.commit()
        logger.success("SQLite wiped ✓")
    except Exception as e:
        logger.error(f"SQLite wipe failed: {e}")

    # 2. Wipe ChromaDB
    logger.info("Wiping ChromaDB collections...")
    try:
        client, _ = get_chroma()
        collections = client.list_collections()
        for col in collections:
            logger.info(f"  Deleting collection: {col.name}")
            client.delete_collection(col.name)
        logger.success("ChromaDB wiped ✓")
    except Exception as e:
        logger.error(f"ChromaDB wipe failed: {e}")

    logger.success("FACTORY RESET COMPLETE. SYSTEM IS CLEAN.")

if __name__ == "__main__":
    confirm = input("Are you sure you want to delete EVERYTHING? (y/n): ")
    if confirm.lower() == 'y':
        factory_reset()
    else:
        logger.info("Reset cancelled.")
