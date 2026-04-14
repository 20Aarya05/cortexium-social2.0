import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from core.storage.db import SessionLocal, Person, get_chroma
from loguru import logger

def cleanup_you():
    session = SessionLocal()
    client, col = get_chroma()

    # 1. Find all people named 'You'
    # Using case-insensitive match just in case
    targets = session.query(Person).filter(Person.name.ilike('you')).all()
    
    if not targets:
        logger.info("[Cleanup] No people named 'You' found.")
        return

    target_ids = [p.id for p in targets]
    target_names = [p.name for p in targets]
    
    logger.info(f"[Cleanup] Identified {len(target_ids)} records to remove: {target_names}")

    # 2. Remove from ChromaDB
    try:
        col.delete(ids=target_ids)
        logger.info(f"[Cleanup] Removed {len(target_ids)} embeddings from ChromaDB.")
    except Exception as e:
        logger.error(f"[Cleanup] ChromaDB deletion failed: {e}")

    # 3. Remove from SQLite
    try:
        session.query(Person).filter(Person.id.in_(target_ids)).delete(synchronize_session=False)
        session.commit()
        logger.info(f"[Cleanup] Removed {len(target_ids)} person records from SQLite.")
    except Exception as e:
        session.rollback()
        logger.error(f"[Cleanup] SQLite deletion failed: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    cleanup_you()
