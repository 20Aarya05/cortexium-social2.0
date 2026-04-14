import cv2
from loguru import logger

def list_available_cameras(max_indices: int = 5) -> list[int]:
    """
    Scans for available video capture devices.
    Returns a list of integer indices that successfully open.
    """
    available = []
    for i in range(max_indices):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        except Exception as e:
            logger.debug(f"[Hardware] Camera index {i} failed: {e}")
    
    logger.info(f"[Hardware] Found {len(available)} available cameras: {available}")
    return available

def update_env_setting(key: str, value: str):
    """
    Safely updates or adds a key-value pair in the .env file.
    """
    import os
    from pathlib import Path
    
    env_path = Path(".env")
    if not env_path.exists():
        logger.error("[Hardware] .env file not found")
        return False
        
    lines = env_path.read_text().splitlines()
    updated = False
    
    new_lines = []
    for line in lines:
        if line.startswith(f"{key}="):
            new_lines.append(f"{key}={value}")
            updated = True
        else:
            new_lines.append(line)
            
    if not updated:
        # Append to the end if not found
        new_lines.append(f"{key}={value}")
        
    env_path.write_text("\n".join(new_lines) + "\n")
    logger.info(f"[Hardware] Updated .env: {key}={value}")
    return True
