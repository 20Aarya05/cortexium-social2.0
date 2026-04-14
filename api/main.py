"""
FastAPI Service — Internal bus for Cortexium.

REST endpoints:
  POST /enroll              — enroll new face (multipart: name + embedding JSON)
  GET  /persons             — list all known persons
  GET  /identify            — identify face by embedding
  GET  /interactions        — recent interaction events
  GET  /knowledge/patterns  — current behavior patterns
  GET  /knowledge/export    — trigger full robot export
  POST /hud/insight         — push AI insight to HUD
  WS   /stream              — WebSocket live event stream

Run with: uvicorn api.main:app --host 0.0.0.0 --port 8765 --reload
"""

from __future__ import annotations

import json
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config as cfg
from core.storage.db import init_db
from core.storage.person_registry import (
    enroll_person,
    identify_person,
    get_all_persons,
    get_recent_interactions,
)
from core.knowledge.behavior_extractor import BehaviorExtractor
from core.storage.person_registry import delete_person_data


app = FastAPI(
    title="Cortexium Social Intelligence API",
    version="1.0.0",
    description="AI agent for smart glasses — face recognition, social graph, robot knowledge",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── WebSocket manager ─────────────────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active.discard(ws) if hasattr(self.active, "discard") else None
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: dict):
        payload = json.dumps(data)
        for ws in list(self.active):
            try:
                await ws.send_text(payload)
            except Exception:
                self.disconnect(ws)


manager = ConnectionManager()


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    init_db()
    logger.info(f"[API] Cortexium API running on {cfg.API_HOST}:{cfg.API_PORT}")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/enroll")
async def enroll(
    name:      str        = Form(...),
    embedding: str        = Form(...),   # JSON array of floats
    context:   Optional[str] = Form(None),
):
    emb = json.loads(embedding)
    person = enroll_person(embedding=emb, name=name, context=context)
    await manager.broadcast({"event": "enrolled", "person_id": person.id, "name": person.name})
    return {"status": "ok", "person_id": person.id, "name": person.name}


@app.post("/broadcast")
async def broadcast_event(data: dict):
    """Generic endpoint to push events to all connected WebSocket clients."""
    await manager.broadcast(data)
    return {"status": "ok"}


@app.get("/persons")
def list_persons():
    persons = get_all_persons()
    return [
        {
            "id":         p.id,
            "name":       p.name,
            "first_seen": p.first_seen.isoformat() if p.first_seen else None,
            "last_seen":  p.last_seen.isoformat()  if p.last_seen  else None,
            "face_count": p.face_count,
            "context":    p.context,
        }
        for p in persons
    ]


@app.delete("/persons/{person_id}")
async def delete_person(person_id: str):
    logger.info(f"[API] Received deletion request for person: {person_id}")
    try:
        delete_person_data(person_id)
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"[API] Deletion failed for {person_id}: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/identify")
async def identify(embedding: str = Form(...)):
    emb    = json.loads(embedding)
    person, conf = identify_person(emb)
    if person:
        return {"matched": True, "person_id": person.id, "name": person.name, "confidence": conf}
    return {"matched": False, "confidence": conf}


@app.get("/interactions")
def recent_interactions(limit: int = 50):
    events = get_recent_interactions(limit)
    return [
        {
            "id":            e.id,
            "timestamp":     e.timestamp.isoformat() if e.timestamp else None,
            "person_a":      e.person_a,
            "person_b":      e.person_b,
            "emotion":       e.emotion,
            "gesture":       e.gesture,
            "speech":        e.speech,
            "social_context":e.social_context,
        }
        for e in events
    ]


@app.get("/knowledge/patterns")
def get_patterns():
    extractor = BehaviorExtractor()
    return extractor.extract_patterns()


@app.get("/knowledge/export")
def trigger_export():
    extractor = BehaviorExtractor()
    paths = extractor.run_full_export()
    return {"status": "exported", "files": paths}


@app.get("/settings/cameras")
def get_available_cameras():
    from core.utils.hardware import list_available_cameras
    return {"cameras": list_available_cameras()}


@app.post("/settings/camera")
async def set_camera(index: int = Form(...)):
    from core.utils.hardware import update_env_setting
    success = update_env_setting("CAMERA_SOURCE", str(index))
    if success:
        return {"status": "ok", "message": "Camera updated. Please restart the agent."}
    return JSONResponse(status_code=500, content={"error": "Failed to update .env"})


@app.post("/hud/insight")
async def push_insight(insight: str = Form(...)):
    await manager.broadcast({"event": "insight", "text": insight})
    return {"status": "ok"}


@app.websocket("/stream")
async def ws_stream(websocket: WebSocket):
    await manager.connect(websocket)
    logger.info("[API] WebSocket client connected")
    try:
        while True:
            await websocket.receive_text()   # keep-alive ping
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("[API] WebSocket client disconnected")


# ── Broadcast helper (called from main loop) ──────────────────────────────────

async def broadcast_event(event: dict):
    await manager.broadcast({"event": "interaction", **event})
