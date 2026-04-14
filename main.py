"""
Cortexium Main Loop — Orchestrates all pipelines.

Usage:
    python main.py                          # default webcam
    python main.py --source video.mp4       # video file
    python main.py --source 0               # webcam index 0
    python main.py --no-audio               # vision only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from rich.console import Console
from rich.panel import Panel

# ── Bootstrap ─────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from core.storage.db import init_db
from core.vision.face_pipeline import FacePipeline, TrackedFace
from core.vision.body_pose import BodyPosePipeline
from core.vision.emotion_detector import EmotionDetector
from core.audio.speech_engine import SpeechEngine
from core.audio.diarization import DiarizationEngine
from core.fusion.interaction_analyzer import InteractionAnalyzer
from core.enrollment.enrollment_flow import EnrollmentFlow
from core.hud.hud_renderer import HUDRenderer
import threading
import queue
import requests
from rich.console import Console


console = Console()


def parse_args():
    p = argparse.ArgumentParser(description="Cortexium Social Intelligence Agent")
    p.add_argument("--source",   default=str(cfg.CAMERA_SOURCE), help="Camera index or video path")
    p.add_argument("--no-audio", action="store_true",            help="Disable audio / mic")
    p.add_argument("--no-llm",   action="store_true",            help="Disable Ollama LLM")
    p.add_argument("--width",    type=int, default=cfg.HUD_WIDTH)
    p.add_argument("--height",   type=int, default=cfg.HUD_HEIGHT)
    return p.parse_args()


def open_capture(source: str) -> cv2.VideoCapture:
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Cannot open camera/video: {source}")
        sys.exit(1)
    logger.info(f"[Main] Capture opened: {source}")
    return cap


class VisionWorker:
    """Runs vision pipelines in a background thread with frame-skipping."""
    def __init__(self, face_pipe: FacePipeline, pose_pipe: BodyPosePipeline, emo_detect: EmotionDetector):
        self.face_pipe = face_pipe
        self.pose_pipe = pose_pipe
        self.emo_detect = emo_detect
        
        self.input_queue = queue.Queue(maxsize=1)
        self.latest_result = ([], None) # (tracked_faces, pose_result)
        self.running = False
        self._thread = None

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("[VisionWorker] Thread started")

    def stop(self):
        self.running = False

    def put_frame(self, frame):
        """Put frame into queue, skip if busy."""
        try:
            self.input_queue.put_nowait(frame)
        except queue.Full:
            pass

    def get_latest(self) -> tuple[list[TrackedFace], any]:
        return self.latest_result

    def _run(self):
        while self.running:
            try:
                frame = self.input_queue.get(timeout=1.0)
                
                faces = self.face_pipe.process_frame(frame)
                pose  = self.pose_pipe.process_frame(frame)
                
                # Optional: emotion only if face is close
                for f in faces:
                    if f.person_id:
                        f.emotion = self.emo_detect.detect(frame, f.box)
                
                self.latest_result = (faces, pose)
                self.input_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[VisionWorker] Error: {e}")


class BroadcastWorker:
    """Handles network-bound dashboard syncing in a separate thread."""
    def __init__(self, api_port: int):
        self.api_port = api_port
        self.queue = queue.Queue(maxsize=100)
        self.running = False
        self._thread = None

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("[BroadcastWorker] Thread started")

    def stop(self):
        self.running = False

    def put_event(self, event_type: str, data: any):
        try:
            self.queue.put_nowait({"event": event_type, "data": data})
        except queue.Full:
            pass

    def _run(self):
        while self.running:
            try:
                item = self.queue.get(timeout=1.0)
                if item["event"] == "transcript":
                    requests.post(
                        f"http://localhost:{self.api_port}/broadcast",
                        json={"event": "transcript", "text": item["data"]},
                        timeout=0.2
                    )
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                pass


def main():
    args = parse_args()

    console.print(Panel.fit(
        "[bold cyan]CORTEXIUM[/bold cyan] Social Intelligence Agent\n"
        "[dim]Face recognition · Social graph · Robot knowledge[/dim]\n"
        f"[green]Source:[/green] {args.source}",
        border_style="cyan",
    ))

    # ── Init storage ──────────────────────────────────────────────────────────
    init_db()

    # ── Init pipelines ────────────────────────────────────────────────────────
    logger.info("[Main] Initialising pipelines …")
    
    face_pipeline = None
    try:
        face_pipeline  = FacePipeline()
    except Exception as e:
        logger.error(f"[Main] Failed to init FacePipeline: {e}")

    pose_pipeline = None
    try:
        pose_pipeline  = BodyPosePipeline()
    except Exception as e:
        logger.error(f"[Main] Failed to init BodyPosePipeline: {e}")

    emotion_detect = None
    try:
        emotion_detect = EmotionDetector()
    except Exception as e:
        logger.error(f"[Main] Failed to init EmotionDetector: {e}")

    speech_engine = None
    try:
        speech_engine  = SpeechEngine(model_size="small")
    except Exception as e:
        logger.error(f"[Main] Failed to init SpeechEngine: {e}")

    diarizer = None
    try:
        diarizer       = DiarizationEngine()
    except Exception as e:
        logger.error(f"[Main] Failed to init DiarizationEngine: {e}")

    analyzer       = InteractionAnalyzer(interval_seconds=10.0)
    enrollment     = EnrollmentFlow()
    hud            = HUDRenderer()

    if speech_engine and not args.no_audio:
        speech_engine.start_mic_stream()

    # ── Open video capture ────────────────────────────────────────────────────
    cap = open_capture(args.source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    fps_counter = 0
    fps_timer   = time.time()
    current_fps = 0.0

    console.print("[green]Running — press Q to quit[/green]")

    # ── Workers ──
    vision_worker = VisionWorker(face_pipeline, pose_pipeline, emotion_detect)
    vision_worker.start()
    
    broadcast_worker = BroadcastWorker(cfg.API_PORT)
    broadcast_worker.start()

    # ── Main loop ─────────────────────────────────────────────────────────────
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("[Main] Frame read failed — end of stream or camera lost")
                if not str(args.source).isdigit():
                    break
                time.sleep(0.01)
                continue

            # ── Push to async vision ──
            vision_worker.put_frame(frame)
            
            # ── Get latest processed results ──
            tracked_faces, pose_result = vision_worker.get_latest()

            # ── Audio ─────────────────────────────────────────────────────────
            transcript = None
            if not args.no_audio:
                transcript = speech_engine.get_latest_transcript()

            # ── Enrollment (Main Thread for state) ────────────────────────────
            for tf in tracked_faces:
                if tf.person_id is None and tf.embedding:
                    enrollment.trigger(tf.track_id, tf.embedding)

            pending_ids = enrollment.get_pending_ids()

            if transcript and pending_ids:
                enrolled_person = enrollment.feed_transcript(transcript)
                if enrolled_person:
                    hud.push_insight(f"✓ Enrolled: {enrolled_person.name}")


            # ── Fusion ────────────────────────────────────────────────────────
            face_dicts = [
                {
                    "person_id": tf.person_id,
                    "name":      tf.person_name,
                    "emotion":   tf.emotion,
                    "track_id":  tf.track_id,
                }
                for tf in tracked_faces
            ]
            analyzer.feed_faces(face_dicts)
            if transcript:
                analyzer.feed_speech(transcript)
            
            if pose_result:
                analyzer.feed_gesture(pose_result.gesture)
                analyzer.feed_proximity(pose_result.proximity_hint)

            event = analyzer.tick()
            if event and event.get("ai_insight"):
                hud.push_insight(event["ai_insight"])

            # ── HUD ───────────────────────────────────────────────────────────
            hud.render(
                frame_bgr          = frame,
                tracked_faces      = tracked_faces,
                pending_enrollment = pending_ids,
                transcript         = transcript,
                insight            = event.get("ai_insight") if event else None,
            )

            # ── Dashboard Broadcast ──
            if transcript:
                broadcast_worker.put_event("transcript", transcript)

            # ── FPS counter ───────────────────────────────────────────────────
            fps_counter += 1
            if time.time() - fps_timer >= 3.0:
                current_fps = fps_counter / (time.time() - fps_timer)
                fps_timer   = time.time()
                fps_counter = 0
                logger.debug(f"[Main] FPS: {current_fps:.1f} | Faces: {len(tracked_faces)}")

            if not hud.handle_events():
                break

    except KeyboardInterrupt:
        logger.info("[Main] Interrupted by user")
    finally:
        cap.release()
        pose_pipeline.close()
        hud.close()
        speech_engine.stop_mic_stream()
        console.print("\n[bold yellow]Cortexium stopped.[/bold yellow]")


if __name__ == "__main__":
    main()
