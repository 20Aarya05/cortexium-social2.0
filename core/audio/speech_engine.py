"""
Speech Engine — local Whisper STT with 3-second chunked streaming.
Runs in a daemon thread; results queued for the main pipeline.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Optional

import numpy as np
from loguru import logger

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import config as cfg

try:
    import whisper
    _WHISPER_OK = True
except ImportError:
    _WHISPER_OK = False
    logger.warning("[Speech] openai-whisper not installed — STT disabled")

try:
    import sounddevice as sd
    _SD_OK = True
except ImportError:
    try:
        import pyaudio
        _PA_OK = True
        _SD_OK = False
    except ImportError:
        _SD_OK = _PA_OK = False
        logger.warning("[Speech] No audio capture library — mic STT disabled")


SAMPLE_RATE   = 16_000
CHUNK_SECONDS = 3
OVERLAP_SECS  = 0.5


class SpeechEngine:
    def __init__(self, model_size: str = "small"):
        self._model = None
        self._transcript_queue: queue.Queue[str] = queue.Queue()
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._running = False

        if _WHISPER_OK:
            logger.info(f"[Speech] Loading Whisper '{model_size}' model …")
            try:
                # Use CPU for stability if GPU is flaky, 
                # but ORT usually handles device internally for whisper if installed.
                self._model = whisper.load_model(model_size)
                logger.info("[Speech] Whisper loaded ✓")
            except Exception as e:
                logger.error(f"[Speech] Whisper load failed: {e}")

    # ── Transcription Worker ──────────────────────────────────────────────────

    def _transcription_worker(self):
        """Processes raw audio chunks from the audio_queue."""
        logger.info("[Speech] Transcription worker started")
        while self._running:
            try:
                # Wait for a chunk with timeout to check self._running
                audio_np = self._audio_queue.get(timeout=1.0)
                text = self.transcribe_chunk(audio_np)
                if text:
                    logger.debug(f"[Speech] Transcribed: {text}")
                    self._transcript_queue.put(text)
                self._audio_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[Speech] Worker error: {e}")

    def transcribe_chunk(self, audio_np: np.ndarray) -> str:
        """Synchronously transcribe a float32 mono audio array at 16 kHz."""
        if self._model is None:
            return ""
        try:
            # Whisper .transcribe handles the heavy lifting
            result = self._model.transcribe(
                audio_np.astype(np.float32),
                fp16=False,
                language="en",
            )
            return result.get("text", "").strip()
        except Exception as e:
            logger.debug(f"[Speech] Transcription error: {e}")
            return ""

    # ── Microphone streaming thread ───────────────────────────────────────────

    def start_mic_stream(self):
        """Start background mic capture and transcription worker."""
        if self._model is None:
            return
        self._running = True
        
        # 1. Start worker
        worker_t = threading.Thread(target=self._transcription_worker, daemon=True)
        worker_t.start()
        
        # 2. Start capture loop
        capture_t = threading.Thread(target=self._mic_loop, daemon=True)
        capture_t.start()
        
        logger.info("[Speech] Mic & Worker threads started")

    def stop_mic_stream(self):
        self._running = False

    def get_latest_transcript(self) -> Optional[str]:
        """Non-blocking. Returns latest transcript or None."""
        try:
            return self._transcript_queue.get_nowait()
        except queue.Empty:
            return None

    def _mic_loop(self):
        if not _SD_OK:
            logger.warning("[Speech] sounddevice unavailable — mic loop disabled")
            return
            
        chunk_size = int(SAMPLE_RATE * CHUNK_SECONDS)
        buffer = np.zeros(chunk_size, dtype=np.float32)

        def callback(indata, frames, time_info, status):
            nonlocal buffer
            if status:
                logger.warning(f"[Speech] Mic status: {status}")
            mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
            buffer = np.roll(buffer, -len(mono))
            buffer[-len(mono):] = mono

        try:
            logger.info(f"[Speech] Opening mic: {cfg.MIC_SOURCE if cfg.MIC_SOURCE is not None else 'Default'}")
            with sd.InputStream(
                device=cfg.MIC_SOURCE,
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                callback=callback,
                blocksize=int(SAMPLE_RATE * 0.1),
            ):
                while self._running:
                    # Capture a snapshot every (CHUNK_SECONDS - OVERLAP)
                    time.sleep(CHUNK_SECONDS - OVERLAP_SECS)
                    # Put current buffer in queue for worker
                    self._audio_queue.put(buffer.copy())
                    
        except Exception as e:
            logger.error(f"[Speech] Mic loop crashed: {e}")

