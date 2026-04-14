"""
HUD Renderer — PyGame transparent overlay.
Draws bounding boxes, name badges, emotion chips,
live transcription ticker, and AI insight bubble.
"""

from __future__ import annotations

import sys
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np
from loguru import logger

import sys as _sys, pathlib
_sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import config as cfg

try:
    import pygame
    _PG_OK = True
except ImportError:
    _PG_OK = False
    logger.warning("[HUD] pygame not installed — HUD overlay disabled")

# ── Palette ───────────────────────────────────────────────────────────────────
C_BG          = (10,  12,  20,  200)    # dark glass
C_KNOWN       = (0,   220, 140, 255)    # teal green
C_UNKNOWN     = (255, 80,  80,  255)    # red
C_EMOTION     = (255, 200, 60,  255)    # amber
C_TEXT        = (255, 255, 255, 255)
C_INSIGHT_BG  = (20,  20,  50,  210)
C_TICKER_BG   = (0,   0,   0,   160)
C_PENDING     = (100, 180, 255, 255)    # blue — awaiting enrollment

EMOTION_EMOJI = {
    "happy":     "😊",
    "sad":       "😢",
    "angry":     "😠",
    "surprised": "😲",
    "disgust":   "🤢",
    "fear":      "😨",
    "neutral":   "😐",
}


class HUDRenderer:
    def __init__(self):
        self._ok = False
        self._screen = None
        self._font_sm = self._font_md = self._font_lg = None
        self._ticker  = deque(maxlen=6)   # recent transcript lines
        self._insight = ""
        self._insight_time = 0.0
        self._init_pygame()

    def _init_pygame(self):
        if not _PG_OK:
            return
        try:
            pygame.init()
            flags  = pygame.SRCALPHA
            if cfg.HUD_FULLSCREEN:
                flags |= pygame.FULLSCREEN
            self._screen = pygame.display.set_mode(
                (cfg.HUD_WIDTH, cfg.HUD_HEIGHT), flags
            )
            pygame.display.set_caption("Cortexium HUD")

            # Fonts (fallback to system font if Consolas unavailable)
            self._font_sm = pygame.font.SysFont("Consolas", 14)
            self._font_md = pygame.font.SysFont("Consolas", 18, bold=True)
            self._font_lg = pygame.font.SysFont("Consolas", 22, bold=True)

            self._ok = True
            logger.info(f"[HUD] PyGame overlay {cfg.HUD_WIDTH}×{cfg.HUD_HEIGHT}")
        except Exception as e:
            logger.error(f"[HUD] Init failed: {e}")

    # ── Public API ────────────────────────────────────────────────────────────

    def render(
        self,
        frame_bgr: np.ndarray,
        tracked_faces: list,
        pending_enrollment: list[int],
        transcript: Optional[str] = None,
        insight: Optional[str] = None,
    ):
        """
        Draws the HUD overlay on top of the camera frame.
        Also renders the combined view in a CV2 window (works without pygame).
        """
        # Draw with OpenCV first (always works)
        canvas = self._draw_cv2(
            frame_bgr.copy(),
            tracked_faces,
            pending_enrollment,
            transcript,
            insight,
        )

        cv2.imshow("Cortexium — Social Intelligence", canvas)

        # Also update PyGame window if available
        if self._ok and self._screen:
            self._draw_pygame(canvas)

    def push_transcript(self, text: str):
        if text:
            self._ticker.appendleft(f"🎙 {text}")

    def push_insight(self, text: str):
        self._insight = text
        self._insight_time = time.time()

    def handle_events(self) -> bool:
        """Returns False if user closed window."""
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False
        if _PG_OK:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    return False
        return True

    def close(self):
        cv2.destroyAllWindows()
        if _PG_OK:
            pygame.quit()

    # ── OpenCV drawing ────────────────────────────────────────────────────────

    def _draw_cv2(self, frame, tracked_faces, pending_ids, transcript, insight):
        h, w = frame.shape[:2]
        overlay = frame.copy()

        for tf in tracked_faces:
            x1, y1, x2, y2 = tf.bbox
            is_pending  = tf.track_id in pending_ids
            is_known    = tf.person_id is not None

            # Bounding box
            color = (
                (100, 180, 255) if is_pending
                else (0, 220, 140) if is_known
                else (80,  80, 255)
            )
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

            # Name badge
            label = f"{tf.person_name}"
            if is_pending:
                label = "👤 Who is this?"
            conf_pct = int(tf.confidence * 100)
            sub = f"{tf.emotion} {conf_pct}%"

            self._draw_badge(overlay, label, sub, x1, y1, color)

        # ── Enrollment prompt ─────────────────────────────────────────────────
        if pending_ids:
            msg = "👤 New person detected — say: 'This is [Name]'"
            self._draw_banner(overlay, msg, w, color=(80, 160, 255))

        # ── Transcription ticker ──────────────────────────────────────────────
        if transcript:
            self._ticker.appendleft(f"🎙 {transcript}")

        ticker_y = h - 30
        for line in list(self._ticker)[:3]:
            cv2.rectangle(overlay, (0, ticker_y - 18), (w, ticker_y + 4), (0, 0, 0), -1)
            cv2.putText(overlay, line[:100], (8, ticker_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1, cv2.LINE_AA)
            ticker_y -= 22

        # ── AI Insight ────────────────────────────────────────────────────────
        if insight and (time.time() - self._insight_time) < 8:
            self._draw_insight_bubble(overlay, insight, w, h)
        elif insight and self._insight:
            self._draw_insight_bubble(overlay, self._insight, w, h)

        # ── Session info (top-right) ──────────────────────────────────────────
        ts = time.strftime("%H:%M:%S")
        cv2.putText(overlay, f"[CORTEXIUM] {ts}", (w - 220, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 200), 1, cv2.LINE_AA)
        cv2.putText(overlay, f"Faces: {len(tracked_faces)}", (w - 220, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        # Blend overlay
        result = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)
        return result

    def _draw_badge(self, img, label: str, sub: str, x: int, y: int, color):
        pad = 4
        fs  = 0.55
        th  = cv2.FONT_HERSHEY_SIMPLEX

        (tw, _), _ = cv2.getTextSize(label, th, fs, 1)
        bg_x2 = x + tw + pad * 2
        bg_y1 = max(0, y - 30)
        bg_y2 = y

        cv2.rectangle(img, (x, bg_y1), (bg_x2, bg_y2), color, -1)
        cv2.putText(img, label, (x + pad, y - 14), th, fs, (10, 10, 10), 1, cv2.LINE_AA)
        cv2.putText(img, sub,   (x + pad, y - 2),  th, 0.38, (40, 40, 40), 1, cv2.LINE_AA)

    def _draw_banner(self, img, text: str, w: int, color=(80, 160, 255)):
        cv2.rectangle(img, (0, 0), (w, 36), (0, 0, 0), -1)
        cv2.putText(img, text, (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

    def _draw_insight_bubble(self, img, text: str, w: int, h: int):
        lines = [text[i:i+80] for i in range(0, min(len(text), 240), 80)]
        bh = len(lines) * 22 + 12
        by = h - 110 - bh
        cv2.rectangle(img, (8, by), (w - 8, by + bh), (20, 20, 50), -1)
        cv2.rectangle(img, (8, by), (w - 8, by + bh), (80, 80, 160), 1)
        cv2.putText(img, "🧠 AI Insight", (14, by + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 255), 1)
        for i, line in enumerate(lines):
            cv2.putText(img, line, (14, by + 32 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

    # ── PyGame blitting ───────────────────────────────────────────────────────

    def _draw_pygame(self, canvas_bgr: np.ndarray):
        try:
            rgb  = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)
            surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
            surf = pygame.transform.scale(surf, (cfg.HUD_WIDTH, cfg.HUD_HEIGHT))
            self._screen.blit(surf, (0, 0))
            pygame.display.flip()
        except Exception as e:
            logger.debug(f"[HUD] PyGame render error: {e}")
