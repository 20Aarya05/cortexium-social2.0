"""
HUD Renderer — Premium Cyber Aesthetic.
Draws futuristic targeting reticles, eye tracking, biometric side panels,
and AI insight cards matching high-end tactical interfaces.
"""

from __future__ import annotations

import sys
import time
import datetime
from collections import deque
from typing import Optional

import cv2
import numpy as np
from loguru import logger

import sys as _sys, pathlib
_sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import config as cfg
from core.storage.person_registry import get_person

try:
    import pygame
    _PG_OK = True
except ImportError:
    _PG_OK = False

# ── Cyber Palette ─────────────────────────────────────────────────────────────
C_CYAN    = (255, 255, 0)   # BGR Cyan
C_WHITE   = (255, 255, 255)
C_RED     = (80, 80, 255)
C_DARK    = (15, 12, 10)
C_ACCENT  = (220, 220, 0)
C_PANEL   = (30, 25, 20, 180) # RGBA panel

class HUDRenderer:
    def __init__(self):
        self._ok = False
        self._screen = None
        self._ticker = deque(maxlen=4)
        self._insight = ""
        self._insight_time = 0.0
        self._scan_y = 0
        self._init_pygame()

    def _init_pygame(self):
        if not _PG_OK: return
        try:
            pygame.init()
            flags = pygame.SRCALPHA
            if cfg.HUD_FULLSCREEN: flags |= pygame.FULLSCREEN
            self._screen = pygame.display.set_mode((cfg.HUD_WIDTH, cfg.HUD_HEIGHT), flags)
            pygame.display.set_caption("Cortexium HUD")
            self._ok = True
            logger.info(f"[HUD] Cyber Overlay Ready")
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
        h, w = frame_bgr.shape[:2]
        overlay = frame_bgr.copy()

        # Update animation states
        self._scan_y = (self._scan_y + 4) % h

        # 1. Draw Global HUD Elements
        self._draw_status_bar(overlay, w, h)
        self._draw_branding(overlay, w, h)

        # 2. Draw Tracked Subjects
        for tf in tracked_faces:
            is_pending = tf.track_id in pending_enrollment
            color = C_CYAN if not is_pending else C_RED
            
            # Targeting Reticle
            self._draw_reticle(overlay, tf.bbox, color)
            
            # Eye Tracker
            if hasattr(tf, "landmarks") and tf.landmarks:
                self._draw_eyes(overlay, tf.landmarks, color)

            # Subject Details (only if identified)
            if tf.person_id:
                person = get_person(tf.person_id)
                if person:
                    self._draw_subject_card(overlay, person, tf, w, h)

        # 3. Transcript & Insights
        if transcript: self._ticker.appendleft(transcript)
        self._draw_ticker(overlay, w, h)
        
        if insight:
            self._insight = insight
            self._insight_time = time.time()
        
        if self._insight and (time.time() - self._insight_time < 10):
            self._draw_insight_card(overlay, self._insight, w, h)

        # 4. Final Compositing
        result = cv2.addWeighted(overlay, 0.85, frame_bgr, 0.15, 0)
        
        # Output
        cv2.imshow("Cortexium — Social Intelligence", result)
        if self._ok and self._screen:
            self._draw_pygame(result)

    def push_transcript(self, text: str):
        if text: self._ticker.appendleft(text)

    def push_insight(self, text: str):
        self._insight = text
        self._insight_time = time.time()

    def handle_events(self) -> bool:
        if cv2.waitKey(1) & 0xFF == ord("q"): return False
        if _PG_OK:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: return False
        return True

    def close(self):
        cv2.destroyAllWindows()
        if _PG_OK: pygame.quit()

    # ── Drawing Components ────────────────────────────────────────────────────

    def _draw_reticle(self, img, bbox, color):
        x1, y1, x2, y2 = bbox
        l = 25  # length of corners
        t = 2   # thickness
        
        # Corners
        # Top-Left
        cv2.line(img, (x1, y1), (x1 + l, y1), color, t)
        cv2.line(img, (x1, y1), (x1, y1 + l), color, t)
        # Top-Right
        cv2.line(img, (x2, y1), (x2 - l, y1), color, t)
        cv2.line(img, (x2, y1), (x2, y1 + l), color, t)
        # Bottom-Left
        cv2.line(img, (x1, y2), (x1 + l, y2), color, t)
        cv2.line(img, (x1, y2), (x1, y2 - l), color, t)
        # Bottom-Right
        cv2.line(img, (x2, y2), (x2 - l, y2), color, t)
        cv2.line(img, (x2, y2), (x2, y2 - l), color, t)

        # Labels
        fs = 0.35
        cv2.putText(img, "FACE TRACKED", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, fs, color, 1)
        cv2.putText(img, f"ID: {bbox[0]:04d}", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, fs, color, 1)

    def _draw_eyes(self, img, kps, color):
        # InsightFace kps: [left_eye, right_eye, nose, left_mouth, right_mouth]
        for i in range(2):
            ex, ey = int(kps[i][0]), int(kps[i][1])
            s = 12
            cv2.rectangle(img, (ex-s, ey-s), (ex+s, ey+s), color, 1)
            cv2.putText(img, "EYE TRACKED", (ex - 20, ey + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    def _draw_subject_card(self, img, person, tf, w, h):
        card_w, card_h = 240, 260
        tx, ty = w - card_w - 20, 100
        
        # Glass Panel
        sub = img[ty:ty+card_h, tx:tx+card_w]
        rect = np.zeros_like(sub)
        cv2.rectangle(rect, (0, 0), (card_w, card_h), (40, 35, 30), -1)
        res = cv2.addWeighted(sub, 0.4, rect, 0.6, 0)
        img[ty:ty+card_h, tx:tx+card_w] = res
        cv2.rectangle(img, (tx, ty), (tx+card_w, ty+card_h), C_CYAN, 1)

        # Content
        px = tx + 15
        py = ty + 30
        
        # Circle Portrait (Placeholder)
        cv2.circle(img, (tx + card_w//2, ty + 50), 35, C_CYAN, 1)
        cv2.circle(img, (tx + card_w//2, ty + 50), 30, (100, 100, 100), -1)
        
        py += 80
        cv2.putText(img, "SUBJECT:", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_CYAN, 1)
        cv2.putText(img, person.name.upper(), (px, py+18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_WHITE, 1)
        
        py += 45
        role = "SOCIAL ENTITY" if person.name != "Unknown" else "UNREGISTERED"
        cv2.putText(img, "ROLE:", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_CYAN, 1)
        cv2.putText(img, role, (px, py+16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_WHITE, 1)

        py += 40
        last = person.last_seen.strftime("%H:%M:%S") if person.last_seen else "N/A"
        cv2.putText(img, "LAST SEEN:", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_CYAN, 1)
        cv2.putText(img, f"TODAY ({last})", (px, py+16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_WHITE, 1)
        
        py += 40
        cv2.putText(img, "EMOTION:", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_CYAN, 1)
        cv2.putText(img, tf.emotion.upper(), (px, py+16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 255, 200), 1)


    def _draw_status_bar(self, img, w, h):
        # Simplified Bottom Status
        bx, by = w // 2 - 80, h - 40
        cv2.putText(img, "SYSTEM: ACTIVE", (bx, by), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)

    def _draw_branding(self, img, w, h):
        # Top Left
        cv2.putText(img, "CORTEXIUM // SOCIAL INTEL", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_CYAN, 1)

    def _draw_ticker(self, img, w, h):
        ty = h - 60
        for i, line in enumerate(list(self._ticker)):
            alpha = 1.0 - (i * 0.25)
            color = (int(255*alpha), int(255*alpha), int(255*alpha))
            cv2.putText(img, f"> {line}", (20, ty - i*20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    def _draw_insight_card(self, img, text, w, h):
        cx, cy = 20, h - 220
        cv2.putText(img, "AI INSIGHT:", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 50), 1)
        
        words = text.split()
        lines = []
        cur = ""
        for word in words:
            if len(cur + word) < 40: cur += word + " "
            else:
                lines.append(cur)
                cur = word + " "
        lines.append(cur)
        
        for i, line in enumerate(lines[:3]):
            cv2.putText(img, line, (cx, cy + 20 + i*16), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_WHITE, 1)

    def _draw_pygame(self, canvas_bgr: np.ndarray):
        try:
            rgb = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)
            surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
            surf = pygame.transform.scale(surf, (cfg.HUD_WIDTH, cfg.HUD_HEIGHT))
            self._screen.blit(surf, (0, 0))
            pygame.display.flip()
        except Exception: pass
