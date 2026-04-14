"""
Body Pose Estimation — MediaPipe Holistic.
Extracts skeleton landmarks and high-level social signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from loguru import logger

try:
    import mediapipe as mp
    _MP_OK = True
except ImportError:
    _MP_OK = False
    logger.warning("[Pose] mediapipe not installed — body pose disabled")


@dataclass
class PoseResult:
    gesture: str = "neutral"          # nodding, waving, pointing, handshake, crossed_arms
    head_orientation: str = "forward" # forward, left, right, down, up
    proximity_hint: str = "medium"    # close, medium, far (based on face/shoulder size)
    raw_landmarks: Optional[object] = None


class BodyPosePipeline:
    def __init__(self):
        self._holistic = None
        if _MP_OK:
            self._holistic = mp.solutions.holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            logger.info("[Pose] MediaPipe Holistic loaded")

    def process_frame(self, frame_rgb: np.ndarray) -> PoseResult:
        """frame_rgb must be RGB. Returns PoseResult."""
        if self._holistic is None:
            return PoseResult()

        result = self._holistic.process(frame_rgb)
        pose = PoseResult(raw_landmarks=result)

        if result.pose_landmarks:
            pose.gesture         = self._detect_gesture(result)
            pose.head_orientation= self._head_orientation(result)

        if result.face_landmarks:
            pose.proximity_hint = self._estimate_proximity(result, frame_rgb.shape)

        return pose

    # ── Signal detectors ──────────────────────────────────────────────────────

    def _detect_gesture(self, result) -> str:
        lm = result.pose_landmarks.landmark
        mp_pose = mp.solutions.pose.PoseLandmark

        # Waving: wrist above shoulder
        r_wrist   = lm[mp_pose.RIGHT_WRIST]
        r_shoulder= lm[mp_pose.RIGHT_SHOULDER]
        l_wrist   = lm[mp_pose.LEFT_WRIST]
        l_shoulder= lm[mp_pose.LEFT_SHOULDER]

        if r_wrist.y < r_shoulder.y - 0.1 or l_wrist.y < l_shoulder.y - 0.1:
            return "waving"

        # Crossed arms: wrists cross the body midline
        nose = lm[mp_pose.NOSE]
        if (r_wrist.x < nose.x - 0.05) and (l_wrist.x > nose.x + 0.05):
            return "crossed_arms"

        # Pointing: one arm extended forward (z-depth heuristic)
        if abs(r_wrist.z) > 0.15 or abs(l_wrist.z) > 0.15:
            return "pointing"

        return "neutral"

    def _head_orientation(self, result) -> str:
        if not result.face_landmarks:
            return "forward"
        lm = result.face_landmarks.landmark
        # Nose tip vs ear positions (simplified)
        nose    = lm[1]
        l_ear   = lm[234]
        r_ear   = lm[454]
        mid_x   = (l_ear.x + r_ear.x) / 2
        if nose.x < mid_x - 0.05:
            return "left"
        if nose.x > mid_x + 0.05:
            return "right"
        if nose.y < lm[0].y - 0.02:
            return "up"
        if nose.y > lm[0].y + 0.02:
            return "down"
        return "forward"

    def _estimate_proximity(self, result, shape) -> str:
        """Estimate proximity by face mesh bounding box size relative to frame."""
        h, w = shape[:2]
        xs = [lm.x for lm in result.face_landmarks.landmark]
        ys = [lm.y for lm in result.face_landmarks.landmark]
        face_w = (max(xs) - min(xs)) * w
        if face_w > w * 0.35:
            return "close"
        if face_w > w * 0.12:
            return "medium"
        return "far"

    def close(self):
        if self._holistic:
            self._holistic.close()
