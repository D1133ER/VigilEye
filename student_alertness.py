#!/usr/bin/env python3
"""Real-time Student Attentive Alertness System.

This script uses MediaPipe's Face Landmarker task API, geometric eye/mouth
features, iris-aware gaze estimation, head pose estimation with solvePnP, and
 blendshape-based engagement heuristics to estimate student attention locally.

The implementation is intentionally modular so you can later add other signals,
such as hand raise detection or a YOLO-based classroom behavior model, without
rewriting the webcam loop.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import mediapipe as mp
import numpy as np

try:
    import pygame
except Exception:
    pygame = None


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "face_landmarker.task"
LOG_DIR = PROJECT_ROOT / "logs"
RUNTIME_DIR = PROJECT_ROOT / "runtime"
LATEST_STATE_PATH = RUNTIME_DIR / "latest_state.json"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# These thresholds are intentionally defined near the top because classroom,
# webcam, and seating setups vary. In 2026 practice, keeping them explicit and
# easy to tune is more maintainable than burying them deep inside the pipeline.
MIN_FACE_DETECTION_CONFIDENCE = 0.55
MIN_FACE_PRESENCE_CONFIDENCE = 0.55
MIN_TRACKING_CONFIDENCE = 0.55

# EAR thresholds reflect common practical values for MediaPipe facial landmarks.
# A hysteresis pair reduces noisy open/closed flips in normal indoor lighting.
EAR_CLOSED_THRESHOLD = 0.18
EAR_RECOVERY_THRESHOLD = 0.21
EAR_OPEN_REFERENCE = 0.30
DROWSY_EYES_CLOSED_SECONDS = 2.0

# MAR is computed from three inner-mouth vertical spans divided by mouth width.
# The yawn trigger is sustained to avoid flagging brief speech or lip movement.
MAR_YAWN_THRESHOLD = 0.38
MAR_RECOVERY_THRESHOLD = 0.30
MAR_WIDE_OPEN_REFERENCE = 0.55
YAWN_SECONDS = 1.2

# Head pose limits are intentionally conservative so small natural movement does
# not count as distraction, but obvious off-axis posture does.
HEAD_YAW_LIMIT_DEG = 22.0
HEAD_PITCH_LIMIT_DEG = 18.0
HEAD_ROLL_LIMIT_DEG = 20.0
LOOK_AWAY_SECONDS = 1.0

# Gaze offsets are normalized around each eye and centered at 0.0. Values above
# these limits usually indicate looking away from the screen for a seated user.
GAZE_HORIZONTAL_LIMIT = 0.18
GAZE_VERTICAL_LIMIT = 0.16

# Attention score weights sum to 1.0. Eyes and gaze dominate because they are
# the strongest attention cues in laptop-based classroom scenarios.
ATTENTION_WEIGHTS = {
    "eyes": 0.32,
    "mouth": 0.10,
    "head": 0.20,
    "gaze": 0.22,
    "engagement": 0.16,
}

ATTENTIVE_THRESHOLD = 75.0
MODERATE_THRESHOLD = 55.0
LOW_ATTENTION_ALERT_THRESHOLD = 45.0
ALERT_COOLDOWN_SECONDS = 3.0

STATE_WRITE_INTERVAL_SECONDS = 0.25
DEFAULT_LOG_INTERVAL_SECONDS = 0.50

GREEN = (60, 200, 90)
YELLOW = (0, 210, 255)
RED = (40, 40, 230)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


# Landmark indices for MediaPipe's 478-point face landmark topology.
RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]
LEFT_EYE_EAR = [263, 387, 385, 362, 380, 373]

RIGHT_EYE_CORNERS = (33, 133)
LEFT_EYE_CORNERS = (263, 362)
RIGHT_EYE_LIDS = (159, 145)
LEFT_EYE_LIDS = (386, 374)
RIGHT_IRIS = [468, 469, 470, 471, 472]
LEFT_IRIS = [473, 474, 475, 476, 477]

MOUTH_LEFT_RIGHT = (78, 308)
MOUTH_VERTICAL_PAIRS = [(13, 14), (82, 87), (312, 317)]

# A small, stable solvePnP point set is fast and practical for real-time use.
HEAD_POSE_INDICES = [1, 33, 61, 199, 263, 291]
MIN_REQUIRED_LANDMARKS = 478
HEAD_POSE_CENTER_INDEX = 1

BLENDSHAPE_LABELS = {
    "browInnerUp": "brows raised / curiosity",
    "browOuterUpLeft": "left brow raise",
    "browOuterUpRight": "right brow raise",
    "mouthPressLeft": "left lip press / concentration",
    "mouthPressRight": "right lip press / concentration",
    "eyeSquintLeft": "left eye squint / strain",
    "eyeSquintRight": "right eye squint / strain",
    "jawOpen": "jaw open / fatigue",
    "mouthPucker": "mouth pucker / uncertainty",
    "mouthFrownLeft": "left frown cue",
    "mouthFrownRight": "right frown cue",
    "eyeBlinkLeft": "left blink",
    "eyeBlinkRight": "right blink",
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def ensure_runtime_dirs() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)


def euclidean(point_a: np.ndarray, point_b: np.ndarray) -> float:
    return float(np.linalg.norm(point_a - point_b))


def point_array(face_landmarks: list[Any], frame_width: int, frame_height: int) -> tuple[np.ndarray, np.ndarray]:
    normalized = np.array(
        [[lm.x, lm.y, lm.z] for lm in face_landmarks],
        dtype=np.float64,
    )
    pixels = normalized.copy()
    pixels[:, 0] *= frame_width
    pixels[:, 1] *= frame_height
    pixels[:, 2] *= frame_width
    return normalized, pixels


def compute_ear(landmarks_px: np.ndarray, indices: list[int]) -> float:
    p1, p2, p3, p4, p5, p6 = landmarks_px[indices]
    horizontal = euclidean(p1[:2], p4[:2])
    if horizontal <= 1e-6:
        return 0.0
    vertical = euclidean(p2[:2], p6[:2]) + euclidean(p3[:2], p5[:2])
    return vertical / (2.0 * horizontal)


def compute_avg_ear(landmarks_px: np.ndarray) -> float:
    right_ear = compute_ear(landmarks_px, RIGHT_EYE_EAR)
    left_ear = compute_ear(landmarks_px, LEFT_EYE_EAR)
    return (right_ear + left_ear) / 2.0


def compute_mar(landmarks_px: np.ndarray) -> float:
    mouth_width = euclidean(
        landmarks_px[MOUTH_LEFT_RIGHT[0]][:2],
        landmarks_px[MOUTH_LEFT_RIGHT[1]][:2],
    )
    if mouth_width <= 1e-6:
        return 0.0
    vertical_values = [
        euclidean(landmarks_px[top][:2], landmarks_px[bottom][:2])
        for top, bottom in MOUTH_VERTICAL_PAIRS
    ]
    return float(np.mean(vertical_values) / mouth_width)


def iris_center(landmarks_px: np.ndarray, iris_indices: list[int]) -> np.ndarray:
    iris_points = landmarks_px[iris_indices, :2]
    return np.mean(iris_points, axis=0)


def compute_eye_gaze_ratio(
    center: np.ndarray,
    outer_corner: np.ndarray,
    inner_corner: np.ndarray,
    upper_lid: np.ndarray,
    lower_lid: np.ndarray,
) -> tuple[float, float]:
    horizontal_denominator = (inner_corner[0] - outer_corner[0])
    vertical_denominator = (lower_lid[1] - upper_lid[1])
    horizontal_ratio = (center[0] - outer_corner[0]) / (horizontal_denominator + 1e-6)
    vertical_ratio = (center[1] - upper_lid[1]) / (vertical_denominator + 1e-6)
    return float(horizontal_ratio), float(vertical_ratio)


def compute_gaze_metrics(landmarks_px: np.ndarray) -> tuple[float, float, float, float]:
    right_center = iris_center(landmarks_px, RIGHT_IRIS)
    left_center = iris_center(landmarks_px, LEFT_IRIS)

    right_h, right_v = compute_eye_gaze_ratio(
        center=right_center,
        outer_corner=landmarks_px[RIGHT_EYE_CORNERS[0]][:2],
        inner_corner=landmarks_px[RIGHT_EYE_CORNERS[1]][:2],
        upper_lid=landmarks_px[RIGHT_EYE_LIDS[0]][:2],
        lower_lid=landmarks_px[RIGHT_EYE_LIDS[1]][:2],
    )
    left_h, left_v = compute_eye_gaze_ratio(
        center=left_center,
        outer_corner=landmarks_px[LEFT_EYE_CORNERS[0]][:2],
        inner_corner=landmarks_px[LEFT_EYE_CORNERS[1]][:2],
        upper_lid=landmarks_px[LEFT_EYE_LIDS[0]][:2],
        lower_lid=landmarks_px[LEFT_EYE_LIDS[1]][:2],
    )

    horizontal_offset = ((right_h - 0.5) + (left_h - 0.5)) / 2.0
    vertical_offset = ((right_v - 0.5) + (left_v - 0.5)) / 2.0
    deviation = (
        abs(horizontal_offset) / (GAZE_HORIZONTAL_LIMIT + 1e-6)
        + abs(vertical_offset) / (GAZE_VERTICAL_LIMIT + 1e-6)
    ) / 2.0
    deviation = clamp(deviation, 0.0, 1.5)
    centered_score = 1.0 - clamp(deviation, 0.0, 1.0)
    return horizontal_offset, vertical_offset, deviation, centered_score


def estimate_head_pose(
    landmarks_normalized: np.ndarray,
    landmarks_px: np.ndarray,
    frame_width: int,
    frame_height: int,
) -> tuple[float, float, float]:
    face_2d = np.array([landmarks_px[index][:2] for index in HEAD_POSE_INDICES], dtype=np.float64)
    scale = float(max(frame_width, frame_height))
    face_center = landmarks_normalized[HEAD_POSE_CENTER_INDEX]
    face_3d = np.array([landmarks_normalized[index] - face_center for index in HEAD_POSE_INDICES], dtype=np.float64)
    face_3d *= scale

    focal_length = float(frame_width)
    camera_matrix = np.array(
        [
            [focal_length, 0.0, frame_width / 2.0],
            [0.0, focal_length, frame_height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortion = np.zeros((4, 1), dtype=np.float64)

    solve_pnp_flag = getattr(cv2, "SOLVEPNP_ITERATIVE", cv2.SOLVEPNP_EPNP)
    success, rotation_vector, translation_vector = cv2.solvePnP(
        face_3d,
        face_2d,
        camera_matrix,
        distortion,
        flags=solve_pnp_flag,
    )
    if not success:
        return 0.0, 0.0, 0.0

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    projection_matrix = np.hstack((rotation_matrix, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(projection_matrix)
    pitch, yaw, roll = [float(angle) for angle in euler_angles.flatten()]
    return pitch, yaw, roll


def face_bounding_box(landmarks_px: np.ndarray, frame_width: int, frame_height: int) -> list[int]:
    x_coords = np.clip(landmarks_px[:, 0], 0, frame_width - 1)
    y_coords = np.clip(landmarks_px[:, 1], 0, frame_height - 1)
    x1 = int(np.min(x_coords))
    y1 = int(np.min(y_coords))
    x2 = int(np.max(x_coords))
    y2 = int(np.max(y_coords))
    return [x1, y1, x2, y2]


def category_scores(blendshapes: list[Any]) -> dict[str, float]:
    return {category.category_name: float(category.score) for category in blendshapes}


@dataclass
class EngagementResult:
    score: float
    label: str
    insights: list[str] = field(default_factory=list)


def analyze_blendshapes(scores: dict[str, float]) -> EngagementResult:
    if not scores:
        return EngagementResult(score=0.50, label="neutral", insights=[])

    def avg(*keys: str) -> float:
        if not keys:
            return 0.0
        return float(sum(scores.get(key, 0.0) for key in keys) / len(keys))

    brow_raise = avg("browInnerUp", "browOuterUpLeft", "browOuterUpRight")
    lip_press = avg("mouthPressLeft", "mouthPressRight")
    eye_squint = avg("eyeSquintLeft", "eyeSquintRight")
    eye_blink = avg("eyeBlinkLeft", "eyeBlinkRight")
    soft_smile = avg("mouthSmileLeft", "mouthSmileRight")
    jaw_open = scores.get("jawOpen", 0.0)
    mouth_pucker = scores.get("mouthPucker", 0.0)
    frown = avg("mouthFrownLeft", "mouthFrownRight")
    brow_down = avg("browDownLeft", "browDownRight")

    focused_signal = clamp(
        0.42 * brow_raise + 0.30 * lip_press + 0.18 * (1.0 - eye_squint) + 0.10 * soft_smile,
        0.0,
        1.0,
    )
    confusion_signal = clamp(
        0.38 * brow_down + 0.24 * eye_squint + 0.20 * mouth_pucker + 0.18 * jaw_open,
        0.0,
        1.0,
    )
    boredom_signal = clamp(
        0.40 * jaw_open + 0.25 * frown + 0.20 * eye_blink + 0.15 * (1.0 - brow_raise),
        0.0,
        1.0,
    )

    engagement = clamp(
        0.58 * focused_signal + 0.12 * soft_smile + 0.30 * (1.0 - max(confusion_signal, boredom_signal)),
        0.0,
        1.0,
    )

    if boredom_signal >= 0.45 and boredom_signal > confusion_signal:
        label = "bored"
    elif confusion_signal >= 0.42:
        label = "confused"
    elif engagement >= 0.68:
        label = "attentive"
    else:
        label = "neutral"

    interesting_scores = sorted(
        ((name, score) for name, score in scores.items() if score >= 0.20 and name in BLENDSHAPE_LABELS),
        key=lambda item: item[1],
        reverse=True,
    )[:3]
    insights = [BLENDSHAPE_LABELS[name] for name, _ in interesting_scores]
    if label == "attentive" and not insights:
        insights = ["focused facial activity"]
    return EngagementResult(score=engagement, label=label, insights=insights)


def score_from_range(value: float, low: float, high: float) -> float:
    return clamp((value - low) / (high - low + 1e-6), 0.0, 1.0)


def inverse_score_from_range(value: float, low: float, high: float) -> float:
    return 1.0 - score_from_range(value, low, high)


def attention_score(
    face_present: bool,
    ear: float,
    mar: float,
    pitch: float,
    yaw: float,
    roll: float,
    gaze_center_score: float,
    engagement_score_value: float,
    drowsy_active: bool,
    yawn_active: bool,
    look_away_active: bool,
) -> tuple[float, dict[str, float]]:
    if not face_present:
        return 0.0, {
            "eyes": 0.0,
            "mouth": 0.0,
            "head": 0.0,
            "gaze": 0.0,
            "engagement": 0.0,
        }

    eyes_score = score_from_range(ear, EAR_CLOSED_THRESHOLD, EAR_OPEN_REFERENCE)
    mouth_score = inverse_score_from_range(mar, MAR_RECOVERY_THRESHOLD, MAR_WIDE_OPEN_REFERENCE)
    head_score = 1.0 - clamp(
        max(
            abs(yaw) / HEAD_YAW_LIMIT_DEG,
            abs(pitch) / HEAD_PITCH_LIMIT_DEG,
            abs(roll) / HEAD_ROLL_LIMIT_DEG,
        ),
        0.0,
        1.0,
    )
    gaze_score = clamp(gaze_center_score, 0.0, 1.0)
    engagement_score_value = clamp(engagement_score_value, 0.0, 1.0)

    weighted_score = 100.0 * (
        ATTENTION_WEIGHTS["eyes"] * eyes_score
        + ATTENTION_WEIGHTS["mouth"] * mouth_score
        + ATTENTION_WEIGHTS["head"] * head_score
        + ATTENTION_WEIGHTS["gaze"] * gaze_score
        + ATTENTION_WEIGHTS["engagement"] * engagement_score_value
    )

    if drowsy_active:
        weighted_score -= 30.0
    elif ear < EAR_CLOSED_THRESHOLD:
        weighted_score -= 10.0

    if yawn_active:
        weighted_score -= 15.0

    if look_away_active:
        weighted_score -= 25.0

    return clamp(weighted_score, 0.0, 100.0), {
        "eyes": eyes_score,
        "mouth": mouth_score,
        "head": head_score,
        "gaze": gaze_score,
        "engagement": engagement_score_value,
    }


def status_for_analysis(
    face_present: bool,
    score_value: float,
    drowsy_active: bool,
    yawn_active: bool,
    look_away_active: bool,
    engagement_label: str,
) -> tuple[str, tuple[int, int, int], bool]:
    if not face_present:
        return "NO FACE DETECTED", RED, True
    if drowsy_active:
        return "DROWSY", RED, True
    if yawn_active and score_value < ATTENTIVE_THRESHOLD:
        return "YAWNING / FATIGUED", RED, True
    if look_away_active and score_value < ATTENTIVE_THRESHOLD:
        return "LOOKING AWAY", RED, True
    if score_value >= ATTENTIVE_THRESHOLD:
        return "ATTENTIVE", GREEN, False
    if score_value >= MODERATE_THRESHOLD:
        if engagement_label == "confused":
            return "MODERATE / CONFUSED", YELLOW, False
        return "MODERATE", YELLOW, False
    if engagement_label == "bored":
        return "LOW / BORED", RED, True
    return "LOW ATTENTION", RED, score_value <= LOW_ATTENTION_ALERT_THRESHOLD


@dataclass
class FrameAnalysis:
    timestamp_iso: str = field(default_factory=now_iso)
    frame_timestamp_ms: int = 0
    face_present: bool = False
    attention_score: float = 0.0
    status: str = "INITIALIZING"
    color_bgr: tuple[int, int, int] = GREEN
    alert_active: bool = False
    alert_reason: str = "warming up"
    ear: float = 0.0
    mar: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    gaze_horizontal: float = 0.0
    gaze_vertical: float = 0.0
    gaze_deviation: float = 0.0
    gaze_center_score: float = 0.0
    engagement_score: float = 0.0
    engagement_label: str = "neutral"
    blendshape_insights: list[str] = field(default_factory=list)
    closure_duration: float = 0.0
    yawn_duration: float = 0.0
    look_away_duration: float = 0.0
    face_bbox: list[int] = field(default_factory=list)
    subscores: dict[str, float] = field(default_factory=dict)
    fps: float = 0.0
    session_name: str = "student_session"
    session_log: str = ""

    def to_state_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["color_bgr"] = list(self.color_bgr)
        return payload


class TemporalState:
    def __init__(self) -> None:
        self.eye_closed_since: Optional[float] = None
        self.yawn_since: Optional[float] = None
        self.look_away_since: Optional[float] = None
        self.filtered_values: dict[str, float] = {}

    def smooth(self, key: str, value: float, alpha: float = 0.35) -> float:
        previous = self.filtered_values.get(key)
        if previous is None:
            self.filtered_values[key] = value
        else:
            self.filtered_values[key] = previous + alpha * (value - previous)
        return self.filtered_values[key]

    def update_eye_duration(self, ear: float, timestamp_s: float) -> float:
        if ear < EAR_CLOSED_THRESHOLD:
            if self.eye_closed_since is None:
                self.eye_closed_since = timestamp_s
        elif ear > EAR_RECOVERY_THRESHOLD:
            self.eye_closed_since = None
        return 0.0 if self.eye_closed_since is None else timestamp_s - self.eye_closed_since

    def update_yawn_duration(self, mar: float, timestamp_s: float) -> float:
        if mar > MAR_YAWN_THRESHOLD:
            if self.yawn_since is None:
                self.yawn_since = timestamp_s
        elif mar < MAR_RECOVERY_THRESHOLD:
            self.yawn_since = None
        return 0.0 if self.yawn_since is None else timestamp_s - self.yawn_since

    def update_look_away_duration(self, looking_away: bool, timestamp_s: float) -> float:
        if looking_away:
            if self.look_away_since is None:
                self.look_away_since = timestamp_s
        else:
            self.look_away_since = None
        return 0.0 if self.look_away_since is None else timestamp_s - self.look_away_since

    def reset(self) -> None:
        self.eye_closed_since = None
        self.yawn_since = None
        self.look_away_since = None


class AlertPlayer:
    def __init__(self, enabled: bool, sound_path: Optional[Path]) -> None:
        self.enabled = enabled
        self.sound_path = sound_path
        self.last_alert_at = 0.0
        self.queue: queue.Queue[str] = queue.Queue(maxsize=8)
        self.audio_ready = False
        self.generated_tone: Any = None
        self._prepare_audio()
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def _prepare_audio(self) -> None:
        if pygame is None:
            return
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
            self.audio_ready = True
            if self.sound_path is None or not self.sound_path.exists():
                self.generated_tone = self._build_tone()
        except Exception:
            self.audio_ready = False
            self.generated_tone = None

    def _build_tone(self) -> Any:
        sample_rate = 22050
        duration_seconds = 0.18
        frequency_hz = 880.0
        samples = np.linspace(0.0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
        waveform = (0.22 * np.sin(2.0 * np.pi * frequency_hz * samples) * 32767.0).astype(np.int16)
        return pygame.sndarray.make_sound(waveform)

    def _run(self) -> None:
        while True:
            _ = self.queue.get()
            try:
                if self.audio_ready and self.sound_path and self.sound_path.exists():
                    pygame.mixer.music.load(str(self.sound_path))
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.05)
                elif self.audio_ready and self.generated_tone is not None:
                    channel = self.generated_tone.play()
                    while channel is not None and channel.get_busy():
                        time.sleep(0.02)
                else:
                    print("\a", end="", flush=True)
            except Exception:
                print("\a", end="", flush=True)

    def maybe_alert(self, reason: str) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        if now - self.last_alert_at < ALERT_COOLDOWN_SECONDS:
            return
        self.last_alert_at = now
        try:
            self.queue.put_nowait(reason)
        except queue.Full:
            pass


class SessionLogger:
    def __init__(self, session_name: str, log_interval_seconds: float) -> None:
        safe_session_name = session_name.strip().replace(" ", "_") or "student_session"
        self.path = LOG_DIR / f"{safe_session_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.log_interval_seconds = log_interval_seconds
        self.last_write_at = 0.0
        self.last_status = ""
        self.handle = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.handle,
            fieldnames=[
                "timestamp",
                "session_name",
                "attention_score",
                "status",
                "ear",
                "mar",
                "pitch",
                "yaw",
                "roll",
                "gaze_horizontal",
                "gaze_vertical",
                "gaze_deviation",
                "engagement_score",
                "engagement_label",
                "closure_duration",
                "yawn_duration",
                "look_away_duration",
                "alert_active",
                "blendshape_insights",
            ],
        )
        self.writer.writeheader()
        self.handle.flush()

    def maybe_write(self, analysis: FrameAnalysis, current_time: float) -> None:
        if (
            current_time - self.last_write_at < self.log_interval_seconds
            and analysis.status == self.last_status
        ):
            return

        self.writer.writerow(
            {
                "timestamp": analysis.timestamp_iso,
                "session_name": analysis.session_name,
                "attention_score": round(analysis.attention_score, 2),
                "status": analysis.status,
                "ear": round(analysis.ear, 4),
                "mar": round(analysis.mar, 4),
                "pitch": round(analysis.pitch, 2),
                "yaw": round(analysis.yaw, 2),
                "roll": round(analysis.roll, 2),
                "gaze_horizontal": round(analysis.gaze_horizontal, 4),
                "gaze_vertical": round(analysis.gaze_vertical, 4),
                "gaze_deviation": round(analysis.gaze_deviation, 4),
                "engagement_score": round(analysis.engagement_score, 4),
                "engagement_label": analysis.engagement_label,
                "closure_duration": round(analysis.closure_duration, 2),
                "yawn_duration": round(analysis.yawn_duration, 2),
                "look_away_duration": round(analysis.look_away_duration, 2),
                "alert_active": analysis.alert_active,
                "blendshape_insights": " | ".join(analysis.blendshape_insights),
            }
        )
        self.handle.flush()
        self.last_write_at = current_time
        self.last_status = analysis.status

    def close(self) -> None:
        self.handle.close()


class AttentionMonitor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.frame_width = args.width
        self.frame_height = args.height
        self.session_name = args.session_name.strip().replace(" ", "_") or "student_session"
        self.temporal_state = TemporalState()
        self.lock = threading.Lock()
        self.latest_analysis = FrameAnalysis(session_name=self.session_name)
        self.capture_fps = 0.0
        self.logger = SessionLogger(self.session_name, args.log_interval_sec)
        self.alert_player = AlertPlayer(enabled=not args.no_audio_alerts, sound_path=args.alert_sound)
        self.last_state_write_at = 0.0

    def build_landmarker(self) -> Any:
        base_options = mp.tasks.BaseOptions(model_asset_path=str(self.args.model))
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_faces=1,
            min_face_detection_confidence=MIN_FACE_DETECTION_CONFIDENCE,
            min_face_presence_confidence=MIN_FACE_PRESENCE_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            result_callback=self.on_result,
        )
        return mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def on_result(self, result: Any, _output_image: Any, timestamp_ms: int) -> None:
        analysis = self._analyze_result(result, timestamp_ms)
        with self.lock:
            self.latest_analysis = analysis

    def _analyze_result(self, result: Any, timestamp_ms: int) -> FrameAnalysis:
        timestamp_s = timestamp_ms / 1000.0

        if not result.face_landmarks:
            self.temporal_state.reset()
            return FrameAnalysis(
                timestamp_iso=now_iso(),
                frame_timestamp_ms=timestamp_ms,
                face_present=False,
                attention_score=0.0,
                status="NO FACE DETECTED",
                color_bgr=RED,
                alert_active=True,
                alert_reason="no face",
                gaze_center_score=0.0,
                engagement_label="neutral",
                session_name=self.session_name,
                session_log=str(self.logger.path),
            )

        face_landmarks = result.face_landmarks[0]
        if timestamp_ms <= self.latest_analysis.frame_timestamp_ms:
            return self.get_latest_analysis()
        if len(face_landmarks) < MIN_REQUIRED_LANDMARKS:
            self.temporal_state.reset()
            return FrameAnalysis(
                timestamp_iso=now_iso(),
                frame_timestamp_ms=timestamp_ms,
                face_present=False,
                attention_score=0.0,
                status="FACE PARTIAL / REPOSITION",
                color_bgr=RED,
                alert_active=True,
                alert_reason="partial face",
                session_name=self.session_name,
                session_log=str(self.logger.path),
            )

        normalized, pixels = point_array(face_landmarks, self.frame_width, self.frame_height)

        ear = self.temporal_state.smooth("ear", compute_avg_ear(pixels))
        mar = self.temporal_state.smooth("mar", compute_mar(pixels))
        pitch, yaw, roll = estimate_head_pose(normalized, pixels, self.frame_width, self.frame_height)
        pitch = self.temporal_state.smooth("pitch", pitch)
        yaw = self.temporal_state.smooth("yaw", yaw)
        roll = self.temporal_state.smooth("roll", roll)

        gaze_horizontal, gaze_vertical, gaze_deviation, gaze_center_score = compute_gaze_metrics(pixels)
        gaze_horizontal = self.temporal_state.smooth("gaze_horizontal", gaze_horizontal)
        gaze_vertical = self.temporal_state.smooth("gaze_vertical", gaze_vertical)
        gaze_deviation = self.temporal_state.smooth("gaze_deviation", gaze_deviation)
        gaze_center_score = self.temporal_state.smooth("gaze_center_score", gaze_center_score)

        blendshape_scores = category_scores(result.face_blendshapes[0]) if result.face_blendshapes else {}
        engagement = analyze_blendshapes(blendshape_scores)
        engagement.score = self.temporal_state.smooth("engagement", engagement.score)

        closure_duration = self.temporal_state.update_eye_duration(ear, timestamp_s)
        yawn_duration = self.temporal_state.update_yawn_duration(mar, timestamp_s)
        instantaneous_looking_away = (
            abs(yaw) > HEAD_YAW_LIMIT_DEG
            or abs(pitch) > HEAD_PITCH_LIMIT_DEG
            or abs(gaze_horizontal) > GAZE_HORIZONTAL_LIMIT
            or abs(gaze_vertical) > GAZE_VERTICAL_LIMIT
        )
        look_away_duration = self.temporal_state.update_look_away_duration(instantaneous_looking_away, timestamp_s)

        drowsy_active = closure_duration >= DROWSY_EYES_CLOSED_SECONDS
        yawn_active = yawn_duration >= YAWN_SECONDS
        look_away_active = look_away_duration >= LOOK_AWAY_SECONDS

        score_value, subscores = attention_score(
            face_present=True,
            ear=ear,
            mar=mar,
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            gaze_center_score=gaze_center_score,
            engagement_score_value=engagement.score,
            drowsy_active=drowsy_active,
            yawn_active=yawn_active,
            look_away_active=look_away_active,
        )
        score_value = self.temporal_state.smooth("attention_score", score_value, alpha=0.25)

        status, color_bgr, alert_active = status_for_analysis(
            face_present=True,
            score_value=score_value,
            drowsy_active=drowsy_active,
            yawn_active=yawn_active,
            look_away_active=look_away_active,
            engagement_label=engagement.label,
        )

        alert_reason = status.lower().replace(" / ", " ")
        return FrameAnalysis(
            timestamp_iso=now_iso(),
            frame_timestamp_ms=timestamp_ms,
            face_present=True,
            attention_score=score_value,
            status=status,
            color_bgr=color_bgr,
            alert_active=alert_active,
            alert_reason=alert_reason,
            ear=ear,
            mar=mar,
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            gaze_horizontal=gaze_horizontal,
            gaze_vertical=gaze_vertical,
            gaze_deviation=gaze_deviation,
            gaze_center_score=gaze_center_score,
            engagement_score=engagement.score,
            engagement_label=engagement.label,
            blendshape_insights=engagement.insights,
            closure_duration=closure_duration,
            yawn_duration=yawn_duration,
            look_away_duration=look_away_duration,
            face_bbox=face_bounding_box(pixels, self.frame_width, self.frame_height),
            subscores=subscores,
            session_name=self.session_name,
            session_log=str(self.logger.path),
        )

    def get_latest_analysis(self) -> FrameAnalysis:
        with self.lock:
            return copy.deepcopy(self.latest_analysis)

    def maybe_write_state(self, analysis: FrameAnalysis, current_time: float) -> None:
        if current_time - self.last_state_write_at < STATE_WRITE_INTERVAL_SECONDS:
            return
        payload = {
            "updated_at": analysis.timestamp_iso,
            "session_name": self.session_name,
            "session_log": str(self.logger.path),
            "privacy": "All processing is local. No cloud inference is used.",
            "analysis": analysis.to_state_payload(),
        }
        temp_path = LATEST_STATE_PATH.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(temp_path, LATEST_STATE_PATH)
        self.last_state_write_at = current_time

    def update_fps(self, loop_started_at: float, loop_ended_at: float) -> None:
        frame_duration = max(loop_ended_at - loop_started_at, 1e-6)
        current_fps = 1.0 / frame_duration
        if self.capture_fps == 0.0:
            self.capture_fps = current_fps
        else:
            self.capture_fps = 0.88 * self.capture_fps + 0.12 * current_fps

    def run(self) -> int:
        if not self.args.model.exists():
            print(
                f"Model not found at {self.args.model}. Download the latest float16 Face Landmarker task bundle first.",
                flush=True,
            )
            return 1

        capture = cv2.VideoCapture(self.args.camera_index)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not capture.isOpened():
            print("Could not open the webcam. Check permissions and camera index.", flush=True)
            return 1

        try:
            with self.build_landmarker() as landmarker:
                processed_frames = 0
                while True:
                    loop_started_at = time.perf_counter()
                    success, frame = capture.read()
                    if not success:
                        print("Failed to read a webcam frame.", flush=True)
                        break

                    frame = cv2.flip(frame, 1)
                    frame = cv2.resize(frame, (self.args.width, self.args.height), interpolation=cv2.INTER_LINEAR)
                    self.frame_height, self.frame_width = frame.shape[:2]

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    timestamp_ms = int(time.perf_counter() * 1000)
                    landmarker.detect_async(mp_image, timestamp_ms)

                    analysis = self.get_latest_analysis()
                    analysis.fps = self.capture_fps
                    analysis.session_log = str(self.logger.path)

                    if analysis.alert_active:
                        self.alert_player.maybe_alert(analysis.alert_reason)

                    current_time = time.perf_counter()
                    self.logger.maybe_write(analysis, current_time)
                    self.maybe_write_state(analysis, current_time)

                    if not self.args.no_display:
                        draw_overlay(frame, analysis)
                        cv2.imshow("Student Attentive Alertness System", frame)

                    loop_ended_at = time.perf_counter()
                    self.update_fps(loop_started_at, loop_ended_at)
                    processed_frames += 1

                    if self.args.max_frames > 0 and processed_frames >= self.args.max_frames:
                        print(f"Reached max frame limit: {self.args.max_frames}", flush=True)
                        break

                    if not self.args.no_display:
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            break
        finally:
            capture.release()
            self.logger.close()
            cv2.destroyAllWindows()

        return 0


def run_self_test(args: argparse.Namespace) -> int:
    print("Running student alertness self-test...")
    ensure_runtime_dirs()

    if not args.model.exists():
        print(f"[FAIL] Model file not found: {args.model}")
        return 1
    print(f"[OK] Model file found: {args.model}")

    try:
        monitor = AttentionMonitor(args)
        with monitor.build_landmarker() as landmarker:
            print(f"[OK] Face Landmarker initialized: {type(landmarker).__name__}")
        monitor.logger.close()
    except Exception as exc:
        print(f"[FAIL] Face Landmarker initialization failed: {exc}")
        return 1

    if args.camera_check:
        capture = cv2.VideoCapture(args.camera_index)
        try:
            if not capture.isOpened():
                print(f"[FAIL] Webcam index {args.camera_index} could not be opened")
                return 1
            success, frame = capture.read()
            if not success or frame is None:
                print(f"[FAIL] Webcam index {args.camera_index} opened but no frame was read")
                return 1
            print(f"[OK] Webcam index {args.camera_index} produced a frame with shape {frame.shape}")
        finally:
            capture.release()

    print("[OK] Self-test completed successfully")
    return 0


def panel_lines(analysis: FrameAnalysis) -> list[str]:
    blendshape_text = ", ".join(analysis.blendshape_insights[:2]) if analysis.blendshape_insights else "stable / neutral"
    return [
        f"Status: {analysis.status}",
        f"Attention: {analysis.attention_score:05.1f}%    FPS: {analysis.fps:04.1f}",
        f"EAR: {analysis.ear:.3f}    Closed: {analysis.closure_duration:.1f}s",
        f"MAR: {analysis.mar:.3f}    Yawn: {analysis.yawn_duration:.1f}s",
        f"Head pitch/yaw/roll: {analysis.pitch:+05.1f} / {analysis.yaw:+05.1f} / {analysis.roll:+05.1f}",
        f"Gaze offset H/V: {analysis.gaze_horizontal:+.3f} / {analysis.gaze_vertical:+.3f}",
        f"Blendshapes: {analysis.engagement_label} | {blendshape_text}",
    ]


def draw_attention_bar(frame: np.ndarray, score_value: float, color_bgr: tuple[int, int, int]) -> None:
    bar_x, bar_y, bar_w, bar_h = 20, frame.shape[0] - 38, 300, 16
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), WHITE, 1)
    fill_w = int((clamp(score_value, 0.0, 100.0) / 100.0) * (bar_w - 2))
    cv2.rectangle(frame, (bar_x + 1, bar_y + 1), (bar_x + fill_w, bar_y + bar_h - 1), color_bgr, -1)
    cv2.putText(
        frame,
        "Attention Score",
        (bar_x, bar_y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        WHITE,
        1,
        cv2.LINE_AA,
    )


def draw_overlay(frame: np.ndarray, analysis: FrameAnalysis) -> None:
    color = analysis.color_bgr
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (3, 3), (width - 4, height - 4), color, 3)

    if analysis.face_present and len(analysis.face_bbox) == 4:
        x1, y1, x2, y2 = analysis.face_bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    overlay = frame.copy()
    cv2.rectangle(overlay, (14, 14), (520, 210), BLACK, -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0.0, frame)

    for index, line in enumerate(panel_lines(analysis)):
        cv2.putText(
            frame,
            line,
            (24, 40 + index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            WHITE,
            1,
            cv2.LINE_AA,
        )

    draw_attention_bar(frame, analysis.attention_score, color)

    if analysis.alert_active:
        flash = int(time.time() * 2) % 2 == 0
        if flash:
            cv2.putText(
                frame,
                "ALERT",
                (width - 140, 42),
                cv2.FONT_HERSHEY_DUPLEX,
                1.0,
                color,
                2,
                cv2.LINE_AA,
            )

    cv2.putText(
        frame,
        "Local-only processing | Press q to quit",
        (width - 300, height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        WHITE,
        1,
        cv2.LINE_AA,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Student Attentive Alertness System")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Path to face_landmarker.task")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index")
    parser.add_argument("--width", type=int, default=960, help="Capture width")
    parser.add_argument("--height", type=int, default=540, help="Capture height")
    parser.add_argument(
        "--session-name",
        default="student_session",
        help="Session label used in CSV logging and the Streamlit dashboard",
    )
    parser.add_argument(
        "--log-interval-sec",
        type=float,
        default=DEFAULT_LOG_INTERVAL_SECONDS,
        help="Minimum delay between CSV log rows unless status changes",
    )
    parser.add_argument(
        "--no-audio-alerts",
        action="store_true",
        help="Disable local audio alerts",
    )
    parser.add_argument(
        "--alert-sound",
        type=Path,
        default=None,
        help="Optional audio file played when alerts fire",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Verify the model, MediaPipe initialization, and optionally the webcam, then exit",
    )
    parser.add_argument(
        "--camera-check",
        action="store_true",
        help="When used with --self-test, also open the webcam and read one frame",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without opening the OpenCV preview window",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional frame limit for automated runs; 0 means unlimited",
    )
    return parser


def main() -> int:
    ensure_runtime_dirs()
    args = build_arg_parser().parse_args()
    if args.self_test:
        return run_self_test(args)
    monitor = AttentionMonitor(args)
    return monitor.run()


if __name__ == "__main__":
    raise SystemExit(main())