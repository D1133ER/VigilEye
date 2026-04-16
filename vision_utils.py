from config_loader import *
import numpy as np
import cv2
import mediapipe as mp
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Any, Optional
import copy
import os
from pathlib import Path
import queue

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
    ear_closed_threshold: float = EAR_CLOSED_THRESHOLD,
    ear_open_reference: float = EAR_OPEN_REFERENCE,
    mar_recovery_threshold: float = MAR_RECOVERY_THRESHOLD,
    mar_wide_open_reference: float = MAR_WIDE_OPEN_REFERENCE,
) -> tuple[float, dict[str, float]]:
    if not face_present:
        return 0.0, {
            "eyes": 0.0,
            "mouth": 0.0,
            "head": 0.0,
            "gaze": 0.0,
            "engagement": 0.0,
        }

    eyes_score = score_from_range(ear, ear_closed_threshold, ear_open_reference)
    mouth_score = inverse_score_from_range(mar, mar_recovery_threshold, mar_wide_open_reference)
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

