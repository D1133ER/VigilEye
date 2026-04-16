from config_loader import *
from vision_utils import *
import numpy as np
import cv2
import time
import threading
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
import csv
import queue
import os
from typing import Any, Optional
try:
    import pygame
except:
    pygame = None

from enum import Enum, auto

class CalibrationStatus(Enum):
    IDLE = auto()
    CALIBRATING = auto()
    CALIBRATED = auto()


class FaceTracker:
    """IoU-based face tracker that assigns stable IDs across frames.

    MediaPipe may reorder face detections between frames. This tracker matches
    new detections to existing tracks using bounding-box IoU and assigns
    persistent integer IDs so that temporal state stays with the correct person.
    """

    def __init__(self, iou_threshold: float = 0.3, max_missing_frames: int = 15) -> None:
        self.iou_threshold = iou_threshold
        self.max_missing_frames = max_missing_frames
        self.next_id: int = 0
        self.tracks: dict[int, dict[str, Any]] = {}  # {id: {"bbox": [...], "missing": int}}
        self._recently_pruned: list[int] = []

    @staticmethod
    def _iou(a: list[int], b: list[int]) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0
        area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def update(self, bboxes: list[list[int]]) -> list[int]:
        """Match detections to tracks and return a stable ID for each detection."""
        self._recently_pruned = []

        # Build scored pairs: (track_id, det_idx, iou)
        pairs: list[tuple[int, int, float]] = []
        for tid, track in self.tracks.items():
            for didx, bbox in enumerate(bboxes):
                iou = self._iou(track["bbox"], bbox)
                if iou >= self.iou_threshold:
                    pairs.append((tid, didx, iou))

        # Greedy match: highest IoU first
        pairs.sort(key=lambda p: p[2], reverse=True)
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        det_to_track: dict[int, int] = {}

        for tid, didx, _ in pairs:
            if tid in matched_tracks or didx in matched_dets:
                continue
            det_to_track[didx] = tid
            matched_tracks.add(tid)
            matched_dets.add(didx)

        # Update matched tracks
        for didx, tid in det_to_track.items():
            self.tracks[tid]["bbox"] = bboxes[didx]
            self.tracks[tid]["missing"] = 0

        # Increment missing counter for unmatched existing tracks
        for tid in list(self.tracks):
            if tid not in matched_tracks:
                self.tracks[tid]["missing"] += 1

        # Prune stale tracks
        for tid in list(self.tracks):
            if self.tracks[tid]["missing"] > self.max_missing_frames:
                del self.tracks[tid]
                self._recently_pruned.append(tid)

        # Create new tracks for unmatched detections (after aging, so they start at 0)
        for didx in range(len(bboxes)):
            if didx not in matched_dets:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"bbox": bboxes[didx], "missing": 0}
                det_to_track[didx] = tid

        # Return stable IDs in detection order
        return [det_to_track[i] for i in range(len(bboxes))]

    def pruned_ids(self) -> list[int]:
        """IDs that were pruned in the most recent update() call."""
        return self._recently_pruned

@dataclass
class CalibrationBaselines:
    ear: float = 0.0
    mar: float = 0.0
    gaze_h_offset: float = 0.0
    gaze_v_offset: float = 0.0
    samples_count: int = 0

@dataclass
class FrameAnalysis:
    student_id: int = 0
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
    calibration_progress: float = 0.0
    calibration_message: str = ""
    session_name: str = "student_session"
    session_log: str = ""

    def to_state_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["color_bgr"] = list(self.color_bgr)
        return payload


from collections import deque

class TemporalState:
    def __init__(self) -> None:
        self.eye_closed_since: Optional[float] = None
        self.yawn_since: Optional[float] = None
        self.look_away_since: Optional[float] = None
        self.filtered_values: dict[str, float] = {}

        # For Sustained Distraction Buffer
        self.look_away_window: deque[bool] = deque(maxlen=90) # ~3s at 30fps
        self.calibration_status = CalibrationStatus.IDLE
        self.baselines = CalibrationBaselines()
        self.calibration_samples: list[dict] = []
        self.calibrated_at: Optional[float] = None
        self.calibration_failed_at: Optional[float] = None

    def smooth(self, key: str, value: float, alpha: float = 0.35) -> float:
        previous = self.filtered_values.get(key)
        if previous is None:
            self.filtered_values[key] = value
        else:
            self.filtered_values[key] = previous + alpha * (value - previous)
        return self.filtered_values[key]

    def update_look_away_buffer(self, looking_away: bool) -> bool:
        self.look_away_window.append(looking_away)
        if len(self.look_away_window) < 30: # Minimum samples before deciding
            return looking_away

        # Sustained distraction: True if >= 70% of window is looking away
        sustained = (sum(self.look_away_window) / len(self.look_away_window)) >= 0.7
        return sustained

    def update_eye_duration(self, ear: float, timestamp_s: float, threshold: float = EAR_CLOSED_THRESHOLD) -> float:
        if ear < threshold:
            if self.eye_closed_since is None:
                self.eye_closed_since = timestamp_s
        elif ear > EAR_RECOVERY_THRESHOLD:
            self.eye_closed_since = None
        return 0.0 if self.eye_closed_since is None else timestamp_s - self.eye_closed_since

    def update_yawn_duration(self, mar: float, timestamp_s: float, threshold: float = MAR_YAWN_THRESHOLD) -> float:
        if mar > threshold:
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
        self.look_away_window.clear()


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
        self.last_write_at_per_student: dict[int, float] = {}
        self.last_status_per_student: dict[int, str] = {}
        self.handle = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.handle,
            fieldnames=[
                "timestamp",
                "student_id",
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

    def maybe_write(self, analyses: list[FrameAnalysis], current_time: float) -> None:
        if not analyses:
            return

        # For CSV logging, we log each student individually
        for analysis in analyses:
            sid = analysis.student_id
            last_write = self.last_write_at_per_student.get(sid, 0.0)
            last_status = self.last_status_per_student.get(sid, "")
            if (
                current_time - last_write < self.log_interval_seconds
                and analysis.status == last_status
            ):
                continue

            self.writer.writerow(
                {
                    "timestamp": analysis.timestamp_iso,
                    "student_id": analysis.student_id,
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
            self.last_write_at_per_student[sid] = current_time
            self.last_status_per_student[sid] = analysis.status

    def close(self) -> None:
        self.handle.close()


