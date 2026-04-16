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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# These thresholds are intentionally defined near the top because classroom,
# webcam, and seating setups vary. In 2026 practice, keeping them explicit and
# easy to tune is more maintainable than burying them deep inside the pipeline.
from config_loader import *


from vision_utils import *
from core import *
class AttentionMonitor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.frame_width = args.width
        self.frame_height = args.height
        self.session_name = args.session_name.strip().replace(" ", "_") or "student_session"
        self.student_states: dict[int, TemporalState] = {}
        self.face_tracker = FaceTracker()
        self.lock = threading.Lock()
        self.latest_analyses: list[FrameAnalysis] = []
        self.capture_fps = 0.0
        self.logger = SessionLogger(self.session_name, args.log_interval_sec)
        self.alert_player = AlertPlayer(enabled=not args.no_audio_alerts, sound_path=args.alert_sound)
        self.last_state_write_at = 0.0

    def build_landmarker(self) -> Any:
        base_options = mp.tasks.BaseOptions(model_asset_path=str(self.args.model))
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_faces=5,
            min_face_detection_confidence=MIN_FACE_DETECTION_CONFIDENCE,
            min_face_presence_confidence=MIN_FACE_PRESENCE_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            result_callback=self.on_result,
        )
        return mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def on_result(self, result: Any, _output_image: Any, timestamp_ms: int) -> None:
        analyses = self._analyze_result(result, timestamp_ms)
        with self.lock:
            self.latest_analyses = analyses

    def _analyze_result(self, result: Any, timestamp_ms: int) -> list[FrameAnalysis]:
        timestamp_s = timestamp_ms / 1000.0
        analyses = []

        if not result.face_landmarks:
            # Age all tracks; stale ones will be pruned automatically
            self.face_tracker.update([])
            with self.lock:
                for stale_id in self.face_tracker.pruned_ids():
                    self.student_states.pop(stale_id, None)
            return []

        # Compute bboxes for all detections first, then assign stable IDs
        detection_bboxes: list[list[int]] = []
        for face_landmarks in result.face_landmarks:
            if len(face_landmarks) >= MIN_REQUIRED_LANDMARKS:
                _, pixels = point_array(face_landmarks, self.frame_width, self.frame_height)
                detection_bboxes.append(face_bounding_box(pixels, self.frame_width, self.frame_height))
            else:
                detection_bboxes.append([0, 0, 0, 0])

        stable_ids = self.face_tracker.update(detection_bboxes)

        with self.lock:
            for stale_id in self.face_tracker.pruned_ids():
                self.student_states.pop(stale_id, None)
            for sid in stable_ids:
                if sid not in self.student_states:
                    self.student_states[sid] = TemporalState()
            local_states = {sid: self.student_states[sid] for sid in stable_ids}

        for idx, face_landmarks in enumerate(result.face_landmarks):
            student_id = stable_ids[idx]
            state = local_states[student_id]

            if len(face_landmarks) < MIN_REQUIRED_LANDMARKS:
                analyses.append(FrameAnalysis(
                    student_id=student_id,
                    timestamp_iso=now_iso(),
                    frame_timestamp_ms=timestamp_ms,
                    face_present=False,
                    attention_score=0.0,
                    status="FACE PARTIAL / REPOSITION",
                    color_bgr=RED,
                    alert_active=True,
                    alert_reason="partial face",
                    fps=self.capture_fps,
                    session_name=self.session_name,
                    session_log=str(self.logger.path),
                ))
                continue

            normalized, pixels = point_array(face_landmarks, self.frame_width, self.frame_height)

            # Calibration Handling
            if state.calibration_status == CalibrationStatus.IDLE:
                state.calibration_status = CalibrationStatus.CALIBRATING

            # Sample for calibration if in CALIBRATING state
            if state.calibration_status == CalibrationStatus.CALIBRATING:
                ear_raw = compute_avg_ear(pixels)
                mar_raw = compute_mar(pixels)
                h_off, v_off, _, gaze_center = compute_gaze_metrics(pixels)
                state.calibration_samples.append({
                    "ear": ear_raw, "mar": mar_raw, "h": h_off, "v": v_off, "score": gaze_center
                })

                if len(state.calibration_samples) >= 150: # ~5s at 30fps
                    # Confidence check: 80% frames must have good gaze center
                    valid_frames = sum(1 for s in state.calibration_samples if s["score"] > 0.7)
                    if valid_frames / len(state.calibration_samples) >= 0.8:
                        # Finalize baselines
                        samples = state.calibration_samples
                        state.baselines.ear = np.mean([s["ear"] for s in samples])
                        state.baselines.mar = np.mean([s["mar"] for s in samples])
                        state.baselines.gaze_h_offset = np.mean([s["h"] for s in samples])
                        state.baselines.gaze_v_offset = np.mean([s["v"] for s in samples])
                        state.baselines.samples_count = len(samples)
                        state.calibration_status = CalibrationStatus.CALIBRATED
                        state.calibrated_at = timestamp_s
                        state.calibration_samples.clear()
                    else:
                        # Reset calibration if confidence low
                        state.calibration_failed_at = timestamp_s
                        state.calibration_samples.clear()

            # Determine dynamic thresholds
            calibrated = state.calibration_status == CalibrationStatus.CALIBRATED
            ear_thresh = state.baselines.ear * 0.7 if calibrated else EAR_CLOSED_THRESHOLD
            ear_open_ref = state.baselines.ear if calibrated else EAR_OPEN_REFERENCE
            mar_thresh = state.baselines.mar * 1.5 if calibrated else MAR_YAWN_THRESHOLD
            _mar_scale = mar_thresh / MAR_YAWN_THRESHOLD
            mar_recovery = MAR_RECOVERY_THRESHOLD * _mar_scale
            mar_wide_ref = MAR_WIDE_OPEN_REFERENCE * _mar_scale

            ear = state.smooth("ear", compute_avg_ear(pixels))
            mar = state.smooth("mar", compute_mar(pixels))
            pitch, yaw, roll = estimate_head_pose(normalized, pixels, self.frame_width, self.frame_height)
            pitch = state.smooth("pitch", pitch)
            yaw = state.smooth("yaw", yaw)
            roll = state.smooth("roll", roll)

            gaze_horizontal, gaze_vertical, gaze_deviation, gaze_center_score = compute_gaze_metrics(pixels)
            gaze_horizontal = state.smooth("gaze_horizontal", gaze_horizontal)
            gaze_vertical = state.smooth("gaze_vertical", gaze_vertical)
            gaze_deviation = state.smooth("gaze_deviation", gaze_deviation)
            gaze_center_score = state.smooth("gaze_center_score", gaze_center_score)

            blendshape_scores = category_scores(result.face_blendshapes[idx]) if result.face_blendshapes else {}
            engagement = analyze_blendshapes(blendshape_scores)
            engagement.score = state.smooth("engagement", engagement.score)

            closure_duration = state.update_eye_duration(ear, timestamp_s, threshold=ear_thresh)
            yawn_duration = state.update_yawn_duration(mar, timestamp_s, threshold=mar_thresh)

            # Instantaneous looking away check
            instantaneous_looking_away = (
                abs(yaw) > HEAD_YAW_LIMIT_DEG
                or abs(pitch) > HEAD_PITCH_LIMIT_DEG
                or abs(gaze_horizontal) > GAZE_HORIZONTAL_LIMIT
                or abs(gaze_vertical) > GAZE_VERTICAL_LIMIT
            )

            # Use the new sustained distraction buffer
            look_away_active_sustained = state.update_look_away_buffer(instantaneous_looking_away)
            look_away_duration = state.update_look_away_duration(instantaneous_looking_away, timestamp_s)

            drowsy_active = closure_duration >= DROWSY_EYES_CLOSED_SECONDS
            yawn_active = yawn_duration >= YAWN_SECONDS
            look_away_active = look_away_active_sustained

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
                ear_closed_threshold=ear_thresh,
                ear_open_reference=ear_open_ref,
                mar_recovery_threshold=mar_recovery,
                mar_wide_open_reference=mar_wide_ref,
            )
            score_value = state.smooth("attention_score", score_value, alpha=0.25)

            status, color_bgr, alert_active = status_for_analysis(
                face_present=True,
                score_value=score_value,
                drowsy_active=drowsy_active,
                yawn_active=yawn_active,
                look_away_active=look_away_active,
                engagement_label=engagement.label,
            )

            if state.calibration_status == CalibrationStatus.CALIBRATING:
                status = "CALIBRATING..."
                color_bgr = (0, 255, 255) # Yellow

            # Compute calibration UX fields
            cal_progress = 0.0
            cal_message = ""
            if state.calibration_status == CalibrationStatus.CALIBRATING:
                cal_progress = len(state.calibration_samples) / 150.0
                remaining_s = max(0, int((150 - len(state.calibration_samples)) / 30))
                cal_message = f"Look at your screen ({remaining_s}s)"
            elif state.calibrated_at and timestamp_s - state.calibrated_at < 1.5:
                cal_progress = 1.0
                cal_message = "Calibrated!"
            elif state.calibration_failed_at and timestamp_s - state.calibration_failed_at < 2.0:
                cal_progress = 0.0
                cal_message = "Restarting - improve lighting"

            alert_reason = status.lower().replace(" / ", " ")
            analyses.append(FrameAnalysis(
                student_id=student_id,
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
                fps=self.capture_fps,
                calibration_progress=cal_progress,
                calibration_message=cal_message,
                session_name=self.session_name,
                session_log=str(self.logger.path),
            ))

        return analyses

    def get_latest_analyses(self) -> list[FrameAnalysis]:
        with self.lock:
            return copy.deepcopy(self.latest_analyses)

    def maybe_write_state(self, analyses: list[FrameAnalysis], current_time: float) -> None:
        if current_time - self.last_state_write_at < STATE_WRITE_INTERVAL_SECONDS:
            return

        # Restructure payload for multi-student support
        with self.lock:
            states_snap = {sid: st for sid, st in self.student_states.items()}
        students_data = {}
        for a in analyses:
            sid = str(a.student_id)
            state = states_snap.get(a.student_id)
            students_data[sid] = {
                "analysis": a.to_state_payload(),
                "calibration_status": state.calibration_status.name if state else "IDLE"
            }

        payload = {
            "updated_at": now_iso(),
            "session_info": {
                "session_name": self.session_name,
                "session_log": str(self.logger.path),
                "student_count": len(analyses)
            },
            "students": students_data,
            "privacy": "All processing is local. No cloud inference is used.",
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

                    analyses = self.get_latest_analyses()

                    for analysis in analyses:
                        if analysis.alert_active:
                            self.alert_player.maybe_alert(analysis.alert_reason)

                    current_time = time.perf_counter()
                    self.logger.maybe_write(analyses, current_time)
                    self.maybe_write_state(analyses, current_time)

                    if not self.args.no_display:
                        # Draw overlays for each detected student
                        for analysis in analyses:
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
    pass # Moved into draw_overlay for per-student positioning


def draw_overlay(frame: np.ndarray, analysis: FrameAnalysis) -> None:
    color = analysis.color_bgr
    height, width = frame.shape[:2]

    PANEL_W, PANEL_H = 370, 200
    BAR_W, BAR_H = 280, 14

    # Draw student-specific bounding box
    if analysis.face_present and len(analysis.face_bbox) == 4:
        x1, y1, x2, y2 = analysis.face_bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID: {analysis.student_id}", (x1, max(y1 - 8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

        # Anchor panel to the right of the face; fall back to left if it overflows
        panel_x = x2 + 8
        if panel_x + PANEL_W > width:
            panel_x = max(x1 - PANEL_W - 8, 4)
        panel_y = max(min(y1, height - PANEL_H - 4), 4)
    else:
        # No bbox: stack panels from the top-left corner
        panel_x = 14
        panel_y = 14 + analysis.student_id * (PANEL_H + 6)
        if panel_y + PANEL_H > height:
            panel_y = 14

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + PANEL_W, panel_y + PANEL_H), BLACK, -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0.0, frame)

    for index, line in enumerate(panel_lines(analysis)):
        cv2.putText(
            frame,
            line,
            (panel_x + 10, panel_y + 26 + index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            WHITE,
            1,
            cv2.LINE_AA,
        )

    # Draw attention bar below the panel
    bar_x = panel_x
    bar_y = panel_y + PANEL_H + 4
    if bar_y + BAR_H > height:
        bar_y = panel_y - BAR_H - 4
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + BAR_W, bar_y + BAR_H), WHITE, 1)
    fill_w = int((clamp(analysis.attention_score, 0.0, 100.0) / 100.0) * (BAR_W - 2))
    cv2.rectangle(frame, (bar_x + 1, bar_y + 1), (bar_x + fill_w, bar_y + BAR_H - 1), color, -1)
    cv2.putText(frame, f"S{analysis.student_id} Score", (bar_x, bar_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)

    # Calibration overlay
    if analysis.calibration_message:
        if analysis.face_present and len(analysis.face_bbox) == 4:
            cx = (analysis.face_bbox[0] + analysis.face_bbox[2]) // 2
            cy = (analysis.face_bbox[1] + analysis.face_bbox[3]) // 2
        else:
            cx, cy = width // 2, height // 2

        prog = analysis.calibration_progress
        if "Calibrated!" in analysis.calibration_message:
            msg_color = (0, 220, 0)  # green
        elif "Restarting" in analysis.calibration_message:
            msg_color = (0, 0, 220)  # red
        else:
            msg_color = (0, 255, 255)  # yellow
            # Draw crosshair reticle
            r = 30
            cv2.circle(frame, (cx, cy), r, msg_color, 1, cv2.LINE_AA)
            cv2.line(frame, (cx - r - 6, cy), (cx - r + 6, cy), msg_color, 2)
            cv2.line(frame, (cx + r - 6, cy), (cx + r + 6, cy), msg_color, 2)
            cv2.line(frame, (cx, cy - r - 6), (cx, cy - r + 6), msg_color, 2)
            cv2.line(frame, (cx, cy + r - 6), (cx, cy + r + 6), msg_color, 2)
            # Progress bar below reticle
            pb_w, pb_h = 120, 8
            pb_x, pb_y = cx - pb_w // 2, cy + r + 12
            cv2.rectangle(frame, (pb_x, pb_y), (pb_x + pb_w, pb_y + pb_h), WHITE, 1)
            fill = int(prog * (pb_w - 2))
            cv2.rectangle(frame, (pb_x + 1, pb_y + 1), (pb_x + 1 + fill, pb_y + pb_h - 1), msg_color, -1)

        # Message text
        ts_size = cv2.getTextSize(analysis.calibration_message, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
        tx = cx - ts_size[0] // 2
        ty = cy + 60
        cv2.putText(frame, analysis.calibration_message, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, msg_color, 1, cv2.LINE_AA)

    if analysis.alert_active:
        flash = int(time.time() * 2) % 2 == 0
        if flash:
            cv2.putText(
                frame,
                "ALERT",
                (panel_x + PANEL_W - 80, panel_y + 30),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
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