#!/usr/bin/env python3
"""Unified launcher for the VigilEye student alertness system.

This script starts the detector and Streamlit dashboard together using the best
available project-local Python environment. It also supports a built-in
preflight check so users can confirm that the model and camera are available
before the live session starts.
"""

from __future__ import annotations

import argparse
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "face_landmarker.task"


def quote_command(command: list[str]) -> str:
    return shlex.join(command)


def detect_python_interpreter(explicit_python: Path | None) -> Path:
    if explicit_python is not None:
        return explicit_python.resolve()

    candidates = [
        PROJECT_ROOT / ".py312" / "bin" / "python",
        PROJECT_ROOT / ".venv" / "bin" / "python",
        Path(sys.executable),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError("No usable Python interpreter was found for the project")


def build_self_test_command(python_executable: Path, args: argparse.Namespace) -> list[str]:
    command = [
        str(python_executable),
        "student_alertness.py",
        "--self-test",
        "--model",
        str(args.model),
    ]
    if args.camera_check:
        command.append("--camera-check")
    return command


def build_detector_command(python_executable: Path, args: argparse.Namespace) -> list[str]:
    command = [
        str(python_executable),
        "student_alertness.py",
        "--model",
        str(args.model),
        "--camera-index",
        str(args.camera_index),
        "--session-name",
        args.session_name,
        "--width",
        str(args.width),
        "--height",
        str(args.height),
    ]
    if args.no_audio_alerts:
        command.append("--no-audio-alerts")
    if args.alert_sound is not None:
        command.extend(["--alert-sound", str(args.alert_sound)])
    if args.headless_detector:
        command.append("--no-display")
    if args.detector_max_frames > 0:
        command.extend(["--max-frames", str(args.detector_max_frames)])
    return command


def build_dashboard_command(python_executable: Path, args: argparse.Namespace) -> list[str]:
    command = [
        str(python_executable),
        "-m",
        "streamlit",
        "run",
        "dashboard.py",
        "--browser.gatherUsageStats",
        "false",
        "--server.port",
        str(args.dashboard_port),
        "--server.headless",
        "true" if args.dashboard_headless else "false",
    ]
    return command


def terminate_process(process: subprocess.Popen[str] | None, name: str) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    print(f"Stopped {name}.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch VigilEye detector and dashboard together")
    parser.add_argument("--python", type=Path, default=None, help="Optional explicit Python interpreter to use")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Path to face_landmarker.task")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index for the detector")
    parser.add_argument("--session-name", default="class_session", help="Session label used in logs and dashboard")
    parser.add_argument("--width", type=int, default=960, help="Detector capture width")
    parser.add_argument("--height", type=int, default=540, help="Detector capture height")
    parser.add_argument("--dashboard-port", type=int, default=8501, help="Streamlit dashboard port")
    parser.add_argument("--skip-self-test", action="store_true", help="Skip the detector preflight check")
    parser.add_argument("--camera-check", action="store_true", help="Include webcam probing in the preflight check")
    parser.add_argument("--detector-only", action="store_true", help="Run only the detector")
    parser.add_argument("--dashboard-only", action="store_true", help="Run only the dashboard")
    parser.add_argument("--dashboard-headless", action="store_true", help="Run Streamlit without opening a browser")
    parser.add_argument("--headless-detector", action="store_true", help="Run detector without opening the OpenCV preview window")
    parser.add_argument("--detector-max-frames", type=int, default=0, help="Optional detector frame limit for automated runs")
    parser.add_argument("--no-audio-alerts", action="store_true", help="Disable detector audio alerts")
    parser.add_argument("--alert-sound", type=Path, default=None, help="Optional alert sound file for the detector")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.detector_only and args.dashboard_only:
        print("Choose only one of --detector-only or --dashboard-only.")
        return 2

    python_executable = detect_python_interpreter(args.python)
    print(f"Using Python interpreter: {python_executable}")

    if not args.skip_self_test and not args.dashboard_only:
        self_test_command = build_self_test_command(python_executable, args)
        print(f"Running preflight: {quote_command(self_test_command)}")
        result = subprocess.run(self_test_command, cwd=PROJECT_ROOT)
        if result.returncode != 0:
            print("Preflight failed. Aborting launcher.")
            return result.returncode

    dashboard_process: subprocess.Popen[str] | None = None
    detector_process: subprocess.Popen[str] | None = None
    try:
        if not args.detector_only:
            dashboard_command = build_dashboard_command(python_executable, args)
            print(f"Starting dashboard: {quote_command(dashboard_command)}")
            dashboard_process = subprocess.Popen(dashboard_command, cwd=PROJECT_ROOT)
            time.sleep(2.0)

        if not args.dashboard_only:
            detector_command = build_detector_command(python_executable, args)
            print(f"Starting detector: {quote_command(detector_command)}")
            detector_process = subprocess.Popen(detector_command, cwd=PROJECT_ROOT)
            detector_return = detector_process.wait()
            if not args.detector_only:
                terminate_process(dashboard_process, "dashboard")
            return detector_return

        assert dashboard_process is not None
        print("Dashboard is running. Press Ctrl+C to stop it.")
        dashboard_process.wait()
        return dashboard_process.returncode or 0
    except KeyboardInterrupt:
        print("Interrupted. Shutting down VigilEye launcher.")
        terminate_process(detector_process, "detector")
        terminate_process(dashboard_process, "dashboard")
        return 130
    finally:
        if args.dashboard_only:
            terminate_process(dashboard_process, "dashboard")


if __name__ == "__main__":
    raise SystemExit(main())