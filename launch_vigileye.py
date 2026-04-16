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
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "face_landmarker.task"

# ANSI color helpers (graceful fallback on terminals that don't support them)
_USE_COLOR = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

def ok(msg: str) -> None:    print(_c("32", f"  ✔  {msg}"), flush=True)
def info(msg: str) -> None:  print(_c("36", f"  ●  {msg}"), flush=True)
def warn(msg: str) -> None:  print(_c("33", f"  ⚠  {msg}"), flush=True)
def err(msg: str) -> None:   print(_c("31", f"  ✘  {msg}"), flush=True)
def bold(msg: str) -> str:   return _c("1", msg)


def print_banner() -> None:
    print()
    print(_c("1;36", "  VigilEye — Student Attentiveness System"))
    print(_c("90",   "  Real-time attention monitoring  |  Local only"))
    print()


def quote_command(command: list[str]) -> str:
    return shlex.join(command)


def ensure_streamlit_credentials() -> None:
    """Pre-answer Streamlit's first-run email prompt so it never blocks the launcher."""
    credentials_path = Path.home() / ".streamlit" / "credentials.toml"
    if not credentials_path.exists():
        credentials_path.parent.mkdir(parents=True, exist_ok=True)
        credentials_path.write_text('[general]\nemail = ""\n', encoding="utf-8")


def cleanup_old_logs(retain_days: int = 14) -> None:
    """Automatically purge log CSVs older than retain_days to prevent disk bloat."""
    log_dir = PROJECT_ROOT / "logs"
    if not log_dir.exists():
        return
    now = time.time()
    purged = 0
    for log_file in log_dir.glob("*.csv"):
        if log_file.is_file():
            age_days = (now - log_file.stat().st_mtime) / 86400.0
            if age_days > retain_days:
                try:
                    log_file.unlink()
                    purged += 1
                except Exception:
                    pass
    if purged:
        info(f"Purged {purged} log file(s) older than {retain_days} days.")


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


def check_model(model_path: Path) -> bool:
    """Returns True if the model file exists; prints actionable help if not."""
    if model_path.exists():
        return True
    err(f"Model file not found: {model_path}")
    print()
    print(_c("33", "  Download it with:"))
    print(_c("97", f"    mkdir -p models && curl -L \\"))
    print(_c("97",  "      https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task \\"))
    print(_c("97", f"      -o {model_path}"))
    print()
    return False


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
    return [
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


def terminate_process(process: subprocess.Popen[str] | None, name: str) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    info(f"Stopped {name}.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch VigilEye detector and dashboard together",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  Launch everything (recommended):
    ./.py312/bin/python launch_vigileye.py --camera-check

  Detector only (no dashboard):
    ./.py312/bin/python launch_vigileye.py --detector-only

  Dashboard only (review past logs):
    ./.py312/bin/python launch_vigileye.py --dashboard-only --dashboard-headless

  Headless smoke test (30 frames):
    ./.py312/bin/python launch_vigileye.py --headless-detector --detector-max-frames 30 --dashboard-headless
        """,
    )
    parser.add_argument("--python", type=Path, default=None, help="Optional explicit Python interpreter to use")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Path to face_landmarker.task")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index for the detector (default: 0)")
    parser.add_argument("--session-name", default="class_session", help="Session label used in logs and dashboard")
    parser.add_argument("--width", type=int, default=960, help="Detector capture width")
    parser.add_argument("--height", type=int, default=540, help="Detector capture height")
    parser.add_argument("--dashboard-port", type=int, default=8501, help="Streamlit dashboard port (default: 8501)")
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

    print_banner()

    if args.detector_only and args.dashboard_only:
        err("Choose only one of --detector-only or --dashboard-only.")
        return 2

    cleanup_old_logs(retain_days=14)

    try:
        python_executable = detect_python_interpreter(args.python)
    except FileNotFoundError as exc:
        err(str(exc))
        return 1
    ok(f"Python interpreter: {python_executable}")

    if not args.dashboard_only and not check_model(args.model):
        return 1

    if not args.skip_self_test and not args.dashboard_only:
        info("Running preflight check...")
        self_test_command = build_self_test_command(python_executable, args)
        result = subprocess.run(self_test_command, cwd=PROJECT_ROOT)
        if result.returncode != 0:
            err("Preflight failed. Fix the issues above before starting a session.")
            return result.returncode
        ok("Preflight passed.")

    print()

    dashboard_process: subprocess.Popen[str] | None = None
    detector_process: subprocess.Popen[str] | None = None
    try:
        if not args.detector_only:
            ensure_streamlit_credentials()
            dashboard_command = build_dashboard_command(python_executable, args)
            dashboard_process = subprocess.Popen(
                dashboard_command,
                cwd=PROJECT_ROOT,
                stdin=subprocess.DEVNULL,
            )
            info("Dashboard starting...")
            time.sleep(2.0)
            ok(bold(f"Dashboard ready → http://localhost:{args.dashboard_port}"))
            print()

        if not args.dashboard_only:
            info(f"Starting detector  (session: {args.session_name}, camera: {args.camera_index})")
            if not args.headless_detector:
                info("OpenCV preview window will open. Press  q  to quit.")
            print()
            detector_command = build_detector_command(python_executable, args)
            detector_process = subprocess.Popen(detector_command, cwd=PROJECT_ROOT)
            detector_return = detector_process.wait()
            if not args.detector_only:
                terminate_process(dashboard_process, "dashboard")
            return detector_return

        assert dashboard_process is not None
        ok(f"Dashboard is running at http://localhost:{args.dashboard_port}")
        info("Press Ctrl+C to stop.")
        dashboard_process.wait()
        return dashboard_process.returncode or 0

    except KeyboardInterrupt:
        print()
        info("Shutting down VigilEye...")
        terminate_process(detector_process, "detector")
        terminate_process(dashboard_process, "dashboard")
        ok("Goodbye.")
        return 130
    finally:
        if args.dashboard_only:
            terminate_process(dashboard_process, "dashboard")


if __name__ == "__main__":
    raise SystemExit(main())