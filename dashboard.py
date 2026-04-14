#!/usr/bin/env python3
"""Streamlit dashboard for the Student Attentive Alertness System.

The dashboard reads the runtime JSON state and CSV session logs produced by the
webcam detector. This keeps the system local, lightweight, and simple to deploy
without a separate backend service.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None


PROJECT_ROOT = Path(__file__).resolve().parent
RUNTIME_STATE_PATH = PROJECT_ROOT / "runtime" / "latest_state.json"
LOG_DIR = PROJECT_ROOT / "logs"

STATUS_COLORS = {
    "ATTENTIVE": "#1f9d55",
    "MODERATE": "#d97706",
    "MODERATE / CONFUSED": "#d97706",
    "LOW ATTENTION": "#dc2626",
    "LOW / BORED": "#dc2626",
    "LOOKING AWAY": "#dc2626",
    "YAWNING / FATIGUED": "#dc2626",
    "DROWSY": "#b91c1c",
    "NO FACE DETECTED": "#b91c1c",
    "FACE PARTIAL / REPOSITION": "#b91c1c",
    "INITIALIZING": "#334155",
}


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def discover_logs() -> list[Path]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(LOG_DIR.glob("*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)


def load_log(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    return frame


def score_color(score_value: float) -> str:
    if score_value >= 75:
        return "#1f9d55"
    if score_value >= 55:
        return "#d97706"
    return "#dc2626"


def guidance_from_state(analysis: dict[str, Any]) -> str:
    status = analysis.get("status", "UNKNOWN")
    attention = float(analysis.get("attention_score", 0.0))
    engagement_label = analysis.get("engagement_label", "neutral")

    if status == "NO FACE DETECTED":
        return "Learner is out of frame. Ask for camera realignment or confirm presence."
    if status == "DROWSY":
        return "Likely fatigue. Switch to a short interactive prompt or ask for a quick break."
    if status == "YAWNING / FATIGUED":
        return "Fatigue cue detected. Shorten the explanation block and ask a recall question."
    if status == "LOOKING AWAY":
        return "Attention shifted off-screen. Re-engage with a direct question or visual cue."
    if engagement_label == "confused":
        return "Facial tension suggests confusion. Slow down and restate the last key concept."
    if attention >= 75:
        return "Engagement is healthy. This is a good time to continue or add a slightly harder example."
    if attention >= 55:
        return "Moderate attention. Insert a brief check-for-understanding before moving on."
    return "Low attention trend. Use a shorter task, a poll, or call for an explicit response."


def render_status_banner(analysis: dict[str, Any]) -> None:
    status = analysis.get("status", "UNKNOWN")
    color = STATUS_COLORS.get(status, "#334155")
    score_value = float(analysis.get("attention_score", 0.0))
    st.markdown(
        f"""
        <div style="
            background:{color};
            color:white;
            padding:0.9rem 1rem;
            border-radius:0.8rem;
            margin-bottom:1rem;
            font-weight:600;
            font-size:1.1rem;
        ">
            Live status: {status} | Attention {score_value:.1f}%
        </div>
        """,
        unsafe_allow_html=True,
    )


def summary_cards(analysis: dict[str, Any]) -> None:
    cols = st.columns(5)
    cols[0].metric("Attention", f"{float(analysis.get('attention_score', 0.0)):.1f}%")
    cols[1].metric("EAR", f"{float(analysis.get('ear', 0.0)):.3f}")
    cols[2].metric("MAR", f"{float(analysis.get('mar', 0.0)):.3f}")
    cols[3].metric("Engagement", analysis.get("engagement_label", "neutral").title())
    cols[4].metric("FPS", f"{float(analysis.get('fps', 0.0)):.1f}")

    pose_cols = st.columns(4)
    pose_cols[0].metric("Pitch", f"{float(analysis.get('pitch', 0.0)):+.1f} deg")
    pose_cols[1].metric("Yaw", f"{float(analysis.get('yaw', 0.0)):+.1f} deg")
    pose_cols[2].metric("Roll", f"{float(analysis.get('roll', 0.0)):+.1f} deg")
    pose_cols[3].metric("Gaze Deviation", f"{float(analysis.get('gaze_deviation', 0.0)):.2f}")


def build_history_chart(log_frame: pd.DataFrame) -> None:
    if log_frame.empty or "timestamp" not in log_frame.columns:
        st.info("No session history yet. Start the detector to populate the dashboard.")
        return

    chart_frame = log_frame.dropna(subset=["timestamp"]).copy()
    if chart_frame.empty:
        st.info("Session history exists but does not contain readable timestamps yet.")
        return

    chart_frame = chart_frame.sort_values("timestamp")
    chart_frame["attention_score"] = pd.to_numeric(chart_frame["attention_score"], errors="coerce")
    chart_frame["attention_trend"] = chart_frame["attention_score"].rolling(window=12, min_periods=1).mean()
    chart_frame = chart_frame.set_index("timestamp")[["attention_score", "attention_trend"]]
    st.line_chart(chart_frame, use_container_width=True)


def build_status_distribution(log_frame: pd.DataFrame) -> None:
    if log_frame.empty or "status" not in log_frame.columns:
        return
    status_counts = log_frame["status"].value_counts().rename_axis("status").to_frame("count")
    if status_counts.empty:
        return
    st.bar_chart(status_counts, use_container_width=True)


def recent_alerts_table(log_frame: pd.DataFrame) -> None:
    if log_frame.empty:
        return
    alert_frame = log_frame[
        (pd.to_numeric(log_frame.get("attention_score"), errors="coerce") < 55)
        | (log_frame.get("alert_active").astype(str).str.lower() == "true")
        | (log_frame.get("status") != "ATTENTIVE")
    ].copy()
    if alert_frame.empty:
        st.success("No notable low-attention or fatigue events in the current log.")
        return
    columns = [
        "timestamp",
        "attention_score",
        "status",
        "engagement_label",
        "blendshape_insights",
    ]
    st.dataframe(alert_frame[columns].tail(20), use_container_width=True, hide_index=True)


def session_summary(log_frame: pd.DataFrame) -> None:
    if log_frame.empty:
        return

    numeric_scores = pd.to_numeric(log_frame.get("attention_score"), errors="coerce")
    avg_score = float(numeric_scores.mean()) if not numeric_scores.empty else 0.0
    low_attention_share = float((numeric_scores < 60).mean() * 100.0) if not numeric_scores.empty else 0.0
    drowsy_share = (
        float((log_frame.get("status") == "DROWSY").mean() * 100.0)
        if "status" in log_frame.columns
        else 0.0
    )

    cols = st.columns(3)
    cols[0].metric("Average attention", f"{avg_score:.1f}%")
    cols[1].metric("Time below 60%", f"{low_attention_share:.1f}%")
    cols[2].metric("Drowsy frames", f"{drowsy_share:.1f}%")


def main() -> None:
    st.set_page_config(page_title="Student Alertness Dashboard", layout="wide")
    st.title("Student Attentive Alertness Dashboard")
    st.caption("Local-only classroom monitoring dashboard powered by the MediaPipe Face Landmarker task API.")

    state = load_json(RUNTIME_STATE_PATH)
    available_logs = discover_logs()

    with st.sidebar:
        st.header("Controls")
        refresh_ms = st.slider("Refresh interval (ms)", 500, 5000, 1000, 250)
        auto_refresh = st.toggle("Auto refresh", value=True)

        if auto_refresh and st_autorefresh is not None:
            st_autorefresh(interval=refresh_ms, key="dashboard-refresh")
        elif auto_refresh:
            st.info("Install streamlit-autorefresh for automatic refresh support.")

        log_paths = available_logs[:]
        state_log = Path(state["session_log"]) if state and state.get("session_log") else None
        if state_log and state_log.exists() and state_log not in log_paths:
            log_paths.insert(0, state_log)

        labels = [path.name for path in log_paths]
        selected_label = st.selectbox("Session log", labels if labels else ["No logs available"], index=0)
        selected_log = None if not log_paths else log_paths[labels.index(selected_label)]

        st.markdown("### Runtime Files")
        st.write(f"State file: {RUNTIME_STATE_PATH}")
        if selected_log is not None:
            st.write(f"Log file: {selected_log}")

        st.markdown("### Privacy")
        st.write("All inference and scoring stay on the local machine.")

    if state is None:
        st.warning(
            "No live runtime state found yet. Start student_alertness.py first and make sure models/face_landmarker.task exists."
        )
        if not available_logs:
            st.stop()

    analysis = state.get("analysis", {}) if state else {}
    log_frame = load_log(selected_log)

    if analysis:
        render_status_banner(analysis)
        summary_cards(analysis)

        blendshape_insights = analysis.get("blendshape_insights", [])
        st.markdown("### Instructor Guidance")
        st.info(guidance_from_state(analysis))

        insight_cols = st.columns(2)
        insight_cols[0].markdown("### Blendshape Insights")
        if blendshape_insights:
            insight_cols[0].write(" | ".join(blendshape_insights))
        else:
            insight_cols[0].write("No strong expression cues at the moment.")

        subscores = analysis.get("subscores", {})
        insight_cols[1].markdown("### Attention Subscores")
        if subscores:
            subscore_frame = pd.DataFrame(
                {
                    "signal": list(subscores.keys()),
                    "score": [round(float(value), 3) for value in subscores.values()],
                }
            ).set_index("signal")
            insight_cols[1].bar_chart(subscore_frame, use_container_width=True)
        else:
            insight_cols[1].write("Subscores will appear after the first successful detection.")

    st.markdown("### Live Attention Trend")
    build_history_chart(log_frame)

    lower_left, lower_right = st.columns(2)
    with lower_left:
        st.markdown("### Session Summary")
        session_summary(log_frame)
        st.markdown("### Status Distribution")
        build_status_distribution(log_frame)

    with lower_right:
        st.markdown("### Recent Events")
        recent_alerts_table(log_frame)


if __name__ == "__main__":
    main()