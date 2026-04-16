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
import copy
import plotly.graph_objects as go
import plotly.express as px
from plotly import figure_factory as ff

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


CONFIG_PATH = PROJECT_ROOT / "config.json"


def discover_logs() -> list[Path]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(LOG_DIR.glob("*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)


def apply_command_center_theme() -> None:
    """Injects the professional Command Center CSS theme into the Streamlit app."""
    st.markdown(
        """
        <style>
            /* Main Background and Fonts */
            .stApp {
                background-color: #0a0a0a;
                color: #e0e0e0;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            }

            /* Sidebar Styling */
            [data-testid="stSidebar"] {
                background-color: #111111 !important;
                border-right: 1px solid #333;
            }

            /* Card Styling for Student Grid */
            .student-card {
                background: #1a1a1a;
                border: 1px solid #333;
                border-radius: 12px;
                padding: 15px;
                text-align: center;
                transition: all 0.2s ease-in-out;
                cursor: pointer;
                margin-bottom: 10px;
                color: white !important;
            }
            .student-card:hover {
                border-color: #00FF41;
                background: #222;
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0, 255, 65, 0.2);
            }
            .student-card.selected {
                border-color: #00FF41;
                background: #252525;
                box-shadow: 0 0 15px rgba(0, 255, 65, 0.4);
            }

            /* Status Indicator Dots */
            .status-dot {
                height: 10px;
                width: 10px;
                border-radius: 50%;
                display: inline-block;
                margin-right: 8px;
            }

            /* Alert Feed Cards */
            .alert-card {
                background: #1a1a1a;
                border-left: 5px solid #FF3131;
                border-radius: 6px;
                padding: 12px;
                margin-bottom: 8px;
                color: #eee;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .alert-card.moderate {
                border-left-color: #FACC15;
            }
            .alert-card.attentive {
                border-left-color: #00FF41;
            }

            /* Metric Overrides */
            [data-testid="stMetricValue"] {
                color: #00FF41 !important;
                font-family: 'JetBrains Mono', monospace !important;
            }

            /* Button Overrides */
            .stButton > button {
                background-color: #222;
                color: white;
                border: 1px solid #444;
                border-radius: 8px;
                transition: all 0.2s;
            }
            .stButton > button:hover {
                border-color: #00FF41;
                color: #00FF41;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_student_grid(state: dict[str, Any]) -> None:
    """Renders a grid of student status cards."""
    students = state.get("students", {})
    if not students:
        st.info("No students detected.")
        return

    # Initialize session state for selection
    if "selected_student" not in st.session_state:
        st.session_state.selected_student = list(students.keys())[0]

    st.markdown("### 👥 Classroom Overview")

    # Calculate columns based on student count
    num_students = len(students)
    cols_per_row = 4 if num_students >= 4 else num_students

    # Iterate through students and create cards
    student_list = list(students.items())
    for i in range(0, num_students, cols_per_row):
        row_students = student_list[i : i + cols_per_row]
        cols = st.columns(cols_per_row)

        for idx, (sid, data) in enumerate(row_students):
            analysis = data.get("analysis", {})
            score = analysis.get("attention_score", 0.0)
            status = analysis.get("status", "UNKNOWN")
            color_bgr = analysis.get("color_bgr", [0, 255, 0])

            # Map BGR to Hex for CSS
            hex_color = f"#{color_bgr[2]:02x}{color_bgr[1]:02x}{color_bgr[0]:02x}"

            is_selected = st.session_state.selected_student == sid
            selected_class = "selected" if is_selected else ""

            with cols[idx]:
                st.markdown(
                    f"""
                    <div class="student-card {selected_class}">
                        <div style="font-size: 0.8rem; color: #888;">Student {sid}</div>
                        <div style="font-size: 1.5rem; font-weight: bold; margin: 5px 0;">{score:.1f}%</div>
                        <div style="font-size: 0.7rem; color: {hex_color}; font-weight: 600;">{status}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                if st.button(f"Focus {sid}", key=f"focus_{sid}", use_container_width=True):
                    st.session_state.selected_student = sid
                    st.rerun()


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


def save_config(config: dict) -> bool:
    config_path = PROJECT_ROOT / "config.json"
    try:
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Failed to save config: {e}")
        return False

def render_config_editor() -> None:
    st.markdown("### ⚙️ Configuration Editor")
    config = load_json(CONFIG_PATH) or {}

    # Create a copy for editing
    edited_config = copy.deepcopy(config)

    with st.expander("Tune Attention Thresholds", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            edited_config["EAR_CLOSED_THRESHOLD"] = st.number_input(
                "EAR Closed Threshold",
                value=float(config.get("EAR_CLOSED_THRESHOLD", 0.18)),
                format="%.3f", step=0.01
            )
            edited_config["MAR_YAWN_THRESHOLD"] = st.number_input(
                "MAR Yawn Threshold",
                value=float(config.get("MAR_YAWN_THRESHOLD", 0.38)),
                format="%.3f", step=0.01
            )
            edited_config["HEAD_YAW_LIMIT_DEG"] = st.number_input(
                "Head Yaw Limit (deg)",
                value=float(config.get("HEAD_YAW_LIMIT_DEG", 22.0)),
                step=1.0
            )
        with col2:
            edited_config["ATTENTIVE_THRESHOLD"] = st.number_input(
                "Attentive Threshold (%)",
                value=float(config.get("ATTENTIVE_THRESHOLD", 75.0)),
                step=1.0
            )
            edited_config["MODERATE_THRESHOLD"] = st.number_input(
                "Moderate Threshold (%)",
                value=float(config.get("MODERATE_THRESHOLD", 55.0)),
                step=1.0
            )
            edited_config["ALERT_COOLDOWN_SECONDS"] = st.number_input(
                "Alert Cooldown (s)",
                value=float(config.get("ALERT_COOLDOWN_SECONDS", 3.0)),
                step=0.5
            )

    if st.button("Save Configuration", type="primary"):
        if save_config(edited_config):
            st.success("Configuration saved! Restart the detector to apply changes.")
            st.rerun()


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
    cols[0].metric("Attention Score", f"{float(analysis.get('attention_score', 0.0)):.1f}%",
                   help="Weighted composite (0–100). Above 75 = attentive, 55–75 = moderate, below 55 = low.")
    cols[1].metric("EAR", f"{float(analysis.get('ear', 0.0)):.3f}",
                   help="Eye Aspect Ratio. Values below ~0.18 indicate closed or nearly closed eyes.")
    cols[2].metric("MAR", f"{float(analysis.get('mar', 0.0)):.3f}",
                   help="Mouth Aspect Ratio. Values above ~0.38 indicate an open mouth or yawn.")
    cols[3].metric("Engagement", analysis.get("engagement_label", "neutral").title(),
                   help="Blendshape-derived label: attentive, neutral, confused, or bored.")
    cols[4].metric("FPS", f"{float(analysis.get('fps', 0.0)):.1f}",
                   help="Detector frames per second. Below 10 FPS may reduce detection accuracy.")

    pose_cols = st.columns(4)
    pose_cols[0].metric("Pitch", f"{float(analysis.get('pitch', 0.0)):+.1f}°",
                        help="Head tilt forward/back. Beyond ±18° means the head is tilted away.")
    pose_cols[1].metric("Yaw", f"{float(analysis.get('yaw', 0.0)):+.1f}°",
                        help="Head turn left/right. Beyond ±22° means the student is looking away.")
    pose_cols[2].metric("Roll", f"{float(analysis.get('roll', 0.0)):+.1f}°",
                        help="Head tilt sideways. Large values may indicate fatigue posture.")
    pose_cols[3].metric("Gaze Deviation", f"{float(analysis.get('gaze_deviation', 0.0)):.2f}",
                        help="Iris deviation from screen center (0 = centered, 1+ = looking away).")


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


def render_alert_feed(log_frame: pd.DataFrame) -> None:
    """Renders a stylized HTML feed of recent alerts."""
    if log_frame.empty:
        return

    # Filter for alerts (low score, explicit alert_active, or non-attentive status)
    alert_frame = log_frame[
        (pd.to_numeric(log_frame.get("attention_score"), errors="coerce") < 55)
        | (log_frame.get("alert_active").astype(str).str.lower() == "true")
        | (log_frame.get("status") != "ATTENTIVE")
    ].copy()

    if alert_frame.empty:
        st.success("No notable low-attention or fatigue events in the current log.")
        return

    # Sort by timestamp descending
    alert_frame = alert_frame.sort_values("timestamp", ascending=False)

    st.markdown("### ⚠️ Alert Feed")

    # Filter toggle
    filter_mode = st.segmented_control(
        "Filter Alerts",
        options=["All", "High Priority"],
        default="All"
    )

    if filter_mode == "High Priority":
        high_priority_statuses = {"DROWSY", "LOW ATTENTION", "NO FACE DETECTED"}
        alert_frame = alert_frame[alert_frame["status"].isin(high_priority_statuses)]

    if alert_frame.empty:
        st.info("No high-priority alerts detected.")
        return

    # Render top 15 alerts as HTML cards
    for i, (_, row) in enumerate(alert_frame.head(15).iterrows()):
        status = row.get("status", "UNKNOWN")
        score = float(row.get("attention_score", 0.0))
        ts = row.get("timestamp", "N/A")
        student_id = row.get("student_id", "Unknown")

        # Determine card class
        card_class = "alert-card"
        if status == "ATTENTIVE": card_class += " attentive"
        elif status in {"MODERATE", "MODERATE / CONFUSED"}: card_class += " moderate"

        # We use a button for the nudge action
        # Since we can't put buttons inside HTML strings, we'll use a row-based approach
        with st.container():
            st.markdown(
                f"""
                <div class="{card_class}">
                    <div>
                        <span style="color: #888; font-size: 0.7rem;">{ts}</span><br>
                        <strong>Student {student_id}</strong> | {status} ({score:.1f}%)
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            # Add a small button underneath the HTML card
            safe_key = f"nudge_{str(ts).replace(':', '-').replace(' ', '_')}_{student_id}_{i}"
            if st.button(f"Nudge Student {student_id}", key=safe_key, use_container_width=False, type="secondary"):
                st.toast(f"Nudge signal sent to Student {student_id}!")



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


def prepare_classroom_data(log_frame: pd.DataFrame) -> pd.DataFrame:
    """Pivots the log data to have students as columns and timestamps as index."""
    if log_frame.empty or "timestamp" not in log_frame.columns:
        return pd.DataFrame()

    df = log_frame.dropna(subset=["timestamp"]).copy()
    df = df.sort_values("timestamp")
    df["attention_score"] = pd.to_numeric(df["attention_score"], errors="coerce")

    # Use student_id column if present (new logs), else treat whole session as student 0
    if "student_id" not in df.columns:
        df["student_id"] = 0

    # pivot_table handles duplicate timestamps gracefully by averaging
    pivot_df = df.pivot_table(
        index="timestamp", columns="student_id", values="attention_score", aggfunc="mean"
    )
    pivot_df = pivot_df.interpolate(method="linear").fillna(0.0)
    return pivot_df


def render_attention_heatmap(log_frame: pd.DataFrame) -> None:
    """Renders an interactive Plotly heatmap of class attention."""
    df = prepare_classroom_data(log_frame)
    if df.empty:
        st.info("Insufficient multi-student data to render heatmap.")
        return

    fig = go.Figure(data=go.Heatmap(
        z=df.values,
        x=df.index,
        y=df.columns,
        colorscale="Viridis",
        colorbar=dict(title="Score (%)"),
        hovertemplate="Time: %{x}<br>Student: %{y}<br>Attention: %{z:.1f}%<extra></extra>"
    ))

    fig.update_layout(
        title="Classroom Attention Heatmap",
        xaxis_title="Session Time",
        yaxis_title="Student ID",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)


def render_comparative_chart(log_frame: pd.DataFrame) -> None:
    """Renders a multi-line chart comparing all students."""
    df = prepare_classroom_data(log_frame)
    if df.empty:
        st.info("Insufficient multi-student data for comparative trends.")
        return

    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[col],
            mode='lines',
            name=f"Student {col}",
            line=dict(width=2)
        ))

    fig.update_layout(
        title="Comparative Attention Trajectories",
        xaxis_title="Session Time",
        yaxis_title="Attention Score (%)",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=20),
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)


def render_session_deadzones(log_frame: pd.DataFrame) -> None:
    """Renders a chart showing overall class attention to find dead-zones."""
    df = prepare_classroom_data(log_frame)
    if df.empty:
        st.info("Insufficient data to analyze session dead-zones.")
        return

    # Calculate mean attention across all students per time slice
    class_avg = df.mean(axis=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=class_avg.index,
        y=class_avg.values,
        mode='lines',
        line=dict(color='#00FF41', width=3),
        fill='tozeroy',
        name="Class Average"
    ))

    fig.update_layout(
        title="Session 'Dead-Zone' Analysis (Class Average)",
        xaxis_title="Session Time",
        yaxis_title="Average Attention (%)",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
        height=300
    )
    st.plotly_chart(fig, use_container_width=True)


def clear_session_logs() -> None:
    if not LOG_DIR.exists():
        return

    logs = list(LOG_DIR.glob("*.csv"))
    if not logs:
        st.info("No logs to clear.")
        return

    try:
        for log_file in logs:
            log_file.unlink()
        st.success(f"Cleared {len(logs)} session logs.")
    except Exception as e:
        st.error(f"Failed to clear logs: {e}")

def main() -> None:
    st.set_page_config(page_title="VigilEye Command Center", layout="wide")
    apply_command_center_theme()

    st.title("🛰️ VigilEye Command Center")
    st.caption("Professional Classroom Monitoring | Real-time Attention Analysis")

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
        session_info = state.get("session_info", {}) if state else {}
        state_log = Path(session_info["session_log"]) if session_info.get("session_log") else None
        if state_log and state_log.exists() and state_log not in log_paths:
            log_paths.insert(0, state_log)

        labels = [path.name for path in log_paths]
        # Make labels human-readable: "class_session  2026-04-14  15:22" instead of raw filename
        def _friendly_label(p: Path) -> str:
            name = p.stem  # e.g. "class_session_20260414_152200"
            parts = name.rsplit("_", 2)
            if len(parts) == 3:
                session, date, hms = parts
                try:
                    return f"{session.replace('_', ' ')}  {date[:4]}-{date[4:6]}-{date[6:]}  {hms[:2]}:{hms[2:4]}"
                except Exception:
                    pass
            return name
        friendly_labels = [_friendly_label(p) for p in log_paths]
        selected_label = st.selectbox("Session log", friendly_labels if friendly_labels else ["No logs available"], index=0)
        selected_log = None if not log_paths else log_paths[friendly_labels.index(selected_label)] if selected_label in friendly_labels else None

        st.markdown("### System")
        if selected_log is not None:
            st.caption(f"Log: `{selected_log.name}`")
        detector_running = state is not None
        st.caption(f"Detector: {'running' if detector_running else 'not running'}")

        st.markdown("### Privacy")
        st.write("All inference and scoring stay on the local machine.")

        st.markdown("### Configuration")
        render_config_editor()

        st.markdown("---")
        if "confirm_clear" not in st.session_state:
            st.session_state.confirm_clear = False
        if not st.session_state.confirm_clear:
            if st.button("Clear All Logs", type="secondary"):
                st.session_state.confirm_clear = True
                st.rerun()
        else:
            st.warning("This will permanently delete all session logs.")
            col_yes, col_no = st.columns(2)
            if col_yes.button("Yes, Delete", type="primary"):
                clear_session_logs()
                st.session_state.confirm_clear = False
                st.rerun()
            if col_no.button("Cancel"):
                st.session_state.confirm_clear = False
                st.rerun()

    if state is None:
        st.markdown(
            """
            <div style="background:#1a1a1a;border:1px solid #333;border-radius:12px;padding:1.5rem;margin-bottom:1rem;">
                <h3 style="color:#facc15;margin-top:0;">No live detector running</h3>
                <p style="color:#aaa;margin-bottom:0.8rem;">The detector is not writing state yet. Follow these steps to start:</p>
                <ol style="color:#ccc;line-height:2;">
                    <li>Make sure <code style="color:#00FF41">models/face_landmarker.task</code> exists
                        &nbsp;<span style="color:#888;font-size:0.8rem;">(see README for download command)</span></li>
                    <li>Open a terminal in the project directory</li>
                    <li>Run: <code style="background:#111;padding:2px 8px;border-radius:4px;color:#00FF41">./.py312/bin/python launch_vigileye.py --camera-check</code></li>
                    <li>Wait for <em>Face Landmarker initialized</em> to appear in the terminal</li>
                    <li>This dashboard will update automatically once the detector is running</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if not available_logs:
            st.info("No previous session logs found either. Start the detector to record your first session.")
            st.stop()

    # 1. Render the Student Grid
    if state is not None:
        render_student_grid(state)
        st.markdown("---")

    # 2. Resolve analysis for the selected student (from session state)
    students_data = state.get("students", {}) if state else {}
    selected_student_id = st.session_state.get("selected_student")

    if selected_student_id is not None and selected_student_id in students_data:
        analysis = students_data[selected_student_id].get("analysis", {})
        calibration_status = students_data[selected_student_id].get("calibration_status", "IDLE")
    else:
        # Fallback to first available student if none selected
        if students_data:
            first_id = list(students_data.keys())[0]
            analysis = students_data[first_id].get("analysis", {})
            calibration_status = students_data[first_id].get("calibration_status", "IDLE")
        else:
            analysis = {}
            calibration_status = "IDLE"

    log_frame = load_log(selected_log)

    if analysis:
        if calibration_status == "CALIBRATING":
            st.warning("Student is currently calibrating. Please wait for a stable baseline.")

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

    st.markdown("### 📊 Session Analytics")
    num_log_students = log_frame["student_id"].nunique() if "student_id" in log_frame.columns and not log_frame.empty else 1
    if num_log_students > 1:
        render_attention_heatmap(log_frame)
        render_comparative_chart(log_frame)
        render_session_deadzones(log_frame)
    else:
        # Single-student: show a simple trend chart with a rolling average
        build_history_chart(log_frame)

    st.markdown("---")
    lower_left, lower_right = st.columns(2)
    with lower_left:
        st.markdown("### Session Summary")
        session_summary(log_frame)
        st.markdown("### Status Distribution")
        build_status_distribution(log_frame)

    with lower_right:
        st.markdown("### 🔔 Notifications")
        render_alert_feed(log_frame)


if __name__ == "__main__":
    main()