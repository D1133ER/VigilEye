from core import TemporalState, FrameAnalysis, CalibrationStatus


class TestSmooth:
    def test_first_call_returns_raw(self):
        ts = TemporalState()
        assert ts.smooth("x", 10.0) == 10.0

    def test_ema_converges(self):
        ts = TemporalState()
        for _ in range(300):
            val = ts.smooth("x", 5.0, alpha=0.35)
        assert abs(val - 5.0) < 0.01

    def test_different_keys_independent(self):
        ts = TemporalState()
        ts.smooth("a", 100.0)
        ts.smooth("b", 0.0)
        assert ts.filtered_values["a"] == 100.0
        assert ts.filtered_values["b"] == 0.0


class TestEyeDuration:
    def test_accumulates_when_closed(self):
        ts = TemporalState()
        ts.update_eye_duration(0.10, 1.0)  # below threshold → starts timer
        d = ts.update_eye_duration(0.10, 2.0)
        assert abs(d - 1.0) < 0.01

    def test_resets_when_open(self):
        ts = TemporalState()
        ts.update_eye_duration(0.10, 1.0)
        ts.update_eye_duration(0.30, 2.0)  # above recovery threshold
        d = ts.update_eye_duration(0.10, 3.0)
        assert d == 0.0  # timer restarted


class TestYawnDuration:
    def test_accumulates(self):
        ts = TemporalState()
        ts.update_yawn_duration(0.50, 1.0)  # above threshold
        d = ts.update_yawn_duration(0.50, 2.5)
        assert abs(d - 1.5) < 0.01

    def test_resets_when_mouth_closed(self):
        ts = TemporalState()
        ts.update_yawn_duration(0.50, 1.0)
        ts.update_yawn_duration(0.20, 2.0)  # below recovery
        d = ts.update_yawn_duration(0.50, 3.0)
        assert d == 0.0


class TestLookAwayDuration:
    def test_accumulates(self):
        ts = TemporalState()
        ts.update_look_away_duration(True, 1.0)
        d = ts.update_look_away_duration(True, 3.0)
        assert abs(d - 2.0) < 0.01

    def test_resets_when_looking_back(self):
        ts = TemporalState()
        ts.update_look_away_duration(True, 1.0)
        ts.update_look_away_duration(False, 2.0)
        d = ts.update_look_away_duration(True, 3.0)
        assert d == 0.0


class TestLookAwayBuffer:
    def test_below_min_samples_returns_raw(self):
        ts = TemporalState()
        assert ts.update_look_away_buffer(True) is True
        assert ts.update_look_away_buffer(False) is False

    def test_sustained_distraction(self):
        ts = TemporalState()
        # Fill buffer with 90 "looking away" frames
        for _ in range(90):
            result = ts.update_look_away_buffer(True)
        assert result is True  # 100% > 70% threshold


class TestFrameAnalysisPayload:
    def test_roundtrip_preserves_fields(self):
        fa = FrameAnalysis(student_id=1, attention_score=85.0, status="ATTENTIVE")
        p = fa.to_state_payload()
        assert p["student_id"] == 1
        assert p["attention_score"] == 85.0
        assert p["status"] == "ATTENTIVE"

    def test_color_bgr_becomes_list(self):
        fa = FrameAnalysis()
        p = fa.to_state_payload()
        assert isinstance(p["color_bgr"], list)
        assert len(p["color_bgr"]) == 3


class TestCalibrationStatus:
    def test_initial_state_is_idle(self):
        ts = TemporalState()
        assert ts.calibration_status == CalibrationStatus.IDLE
        assert ts.baselines.ear == 0.0
        assert len(ts.calibration_samples) == 0

    def test_reset_clears_timers(self):
        ts = TemporalState()
        ts.update_eye_duration(0.10, 1.0)
        ts.update_look_away_duration(True, 1.0)
        ts.reset()
        assert ts.eye_closed_since is None
        assert ts.look_away_since is None
