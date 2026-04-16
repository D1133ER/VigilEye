import numpy as np

from vision_utils import (
    clamp,
    euclidean,
    score_from_range,
    inverse_score_from_range,
    compute_ear,
    compute_avg_ear,
    compute_mar,
    face_bounding_box,
    analyze_blendshapes,
    attention_score,
    status_for_analysis,
    compute_gaze_metrics,
    estimate_head_pose,
    RIGHT_EYE_EAR,
    LEFT_EYE_EAR,
)


# ---------------------------------------------------------------------------
# Pure scalar helpers
# ---------------------------------------------------------------------------

class TestClamp:
    def test_within_range(self):
        assert clamp(0.5, 0.0, 1.0) == 0.5

    def test_below(self):
        assert clamp(-3.0, 0.0, 1.0) == 0.0

    def test_above(self):
        assert clamp(5.0, 0.0, 1.0) == 1.0

    def test_at_boundary(self):
        assert clamp(0.0, 0.0, 1.0) == 0.0
        assert clamp(1.0, 0.0, 1.0) == 1.0


class TestEuclidean:
    def test_zero_distance(self):
        assert euclidean(np.array([0, 0]), np.array([0, 0])) == 0.0

    def test_known_distance(self):
        assert abs(euclidean(np.array([0, 0]), np.array([3, 4])) - 5.0) < 1e-9


class TestScoreFromRange:
    def test_midpoint(self):
        assert abs(score_from_range(0.5, 0.0, 1.0) - 0.5) < 0.01

    def test_below_low(self):
        assert score_from_range(-1.0, 0.0, 1.0) == 0.0

    def test_above_high(self):
        assert score_from_range(2.0, 0.0, 1.0) == 1.0

    def test_inverse_at_low(self):
        assert abs(inverse_score_from_range(0.0, 0.0, 1.0) - 1.0) < 0.01

    def test_inverse_at_high(self):
        assert abs(inverse_score_from_range(1.0, 0.0, 1.0) - 0.0) < 0.01


# ---------------------------------------------------------------------------
# Geometric features (EAR, MAR)
# ---------------------------------------------------------------------------

class TestEarMar:
    def test_ear_positive(self, landmarks_478):
        ear = compute_ear(landmarks_478, RIGHT_EYE_EAR)
        assert 0.0 < ear < 1.0

    def test_avg_ear_positive(self, landmarks_478):
        avg = compute_avg_ear(landmarks_478)
        assert 0.0 < avg < 1.0

    def test_left_right_similar(self, landmarks_478):
        r = compute_ear(landmarks_478, RIGHT_EYE_EAR)
        l = compute_ear(landmarks_478, LEFT_EYE_EAR)
        assert abs(r - l) < 0.2  # symmetric fixture → should be close

    def test_mar_positive(self, landmarks_478):
        mar = compute_mar(landmarks_478)
        assert 0.0 < mar < 1.0


# ---------------------------------------------------------------------------
# Bounding box
# ---------------------------------------------------------------------------

class TestBoundingBox:
    def test_valid_bounds(self, landmarks_478):
        bbox = face_bounding_box(landmarks_478, 960, 540)
        x1, y1, x2, y2 = bbox
        assert x1 <= x2
        assert y1 <= y2
        assert x1 >= 0 and x2 < 960
        assert y1 >= 0 and y2 < 540


# ---------------------------------------------------------------------------
# Gaze and head pose
# ---------------------------------------------------------------------------

class TestGaze:
    def test_centered_gaze_near_zero(self, landmarks_478):
        h, v, dev, center = compute_gaze_metrics(landmarks_478)
        # Fixture has iris centered within eye corners → offsets should be moderate
        assert abs(h) < 0.5
        assert abs(v) < 0.5
        assert 0.0 <= center <= 1.0


class TestHeadPose:
    def test_returns_three_angles(self, landmarks_478):
        norm = landmarks_478.copy()
        norm[:, 0] /= 960
        norm[:, 1] /= 540
        norm[:, 2] /= 960
        pitch, yaw, roll = estimate_head_pose(norm, landmarks_478, 960, 540)
        assert isinstance(pitch, float)
        assert isinstance(yaw, float)
        assert isinstance(roll, float)


# ---------------------------------------------------------------------------
# Blendshape engagement analysis
# ---------------------------------------------------------------------------

class TestBlendshapes:
    def test_empty_neutral(self):
        r = analyze_blendshapes({})
        assert r.label == "neutral"
        assert 0.0 <= r.score <= 1.0

    def test_focused_signals_attentive(self):
        scores = {
            "browInnerUp": 0.8, "browOuterUpLeft": 0.7, "browOuterUpRight": 0.7,
            "mouthPressLeft": 0.6, "mouthPressRight": 0.6,
            "mouthSmileLeft": 0.3, "mouthSmileRight": 0.3,
        }
        r = analyze_blendshapes(scores)
        assert r.label == "attentive"

    def test_fatigue_signals_bored(self):
        scores = {
            "jawOpen": 0.8,
            "mouthFrownLeft": 0.6, "mouthFrownRight": 0.6,
            "eyeBlinkLeft": 0.5, "eyeBlinkRight": 0.5,
        }
        r = analyze_blendshapes(scores)
        assert r.label == "bored"


# ---------------------------------------------------------------------------
# Attention score
# ---------------------------------------------------------------------------

class TestAttentionScore:
    def test_no_face_zero(self):
        score, subs = attention_score(
            face_present=False, ear=0.0, mar=0.0, pitch=0.0, yaw=0.0, roll=0.0,
            gaze_center_score=0.0, engagement_score_value=0.0,
            drowsy_active=False, yawn_active=False, look_away_active=False,
        )
        assert score == 0.0
        assert all(v == 0.0 for v in subs.values())

    def test_ideal_inputs_high(self):
        score, _ = attention_score(
            face_present=True, ear=0.28, mar=0.15, pitch=0.0, yaw=0.0, roll=0.0,
            gaze_center_score=1.0, engagement_score_value=0.9,
            drowsy_active=False, yawn_active=False, look_away_active=False,
        )
        assert score >= 70.0

    def test_drowsy_penalty(self):
        base, _ = attention_score(
            face_present=True, ear=0.28, mar=0.15, pitch=0.0, yaw=0.0, roll=0.0,
            gaze_center_score=1.0, engagement_score_value=0.9,
            drowsy_active=False, yawn_active=False, look_away_active=False,
        )
        penalized, _ = attention_score(
            face_present=True, ear=0.28, mar=0.15, pitch=0.0, yaw=0.0, roll=0.0,
            gaze_center_score=1.0, engagement_score_value=0.9,
            drowsy_active=True, yawn_active=False, look_away_active=False,
        )
        assert penalized < base


# ---------------------------------------------------------------------------
# Status classification
# ---------------------------------------------------------------------------

class TestStatusForAnalysis:
    def test_no_face(self):
        s, _, alert = status_for_analysis(False, 50.0, False, False, False, "neutral")
        assert s == "NO FACE DETECTED"
        assert alert is True

    def test_drowsy(self):
        s, _, _ = status_for_analysis(True, 30.0, True, False, False, "neutral")
        assert s == "DROWSY"

    def test_attentive(self):
        s, _, alert = status_for_analysis(True, 80.0, False, False, False, "attentive")
        assert s == "ATTENTIVE"
        assert alert is False

    def test_looking_away(self):
        s, _, _ = status_for_analysis(True, 40.0, False, False, True, "neutral")
        assert s == "LOOKING AWAY"

    def test_moderate_confused(self):
        s, _, _ = status_for_analysis(True, 60.0, False, False, False, "confused")
        assert s == "MODERATE / CONFUSED"
