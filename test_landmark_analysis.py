"""
Tests for landmark_analysis module.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from landmark_analysis import (
    calculate_angle, JOINTS, UPPER_BODY_JOINTS, LOWER_BODY_JOINTS,
    AngleSmoother, PoseStabilityDetector, cosine_similarity,
    BODY_SEGMENTS, extract_joint_angles,
)


def test_calculate_angle_straight():
    a = [0, 0, 0]
    b = [1, 0, 0]
    c = [2, 0, 0]
    angle = calculate_angle(a, b, c)
    assert angle is not None
    assert abs(angle - 180.0) < 0.1


def test_calculate_angle_right():
    a = [1, 0, 0]
    b = [0, 0, 0]
    c = [0, 1, 0]
    angle = calculate_angle(a, b, c)
    assert angle is not None
    assert abs(angle - 90.0) < 0.1


def test_calculate_angle_zero_vector():
    a = [0, 0, 0]
    b = [0, 0, 0]
    c = [1, 0, 0]
    angle = calculate_angle(a, b, c)
    assert angle is None


def test_joints_structure():
    assert len(JOINTS) == 8
    for name, idxs in JOINTS.items():
        assert len(idxs) == 3
        assert all(isinstance(i, int) for i in idxs)


def test_upper_lower_body_joints():
    assert len(UPPER_BODY_JOINTS) == 4
    assert len(LOWER_BODY_JOINTS) == 4
    for j in UPPER_BODY_JOINTS:
        assert j in JOINTS
    for j in LOWER_BODY_JOINTS:
        assert j in JOINTS


def test_angle_smoother():
    smoother = AngleSmoother(alpha=0.5, max_missing_frames=2)
    result = smoother.update({"Left Elbow": 90.0, "Right Elbow": None,
        "Left Shoulder": None, "Right Shoulder": None,
        "Left Hip": None, "Right Hip": None,
        "Left Knee": None, "Right Knee": None})
    assert result["Left Elbow"] == 90.0
    assert result["Right Elbow"] is None

    result2 = smoother.update({"Left Elbow": 100.0, "Right Elbow": None,
        "Left Shoulder": None, "Right Shoulder": None,
        "Left Hip": None, "Right Hip": None,
        "Left Knee": None, "Right Knee": None})
    assert result2["Left Elbow"] == 95.0  # 0.5*100 + 0.5*90


def test_angle_smoother_reset():
    smoother = AngleSmoother()
    smoother.update({"Left Elbow": 90.0, "Right Elbow": 90.0,
        "Left Shoulder": 90.0, "Right Shoulder": 90.0,
        "Left Hip": 90.0, "Right Hip": 90.0,
        "Left Knee": 90.0, "Right Knee": 90.0})
    smoother.reset()
    assert all(v is None for v in smoother.values.values())


def test_stability_detector_initially_unstable():
    sd = PoseStabilityDetector(required_stable_frames=3)
    angles = {"Left Elbow": 90, "Right Elbow": 90, "Left Shoulder": 90,
              "Right Shoulder": 90, "Left Hip": 90, "Right Hip": 90,
              "Left Knee": 90, "Right Knee": 90}
    assert sd.update(angles) is False


def test_stability_detector_becomes_stable():
    sd = PoseStabilityDetector(angle_threshold=5.0, required_stable_frames=3)
    angles = {"Left Elbow": 90, "Right Elbow": 90, "Left Shoulder": 90,
              "Right Shoulder": 90, "Left Hip": 90, "Right Hip": 90,
              "Left Knee": 90, "Right Knee": 90}
    for _ in range(5):
        result = sd.update(angles)
    assert result is True


def test_stability_detector_reset():
    sd = PoseStabilityDetector(required_stable_frames=2)
    angles = {"Left Elbow": 90, "Right Elbow": 90, "Left Shoulder": 90,
              "Right Shoulder": 90, "Left Hip": 90, "Right Hip": 90,
              "Left Knee": 90, "Right Knee": 90}
    for _ in range(5):
        sd.update(angles)
    sd.reset()
    assert sd.stable_count == 0
    assert len(sd.history) == 0


def test_cosine_similarity():
    assert abs(cosine_similarity([1, 0], [1, 0]) - 1.0) < 0.01
    assert abs(cosine_similarity([1, 0], [-1, 0]) - (-1.0)) < 0.01
    assert abs(cosine_similarity([1, 0], [0, 1])) < 0.01
    assert cosine_similarity([0, 0], [1, 0]) == 0.0


def test_body_segments_structure():
    assert len(BODY_SEGMENTS) == 10
    for name, (a, b) in BODY_SEGMENTS.items():
        assert isinstance(a, int)
        assert isinstance(b, int)


def test_extract_joint_angles_none():
    result = extract_joint_angles(None)
    assert all(v is None for v in result.values())


if __name__ == "__main__":
    test_calculate_angle_straight()
    test_calculate_angle_right()
    test_calculate_angle_zero_vector()
    test_joints_structure()
    test_upper_lower_body_joints()
    test_angle_smoother()
    test_angle_smoother_reset()
    test_stability_detector_initially_unstable()
    test_stability_detector_becomes_stable()
    test_stability_detector_reset()
    test_cosine_similarity()
    test_body_segments_structure()
    test_extract_joint_angles_none()
    print("All landmark_analysis tests passed!")
