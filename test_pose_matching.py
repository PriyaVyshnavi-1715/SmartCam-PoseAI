"""
Tests for pose_matching module with 15 reference poses.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pose_matching import REFERENCE_POSES, compare_pose, JOINT_NAMES, HUMAN_FEEDBACK


def test_all_15_poses_defined():
    expected = [
        "T-Pose", "Standing Pose", "Sitting Pose", "Warrior Pose",
        "Squat Pose", "Tree Pose", "Mountain Pose", "Victory Pose",
        "Side Stretch", "Model Pose", "Confident Stance", "Hip Pop",
        "Head Tilt", "Star Pose", "Lunge Pose",
    ]
    for pose in expected:
        assert pose in REFERENCE_POSES, f"Missing pose: {pose}"
    assert len(REFERENCE_POSES) == 15


def test_all_poses_have_8_joints():
    for pose, angles in REFERENCE_POSES.items():
        for joint in JOINT_NAMES:
            assert joint in angles, f"{pose} missing joint {joint}"
            lo, hi = angles[joint]
            assert 0 <= lo <= hi <= 180, f"{pose} {joint}: invalid range ({lo}, {hi})"


def test_compare_pose_returns_structure():
    angles = {
        "Left Elbow": 170, "Right Elbow": 170,
        "Left Shoulder": 90, "Right Shoulder": 90,
        "Left Hip": 170, "Right Hip": 170,
        "Left Knee": 170, "Right Knee": 170,
    }
    result = compare_pose(angles, "T-Pose")
    assert "ready" in result
    assert "score" in result
    assert "feedback" in result
    assert "joint_scores" in result
    assert isinstance(result["feedback"], list)


def test_perfect_t_pose():
    angles = {
        "Left Elbow": 170, "Right Elbow": 170,
        "Left Shoulder": 90, "Right Shoulder": 90,
        "Left Hip": 170, "Right Hip": 170,
        "Left Knee": 170, "Right Knee": 170,
    }
    result = compare_pose(angles, "T-Pose")
    assert result["ready"] is True
    assert result["score"] is not None
    assert result["score"] >= 90


def test_bad_pose_gives_feedback():
    angles = {
        "Left Elbow": 45, "Right Elbow": 45,
        "Left Shoulder": 10, "Right Shoulder": 10,
        "Left Hip": 90, "Right Hip": 90,
        "Left Knee": 60, "Right Knee": 60,
    }
    result = compare_pose(angles, "T-Pose")
    assert result["ready"] is True
    assert result["score"] is not None
    assert result["score"] < 50
    assert len(result["feedback"]) > 0


def test_human_feedback_dict():
    assert len(HUMAN_FEEDBACK) == 16
    for joint in JOINT_NAMES:
        assert (joint, "increase") in HUMAN_FEEDBACK
        assert (joint, "decrease") in HUMAN_FEEDBACK


def test_insufficient_joints():
    angles = {"Left Elbow": 170, "Right Elbow": 170}
    result = compare_pose(angles, "T-Pose")
    assert result["ready"] is False
    assert result["score"] is None


def test_each_pose_scores_correctly():
    """Test that each pose scores 100% when given perfect angles."""
    for pose_name, targets in REFERENCE_POSES.items():
        angles = {j: (lo + hi) / 2 for j, (lo, hi) in targets.items()}
        result = compare_pose(angles, pose_name)
        assert result["ready"] is True, f"{pose_name} not ready"
        assert result["score"] is not None, f"{pose_name} score is None"
        assert result["score"] == 100.0, f"{pose_name} score {result['score']} != 100"


if __name__ == "__main__":
    test_all_15_poses_defined()
    test_all_poses_have_8_joints()
    test_compare_pose_returns_structure()
    test_perfect_t_pose()
    test_bad_pose_gives_feedback()
    test_human_feedback_dict()
    test_insufficient_joints()
    test_each_pose_scores_correctly()
    print("All pose_matching tests passed!")
