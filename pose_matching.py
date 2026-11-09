"""
Rule-based reference pose matching with 15 poses.
Scores are heuristic similarities, not medical assessments.
"""
from landmark_analysis import UPPER_BODY_JOINTS, LOWER_BODY_JOINTS

JOINT_NAMES = (
    "Left Elbow", "Right Elbow", "Left Shoulder", "Right Shoulder",
    "Left Hip", "Right Hip", "Left Knee", "Right Knee",
)

REFERENCE_POSES = {
    "T-Pose": {
        "Left Elbow": (155, 180), "Right Elbow": (155, 180),
        "Left Shoulder": (75, 110), "Right Shoulder": (75, 110),
        "Left Hip": (155, 180), "Right Hip": (155, 180),
        "Left Knee": (155, 180), "Right Knee": (155, 180),
    },
    "Standing Pose": {
        "Left Elbow": (145, 180), "Right Elbow": (145, 180),
        "Left Shoulder": (0, 50), "Right Shoulder": (0, 50),
        "Left Hip": (145, 180), "Right Hip": (145, 180),
        "Left Knee": (150, 180), "Right Knee": (150, 180),
    },
    "Sitting Pose": {
        "Left Elbow": (70, 125), "Right Elbow": (70, 125),
        "Left Shoulder": (0, 65), "Right Shoulder": (0, 65),
        "Left Hip": (75, 125), "Right Hip": (75, 125),
        "Left Knee": (70, 130), "Right Knee": (70, 130),
    },
    "Warrior Pose": {
        "Left Elbow": (140, 180), "Right Elbow": (140, 180),
        "Left Shoulder": (35, 150), "Right Shoulder": (35, 150),
        "Left Hip": (65, 145), "Right Hip": (65, 180),
        "Left Knee": (65, 125), "Right Knee": (135, 180),
    },
    "Squat Pose": {
        "Left Elbow": (65, 180), "Right Elbow": (65, 180),
        "Left Shoulder": (0, 145), "Right Shoulder": (0, 145),
        "Left Hip": (55, 125), "Right Hip": (55, 125),
        "Left Knee": (55, 125), "Right Knee": (55, 125),
    },
    "Tree Pose": {
        "Left Elbow": (65, 180), "Right Elbow": (65, 180),
        "Left Shoulder": (0, 150), "Right Shoulder": (0, 150),
        "Left Hip": (25, 115), "Right Hip": (135, 180),
        "Left Knee": (25, 115), "Right Knee": (135, 180),
    },
    "Mountain Pose": {
        "Left Elbow": (145, 180), "Right Elbow": (145, 180),
        "Left Shoulder": (0, 30), "Right Shoulder": (0, 30),
        "Left Hip": (160, 180), "Right Hip": (160, 180),
        "Left Knee": (160, 180), "Right Knee": (160, 180),
    },
    "Victory Pose": {
        "Left Elbow": (150, 180), "Right Elbow": (150, 180),
        "Left Shoulder": (120, 170), "Right Shoulder": (120, 170),
        "Left Hip": (160, 180), "Right Hip": (160, 180),
        "Left Knee": (160, 180), "Right Knee": (160, 180),
    },
    "Side Stretch": {
        "Left Elbow": (140, 180), "Right Elbow": (140, 180),
        "Left Shoulder": (0, 40), "Right Shoulder": (130, 175),
        "Left Hip": (140, 180), "Right Hip": (140, 180),
        "Left Knee": (155, 180), "Right Knee": (155, 180),
    },
    "Model Pose": {
        "Left Elbow": (40, 100), "Right Elbow": (140, 180),
        "Left Shoulder": (15, 65), "Right Shoulder": (0, 40),
        "Left Hip": (155, 180), "Right Hip": (155, 180),
        "Left Knee": (155, 180), "Right Knee": (155, 180),
    },
    "Confident Stance": {
        "Left Elbow": (40, 100), "Right Elbow": (40, 100),
        "Left Shoulder": (15, 60), "Right Shoulder": (15, 60),
        "Left Hip": (155, 180), "Right Hip": (155, 180),
        "Left Knee": (155, 180), "Right Knee": (155, 180),
    },
    "Hip Pop": {
        "Left Elbow": (40, 100), "Right Elbow": (140, 180),
        "Left Shoulder": (15, 65), "Right Shoulder": (0, 45),
        "Left Hip": (140, 175), "Right Hip": (155, 180),
        "Left Knee": (155, 180), "Right Knee": (155, 180),
    },
    "Head Tilt": {
        "Left Elbow": (130, 180), "Right Elbow": (130, 180),
        "Left Shoulder": (0, 45), "Right Shoulder": (0, 45),
        "Left Hip": (155, 180), "Right Hip": (155, 180),
        "Left Knee": (155, 180), "Right Knee": (155, 180),
    },
    "Star Pose": {
        "Left Elbow": (155, 180), "Right Elbow": (155, 180),
        "Left Shoulder": (100, 160), "Right Shoulder": (100, 160),
        "Left Hip": (100, 150), "Right Hip": (100, 150),
        "Left Knee": (155, 180), "Right Knee": (155, 180),
    },
    "Lunge Pose": {
        "Left Elbow": (140, 180), "Right Elbow": (140, 180),
        "Left Shoulder": (0, 50), "Right Shoulder": (0, 50),
        "Left Hip": (70, 130), "Right Hip": (140, 180),
        "Left Knee": (70, 110), "Right Knee": (155, 180),
    },
}

UPPER_BODY_COMPATIBLE = {
    "T-Pose", "Standing Pose", "Sitting Pose",
    "Head Tilt", "Model Pose", "Confident Stance", "Mountain Pose",
}

HUMAN_FEEDBACK = {
    ("Left Elbow", "increase"): "Straighten your left arm more",
    ("Left Elbow", "decrease"): "Bend your left elbow more",
    ("Right Elbow", "increase"): "Straighten your right arm more",
    ("Right Elbow", "decrease"): "Bend your right elbow more",
    ("Left Shoulder", "increase"): "Raise your left arm higher",
    ("Left Shoulder", "decrease"): "Lower your left arm slightly",
    ("Right Shoulder", "increase"): "Raise your right arm higher",
    ("Right Shoulder", "decrease"): "Lower your right arm slightly",
    ("Left Hip", "increase"): "Straighten your left hip",
    ("Left Hip", "decrease"): "Bend at your left hip more",
    ("Right Hip", "increase"): "Straighten your right hip",
    ("Right Hip", "decrease"): "Bend at your right hip more",
    ("Left Knee", "increase"): "Straighten your left leg",
    ("Left Knee", "decrease"): "Bend your left knee more",
    ("Right Knee", "increase"): "Straighten your right leg",
    ("Right Knee", "decrease"): "Bend your right knee more",
}


def _score_joint(angle, target_range, falloff_degrees=45.0):
    low, high = target_range
    if low <= angle <= high:
        return 100.0, 0.0
    distance = low - angle if angle < low else angle - high
    return max(0.0, 100.0 * (1.0 - distance / falloff_degrees)), distance


def compare_pose(current_angles, selected_pose, min_valid_joints=None):
    if selected_pose not in REFERENCE_POSES:
        raise ValueError(
            f"Unknown pose {selected_pose!r}. Choose from: {', '.join(REFERENCE_POSES)}"
        )

    targets = REFERENCE_POSES[selected_pose]
    valid = {}
    for name in JOINT_NAMES:
        value = current_angles.get(name)
        try:
            value = float(value) if value is not None else None
        except (TypeError, ValueError):
            value = None
        if value is not None and 0.0 <= value <= 180.0:
            valid[name] = value

    upper_count = sum(name in valid for name in UPPER_BODY_JOINTS)
    lower_count = sum(name in valid for name in LOWER_BODY_JOINTS)

    if selected_pose in UPPER_BODY_COMPATIBLE and lower_count < 2:
        mode = "upper_body"
        selected_names = UPPER_BODY_JOINTS
        required = 3 if min_valid_joints is None else min(min_valid_joints, 3)
    else:
        mode = "full_body"
        selected_names = JOINT_NAMES
        required = 5 if min_valid_joints is None else min_valid_joints

    joint_scores = {name: None for name in JOINT_NAMES}
    feedback = []
    for name in selected_names:
        angle = valid.get(name)
        if angle is None:
            continue
        score, distance = _score_joint(angle, targets[name])
        joint_scores[name] = round(score, 1)
        if distance > 0:
            low, high = targets[name]
            direction = "increase" if angle < low else "decrease"
            msg = HUMAN_FEEDBACK.get((name, direction))
            if msg is None:
                msg = f"Adjust your {name.lower()}"
            feedback.append(msg)

    scores = [
        joint_scores[name] for name in selected_names
        if joint_scores[name] is not None
    ]
    valid_count = len(scores)
    ready = valid_count >= required

    if mode == "upper_body" and not ready:
        waiting = (
            f"Need at least {required} visible upper-body joints "
            f"(currently {upper_count}/4). Keep both shoulders and arms visible."
        )
    elif mode == "full_body" and lower_count < 2:
        ready = False
        waiting = "Show your hips and knees for full-body pose matching."
    elif not ready:
        waiting = f"Need at least {required} visible full-body joints."
    else:
        waiting = ""

    if not ready:
        return {
            "ready": False,
            "score": None,
            "valid_joints": valid_count,
            "required_joints": required,
            "analysis_mode": mode,
            "feedback": [waiting],
            "joint_scores": joint_scores,
        }

    return {
        "ready": True,
        "score": round(sum(scores) / valid_count, 1),
        "valid_joints": valid_count,
        "required_joints": required,
        "analysis_mode": mode,
        "feedback": feedback[:3] or ["Great alignment! Hold this pose."],
        "joint_scores": joint_scores,
    }
