"""
Pose library with metadata for all 15 reference poses.
Provides structured information for the frontend UI.
"""

POSE_CATALOG = {
    "T-Pose": {
        "name": "T-Pose",
        "category": "Creative",
        "difficulty": 1,
        "description": "Stand with arms extended horizontally to form a T shape.",
        "instructions": "Stand upright with feet together. Extend both arms straight out to the sides at shoulder height. Keep your palms facing down and fingers extended.",
        "required_visible_joints": ["Left Elbow", "Right Elbow", "Left Shoulder", "Right Shoulder"],
    },
    "Standing Pose": {
        "name": "Standing Pose",
        "category": "Standing",
        "difficulty": 1,
        "description": "A natural upright stance with arms relaxed at your sides.",
        "instructions": "Stand tall with feet hip-width apart. Let your arms hang naturally at your sides. Keep your shoulders relaxed and look straight ahead.",
        "required_visible_joints": ["Left Shoulder", "Right Shoulder", "Left Hip", "Right Hip"],
    },
    "Sitting Pose": {
        "name": "Sitting Pose",
        "category": "Sitting",
        "difficulty": 1,
        "description": "A seated position with back straight and arms resting on knees.",
        "instructions": "Sit on a chair with your back straight. Place your hands on your knees with elbows slightly bent. Keep your feet flat on the floor.",
        "required_visible_joints": ["Left Elbow", "Right Elbow", "Left Shoulder", "Right Shoulder"],
    },
    "Warrior Pose": {
        "name": "Warrior Pose",
        "category": "Fitness",
        "difficulty": 2,
        "description": "A strong lunge position with arms extended horizontally.",
        "instructions": "Step one foot forward into a deep lunge. Bend your front knee to about 90 degrees while keeping your back leg straight. Extend both arms out to the sides at shoulder height.",
        "required_visible_joints": ["Left Shoulder", "Right Shoulder", "Left Hip", "Right Hip", "Left Knee", "Right Knee"],
    },
    "Squat Pose": {
        "name": "Squat Pose",
        "category": "Fitness",
        "difficulty": 2,
        "description": "A deep squat with hips below knee level.",
        "instructions": "Stand with feet shoulder-width apart. Lower your hips back and down as if sitting in a chair. Keep your chest up and arms extended forward for balance.",
        "required_visible_joints": ["Left Hip", "Right Hip", "Left Knee", "Right Knee"],
    },
    "Tree Pose": {
        "name": "Tree Pose",
        "category": "Fitness",
        "difficulty": 3,
        "description": "A single-leg balance pose with one foot resting on the opposite thigh.",
        "instructions": "Stand on your right leg. Place your left foot on your right inner thigh. Bring your hands together in front of your chest or raise them above your head.",
        "required_visible_joints": ["Left Hip", "Right Hip", "Left Knee", "Right Knee"],
    },
    "Mountain Pose": {
        "name": "Mountain Pose",
        "category": "Standing",
        "difficulty": 1,
        "description": "A tall, grounded standing position with arms at your sides.",
        "instructions": "Stand with feet together and arms at your sides. Press your feet firmly into the ground. Lengthen your spine and relax your shoulders down.",
        "required_visible_joints": ["Left Shoulder", "Right Shoulder", "Left Hip", "Right Hip"],
    },
    "Victory Pose": {
        "name": "Victory Pose",
        "category": "Creative",
        "difficulty": 1,
        "description": "Arms raised overhead in a V shape celebrating victory.",
        "instructions": "Stand tall with feet hip-width apart. Raise both arms up and outward to form a V shape above your head. Keep your elbows straight and look up slightly.",
        "required_visible_joints": ["Left Elbow", "Right Elbow", "Left Shoulder", "Right Shoulder"],
    },
    "Side Stretch": {
        "name": "Side Stretch",
        "category": "Fitness",
        "difficulty": 2,
        "description": "A lateral body stretch with one arm reaching overhead.",
        "instructions": "Stand with feet wider than hip-width. Raise your right arm overhead and lean your torso to the left. Keep your left arm relaxed at your side. Feel the stretch along your right side.",
        "required_visible_joints": ["Left Shoulder", "Right Shoulder", "Left Hip", "Right Hip"],
    },
    "Model Pose": {
        "name": "Model Pose",
        "category": "Fashion",
        "difficulty": 2,
        "description": "A confident fashion pose with one hand on your hip.",
        "instructions": "Stand with one foot slightly forward. Place your left hand on your hip with elbow bent. Let your right arm hang naturally. Shift your weight to your back foot.",
        "required_visible_joints": ["Left Elbow", "Right Elbow", "Left Shoulder", "Right Shoulder"],
    },
    "Confident Stance": {
        "name": "Confident Stance",
        "category": "Fashion",
        "difficulty": 1,
        "description": "A power pose with both hands on your hips.",
        "instructions": "Stand with feet shoulder-width apart. Place both hands on your hips with elbows pointing outward. Stand tall with your chest open and chin slightly raised.",
        "required_visible_joints": ["Left Elbow", "Right Elbow", "Left Shoulder", "Right Shoulder"],
    },
    "Hip Pop": {
        "name": "Hip Pop",
        "category": "Fashion",
        "difficulty": 2,
        "description": "A dynamic fashion pose with one hip shifted out to the side.",
        "instructions": "Stand with feet together. Shift your weight to one leg and push that hip out to the side. Place one hand on your hip and let the other arm hang relaxed.",
        "required_visible_joints": ["Left Elbow", "Left Shoulder", "Right Shoulder", "Left Hip", "Right Hip"],
    },
    "Head Tilt": {
        "name": "Head Tilt",
        "category": "Portrait",
        "difficulty": 1,
        "description": "A relaxed portrait pose with a gentle head tilt.",
        "instructions": "Stand or sit with a relaxed posture. Tilt your head slightly to one side. Keep your arms relaxed at your sides and maintain a natural expression.",
        "required_visible_joints": ["Left Shoulder", "Right Shoulder"],
    },
    "Star Pose": {
        "name": "Star Pose",
        "category": "Creative",
        "difficulty": 2,
        "description": "A wide X shape with arms and legs spread apart.",
        "instructions": "Stand with feet wide apart, wider than shoulder-width. Raise both arms up and out to form an X or star shape. Keep your elbows and knees straight.",
        "required_visible_joints": ["Left Shoulder", "Right Shoulder", "Left Hip", "Right Hip", "Left Knee", "Right Knee"],
    },
    "Lunge Pose": {
        "name": "Lunge Pose",
        "category": "Fitness",
        "difficulty": 2,
        "description": "A forward lunge with one knee bent at 90 degrees.",
        "instructions": "Step your left foot forward into a deep lunge. Bend your left knee to about 90 degrees. Keep your right leg extended behind you. Arms can rest at your sides or on your hips.",
        "required_visible_joints": ["Left Hip", "Right Hip", "Left Knee", "Right Knee"],
    },
}

CATEGORIES = ["Standing", "Sitting", "Fashion", "Portrait", "Creative", "Fitness"]


def get_pose_catalog():
    """Return the full pose catalog as a list of dicts."""
    return list(POSE_CATALOG.values())


def get_pose_by_name(name):
    """Return a single pose dict by name, or None if not found."""
    return POSE_CATALOG.get(name)


def get_poses_by_category(category):
    """Return all poses in a given category."""
    return [p for p in POSE_CATALOG.values() if p["category"] == category]
