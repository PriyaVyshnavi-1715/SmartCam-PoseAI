"""
Tests for backend API endpoints, authentication, and database.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Override database to use in-memory SQLite for testing
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

from backend.database import init_db, SessionLocal, User, PoseSession, Capture, UserPreference
from backend.auth import hash_password, verify_password, create_access_token, decode_token
from backend.pose_library import get_pose_catalog, get_pose_by_name, get_poses_by_category


def test_password_hashing():
    pw = "testpassword123"
    hashed = hash_password(pw)
    assert hashed != pw
    assert verify_password(pw, hashed)
    assert not verify_password("wrong", hashed)


def test_jwt_tokens():
    token = create_access_token({"sub": "42"})
    assert isinstance(token, str)
    payload = decode_token(token)
    assert payload["sub"] == "42"
    assert "exp" in payload


def test_database_init():
    init_db()
    db = SessionLocal()
    try:
        # Create a test user
        user = User(name="Test User", email="test@example.com", password_hash=hash_password("pass123"))
        db.add(user)
        db.commit()
        db.refresh(user)
        assert user.id is not None
        assert user.name == "Test User"

        # Create preferences
        prefs = UserPreference(user_id=user.id, voice_enabled=True, language="en")
        db.add(prefs)
        db.commit()
        assert prefs.user_id == user.id

        # Create session
        session = PoseSession(user_id=user.id, pose_name="T-Pose", alignment_score=85.5, duration_seconds=120)
        db.add(session)
        db.commit()
        assert session.id is not None

        # Create capture
        capture = Capture(user_id=user.id, session_id=session.id, image_path="test.jpg", alignment_score=85.5, pose_name="T-Pose")
        db.add(capture)
        db.commit()
        assert capture.id is not None

        # Query relationships
        found = db.query(User).filter(User.email == "test@example.com").first()
        assert found is not None
        assert len(found.sessions) == 1
        assert len(found.captures) == 1

        # Cleanup
        db.delete(user)
        db.commit()
    finally:
        db.close()


def test_pose_catalog():
    catalog = get_pose_catalog()
    assert len(catalog) == 15
    names = [p["name"] for p in catalog]
    assert "T-Pose" in names
    assert "Mountain Pose" in names
    assert "Lunge Pose" in names


def test_pose_by_name():
    pose = get_pose_by_name("Warrior Pose")
    assert pose is not None
    assert pose["name"] == "Warrior Pose"
    assert pose["category"] == "Fitness"
    assert pose["difficulty"] == 2
    assert "instructions" in pose

    none_pose = get_pose_by_name("Nonexistent")
    assert none_pose is None


def test_pose_by_category():
    fitness = get_poses_by_category("Fitness")
    assert len(fitness) >= 4
    for p in fitness:
        assert p["category"] == "Fitness"

    standing = get_poses_by_category("Standing")
    assert len(standing) >= 2


def test_all_poses_have_metadata():
    catalog = get_pose_catalog()
    for pose in catalog:
        assert "name" in pose
        assert "category" in pose
        assert "difficulty" in pose
        assert "description" in pose
        assert "instructions" in pose
        assert "required_visible_joints" in pose
        assert 1 <= pose["difficulty"] <= 3
        assert len(pose["description"]) > 10
        assert len(pose["instructions"]) > 20


if __name__ == "__main__":
    test_password_hashing()
    test_jwt_tokens()
    test_database_init()
    test_pose_catalog()
    test_pose_by_name()
    test_pose_by_category()
    test_all_poses_have_metadata()
    print("All backend tests passed!")
