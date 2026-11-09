"""
Smart Camera with Real-Time Pose Suggestion and Alignment - Backend API
"""
import cv2
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.database import init_db, get_db, User, PoseSession, Capture, UserPreference
from backend.auth import (
    hash_password, verify_password, create_access_token,
    get_current_user, get_optional_user,
)
from backend.pose_engine import PoseEngine, CAPTURES_DIR
from backend.pose_library import get_pose_catalog, get_pose_by_name
from pose_matching import REFERENCE_POSES


# =============================
# CAMERA MANAGER
# =============================

class CameraManager:
    def __init__(self):
        self.engine = PoseEngine()
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.worker = None
        self.running = False
        self.latest_frame = None
        self.latest_results = {
            "body_detected": False, "selected_pose": "T-Pose",
            "match_ready": False, "match_score": 0,
            "visible_joints": 0, "joint_angles": {},
            "feedback": ["Waiting for camera..."]
        }
        self.error = None
        self.target_pose = "T-Pose"
        self.session_start_time = None

    def start(self):
        if self.running:
            return
        try:
            with self.lock:
                self.engine.start()
            self.stop_event.clear()
            self.error = None
            self.running = True
            self.session_start_time = time.time()
            self.worker = threading.Thread(target=self._camera_loop, daemon=True)
            self.worker.start()
        except Exception as exc:
            self.running = False
            self.error = str(exc)
            raise RuntimeError(str(exc))

    def _camera_loop(self):
        while not self.stop_event.is_set():
            try:
                with self.lock:
                    self.engine.set_target_pose(self.target_pose)
                    frame, results = self.engine.process_frame()
                if frame is None:
                    time.sleep(0.02)
                    continue
                success, buffer = cv2.imencode(".jpg", frame)
                if success:
                    with self.lock:
                        self.latest_frame = buffer.tobytes()
                        self.latest_results = results
            except Exception as exc:
                self.error = str(exc)
                self.running = False
                break
            time.sleep(0.01)

    def stop(self):
        self.stop_event.set()
        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=3)
        with self.lock:
            self.engine.stop()
            self.latest_frame = None
        duration = 0
        if self.session_start_time:
            duration = int(time.time() - self.session_start_time)
        self.running = False
        self.worker = None
        self.session_start_time = None
        return duration

    def select_pose(self, pose_name):
        if pose_name not in REFERENCE_POSES:
            raise ValueError("Invalid pose name")
        self.target_pose = pose_name

    def get_results(self):
        with self.lock:
            return self.latest_results.copy() if self.latest_results else {}

    def capture(self):
        with self.lock:
            return self.engine.capture_frame()


camera = CameraManager()


# =============================
# APPLICATION LIFECYCLE
# =============================

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
    yield
    if camera.running:
        camera.stop()


app = FastAPI(
    title="Smart Camera API",
    description="Real-Time Pose Suggestion and Alignment",
    version="2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================
# REQUEST / RESPONSE MODELS
# =============================

class RegisterRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=6, max_length=128)

class LoginRequest(BaseModel):
    email: str
    password: str

class PoseRequest(BaseModel):
    pose: str

class SessionSaveRequest(BaseModel):
    pose_name: str
    alignment_score: float | None = None
    duration_seconds: int = 0
    feedback_summary: str | None = None

class PreferencesUpdate(BaseModel):
    voice_enabled: bool | None = None
    language: str | None = None
    speech_rate: float | None = None
    dark_mode: bool | None = None


# =============================
# BASIC ENDPOINTS
# =============================

@app.get("/")
def home():
    return {"message": "Smart Camera API", "status": "running"}

@app.get("/api/health")
def health():
    return {"status": "healthy", "camera_running": camera.running}


# =============================
# AUTHENTICATION
# =============================

@app.post("/api/auth/register")
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == req.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(
        name=req.name,
        email=req.email,
        password_hash=hash_password(req.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    # Create default preferences
    prefs = UserPreference(user_id=user.id)
    db.add(prefs)
    db.commit()
    token = create_access_token({"sub": str(user.id)})
    return {
        "user_id": user.id, "name": user.name, "email": user.email,
        "token": token,
    }

@app.post("/api/auth/login")
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_access_token({"sub": str(user.id)})
    return {
        "user_id": user.id, "name": user.name, "email": user.email,
        "token": token,
    }

@app.get("/api/auth/me")
def get_me(user: User = Depends(get_current_user)):
    return {"user_id": user.id, "name": user.name, "email": user.email}


# =============================
# POSES
# =============================

@app.get("/api/poses")
def get_poses():
    return {"poses": list(REFERENCE_POSES.keys())}

@app.get("/api/poses/catalog")
def get_poses_catalog():
    return {"poses": get_pose_catalog()}

@app.get("/api/poses/{name}")
def get_pose_detail(name: str):
    pose = get_pose_by_name(name)
    if pose is None:
        raise HTTPException(status_code=404, detail="Pose not found")
    return pose


# =============================
# CAMERA
# =============================

@app.post("/api/camera/start")
def start_camera():
    try:
        camera.start()
        return {"success": True, "message": "Camera started"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not start camera: {exc}")

@app.post("/api/camera/stop")
def stop_camera(
    user: User = Depends(get_optional_user),
    db: Session = Depends(get_db),
):
    try:
        duration = camera.stop()
        # Auto-save session if user is logged in
        if user and duration > 5:
            results = camera.get_results()
            session = PoseSession(
                user_id=user.id,
                pose_name=results.get("selected_pose", camera.target_pose),
                alignment_score=results.get("match_score"),
                duration_seconds=duration,
                feedback_summary="; ".join(results.get("feedback", [])),
            )
            db.add(session)
            db.commit()
        return {"success": True, "message": "Camera stopped", "duration": duration}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/api/pose/select")
def select_pose(request: PoseRequest):
    try:
        camera.select_pose(request.pose)
        return {"success": True, "selected_pose": request.pose}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# =============================
# CAMERA STREAM
# =============================

def video_generator():
    while True:
        if not camera.running:
            break
        with camera.lock:
            frame = camera.latest_frame
        if frame is None:
            time.sleep(0.03)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame
            + b"\r\n"
        )
        time.sleep(0.03)

@app.get("/api/camera/stream")
def camera_stream():
    if not camera.running:
        raise HTTPException(status_code=400, detail="Start the camera first")
    return StreamingResponse(
        video_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

@app.get("/api/results")
def get_results():
    return {
        "camera_running": camera.running,
        "error": camera.error,
        "results": camera.get_results(),
    }


# =============================
# CAPTURE
# =============================

@app.post("/api/camera/capture")
def capture_photo(
    user: User = Depends(get_optional_user),
    db: Session = Depends(get_db),
):
    if not camera.running:
        raise HTTPException(status_code=400, detail="Camera is not running")
    result = camera.capture()
    if result is None:
        raise HTTPException(status_code=500, detail="No frame available for capture")
    # Save to database if user is logged in
    if user:
        capture = Capture(
            user_id=user.id,
            image_path=result["filename"],
            alignment_score=result.get("alignment_score"),
            pose_name=result.get("pose_name"),
        )
        db.add(capture)
        db.commit()
        db.refresh(capture)
        result["capture_id"] = capture.id
    return result

@app.get("/api/captures")
def list_captures(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    captures = (
        db.query(Capture)
        .filter(Capture.user_id == user.id)
        .order_by(Capture.created_at.desc())
        .all()
    )
    return {
        "captures": [
            {
                "id": c.id,
                "filename": c.image_path,
                "alignment_score": c.alignment_score,
                "pose_name": c.pose_name,
                "created_at": c.created_at.isoformat() if c.created_at else None,
            }
            for c in captures
        ]
    }

@app.get("/api/captures/{capture_id}/image")
def get_capture_image(capture_id: int, db: Session = Depends(get_db)):
    capture = db.query(Capture).filter(Capture.id == capture_id).first()
    if not capture:
        raise HTTPException(status_code=404, detail="Capture not found")
    filepath = CAPTURES_DIR / capture.image_path
    if not filepath.is_file():
        raise HTTPException(status_code=404, detail="Image file not found")
    return FileResponse(str(filepath), media_type="image/jpeg")

@app.delete("/api/captures/{capture_id}")
def delete_capture(
    capture_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    capture = db.query(Capture).filter(
        Capture.id == capture_id, Capture.user_id == user.id
    ).first()
    if not capture:
        raise HTTPException(status_code=404, detail="Capture not found")
    filepath = CAPTURES_DIR / capture.image_path
    if filepath.is_file():
        filepath.unlink()
    db.delete(capture)
    db.commit()
    return {"success": True}


# =============================
# SESSIONS
# =============================

@app.get("/api/sessions")
def list_sessions(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    sessions = (
        db.query(PoseSession)
        .filter(PoseSession.user_id == user.id)
        .order_by(PoseSession.created_at.desc())
        .limit(50)
        .all()
    )
    return {
        "sessions": [
            {
                "id": s.id,
                "pose_name": s.pose_name,
                "alignment_score": s.alignment_score,
                "duration_seconds": s.duration_seconds,
                "feedback_summary": s.feedback_summary,
                "created_at": s.created_at.isoformat() if s.created_at else None,
            }
            for s in sessions
        ]
    }

@app.post("/api/sessions")
def save_session(
    req: SessionSaveRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    session = PoseSession(
        user_id=user.id,
        pose_name=req.pose_name,
        alignment_score=req.alignment_score,
        duration_seconds=req.duration_seconds,
        feedback_summary=req.feedback_summary,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return {"session_id": session.id}

@app.delete("/api/sessions/{session_id}")
def delete_session(
    session_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    session = db.query(PoseSession).filter(
        PoseSession.id == session_id, PoseSession.user_id == user.id
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    db.delete(session)
    db.commit()
    return {"success": True}


# =============================
# ANALYTICS
# =============================

@app.get("/api/analytics/summary")
def analytics_summary(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    sessions = db.query(PoseSession).filter(PoseSession.user_id == user.id).all()
    captures = db.query(Capture).filter(Capture.user_id == user.id).count()

    if not sessions:
        return {
            "total_sessions": 0,
            "photos_captured": captures,
            "avg_score": None,
            "best_score": None,
            "most_practiced": None,
            "total_duration_minutes": 0,
        }

    scored = [s for s in sessions if s.alignment_score is not None]
    avg_score = round(sum(s.alignment_score for s in scored) / len(scored), 1) if scored else None
    best_score = round(max(s.alignment_score for s in scored), 1) if scored else None
    total_duration = sum(s.duration_seconds or 0 for s in sessions)

    # Most practiced pose
    pose_counts = {}
    for s in sessions:
        pose_counts[s.pose_name] = pose_counts.get(s.pose_name, 0) + 1
    most_practiced = max(pose_counts, key=pose_counts.get) if pose_counts else None

    return {
        "total_sessions": len(sessions),
        "photos_captured": captures,
        "avg_score": avg_score,
        "best_score": best_score,
        "most_practiced": most_practiced,
        "total_duration_minutes": round(total_duration / 60, 1),
    }

@app.get("/api/analytics/daily")
def analytics_daily(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    sessions = (
        db.query(PoseSession)
        .filter(PoseSession.user_id == user.id)
        .order_by(PoseSession.created_at.desc())
        .limit(30)
        .all()
    )
    daily = {}
    for s in sessions:
        if s.created_at:
            day = s.created_at.strftime("%Y-%m-%d")
            if day not in daily:
                daily[day] = {"date": day, "sessions": 0, "avg_score": 0, "scores": []}
            daily[day]["sessions"] += 1
            if s.alignment_score is not None:
                daily[day]["scores"].append(s.alignment_score)

    result = []
    for day_data in daily.values():
        scores = day_data.pop("scores")
        day_data["avg_score"] = round(sum(scores) / len(scores), 1) if scores else None
        result.append(day_data)

    return {"daily": sorted(result, key=lambda x: x["date"])}


# =============================
# USER PREFERENCES
# =============================

@app.get("/api/preferences")
def get_preferences(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
    if not prefs:
        prefs = UserPreference(user_id=user.id)
        db.add(prefs)
        db.commit()
        db.refresh(prefs)
    return {
        "voice_enabled": prefs.voice_enabled,
        "language": prefs.language,
        "speech_rate": prefs.speech_rate,
        "dark_mode": prefs.dark_mode,
    }

@app.put("/api/preferences")
def update_preferences(
    req: PreferencesUpdate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
    if not prefs:
        prefs = UserPreference(user_id=user.id)
        db.add(prefs)
    if req.voice_enabled is not None:
        prefs.voice_enabled = req.voice_enabled
    if req.language is not None:
        prefs.language = req.language
    if req.speech_rate is not None:
        prefs.speech_rate = req.speech_rate
    if req.dark_mode is not None:
        prefs.dark_mode = req.dark_mode
    db.commit()
    return {"success": True}


# =============================
# USER PROFILE
# =============================

@app.delete("/api/auth/account")
def delete_account(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Delete user's captures from disk
    captures = db.query(Capture).filter(Capture.user_id == user.id).all()
    for c in captures:
        filepath = CAPTURES_DIR / c.image_path
        if filepath.is_file():
            filepath.unlink()
    db.delete(user)
    db.commit()
    return {"success": True, "message": "Account and all data deleted"}
