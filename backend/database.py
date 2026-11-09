"""
Database models and session management for Smart Camera application.
Uses SQLAlchemy with SQLite for local data persistence.
"""

import os
from datetime import datetime, timezone
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean,
    DateTime, Text, ForeignKey
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./smart_camera.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def _utcnow():
    return datetime.now(timezone.utc)


# ---------------------
# USER MODEL
# ---------------------

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=_utcnow)

    sessions = relationship("PoseSession", back_populates="user", cascade="all, delete-orphan")
    captures = relationship("Capture", back_populates="user", cascade="all, delete-orphan")
    preferences = relationship("UserPreference", back_populates="user", uselist=False, cascade="all, delete-orphan")


# ---------------------
# POSE SESSION MODEL
# ---------------------

class PoseSession(Base):
    __tablename__ = "pose_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    pose_name = Column(String(100), nullable=False)
    alignment_score = Column(Float, nullable=True)
    duration_seconds = Column(Integer, default=0)
    feedback_summary = Column(Text, nullable=True)
    created_at = Column(DateTime, default=_utcnow)

    user = relationship("User", back_populates="sessions")
    captures = relationship("Capture", back_populates="session", cascade="all, delete-orphan")


# ---------------------
# CAPTURE MODEL
# ---------------------

class Capture(Base):
    __tablename__ = "captures"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    session_id = Column(Integer, ForeignKey("pose_sessions.id"), nullable=True)
    image_path = Column(String(500), nullable=False)
    alignment_score = Column(Float, nullable=True)
    pose_name = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=_utcnow)

    user = relationship("User", back_populates="captures")
    session = relationship("PoseSession", back_populates="captures")


# ---------------------
# USER PREFERENCES MODEL
# ---------------------

class UserPreference(Base):
    __tablename__ = "user_preferences"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    voice_enabled = Column(Boolean, default=True)
    language = Column(String(20), default="en")
    speech_rate = Column(Float, default=1.0)
    dark_mode = Column(Boolean, default=True)

    user = relationship("User", back_populates="preferences")


# ---------------------
# DATABASE INITIALIZATION
# ---------------------

def init_db():
    """Create all tables if they do not exist."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency for database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
