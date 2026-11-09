# Smart Camera with Real-Time Pose Suggestion and Alignment

An AI-powered photography assistant that detects your body in real-time, provides pose suggestions, and guides you to achieve perfect alignment with visual feedback, scoring, and voice guidance.

## Features

- **33-Point Body Detection** — MediaPipe Pose Landmarker with real-time skeleton overlay
- **15 Reference Poses** — Standing, sitting, fashion, portrait, creative, and fitness categories
- **Real-Time Alignment Scoring** — 0-100% score with joint-level color coding (green/yellow/red)
- **AI Correction Feedback** — Human-readable suggestions like "Raise your left arm higher"
- **Pose Stability Detection** — Knows when you've held a pose steady
- **Voice Guidance** — English and Telugu spoken corrections (browser TTS)
- **Photo Capture** — 3-second countdown, preview, save, retake, delete
- **Hand & Face Detection** — Gesture classification and facial expression recognition
- **Session History** — All practice sessions and captures stored in SQLite
- **Performance Analytics** — Track scores, sessions, and improvement over time
- **Authentication** — Register/login with JWT tokens and bcrypt password hashing
- **Dark/Light Theme** — Premium design with smooth animations

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, FastAPI, OpenCV, MediaPipe |
| Frontend | React, TypeScript, Vite |
| Database | SQLite (SQLAlchemy ORM) |
| Auth | JWT (python-jose), bcrypt (passlib) |
| CV Models | MediaPipe Pose, Hand, and Face Landmarker |

## Requirements

- **Python 3.10+**
- **Node.js 18+**
- **Webcam** (built-in or USB)
- **Windows 10/11** (tested on Windows; Linux/macOS should work with minor changes)

## Installation & Setup (Windows)

### Step 1: Extract the project

Extract the ZIP file to any location (e.g., `C:\Projects\`).

### Step 2: Install Python dependencies

Open a **terminal** (PowerShell or Command Prompt) in the project folder:

```bash
cd Smart_Camera_with_Real_Time_Pose_Suggestion_and_Alignment

# Create virtual environment
python -m venv .venv

# Activate it
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Download MediaPipe model files

Download these 3 files into the `assets/` folder:

1. **Pose Landmarker**: https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
2. **Hand Landmarker**: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
3. **Face Landmarker**: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task

```bash
# Or use curl/wget:
mkdir assets 2>nul
curl -o assets/pose_landmarker_lite.task "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
curl -o assets/hand_landmarker.task "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
curl -o assets/face_landmarker.task "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
```

### Step 4: Create environment file

```bash
copy .env.example .env
```

Edit `.env` and set a random secret key (or keep the default for local development).

### Step 5: Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

### Step 6: Run the application

**Terminal 1 — Start the backend:**

```bash
.venv\Scripts\activate
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

**Terminal 2 — Start the frontend:**

```bash
cd frontend
npm run dev
```

### Step 7: Open the application

Open your browser and go to: **http://localhost:5173**

1. Click "Get Started" to register a new account
2. Go to "Smart Camera" and click "Start Camera"
3. Allow webcam access when prompted
4. Select a pose from the list and follow the alignment feedback
5. When your score is above 70% and the pose is stable, click "Capture"

## Running Tests

```bash
.venv\Scripts\activate

# Run all tests
python test_pose_matching.py
python test_landmark_analysis.py
python test_backend.py

# Or use pytest
pytest test_pose_matching.py test_landmark_analysis.py test_backend.py -v
```

## Project Structure

```
Smart_Camera_with_Real_Time_Pose_Suggestion_and_Alignment/
├── backend/
│   ├── __init__.py          # Package init
│   ├── main.py              # FastAPI app with all API routes
│   ├── pose_engine.py       # OpenCV + MediaPipe camera pipeline
│   ├── pose_library.py      # 15 pose metadata catalog
│   ├── database.py          # SQLAlchemy models (User, Session, Capture, Preferences)
│   └── auth.py              # JWT + bcrypt authentication
├── frontend/
│   ├── src/
│   │   ├── App.tsx           # Complete React SPA (all pages)
│   │   ├── App.css           # Premium design system
│   │   ├── main.tsx          # React entry point
│   │   └── index.css         # Base styles
│   ├── index.html
│   ├── vite.config.ts        # Vite config with API proxy
│   └── package.json
├── assets/                   # MediaPipe .task model files
├── captures/                 # Saved photos (created at runtime)
├── landmark_analysis.py      # Joint angle calculation & stability detection
├── pose_matching.py          # 15 reference poses with angle ranges
├── sitting_posture.py        # Sitting posture analysis
├── test_pose_matching.py     # Pose matching tests
├── test_landmark_analysis.py # Landmark analysis tests
├── test_backend.py           # Backend + database tests
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variable template
├── .gitignore
└── README.md
```

## API Endpoints

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | /api/health | Server health check | No |
| POST | /api/auth/register | Register new user | No |
| POST | /api/auth/login | Login | No |
| GET | /api/auth/me | Current user info | Yes |
| GET | /api/poses | List pose names | No |
| GET | /api/poses/catalog | Full pose catalog with metadata | No |
| POST | /api/camera/start | Start webcam | No |
| POST | /api/camera/stop | Stop webcam | Optional |
| GET | /api/camera/stream | MJPEG video stream | No |
| GET | /api/results | Latest analysis results | No |
| POST | /api/pose/select | Select target pose | No |
| POST | /api/camera/capture | Take a photo | Optional |
| GET | /api/captures | List user's captures | Yes |
| DELETE | /api/captures/{id} | Delete a capture | Yes |
| GET | /api/sessions | List practice sessions | Yes |
| POST | /api/sessions | Save a session | Yes |
| GET | /api/analytics/summary | Performance summary | Yes |
| GET | /api/analytics/daily | Daily performance data | Yes |
| GET | /api/preferences | Get user preferences | Yes |
| PUT | /api/preferences | Update preferences | Yes |

## License

This project is for educational purposes.
