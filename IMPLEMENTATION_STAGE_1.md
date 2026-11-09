# PoseAI — Stage 1 implementation

## Included
- Real page-specific rendering for Overview, Live Analysis, Pose Matching, Exercise Tracking, Analytics, History, and Profile.
- Functional sidebar navigation between those pages.
- Dark/light theme toggle in the top bar and Profile settings; theme preference is stored in this browser.
- Editable local workspace display name.
- Pose Matching page uses the existing `/api/pose/select` flow and displays live joint measurements and feedback.
- Exercise Tracking page reports readiness but does not fabricate repetition counts.
- Analytics and History show honest empty states because session persistence is not implemented yet.
- Existing camera stream and CV engine source retained.

## Run
Backend (from project root):
```powershell
python -m uvicorn backend.main:app --reload
```

Frontend (from `frontend`):
```powershell
npm install
npm run dev
```

## Validation
- Python backend and CV source files passed `py_compile`.
- Frontend dependency installation could not complete in the build environment, so `npm run build` / TypeScript compilation was not verified here. Run `npm install` then `npm run build` locally.
- The current backend does not yet save sessions or calculate exercise repetitions. Those are future implementation stages, not completed features.
