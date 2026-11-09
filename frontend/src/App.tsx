import { useEffect, useState, useRef, useCallback } from 'react'
import './App.css'


// ─── Types ──────────────────────────────────────────────
type SittingAnalysis = {
  ready: boolean; shoulder_tilt_degrees: number | null
  head_offset: number | null; feedback: string[]
}
type AnalysisResults = {
  body_detected: boolean; selected_pose: string
  match_ready: boolean; match_score: number | null
  visible_joints: number; analysis_mode: string; required_joints: number
  joint_angles: Record<string, number | null>
  joint_scores: Record<string, number | null>
  hand_count: number; hand_gestures: string[]
  face_detected: boolean; expression: string
  feedback: string[]; pose_stable: boolean
  sitting_analysis: SittingAnalysis
}
type PoseInfo = {
  name: string; category: string; difficulty: number
  description: string; instructions: string
  required_visible_joints: string[]
}
type SessionRecord = {
  id: number; pose_name: string; alignment_score: number | null
  duration_seconds: number; feedback_summary: string | null; created_at: string
}
type CaptureRecord = {
  id: number; filename: string; alignment_score: number | null
  pose_name: string | null; created_at: string | null
}
type Analytics = {
  total_sessions: number; photos_captured: number
  avg_score: number | null; best_score: number | null
  most_practiced: string | null; total_duration_minutes: number
}

const emptyResults: AnalysisResults = {
  body_detected: false, selected_pose: 'T-Pose', match_ready: false,
  match_score: null, visible_joints: 0, analysis_mode: 'full_body',
  required_joints: 5, joint_angles: {}, joint_scores: {},
  hand_count: 0, hand_gestures: [], face_detected: false,
  expression: 'NO FACE', feedback: ['Start the camera to begin analysis.'],
  pose_stable: false,
  sitting_analysis: { ready: false, shoulder_tilt_degrees: null, head_offset: null, feedback: [] }
}

const JOINTS = ['Left Elbow','Right Elbow','Left Shoulder','Right Shoulder','Left Hip','Right Hip','Left Knee','Right Knee']
const CATEGORIES = ['All','Standing','Sitting','Fashion','Portrait','Creative','Fitness']
const DIFFICULTY_LABEL = ['','Easy','Medium','Hard']

// ─── Voice Guidance ─────────────────────────────────────
class VoiceGuide {
  enabled = false; lang = 'en'; rate = 1.0; lastMsg = ''; lastTime = 0
  speak(text: string) {
    if (!this.enabled || !window.speechSynthesis) return
    const now = Date.now()
    if (text === this.lastMsg && now - this.lastTime < 4000) return
    window.speechSynthesis.cancel()
    const u = new SpeechSynthesisUtterance(text)
    u.rate = this.rate
    if (this.lang === 'te') {
      const voices = window.speechSynthesis.getVoices()
      const te = voices.find(v => v.lang.startsWith('te'))
      if (te) u.voice = te; else u.lang = 'en-US'
    } else { u.lang = 'en-US' }
    window.speechSynthesis.speak(u)
    this.lastMsg = text; this.lastTime = now
  }
}
const voice = new VoiceGuide()

// ─── App Component ──────────────────────────────────────
function App() {
  // Auth
  const [page, setPage] = useState('landing')
  const [token, setToken] = useState(() => localStorage.getItem('sc-token') || '')
  const [userName, setUserName] = useState(() => localStorage.getItem('sc-name') || '')
  const [authMode, setAuthMode] = useState<'login'|'register'>('login')
  const [authEmail, setAuthEmail] = useState('')
  const [authPass, setAuthPass] = useState('')
  const [authName, setAuthName] = useState('')
  const [authError, setAuthError] = useState('')

  // Camera
  const [selectedPose, setSelectedPose] = useState('T-Pose')
  const [backendStatus, setBackendStatus] = useState('Checking...')
  const [apiPoses, setApiPoses] = useState<string[]>([])
  const [poseCatalog, setPoseCatalog] = useState<PoseInfo[]>([])
  const [cameraRunning, setCameraRunning] = useState(false)
  const [cameraLoading, setCameraLoading] = useState(false)
  const [cameraError, setCameraError] = useState('')
  const [analysis, setAnalysis] = useState<AnalysisResults>(emptyResults)

  // Settings
  const [theme, setTheme] = useState<'dark'|'light'>(() => {
    const s = localStorage.getItem('sc-theme'); return s === 'light' ? 'light' : 'dark'
  })
  const [voiceEnabled, setVoiceEnabled] = useState(true)
  const [voiceLang, setVoiceLang] = useState('en')
  const [speechRate, setSpeechRate] = useState(1.0)

  // Data
  const [sessions, setSessions] = useState<SessionRecord[]>([])
  const [captures, setCaptures] = useState<CaptureRecord[]>([])
  const [analytics, setAnalytics] = useState<Analytics|null>(null)
  const [poseFilter, setPoseFilter] = useState('All')

  // Capture
  const [captureCountdown, setCaptureCountdown] = useState(0)
  const [lastCapture, setLastCapture] = useState<CaptureRecord|null>(null)
  const countdownRef = useRef<number>(0)

  // Theme
  useEffect(() => {
    document.documentElement.dataset.theme = theme
    localStorage.setItem('sc-theme', theme)
  }, [theme])

  // Voice settings sync
  useEffect(() => { voice.enabled = voiceEnabled; voice.lang = voiceLang; voice.rate = speechRate }, [voiceEnabled, voiceLang, speechRate])

  // Auth headers helper
  const authHeaders = useCallback((): HeadersInit => {
    const h: HeadersInit = { 'Content-Type': 'application/json' }
    if (token) h['Authorization'] = `Bearer ${token}`
    return h
  }, [token])

  // Check backend
  useEffect(() => {
    fetch('/api/health').then(r => r.json()).then(d => setBackendStatus(d.status)).catch(() => setBackendStatus('Offline'))
    fetch('/api/poses').then(r => r.json()).then(d => setApiPoses(d.poses)).catch(() => {})
    fetch('/api/poses/catalog').then(r => r.json()).then(d => setPoseCatalog(d.poses)).catch(() => {})
  }, [])

  // Auto-login check
  useEffect(() => {
    if (!token) return
    fetch('/api/auth/me', { headers: authHeaders() })
      .then(r => { if (!r.ok) throw new Error(); return r.json() })
      .then(d => { setUserName(d.name); setPage('dashboard') })
      .catch(() => { setToken(''); localStorage.removeItem('sc-token') })
  }, []) // eslint-disable-line

  // Poll results
  useEffect(() => {
    if (!cameraRunning) return
    let cancelled = false
    const poll = async () => {
      try {
        const r = await fetch('/api/results')
        if (!r.ok) throw new Error()
        const d = await r.json()
        if (!cancelled) {
          const res = d.results ?? {}
          setAnalysis({ ...emptyResults, ...res,
            joint_angles: { ...emptyResults.joint_angles, ...(res.joint_angles ?? {}) },
            joint_scores: { ...(res.joint_scores ?? {}) },
            sitting_analysis: { ...emptyResults.sitting_analysis, ...(res.sitting_analysis ?? {}) },
          })
          if (d.error) setCameraError(d.error)
          // Voice feedback
          if (res.feedback?.length && res.match_ready) voice.speak(res.feedback[0])
          if (res.pose_stable && res.match_score > 70) voice.speak('Pose is stable. Ready for capture!')
        }
      } catch { if (!cancelled) setCameraError('Lost connection to backend.') }
    }
    poll()
    const t = setInterval(poll, 500)
    return () => { cancelled = true; clearInterval(t) }
  }, [cameraRunning])

  // Load data when page changes
  useEffect(() => {
    if (!token) return
    if (page === 'history') {
      fetch('/api/sessions', { headers: authHeaders() }).then(r => r.json()).then(d => setSessions(d.sessions || [])).catch(() => {})
      fetch('/api/captures', { headers: authHeaders() }).then(r => r.json()).then(d => setCaptures(d.captures || [])).catch(() => {})
    }
    if (page === 'analytics') {
      fetch('/api/analytics/summary', { headers: authHeaders() }).then(r => r.json()).then(d => setAnalytics(d)).catch(() => {})
    }
  }, [page, token]) // eslint-disable-line

  const poses = apiPoses.length > 0 ? apiPoses : ['T-Pose','Standing Pose','Sitting Pose','Warrior Pose','Squat Pose','Tree Pose']
  const score = analysis.match_ready && analysis.match_score !== null ? analysis.match_score.toFixed(1) : '--'

  // ─── Auth ──────────────────────────
  const handleAuth = async (e: React.FormEvent) => {
    e.preventDefault(); setAuthError('')
    const endpoint = authMode === 'register' ? '/api/auth/register' : '/api/auth/login'
    const body = authMode === 'register'
      ? { name: authName, email: authEmail, password: authPass }
      : { email: authEmail, password: authPass }
    try {
      const r = await fetch(endpoint, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
      const d = await r.json()
      if (!r.ok) throw new Error(d.detail || 'Authentication failed')
      setToken(d.token); setUserName(d.name)
      localStorage.setItem('sc-token', d.token); localStorage.setItem('sc-name', d.name)
      setPage('dashboard')
    } catch (err) { setAuthError(err instanceof Error ? err.message : 'Error') }
  }

  const logout = () => {
    setToken(''); setUserName(''); localStorage.removeItem('sc-token'); localStorage.removeItem('sc-name')
    if (cameraRunning) stopCamera()
    setPage('landing')
  }

  // ─── Camera ────────────────────────
  const startCamera = async () => {
    setCameraLoading(true); setCameraError('')
    try {
      let r = await fetch('/api/camera/start', { method: 'POST' })
      let d = await r.json(); if (!r.ok) throw new Error(d.detail)
      r = await fetch('/api/pose/select', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ pose: selectedPose }) })
      d = await r.json(); if (!r.ok) throw new Error(d.detail)
      setCameraRunning(true)
    } catch (e) { setCameraError(e instanceof Error ? e.message : 'Camera failed'); setCameraRunning(false) }
    finally { setCameraLoading(false) }
  }
  const stopCamera = async () => {
    setCameraLoading(true)
    try {
      await fetch('/api/camera/stop', { method: 'POST', headers: authHeaders() })
      setCameraRunning(false); setAnalysis(emptyResults)
    } catch {} finally { setCameraLoading(false) }
  }
  const changePose = async (pose: string) => {
    setSelectedPose(pose)
    if (!cameraRunning) return
    try { await fetch('/api/pose/select', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ pose }) }) } catch {}
  }

  // ─── Capture ───────────────────────
  const startCapture = () => {
    setCaptureCountdown(3)
    let c = 3
    countdownRef.current = window.setInterval(() => {
      c--; setCaptureCountdown(c)
      if (c <= 0) {
        clearInterval(countdownRef.current)
        doCapture()
      }
    }, 1000)
  }
  const doCapture = async () => {
    try {
      const r = await fetch('/api/camera/capture', { method: 'POST', headers: authHeaders() })
      const d = await r.json()
      if (r.ok) { setLastCapture({ id: d.capture_id || 0, filename: d.filename, alignment_score: d.alignment_score, pose_name: d.pose_name, created_at: new Date().toISOString() }) }
    } catch {} finally { setCaptureCountdown(0) }
  }
  const deleteCapture = async (id: number) => {
    try { await fetch(`/api/captures/${id}`, { method: 'DELETE', headers: authHeaders() }) } catch {}
    setCaptures(prev => prev.filter(c => c.id !== id))
  }

  // ─── Filtered Poses ────────────────
  const filteredPoses = poseFilter === 'All' ? poseCatalog : poseCatalog.filter(p => p.category === poseFilter)

  // ─── NAV ───────────────────────────
  const nav = [
    { name: 'dashboard', label: 'Dashboard', icon: '⌂' },
    { name: 'camera', label: 'Smart Camera', icon: '◉' },
    { name: 'poses', label: 'Pose Library', icon: '♧' },
    { name: 'history', label: 'History', icon: '◷' },
    { name: 'analytics', label: 'Analytics', icon: '▥' },
    { name: 'settings', label: 'Settings', icon: '⚙' },
    { name: 'help', label: 'Help', icon: '?' },
  ]

  // ═══════════════════════════════════
  // LANDING PAGE
  // ═══════════════════════════════════
  if (page === 'landing') return (
    <div className="landing">
      <div className="landing-hero">
        <div className="landing-badge">AI-POWERED PHOTOGRAPHY</div>
        <h1>Smart Camera<br/><span>Pose Suggestion & Alignment</span></h1>
        <p>Real-time body detection, intelligent pose matching, and guided photography. Get perfect poses every time with AI-powered alignment scoring and voice guidance.</p>
        <div className="landing-buttons">
          <button className="btn-primary" onClick={() => setPage('auth')}>Get Started →</button>
          <button className="btn-secondary" onClick={() => setPage('auth')}>Sign In</button>
        </div>
        <div className="landing-features">
          <div className="lf-card"><span>◎</span><strong>33 Body Landmarks</strong><p>MediaPipe detection with confidence filtering</p></div>
          <div className="lf-card"><span>♧</span><strong>15 Reference Poses</strong><p>Standing, sitting, fashion, fitness & creative</p></div>
          <div className="lf-card"><span>▥</span><strong>Real-Time Scoring</strong><p>0-100% alignment with joint-level feedback</p></div>
          <div className="lf-card"><span>⚙</span><strong>Voice Guidance</strong><p>English & Telugu spoken corrections</p></div>
        </div>
      </div>
    </div>
  )

  // ═══════════════════════════════════
  // AUTH PAGE
  // ═══════════════════════════════════
  if (page === 'auth') return (
    <div className="auth-page">
      <div className="auth-card">
        <div className="auth-brand">
          <div className="brand-icon">✦</div>
          <h2>Smart Camera</h2>
          <p>Pose Suggestion & Alignment</p>
        </div>
        <div className="auth-tabs">
          <button className={authMode === 'login' ? 'active' : ''} onClick={() => setAuthMode('login')}>Sign In</button>
          <button className={authMode === 'register' ? 'active' : ''} onClick={() => setAuthMode('register')}>Register</button>
        </div>
        <form onSubmit={handleAuth}>
          {authMode === 'register' && <input placeholder="Full Name" value={authName} onChange={e => setAuthName(e.target.value)} required />}
          <input type="email" placeholder="Email" value={authEmail} onChange={e => setAuthEmail(e.target.value)} required />
          <input type="password" placeholder="Password" value={authPass} onChange={e => setAuthPass(e.target.value)} required minLength={6} />
          {authError && <p className="auth-error">{authError}</p>}
          <button type="submit" className="btn-primary">{authMode === 'register' ? 'Create Account' : 'Sign In'} →</button>
        </form>
        <button className="btn-link" onClick={() => setPage('landing')}>← Back to home</button>
      </div>
    </div>
  )

  // ═══════════════════════════════════
  // MAIN APP LAYOUT
  // ═══════════════════════════════════
  return (
    <div className="app-layout">
      <aside className="sidebar">
        <div className="brand" onClick={() => setPage('dashboard')}>
          <div className="brand-icon">✦</div>
          <div><h2>SMART<span>CAM</span></h2><small>POSE AI</small></div>
        </div>
        <div className="nav-label">NAVIGATION</div>
        <nav>
          {nav.map(n => (
            <button key={n.name} className={`nav-item ${page === n.name ? 'active' : ''}`} onClick={() => setPage(n.name)}>
              <span className="nav-icon">{n.icon}</span>{n.label}
              {n.name === 'camera' && cameraRunning && <span className="live-dot" />}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="status-bar">
            <span className={`status-dot ${backendStatus === 'healthy' ? '' : 'muted'}`} />
            <span>Engine {backendStatus === 'healthy' ? 'Online' : backendStatus}</span>
          </div>
          <div className="profile">
            <div className="avatar">{userName.slice(0,1).toUpperCase()}</div>
            <div className="profile-copy"><strong>{userName}</strong><small>Active</small></div>
            <button className="btn-link" onClick={logout}>Logout</button>
          </div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div><h1>{nav.find(n => n.name === page)?.label || 'Dashboard'}</h1></div>
          <div className="topbar-right">
            <span className="engine-badge"><span className="status-dot" /> Engine {backendStatus === 'healthy' ? 'online' : 'offline'}</span>
            <button className="icon-button" onClick={() => setTheme(t => t === 'dark' ? 'light' : 'dark')}>{theme === 'dark' ? '☼' : '☾'}</button>
          </div>
        </header>

        {/* ─── DASHBOARD ─── */}
        {page === 'dashboard' && <>
          <section className="welcome-row">
            <div><h2>Welcome, {userName} <span>✦</span></h2><p>Your AI photography assistant is ready.</p></div>
          </section>
          <section className="stats-grid">
            <div className="stat-card" onClick={() => setPage('camera')}>
              <div className="stat-top"><span>POSE ACCURACY</span><span className="stat-icon purple">◎</span></div>
              <div className="stat-value">{score}<small>%</small></div>
              <p>{analysis.match_ready ? 'Live matching score' : 'Start camera to begin'}</p>
            </div>
            <div className="stat-card" onClick={() => setPage('history')}>
              <div className="stat-top"><span>SESSIONS</span><span className="stat-icon blue">◷</span></div>
              <div className="stat-value">{analytics?.total_sessions ?? '—'}</div>
              <p>{analytics?.total_sessions ? `${analytics.total_duration_minutes}min total` : 'No sessions yet'}</p>
            </div>
            <div className="stat-card" onClick={() => setPage('poses')}>
              <div className="stat-top"><span>POSES</span><span className="stat-icon green">♧</span></div>
              <div className="stat-value">{poses.length}</div>
              <p>Reference poses available</p>
            </div>
            <div className="stat-card" onClick={() => setPage('analytics')}>
              <div className="stat-top"><span>BEST SCORE</span><span className="stat-icon orange">⌁</span></div>
              <div className="stat-value">{analytics?.best_score?.toFixed(0) ?? '—'}<small>%</small></div>
              <p>{analytics?.most_practiced ? `Best at ${analytics.most_practiced}` : 'Practice to improve'}</p>
            </div>
          </section>
          <section className="quick-actions">
            <button className="btn-primary" onClick={() => { setPage('camera'); if (!cameraRunning) startCamera() }}>Open Smart Camera →</button>
            <button className="btn-secondary" onClick={() => setPage('poses')}>Browse Poses</button>
            <button className="btn-secondary" onClick={() => setPage('history')}>View History</button>
          </section>
        </>}

        {/* ─── SMART CAMERA ─── */}
        {page === 'camera' && <>
          <section className="content-grid">
            <div className="panel camera-panel">
              <div className="panel-heading">
                <div><h3>Live Pose Analysis</h3><p>Real-time camera and body alignment</p></div>
                <span className="coming-badge">{cameraRunning ? '● LIVE' : 'CAMERA OFF'}</span>
              </div>
              <div className="camera-placeholder">
                {cameraRunning ? (
                  <img src="/api/camera/stream" alt="Live camera" style={{ width: '100%', height: '100%', objectFit: 'contain', borderRadius: 12 }} />
                ) : (
                  <><div className="camera-symbol">◎</div><h3>Camera Preview</h3><p>Start the camera to begin pose analysis</p></>
                )}
                {captureCountdown > 0 && <div className="countdown-overlay">{captureCountdown}</div>}
                {!cameraRunning ? (
                  <button className="btn-primary" onClick={startCamera} disabled={cameraLoading || backendStatus !== 'healthy'}>{cameraLoading ? 'Starting...' : 'Start Camera →'}</button>
                ) : (
                  <div className="camera-controls">
                    <button className="btn-capture" onClick={startCapture} disabled={captureCountdown > 0 || !analysis.match_ready}>📸 Capture</button>
                    <button className="btn-danger" onClick={stopCamera} disabled={cameraLoading}>{cameraLoading ? 'Stopping...' : 'Stop Camera'}</button>
                  </div>
                )}
              </div>
              {cameraError && <p className="error-text">{cameraError}</p>}
              <div className="camera-footer">
                <span><span className={`status-dot ${cameraRunning ? '' : 'muted'}`} />{cameraRunning ? 'Connected' : 'Disconnected'}</span>
                <span>{analysis.pose_stable ? '✓ Stable' : 'Adjusting...'}</span>
              </div>
            </div>
            <div className="panel pose-panel">
              <div className="panel-heading">
                <div><h3>Target Pose</h3><p>Select your reference pose</p></div>
                <span className="small-count">{poses.length} POSES</span>
              </div>
              <div className="pose-list">
                {poses.map((pose, i) => (
                  <button key={pose} className={`pose-option ${selectedPose === pose ? 'selected' : ''}`} onClick={() => changePose(pose)}>
                    <span className="pose-number">{String(i+1).padStart(2,'0')}</span>
                    <span className="pose-name">{pose}</span>
                    <span className="pose-arrow">{selectedPose === pose ? '✓' : '→'}</span>
                  </button>
                ))}
              </div>
              <div className="selected-pose">
                <span>SELECTED TARGET</span><strong>{selectedPose}</strong>
                <small>{analysis.match_ready ? `Score: ${score}%` : 'Waiting for joints'}</small>
              </div>
            </div>
          </section>
          <section className="bottom-grid">
            <div className="panel joint-panel">
              <div className="panel-heading"><div><h3>Joint Angles</h3><p>Live measurements</p></div>
                <span className="small-count">{analysis.visible_joints}/08 JOINTS</span></div>
              <div className="joint-grid">
                {JOINTS.map(j => {
                  const a = analysis.joint_angles[j]; const s = analysis.joint_scores[j]
                  return (<div className={`joint-item ${s !== null && s !== undefined ? (s >= 80 ? 'good' : s >= 50 ? 'warn' : 'bad') : ''}`} key={j}>
                    <span>{j}</span><strong>{a == null ? '—°' : `${a.toFixed(1)}°`}</strong>
                    {s != null && <small className="joint-score">{s.toFixed(0)}%</small>}
                  </div>)
                })}
              </div>
            </div>
            <div className="panel feedback-panel">
              <div className="panel-heading"><div><h3>AI Feedback</h3><p>Alignment corrections</p></div><span className="feedback-icon">✦</span></div>
              <div className="feedback-list">
                {analysis.feedback.map((m, i) => <p key={i} className="feedback-msg">{m}</p>)}
                {analysis.pose_stable && analysis.match_ready && analysis.match_score !== null && analysis.match_score > 70 &&
                  <p className="feedback-success">✓ Pose is stable — ready for capture!</p>}
              </div>
            </div>
          </section>
          {lastCapture && <div className="capture-preview">
            <div className="panel">
              <h3>📸 Photo Captured!</h3>
              <img src={`/api/captures/${lastCapture.id}/image`} alt="Capture" className="capture-img" />
              <p>Score: {lastCapture.alignment_score?.toFixed(1) ?? '—'}% · {lastCapture.pose_name}</p>
              <div className="capture-actions">
                <button className="btn-secondary" onClick={() => setLastCapture(null)}>Keep</button>
                <button className="btn-danger" onClick={() => { if (lastCapture.id) deleteCapture(lastCapture.id); setLastCapture(null) }}>Delete</button>
                <button className="btn-secondary" onClick={() => { setLastCapture(null); startCapture() }}>Retake</button>
              </div>
            </div>
          </div>}
        </>}

        {/* ─── POSE LIBRARY ─── */}
        {page === 'poses' && <section className="page-stack">
          <div className="page-intro">
            <div><h2>Pose Library</h2><p>{poseCatalog.length} reference poses across {CATEGORIES.length - 1} categories</p></div>
          </div>
          <div className="category-tabs">
            {CATEGORIES.map(c => <button key={c} className={poseFilter === c ? 'active' : ''} onClick={() => setPoseFilter(c)}>{c}</button>)}
          </div>
          <div className="pose-library">
            {filteredPoses.map(p => (
              <div key={p.name} className={`pose-card ${selectedPose === p.name ? 'selected' : ''}`}>
                <div className="pose-card-header">
                  <span className="pose-cat-badge">{p.category}</span>
                  <span className="pose-diff">{'★'.repeat(p.difficulty)} {DIFFICULTY_LABEL[p.difficulty]}</span>
                </div>
                <strong>{p.name}</strong>
                <p>{p.description}</p>
                <small>{p.instructions}</small>
                <button className="btn-primary" onClick={() => { changePose(p.name); setPage('camera') }}>
                  {selectedPose === p.name ? 'Selected ✓' : 'Select & Practice →'}
                </button>
              </div>
            ))}
            {filteredPoses.length === 0 && <p className="empty-msg">No poses in this category.</p>}
          </div>
        </section>}

        {/* ─── HISTORY ─── */}
        {page === 'history' && <section className="page-stack">
          <div className="page-intro">
            <div><h2>Session History</h2><p>Your past pose practice sessions and captures</p></div>
            <span className="pill">{sessions.length} SESSIONS · {captures.length} PHOTOS</span>
          </div>
          {sessions.length === 0 && captures.length === 0 ? (
            <div className="panel page-panel"><div className="empty-state"><span>◷</span><strong>No sessions recorded yet</strong><p>Complete a camera session to see your history here.</p>
              <button className="btn-primary" onClick={() => setPage('camera')}>Open Smart Camera →</button></div></div>
          ) : (<>
            {sessions.length > 0 && <div className="panel page-panel">
              <h3>Sessions</h3>
              <div className="session-list">
                {sessions.map(s => (
                  <div className="session-row" key={s.id}>
                    <div><strong>{s.pose_name}</strong><small>{s.created_at ? new Date(s.created_at).toLocaleString() : ''}</small></div>
                    <div className="session-stats">
                      <span>{s.alignment_score?.toFixed(1) ?? '—'}%</span>
                      <span>{Math.floor(s.duration_seconds/60)}:{String(s.duration_seconds%60).padStart(2,'0')}</span>
                      <button className="btn-sm" onClick={() => {
                        fetch(`/api/sessions/${s.id}`, { method: 'DELETE', headers: authHeaders() })
                        setSessions(prev => prev.filter(x => x.id !== s.id))
                      }}>✕</button>
                    </div>
                  </div>
                ))}
              </div>
            </div>}
            {captures.length > 0 && <div className="panel page-panel">
              <h3>Captured Photos</h3>
              <div className="capture-grid">
                {captures.map(c => (
                  <div className="capture-card" key={c.id}>
                    <img src={`/api/captures/${c.id}/image`} alt={c.pose_name || 'Capture'} />
                    <div className="capture-info">
                      <strong>{c.pose_name || 'Photo'}</strong>
                      <span>{c.alignment_score?.toFixed(1) ?? '—'}%</span>
                      <button className="btn-sm" onClick={() => deleteCapture(c.id)}>Delete</button>
                    </div>
                  </div>
                ))}
              </div>
            </div>}
          </>)}
        </section>}

        {/* ─── ANALYTICS ─── */}
        {page === 'analytics' && <section className="page-stack">
          <div className="page-intro">
            <div><h2>Performance Analytics</h2><p>Track your pose alignment progress over time</p></div>
          </div>
          <div className="stats-grid">
            <div className="stat-card"><div className="stat-top">TOTAL SESSIONS</div><div className="stat-value">{analytics?.total_sessions ?? 0}</div><p>Practice sessions completed</p></div>
            <div className="stat-card"><div className="stat-top">PHOTOS CAPTURED</div><div className="stat-value">{analytics?.photos_captured ?? 0}</div><p>Photos saved</p></div>
            <div className="stat-card"><div className="stat-top">AVG SCORE</div><div className="stat-value">{analytics?.avg_score?.toFixed(1) ?? '—'}<small>%</small></div><p>Average alignment score</p></div>
            <div className="stat-card"><div className="stat-top">BEST SCORE</div><div className="stat-value">{analytics?.best_score?.toFixed(1) ?? '—'}<small>%</small></div><p>{analytics?.most_practiced ? `Best pose: ${analytics.most_practiced}` : 'Keep practicing'}</p></div>
          </div>
          {analytics && analytics.total_sessions > 0 ? (
            <div className="panel page-panel">
              <h3>Practice Summary</h3>
              <div className="analytics-chart">
                <div className="chart-bar" style={{height: `${Math.min(100, analytics.avg_score ?? 0)}%`}}><span>{analytics.avg_score?.toFixed(0)}%</span></div>
                <div className="chart-bar best" style={{height: `${Math.min(100, analytics.best_score ?? 0)}%`}}><span>{analytics.best_score?.toFixed(0)}%</span></div>
              </div>
              <div className="chart-labels"><span>Average</span><span>Best</span></div>
              <p className="waiting-note">Total practice time: {analytics.total_duration_minutes} minutes across {analytics.total_sessions} sessions</p>
            </div>
          ) : (
            <div className="panel page-panel"><div className="empty-state"><span>▥</span><strong>No analytics yet</strong><p>Complete sessions to see your performance data.</p></div></div>
          )}
        </section>}

        {/* ─── SETTINGS ─── */}
        {page === 'settings' && <section className="page-stack">
          <div className="page-intro"><div><h2>Settings</h2><p>Configure your Smart Camera experience</p></div></div>
          <div className="panel page-panel settings-panel">
            <h3>Voice Guidance</h3>
            <label className="setting-row">
              <span>Enable voice guidance</span>
              <input type="checkbox" checked={voiceEnabled} onChange={e => setVoiceEnabled(e.target.checked)} />
            </label>
            <label className="setting-row">
              <span>Language</span>
              <select value={voiceLang} onChange={e => setVoiceLang(e.target.value)}>
                <option value="en">English</option><option value="te">Telugu</option>
              </select>
            </label>
            <label className="setting-row">
              <span>Speech rate ({speechRate.toFixed(1)}x)</span>
              <input type="range" min="0.5" max="2" step="0.1" value={speechRate} onChange={e => setSpeechRate(parseFloat(e.target.value))} />
            </label>
            <h3>Appearance</h3>
            <label className="setting-row">
              <span>Theme</span>
              <select value={theme} onChange={e => setTheme(e.target.value as 'dark'|'light')}>
                <option value="dark">Dark</option><option value="light">Light</option>
              </select>
            </label>
            <h3>Account</h3>
            <p className="waiting-note">Logged in as: {userName}</p>
            <button className="btn-danger" onClick={logout}>Sign Out</button>
          </div>
        </section>}

        {/* ─── HELP ─── */}
        {page === 'help' && <section className="page-stack">
          <div className="page-intro"><div><h2>Help & About</h2><p>Smart Camera with Real-Time Pose Suggestion and Alignment</p></div></div>
          <div className="panel page-panel">
            <h3>How to Use</h3>
            <div className="help-steps">
              <div className="help-step"><span>1</span><p><strong>Select a Pose</strong> — Browse the Pose Library and choose a reference pose to practice.</p></div>
              <div className="help-step"><span>2</span><p><strong>Start Camera</strong> — Open the Smart Camera and allow webcam access. Stand where your full body is visible.</p></div>
              <div className="help-step"><span>3</span><p><strong>Follow Feedback</strong> — Watch the alignment score and follow the AI correction suggestions. Green joints are aligned; red need adjustment.</p></div>
              <div className="help-step"><span>4</span><p><strong>Capture Photo</strong> — When your pose is stable and the score is above 70%, click Capture to save your photo.</p></div>
              <div className="help-step"><span>5</span><p><strong>Enable Voice</strong> — Turn on voice guidance in Settings for spoken corrections in English or Telugu.</p></div>
            </div>
            <h3>About</h3>
            <p>This application uses MediaPipe Pose Landmarker to detect 33 body landmarks in real-time via your webcam. Joint angles are calculated using 3D vector mathematics, and alignment scores are computed by comparing your current pose against predefined reference angle ranges.</p>
            <p>Scores are heuristic similarities for photography guidance, not medical assessments.</p>
            <h3>Technology Stack</h3>
            <p>Python · FastAPI · OpenCV · MediaPipe · React · TypeScript · SQLite · Vite</p>
          </div>
        </section>}

        <footer className="footer">
          <span>SMART CAMERA • POSE SUGGESTION & ALIGNMENT</span>
          <span>Computer Vision Powered</span>
        </footer>
      </main>
    </div>
  )
}

export default App
