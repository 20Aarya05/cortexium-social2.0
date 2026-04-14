import { useState, useEffect, useCallback, useRef } from 'react'

const API = '/api'

/* ── Custom hook: fetch with auto-refresh ─────────────────────────────── */
function useAPI(path, interval = 5000) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)

  const fetchData = useCallback(async () => {
    try {
      const res = await fetch(`${API}${path}`)
      if (res.ok) setData(await res.json())
    } catch { /* server not running */ }
    finally { setLoading(false) }
  }, [path])

  useEffect(() => {
    fetchData()
    const id = setInterval(fetchData, interval)
    return () => clearInterval(id)
  }, [fetchData, interval])

  return { data, loading, refetch: fetchData }
}

/* ── Emotion emoji map ─────────────────────────────────────────────────── */
const EMO = {
  happy:'😊', sad:'😢', angry:'😠', surprised:'😲',
  disgust:'🤢', fear:'😨', neutral:'😐',
}

/* ── Views ─────────────────────────────────────────────────────────────── */

function Overview({ persons, interactions, liveEvents, liveTranscript }) {
  const totalPersons = persons?.length ?? 0
  const totalEvents  = interactions?.length ?? 0
  const emotions     = interactions?.map(i => i.emotion).filter(Boolean) ?? []
  const topEmotion   = emotions.length
    ? Object.entries(emotions.reduce((a,e)=>{a[e]=(a[e]||0)+1;return a},{}))
             .sort((a,b)=>b[1]-a[1])[0][0]
    : '—'
  const activeToday  = interactions?.filter(i => {
    if (!i.timestamp) return false
    return new Date(i.timestamp).toDateString() === new Date().toDateString()
  }).length ?? 0

  return (
    <div className="fade-in">
      <div className="stat-grid">
        {[
          { label: 'Known People',      value: totalPersons, sub: 'enrolled faces',      icon: '👤' },
          { label: 'Interactions',      value: totalEvents,  sub: 'total logged events', icon: '💬' },
          { label: 'Active Today',      value: activeToday,  sub: 'events today',        icon: '📅' },
          { label: 'Top Emotion',       value: EMO[topEmotion]||topEmotion, sub: topEmotion, icon: '🧠' },
        ].map(s => (
          <div className="card" key={s.label}>
            <div className="card-title">{s.icon} {s.label}</div>
            <div className="card-value">{s.value}</div>
            <div className="card-sub">{s.sub}</div>
          </div>
        ))}
      </div>

      <div className="section-header">
        <span className="section-title">🎙️ Live Voice Captions</span>
      </div>
      <div className="card" style={{borderColor:'var(--c-primary)', minHeight:60, display:'flex', alignItems:'center', justifyContent:'center', textAlign:'center', fontSize:18, fontWeight:500, fontStyle:'italic'}}>
        {liveTranscript ? `"${liveTranscript}"` : <span style={{color:'var(--c-muted)', fontSize:14}}>(Say something to see it here)</span>}
      </div>

      <div className="section-header">
        <span className="section-title">🔴 Live Event Stream</span>
        <span className="chip emotion">{liveEvents.length} events</span>
      </div>
      <div className="live-feed">
        {liveEvents.length === 0 && (
          <div style={{color:'var(--c-muted)', padding:'8px 0'}}>
            Waiting for events… (start main.py)
          </div>
        )}
        {liveEvents.slice(0, 30).map((ev, i) => (
          <div className="live-item fade-in" key={i}>
            <span className="ts">[{new Date(ev.ts).toLocaleTimeString()}]</span>
            <span>{ev.text}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function PeopleView({ persons, onRefetch }) {
  return (
    <div className="fade-in">
      <div className="section-header">
        <span className="section-title">👥 Known People</span>
        <button className="btn btn-ghost" onClick={onRefetch}>↻ Refresh</button>
      </div>
      {(!persons || persons.length === 0) && (
        <div className="empty-state">
          <div className="icon">👤</div>
          <p>No people enrolled yet.</p>
          <p>Start the agent and say <em>"This is [Name]"</em> when a face is detected.</p>
        </div>
      )}
      <div className="persons-grid">
        {(persons || []).map(p => (
          <div className="person-card fade-in" key={p.id}>
            <div style={{position:'absolute', top:8, right:8}}>
                <button 
                  className="btn btn-ghost" 
                  style={{fontSize:12, padding:4, minWidth:0, height:'auto', color:'var(--c-muted)'}}
                  onClick={async () => {
                    if (window.confirm(`Are you sure you want to delete all data for ${p.name}?`)) {
                        try {
                            const res = await fetch(`${API}/persons/${p.id}`, { method: 'DELETE' })
                            if (res.ok) {
                                alert(`Successfully deleted ${p.name}`);
                                onRefetch();
                            } else {
                                const err = await res.json().catch(() => ({}));
                                alert(`Error: ${res.status} - ${err.error || 'Server error'}`);
                            }
                        } catch(e) { 
                            alert('Network error: Is the backend running?');
                            console.error('Delete failed', e);
                        }
                    }
                  }}
                  title="Delete person"
                >
                  🗑️
                </button>
            </div>
            <div className="person-avatar">{p.name[0].toUpperCase()}</div>
            <div className="person-name">{p.name}</div>
            <div className="person-meta">
              {p.face_count} sightings<br />
              First: {p.first_seen ? new Date(p.first_seen).toLocaleDateString() : '—'}<br />
              Last:  {p.last_seen  ? new Date(p.last_seen).toLocaleDateString()  : '—'}
            </div>
            {p.context && <div style={{fontSize:11,color:'var(--c-accent)'}}>{p.context}</div>}
          </div>
        ))}
      </div>
    </div>
  )
}

function TimelineView({ interactions }) {
  return (
    <div className="fade-in">
      <div className="section-header">
        <span className="section-title">🕑 Interaction Timeline</span>
        <span className="chip emotion">{interactions?.length ?? 0} total</span>
      </div>
      <div className="timeline">
        {(!interactions || interactions.length === 0) && (
          <div className="empty-state">
            <div className="icon">💬</div>
            <p>No interactions logged yet.</p>
          </div>
        )}
        {(interactions || []).map(ev => (
          <div className="timeline-item fade-in" key={ev.id}>
            <div className="t-emotion">{EMO[ev.emotion] || '😐'}</div>
            <div className="t-body">
              <div className="t-persons">
                {ev.participants?.join(', ') || ev.persons?.join(', ') || 'Unknown'}
              </div>
              {ev.speech && <div className="t-speech">"{ev.speech.slice(0, 120)}"</div>}
              <div className="t-context">{ev.social_context || 'conversation'}</div>
            </div>
            <div className="t-time">
              {ev.timestamp ? new Date(ev.timestamp).toLocaleTimeString() : '—'}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function KnowledgeView() {
  const { data: patterns, loading, refetch } = useAPI('/knowledge/patterns', 30000)
  const [exporting, setExporting] = useState(false)
  const [exportResult, setExportResult] = useState(null)

  const triggerExport = async () => {
    setExporting(true)
    try {
      const res = await fetch(`${API}/knowledge/export`)
      const d = await res.json()
      setExportResult(d)
    } catch(e) { setExportResult({ error: String(e) }) }
    setExporting(false)
  }

  return (
    <div className="fade-in">
      <div className="section-header">
        <span className="section-title">🤖 Robot Behavioral Knowledge</span>
        <button className="btn btn-primary" onClick={triggerExport} disabled={exporting}>
          {exporting ? '⏳ Exporting…' : '⬇ Export'}
        </button>
      </div>

      {exportResult && (
        <div className="card" style={{marginBottom:16,borderColor:'var(--c-primary)'}}>
          <div className="card-title">✅ Export complete</div>
          {Object.entries(exportResult.files||{}).map(([k,v]) => (
            <div key={k} style={{fontSize:12,fontFamily:'var(--font-mono)',color:'var(--c-accent)',marginTop:4}}>
              {k}: {v}
            </div>
          ))}
        </div>
      )}

      {loading && <div style={{color:'var(--c-muted)'}}>Loading patterns…</div>}

      <div style={{display:'flex',flexDirection:'column',gap:12}}>
        {(patterns||[]).map((p,i) => (
          <div className="card" key={i}>
            <div style={{display:'flex',justifyContent:'space-between',alignItems:'flex-start'}}>
              <div>
                <div style={{fontWeight:600,marginBottom:6}}>{p.trigger}</div>
                <div style={{fontSize:12,color:'var(--c-muted)',marginBottom:8}}>
                  Context: <span style={{color:'var(--c-accent)'}}>{p.context}</span> |
                  Outcome: <span style={{color:'var(--c-primary)'}}>{p.outcome}</span>
                </div>
              </div>
              <div style={{textAlign:'right'}}>
                <div className="chip emotion">{p.frequency}× observed</div>
                <div style={{fontSize:11,color:'var(--c-muted)',marginTop:4}}>
                  conf: {(p.confidence*100).toFixed(0)}%
                </div>
              </div>
            </div>
            {p.sample_speech && (
              <div style={{fontSize:12,fontStyle:'italic',color:'var(--c-muted)',borderTop:'1px solid var(--c-border)',paddingTop:8}}>
                "{p.sample_speech.slice(0,160)}"
              </div>
            )}
            {p.observed_from?.length > 0 && (
              <div style={{fontSize:11,color:'var(--c-muted)',marginTop:6}}>
                Observed from: {p.observed_from.join(', ')}
              </div>
            )}
          </div>
        ))}
        {(!loading && (!patterns || patterns.length === 0)) && (
          <div className="empty-state">
            <div className="icon">🤖</div>
            <p>No patterns yet. Log some interactions first.</p>
          </div>
        )}
      </div>
    </div>
  )
}

function SettingsView() {
  const { data: camData, loading } = useAPI('/settings/cameras', 30000)
  const [selected, setSelected] = useState(null)
  const [saving, setSaving] = useState(false)

  const saveCamera = async () => {
    if (selected === null) return
    setSaving(true)
    try {
      const form = new FormData()
      form.append('index', selected)
      const res = await fetch(`${API}/settings/camera`, { method: 'POST', body: form })
      const d = await res.json()
      if (res.ok) {
        alert(d.message || 'Settings saved')
      } else {
        alert('Error: ' + d.error)
      }
    } catch(e) { alert('Save failed: ' + e) }
    setSaving(false)
  }

  return (
    <div className="fade-in">
      <div className="section-header">
        <span className="section-title">⚙️ Hardware Settings</span>
      </div>

      <div className="card">
        <div className="card-title">Camera Source</div>
        <p style={{fontSize:12,color:'var(--c-muted)',marginBottom:16}}>
          Select the local camera index to use. Index 0 is typically your default webcam.
        </p>
        
        {loading ? (
          <div style={{color:'var(--c-muted)'}}>Scanning cameras…</div>
        ) : (
          <div style={{display:'flex',gap:12,alignItems:'center'}}>
            <select 
              className="btn btn-ghost" 
              style={{padding:'8px 12px', flex:1}}
              onChange={(e) => setSelected(e.target.value)}
              value={selected ?? ''}
            >
              <option value="" disabled>— Select Camera —</option>
              {(camData?.cameras || [0,1,2,3]).map(idx => (
                <option key={idx} value={idx}>Camera Index {idx}</option>
              ))}
            </select>
            <button 
              className="btn btn-primary" 
              disabled={saving || selected === null}
              onClick={saveCamera}
            >
              {saving ? 'Saving…' : 'Save & Notify'}
            </button>
          </div>
        )}
      </div>

      <div style={{marginTop:24,padding:16,borderRadius:8,background:'rgba(255,100,100,0.1)',border:'1px solid rgba(255,100,100,0.2)'}}>
        <div style={{fontWeight:600,color:'var(--c-primary)',marginBottom:8}}>⚠️ Note for Multi-Camera Setup</div>
        <p style={{fontSize:12,color:'var(--c-text)',margin:0}}>
          Changing the camera source updates the hidden configuration. You must **restart the agent** (close and re-run <code>run_all.bat</code>) for the AI vision thread to pick up the new hardware.
        </p>
      </div>
    </div>
  )
}

/* ── Main App ─────────────────────────────────────────────────────────── */
export default function App() {
  const [view, setView] = useState('overview')
  const [wsLive, setWsLive] = useState(false)
  const [liveEvents, setLiveEvents] = useState([])
  const [liveTranscript, setLiveTranscript] = useState('')
  const wsRef = useRef(null)
  const transcriptTimeoutRef = useRef(null)

  const { data: persons, refetch: refetchPersons } = useAPI('/persons', 8000)
  const { data: interactions }                      = useAPI('/interactions?limit=100', 8000)

  // WebSocket live stream
  useEffect(() => {
    const connect = () => {
      // Since vite.config.js handles the /stream proxy to the backend,
      // we can use the same host/port the user is currently visiting.
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const wsUrl = `${protocol}//${window.location.host}/stream`
      
      const ws = new WebSocket(wsUrl)
      wsRef.current = ws
      ws.onopen  = () => setWsLive(true)
      ws.onclose = () => { setWsLive(false); setTimeout(connect, 3000) }
      ws.onerror = () => ws.close()
      ws.onmessage = (e) => {
        try {
          const d = JSON.parse(e.data)
          if (d.event === 'transcript') {
            setLiveTranscript(d.text)
            if (transcriptTimeoutRef.current) clearTimeout(transcriptTimeoutRef.current)
            transcriptTimeoutRef.current = setTimeout(() => setLiveTranscript(''), 3000)
            return
          }
          const text = d.event === 'enrolled'
            ? `✓ Enrolled: ${d.name}`
            : d.event === 'insight'
            ? `🧠 ${d.text}`
            : `💬 ${d.persons?.join(', ')||'—'} | ${d.emotion||''} | ${(d.speech||'').slice(0,60)}`
          setLiveEvents(ev => [{ ts: Date.now(), text }, ...ev].slice(0, 50))
        } catch {}
      }
    }
    connect()
    return () => wsRef.current?.close()
  }, [])

  const navItems = [
    { id: 'overview',   label: 'Overview',    icon: '⬡' },
    { id: 'people',     label: 'People',      icon: '👤' },
    { id: 'timeline',   label: 'Timeline',    icon: '📅' },
    { id: 'knowledge',  label: 'Knowledge',   icon: '🤖' },
    { id: 'settings',   label: 'Settings',    icon: '⚙️' },
  ]

  return (
    <div className="layout">
      {/* Topbar */}
      <header className="topbar">
        <div className="logo">
          <div className="logo-dot" />
          <span className="logo-text">CORTEXIUM</span>
        </div>
        <div style={{display:'flex',alignItems:'center',gap:16}}>
          <div className={`status-badge ${wsLive ? 'live' : ''}`}>
            <div className="dot" />
            {wsLive ? 'Live' : 'Offline'}
          </div>
          <div style={{fontSize:12,color:'var(--c-muted)'}}>
            {persons?.length ?? 0} people · {interactions?.length ?? 0} events
          </div>
        </div>
      </header>

      {/* Sidebar */}
      <aside className="sidebar">
        <div style={{fontSize:11,color:'var(--c-muted)',padding:'8px 12px',marginBottom:4}}>
          NAVIGATION
        </div>
        {navItems.map(n => (
          <button
            key={n.id}
            className={`nav-item ${view === n.id ? 'active' : ''}`}
            onClick={() => setView(n.id)}
          >
            <span>{n.icon}</span>
            <span>{n.label}</span>
          </button>
        ))}
        <div style={{marginTop:'auto',padding:'8px 12px',fontSize:11,color:'var(--c-muted)'}}>
          v1.0.0 · Social AI
        </div>
      </aside>

      {/* Main content */}
      <main className="content">
        {view === 'overview'  && <Overview persons={persons} interactions={interactions} liveEvents={liveEvents} liveTranscript={liveTranscript} />}
        {view === 'people'    && <PeopleView persons={persons} onRefetch={refetchPersons} />}
        {view === 'timeline'  && <TimelineView interactions={interactions} />}
        {view === 'knowledge' && <KnowledgeView />}
        {view === 'settings'  && <SettingsView />}
      </main>
    </div>
  )
}
