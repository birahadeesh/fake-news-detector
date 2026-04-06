import os
import re
import string
import time

import requests

import joblib
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Hybrid AI Fake News Intelligence System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "app_loaded" not in st.session_state:
    st.session_state.app_loaded = False

if not st.session_state.app_loaded:
    _loader = st.empty()

    def _show_loader(pct: int, label: str) -> None:
        _loader.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');
        html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
        .stApp {{ background: #030712 !important; }}
        #MainMenu, footer, header {{ visibility: hidden; }}

        @keyframes iconPulse {{
            0%,100% {{ filter: drop-shadow(0 0 18px rgba(99,102,241,.5)); transform: scale(1); }}
            50%      {{ filter: drop-shadow(0 0 36px rgba(139,92,246,.9)); transform: scale(1.08); }}
        }}
        @keyframes fadeLoader {{
            from {{ opacity:0; transform:translateY(14px); }}
            to   {{ opacity:1; transform:translateY(0); }}
        }}
        @keyframes fillBar {{
            from {{ width: 0%; }}
            to   {{ width: {pct}%; }}
        }}
        @keyframes bgAnim {{
            0%,100% {{ opacity:.6; }} 50% {{ opacity:1; }}
        }}

        .loader-wrap {{
            position:fixed; inset:0; z-index:9999;
            background: #030712;
            background-image: radial-gradient(ellipse 80% 50% at 50% 0%, rgba(99,102,241,.22) 0%, transparent 65%);
            display:flex; flex-direction:column;
            align-items:center; justify-content:center;
            animation: fadeLoader .4s ease both;
        }}
        .loader-icon {{
            font-size: 4.5rem;
            animation: iconPulse 2s ease-in-out infinite;
            margin-bottom: 1.6rem;
        }}
        .loader-title {{
            font-size: 1.7rem; font-weight: 900;
            background: linear-gradient(135deg, #e0f2fe 0%, #a5b4fc 50%, #c084fc 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text; letter-spacing: -0.8px;
            margin-bottom: .5rem;
        }}
        .loader-sub {{
            font-size: .88rem; color: #475569;
            margin-bottom: 2.2rem; letter-spacing: .2px;
        }}
        .loader-bar-wrap {{
            width: 320px; height: 4px;
            background: rgba(255,255,255,.06);
            border-radius: 100px; overflow: hidden;
            margin-bottom: 1.1rem;
        }}
        .loader-bar {{
            height: 4px; border-radius: 100px;
            background: linear-gradient(90deg, #6366f1, #8b5cf6, #c084fc);
            box-shadow: 0 0 12px rgba(99,102,241,.6);
            width: {pct}%;
            transition: width .4s ease;
        }}
        .loader-pct {{
            font-size: .72rem; font-weight: 700;
            color: #334155; letter-spacing: 1px;
            font-family: 'JetBrains Mono', monospace;
        }}
        </style>
        <div class="loader-wrap">
            <div class="loader-icon">🧠</div>
            <div class="loader-title">Initializing Hybrid AI System</div>
            <div class="loader-sub">{label}</div>
            <div class="loader-bar-wrap">
                <div class="loader-bar"></div>
            </div>
            <div class="loader-pct">{pct}%</div>
        </div>
        """, unsafe_allow_html=True)

    _show_loader(0,  "Loading ML model and AI services…")
    time.sleep(0.4)
    _show_loader(35, "Initializing TF-IDF vectorizer…")
    time.sleep(0.4)
    _show_loader(65, "Connecting to Gemini API…")
    time.sleep(0.5)
    _show_loader(90, "Warming up inference engine…")
    time.sleep(0.3)
    _show_loader(100, "Ready.")
    time.sleep(0.2)

    _loader.empty()
    st.session_state.app_loaded = True

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #e2e8f0; }

/* ══ ANIMATED BACKGROUND ═════════════════════════════════════════ */
.stApp {
    background-color: #030712;
    background-image:
        linear-gradient(rgba(129,140,248,0.033) 1px, transparent 1px),
        linear-gradient(90deg, rgba(129,140,248,0.033) 1px, transparent 1px);
    background-size: 52px 52px, 52px 52px;
    min-height: 100vh;
    position: relative;
    overflow-x: hidden;
}
.stApp::before {
    content: '';
    position: fixed;
    top: -40%;
    left: 50%;
    transform: translateX(-50%);
    width: 120%;
    height: 70%;
    background: radial-gradient(ellipse at center, rgba(99,102,241,0.18) 0%, rgba(139,92,246,0.08) 40%, transparent 70%);
    animation: bgPulse 18s ease-in-out infinite;
    pointer-events: none;
    z-index: 0;
}
@keyframes bgPulse {
    0%, 100% { opacity: 0.7; transform: translateX(-50%) scaleX(1); }
    50%       { opacity: 1;   transform: translateX(-50%) scaleX(1.12); }
}

#MainMenu, footer, header { visibility: hidden; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0a0f1e; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }

/* ══ FADE-IN ═════════════════════════════════════════════════════ */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
.fade-in-up  { animation: fadeInUp .55s cubic-bezier(.22,.68,0,1.2) both; }
.fade-in     { animation: fadeIn .4s ease both; }
.fade-delay-1 { animation-delay: .08s; }
.fade-delay-2 { animation-delay: .18s; }
.fade-delay-3 { animation-delay: .28s; }

/* ══ PULSE GLOWS ════════════════════════════════════════════════ */
@keyframes pulseReal {
    0%, 100% { box-shadow: 0 0 40px rgba(34,197,94,0.12), inset 0 0 40px rgba(34,197,94,0.04); }
    50%       { box-shadow: 0 0 64px rgba(34,197,94,0.24), inset 0 0 48px rgba(34,197,94,0.08); }
}
@keyframes pulseFake {
    0%, 100% { box-shadow: 0 0 40px rgba(239,68,68,0.12), inset 0 0 40px rgba(239,68,68,0.04); }
    50%       { box-shadow: 0 0 64px rgba(239,68,68,0.24), inset 0 0 48px rgba(239,68,68,0.08); }
}
@keyframes badgePulseReal {
    0%, 100% { box-shadow: 0 0 0 0 rgba(34,197,94,0); }
    50%       { box-shadow: 0 0 12px 3px rgba(34,197,94,0.35); }
}
@keyframes badgePulseFake {
    0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0); }
    50%       { box-shadow: 0 0 12px 3px rgba(239,68,68,0.35); }
}

/* ══ BAR ANIMATION ═══════════════════════════════════════════════ */
@keyframes barGrow { from { width: 0; } to { width: var(--target-w, 100%); } }
.bar-inner { animation: barGrow .85s cubic-bezier(.22,.68,0,1) forwards; }

/* ══ PULSE DOT ══════════════════════════════════════════════════ */
.pulse-dot {
    display: inline-block;
    width: 10px; height: 10px;
    background: #4ade80;
    border-radius: 50%;
    margin-right: 8px;
    position: relative; top: -1px;
    animation: livePulse 2s ease-in-out infinite;
}
@keyframes livePulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(74,222,128,0.6); }
    50%       { box-shadow: 0 0 0 8px rgba(74,222,128,0); }
}

/* ══ HERO ════════════════════════════════════════════════════════ */
.hero-wrap { text-align: center; padding: 2.8rem 2rem 0.5rem; }
.hero-eyebrow {
    display: inline-flex; align-items: center;
    background: rgba(99,102,241,0.1); border: 1px solid rgba(99,102,241,0.25);
    border-radius: 100px; padding: .28rem 1rem;
    font-size: .7rem; font-weight: 700; color: #a5b4fc;
    letter-spacing: 1px; text-transform: uppercase; margin-bottom: 1rem;
}
.hero-title {
    font-size: 3.4rem; font-weight: 900; line-height: 1.12; letter-spacing: -1.5px;
    background: linear-gradient(135deg, #f0f9ff 0%, #bfdbfe 30%, #a5b4fc 65%, #c084fc 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: .9rem;
}
.hero-sub {
    font-size: 1.12rem; color: #cbd5e1; line-height: 1.72;
    max-width: 520px; margin: 0 auto 1.8rem;
}

/* ── Divider ─────────────────────────────────────────────────── */
.glow-div {
    height: 1px;
    background: linear-gradient(90deg, transparent 0%, rgba(99,102,241,0.5) 30%, rgba(192,132,252,0.5) 70%, transparent 100%);
    margin: 1.2rem 0; border: none;
}

/* ══ STATUS BAR ══════════════════════════════════════════════════ */
.status-bar { display:flex; justify-content:center; gap:1.4rem; padding:.5rem 0 1.3rem; flex-wrap:wrap; }
.status-pill {
    display:inline-flex; align-items:center; gap:.45rem;
    background:rgba(15,23,42,0.7); border:1px solid rgba(99,102,241,0.18);
    border-radius:100px; padding:.32rem 1rem;
    font-size:.76rem; font-weight:600; color:#cbd5e1;
    letter-spacing:.5px; backdrop-filter:blur(10px);
    transition: border-color .2s;
}
.dot-green  { width:7px;height:7px;border-radius:50%;background:#4ade80;box-shadow:0 0 6px rgba(74,222,128,.7);flex-shrink:0; }
.dot-yellow { width:7px;height:7px;border-radius:50%;background:#facc15;box-shadow:0 0 6px rgba(250,204,21,.7);flex-shrink:0; }
.dot-red    { width:7px;height:7px;border-radius:50%;background:#f87171;box-shadow:0 0 6px rgba(248,113,113,.7);flex-shrink:0; }

/* ══ FEATURE CARDS ═══════════════════════════════════════════════ */
@keyframes cardFloat {
    0%, 100% { transform: translateY(0px); }
    50%       { transform: translateY(-6px); }
}

.feat-grid  { display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin-bottom:1.8rem; }
.feat-card  {
    position:relative; background:rgba(15,23,42,.65);
    border:1px solid rgba(99,102,241,.15); border-radius:18px;
    padding:1.3rem 1.2rem; text-align:center;
    backdrop-filter:blur(18px);
    animation: cardFloat 7s ease-in-out infinite;
    transition: all 0.35s cubic-bezier(0.22, 1, 0.36, 1);
    overflow:hidden; cursor:default;
}
/* Staggered float start so cards drift independently */
.feat-card:nth-child(1) { animation-delay: 0s; }
.feat-card:nth-child(2) { animation-delay: 2.3s; }
.feat-card:nth-child(3) { animation-delay: 4.6s; }

.feat-card::after {
    content:''; position:absolute; inset:0; border-radius:18px;
    background:linear-gradient(135deg,rgba(99,102,241,.06) 0%,transparent 55%);
    pointer-events:none;
}
.feat-card:hover {
    transform:translateY(-5px) !important;
    border-color:rgba(129,140,248,.4);
    box-shadow: 0 24px 56px rgba(99,102,241,.12), 0 0 0 1px rgba(129,140,248,.08);
    animation-play-state: paused;
}

/* Icon micro-animation on card hover */
.feat-card .feat-icon {
    font-size:1.75rem; margin-bottom:.55rem; display:block;
    transition: transform 0.35s cubic-bezier(0.22, 1, 0.36, 1);
}
.feat-card:hover .feat-icon {
    transform: scale(1.03) rotate(2deg);
}

.feat-name { font-size:.82rem; font-weight:700; color:#a5b4fc; text-transform:uppercase; letter-spacing:1.1px; margin-bottom:.28rem; }
.feat-desc { font-size:.82rem; color:#94a3b8; line-height:1.55; }

/* ══ INPUT CARD ══════════════════════════════════════════════════ */
.input-card {
    background:rgba(15,23,42,.6); border:1px solid rgba(99,102,241,.16);
    border-radius:20px; padding:1.6rem 1.8rem 1.4rem;
    backdrop-filter:blur(24px); margin-bottom:.5rem;
}
div[data-testid="stTextArea"] textarea {
    background:rgba(3,7,18,.75) !important; border:1px solid rgba(99,102,241,.18) !important;
    border-radius:12px !important; color:#e2e8f0 !important;
    font-size:1rem !important; font-family:'Inter',sans-serif !important;
    line-height:1.7 !important; padding:.85rem 1rem !important;
    transition:border-color .2s, box-shadow .2s !important;
}
div[data-testid="stTextArea"] textarea:focus {
    border-color:rgba(99,102,241,.55) !important;
    box-shadow:0 0 0 3px rgba(99,102,241,.1) !important;
}
div[data-testid="stTextArea"] textarea::placeholder { color:#334155 !important; }
div[data-testid="stTextArea"] label {
    color:#94a3b8 !important; font-weight:600 !important;
    font-size:.78rem !important; text-transform:uppercase !important; letter-spacing:1px !important;
}
div[data-testid="stButton"] > button[kind="primary"] {
    background:linear-gradient(135deg,#4f46e5 0%,#7c3aed 100%) !important;
    color:#fff !important; border:none !important; border-radius:12px !important;
    font-weight:700 !important; font-size:.95rem !important;
    box-shadow:0 4px 22px rgba(79,70,229,.42),inset 0 1px 0 rgba(255,255,255,.12) !important;
    transition:transform .18s, box-shadow .18s !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    transform:translateY(-2px) scale(1.012) !important;
    box-shadow:0 8px 34px rgba(79,70,229,.58),inset 0 1px 0 rgba(255,255,255,.12) !important;
}

/* ══ RESULT CARD ═════════════════════════════════════════════════ */
.result-card {
    position:relative; border-radius:22px; padding:1.9rem 2.1rem 1.7rem;
    margin-bottom:1.1rem; backdrop-filter:blur(24px); overflow:hidden;
}
.result-card::before {
    content:''; position:absolute; inset:0; border-radius:22px;
    background:linear-gradient(135deg,rgba(255,255,255,.04) 0%,transparent 55%);
    pointer-events:none;
}
.card-fake { background:rgba(80,8,8,.15); border:1px solid rgba(239,68,68,.4);
             animation:pulseFake 3s ease-in-out infinite;
             transition:transform .3s ease, box-shadow .3s ease; }
.card-fake:hover { transform:translateY(-4px);
                   box-shadow:0 0 52px rgba(239,68,68,0.28), 0 16px 40px rgba(239,68,68,0.1); }
.card-real { background:rgba(8,45,22,.15); border:1px solid rgba(34,197,94,.4);
             animation:pulseReal 3s ease-in-out infinite;
             transition:transform .3s ease, box-shadow .3s ease; }
.card-real:hover { transform:translateY(-4px);
                   box-shadow:0 0 52px rgba(34,197,94,0.28), 0 16px 40px rgba(34,197,94,0.1); }

.verdict-badge {
    display:inline-flex; align-items:center; gap:.45rem;
    border-radius:100px; padding:.3rem .9rem;
    font-size:.68rem; font-weight:800; text-transform:uppercase;
    letter-spacing:1.3px; margin-bottom:.9rem;
}
.badge-fake { background:rgba(239,68,68,.14);color:#fca5a5;border:1px solid rgba(239,68,68,.3);
              animation:badgePulseFake 2s ease-in-out infinite; }
.badge-real { background:rgba(34,197,94,.14);color:#86efac;border:1px solid rgba(34,197,94,.3);
              animation:badgePulseReal 2s ease-in-out infinite; }

.verdict-text { font-size:2.5rem; font-weight:900; letter-spacing:-1.5px; line-height:1; margin-bottom:.35rem; }
.verdict-fake { color:#fca5a5; }
.verdict-real { color:#86efac; }
.verdict-conf { font-size:.9rem; color:#94a3b8; margin-top:.35rem; font-weight:500; }
.verdict-conf span { color:#f1f5f9; font-weight:800; font-size:1rem; font-family:'JetBrains Mono',monospace; }

/* Bar */
.bar-outer { background:rgba(255,255,255,.05); border-radius:100px; height:8px; margin-top:1rem; overflow:hidden; }
.bar-inner { height:8px; border-radius:100px; }
.bar-fake { background:linear-gradient(90deg,#dc2626,#f87171); box-shadow:0 0 14px rgba(239,68,68,.55); }
.bar-real { background:linear-gradient(90deg,#16a34a,#4ade80); box-shadow:0 0 14px rgba(34,197,94,.55); }

/* ══ KPI TILES ═══════════════════════════════════════════════════ */
.kpi-row { display:grid; grid-template-columns:1fr 1fr; gap:.85rem; margin-top:1rem; }
.kpi-tile {
    background:rgba(15,23,42,.7); border:1px solid rgba(99,102,241,.14);
    border-radius:16px; padding:1.15rem 1rem 1rem; text-align:center;
    backdrop-filter:blur(12px);
    transition:transform .2s, border-color .2s, box-shadow .2s;
    cursor:default;
}
.kpi-tile:hover { transform:translateY(-4px); border-color:rgba(99,102,241,.35); box-shadow:0 14px 32px rgba(99,102,241,.12); }
.kpi-num { font-size:2.3rem; font-weight:900; line-height:1; font-family:'JetBrains Mono',monospace; letter-spacing:-1px; }
.kpi-fake { color:#f87171; }
.kpi-real { color:#4ade80; }
.kpi-lbl  { font-size:.7rem; text-transform:uppercase; letter-spacing:.9px; color:#94a3b8; margin-top:.35rem; font-weight:600; }

/* ══ RIGHT PANELS ════════════════════════════════════════════════ */
.sec-lbl { font-size:.72rem; font-weight:800; color:#818cf8; text-transform:uppercase; letter-spacing:1.3px; margin:1.3rem 0 .6rem; }
.panel {
    position:relative; background:rgba(15,23,42,.65); border:1px solid rgba(99,102,241,.14);
    border-radius:18px; padding:1.4rem 1.5rem 1.3rem; margin-bottom:.9rem;
    backdrop-filter:blur(20px); overflow:hidden;
    transition: border-color .2s, box-shadow .2s;
}
.panel:hover { border-color:rgba(99,102,241,.32); box-shadow:0 12px 36px rgba(99,102,241,.1); }
.panel::before {
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background:linear-gradient(90deg,#6366f1,#8b5cf6,#c084fc); border-radius:18px 18px 0 0;
}
.panel-head { font-size:.75rem; font-weight:800; color:#a5b4fc; text-transform:uppercase; letter-spacing:1.1px; margin-bottom:.85rem; padding-top:.05rem; }
.panel-body { font-size:.9rem; color:#94a3b8; line-height:1.8; }
.panel-body strong { color:#e2e8f0; }
.panel-row { display:flex; justify-content:space-between; align-items:center; padding:.4rem 0; border-bottom:1px solid rgba(99,102,241,.08); font-size:.85rem; }
.panel-row:last-child { border-bottom:none; }
.prow-lbl { color:#64748b; font-size:.78rem; font-weight:500; }
.prow-val { font-weight:700; font-family:'JetBrains Mono',monospace; font-size:.84rem; }
.g-unavail { font-size:.84rem; color:#475569; font-style:italic; }

/* ══ FOOTER ══════════════════════════════════════════════════════ */
.site-footer { text-align:center; padding:1.2rem 0 .8rem; }
.site-footer p { font-size:.76rem; color:#334155; line-height:2; letter-spacing:.3px; }
.site-footer span { color:#475569; font-weight:600; }
/* ══ CUSTOM CURSOR ═══════════════════════════════════════════════ */
* { cursor: none !important; }
#cursor-dot {
    position: fixed; width:6px; height:6px;
    background: #a5b4fc;
    border-radius: 50%;
    pointer-events: none; z-index: 999999;
    transform: translate(-50%,-50%);
    box-shadow: 0 0 6px rgba(165,180,252,0.8);
}
#cursor-ring {
    position: fixed; width:36px; height:36px;
    border: 2px solid rgba(165,180,252,0.75);
    border-radius: 50%;
    pointer-events: none; z-index: 999998;
    transform: translate(-50%,-50%);
    background: rgba(99,102,241,0.04);
    box-shadow: 0 0 16px rgba(129,140,248,0.3), inset 0 0 8px rgba(129,140,248,0.06);
    transition: width 0.3s cubic-bezier(0.22,1,0.36,1),
                height 0.3s cubic-bezier(0.22,1,0.36,1),
                border-color 0.3s ease,
                box-shadow 0.3s ease,
                background 0.3s ease;
}
#cursor-ring.hovering {
    width: 52px; height: 52px;
    border-color: rgba(192,132,252,0.85);
    background: rgba(139,92,246,0.06);
    box-shadow: 0 0 24px rgba(192,132,252,0.35), 0 0 8px rgba(192,132,252,0.15), inset 0 0 12px rgba(139,92,246,0.08);
}
</style>
""", unsafe_allow_html=True)

# ── Custom cursor follower (JS via parent document access) ─────────
components.html("""
<script>
(function() {
    const doc = window.parent.document;

    // Create elements only once
    if (!doc.getElementById('cursor-dot')) {
        const dot = doc.createElement('div');
        dot.id = 'cursor-dot';
        doc.body.appendChild(dot);
    }
    if (!doc.getElementById('cursor-ring')) {
        const ring = doc.createElement('div');
        ring.id = 'cursor-ring';
        doc.body.appendChild(ring);
    }

    let mx = 0, my = 0, rx = 0, ry = 0;
    const lerp = 0.10; // lag factor — lower = more delay

    doc.addEventListener('mousemove', (e) => {
        mx = e.clientX;
        my = e.clientY;
        const dot = doc.getElementById('cursor-dot');
        if (dot) { dot.style.left = mx + 'px'; dot.style.top = my + 'px'; }
    });

    // Hover detection
    const SELECTORS = '.feat-card,.kpi-tile,.panel,.result-card,button,a,.input-card';
    doc.addEventListener('mouseover', (e) => {
        const ring = doc.getElementById('cursor-ring');
        if (ring && e.target.closest(SELECTORS)) ring.classList.add('hovering');
    });
    doc.addEventListener('mouseout', (e) => {
        const ring = doc.getElementById('cursor-ring');
        if (ring && e.target.closest(SELECTORS)) ring.classList.remove('hovering');
    });

    // Smooth animation loop
    function tick() {
        rx += (mx - rx) * lerp;
        ry += (my - ry) * lerp;
        const ring = doc.getElementById('cursor-ring');
        if (ring) { ring.style.left = rx + 'px'; ring.style.top = ry + 'px'; }
        window.parent.requestAnimationFrame(tick);
    }
    tick();
})();
</script>
""", height=0)

# ── Backend — logic unchanged ──────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    mdl = joblib.load("model/model.pkl")
    vec = joblib.load("model/vectorizer.pkl")
    return mdl, vec


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()


def get_gemini_client():
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")
    except Exception:
        api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return None
    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except Exception:
        return None


GEMINI_MODEL = "gemini-2.0-flash-001"


def gemini_summarize(client, article: str) -> str:
    prompt = (
        "Summarize the following news article in exactly 3-4 sentences. Be concise and factual.\n\n"
        f"Article:\n{article[:1200]}"
    )
    return client.models.generate_content(model=GEMINI_MODEL, contents=prompt).text.strip()


def gemini_credibility(client, article: str, label: str) -> str:
    tone = "unreliable or fake" if label == "FAKE" else "credible or real"
    prompt = (
        f"A machine-learning model classified this article as '{label}'. "
        f"In 3-5 sentences explain why it might be {tone}, based only on writing style, tone, and content cues.\n\n"
        f"Article:\n{article[:1200]}"
    )
    return client.models.generate_content(model=GEMINI_MODEL, contents=prompt).text.strip()


@st.cache_data(ttl=120, show_spinner=False)
def fetch_latest_news() -> list[dict]:
    """Fetch top 5 headlines from Indian news RSS feeds. No API key needed."""
    import xml.etree.ElementTree as ET

    RSS_FEEDS = [
        ("NDTV",          "https://feeds.feedburner.com/ndtvnews-india-news"),
        ("The Hindu",     "https://www.thehindu.com/news/national/feeder/default.rss"),
        ("Times of India","https://timesofindia.indiatimes.com/rssfeeds/296589292.cms"),
    ]
    HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; FakeNewsDetector/1.0)"}
    result = []

    for source_name, url in RSS_FEEDS:
        if len(result) >= 5:
            break
        try:
            resp = requests.get(url, timeout=8, headers=HEADERS)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
            items = root.findall(".//item")
            for item in items:
                if len(result) >= 5:
                    break
                title = (item.findtext("title") or "").strip()
                raw_desc = (item.findtext("description") or "").strip()
                # Strip any HTML tags that may appear in RSS descriptions
                desc = re.sub(r"<[^>]+>", "", raw_desc).strip()
                link = (item.findtext("link") or "").strip()
                if not title or "[Removed]" in title:
                    continue
                result.append({
                    "title":       title,
                    "description": desc[:220],
                    "text":        f"{title}. {desc}".strip(),
                    "url":         link,
                    "source":      source_name,
                })
        except Exception:
            continue   # try next feed

    return result


model, vectorizer = load_model()
gemini_client = get_gemini_client()

# ══ HERO ══════════════════════════════════════════════════════════
_, hero_col, _ = st.columns([1, 3, 1])
with hero_col:
    st.markdown("""
    <div class="hero-wrap fade-in">
        <div class="hero-eyebrow">
            <span class="pulse-dot"></span>NLP · Machine Learning · Generative AI
        </div>
        <div class="hero-title">Hybrid AI Fake News<br>Intelligence System</div>
        <div class="hero-sub">
            Paste any news article for an instant credibility assessment powered
            by a high-accuracy ML model and Gemini AI reasoning.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div class='glow-div'></div>", unsafe_allow_html=True)

# ══ STATUS BAR ════════════════════════════════════════════════════
gemini_dot   = "dot-green" if gemini_client else "dot-red"
gemini_label = "Active" if gemini_client else "Unavailable — set GEMINI_API_KEY"

st.markdown(f"""
<div class="status-bar fade-in">
    <span class="status-pill"><span class="dot-green"></span>ML Model: Active</span>
    <span class="status-pill"><span class="{gemini_dot}"></span>Gemini API: {gemini_label}</span>
</div>
""", unsafe_allow_html=True)

# ══ FEATURE CARDS ═════════════════════════════════════════════════
_, feat_col, _ = st.columns([0.5, 5, 0.5])
with feat_col:
    st.markdown("""
    <div class="feat-grid">
        <div class="feat-card fade-in-up fade-delay-1">
            <span class="feat-icon">⚡</span>
            <div class="feat-name">ML Engine</div>
            <div class="feat-desc">TF-IDF bigrams + Logistic Regression · 40K+ articles · 99% accuracy</div>
        </div>
        <div class="feat-card fade-in-up fade-delay-2">
            <span class="feat-icon">📊</span>
            <div class="feat-name">Confidence Scoring</div>
            <div class="feat-desc">Raw probability scores with low / high confidence interpretation</div>
        </div>
        <div class="feat-card fade-in-up fade-delay-3">
            <span class="feat-icon">🤖</span>
            <div class="feat-name">Gemini Reasoning</div>
            <div class="feat-desc">Article summary and credibility analysis via Gemini 2.0 Flash</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══ INPUT CARD ════════════════════════════════════════════════════
_, inp_col, _ = st.columns([0.5, 5, 0.5])
with inp_col:
    if not gemini_client:
        st.info("💡 **Gemini AI insights are disabled.** Add `GEMINI_API_KEY` to enable article summaries and credibility analysis.", icon="ℹ️")
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    article_text = st.text_area(
        label="📰 Article Text",
        placeholder="Paste the full body of a news article here…",
        height=210,
        label_visibility="visible",
    )
    analyze_btn = st.button("🔎  Run Analysis", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ══ ANALYSIS ══════════════════════════════════════════════════════
if analyze_btn:
    if not article_text.strip():
        _, w_col, _ = st.columns([0.5, 5, 0.5])
        with w_col:
            st.warning("⚠️ The text area is empty. Paste a news article and try again.", icon="⚠️")
    else:
        # ── Staged loading ────────────────────────────────────────
        status_box = st.empty()

        with status_box.status("🧠 Analyzing article\u2026", expanded=True) as job_status:

            # Stage 1 — ML inference
            step1 = st.empty()
            step1.write("🧠 Running ML inference\u2026")
            cleaned  = clean_text(article_text)
            vec_text = vectorizer.transform([cleaned])
            pred     = model.predict(vec_text)[0]
            proba    = model.predict_proba(vec_text)[0]
            time.sleep(0.5)
            step1.write("✅ ML inference completed.")

            # Stage 2 — Probabilities
            step2 = st.empty()
            step2.write("📊 Calculating probabilities\u2026")
            label    = "REAL" if pred == 1 else "FAKE"
            conf     = proba[pred] * 100
            fake_pct = proba[0] * 100
            real_pct = proba[1] * 100
            time.sleep(0.4)
            step2.write("✅ Probability calculation completed.")

            # Stage 3 — Gemini (optional)
            summary     = None
            credibility = None
            if gemini_client:
                step3 = st.empty()
                step3.write("🤖 Running AI reasoning\u2026")
                try:    summary     = gemini_summarize(gemini_client, article_text)
                except: summary     = None
                try:    credibility = gemini_credibility(gemini_client, article_text, label)
                except: credibility = None
                step3.write("✅ AI reasoning completed.")

            job_status.update(label="✅ Analysis completed successfully.", state="complete", expanded=False)


        st.markdown("<div class='glow-div'></div>", unsafe_allow_html=True)

        # ── Dashboard ─────────────────────────────────────────────
        _, dash_col, _ = st.columns([0.5, 5, 0.5])
        with dash_col:
            left, right = st.columns([13, 8], gap="large")

            # ════ LEFT column ═════════════════════════════════════
            with left:
                if label == "REAL":
                    card_cls, badge_cls, verdict_cls = "card-real", "badge-real", "verdict-real"
                    bar_cls, icon, bar_w = "bar-real", "✅", real_pct
                else:
                    card_cls, badge_cls, verdict_cls = "card-fake", "badge-fake", "verdict-fake"
                    bar_cls, icon, bar_w = "bar-fake", "🚨", fake_pct

                st.markdown(f"""
                <div class="result-card {card_cls} fade-in-up">
                    <div class="verdict-badge {badge_cls}">{icon}&nbsp;{label}</div>
                    <div class="verdict-text {verdict_cls}">{label}</div>
                    <div class="verdict-conf">Model confidence &nbsp;·&nbsp; <span>{conf:.1f}%</span></div>
                    <div class="bar-outer">
                        <div class="bar-inner {bar_cls}" style="--target-w:{bar_w:.1f}%;width:var(--target-w)"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if conf < 60:
                    st.warning(
                        f"⚠️ **Low confidence ({conf:.1f}%)** — Model is uncertain. Cross-check with a trusted source.",
                        icon="⚠️",
                    )
                else:
                    st.success(
                        f"✅ **Strong confidence ({conf:.1f}%)** — Model is confident this article is **{label}**.",
                        icon="✅",
                    )

                st.markdown("<div class='sec-lbl'>Probability Breakdown</div>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class="kpi-row fade-in-up fade-delay-1">
                    <div class="kpi-tile">
                        <div class="kpi-num kpi-fake">{fake_pct:.1f}<span style="font-size:1rem;opacity:.6">%</span></div>
                        <div class="kpi-lbl">🚨 Fake probability</div>
                    </div>
                    <div class="kpi-tile">
                        <div class="kpi-num kpi-real">{real_pct:.1f}<span style="font-size:1rem;opacity:.6">%</span></div>
                        <div class="kpi-lbl">✅ Real probability</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ════ RIGHT column ═════════════════════════════════════
            with right:
                dec_color = "#86efac" if label == "REAL" else "#fca5a5"
                st.markdown(f"""
                <div class="panel fade-in-up">
                    <div class="panel-head">🧠 Hybrid AI Decision</div>
                    <div class="panel-row">
                        <span class="prow-lbl">Verdict</span>
                        <span class="prow-val" style="color:{dec_color}">{icon} {label}</span>
                    </div>
                    <div class="panel-row">
                        <span class="prow-lbl">Confidence</span>
                        <span class="prow-val" style="color:#e2e8f0">{conf:.2f}%</span>
                    </div>
                    <div class="panel-row">
                        <span class="prow-lbl">Fake probability</span>
                        <span class="prow-val" style="color:#f87171">{fake_pct:.2f}%</span>
                    </div>
                    <div class="panel-row">
                        <span class="prow-lbl">Real probability</span>
                        <span class="prow-val" style="color:#4ade80">{real_pct:.2f}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<div class='glow-div' style='margin:.5rem 0'></div>", unsafe_allow_html=True)

                summary_html = summary if summary else "<span class='g-unavail'>&#9888; Gemini AI temporarily unavailable.</span>"
                cred_html    = credibility if credibility else "<span class='g-unavail'>&#9888; Gemini AI temporarily unavailable.</span>"
                # Guard against any non-BMP characters in Gemini response that crash Python 3.13
                def _safe(s: str) -> str:
                    return re.sub(r'[\ud800-\udfff]', '', s) if s else s
                summary_html = _safe(summary_html)
                cred_html    = _safe(cred_html)

                if gemini_client:
                    st.markdown(f"""
                    <div class="panel fade-in-up fade-delay-1">
                        <div class="panel-head">&#128221; Article Summary</div>
                        <div class="panel-body">{summary_html}</div>
                    </div>
                    <div class="panel fade-in-up fade-delay-2">
                        <div class="panel-head">&#129488; Credibility Analysis</div>
                        <div class="panel-body">{cred_html}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="panel fade-in-up">
                        <div class="panel-head">&#129302; Gemini AI Insights</div>
                        <div class="panel-body" style="color:#334155">
                            Set <code style="color:#818cf8">GEMINI_API_KEY</code> to enable AI summaries and credibility analysis.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# \u2550\u2550 LIVE NEWS ANALYSIS \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
_, live_col, _ = st.columns([0.5, 5, 0.5])
with live_col:
    st.markdown("<div class='glow-div'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center;padding:1.2rem 0 1rem'>
        <p style='font-size:1.45rem;font-weight:900;letter-spacing:-0.5px;
                  background:linear-gradient(135deg,#bfdbfe 0%,#a5b4fc 50%,#c084fc 100%);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                  background-clip:text;margin-bottom:.3rem;'>&#128240; Live News Analysis</p>
        <p style='font-size:.88rem;color:#475569;'>Fetch today's top headlines and run instant credibility checks.</p>
    </div>
    """, unsafe_allow_html=True)

    fetch_btn = st.button("🔄 Fetch Latest News", use_container_width=True)

    if "live_articles" not in st.session_state:
        st.session_state.live_articles = []
    if "live_result" not in st.session_state:
        st.session_state.live_result = {}

    if fetch_btn:
        with st.spinner("Pulling latest headlines from news feeds..."):
            articles = fetch_latest_news()
        if not articles:
            st.warning("Could not fetch headlines right now. Please try again in a moment.", icon="⚠️")
        else:
            st.session_state.live_articles = articles
            st.session_state.live_result   = {}

    if st.session_state.live_articles:
        st.markdown("<div class='sec-lbl' style='margin-bottom:.8rem'>Today's Headlines</div>", unsafe_allow_html=True)
        for idx, art in enumerate(st.session_state.live_articles):
            with st.container():
                title_display = art["title"][:90] + "..." if len(art["title"]) > 90 else art["title"]
                desc_display  = art["description"][:180] + "..." if len(art["description"]) > 180 else (art["description"] or "<em style='color:#334155'>No description available.</em>")
                source_prefix = art["source"] + " &nbsp;&middot;&nbsp; " if art["source"] else ""

                st.markdown(f"""
                <div class="panel" style="margin-bottom:.6rem;">
                    <div class="panel-head" style="text-transform:none;font-size:.82rem;letter-spacing:0">
                        {source_prefix}<span style="color:#64748b;font-weight:400;">{title_display}</span>
                    </div>
                    <div class="panel-body" style="margin-bottom:.75rem;font-size:.84rem;">
                        {desc_display}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                col_btn, col_res = st.columns([1, 3], gap="small")
                with col_btn:
                    if st.button("🔎 Analyze", key=f"live_analyze_{idx}", use_container_width=True):
                        cleaned  = clean_text(art["text"])
                        vec_text = vectorizer.transform([cleaned])
                        pred     = model.predict(vec_text)[0]
                        proba    = model.predict_proba(vec_text)[0]
                        lbl      = "REAL" if pred == 1 else "FAKE"
                        conf_pct = proba[pred] * 100
                        st.session_state.live_result[idx] = {"label": lbl, "conf": conf_pct}

                with col_res:
                    if idx in st.session_state.live_result:
                        r = st.session_state.live_result[idx]
                        lbl, conf_v = r["label"], r["conf"]
                        v_color = "#86efac" if lbl == "REAL" else "#fca5a5"
                        b_cls   = "badge-real" if lbl == "REAL" else "badge-fake"
                        v_icon  = "✅" if lbl == "REAL" else "🚨"
                        st.markdown(f"""
                        <div style="display:flex;align-items:center;gap:.8rem;padding:.5rem 0">
                            <span class="verdict-badge {b_cls}" style="animation:none">{v_icon}&nbsp;{lbl}</span>
                            <span style="font-family:'JetBrains Mono',monospace;font-size:.82rem;
                                         font-weight:700;color:{v_color}">{conf_v:.1f}% confidence</span>
                        </div>
                        """, unsafe_allow_html=True)

# ══ FOOTER ════════════════════════════════════════════════════════
st.markdown("<div class='glow-div'></div>", unsafe_allow_html=True)
st.markdown("""
<div class="site-footer">
    <p>
        <span>Hybrid AI Fake News Intelligence System</span><br>
        TF-IDF + Logistic Regression &nbsp;·&nbsp; Gemini 2.0 Flash API<br>
        Developed by <span>Birahadeeshwaran S.</span>
    </p>
</div>
""", unsafe_allow_html=True)