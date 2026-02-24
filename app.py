import os
import re
import string

import joblib
import streamlit as st

st.set_page_config(
    page_title="Hybrid AI Fake News Intelligence System",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #e2e8f0;
}

.stApp {
    background: linear-gradient(135deg, #020817 0%, #0a1628 40%, #0d1f3c 100%);
    min-height: 100vh;
}

#MainMenu, footer, header { visibility: hidden; }

.hero {
    text-align: center;
    padding: 2.8rem 1rem 1.2rem;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.5px;
    line-height: 1.2;
    text-shadow: none;
    margin-bottom: 0.6rem;
}
.hero-glow {
    display: inline-block;
    filter: drop-shadow(0 0 24px rgba(129,140,248,0.45));
}
.hero-subtitle {
    font-size: 0.97rem;
    color: #64748b;
    font-weight: 400;
    letter-spacing: 0.3px;
}

.neon-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #38bdf8 30%, #818cf8 70%, transparent);
    margin: 1.4rem 0 1.8rem;
    border: none;
    opacity: 0.4;
}

.feature-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.85rem;
    margin-bottom: 1.8rem;
}
.feature-card {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(129,140,248,0.18);
    border-radius: 14px;
    padding: 1.1rem 1rem;
    text-align: center;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    transition: transform 0.22s ease, border-color 0.22s ease, box-shadow 0.22s ease;
}
.feature-card:hover {
    transform: translateY(-4px);
    border-color: rgba(129,140,248,0.5);
    box-shadow: 0 8px 32px rgba(129,140,248,0.12);
}
.feature-icon {
    font-size: 1.6rem;
    margin-bottom: 0.35rem;
}
.feature-title {
    font-size: 0.82rem;
    font-weight: 700;
    color: #c7d2fe;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 0.2rem;
}
.feature-desc {
    font-size: 0.78rem;
    color: #64748b;
    line-height: 1.4;
}

div[data-testid="stTextArea"] textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(129,140,248,0.22) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-size: 0.93rem !important;
    resize: vertical;
    transition: border-color 0.2s ease;
}
div[data-testid="stTextArea"] textarea:focus {
    border-color: rgba(129,140,248,0.6) !important;
    box-shadow: 0 0 0 2px rgba(129,140,248,0.12) !important;
}
div[data-testid="stTextArea"] label {
    color: #94a3b8 !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
}

div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.65rem 1.5rem !important;
    transition: transform 0.18s ease, box-shadow 0.18s ease !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.35) !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    transform: scale(1.025) !important;
    box-shadow: 0 6px 28px rgba(99,102,241,0.5) !important;
}

.result-card {
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-top: 0.8rem;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
}
.result-card-fake {
    background: rgba(220,38,38,0.08);
    border: 1px solid rgba(239,68,68,0.45);
    box-shadow: 0 0 32px rgba(239,68,68,0.18), inset 0 0 32px rgba(239,68,68,0.04);
}
.result-card-real {
    background: rgba(22,163,74,0.08);
    border: 1px solid rgba(34,197,94,0.45);
    box-shadow: 0 0 32px rgba(34,197,94,0.18), inset 0 0 32px rgba(34,197,94,0.04);
}
.result-label {
    font-size: 1.9rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin: 0 0 0.25rem;
}
.result-label-fake { color: #f87171; }
.result-label-real { color: #4ade80; }
.conf-text {
    font-size: 0.88rem;
    color: #94a3b8;
    margin-top: 0.1rem;
}

.bar-wrap {
    background: rgba(255,255,255,0.06);
    border-radius: 100px;
    height: 10px;
    margin-top: 0.8rem;
    overflow: hidden;
}
.bar-fill {
    border-radius: 100px;
    height: 10px;
}
.bar-fake { background: linear-gradient(90deg, #f87171, #ef4444); box-shadow: 0 0 10px rgba(239,68,68,0.6); }
.bar-real { background: linear-gradient(90deg, #4ade80, #22c55e); box-shadow: 0 0 10px rgba(34,197,94,0.6); }

.section-title {
    font-weight: 600;
    font-size: 0.88rem;
    color: #818cf8;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 1.4rem 0 0.5rem;
}

.gemini-box {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(129,140,248,0.2);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    font-size: 0.91rem;
    color: #cbd5e1;
    line-height: 1.7;
    backdrop-filter: blur(8px);
}

div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(129,140,248,0.15);
    border-radius: 12px;
    padding: 0.8rem 1rem;
}
div[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.82rem !important; }
div[data-testid="stMetricValue"] { color: #e2e8f0 !important; font-weight: 700 !important; }

.footer {
    text-align: center;
    padding: 1.5rem 0 0.5rem;
    color: #334155;
    font-size: 0.78rem;
    letter-spacing: 0.4px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model():
    model = joblib.load("model/model.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
    return model, vectorizer


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.strip()
    return text


def get_gemini_client():
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
        "Summarize the following news article in exactly 3-4 sentences. "
        "Be concise and factual.\n\n"
        f"Article:\n{article[:1200]}"
    )
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return response.text.strip()

def gemini_credibility(client, article: str, label: str) -> str:
    prompt = (
        f"A machine-learning model classified the following news article as '{label}'. "
        "In 3-5 sentences, briefly explain why this article might be "
        f"{'unreliable or fake' if label == 'FAKE' else 'credible or real'}, "
        "based only on its writing style, tone, and content cues.\n\n"
        f"Article:\n{article[:1200]}"
    )
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return response.text.strip()


model, vectorizer = load_model()
gemini_client = get_gemini_client()


st.markdown("""
<div class="hero">
    <div class="hero-glow">
        <p class="hero-title">🧠 Hybrid AI Fake News<br>Intelligence System</p>
    </div>
    <p class="hero-subtitle">Powered by Machine Learning + Gemini AI Reasoning</p>
</div>
<div class="neon-divider"></div>
""", unsafe_allow_html=True)


st.markdown("""
<div class="feature-row">
    <div class="feature-card">
        <div class="feature-icon">⚡</div>
        <div class="feature-title">ML Engine</div>
        <div class="feature-desc">TF-IDF + Logistic Regression</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">📊</div>
        <div class="feature-title">Confidence Scoring</div>
        <div class="feature-desc">Probability-based trust metric</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">🤖</div>
        <div class="feature-title">Gemini Intelligence</div>
        <div class="feature-desc">Reasoning-based credibility analysis</div>
    </div>
</div>
""", unsafe_allow_html=True)

if not gemini_client:
    st.info(
        "💡 **Gemini AI insights are disabled.** "
        "Set the `GEMINI_API_KEY` environment variable to enable summaries and credibility analysis.",
        icon="ℹ️",
    )

article_text = st.text_area(
    label="📰 Paste your news article here",
    placeholder="Enter the full text of a news article …",
    height=240,
    label_visibility="visible",
)

analyze_btn = st.button("🔎 Analyze News", type="primary", use_container_width=True)

if analyze_btn:
    if not article_text.strip():
        st.warning("⚠️ Please paste a news article before clicking Analyze.", icon="⚠️")
    else:
        with st.spinner("Analyzing article …"):

            cleaned  = clean_text(article_text)
            vec_text = vectorizer.transform([cleaned])
            pred     = model.predict(vec_text)[0]
            proba    = model.predict_proba(vec_text)[0]

            label    = "REAL" if pred == 1 else "FAKE"
            conf     = proba[pred] * 100
            fake_pct = proba[0] * 100
            real_pct = proba[1] * 100

            if label == "REAL":
                card_cls   = "result-card-real"
                label_cls  = "result-label-real"
                bar_cls    = "bar-real"
                icon       = "✅"
                bar_width  = real_pct
            else:
                card_cls   = "result-card-fake"
                label_cls  = "result-label-fake"
                bar_cls    = "bar-fake"
                icon       = "🚨"
                bar_width  = fake_pct

            st.markdown(f"""
            <div class="result-card {card_cls}">
                <p class="result-label {label_cls}">{icon} {label}</p>
                <p class="conf-text">Model confidence: <strong style="color:#e2e8f0">{conf:.1f}%</strong></p>
                <div class="bar-wrap">
                    <div class="bar-fill {bar_cls}" style="width:{bar_width:.1f}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if conf < 60:
                st.warning(
                    f"⚠️ **Low confidence ({conf:.1f}%)** — The model is uncertain about this article. "
                    "Treat this result with caution and verify through trusted sources.",
                    icon="⚠️",
                )
            else:
                st.success(
                    f"✅ **Strong confidence ({conf:.1f}%)** — The model is confident this article is **{label}**.",
                    icon="✅",
                )

            st.markdown("<div class='section-title'>📊 Probability Breakdown</div>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            col1.metric("🚨 Fake", f"{fake_pct:.1f}%")
            col2.metric("✅ Real", f"{real_pct:.1f}%")

            if gemini_client:
                st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)
                st.markdown("<p style='font-size:1.05rem;font-weight:700;color:#818cf8;'>🤖 Gemini AI Insights</p>", unsafe_allow_html=True)

                try:
                    summary = gemini_summarize(gemini_client, article_text)
                    st.markdown("<div class='section-title'>📝 Article Summary</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='gemini-box'>{summary}</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"Could not generate summary: {e}")

                try:
                    credibility = gemini_credibility(gemini_client, article_text, label)
                    st.markdown("<div class='section-title'>🧐 Credibility Analysis</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='gemini-box'>{credibility}</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"Could not generate credibility analysis: {e}")


st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)
st.markdown("""
<div class="footer">
    © 2026 Developed by Birahadeeshwaran S.
</div>
""", unsafe_allow_html=True)