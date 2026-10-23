"""
⚖️ The Legal Eagle  -  Enterprise Legal-Tech Edition
AI-Powered Legal Document Analysis & Public Awareness Platform (RAG)
Stack: Streamlit + LangChain + ChromaDB + Google Gemini + Plotly

Run:  streamlit run app.py

API key: put GOOGLE_API_KEY=your_key in a .env file next to app.py (or set it as an
OS environment variable). It is read silently in the background; the UI only asks for
a key if none can be found.
"""
from dotenv import load_dotenv
load_dotenv()
import ast
import html
import inspect
import json
import os
import re
import tempfile
import time
import uuid
from datetime import date
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from fpdf import FPDF
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from google import genai  # used only to silently discover embedding models
except Exception:  # pragma: no cover
    genai = None

try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
    load_dotenv(Path(__file__).resolve().parent / ".env")
except Exception:  # python-dotenv is optional
    pass

# =====================================================================
# PAGE CONFIG
# =====================================================================
st.set_page_config(
    page_title="The Legal Eagle",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
# DESIGN SYSTEM (Deep Slate / Indigo-Violet / Emerald / Crimson Rose)
# =====================================================================
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root {
  --bg: #0F172A; --card: #1E293B; --card-2: #172033; --input: #0B1222; --border: #334155;
  --text: #E2E8F0; --muted: #94A3B8; --a1: #6366F1; --a2: #8B5CF6;
  --ok: #10B981; --warn: #F59E0B; --bad: #F43F5E;
}
.stApp { background: radial-gradient(1200px 500px at 85% -10%, rgba(99,102,241,.14), transparent 60%), var(--bg); color: var(--text); }
.stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp button, .stApp input, .stApp textarea { font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif; }
.stApp p, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
[data-testid="stMarkdownContainer"] { color: var(--text); }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--muted) !important; }
#MainMenu, footer { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 4.6rem !important; padding-bottom: 3rem; max-width: 1240px; }
hr { border-color: var(--border) !important; }

[data-testid="stSidebar"] { background: linear-gradient(180deg, #0B1222 0%, #111B30 100%); border-right: 1px solid var(--border); }
.side-brand { display: flex; align-items: center; gap: .7rem; margin: .2rem 0 1.1rem 0; }
.side-t { font-weight: 800; font-size: 1.05rem; color: #F8FAFC; line-height: 1.1; }
.side-s { font-size: .72rem; color: var(--muted); letter-spacing: .08em; text-transform: uppercase; }

.logo { width: 54px; height: 54px; border-radius: 14px; display: flex; align-items: center; justify-content: center;
  font-size: 1.7rem; background: linear-gradient(135deg, #6366F1, #8B5CF6); box-shadow: 0 8px 24px rgba(99,102,241,.45); flex-shrink: 0; }
.logo.sm { width: 38px; height: 38px; font-size: 1.2rem; border-radius: 11px; }

.hero { position: relative; overflow: hidden; display: flex; align-items: center; justify-content: space-between; gap: 1rem; flex-wrap: wrap;
  padding: 1.5rem 1.7rem; margin: 0 0 1.4rem 0; border-radius: 16px; border: 1px solid var(--border);
  background: linear-gradient(135deg, rgba(99,102,241,.20), rgba(139,92,246,.07) 55%, rgba(16,185,129,.05)), var(--card); }
.hero::after { content: ""; position: absolute; right: -60px; top: -80px; width: 260px; height: 260px; border-radius: 50%;
  background: radial-gradient(circle, rgba(139,92,246,.35), transparent 70%); pointer-events: none; }
.hero-left { display: flex; align-items: center; gap: 1.1rem; position: relative; z-index: 1; }
.hero-title { font-size: 2.1rem; font-weight: 800; line-height: 1.15; letter-spacing: -.02em; padding-bottom: .1rem;
  background: linear-gradient(90deg, #FFFFFF, #C4B5FD 50%, #818CF8); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
.hero-sub { color: var(--muted); font-size: .95rem; margin-top: .25rem; }
.hero-right { display: flex; gap: .45rem; flex-wrap: wrap; position: relative; z-index: 1; }

.stTabs [data-baseweb="tab-list"] { gap: 6px; background: var(--card); padding: 6px; border-radius: 14px; border: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] { height: auto; padding: .65rem 1.15rem; border-radius: 10px; background: transparent; color: var(--muted); font-weight: 600; }
.stTabs [data-baseweb="tab"] p { color: inherit !important; margin: 0; font-weight: 600; }
.stTabs [data-baseweb="tab"]:hover { color: #F8FAFC; background: rgba(99,102,241,.12); }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, #6366F1, #8B5CF6) !important; color: #FFFFFF !important; box-shadow: 0 6px 18px rgba(99,102,241,.4); }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.2rem; }

.sec { display: flex; align-items: center; gap: .75rem; margin: 1.7rem 0 .9rem 0; }
.sec .ico { width: 38px; height: 38px; border-radius: 11px; display: flex; align-items: center; justify-content: center; font-size: 1.1rem;
  background: linear-gradient(135deg, rgba(99,102,241,.28), rgba(139,92,246,.18)); border: 1px solid rgba(139,92,246,.4); }
.sec-t { font-size: 1.15rem; font-weight: 700; color: #F8FAFC; line-height: 1.2; }
.sec-s { font-size: .84rem; color: var(--muted); margin-top: .1rem; }
.sec.first { margin-top: .2rem; }

.card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 1.2rem; margin-bottom: .9rem;
  box-shadow: 0 6px 22px rgba(2,6,23,.35); line-height: 1.6; color: var(--text); }
.card.accent { border-left: 4px solid var(--a1); }
.card .who { font-size: .72rem; text-transform: uppercase; letter-spacing: .09em; font-weight: 700; color: #A5B4FC; margin-bottom: .4rem; }
.card-title { display: flex; justify-content: space-between; align-items: center; gap: .6rem; flex-wrap: wrap; margin-bottom: .55rem; }

.metric { position: relative; overflow: hidden; background: var(--card); border: 1px solid var(--border); border-radius: 12px;
  padding: 1rem 1.1rem 1rem 1.3rem; margin-bottom: .8rem; box-shadow: 0 6px 22px rgba(2,6,23,.35); }
.metric::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: linear-gradient(180deg, #6366F1, #8B5CF6); }
.metric.ok::before { background: var(--ok); } .metric.warn::before { background: var(--warn); } .metric.bad::before { background: var(--bad); }
.m-l { color: var(--muted); font-size: .7rem; text-transform: uppercase; letter-spacing: .09em; font-weight: 600; }
.m-v { font-size: 1.7rem; font-weight: 800; color: #F8FAFC; line-height: 1.25; margin-top: .15rem; letter-spacing: -.01em; }
.m-s { color: var(--muted); font-size: .76rem; margin-top: .1rem; min-height: 1rem; }

.pill { display: inline-flex; align-items: center; gap: .4rem; padding: .2rem .72rem; border-radius: 999px; font-size: .72rem; font-weight: 700;
  letter-spacing: .05em; text-transform: uppercase; border: 1px solid; white-space: nowrap; }
.pill .dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; box-shadow: 0 0 8px currentColor; }
.pill.high { color: #FDA4AF; background: rgba(244,63,94,.14); border-color: rgba(244,63,94,.5); box-shadow: 0 0 16px rgba(244,63,94,.28); }
.pill.medium { color: #FCD34D; background: rgba(245,158,11,.14); border-color: rgba(245,158,11,.5); box-shadow: 0 0 16px rgba(245,158,11,.24); }
.pill.low { color: #6EE7B7; background: rgba(16,185,129,.14); border-color: rgba(16,185,129,.5); box-shadow: 0 0 16px rgba(16,185,129,.24); }
.pill.info { color: #C4B5FD; background: rgba(99,102,241,.16); border-color: rgba(139,92,246,.5); }
.pill.neutral { color: #CBD5E1; background: rgba(148,163,184,.1); border-color: var(--border); }
.pill.plain { text-transform: none; letter-spacing: .01em; }

.callout { border-radius: 12px; padding: .9rem 1.1rem; margin: .5rem 0 .9rem 0; line-height: 1.55; font-size: .93rem;
  border: 1px solid rgba(99,102,241,.4); background: rgba(99,102,241,.10); color: #CBD5E1; }
.callout.ok { border-color: rgba(16,185,129,.45); background: rgba(16,185,129,.09); }
.callout.warn { border-color: rgba(245,158,11,.45); background: rgba(245,158,11,.09); }
.callout.bad { border-color: rgba(244,63,94,.45); background: rgba(244,63,94,.09); }
.callout b, .callout strong { color: #F8FAFC; }
.disclaimer { margin-top: 1.6rem; padding: .8rem 1rem; border-radius: 10px; border: 1px dashed var(--border); color: var(--muted); font-size: .8rem; line-height: 1.5; }

.flag { background: var(--card); border: 1px solid var(--border); border-left: 4px solid var(--border); border-radius: 12px; padding: 1.1rem 1.2rem;
  margin-bottom: .9rem; box-shadow: 0 6px 22px rgba(2,6,23,.35); }
.flag.high { border-left-color: var(--bad); } .flag.medium { border-left-color: var(--warn); } .flag.low { border-left-color: var(--ok); }
.flag-t { font-weight: 700; color: #F8FAFC; font-size: 1.02rem; }
.flag-pills { display: flex; gap: .4rem; flex-wrap: wrap; }
.quote { border-left: 3px solid #475569; background: rgba(11,18,34,.7); color: #CBD5E1; font-style: italic; padding: .55rem .9rem;
  border-radius: 0 8px 8px 0; margin: .4rem 0 .7rem 0; font-size: .92rem; }
.why { color: var(--text); margin-bottom: .6rem; font-size: .93rem; }
.why b { color: #F8FAFC; }
.tip { background: rgba(16,185,129,.09); border: 1px solid rgba(16,185,129,.35); border-radius: 10px; padding: .6rem .85rem; color: #D1FAE5; font-size: .9rem; }

.kt { width: 100%; border-collapse: separate; border-spacing: 0; overflow: hidden; border-radius: 12px; border: 1px solid var(--border); background: var(--card); }
.kt th { text-align: left; padding: .7rem 1rem; font-size: .72rem; letter-spacing: .09em; text-transform: uppercase; color: var(--muted); background: var(--card-2); }
.kt td { padding: .7rem 1rem; border-top: 1px solid var(--border); color: var(--text); font-size: .92rem; vertical-align: top; }
.kt td:first-child { font-weight: 700; color: #F8FAFC; width: 32%; }

.qa-q { background: linear-gradient(135deg, rgba(99,102,241,.22), rgba(139,92,246,.14)); border: 1px solid rgba(139,92,246,.4); border-radius: 12px;
  padding: .8rem 1.1rem; margin: 1rem 0 .6rem 0; color: #F8FAFC; }
.qa-q .who { display: block; font-size: .7rem; text-transform: uppercase; letter-spacing: .09em; color: #C4B5FD; font-weight: 700; margin-bottom: .15rem; }
.src { border: 1px solid var(--border); border-radius: 10px; background: var(--card-2); padding: .7rem .9rem; margin-bottom: .6rem; }
.src-t { color: #CBD5E1; font-size: .86rem; margin-top: .45rem; line-height: 1.5; }
.muted { color: var(--muted); font-size: .8rem; }

.md p { margin: .35rem 0 .55rem 0; } .md ul, .md ol { padding-left: 1.3rem; margin: .3rem 0 .7rem 0; } .md li { margin: .25rem 0; }
.md strong { color: #F8FAFC; } .md-h { font-weight: 700; color: #F8FAFC; margin: .9rem 0 .3rem 0; font-size: 1.02rem; }
.md blockquote { border-left: 3px solid var(--a1); margin: .5rem 0; padding: .3rem .9rem; color: #CBD5E1; background: rgba(11,18,34,.6); border-radius: 0 8px 8px 0; }
.md code { background: var(--input); border: 1px solid var(--border); border-radius: 6px; padding: .05rem .35rem; font-size: .85em; color: #C4B5FD; }
.md-table { width: 100%; border-collapse: collapse; margin: .5rem 0; font-size: .9rem; }
.md-table th, .md-table td { border: 1px solid var(--border); padding: .45rem .7rem; text-align: left; }
.md-table th { background: var(--card-2); color: #F8FAFC; }

[data-testid="stForm"] { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 1.2rem; box-shadow: 0 6px 22px rgba(2,6,23,.35); }
[data-testid="stWidgetLabel"] p, .stApp label p { color: #CBD5E1 !important; font-weight: 600; font-size: .86rem; }
.stTextInput input, .stTextArea textarea, .stNumberInput input { background: var(--input) !important; color: var(--text) !important; border-radius: 10px !important; }
[data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="base-input"] { background: var(--input) !important; border-color: var(--border) !important; border-radius: 10px !important; }
.stTextInput input::placeholder, .stTextArea textarea::placeholder { color: #64748B !important; }
[data-baseweb="input"]:focus-within, [data-baseweb="textarea"]:focus-within { border-color: var(--a1) !important; box-shadow: 0 0 0 3px rgba(99,102,241,.25) !important; }
.stNumberInput button { background: var(--input) !important; color: var(--text) !important; }
[data-baseweb="select"] > div { background: var(--input) !important; border-color: var(--border) !important; border-radius: 10px !important; color: var(--text) !important; }
[data-baseweb="select"] * { color: var(--text); }
[data-baseweb="popover"] ul, [data-baseweb="menu"] { background: var(--card) !important; border: 1px solid var(--border); }
[data-baseweb="popover"] li { color: var(--text) !important; background: transparent !important; }
[data-baseweb="popover"] li:hover, [data-baseweb="popover"] li[aria-selected="true"] { background: rgba(99,102,241,.22) !important; }

[data-testid="stFileUploaderDropzone"] { background: var(--input) !important; border: 1.5px dashed #475569 !important; border-radius: 12px !important; }
[data-testid="stFileUploaderDropzone"] * { color: var(--muted) !important; }
[data-testid="stFileUploaderDropzone"] button { color: var(--text) !important; }

.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {
  width: 100%; border-radius: 10px; border: 1px solid var(--border); background: var(--card); color: var(--text) !important;
  font-weight: 600; padding: .62rem 1rem; transition: all .2s ease; }
.stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
  border-color: var(--a2); transform: translateY(-1px); box-shadow: 0 8px 20px rgba(99,102,241,.28); background: #243247; }
button[kind="primary"], button[kind="primaryFormSubmit"], button[data-testid="stBaseButton-primary"], button[data-testid="stBaseButton-primaryFormSubmit"] {
  background: linear-gradient(135deg, #6366F1, #8B5CF6) !important; border: none !important; color: #FFFFFF !important;
  box-shadow: 0 8px 22px rgba(99,102,241,.4); }
button[kind="primary"]:hover, button[kind="primaryFormSubmit"]:hover, button[data-testid="stBaseButton-primary"]:hover, button[data-testid="stBaseButton-primaryFormSubmit"]:hover {
  filter: brightness(1.12); box-shadow: 0 10px 28px rgba(139,92,246,.55); background: linear-gradient(135deg, #6366F1, #8B5CF6) !important; }
.stButton > button p, .stDownloadButton > button p, [data-testid="stFormSubmitButton"] > button p { color: inherit !important; font-weight: 600; }

[data-testid="stExpander"] { background: var(--card); border: 1px solid var(--border); border-radius: 12px; }
[data-testid="stExpander"] details { border: none !important; background: transparent !important; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary p { color: var(--text) !important; font-weight: 600; }
[data-testid="stPlotlyChart"] { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: .4rem; box-shadow: 0 6px 22px rgba(2,6,23,.35); margin-bottom: .8rem; }
[data-testid="stSpinner"] * { color: #C4B5FD !important; }
"""


def h(s: str) -> str:
    """Flatten an HTML snippet to one line (prevents Markdown from treating indented lines as code)."""
    return " ".join(line.strip() for line in s.splitlines() if line.strip())


def esc(value) -> str:
    """HTML-escape any value (and neutralise `$` so Markdown math never triggers)."""
    return html.escape(str(value if value is not None else ""), quote=True).replace("$", "&#36;")


def esc_br(value) -> str:
    return esc(value).replace("\r\n", "\n").replace("\n", "<br>")


def render(markup: str):
    st.markdown(h(markup), unsafe_allow_html=True)


st.markdown(f"<style>{h(CSS)}</style>", unsafe_allow_html=True)

# =====================================================================
# LANGUAGES (selected language is injected into every LLM prompt)
# =====================================================================


def _native(name: str, script: str) -> str:
    return (
        f"Respond entirely in {name}, written in the {script} script, using simple everyday words that a "
        "common person without legal training understands. Keep statute names, section/article numbers and "
        "well-known legal terms (for example FIR, security deposit, notice) in English or in brackets next to "
        "the local term."
    )


LANGUAGES = {
    "English": "Respond in clear, simple English.",
    "Hinglish (Hindi in Roman script)": (
        "Respond in simple Hinglish (Hindi written in Roman/English script mixed naturally with English "
        "words), easy for a common person to understand. Keep legal terms in English."
    ),
    "Hindi (हिंदी)": _native("Hindi", "Devanagari"),
    "Marathi (मराठी)": _native("Marathi", "Devanagari"),
    "Tamil (தமிழ்)": _native("Tamil", "Tamil"),
    "Telugu (తెలుగు)": _native("Telugu", "Telugu"),
    "Bengali (বাংলা)": _native("Bengali", "Bengali"),
    "Gujarati (ગુજરાતી)": _native("Gujarati", "Gujarati"),
}
NATIVE_SCRIPT_LANGS = [k for k in LANGUAGES if k not in ("English", "Hinglish (Hindi in Roman script)")]

DISCLAIMER = (
    "⚠️ <b>Disclaimer:</b> The Legal Eagle provides general legal information, not professional "
    "legal advice. AI can make mistakes. Please consult a qualified lawyer before acting."
)

# =====================================================================
# SIDEBAR (kept intentionally minimal: language + ethics)
# =====================================================================
with st.sidebar:
    render(
        """
        <div class="side-brand"><div class="logo sm">⚖️</div>
        <div><div class="side-t">The Legal Eagle</div><div class="side-s">Legal-Tech Suite</div></div></div>
        """
    )
    lang = st.selectbox("🌐 Response language", list(LANGUAGES.keys()), index=0, key="ui_lang")
    LANG_INSTRUCTION = LANGUAGES[lang]
    st.caption("Summaries, answers, guidance and notice explanations follow this language.")
    st.divider()
    with st.expander("🛡️ Ethics & Privacy"):
        st.markdown(
            "- Output is **informational, not legal advice**.\n"
            "- AI may **hallucinate**; verify citations and section numbers.\n"
            "- Uploaded PDFs are processed in memory and sent to the Gemini API; "
            "avoid uploading highly sensitive personal data.\n"
            "- Risk scores are heuristic and should not replace a lawyer's review."
        )

# =====================================================================
# HERO HEADER
# =====================================================================
render(
    """
    <div class="hero">
      <div class="hero-left">
        <div class="logo">⚖️</div>
        <div>
          <div class="hero-title">The Legal Eagle</div>
          <div class="hero-sub">AI-powered legal document analysis &amp; public legal awareness platform for every Indian citizen</div>
        </div>
      </div>
      <div class="hero-right">
        <span class="pill info"><span class="dot"></span>RAG Engine</span>
        <span class="pill low"><span class="dot"></span>8 Languages</span>
        <span class="pill neutral">Gemini · ChromaDB</span>
      </div>
    </div>
    """
)


# =====================================================================
# SMALL UI BUILDERS
# =====================================================================
def callout(text_html: str, kind: str = "info"):
    cls = {"info": "callout", "warn": "callout warn", "ok": "callout ok", "bad": "callout bad"}[kind]
    render(f'<div class="{cls}">{text_html}</div>')


def section(icon: str, title: str, sub: str = "", first: bool = False):
    sub_html = f'<div class="sec-s">{esc(sub)}</div>' if sub else ""
    render(
        f'<div class="sec{" first" if first else ""}"><div class="ico">{icon}</div>'
        f'<div><div class="sec-t">{esc(title)}</div>{sub_html}</div></div>'
    )


def pill(text: str, tone: str = "neutral", dot: bool = True, plain: bool = False) -> str:
    dot_html = '<span class="dot"></span>' if dot else ""
    cls = f"pill {tone}{' plain' if plain else ''}"
    return f'<span class="{cls}">{dot_html}{esc(text)}</span>'


def metric_card(label: str, value, sub: str = "", tone: str = "") -> str:
    return (
        f'<div class="metric {tone}"><div class="m-l">{esc(label)}</div>'
        f'<div class="m-v">{esc(value)}</div><div class="m-s">{esc(sub)}</div></div>'
    )


def metric_row(items):
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col:
            render(metric_card(*item))


def disclaimer():
    render(f'<div class="disclaimer">{DISCLAIMER}</div>')


def conf_pill(score: float) -> str:
    pct = score * 100
    tone = "low" if pct >= 70 else ("medium" if pct >= 40 else "high")
    return pill(f"{pct:.1f}% match", tone)


# ---------------------------------------------------------------------
# Minimal, safe Markdown -> HTML (so LLM answers can live inside styled cards)
# ---------------------------------------------------------------------
def _inline(s: str) -> str:
    s = html.escape(s, quote=False).replace("$", "&#36;")
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"__(.+?)__", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\*\w])\*(?![\s\*])(.+?)(?<![\s\*])\*(?![\*\w])", r"<em>\1</em>", s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    return s


def md_to_html(text: str) -> str:
    if not text:
        return ""
    out, table, state = [], [], {"mode": None}

    def close_list():
        if state["mode"] in ("ul", "ol"):
            out.append(f"</{state['mode']}>")
        state["mode"] = None

    def flush_table():
        if not table:
            return
        head, body = table[0], table[1:]
        rows = "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in body)
        out.append(
            '<table class="md-table"><thead><tr>'
            + "".join(f"<th>{_inline(c)}</th>" for c in head)
            + f"</tr></thead><tbody>{rows}</tbody></table>"
        )
        table.clear()

    for raw in text.replace("\r\n", "\n").split("\n"):
        s = raw.strip()
        if s.startswith("|"):
            close_list()
            if re.fullmatch(r"[\s:\-|]+", s):
                continue
            table.append([c.strip() for c in s.strip("|").split("|")])
            continue
        flush_table()
        if not s:
            close_list()
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", s)
        if m:
            close_list()
            out.append(f'<div class="md-h">{_inline(m.group(2))}</div>')
            continue
        if re.fullmatch(r"[-*_]{3,}", s):
            close_list()
            out.append("<hr>")
            continue
        m = re.match(r"^[\*\-•]\s+(.*)$", s)
        if m:
            if state["mode"] != "ul":
                close_list()
                out.append("<ul>")
                state["mode"] = "ul"
            out.append(f"<li>{_inline(m.group(1))}</li>")
            continue
        m = re.match(r"^(\d+)[\.\)]\s+(.*)$", s)
        if m:
            if state["mode"] != "ol":
                close_list()
                out.append(f'<ol start="{int(m.group(1))}">')
                state["mode"] = "ol"
            out.append(f"<li>{_inline(m.group(2))}</li>")
            continue
        m = re.match(r"^>\s?(.*)$", s)
        if m:
            close_list()
            out.append(f"<blockquote>{_inline(m.group(1))}</blockquote>")
            continue
        close_list()
        out.append(f"<p>{_inline(s)}</p>")
    close_list()
    flush_table()
    return "".join(out)


def md_card(markdown_text: str, accent: bool = False, who: str = ""):
    who_html = f'<div class="who">{esc(who)}</div>' if who else ""
    render(f'<div class="card{" accent" if accent else ""}">{who_html}<div class="md">{md_to_html(markdown_text)}</div></div>')


# =====================================================================
# RESPONSE CLEANING
# =====================================================================
def extract_text(content) -> str:
    """Convert ANY Gemini/LangChain response shape into a clean string (drops thinking blocks)."""
    if content is None:
        return ""
    if not isinstance(content, (str, list, tuple, dict)) and hasattr(content, "content"):
        return extract_text(content.content)
    if isinstance(content, str):
        s = content.strip()
        if s.startswith("[{") and ("'text'" in s or '"text"' in s):
            for parser in (ast.literal_eval, json.loads):
                try:
                    return extract_text(parser(s))
                except Exception:
                    continue
        return content
    if isinstance(content, dict):
        if content.get("type") in ("thinking", "reasoning", "signature"):
            return ""
        txt = content.get("text")
        return txt if isinstance(txt, str) else ""
    if isinstance(content, (list, tuple)):
        return "".join(extract_text(item) for item in content)
    return str(content)


def parse_json(text: str):
    """Pull a JSON object out of a model reply (handles ```json fences and extra prose)."""
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(cleaned[start : end + 1])
    except Exception:
        return None


# =====================================================================
# API KEY (read silently from .env / environment / secrets)
# =====================================================================
def find_api_key() -> str:
    load_dotenv()
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
    if not key:
        try:
            key = st.secrets.get("GOOGLE_API_KEY", "") or ""
        except Exception:
            key = ""
    return str(key).strip()


api_key = find_api_key()
if not api_key:
    callout(
        "<b>Gemini API key not found.</b> Create a file named <code>.env</code> next to <code>app.py</code> "
        "containing <code>GOOGLE_API_KEY=your_key_here</code> and restart the app. "
        "Alternatively, paste a key below for this session only.",
        "warn",
    )
    typed_key = st.text_input("Gemini API Key", type="password")
    if not typed_key.strip():
        st.stop()
    api_key = typed_key.strip()

os.environ["GOOGLE_API_KEY"] = api_key

# =====================================================================
# FAILSAFE AI LAYER  (Only 100% stable & active models)
# =====================================================================
FAILSAFE_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
]
PREFERRED_EMBED = ["models/gemini-embedding-001", "models/text-embedding-004", "models/embedding-001"]

QUOTA_MARKERS = ("429", "resource_exhausted", "quota", "rate limit", "rate_limit", "too many requests", "exhausted")
BUSY_MARKERS = ("503", "502", "504", "unavailable", "overloaded", "deadline", "timeout", "timed out", "internal", "temporarily")
MISSING_MARKERS = ("404", "not_found", "not found", "no longer available", "is not supported", "deprecated", "unsupported")
KEY_MARKERS = ("api key not valid", "api_key_invalid", "permission_denied", "invalid api key", "api key expired")

BUSY_MESSAGE = (
    "⏳ <b>Our AI engines are very busy right now.</b> Your request is safe — please wait about a minute and "
    "press the button again."
)


class AIUnavailable(Exception):
    """Raised only after EVERY fallback model/embedding has failed."""

    def __init__(self, message: str = BUSY_MESSAGE, key_problem: bool = False):
        super().__init__(message)
        self.message = message
        self.key_problem = key_problem


def _err_text(e: Exception) -> str:
    return f"{type(e).__name__} {e}".lower()


def _has(msg: str, markers) -> bool:
    return any(m in msg for m in markers)


def _make_llm(model: str, temperature: float):
    try:
        return ChatGoogleGenerativeAI(model=model, temperature=temperature, max_retries=1, timeout=90)
    except Exception:
        return ChatGoogleGenerativeAI(model=model, temperature=temperature)


def ask_llm_failsafe(prompt: str, temperature: float = 0.3) -> str:
    """Try Gemini models in order and fall back when a model fails."""

    last_error = None

    for model in FAILSAFE_MODELS:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature,
                max_retries=1,
                timeout=90,
                google_api_key=api_key,
            )

            result = llm.invoke(prompt)
            response_text = extract_text(result).strip()

            if response_text:
                st.session_state["active_model"] = model
                return response_text

            last_error = RuntimeError(
                f"{model} returned an empty response."
            )

        except Exception as e:
            last_error = e
            error_text = _err_text(e)

            # Stop trying models if the API key is invalid.
            if _has(error_text, KEY_MARKERS):
                raise AIUnavailable(
                    "Gemini API key invalid or unauthorized. "
                    "Check GOOGLE_API_KEY in your .env file.",
                    key_problem=True,
                ) from e

            # Try the next model for unavailable models,
            # quota errors, and temporary service errors.
            continue

    if last_error:
        raise last_error

    raise AIUnavailable()

@st.cache_data(show_spinner=False, ttl=3600)
def discover_embed_models(key: str):
    """Silently ask the API which embedding models this key can use (empty list on any failure)."""
    if genai is None:
        return []
    try:
        client = genai.Client(api_key=key)
        names = []
        for m in client.models.list():
            actions = list(getattr(m, "supported_actions", None) or [])
            if "embedContent" in actions:
                names.append(m.name)
        return names
    except Exception:
        return []


def embed_candidates() -> list:
    found = discover_embed_models(api_key)
    if not found:
        return list(PREFERRED_EMBED)
    ordered = [m for m in PREFERRED_EMBED if m in found] + [m for m in found if m not in PREFERRED_EMBED]
    return ordered[:4]


def build_vectorstore(splits):
    """Embed + index with automatic embedding-model fallback. Raises AIUnavailable if all fail."""
    last_error, key_problem = None, False
    for name in embed_candidates():
        for attempt in range(2):
            try:
                embeddings = GoogleGenerativeAIEmbeddings(model=name)
                collection = f"doc_{uuid.uuid4().hex[:10]}"
                try:
                    vs = Chroma.from_documents(
                        splits, embeddings, collection_name=collection, collection_metadata={"hnsw:space": "cosine"}
                    )
                except TypeError:
                    vs = Chroma.from_documents(splits, embeddings, collection_name=collection)
                st.session_state["embed_model"] = name
                return vs
            except Exception as e:  # noqa: BLE001
                last_error = e
                msg = _err_text(e)
                if _has(msg, KEY_MARKERS):
                    key_problem = True
                    break
                if (_has(msg, QUOTA_MARKERS) or _has(msg, BUSY_MARKERS)) and attempt == 0:
                    time.sleep(2)
                    continue
                break
        if key_problem:
            break
    if key_problem:
        raise AIUnavailable(
            "🔑 <b>The Gemini API key looks invalid or expired.</b> Check <code>GOOGLE_API_KEY</code> in your "
            "<code>.env</code> file and restart the app.",
            key_problem=True,
        ) from last_error
    raise AIUnavailable() from last_error


def show_problem(e: Exception, context: str = "That request"):
    """Friendly, non-alarming error display (amber for busy engines, rose only for real faults)."""
    if isinstance(e, AIUnavailable):
        callout(e.message, "bad" if e.key_problem else "warn")
        return
    msg = _err_text(e)
    if _has(msg, QUOTA_MARKERS) or _has(msg, BUSY_MARKERS):
        callout(BUSY_MESSAGE, "warn")
    else:
        callout(f"<b>{esc(context)} could not be completed.</b> {esc(str(e)[:300])}", "bad")


# =====================================================================
# PROMPTS
# =====================================================================
MAX_FULLTEXT_CHARS = 150_000

ANALYSIS_PROMPT = """You are a senior Indian-law-aware legal analyst helping an ordinary person understand a contract.
{lang_instruction}
Analyze the document below and respond with ONLY a valid JSON object (no markdown fences, no commentary) with this exact schema.
Keep all JSON keys and the enum values (High/Medium/Low, category names) in English; write the free-text values in the requested language.

{{
  "document_type": "e.g. Rental Agreement / Employment Contract / NDA / Other",
  "executive_summary": "A clear 5-8 sentence summary: parties, purpose, duration, money involved, key obligations.",
  "risk_level": "High" | "Medium" | "Low",
  "risk_score": integer 0-100 (0 = very safe, 100 = extremely risky),
  "risk_reasons": ["2-4 short reasons for this risk level"],
  "key_terms": [{{"term": "e.g. Notice period", "value": "what the document says, or 'Not specified'"}}],
  "red_flags": [
    {{
      "clause": "short title of the clause",
      "category": "Penalty | Lock-in | Termination | Deposit | Non-compete | Indemnity | Arbitration/Jurisdiction | Auto-renewal | One-sided right | Other",
      "severity": "High" | "Medium" | "Low",
      "quote": "short exact excerpt (under 25 words) from the document, in the original language",
      "why_it_matters": "plain explanation of the risk",
      "suggestion": "what the person could negotiate or ask to change"
    }}
  ]
}}

Rules:
- Base everything ONLY on the document text. Do not invent clauses.
- Look hard for hidden penalties, strict lock-in periods, unilateral termination, forfeited deposits, heavy interest/late fees, non-compete or non-solicit restrictions, one-sided indemnity, auto-renewal and unfair jurisdiction/arbitration terms.
- If there are no red flags, return an empty list.
- key_terms should cover (if present): notice period, security deposit, lock-in period, rent/salary/consideration, penalty/late fee, termination conditions, renewal, governing law.

DOCUMENT:
{document}
"""

QA_PROMPT = """You are an expert legal advisor. Using ONLY the document excerpts below, answer the question.
{lang_instruction}
Format the answer as clean Markdown with **bold headings** and bullet points.
Mention clause numbers or page numbers when visible. List key clauses, red flags and potential penalties clearly.
If the answer is not in the excerpts, say so honestly instead of guessing.

DOCUMENT EXCERPTS:
{context}

QUESTION: {question}

ANSWER:"""

RIGHTS_PROMPT = """You are a legal-awareness assistant for Indian citizens.
{lang_instruction}

Topic area: {topic}
User's question: {question}

Reply in clean Markdown using EXACTLY these sections with bold headings:

**1. Short Answer** - 2-3 plain sentences.
**2. Your Rights & Relevant Laws** - bullet points naming the relevant Indian laws/sections (e.g. Consumer Protection Act, 2019; Constitution of India Articles; IT Act, 2000; Motor Vehicles Act, 1988) and what they say in simple terms. Only cite sections you are confident about; if unsure, say "check the exact section".
**3. Step-by-Step Action Plan** - a numbered list of practical steps the person should take, in order (collect evidence, written complaint, escalation, deadlines, where to file).
**4. Where to Complain** - bullet points with the correct authority/portal/helpline for this specific issue.
**5. Things to Keep in Mind** - 2-4 short bullets (limitation periods, costs, free legal aid, risks).

Be honest about uncertainty. Do not invent laws, case names, or phone numbers.
"""

NOTICE_PROMPT = """You are an experienced Indian advocate drafting a formal LEGAL NOTICE. Write it in formal English.

Details provided:
- Sender (Notice issuer) name: {sender}
- Sender address: {sender_addr}
- Recipient ({role}) name: {recipient}
- Recipient address: {recipient_addr}
- Notice category: {category}
- Date of transaction / agreement: {txn_date}
- Dispute reason / facts: {reason}
- Claim amount: {amount}
- Time allowed to comply: {days} days
- City / jurisdiction: {city}
- Date of notice: {today}

Instructions:
- Output clean Markdown: a bold title "LEGAL NOTICE", "Date", "To", "Subject" line, numbered paragraphs of facts, the legal grounds (cite Indian statutes only where you are confident, e.g. Consumer Protection Act, 2019 / Indian Contract Act, 1872 / Transfer of Property Act where relevant), the demand (exact claim amount), the compliance deadline of {days} days, a clear warning of legal action (civil / consumer complaint / criminal where applicable) at the recipient's risk and cost, and a signature block.
- Use [PLACEHOLDER IN CAPITALS] for any missing detail instead of inventing facts.
- Keep a firm but professional tone. Do not add commentary before or after the notice.
"""

NOTICE_EXPLAIN_PROMPT = """Below is a formal legal notice. {lang_instruction}
Explain in 4-6 short bullet points what this notice says, what the sender demands, the deadline, and what the sender should do next (e.g. send by Registered Post A.D., keep proof, consult an advocate). Do not rewrite the notice.

NOTICE:
{notice}
"""

TOPICS = [
    "General / Not sure",
    "Consumer Protection (refunds, defective goods, services)",
    "Fundamental Rights (Constitution of India)",
    "IT Act & Cyber Crime",
    "Traffic Rules & Motor Vehicles Act",
    "Tenancy & Rent",
    "Employment & Labour",
    "Police, FIR & Arrest Rights",
    "RTI (Right to Information)",
]

HELP_RESOURCES = """
| Issue | Where to go |
|---|---|
| Consumer complaints | **National Consumer Helpline: 1915** (also 1800-11-4000) · File online: **e-Daakhil** portal (edaakhil.nic.in) · consumerhelpline.gov.in |
| Cyber crime / online fraud | **1930** helpline · **cybercrime.gov.in** |
| Emergency / police | **112** |
| Free legal aid | **NALSA helpline 15100** · District Legal Services Authority |
| Government service grievances | **CPGRAMS** (pgportal.gov.in) |
| Women in distress | **181** · Child helpline **1098** |

*Helpline numbers and portal URLs can change. Always confirm on the official government website.*
"""

# =====================================================================
# SAMPLE DATA FOR THE 1-CLICK DEMO
# =====================================================================
SAMPLE_NAME = "Sample_Rental_Agreement.pdf (demo data)"
SAMPLE_PAGES = [
    """RESIDENTIAL RENTAL AGREEMENT

This Rental Agreement is made on 1 August 2026 at Pune, Maharashtra, between Mr. Ramesh Kulkarni (hereinafter "the Landlord") and Ms. Neha Sharma (hereinafter "the Tenant").

1. Premises. The Landlord lets out Flat No. 402, Sunrise Apartments, Pimpri, Pune 411018 to the Tenant for residential use only.

2. Term. The tenancy is for 11 months commencing 1 August 2026. This agreement shall be automatically renewed for a further 11 months unless the Tenant gives written notice 90 days before expiry, and rent shall increase by 15 percent on each renewal.

3. Rent. The monthly rent is Rs. 25,000, payable on or before the 5th day of each month. Late payment shall attract interest at 3 percent per month (36 percent per annum) and a fixed late fee of Rs. 1,000 per day of delay.

4. Security Deposit. The Tenant shall pay an interest-free security deposit of Rs. 1,50,000 (six months' rent). The Landlord may deduct any amount at his sole discretion for damages, repainting, cleaning or any other reason, and the balance shall be refunded within 90 days after the Tenant vacates.

5. Lock-in Period. There is a strict lock-in period of 10 months. If the Tenant vacates earlier for any reason, the entire security deposit shall stand forfeited and the Tenant shall pay rent for the remaining lock-in period.""",
    """6. Termination by Landlord. The Landlord may terminate this agreement at any time without assigning any reason by giving 7 days' notice. The Tenant may terminate only after the lock-in period by giving 90 days' written notice.

7. Maintenance and Repairs. The Tenant shall bear all repairs, including structural and plumbing repairs, regardless of cause. The Landlord shall not be liable for any defect in the premises.

8. Entry. The Landlord or his agents may enter the premises at any time without prior notice for inspection or showing the flat to prospective tenants.

9. Indemnity. The Tenant shall indemnify the Landlord against all losses, claims, damages and legal costs arising from the Tenant's use of the premises, including losses caused by third parties or events beyond the Tenant's control.

10. Sub-letting and Guests. The Tenant shall not sub-let the premises or allow any guest to stay for more than 3 consecutive days. Breach shall entitle the Landlord to immediate eviction and forfeiture of the deposit.

11. Dispute Resolution. All disputes shall be referred to a sole arbitrator appointed by the Landlord, and the courts at Mumbai alone shall have jurisdiction, even though the premises are in Pune.

12. Governing Law. This agreement is governed by the laws of India, including the Maharashtra Rent Control Act, 1999, where applicable.

Signed by both parties on the date mentioned above.
Landlord: Ramesh Kulkarni          Tenant: Neha Sharma""",
]

# =====================================================================
# HELPERS
# =====================================================================


def format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)


def normalize_severity(sev: str) -> str:
    s = (sev or "").upper()
    return "High" if "HIGH" in s else ("Medium" if "MED" in s else "Low")


LEGAL_TERMS = [
    "indemnify", "indemnity", "hereinafter", "jurisdiction", "arbitration", "arbitrator",
    "forfeit", "lock-in", "termination", "liability", "covenant", "notwithstanding",
    "pursuant", "whereas", "thereof", "herein", "breach", "tenant", "landlord", "lessor",
    "lessee", "confidential", "non-compete", "damages", "governing law", "sub-let",
]


def doc_stats(text: str) -> dict:
    """Document complexity metrics shown in the analytics dashboard."""
    words = re.findall(r"\w+", text)
    n_words = len(words)
    sentences = max(1, len(re.findall(r"[.!?]+(?:\s|$)", text)))
    clause_re = r"(?im)^\s*(?:\d+(?:\.\d+)*[\.\)]\s|\([a-z0-9]{1,3}\)\s|clause\s+\d+|article\s+\d+)"
    clauses = len(re.findall(clause_re, text))
    lower = text.lower()
    legal_hits = sum(lower.count(t) for t in LEGAL_TERMS)
    avg_sentence = n_words / sentences
    legal_density = legal_hits / max(1, n_words) * 100
    complexity = min(100, int(avg_sentence * 2.2 + legal_density * 6))
    return {
        "words": n_words,
        "chars": len(text),
        "tokens": int(len(text) / 4),
        "clauses": clauses,
        "clause_density": round(clauses / max(1, n_words) * 1000, 1),
        "avg_sentence": round(avg_sentence, 1),
        "legal_density": round(legal_density, 2),
        "complexity": complexity,
        "read_minutes": max(1, round(n_words / 200)),
    }


def retrieve_with_scores(vectorstore, query: str, k: int = 5):
    """Return [(Document, confidence 0..1)]; works across Chroma distance configurations."""
    try:
        pairs = vectorstore.similarity_search_with_relevance_scores(query, k=k)
        return [(d, float(max(0.0, min(1.0, s)))) for d, s in pairs]
    except Exception:
        pairs = vectorstore.similarity_search_with_score(query, k=k)
        return [(d, float(max(0.0, min(1.0, 1.0 - s / 2.0)))) for d, s in pairs]


PROBE_QUERIES = [
    "termination and notice period",
    "payment, deposit and penalties",
    "liability, indemnity and disputes",
]


def build_index(pages, source_name: str) -> bool:
    """Split -> embed -> Chroma, compute stats and retrieval probes. Returns True on success."""
    full_text = "\n\n".join(p.page_content for p in pages).strip()
    if len(full_text) < 50:
        callout(
            "<b>No readable text found.</b> This looks like a scanned/image PDF. "
            "Please upload a text-based PDF (or run OCR on it first).",
            "bad",
        )
        return False
    splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(pages)
    vs = build_vectorstore(splits)

    probes = []
    try:
        for q in PROBE_QUERIES:
            for rank, (d, s) in enumerate(retrieve_with_scores(vs, q, k=3), start=1):
                probes.append({"query": q, "rank": rank, "score": s, "page": d.metadata.get("page")})
    except Exception:
        probes = []  # probes are cosmetic; never block the document on them

    st.session_state.update(
        {
            "vectorstore": vs,
            "full_text": full_text,
            "num_chunks": len(splits),
            "num_pages": len(pages),
            "stats": doc_stats(full_text),
            "probes": probes,
            "doc_name": source_name,
            "analyses": {},
            "analysis_raw": {},
            "analysis_failed": {},
            "qa_history": [],
            "last_scores": None,
        }
    )
    return True


def run_analysis(language: str):
    """Run (once per language) the structured analysis; stores parsed JSON or the raw text."""
    doc_text = st.session_state["full_text"]
    truncated = len(doc_text) > MAX_FULLTEXT_CHARS
    raw = ask_llm_failsafe(
        ANALYSIS_PROMPT.format(lang_instruction=LANGUAGES[language], document=doc_text[:MAX_FULLTEXT_CHARS]),
        temperature=0.1,
    )
    parsed = parse_json(raw)
    if parsed:
        parsed["_truncated"] = truncated
        st.session_state["analyses"][language] = parsed
    else:
        st.session_state["analysis_raw"][language] = raw


def analysis_to_text(a: dict, doc_name: str, stats: dict) -> str:
    lines = [
        "THE LEGAL EAGLE - DOCUMENT ANALYSIS SUMMARY",
        f"Document: {doc_name}",
        f"Generated on: {date.today().strftime('%d %B %Y')}",
        "",
        f"Document type: {a.get('document_type', 'N/A')}",
        f"Overall risk: {a.get('risk_level', 'N/A')} ({a.get('risk_score', 'N/A')}/100)",
        "",
        "EXECUTIVE SUMMARY",
        str(a.get("executive_summary", "")),
        "",
        "WHY THIS RISK RATING",
    ]
    lines += [f"- {r}" for r in (a.get("risk_reasons") or [])]
    lines += ["", "KEY TERMS"]
    lines += [f"- {k.get('term', '')}: {k.get('value', '')}" for k in (a.get("key_terms") or [])]
    lines += ["", "RED FLAGS"]
    flags = a.get("red_flags") or []
    if not flags:
        lines.append("- None detected.")
    for i, f in enumerate(flags, start=1):
        lines += [
            f"{i}. [{normalize_severity(f.get('severity'))}] {f.get('clause', '')} ({f.get('category', 'Other')})",
            f"   Excerpt: {f.get('quote', '')}",
            f"   Why it matters: {f.get('why_it_matters', '')}",
            f"   Suggestion: {f.get('suggestion', '')}",
        ]
    lines += [
        "",
        "DOCUMENT METRICS",
        f"- Words: {stats['words']} | Est. tokens: {stats['tokens']} | Clauses: {stats['clauses']} "
        f"| Clause density: {stats['clause_density']}/1000 words | Complexity index: {stats['complexity']}/100",
        "",
        "DISCLAIMER: General legal information, not legal advice. AI can make mistakes; consult a lawyer.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------
# PDF export (Unicode-aware when a suitable system font exists)
# ---------------------------------------------------------------------
UNICODE_FONT_CANDIDATES = [
    os.getenv("LEGAL_EAGLE_PDF_FONT", ""),
    r"C:\Windows\Fonts\Nirmala.ttf",
    "C:/Windows/Fonts/Nirmala.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
    "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def find_unicode_font() -> str:
    for p in UNICODE_FONT_CANDIDATES:
        if p and Path(p).is_file():
            return p
    return ""


def _render_pdf(text: str, font_path: str = "") -> bytes:
    pdf = FPDF(format="A4")
    pdf.set_margins(20, 20, 20)
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    if font_path:
        pdf.add_font("UF", "", font_path)
        pdf.set_font("UF", size=11)
        try:
            pdf.set_text_shaping(True)  # needs `pip install uharfbuzz`; improves Indic conjuncts
        except Exception:
            pass
    else:
        pdf.set_font("Helvetica", size=11)
    for line in text.split("\n"):
        if not line.strip():
            pdf.ln(4)
        else:
            pdf.multi_cell(0, 6, line, new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())


def text_to_pdf_bytes(text: str):
    """Convert (markdown-ish) text into a clean PDF. Returns bytes, or None if generation fails."""
    replacements = {
        "₹": "Rs. ", "–": "-", "—": "-", "‘": "'", "’": "'", "“": '"', "”": '"',
        "…": "...", "•": "-", "\u00a0": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[\*\-]\s+", "- ", text, flags=re.MULTILINE)
    text = text.replace("`", "")
    latin = text.encode("latin-1", "replace").decode("latin-1")
    try:
        if latin == text:
            return _render_pdf(text)
        font = find_unicode_font()
        if font:
            try:
                return _render_pdf(text, font)
            except Exception:
                pass
        return _render_pdf(latin)
    except Exception:
        return None


def pdf_button(label: str, text: str, file_name: str, key: str):
    data = text_to_pdf_bytes(text)
    if data:
        st.download_button(label, data=data, file_name=file_name, mime="application/pdf", key=key)
    else:
        callout("PDF export is unavailable for this text — please use the .txt download.", "warn")


# ---------------------------------------------------------------------
# Plotly styling (transparent, dark-slate, Inter typography)
# ---------------------------------------------------------------------
FONT = dict(family="Inter, Segoe UI, system-ui, sans-serif", color="#E2E8F0")
EMERALD, AMBER, ROSE, INDIGO, VIOLET = "#10B981", "#F59E0B", "#F43F5E", "#6366F1", "#8B5CF6"

try:
    _PLOTLY_WIDTH = (
        {"width": "stretch"}
        if "width" in inspect.signature(st.plotly_chart).parameters
        else {"use_container_width": True}
    )
except Exception:
    _PLOTLY_WIDTH = {"use_container_width": True}


def show_chart(fig, key: str):
    try:
        st.plotly_chart(fig, key=key, theme=None, **_PLOTLY_WIDTH)
    except TypeError:
        st.plotly_chart(fig, key=key, use_container_width=True)


def style_fig(fig, height=320):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=24, r=24, t=56, b=24),
        font=FONT,
        title_font=dict(size=15, color="#F1F5F9", family=FONT["family"]),
        legend=dict(font=dict(color="#CBD5E1")),
        hoverlabel=dict(bgcolor="#0F172A", bordercolor="#334155", font=FONT),
    )
    fig.update_xaxes(gridcolor="#334155", zerolinecolor="#334155", linecolor="#334155", tickfont=dict(color="#94A3B8"))
    fig.update_yaxes(gridcolor="#334155", zerolinecolor="#334155", linecolor="#334155", tickfont=dict(color="#94A3B8"))
    return fig


def gauge(value: float, title: str, higher_is_risk: bool, height=300):
    if higher_is_risk:
        steps = [
            {"range": [0, 33], "color": "rgba(16,185,129,.40)"},
            {"range": [33, 66], "color": "rgba(245,158,11,.40)"},
            {"range": [66, 100], "color": "rgba(244,63,94,.45)"},
        ]
    else:
        steps = [
            {"range": [0, 40], "color": "rgba(244,63,94,.45)"},
            {"range": [40, 70], "color": "rgba(245,158,11,.40)"},
            {"range": [70, 100], "color": "rgba(16,185,129,.40)"},
        ]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title, "font": {"size": 14, "color": "#94A3B8", "family": FONT["family"]}},
            number={"suffix": "/100" if higher_is_risk else "%", "font": {"size": 42, "color": "#F8FAFC", "family": FONT["family"]}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#64748B", "tickfont": {"color": "#94A3B8"}},
                "bar": {"color": "#818CF8", "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": steps,
                "threshold": {"line": {"color": "#F8FAFC", "width": 3}, "thickness": 0.8, "value": value},
            },
        )
    )
    return style_fig(fig, height)


def safe_score(analysis: dict) -> int:
    try:
        return max(0, min(100, int(float(analysis.get("risk_score", 0)))))
    except Exception:
        return 0


def render_dashboard(analysis: dict):
    section("📈", "Analytics Dashboard", "Complexity metrics, risk distribution and retrieval quality")
    stats = st.session_state["stats"]

    metric_row(
        [
            ("Word count", f"{stats['words']:,}", f"≈ {stats['read_minutes']} min read", ""),
            ("Token estimate", f"{stats['tokens']:,}", "characters ÷ 4", ""),
            ("Clauses detected", stats["clauses"], f"{stats['clause_density']} per 1k words", ""),
            ("Avg sentence", f"{stats['avg_sentence']}", "words per sentence", ""),
            (
                "Complexity index",
                f"{stats['complexity']}/100",
                f"Legal-term density {stats['legal_density']}%",
                "bad" if stats["complexity"] >= 66 else ("warn" if stats["complexity"] >= 33 else "ok"),
            ),
        ]
    )

    flags = analysis.get("red_flags") or []
    counts = {"High": 0, "Medium": 0, "Low": 0}
    for f in flags:
        counts[normalize_severity(f.get("severity"))] += 1

    c1, c2 = st.columns(2)
    with c1:
        if sum(counts.values()) == 0:
            callout("No red-flag clauses detected, so there is no risk distribution to plot.", "ok")
        else:
            fig = go.Figure(
                go.Pie(
                    labels=list(counts.keys()),
                    values=list(counts.values()),
                    hole=0.58,
                    marker=dict(colors=[ROSE, AMBER, EMERALD], line=dict(color="#1E293B", width=3)),
                    textinfo="label+value",
                    textfont=dict(color="#F8FAFC", family=FONT["family"]),
                    sort=False,
                )
            )
            fig.update_layout(title="Risk Distribution of Flagged Clauses")
            show_chart(style_fig(fig), "chart_pie")
    with c2:
        cat_counts = {}
        for f in flags:
            cat = str(f.get("category", "Other"))
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
        if cat_counts:
            fig = go.Figure(
                go.Bar(
                    x=list(cat_counts.values()),
                    y=list(cat_counts.keys()),
                    orientation="h",
                    marker=dict(color=VIOLET, line=dict(color=INDIGO, width=1)),
                )
            )
            fig.update_layout(title="Red Flags by Category", xaxis_title="Count")
            show_chart(style_fig(fig), "chart_cat")
        else:
            callout("No categories to display.", "info")

    last = st.session_state.get("last_scores")
    if last:
        scores, label = last["scores"], f"Latest question: {last['query'][:40]}"
        labels = [f"#{i + 1} (p.{p})" for i, p in enumerate(last["pages"])]
        values = [round(s * 100, 1) for s in last["scores"]]
        title = f"Retrieved Chunk Similarity: “{last['query'][:60]}”"
    else:
        probes = st.session_state.get("probes") or []
        scores, label = [p["score"] for p in probes], "automatic retrieval probes"
        labels = [f"{p['query'].split(' ')[0]} #{p['rank']}" for p in probes]
        values = [round(p["score"] * 100, 1) for p in probes]
        title = "Chunk Similarity Confidence (automatic probes)"

    if not scores:
        callout("Ask a question below to see retrieval confidence for your document.", "info")
        return

    avg = round(sum(scores) / len(scores) * 100, 1)
    c3, c4 = st.columns(2)
    with c3:
        show_chart(gauge(avg, "Avg Chunk Similarity Confidence", False), "chart_conf_gauge")
        st.caption(f"Based on: {label}")
    with c4:
        fig = go.Figure(
            go.Bar(
                x=labels,
                y=values,
                marker_color=[EMERALD if v >= 70 else AMBER if v >= 40 else ROSE for v in values],
                text=[f"{v}%" for v in values],
                textposition="outside",
                textfont=dict(color="#E2E8F0"),
            )
        )
        fig.update_layout(title=title, yaxis=dict(range=[0, 115], title="Confidence %"))
        show_chart(style_fig(fig, 300), "chart_conf_bar")
    st.caption(
        "Confidence = Chroma relevance score between the question and each retrieved chunk "
        "(higher means the passage is semantically closer). Ask a question below to update these charts."
    )


# =====================================================================
# SESSION CALLBACKS
# =====================================================================
def start_demo():
    st.session_state["use_demo"] = True


def clear_demo():
    st.session_state["use_demo"] = False
    for k in ("file_id", "vectorstore", "analyses", "analysis_raw", "analysis_failed", "qa_history", "index_failed_id"):
        st.session_state.pop(k, None)


def retry_index():
    st.session_state.pop("index_failed_id", None)


def retry_analysis(language: str):
    st.session_state.get("analysis_failed", {}).pop(language, None)


def retry_explain():
    st.session_state.pop("notice_explain_lang", None)


# =====================================================================
# TABS
# =====================================================================
tab1, tab2, tab3 = st.tabs(
    ["📄 Personal Legal Document Analyzer", "🌐 Public Legal Rights & Awareness", "✍️ Legal Notice Generator"]
)

# =====================================================================
# TAB 1: PERSONAL LEGAL DOCUMENT ANALYZER (RAG)
# =====================================================================
with tab1:
    section(
        "📄",
        "Analyze Rental Agreements, Employment Contracts & NDAs",
        "Upload a text-based PDF to get a summary, risk meter, red flags, analytics and a document-grounded Q&A.",
        first=True,
    )

    up_col, demo_col = st.columns([3, 1])
    with up_col:
        uploaded_file = st.file_uploader("Upload legal document (PDF)", type=["pdf"], key="pdf_uploader")
    with demo_col:
        st.write("")
        st.write("")
        st.button("🚀 1-Click Demo (Sample Rental Agreement)", on_click=start_demo, key="demo_btn_tab1")
        if st.session_state.get("use_demo"):
            st.button("🧹 Clear demo", on_click=clear_demo, key="clear_demo_tab1")

    source = None
    if uploaded_file:
        source = ("upload", uploaded_file)
    elif st.session_state.get("use_demo"):
        source = ("demo", None)

    if source is None:
        callout("⬆️ Upload a PDF, or click the demo button to test the analyzer instantly.", "info")
    else:
        kind, handle = source
        if kind == "upload":
            file_id = f"upload-{handle.name}-{handle.size}"
            doc_name = handle.name
        else:
            file_id = "demo"
            doc_name = SAMPLE_NAME

        ready = True
        if st.session_state.get("file_id") != file_id:
            if st.session_state.get("index_failed_id") == file_id:
                callout(BUSY_MESSAGE, "warn")
                st.button("🔄 Retry indexing", on_click=retry_index, key="retry_index_btn")
                ready = False
            else:
                try:
                    with st.spinner("Reading and indexing your document..."):
                        if kind == "upload":
                            tmp_path = None
                            try:
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                                    tmp.write(handle.getvalue())
                                    tmp_path = tmp.name
                                pages = PyPDFLoader(tmp_path).load()
                            finally:
                                if tmp_path and os.path.exists(tmp_path):
                                    os.remove(tmp_path)
                        else:
                            pages = [
                                Document(page_content=t, metadata={"page": i, "source": SAMPLE_NAME})
                                for i, t in enumerate(SAMPLE_PAGES)
                            ]
                        ready = build_index(pages, doc_name)
                        if ready:
                            st.session_state["file_id"] = file_id
                            st.session_state.pop("index_failed_id", None)
                except AIUnavailable as e:
                    st.session_state["index_failed_id"] = file_id
                    show_problem(e)
                    st.button("🔄 Retry indexing", on_click=retry_index, key="retry_index_btn2")
                    ready = False
                except Exception as e:  # noqa: BLE001
                    show_problem(e, "Reading this document")
                    ready = False

        if ready and st.session_state.get("file_id") == file_id:
            callout(
                f"✅ <b>Document ready:</b> {esc(st.session_state['doc_name'])} · "
                f"{st.session_state['num_pages']} page(s), {st.session_state['num_chunks']} chunks indexed.",
                "ok",
            )

            # ---------- Analysis (once per language) ----------
            failed = st.session_state.setdefault("analysis_failed", {})
            if (
                lang not in st.session_state["analyses"]
                and lang not in st.session_state["analysis_raw"]
                and not failed.get(lang)
            ):
                try:
                    with st.spinner("Running legal analysis: summary, risk index, red flags..."):
                        run_analysis(lang)
                except Exception as e:  # noqa: BLE001
                    failed[lang] = True
                    show_problem(e, "The analysis")
            if failed.get(lang):
                st.button("🔄 Retry analysis", on_click=retry_analysis, args=(lang,), key="retry_analysis_btn")

            analysis = st.session_state["analyses"].get(lang)

            if analysis:
                if analysis.get("_truncated"):
                    callout(
                        "This document is very long; only the first part was used for the summary and risk analysis.",
                        "warn",
                    )

                score = safe_score(analysis)
                level = normalize_severity(analysis.get("risk_level", ""))
                tone = level.lower()
                flags_all = analysis.get("red_flags") or []
                high_n = sum(1 for f in flags_all if normalize_severity(f.get("severity")) == "High")
                kpi_tone = {"high": "bad", "medium": "warn", "low": "ok"}[tone]

                section("📑", "Executive Summary", str(analysis.get("document_type", "Legal document")))
                render(
                    f'<div class="card accent"><div class="card-title">'
                    f'{pill(str(analysis.get("document_type", "Legal document")), "info", dot=False, plain=True)}'
                    f'{pill(f"{level} risk", tone)}</div>'
                    f'<div>{esc_br(analysis.get("executive_summary", "No summary available."))}</div></div>'
                )

                section("🎯", "Risk Index Meter", "Heuristic score — higher means riskier terms")
                metric_row(
                    [
                        ("Risk score", f"{score}/100", f"{level} risk", kpi_tone),
                        ("Red flags", len(flags_all), "potential issues found", "bad" if flags_all else "ok"),
                        ("High severity", high_n, "need attention first", "bad" if high_n else "ok"),
                        ("Pages · Chunks", f"{st.session_state['num_pages']} · {st.session_state['num_chunks']}", "indexed in ChromaDB", ""),
                    ]
                )
                g1, g2 = st.columns([1, 1])
                with g1:
                    show_chart(gauge(score, "Overall Risk Score", True), "chart_risk_gauge")
                with g2:
                    reasons = analysis.get("risk_reasons") or []
                    items = "".join(f"<li>{esc(r)}</li>" for r in reasons) or "<li>No specific reasons returned.</li>"
                    render(
                        f'<div class="card"><div class="card-title"><div class="who">Why this rating</div>'
                        f'{pill(level, tone)}</div><div class="md"><ul>{items}</ul></div></div>'
                    )

                render_dashboard(analysis)

                key_terms = analysis.get("key_terms") or []
                if key_terms:
                    section("🔑", "Key Terms at a Glance")
                    rows = "".join(
                        f"<tr><td>{esc(k.get('term', ''))}</td><td>{esc_br(k.get('value', ''))}</td></tr>"
                        for k in key_terms
                        if isinstance(k, dict)
                    )
                    render(f'<table class="kt"><thead><tr><th>Term</th><th>What the document says</th></tr></thead><tbody>{rows}</tbody></table>')

                section("🚩", "Red Flag Detector", "Sorted by severity — highest risk first")
                if not flags_all:
                    callout("✅ No major red flags were detected. Still read the full document carefully.", "ok")
                else:
                    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
                    flags_sorted = sorted(
                        [f for f in flags_all if isinstance(f, dict)],
                        key=lambda f: order.get(normalize_severity(f.get("severity")).upper(), 3),
                    )
                    st.caption(f"{len(flags_sorted)} potential issue(s) found.")
                    for f in flags_sorted:
                        sev = normalize_severity(f.get("severity"))
                        quote_html = f'<div class="quote">{esc_br(f["quote"])}</div>' if f.get("quote") else ""
                        tip_html = (
                            f'<div class="tip">💡 <b>What you can do:</b> {esc_br(f["suggestion"])}</div>'
                            if f.get("suggestion")
                            else ""
                        )
                        render(
                            f'<div class="flag {sev.lower()}"><div class="card-title">'
                            f'<div class="flag-t">{esc(f.get("clause", "Clause"))}</div>'
                            f'<div class="flag-pills">{pill(sev, sev.lower())}'
                            f'{pill(str(f.get("category", "Other")), "neutral", dot=False, plain=True)}</div></div>'
                            f'{quote_html}'
                            f'<div class="why"><b>Why it matters:</b> {esc_br(f.get("why_it_matters", ""))}</div>'
                            f'{tip_html}</div>'
                        )

                # ---------- Export ----------
                section("📥", "Export Analysis")
                summary_txt = analysis_to_text(analysis, st.session_state["doc_name"], st.session_state["stats"])
                e1, e2, _ = st.columns([1, 1, 2])
                with e1:
                    pdf_button("⬇️ Summary (PDF)", summary_txt, "legal_eagle_analysis.pdf", "dl_analysis_pdf")
                with e2:
                    st.download_button(
                        "⬇️ Summary (.txt)",
                        data=summary_txt.encode("utf-8"),
                        file_name="legal_eagle_analysis.txt",
                        mime="text/plain",
                        key="dl_analysis_txt",
                    )
                if lang in NATIVE_SCRIPT_LANGS:
                    st.caption(
                        "Tip: the .txt export keeps every character of your language. The PDF uses a system Unicode "
                        "font (e.g. Nirmala UI on Windows) when one is available."
                    )

            elif lang in st.session_state["analysis_raw"]:
                section("📑", "Document Analysis")
                md_card(st.session_state["analysis_raw"][lang], accent=True)

            # ---------- Interactive Q&A ----------
            section("💬", "Ask Anything About Your Document", "Answers are grounded in your PDF, with similarity confidence for every source passage")
            with st.form("doc_qa_form", clear_on_submit=True):
                user_q = st.text_input(
                    "Your question",
                    placeholder="e.g. What is the notice period? Can the landlord keep my deposit?",
                )
                ask_btn = st.form_submit_button("🔍 Query Document", type="primary")

            qa_ok = False
            if ask_btn and user_q.strip():
                try:
                    with st.spinner("Searching the document and analyzing clauses..."):
                        scored = retrieve_with_scores(st.session_state["vectorstore"], user_q, k=5)
                        docs = [d for d, _ in scored]
                        answer = ask_llm_failsafe(
                            QA_PROMPT.format(
                                lang_instruction=LANGUAGES[lang],
                                context=format_docs(docs),
                                question=user_q,
                            ),
                            temperature=0.2,
                        )
                        st.session_state["last_scores"] = {
                            "query": user_q,
                            "scores": [s for _, s in scored],
                            "pages": [
                                (d.metadata.get("page") + 1) if isinstance(d.metadata.get("page"), int) else "?"
                                for d in docs
                            ],
                        }
                        st.session_state["qa_history"].insert(
                            0,
                            {
                                "q": user_q,
                                "a": answer,
                                "sources": [(d.metadata.get("page"), d.page_content, s) for d, s in scored],
                            },
                        )
                        qa_ok = True
                except Exception as e:  # noqa: BLE001
                    show_problem(e, "Answering that question")
            if qa_ok:
                st.rerun()

            for item in st.session_state.get("qa_history", []):
                render(f'<div class="qa-q"><span class="who">🧑 You asked</span>{esc_br(item["q"])}</div>')
                md_card(item["a"], accent=True, who="⚖️ Legal Eagle")
                with st.expander(f"📎 Source passages used ({len(item['sources'])}) · with similarity confidence"):
                    for page, content, s in item["sources"]:
                        pg = page + 1 if isinstance(page, int) else "?"
                        render(
                            f'<div class="src">{conf_pill(s)} <span class="muted">&nbsp;Page {pg}</span>'
                            f'<div class="src-t">{esc_br(content)}</div></div>'
                        )

            disclaimer()

# =====================================================================
# TAB 2: PUBLIC LEGAL RIGHTS & AWARENESS
# =====================================================================
with tab2:
    section(
        "🌐",
        "Know Your Rights",
        "Consumer Act, Fundamental Rights, IT Act, Traffic Rules, Tenancy, Employment, RTI and more",
        first=True,
    )
    topic = st.selectbox("📚 Topic area", TOPICS, key="topic_choice")
    callout(f"🌐 Response language: <b>{esc(lang)}</b> — change it from the sidebar.", "info")

    with st.form("rights_form"):
        public_q = st.text_area(
            "Describe your situation or question",
            placeholder="e.g. A shopkeeper is refusing to refund a defective phone. What can I do?",
            height=110,
        )
        go_btn = st.form_submit_button("⚖️ Get Legal Guidance", type="primary")

    if go_btn:
        if not public_q.strip():
            callout("Please describe your situation or question first.", "warn")
        else:
            try:
                with st.spinner("Preparing your rights and action plan..."):
                    st.session_state["rights_answer"] = ask_llm_failsafe(
                        RIGHTS_PROMPT.format(lang_instruction=LANGUAGES[lang], topic=topic, question=public_q),
                        temperature=0.3,
                    )
                    st.session_state["rights_lang"] = lang
            except Exception as e:  # noqa: BLE001
                show_problem(e, "Fetching legal guidance")

    if st.session_state.get("rights_answer"):
        section("⚖️", "Legal Explanation & Action Plan", f"Language: {st.session_state.get('rights_lang', lang)}")
        md_card(st.session_state["rights_answer"], accent=True)
        r1, r2, _ = st.columns([1, 1, 2])
        with r1:
            pdf_button("⬇️ Download PDF", st.session_state["rights_answer"], "legal_guidance.pdf", "dl_rights_pdf")
        with r2:
            st.download_button(
                "⬇️ Download .txt",
                data=st.session_state["rights_answer"].encode("utf-8"),
                file_name="legal_guidance.txt",
                mime="text/plain",
                key="dl_rights_txt",
            )

    st.write("")
    with st.expander("📞 Quick Help: Official Helplines & Portals", expanded=False):
        st.markdown(HELP_RESOURCES)

    disclaimer()

# =====================================================================
# TAB 3: LEGAL NOTICE GENERATOR
# =====================================================================
with tab3:
    section(
        "✍️",
        "Generate a Professional Legal Notice Draft",
        "Anything you leave blank becomes a [PLACEHOLDER] in the draft. The notice is drafted in formal English; "
        "a plain-language explanation follows in your selected language.",
        first=True,
    )

    with st.form("notice_form"):
        n1, n2 = st.columns(2)
        with n1:
            sender = st.text_input("Your Name (Sender)")
            sender_addr = st.text_area("Your Address", height=80)
            category = st.selectbox(
                "Notice Type",
                [
                    "Consumer dispute (defective product / deficient service / refund)",
                    "Landlord - security deposit refund",
                    "Tenant / rent dispute",
                    "Payment recovery / unpaid dues",
                    "Employment - unpaid salary / wrongful termination",
                    "Other",
                ],
            )
            txn_date = st.text_input("Date of purchase / agreement (optional)")
        with n2:
            role = st.selectbox("Recipient is a", ["Seller", "Landlord", "Service Provider", "Employer", "Other"])
            recipient = st.text_input("Recipient Name (Seller / Landlord / Employer)")
            recipient_addr = st.text_area("Recipient's Address", height=80)
            amount = st.text_input("Claim Amount (₹)", placeholder="e.g. 25000")
        reason = st.text_area(
            "Dispute Reason (describe what happened)",
            placeholder="e.g. I bought a laptop on 5 Aug 2026 for ₹60,000. It stopped working in 10 days. The seller refuses replacement or refund.",
            height=120,
        )
        n3, n4 = st.columns(2)
        with n3:
            days = st.number_input("Days given to comply", min_value=7, max_value=60, value=15)
        with n4:
            city = st.text_input("City / Jurisdiction", placeholder="e.g. Pune, Maharashtra")
        make_btn = st.form_submit_button("✍️ Generate Notice", type="primary")

    if make_btn:
        if not reason.strip():
            callout("Please describe the dispute reason.", "warn")
        else:
            try:
                with st.spinner("Drafting your legal notice..."):
                    notice = ask_llm_failsafe(
                        NOTICE_PROMPT.format(
                            sender=sender or "[SENDER NAME]",
                            sender_addr=sender_addr or "[SENDER ADDRESS]",
                            role=role,
                            recipient=recipient or "[RECIPIENT NAME]",
                            recipient_addr=recipient_addr or "[RECIPIENT ADDRESS]",
                            category=category,
                            txn_date=txn_date or "[DATE]",
                            reason=reason,
                            amount=f"Rs. {amount}" if amount.strip() else "[CLAIM AMOUNT]",
                            days=int(days),
                            city=city or "[CITY]",
                            today=date.today().strftime("%d %B %Y"),
                        ),
                        temperature=0.3,
                    )
                st.session_state["notice"] = notice
                st.session_state["notice_editor"] = notice
                st.session_state.pop("notice_explain", None)
                st.session_state.pop("notice_explain_lang", None)
            except Exception as e:  # noqa: BLE001
                show_problem(e, "Drafting the notice")

    if st.session_state.get("notice"):
        section("📜", "Notice Preview")
        md_card(st.session_state["notice"])

        # Plain-language explanation in the selected language (the notice itself stays formal English)
        if st.session_state.get("notice_explain_lang") != lang:
            try:
                with st.spinner("Preparing a plain-language explanation..."):
                    st.session_state["notice_explain"] = ask_llm_failsafe(
                        NOTICE_EXPLAIN_PROMPT.format(
                            lang_instruction=LANGUAGES[lang], notice=st.session_state["notice"]
                        ),
                        temperature=0.2,
                    )
                    st.session_state["notice_explain_lang"] = lang
            except Exception as e:  # noqa: BLE001
                st.session_state["notice_explain"] = ""
                st.session_state["notice_explain_lang"] = lang
                show_problem(e, "The explanation")
                st.button("🔄 Retry explanation", on_click=retry_explain, key="retry_explain_btn")
        if st.session_state.get("notice_explain"):
            section("🗣️", f"Notice Explained — {lang}")
            md_card(st.session_state["notice_explain"], accent=True)

        section("✏️", "Edit Before Downloading", "Make final changes here — the downloads below use this text.")
        edited = st.text_area("Editable draft", key="notice_editor", height=350, label_visibility="collapsed")

        d1, d2, _ = st.columns([1, 1, 2])
        with d1:
            pdf_button("⬇️ Download PDF", edited, "legal_notice.pdf", "dl_notice_pdf")
        with d2:
            st.download_button(
                "⬇️ Download .txt",
                data=edited.encode("utf-8"),
                file_name="legal_notice.txt",
                mime="text/plain",
                key="dl_notice_txt",
            )

        callout(
            "This is an AI-generated draft. Fill in all [PLACEHOLDERS], verify facts and laws, "
            "and have an advocate review it before sending (preferably by Registered Post A.D.).",
            "warn",
        )
    disclaimer()