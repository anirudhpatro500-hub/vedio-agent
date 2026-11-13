import html
import inspect
import time

from dotenv import load_dotenv

# FIX: load env vars BEFORE importing project modules (they may read API keys at import time)
load_dotenv()

import streamlit as st

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

try:
    import markdown as _md  # optional: pip install markdown  # type: ignore[import-not-found]
except ImportError:
    _md = None

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def wide_button(label: str, *, key: str | None = None) -> bool:
    """Streamlit button wrapper that works across older and newer versions."""
    if "width" in inspect.signature(st.button).parameters:
        return st.button(label, key=key, width="stretch")
    return st.button(label, key=key, use_container_width=True)


def wide_submit(label: str) -> bool:
    """Streamlit form submit wrapper that works across older and newer versions."""
    if "width" in inspect.signature(st.form_submit_button).parameters:
        return st.form_submit_button(label, width="stretch")
    return st.form_submit_button(label, use_container_width=True)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
# To recolour the whole app, change the variables in :root (the *-rgb ones are
# comma-separated RGB triplets so transparent tints follow the main colour).
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* ── Root Variables ── */
:root {
    --bg: #0a0a0f;
    --surface: #111118;
    --surface-2: #1a1a25;
    --border: #2a2a3a;

    --accent: #7c3aed;
    --accent-rgb: 124, 58, 237;
    --accent-dark: #5b21b6;
    --accent-glow: #9f67ff;

    --accent-2: #06b6d4;
    --accent-2-rgb: 6, 182, 212;

    --text: #e8e8f0;
    --text-strong: #ffffff;
    --text-muted: #9090b8;

    --success: #10b981;
    --success-rgb: 16, 185, 129;
    --warning: #f59e0b;
    --danger: #ef4444;
}

/* ── Global ── */
html, body { background-color: var(--bg); }

.stApp, .stApp input, .stApp textarea, .stApp button {
    font-family: 'JetBrains Mono', monospace;   /* FIX: not applied to every element, so icon fonts keep working */
}

.stApp {
    background: var(--bg) !important;
    color: var(--text);
}

/* Grid background */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image:
        linear-gradient(rgba(var(--accent-rgb), 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(var(--accent-rgb), 0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

/* FIX: no more `* { color: ... !important }` — it was wiping badge colours */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    color: var(--text);
}

/* ── Headings ── */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Syne', sans-serif !important;
    color: var(--text) !important;
}

/* ── Hero Title ── */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2rem, 5vw, 3.5rem);
    font-weight: 800;
    line-height: 1.1;
    margin: 0;
    background: linear-gradient(135deg, var(--text-strong) 0%, var(--accent-glow) 50%, var(--accent-2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: var(--text-muted);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: 0.5rem;
}

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}

.card:hover { border-color: var(--accent); }

.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, var(--accent), var(--accent-2));
}

.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.card-content {
    font-size: 0.875rem;
    line-height: 1.7;
    color: var(--text);
    word-break: break-word;
}

.card-content p, .chat-bubble p { margin: 0 0 0.5rem 0; }
.card-content p:last-child, .chat-bubble p:last-child { margin-bottom: 0; }
.card-content ul, .card-content ol, .chat-bubble ul, .chat-bubble ol { margin: 0 0 0.5rem 0; padding-left: 1.2rem; }

/* ── Badges ── */
.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.badge-purple { background: rgba(var(--accent-rgb), 0.2);   color: var(--accent-glow); border: 1px solid rgba(var(--accent-rgb), 0.3); }
.badge-cyan   { background: rgba(var(--accent-2-rgb), 0.15); color: var(--accent-2);    border: 1px solid rgba(var(--accent-2-rgb), 0.3); }
.badge-green  { background: rgba(var(--success-rgb), 0.15);  color: var(--success);     border: 1px solid rgba(var(--success-rgb), 0.3); }

/* ── Inputs ── */
.stTextInput > div > div > input,
.stSelectbox > div > div {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(var(--accent-rgb), 0.2) !important;
}

/* ── Buttons ── */
/* FIX: Streamlit gives EVERY default button kind="secondary", so the old
   `button[kind="secondary"]` rule was turning Analyse/Send grey. All buttons are
   gradient now; Clear Chat gets its own override via its key. */
.stButton > button,
[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, var(--accent), var(--accent-dark)) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s !important;
    text-transform: uppercase !important;
}

.stButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(var(--accent-rgb), 0.4) !important;
}

.st-key-clear_btn button {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
}

/* Remove the default form border so the chat input looks the same as before */
[data-testid="stForm"] { border: none !important; padding: 0 !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--surface);
    border: 1px solid var(--border) !important;
    border-radius: 12px;
}

/* ── Pipeline status ── */
.status-bar {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.75rem 1rem;
    background: var(--surface-2);
    border-radius: 8px;
    margin: 0.4rem 0;
    border: 1px solid var(--border);
    font-size: 0.8rem;
}

.status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}

.dot-active  { background: var(--accent-glow); box-shadow: 0 0 8px var(--accent-glow); animation: pulse 1.5s infinite; }
.dot-done    { background: var(--success); }
.dot-error   { background: var(--danger); }
.dot-pending { background: var(--border); }

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%      { opacity: 0.4; }
}

/* ── Chat ── */
.chat-container {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem;
    max-height: 420px;
    overflow-y: auto;
    margin-bottom: 1rem;
}

.chat-msg {
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
}

.chat-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

.chat-bubble {
    display: inline-block;
    padding: 0.6rem 1rem;
    border-radius: 10px;
    font-size: 0.85rem;
    line-height: 1.6;
    max-width: 90%;
    word-break: break-word;
}

.user-label { color: var(--accent-glow); }
.bot-label  { color: var(--accent-2); }

.user-bubble { background: rgba(var(--accent-rgb), 0.15);   border: 1px solid rgba(var(--accent-rgb), 0.25);   align-self: flex-end; }
.bot-bubble  { background: rgba(var(--accent-2-rgb), 0.10); border: 1px solid rgba(var(--accent-2-rgb), 0.20); align-self: flex-start; }

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.5rem 0 !important;
}

/* ── Transcript box ── */
.transcript-box {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.25rem;
    font-size: 0.82rem;
    line-height: 1.8;
    max-height: 300px;
    overflow-y: auto;
    color: var(--text-muted);
    word-break: break-word;
}

/* ── Misc Streamlit elements ── */
.stProgress > div > div > div { background: var(--accent) !important; }
.stSpinner > div { border-top-color: var(--accent) !important; }
[data-testid="stMarkdownContainer"] p { color: var(--text) !important; }
label { color: var(--text-muted) !important; font-size: 0.8rem !important; }

/* scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ──────────────────────────────────────────────────────────
for key, default in {
    "result": None,
    "chat_history": [],
    "processing": False,
    "pipeline_done": False,
    "pipeline_steps": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Helpers ────────────────────────────────────────────────────────────────────
PIPELINE = [
    ("audio",      "🔊", "Audio Processing"),
    ("transcript", "📝", "Transcription"),
    ("title",      "🏷️", "Title Generation"),
    ("summary",    "📋", "Summarisation"),
    ("extract",    "🔍", "Extraction"),
    ("rag",        "🧠", "RAG Engine"),
]


def plain_html(value) -> str:
    """Escape untrusted text (so < > & can't break the layout) and keep line breaks."""
    return html.escape("" if value is None else str(value).strip(), quote=False).replace("\n", "<br>")


def to_html(value) -> str:
    """LLM output -> safe HTML. Renders markdown (bullets, bold) if `markdown` is installed."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = "\n".join(f"- {v}" for v in value)
    text = str(value).strip()
    if _md is None:
        return plain_html(text)
    out = _md.markdown(html.escape(text, quote=False), extensions=["nl2br", "sane_lists"])
    # Blank lines would end the HTML block inside st.markdown and break the card
    return "\n".join(line for line in out.splitlines() if line.strip())


def card(title: str, body_html: str) -> str:
    # Built without indentation so Streamlit's markdown parser never treats it as a code block
    return (
        '<div class="card">'
        f'<div class="card-title">{title}</div>'
        f'<div class="card-content">{body_html}</div>'
        '</div>'
    )


def chat_html(history) -> str:
    parts = ['<div class="chat-container">']
    for msg in history:
        is_user = msg["role"] == "user"
        parts.append(
            f'<div class="chat-msg" style="align-items:{"flex-end" if is_user else "flex-start"}">'
            f'<span class="chat-label {"user-label" if is_user else "bot-label"}">{"You" if is_user else "🤖 Assistant"}</span>'
            f'<div class="chat-bubble {"user-bubble" if is_user else "bot-bubble"}">{to_html(msg["content"])}</div>'
            '</div>'
        )
    parts.append('</div>')
    return "".join(parts)


def step_status(steps: dict, key: str) -> str:
    return {
        "active": "dot-active",
        "done": "dot-done",
        "error": "dot-error",
    }.get(steps.get(key, "pending"), "dot-pending")


def render_step_bar(label: str, key: str, icon: str):
    css = step_status(st.session_state.pipeline_steps, key)
    st.markdown(
        f'<div class="status-bar"><div class="status-dot {css}"></div><span>{icon} {label}</span></div>',
        unsafe_allow_html=True,
    )


def draw_status():
    """Render the pipeline status into the sidebar slot (called again on every step -> live updates)."""
    if not st.session_state.pipeline_steps:
        status_slot.empty()
        return
    with status_slot.container():
        st.markdown("---")
        st.markdown('<span class="badge badge-green">Pipeline Status</span>', unsafe_allow_html=True)
        for step, icon, label in PIPELINE:
            render_step_bar(label, step, icon)


def update_step(key: str, state: str):
    st.session_state.pipeline_steps[key] = state
    draw_status()


# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:1.6rem">🎬 AI<br>Video</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Meeting Intelligence</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<span class="badge badge-purple">Input</span>', unsafe_allow_html=True)
    source = st.text_input("YouTube URL or File Path", placeholder="https://youtube.com/watch?v=... or /path/to/file.mp4")

    language = st.selectbox("Language", ["english", "hinglish"], index=0)

    run_btn = st.button("⚡  Analyse", use_container_width=True)

    # FIX: status now lives in a placeholder so it really updates live while the pipeline runs
    status_slot = st.empty()
    draw_status()

# ─── Main Area ──────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">AI Video Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Transcribe · Summarise · Chat with your meetings</div>', unsafe_allow_html=True)
st.markdown("---")

# ── Run Pipeline ────────────────────────────────────────────────────────────────
if run_btn:
    if not source.strip():
        st.error("Please enter a YouTube URL or file path.")
    else:
        st.session_state.pipeline_done = False
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.pipeline_steps = {}
        draw_status()

        progress_placeholder = st.empty()
        succeeded = False

        try:
            with progress_placeholder.container():
                st.info("⚙️ Pipeline running — see sidebar for live status…")

            update_step("audio", "active")
            chunks = process_input(source.strip())
            update_step("audio", "done")

            update_step("transcript", "active")
            transcript = transcribe_all(chunks, language)
            update_step("transcript", "done")

            update_step("title", "active")
            title = generate_title(transcript)
            update_step("title", "done")

            update_step("summary", "active")
            summary = summarize(transcript)
            update_step("summary", "done")

            update_step("extract", "active")
            action_items = extract_action_items(transcript)
            decisions    = extract_key_decisions(transcript)
            questions    = extract_questions(transcript)
            update_step("extract", "done")

            update_step("rag", "active")
            rag_chain = build_rag_chain(transcript)
            update_step("rag", "done")

            st.session_state.result = {
                "title": title,
                "transcript": transcript,
                "summary": summary,
                "action_items": action_items,
                "key_decisions": decisions,
                "open_questions": questions,
                "rag_chain": rag_chain,
            }
            st.session_state.pipeline_done = True
            succeeded = True

        except Exception as e:
            # FIX: failed step is shown in red in the sidebar instead of silently reset
            for k, _, _ in PIPELINE:
                if st.session_state.pipeline_steps.get(k) == "active":
                    st.session_state.pipeline_steps[k] = "error"
            draw_status()
            progress_placeholder.error(f"❌ Error: {e}")

        # FIX: rerun moved outside try/except so it can never be swallowed as an "error"
        if succeeded:
            progress_placeholder.success("✅ Analysis complete!")
            time.sleep(0.5)
            progress_placeholder.empty()
            st.rerun()

# ── Results ──────────────────────────────────────────────────────────────────────
if st.session_state.result:
    r = st.session_state.result

    # Title banner
    st.markdown(
        '<div class="card">'
        '<div class="card-title">📌 Session Title</div>'
        '<div style="font-family:\'Syne\',sans-serif;font-size:1.4rem;font-weight:700;color:var(--text)">'
        f'{plain_html(r["title"])}</div></div>',
        unsafe_allow_html=True,
    )

    # Top row: summary + transcript
    col1, col2 = st.columns([3, 2], gap="medium")

    with col1:
        st.markdown(card("📋 Summary", to_html(r["summary"])), unsafe_allow_html=True)

    with col2:
        with st.expander("📝 Full Transcript", expanded=False):
            st.markdown(f'<div class="transcript-box">{plain_html(r["transcript"])}</div>', unsafe_allow_html=True)

    # Second row: action items | decisions | questions
    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown(card("✅ Action Items", to_html(r["action_items"])), unsafe_allow_html=True)
    with c2:
        st.markdown(card("🔑 Key Decisions", to_html(r["key_decisions"])), unsafe_allow_html=True)
    with c3:
        st.markdown(card("❓ Open Questions", to_html(r["open_questions"])), unsafe_allow_html=True)

    st.markdown("---")

    # ── RAG Chat ──────────────────────────────────────────────────────────────
    st.markdown(
        '<div style="font-family:\'Syne\',sans-serif;font-size:1.2rem;font-weight:700;margin-bottom:1rem">💬 Chat with your Meeting</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.chat_history:
        st.markdown(chat_html(st.session_state.chat_history), unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="card" style="text-align:center;padding:2rem">'
            '<div style="font-size:2rem;margin-bottom:0.5rem">💬</div>'
            '<div style="color:var(--text-muted);font-size:0.85rem">Ask anything about your meeting transcript</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # FIX: form -> Enter key sends, and the box clears itself after sending
    with st.form("chat_form", clear_on_submit=True):
        chat_col1, chat_col2 = st.columns([5, 1], gap="small")
        with chat_col1:
            user_input = st.text_input(
                "Your question",
                placeholder="What were the main decisions made?",
                label_visibility="collapsed",
            )
        with chat_col2:
            if "width" in inspect.signature(st.form_submit_button).parameters:
                send_btn = st.form_submit_button("Send →", width="stretch")
            else:
                send_btn = st.form_submit_button("Send →", use_container_width=True)

    if send_btn and user_input.strip():
        question = user_input.strip()
        try:
            with st.spinner("Thinking…"):
                answer = ask_question(r["rag_chain"], question)
        except Exception as e:
            answer = f"⚠️ Sorry, I couldn't answer that: {e}"
        st.session_state.chat_history.append({"role": "user", "content": question})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", key="clear_btn"):
            st.session_state.chat_history = []
            st.rerun()

else:
    # Empty state
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:5rem 2rem;text-align:center">
        <div style="font-size:4rem;margin-bottom:1rem">🎬</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:700;color:var(--text);margin-bottom:0.5rem">
            Ready to Analyse
        </div>
        <div style="color:var(--text-muted);font-size:0.85rem;max-width:380px;line-height:1.7">
            Paste a YouTube URL or local file path in the sidebar, choose your language, and hit <strong>Analyse</strong> to get started.
        </div>
        <div style="margin-top:2rem;display:flex;gap:1rem;flex-wrap:wrap;justify-content:center">
            <span class="badge badge-purple">Transcription</span>
            <span class="badge badge-cyan">Summarisation</span>
            <span class="badge badge-green">RAG Chat</span>
        </div>
    </div>""", unsafe_allow_html=True)
