"""Streamlit chat UI for the NileTel RAG assistant — modern branded edition."""
from __future__ import annotations

import os
import re
import uuid

import httpx
import streamlit as st


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
USE_STREAMING = os.getenv("USE_STREAMING", "true").lower() == "true"


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="NileTel Assistant",
    page_icon="📡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ============================================================
# CSS — modern, light, branded
# ============================================================
st.markdown(
    """
<style>
/* ---------- global ---------- */
html, body, .stApp, [data-testid="stAppViewContainer"] {
    background: #FFFFFF !important;
    color: #1A2138 !important;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none !important; }
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 8rem !important;
    max-width: 820px !important;
}

/* hide Streamlit default footer/branding */
footer, [data-testid="stStatusWidget"] { display: none !important; }

/* ---------- HERO HEADER ---------- */
.nt-hero {
    background:
      radial-gradient(circle at 90% 0%, rgba(255,255,255,.18) 0%, transparent 40%),
      linear-gradient(135deg, #00B4DB 0%, #0083B0 60%, #00568C 100%);
    color: #FFFFFF;
    padding: 28px 32px;
    border-radius: 24px;
    box-shadow: 0 20px 50px -12px rgba(0, 131, 176, .35);
    margin-bottom: 28px;
    display: flex; align-items: center; justify-content: space-between;
    position: relative; overflow: hidden;
}
.nt-hero::after {
    content: "";
    position: absolute; top: -40%; right: -10%;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(255,255,255,.14) 0%, transparent 70%);
    border-radius: 50%;
}
.nt-hero-text { position: relative; z-index: 1; }
.nt-hero-text h1 {
    color: #FFFFFF !important;
    margin: 0; font-size: 1.7rem; font-weight: 800;
    letter-spacing: -.5px;
}
.nt-hero-text p {
    color: rgba(255,255,255,.92) !important;
    margin: 6px 0 0; font-size: .92rem; font-weight: 500;
}
.nt-status-pill {
    background: rgba(255,255,255,.22);
    color: #FFFFFF !important;
    padding: 8px 16px; border-radius: 999px;
    font-size: .8rem; font-weight: 700;
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,.28);
    position: relative; z-index: 1;
    display: inline-flex; align-items: center; gap: 8px;
}
.nt-pulse {
    width: 8px; height: 8px; border-radius: 50%;
    background: #5CFFAE;
    box-shadow: 0 0 0 0 rgba(92, 255, 174, .6);
    animation: nt-pulse 1.6s infinite;
}
@keyframes nt-pulse {
    0% { box-shadow: 0 0 0 0 rgba(92, 255, 174, .6); }
    70% { box-shadow: 0 0 0 10px rgba(92, 255, 174, 0); }
    100% { box-shadow: 0 0 0 0 rgba(92, 255, 174, 0); }
}

/* ---------- section title ---------- */
.nt-section-title {
    font-weight: 700; font-size: .75rem;
    color: #6B7488 !important;
    text-transform: uppercase; letter-spacing: 1.2px;
    margin: 8px 0 12px;
    display: flex; align-items: center; gap: 8px;
}
.nt-section-title::before {
    content: ""; width: 16px; height: 2px;
    background: linear-gradient(90deg, #00B4DB, #0083B0);
    border-radius: 2px;
}

/* ---------- email input ---------- */
.stTextInput > div > div > input {
    background: #F7F9FC !important;
    border: 2px solid #E7EDF3 !important;
    border-radius: 14px !important;
    color: #1A2138 !important;
    padding: 10px 14px !important;
    font-size: .9rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #00B4DB !important;
    box-shadow: 0 0 0 4px rgba(0, 180, 219, .12) !important;
}
.stTextInput label { color: #6B7488 !important; font-weight: 600 !important; }

/* ---------- chip buttons ---------- */
.stButton > button {
    background: #FFFFFF !important;
    color: #0083B0 !important;
    border: 1.5px solid #D5E3EE !important;
    border-radius: 999px !important;
    padding: 8px 14px !important;
    font-weight: 600 !important;
    font-size: .82rem !important;
    box-shadow: 0 1px 3px rgba(0,0,0,.04) !important;
    transition: all 200ms cubic-bezier(.4,0,.2,1) !important;
    white-space: nowrap !important;
}
.stButton > button:hover:not(:disabled) {
    background: linear-gradient(135deg, #00B4DB 0%, #0083B0 100%) !important;
    color: #FFFFFF !important;
    border-color: transparent !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 18px rgba(0, 131, 176, .3) !important;
}
.stButton > button:disabled {
    opacity: .55 !important;
    cursor: not-allowed !important;
}

/* ---------- chat messages ---------- */
[data-testid="stChatMessage"] {
    background: #FFFFFF !important;
    border: 1px solid #E7EDF3 !important;
    border-radius: 18px !important;
    box-shadow: 0 4px 12px rgba(0,0,0,.04) !important;
    padding: 16px 20px !important;
    margin-bottom: 14px !important;
    color: #1A2138 !important;
    animation: nt-slide-in .35s cubic-bezier(.16,1,.3,1);
}
@keyframes nt-slide-in {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
[data-testid="stChatMessage"] * { color: #1A2138 !important; }

/* user bubble — tinted gradient */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: linear-gradient(135deg, #EAF8FC 0%, #D8F1F8 100%) !important;
    border-color: #B8E4EF !important;
}

/* RTL support — Arabic content reads right-to-left.
   We apply RTL to ALL chat messages because our assistant always replies
   in Arabic and most user input is Arabic. English words inside (like
   "restart", "RSRP") still render correctly thanks to Unicode bidi. */
[data-testid="stChatMessage"] [data-testid="stMarkdown"],
[data-testid="stChatMessage"] [data-testid="stMarkdown"] * {
    direction: rtl !important;
    text-align: right !important;
    unicode-bidi: plaintext !important;
}
/* Numbered/bulleted lists need their markers on the right side */
[data-testid="stChatMessage"] ol,
[data-testid="stChatMessage"] ul {
    padding-right: 1.4em !important;
    padding-left: 0 !important;
}
/* Chat input box: right-align placeholder + content for Arabic */
[data-testid="stChatInput"] textarea {
    direction: rtl !important;
    text-align: right !important;
    unicode-bidi: plaintext !important;
}

/* bold text in answers — extra emphasis */
[data-testid="stChatMessage"] strong {
    color: #0083B0 !important;
    font-weight: 700 !important;
    background: linear-gradient(180deg, transparent 60%, #E5F5FB 60%);
    padding: 0 2px;
}

/* ---------- chat input bar (BOTTOM) ---------- */
[data-testid="stChatInputContainer"],
[data-testid="stBottomBlockContainer"],
[data-testid="stBottom"] {
    background: rgba(255,255,255,.92) !important;
    border-top: 1px solid #E7EDF3 !important;
    backdrop-filter: blur(12px);
}
[data-testid="stChatInput"] {
    background: #F7F9FC !important;
    border: 2px solid #E7EDF3 !important;
    border-radius: 999px !important;
    box-shadow: 0 6px 18px rgba(0,0,0,.06) !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #00B4DB !important;
    box-shadow: 0 6px 22px rgba(0, 180, 219, .2) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #1A2138 !important;
    font-size: .96rem !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #9AA3B7 !important; }

/* ---------- expander (sources) ---------- */
[data-testid="stExpander"] {
    background: #FAFBFD !important;
    border: 1px solid #E7EDF3 !important;
    border-radius: 14px !important;
    margin-top: 12px !important;
}
[data-testid="stExpander"] summary p {
    color: #0083B0 !important; font-weight: 700 !important;
}

/* ---------- source mini-cards inside expander ---------- */
.nt-src-card {
    background: #FFFFFF;
    border: 1px solid #E7EDF3;
    border-radius: 12px;
    padding: 10px 14px;
    margin: 6px 0;
    display: flex; justify-content: space-between; align-items: center;
}
.nt-src-name {
    color: #1A2138 !important; font-weight: 600; font-size: .88rem;
}
.nt-src-crumb {
    color: #6B7488 !important; font-size: .78rem; margin-top: 2px;
}
.nt-src-score {
    background: #ECFDF5; color: #047857 !important;
    padding: 3px 10px; border-radius: 999px;
    font-size: .72rem; font-weight: 700;
}

/* ---------- badges row ---------- */
.nt-badges { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }
.nt-badge {
    background: #EEF3F7;
    color: #1A2138 !important;
    border-radius: 999px;
    padding: 4px 11px;
    font-size: .72rem;
    font-weight: 700;
    border: 1px solid transparent;
}
.nt-badge.intent  { background: #E5F5FB; color: #0083B0 !important; border-color: #B8E4EF; }
.nt-badge.tier    { background: #FFF4E6; color: #B26F00 !important; border-color: #FFE0B2; }
.nt-badge.latency { background: #ECFDF5; color: #047857 !important; border-color: #C8F0DC; }

/* ---------- TICKET CARD (the special one) ---------- */
.nt-ticket {
    background:
      radial-gradient(circle at 0% 0%, rgba(255,255,255,.25) 0%, transparent 40%),
      linear-gradient(135deg, #FF6B9E 0%, #B30060 100%);
    color: #FFFFFF !important;
    border-radius: 18px;
    padding: 18px 22px;
    margin-top: 14px;
    box-shadow: 0 12px 28px -8px rgba(179, 0, 96, .4);
    position: relative; overflow: hidden;
    animation: nt-card-pop .5s cubic-bezier(.16,1,.3,1);
}
@keyframes nt-card-pop {
    from { opacity: 0; transform: scale(.96) translateY(6px); }
    to   { opacity: 1; transform: scale(1) translateY(0); }
}
.nt-ticket * { color: #FFFFFF !important; }
.nt-ticket-head {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 12px;
}
.nt-ticket-title { font-weight: 800; font-size: 1.05rem; letter-spacing: .2px; }
.nt-ticket-status {
    background: rgba(255,255,255,.22);
    padding: 4px 12px; border-radius: 999px;
    font-size: .72rem; font-weight: 800;
    border: 1px solid rgba(255,255,255,.3);
}
.nt-ticket-grid {
    display: grid; grid-template-columns: 100px 1fr; gap: 6px 12px;
    font-size: .85rem;
}
.nt-ticket-grid .k {
    color: rgba(255,255,255,.78) !important; font-weight: 600;
}
.nt-ticket-grid .v {
    color: #FFFFFF !important; font-weight: 700;
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
}
.nt-ticket-foot {
    display: flex; gap: 10px; flex-wrap: wrap;
    margin-top: 14px; padding-top: 12px;
    border-top: 1px solid rgba(255,255,255,.18);
    font-size: .78rem;
}
.nt-ticket-foot span {
    background: rgba(255,255,255,.15);
    padding: 4px 10px; border-radius: 999px;
    font-weight: 600;
}
.nt-ticket-foot a {
    color: #FFFFFF !important; text-decoration: underline; font-weight: 600;
}

/* ---------- typing indicator ---------- */
.nt-typing {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 10px 16px;
    background: linear-gradient(135deg, #F0F4F8 0%, #E5EDF5 100%);
    border-radius: 18px;
    border: 1px solid #E7EDF3;
}
.nt-typing span {
    display: inline-block;
    width: 8px; height: 8px;
    background: #0083B0;
    border-radius: 50%;
    animation: nt-bounce 1.2s infinite ease-in-out;
}
.nt-typing span:nth-child(2) { animation-delay: .15s; }
.nt-typing span:nth-child(3) { animation-delay: .30s; }
@keyframes nt-bounce {
    0%, 60%, 100% { transform: translateY(0); opacity: .4; }
    30% { transform: translateY(-7px); opacity: 1; }
}

/* column padding tighten */
[data-testid="column"] { padding: 0 4px !important; }

/* footer */
.nt-footer {
    text-align: center;
    color: #95A0B5 !important;
    font-size: .75rem;
    padding: 16px 0 4px;
}
.nt-footer code {
    background: #F0F4F8 !important;
    color: #5A6378 !important;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: .72rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "is_busy" not in st.session_state:
    st.session_state.is_busy = False
if "customer_email" not in st.session_state:
    st.session_state.customer_email = ""


# ============================================================
# HERO HEADER
# ============================================================
st.markdown(
    """
<div class="nt-hero">
    <div class="nt-hero-text">
        <h1>📡 NileTel Support Assistant</h1>
        <p>Bilingual telecom support · Hybrid RAG · Gemini 2.5 Flash Lite</p>
    </div>
    <div class="nt-status-pill"><span class="nt-pulse"></span> Online</div>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# CUSTOMER EMAIL (for ticket notifications)
# ============================================================
st.session_state.customer_email = st.text_input(
    "📧 Email (optional — for ticket confirmation)",
    value=st.session_state.customer_email,
    placeholder="you@example.com",
    label_visibility="visible",
)


# ============================================================
# SAMPLE CHIPS
# ============================================================
st.markdown('<div class="nt-section-title">💡 Try a sample message</div>',
            unsafe_allow_html=True)

samples = [
    ("ℹ️ Info",      "ازاي أحل مشكلة 5G throttling؟"),
    ("📋 Procedure", "ايه إجراءات إلغاء العقد؟"),
    ("🎫 Complaint", "النت بطيء جداً عندي، اعمل تذكرة"),
    ("👋 Greeting",  "ازيك يا فندم"),
    ("🚫 Off-topic", "ايه آخر فيلم نزل؟"),
]
cols = st.columns(len(samples))
for i, (label, text) in enumerate(samples):
    with cols[i]:
        if st.button(label, use_container_width=True, key=f"sample-{i}",
                     help=text, disabled=st.session_state.is_busy):
            st.session_state.pending_input = text
            st.rerun()

# clear button
c1, c2 = st.columns([1, 4])
with c1:
    if st.button("🗑️ Clear", use_container_width=True, key="clear-btn",
                 disabled=st.session_state.is_busy):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()
with c2:
    st.markdown(
        f'<div style="padding-top:8px;color:#95A0B5;font-size:.78rem;">'
        f'session <code style="background:#F0F4F8;padding:2px 6px;'
        f'border-radius:6px;">{st.session_state.session_id[:8]}…</code></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# RENDERING HELPERS
# ============================================================
TICKET_RE = re.compile(
    r"\[\[TICKET\]\]id=([^|]+)\|created=([^|]+)\|email=([01])\|sheet=([^\[]*)\[\[/TICKET\]\]"
)


def _ticket_card_html(tid: str, created: str, email_sent: bool,
                      sheet_url: str | None) -> str:
    foot = []
    if email_sent:
        foot.append('<span>📧 Confirmation email sent</span>')
    foot.append('<span>📋 Support team notified</span>')
    if sheet_url:
        foot.append(f'<a href="{sheet_url}" target="_blank">🔗 View in Google Sheet</a>')
    foot_html = "".join(foot)
    return f"""
<div class="nt-ticket">
    <div class="nt-ticket-head">
        <div class="nt-ticket-title">🎫 Ticket created</div>
        <div class="nt-ticket-status">🟢 OPEN</div>
    </div>
    <div class="nt-ticket-grid">
        <div class="k">ID</div>      <div class="v">{tid}</div>
        <div class="k">Created</div> <div class="v">{created}</div>
    </div>
    <div class="nt-ticket-foot">{foot_html}</div>
</div>
"""


def _sources_html(sources) -> str:
    rows = []
    for src in sources:
        crumb = " > ".join(src.get("heading_path", [])) or "—"
        score = src.get("rerank_score", 0)
        rows.append(
            f'<div class="nt-src-card">'
            f'  <div>'
            f'    <div class="nt-src-name">📄 {src["source"]}</div>'
            f'    <div class="nt-src-crumb">{crumb}</div>'
            f'  </div>'
            f'  <div class="nt-src-score">{score:.3f}</div>'
            f'</div>'
        )
    return "".join(rows)


def _badges_html(intent: str, tier: int, latency_ms: float) -> str:
    return (
        '<div class="nt-badges">'
        f'<span class="nt-badge intent">🧭 {intent}</span>'
        f'<span class="nt-badge tier">⚙️ tier {tier}</span>'
        f'<span class="nt-badge latency">⚡ {latency_ms:.0f} ms</span>'
        '</div>'
    )


def _split_ticket_marker(text: str) -> tuple[str, dict | None]:
    """Pull the [[TICKET]]…[[/TICKET]] marker out of streamed text, if present."""
    m = TICKET_RE.search(text)
    if not m:
        return text, None
    clean = TICKET_RE.sub("", text).rstrip()
    ticket = {
        "id": m.group(1),
        "created_at": m.group(2),
        "email_sent": m.group(3) == "1",
        "sheet_url": m.group(4) or None,
    }
    return clean, ticket


def _render_message(role: str, content: str, sources=None,
                    badges_html: str | None = None, ticket: dict | None = None):
    avatar = "🧑" if role == "user" else "📡"
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)
        if sources:
            with st.expander(f"📄 Sources ({len(sources)})"):
                st.markdown(_sources_html(sources), unsafe_allow_html=True)
        if ticket:
            st.markdown(
                _ticket_card_html(
                    ticket["id"], ticket["created_at"],
                    ticket["email_sent"], ticket.get("sheet_url"),
                ),
                unsafe_allow_html=True,
            )
        if badges_html:
            st.markdown(badges_html, unsafe_allow_html=True)


# ============================================================
# CHAT HISTORY
# ============================================================
for msg in st.session_state.messages:
    _render_message(
        msg["role"], msg["content"],
        sources=msg.get("sources"),
        badges_html=msg.get("badges_html"),
        ticket=msg.get("ticket"),
    )


# ============================================================
# SEND HANDLER
# ============================================================
def _send(query: str) -> None:
    st.session_state.is_busy = True
    st.session_state.messages.append({"role": "user", "content": query})
    _render_message("user", query)

    with st.chat_message("assistant", avatar="📡"):
        try:
            payload = {
                "query": query,
                "session_id": st.session_state.session_id,
                "customer_email": st.session_state.customer_email or None,
            }
            if USE_STREAMING:
                placeholder = st.empty()
                placeholder.markdown(
                    '<div class="nt-typing"><span></span><span></span><span></span></div>',
                    unsafe_allow_html=True,
                )
                full_text = []
                with httpx.stream(
                    "POST", f"{API_URL}/stream",
                    json=payload, timeout=120.0,
                ) as response:
                    response.raise_for_status()
                    for chunk in response.iter_text():
                        if chunk:
                            full_text.append(chunk)
                            partial, _ = _split_ticket_marker("".join(full_text))
                            placeholder.markdown(partial + " ▌")
                final_raw = "".join(full_text)
                final, ticket = _split_ticket_marker(final_raw)
                placeholder.markdown(final)
                if ticket:
                    st.markdown(
                        _ticket_card_html(
                            ticket["id"], ticket["created_at"],
                            ticket["email_sent"], ticket.get("sheet_url"),
                        ),
                        unsafe_allow_html=True,
                    )
                st.session_state.messages.append({
                    "role": "assistant", "content": final,
                    "sources": [], "ticket": ticket,
                })
            else:
                placeholder = st.empty()
                placeholder.markdown(
                    '<div class="nt-typing"><span></span><span></span><span></span></div>',
                    unsafe_allow_html=True,
                )
                response = httpx.post(
                    f"{API_URL}/ask", json=payload, timeout=120.0,
                )
                response.raise_for_status()
                data = response.json()
                placeholder.markdown(data["answer"])
                if data.get("sources"):
                    with st.expander(f"📄 Sources ({len(data['sources'])})"):
                        st.markdown(_sources_html(data["sources"]),
                                    unsafe_allow_html=True)
                ticket = data.get("ticket")
                if ticket:
                    st.markdown(
                        _ticket_card_html(
                            ticket["id"], ticket["created_at"],
                            ticket.get("email_sent", False),
                            ticket.get("sheet_url"),
                        ),
                        unsafe_allow_html=True,
                    )
                badges = _badges_html(
                    data.get("intent", "-"),
                    data.get("routing_tier", 0),
                    data.get("latency_ms", 0),
                )
                st.markdown(badges, unsafe_allow_html=True)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": data["answer"],
                    "sources": data.get("sources", []),
                    "badges_html": badges,
                    "ticket": ticket,
                })
        except Exception as e:
            err = f"⚠️ Error contacting backend: {e}"
            st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err})
        finally:
            st.session_state.is_busy = False


# Sample-button pending input
if "pending_input" in st.session_state:
    pending = st.session_state.pending_input
    del st.session_state.pending_input
    _send(pending)

# Chat input box
if prompt := st.chat_input("اكتب رسالتك هنا..."):
    _send(prompt)


# ============================================================
# FOOTER
# ============================================================
st.markdown(
    f'<div class="nt-footer">📡 NileTel · API <code>{API_URL}</code> · '
    f'streaming <code>{USE_STREAMING}</code></div>',
    unsafe_allow_html=True,
)
