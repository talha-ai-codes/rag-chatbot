import os
import tempfile
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo
import streamlit as st
from PIL import Image
from rag_engine import RagEngine
from answer_generator import build_client, generate_answer_stream

st.set_page_config(page_title="DocuMind", page_icon=Image.open("logo_small.png"), layout="centered")

# ---- Custom styling: dark theme + centered greeting, Claude-style ----
st.markdown("""
<style>
    .stApp { background-color: #0d0d0f; }
    .greeting-wrap {
        text-align: center;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
    }
    .greeting-emoji { font-size: 2.2rem; }
    .greeting-text {
        font-family: Georgia, 'Times New Roman', serif;
        font-size: 2.4rem;
        color: #f2ede7;
        display: inline;
        margin-left: 0.4rem;
    }
    .greeting-sub {
        text-align: center;
        color: #8a8a8f;
        font-size: 0.95rem;
        margin-bottom: 2rem;
    }
    div[data-testid="stChatInput"] { border-radius: 16px; }
    .brand-wrap {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 0.4rem;
    }
    .brand-text {
        font-family: Georgia, 'Times New Roman', serif;
        font-size: 1.3rem;
        color: #f2ede7;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


def time_based_greeting(name: str) -> tuple[str, str]:
    hour = datetime.now(ZoneInfo("Asia/Karachi")).hour
    if 5 <= hour < 12:
        period, emoji = "Morning", "☀️"
    elif 12 <= hour < 17:
        period, emoji = "Afternoon", "🌤️"
    elif 17 <= hour < 21:
        period, emoji = "Evening", "🌇"
    else:  # 9 PM to 5 AM
        period, emoji = "Night", "🌙"
    return f"{period}, {name}", emoji


USER_NAME = st.secrets.get("DISPLAY_NAME", "DocuMind")

# ============================================================
# Multi-chat session management (ChatGPT/Claude-style history)
# ============================================================
if "chats" not in st.session_state:
    first_id = str(uuid.uuid4())
    st.session_state.chats = {
        first_id: {"title": "New chat", "messages": []}
    }
    st.session_state.current_chat_id = first_id


def new_chat():
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {"title": "New chat", "messages": []}
    st.session_state.current_chat_id = chat_id


def switch_chat(chat_id: str):
    st.session_state.current_chat_id = chat_id


current_chat = st.session_state.chats[st.session_state.current_chat_id]
messages = current_chat["messages"]

# ---- Load settings from Streamlit Secrets (set once by the app owner) ----
api_key = st.secrets.get("GROQ_API_KEY", "")
base_url = st.secrets.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
model_name = st.secrets.get("GROQ_MODEL", "openai/gpt-oss-20b")
docs_folder = "./documents"

with st.sidebar:
    col_logo, col_name = st.columns([0.22, 0.78])
    with col_logo:
        st.image("logo_small.png", width=36)
    with col_name:
        st.markdown('<div class="brand-text">DocuMind</div>', unsafe_allow_html=True)

    if st.button("➕ New chat", use_container_width=True):
        new_chat()
        st.rerun()

    st.divider()
    st.caption("CHATS")
    for cid, chat in reversed(list(st.session_state.chats.items())):
        is_active = cid == st.session_state.current_chat_id
        label = ("💬 " if is_active else "🗨️ ") + chat["title"]
        if st.button(label, key=f"chat_{cid}", use_container_width=True):
            switch_chat(cid)
            st.rerun()

    st.divider()
    st.header("Documents")

    doc_source = st.radio(
        "Which documents should I use?",
        ["Shared class documents", "Upload my own documents"],
    )

    uploaded_files = None
    if doc_source == "Upload my own documents":
        uploaded_files = st.file_uploader(
            "Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True
        )

    with st.expander("Advanced settings (optional)"):
        api_key = st.text_input("LLM API key", value=api_key, type="password")
        base_url = st.text_input("Base URL", value=base_url)
        model_name = st.text_input("Model name", value=model_name)


@st.cache_resource(show_spinner="Loading class documents...")
def load_shared_engine():
    engine = RagEngine()
    engine.build_index(docs_folder)
    return engine


# ---- Build / load the knowledge base ----
if doc_source == "Shared class documents":
    if not api_key:
        st.warning("The app owner hasn't configured an API key yet. Ask them to set it up in Advanced settings.")
        st.stop()
    st.session_state.engine = load_shared_engine()
else:
    if uploaded_files:
        if "own_engine" not in st.session_state or st.session_state.get("own_files") != [f.name for f in uploaded_files]:
            with st.spinner("Reading your documents..."):
                engine = RagEngine()
                temp_dir = tempfile.mkdtemp()
                for f in uploaded_files:
                    with open(os.path.join(temp_dir, f.name), "wb") as out:
                        out.write(f.getbuffer())
                engine.build_index(temp_dir)
                st.session_state.own_engine = engine
                st.session_state.own_files = [f.name for f in uploaded_files]
        st.session_state.engine = st.session_state.own_engine
    else:
        st.info("Upload at least one file in the sidebar to start chatting.")
        st.stop()

# ---- Greeting (only shown on an empty chat) ----
if len(messages) == 0:
    greeting_text, greeting_emoji = time_based_greeting(USER_NAME)
    st.markdown(
        f'<div class="greeting-wrap"><span class="greeting-emoji">{greeting_emoji}</span>'
        f'<span class="greeting-text">{greeting_text}</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="greeting-sub">Ask me anything about your documents \u2014 or general AI/ML/programming topics.</div>',
        unsafe_allow_html=True,
    )


def render_feedback(index: int):
    """Show thumbs up/down under an assistant message and store the choice."""
    msg = messages[index]
    current = msg.get("feedback")
    col1, col2, col_spacer = st.columns([0.06, 0.06, 0.88])

    with col1:
        if st.button("👍" if current != "up" else "✅", key=f"up_{st.session_state.current_chat_id}_{index}"):
            messages[index]["feedback"] = "up"
            st.rerun()
    with col2:
        if st.button("👎" if current != "down" else "❌", key=f"down_{st.session_state.current_chat_id}_{index}"):
            messages[index]["feedback"] = "down"
            st.rerun()


for i, msg in enumerate(messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("sources"):
                with st.expander("📄 Sources used"):
                    for c in msg["sources"]:
                        st.markdown(f"**{c['source']}**: {c['text'][:200]}...")
            render_feedback(i)

question = st.chat_input("Ask something about your documents...")

if question:
    if not api_key:
        st.error("No API key configured. Ask the app owner to set it up.")
        st.stop()

    # Name the chat after the first question, like ChatGPT/Claude do
    if current_chat["title"] == "New chat":
        current_chat["title"] = (question[:32] + "…") if len(question) > 32 else question

    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching your documents..."):
            chunks = st.session_state.engine.retrieve(question)
            client = build_client(api_key, base_url or None)

        answer = st.write_stream(
            generate_answer_stream(
                client, model_name, question, chunks,
                chat_history=messages[:-1],
            )
        )

        if chunks:
            with st.expander("📄 Sources used"):
                for c in chunks:
                    st.markdown(f"**{c['source']}**: {c['text'][:200]}...")

        new_index = len(messages)
        messages.append({
            "role": "assistant",
            "content": answer,
            "sources": chunks,
            "feedback": None,
        })
        render_feedback(new_index)

    st.rerun()  # refresh so the sidebar shows the new chat title immediately
