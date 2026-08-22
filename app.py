import os
import tempfile
import streamlit as st
from rag_engine import RagEngine
from answer_generator import build_client, generate_answer

st.set_page_config(page_title="Bootcamp RAG Chatbot", page_icon="🤖")
st.title("🤖 Ask My Documents")
st.caption("A RAG chatbot that answers from the documents you provide.")

# ---- Load settings from Streamlit Secrets (set once by the app owner) ----
# Falls back to manual entry if secrets aren't configured, so the app still
# works for anyone testing locally without secrets set up.
api_key = st.secrets.get("GROQ_API_KEY", "")
base_url = st.secrets.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
model_name = st.secrets.get("GROQ_MODEL", "openai/gpt-oss-20b")
docs_folder = "./documents"

with st.sidebar:
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

    # Advanced settings only shown if secrets aren't already configured,
    # or if the user wants to override them manually.
    with st.expander("Advanced settings (optional)"):
        api_key = st.text_input("LLM API key", value=api_key, type="password")
        base_url = st.text_input("Base URL", value=base_url)
        model_name = st.text_input("Model name", value=model_name)

    if st.button("🔄 Reset conversation"):
        st.session_state.messages = []
        st.rerun()


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

# ---- Main chat ----
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask something about your documents...")

if question:
    if not api_key:
        st.error("No API key configured. Ask the app owner to set it up.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            chunks = st.session_state.engine.retrieve(question)
            client = build_client(api_key, base_url or None)
            answer = generate_answer(
                client, model_name, question, chunks,
                chat_history=st.session_state.messages[:-1],
            )
            st.markdown(answer)

            with st.expander("📄 Sources used"):
                for c in chunks:
                    st.markdown(f"**{c['source']}**: {c['text'][:200]}...")

    st.session_state.messages.append({"role": "assistant", "content": answer})
