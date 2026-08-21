import os
import streamlit as st
from rag_engine import RagEngine
from answer_generator import build_client, generate_answer

st.set_page_config(page_title="Bootcamp RAG Chatbot", page_icon="🤖")
st.title("🤖 Ask My Documents")
st.caption("A RAG chatbot that answers only from the documents you provide.")

# ---- Sidebar: setup ----
with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("LLM API key", type="password")
    base_url = st.text_input(
        "Base URL (leave empty for OpenAI, or use https://api.groq.com/openai/v1 for Groq)",
        value="",
    )
    model_name = st.text_input("Model name", value="gpt-4o-mini")
    docs_folder = st.text_input("Folder path with your PDFs/txt files", value="./documents")

    if st.button("Build knowledge base"):
        with st.spinner("Reading documents and building the search index..."):
            engine = RagEngine()
            num_chunks = engine.build_index(docs_folder)
            st.session_state.engine = engine
        st.success(f"Indexed {num_chunks} chunks. Ready to chat!")

# ---- Main chat ----
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask something about your documents...")

if question:
    if "engine" not in st.session_state:
        st.error("Build the knowledge base first using the sidebar.")
    elif not api_key:
        st.error("Enter your API key in the sidebar.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chunks = st.session_state.engine.retrieve(question)
                client = build_client(api_key, base_url or None)
                answer = generate_answer(client, model_name, question, chunks)
                st.markdown(answer)

                with st.expander("📄 Sources used"):
                    for c in chunks:
                        st.markdown(f"**{c['source']}**: {c['text'][:200]}...")

        st.session_state.messages.append({"role": "assistant", "content": answer})
