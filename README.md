# RAG Chatbot — Run Instructions

## 1. Install dependencies
```
pip install -r requirements.txt
```
(First run will also download a small local embedding model automatically — needs internet once.)

## 2. Add your documents
Create a folder called `documents` in this same directory and put your PDFs/.txt files inside it.

## 3. Get a free API key (pick one)
- **Groq (free, fast, recommended for a demo)**: https://console.groq.com → API Keys → Create key.
  In the app sidebar, set Base URL to `https://api.groq.com/openai/v1` and Model name to `llama-3.1-8b-instant`.
- **OpenAI**: https://platform.openai.com/api-keys → Create key. Leave Base URL empty, use model `gpt-4o-mini`.

## 4. Run the app
```
streamlit run app.py
```
It opens in your browser automatically.

## 5. In the app
1. Paste your API key and model name in the sidebar
2. Click "Build knowledge base"
3. Ask a question in the chat box at the bottom
4. Expand "Sources used" to show the judges *where* the answer came from — this is the best part to highlight

## Explaining it on stage (30-second version)
"I built a RAG — Retrieval-Augmented Generation — chatbot. It reads our documents, breaks them into chunks, and stores them as vectors in a FAISS database. When someone asks a question, it finds the most relevant chunks and gives them to an LLM, which answers using only that information — so it doesn't hallucinate and it's grounded in real, cited sources."
