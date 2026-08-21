"""
Core RAG (Retrieval-Augmented Generation) engine.
This file has 3 jobs:
1. Read your documents and split them into small chunks
2. Turn chunks into vectors and store them in FAISS
3. Given a question, find the most relevant chunks
"""

import os
import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


def load_documents(folder_path: str) -> list[dict]:
    """Read every .pdf and .txt file in folder_path and return raw text per file."""
    documents = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)

        if filename.lower().endswith(".pdf"):
            reader = PdfReader(filepath)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            documents.append({"source": filename, "text": text})

        elif filename.lower().endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            documents.append({"source": filename, "text": text})

    return documents


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Split long text into overlapping chunks so context isn't cut off mid-idea."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


class RagEngine:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        # This model runs locally on your laptop — free, fast, no API key needed
        self.embedder = SentenceTransformer(embedding_model)
        self.index = None
        self.chunk_records = []  # keeps text + source for each vector

    def build_index(self, folder_path: str):
        """Load docs, chunk them, embed them, and build the FAISS index."""
        documents = load_documents(folder_path)
        all_chunks = []

        for doc in documents:
            for chunk in chunk_text(doc["text"]):
                all_chunks.append({"text": chunk, "source": doc["source"]})

        if not all_chunks:
            raise ValueError(f"No readable text found in {folder_path}")

        texts = [c["text"] for c in all_chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        self.chunk_records = all_chunks

        return len(all_chunks)

    def retrieve(self, question: str, top_k: int = 4) -> list[dict]:
        """Return the top_k most relevant chunks for a given question."""
        if self.index is None:
            raise ValueError("Index not built yet. Call build_index() first.")

        query_vector = self.embedder.encode([question]).astype("float32")
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for idx in indices[0]:
            if idx != -1:
                results.append(self.chunk_records[idx])
        return results
