"""
rag.py — Retrieval + grounded generation for IITB Insti-Assist.

Loads the FAISS index built by ingest.py, retrieves the top-k most relevant
chunks for a user question, and calls the Groq API (OpenAI-SDK-compatible,
free tier available at https://console.groq.com/keys) with those chunks
injected into the prompt. The model is instructed to answer ONLY from the
provided context and to say "I don't know" if the answer isn't supported.
"""

import json
import os
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

INDEX_DIR = Path("data/index")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 4
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
# If the best chunk's cosine similarity is below this, we treat the question
# as out-of-scope and refuse rather than let the LLM guess from weak context.
MIN_SIMILARITY_THRESHOLD = 0.30

SYSTEM_PROMPT = """You are IITB Insti-Assist, an academic assistant for IIT Bombay students.

Rules you MUST follow:
1. Answer ONLY using the information in the "CONTEXT" section below. Do not use outside knowledge about IIT Bombay or any other institute.
2. If the context does not contain enough information to answer the question, respond exactly with: "I don't know — I couldn't find this in my current knowledge base." Do not guess or make up dates, numbers, or rules.
3. When you do answer, be concise and specific (dates, credit numbers, grade names, etc. should be quoted exactly as given in the context).
4. Do not mention "the context" or "the documents" in your answer — answer as if you simply know this information, but never invent anything beyond what's given.
"""


class InstiAssistRAG:
    def __init__(self, groq_api_key: str | None = None):
        self.index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
        with open(INDEX_DIR / "chunks.json", encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        self.client = OpenAI(
            api_key=groq_api_key or os.environ.get("GROQ_API_KEY"),
            base_url=GROQ_BASE_URL,
        )

    def retrieve(self, query: str, k: int = TOP_K):
        query_vec = self.embed_model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        scores, indices = self.index.search(query_vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append({**chunk, "score": float(score)})
        return results

    def answer(self, query: str, k: int = TOP_K, model: str = DEFAULT_MODEL):
        retrieved = self.retrieve(query, k=k)

        if not retrieved or retrieved[0]["score"] < MIN_SIMILARITY_THRESHOLD:
            return {
                "answer": "I don't know — I couldn't find this in my current knowledge base.",
                "sources": [],
                "grounded": False,
            }

        context_block = "\n\n---\n\n".join(
            f"[Source: {c['source_label']}]\n{c['text']}" for c in retrieved
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"CONTEXT:\n{context_block}\n\nQUESTION: {query}"},
        ]

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=500,
        )
        answer_text = response.choices[0].message.content.strip()

        is_idk = answer_text.lower().startswith("i don't know")

        return {
            "answer": answer_text,
            "sources": [] if is_idk else [
                {"label": c["source_label"], "file": c["source_file"], "score": round(c["score"], 3)}
                for c in retrieved
            ],
            "grounded": not is_idk,
        }
