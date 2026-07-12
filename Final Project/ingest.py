"""
ingest.py — Data ingestion + chunking + embedding for IITB Insti-Assist.

Reads every .txt file in data/raw/, splits each into overlapping chunks
along paragraph/section boundaries, embeds the chunks with a
sentence-transformers model, and saves:
    data/index/faiss.index     -- the FAISS vector index
    data/index/chunks.json     -- chunk text + metadata (source file, chunk id)

Run:  python ingest.py
"""

import json
import os
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

RAW_DIR = Path("data/raw")
INDEX_DIR = Path("data/index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # 384-dim, fast, good enough for this scale
CHUNK_SIZE_CHARS = 1000                  # ~150-200 tokens
CHUNK_OVERLAP_CHARS = 200                # keeps context continuous across chunk boundaries


def load_documents():
    """Load every .txt file in data/raw/, returning (filename, full_text) pairs."""
    docs = []
    for path in sorted(RAW_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        docs.append((path.name, text))
    return docs


def split_into_sections(text: str):
    """Split on blank lines / all-caps headings so we don't cut mid-topic when possible."""
    # Split on lines that look like headings (ALL CAPS, numbered sections) as soft boundaries
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def chunk_text(text: str, chunk_size=CHUNK_SIZE_CHARS, overlap=CHUNK_OVERLAP_CHARS):
    """
    Chunk text into overlapping windows, preferring to break at paragraph
    boundaries so a chunk doesn't split a rule/date/definition in half.
    """
    paragraphs = split_into_sections(text)
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 1 <= chunk_size:
            current = f"{current}\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            # start new chunk; carry over overlap from the end of the previous chunk
            overlap_text = current[-overlap:] if current else ""
            current = f"{overlap_text}\n{para}".strip()
            # If a single paragraph itself exceeds chunk_size, hard-split it
            while len(current) > chunk_size:
                chunks.append(current[:chunk_size])
                current = current[chunk_size - overlap:]

    if current:
        chunks.append(current)

    return chunks


def build_index():
    print("Loading source documents from data/raw/ ...")
    docs = load_documents()
    if not docs:
        raise SystemExit("No .txt files found in data/raw/. Add source documents first.")

    all_chunks = []   # list of dicts: {id, text, source, chunk_index}
    for filename, text in docs:
        # First line of each file is "SOURCE: ..." — keep it out of the chunk body
        # but attach the human-readable source name to every chunk's metadata.
        lines = text.split("\n")
        source_label = filename
        for line in lines[:3]:
            if line.startswith("SOURCE:"):
                source_label = line.replace("SOURCE:", "").strip()
                break

        chunks = chunk_text(text)
        for i, c in enumerate(chunks):
            all_chunks.append({
                "id": f"{filename}::chunk{i}",
                "text": c,
                "source_file": filename,
                "source_label": source_label,
                "chunk_index": i,
            })
        print(f"  {filename}: {len(chunks)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"Loading embedding model '{EMBED_MODEL_NAME}' ...")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    texts = [c["text"] for c in all_chunks]
    print("Embedding chunks ...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # cosine similarity via inner product on normalized vectors
    index.add(embeddings.astype(np.float32))

    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    with open(INDEX_DIR / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\nSaved index to {INDEX_DIR/'faiss.index'}")
    print(f"Saved chunk metadata to {INDEX_DIR/'chunks.json'}")


if __name__ == "__main__":
    build_index()
