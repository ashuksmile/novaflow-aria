"""Lightweight local RAG engine using FastEmbed (ONNX) + NumPy cosine similarity."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List
import numpy as np
from fastembed import TextEmbedding

from app.config import KNOWLEDGE_BASE_DIR


@dataclass
class Chunk:
    chunk_id: str
    source: str
    heading: str
    content: str


class LocalRAGEngine:
    """In-memory vector index over Markdown documents in knowledge_base/."""

    def __init__(self, kb_dir: Path = KNOWLEDGE_BASE_DIR, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.kb_dir = kb_dir
        self.model_name = model_name
        self._embedder: TextEmbedding | None = None
        self.chunks: List[Chunk] = []
        self.embeddings: np.ndarray | None = None

    def initialize(self) -> int:
        """Load markdown files, chunk by section, and precompute normalized embeddings."""
        self._embedder = TextEmbedding(model_name=self.model_name)
        self.chunks = self._load_and_chunk_docs()
        if not self.chunks:
            self.embeddings = np.zeros((0, 384), dtype=np.float32)
            return 0

        texts = [f"{c.heading}\n{c.content}" for c in self.chunks]
        vecs = np.array(list(self._embedder.embed(texts)), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
        self.embeddings = vecs / norms
        return len(self.chunks)

    def _load_and_chunk_docs(self) -> List[Chunk]:
        chunks: List[Chunk] = []
        if not self.kb_dir.exists():
            return chunks

        for md_path in sorted(self.kb_dir.glob("*.md")):
            raw = md_path.read_text(encoding="utf-8").strip()
            sections = raw.split("\n## ")
            doc_title = md_path.stem.replace("_", " ").title()
            for idx, sec in enumerate(sections):
                sec = sec.strip()
                if not sec:
                    continue
                lines = sec.splitlines()
                if idx == 0:
                    heading = lines[0].lstrip("# ").strip()
                    body = "\n".join(lines[1:]).strip()
                else:
                    heading = f"{doc_title} — {lines[0].strip()}"
                    body = "\n".join(lines[1:]).strip()
                if not body:
                    continue
                chunks.append(
                    Chunk(
                        chunk_id=f"{md_path.stem}_{idx}",
                        source=md_path.name,
                        heading=heading,
                        content=body,
                    )
                )
        return chunks

    def search(self, query: str, top_k: int = 3) -> List[dict]:
        """Return top-k most relevant knowledge chunks with similarity scores."""
        if self._embedder is None or self.embeddings is None or len(self.chunks) == 0:
            return []

        q_vec = np.array(list(self._embedder.embed([query]))[0], dtype=np.float32)
        q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-10)
        scores = self.embeddings @ q_vec
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for i in top_indices:
            score = float(scores[i])
            if score < 0.25:
                continue
            c = self.chunks[int(i)]
            results.append(
                {
                    "chunk_id": c.chunk_id,
                    "source": c.source,
                    "heading": c.heading,
                    "content": c.content,
                    "score": round(score, 3),
                }
            )
        return results
