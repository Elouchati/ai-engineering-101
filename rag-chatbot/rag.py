"""RAG over your own documents — the whole pipeline, in one readable file.

Episode 6 of AI Engineering 101. Episode 5 explained the idea; this is it built.

    chunk  →  embed  →  store  →  retrieve  →  prompt  →  answer

Two deliberate choices, both explained in the video:

**The vector store is 30 lines of numpy, not a database.** Cosine similarity
over a matrix is all a vector DB does at this scale — writing it out once shows
there is no magic in the box. Past roughly 50k chunks you want a real one
(Chroma, pgvector, Pinecone); `search()` is the only function you would replace,
and its signature is the same in all of them.

**Embeddings default to a local model.** all-MiniLM-L6-v2 is ~90MB, runs on a
laptop CPU, costs nothing, and is good enough for most document search. Set
EMBED_API=1 to use an API instead. Being able to build this with no key at all
is the point.
"""

from __future__ import annotations

import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ── 1 · chunking ──────────────────────────────────────────────────────
# The single most consequential setting in a RAG system, and the one people
# leave at whatever the tutorial said. Too big and you retrieve a page to answer
# a sentence, burning context and burying the fact. Too small and you cut a
# thought in half, so neither piece means anything on its own.

CHUNK_WORDS = 120      # a paragraph or two
OVERLAP_WORDS = 25     # carries a sentence across the seam


@dataclass
class Chunk:
    text: str
    source: str        # filename, so answers can cite where they came from
    index: int         # position within that file


_HEADING = re.compile(r"^#{1,6}\s+", re.MULTILINE)


def chunk_text(text: str, source: str) -> list[Chunk]:
    """Split on headings first, then paragraphs, then pack with overlap.

    The heading split is not decoration — it was a bug fix. The first version
    of this function split on blank lines only, so a chunk could start in
    "Returns" and end in "Delivery". Asking "is delivery free" then returned
    the returns section as the top hit, because it happened to contain the word
    delivery twice. The right answer came second.

    That is the single most common RAG failure, and it is not a model problem:
    a chunk that spans two topics is about neither, and no amount of prompt
    tuning recovers it. Where a document tells you its own structure — headings,
    sections, articles — split there before you count words.
    """
    # Sections first: a heading is a hard boundary, never crossed.
    sections: list[str] = []
    last = 0
    for m in _HEADING.finditer(text):
        if m.start() > last:
            sections.append(text[last:m.start()])
        last = m.start()
    sections.append(text[last:])

    chunks: list[Chunk] = []
    for section in sections:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section) if p.strip()]
        if not paragraphs:
            continue
        buf: list[str] = []

        def flush():
            if buf:
                chunks.append(Chunk(" ".join(buf), source, len(chunks)))

        for para in paragraphs:
            words = para.split()
            if len(buf) + len(words) <= CHUNK_WORDS:
                buf.extend(words)
                continue
            flush()
            # Overlap carries a sentence across the seam — but only within a
            # section, never across a heading.
            buf = buf[-OVERLAP_WORDS:] if buf else []
            buf.extend(words)
            while len(buf) > CHUNK_WORDS:
                chunks.append(Chunk(" ".join(buf[:CHUNK_WORDS]), source, len(chunks)))
                buf = buf[CHUNK_WORDS - OVERLAP_WORDS:]
        flush()
    return chunks


# ── 2 · embedding ─────────────────────────────────────────────────────
# The rule that matters: chunks and questions must be embedded by the SAME
# model. Mixing them puts the query in a different space from the documents and
# retrieval returns noise — with no error to tell you.

_local_model = None


def _local_encode(texts: list[str]) -> np.ndarray:
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        _local_model = SentenceTransformer(
            os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        )
    return np.asarray(_local_model.encode(texts, normalize_embeddings=True))


def _api_encode(texts: list[str]) -> np.ndarray:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["API_KEY"],
        base_url=os.environ.get("API_BASE") or None,
    )
    resp = client.embeddings.create(
        model=os.environ.get("EMBED_MODEL", "text-embedding-3-small"),
        input=texts,
    )
    vecs = np.asarray([d.embedding for d in resp.data], dtype=np.float32)
    # Normalising here means cosine similarity is just a dot product later.
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def embed(texts: list[str]) -> np.ndarray:
    return _api_encode(texts) if os.environ.get("EMBED_API") else _local_encode(texts)


# ── 3 · the store ─────────────────────────────────────────────────────

@dataclass
class Store:
    chunks: list[Chunk]
    vectors: np.ndarray           # (n, dim), rows already unit length

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: str | Path) -> "Store":
        return pickle.loads(Path(path).read_bytes())

    def search(self, query: str, k: int = 4) -> list[tuple[float, Chunk]]:
        """Top-k chunks by cosine similarity.

        Because every vector is unit length, cosine similarity is a dot
        product, and searching the whole store is one matrix multiply. That is
        genuinely all a vector database does — it just adds an index so this
        stays fast at millions of rows instead of thousands.
        """
        q = embed([query])[0]
        scores = self.vectors @ q
        top = np.argsort(-scores)[:k]
        return [(float(scores[i]), self.chunks[i]) for i in top]


def build_store(doc_dir: str | Path) -> Store:
    """Read every .txt and .md in a directory, chunk it, embed it, store it."""
    chunks: list[Chunk] = []
    for path in sorted(Path(doc_dir).glob("*")):
        if path.suffix.lower() not in {".txt", ".md"}:
            continue
        chunks.extend(chunk_text(path.read_text(encoding="utf-8"), path.name))
    if not chunks:
        raise ValueError(f"no .txt or .md files found in {doc_dir}")
    vectors = embed([c.text for c in chunks])
    return Store(chunks, vectors)


# ── 4 · the prompt ────────────────────────────────────────────────────
# Two instructions carry the whole thing. "Only the context" is what stops the
# model answering from training data and quietly passing it off as your
# documentation. The "say you don't know" line is what makes the failure
# visible instead of silent — and it is the line people leave out.

SYSTEM = """You answer questions using only the context provided.

Rules:
- Use only the numbered context passages. Do not use outside knowledge.
- Cite the passages you used, like [1] or [2, 3].
- If the context does not contain the answer, say exactly:
  "That isn't in the documents I have." Then stop. Do not guess.
- Be brief. Two or three sentences unless asked for more."""


def build_prompt(question: str, hits: list[tuple[float, Chunk]]) -> list[dict]:
    context = "\n\n".join(
        f"[{i + 1}] ({c.source}) {c.text}" for i, (_, c) in enumerate(hits)
    )
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]


# ── 5 · the answer ────────────────────────────────────────────────────

@dataclass
class Answer:
    text: str
    hits: list[tuple[float, Chunk]]

    @property
    def grounded(self) -> bool:
        """Did the model admit it could not answer? Worth surfacing in the UI —
        an honest refusal is a success, and it should not look like a failure."""
        return "isn't in the documents" not in self.text


def ask(store: Store, question: str, k: int = 4) -> Answer:
    from openai import OpenAI

    hits = store.search(question, k=k)
    client = OpenAI(
        api_key=os.environ["API_KEY"],
        base_url=os.environ.get("API_BASE") or None,
    )
    resp = client.chat.completions.create(
        model=os.environ.get("MODEL", "gpt-4o-mini"),
        messages=build_prompt(question, hits),
        temperature=0.1,      # low: we want it reading, not composing
        max_tokens=400,
    )
    if resp.usage:
        print(f"[tokens] in={resp.usage.prompt_tokens} out={resp.usage.completion_tokens}")
    return Answer(resp.choices[0].message.content.strip(), hits)
