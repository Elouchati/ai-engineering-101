"""Build the index. Run once, and again whenever the documents change.

That re-indexing story is the whole argument for retrieval over fine-tuning:
this takes seconds, a training run does not.

    python ingest.py [doc_dir]
"""
import sys
from pathlib import Path

from rag import build_store

doc_dir = sys.argv[1] if len(sys.argv) > 1 else "docs"
store = build_store(doc_dir)
out = Path(__file__).parent / "store.pkl"
store.save(out)

sources = sorted({c.source for c in store.chunks})
print(f"indexed {len(store.chunks)} chunks from {len(sources)} file(s): {', '.join(sources)}")
print(f"vectors {store.vectors.shape} -> {out.name}")
