"""A retrieval eval set. Thirty lines, and it is the only reason any claim in
episode 7 is checkable.

RAG has a measurement problem. You change the chunk size, you ask it one
question, the answer looks better, you ship it. That is not evidence — it is a
sample of one, chosen after the fact. The fix is boring and it works: write down
a dozen questions with the section that ought to answer each, then score
retrieval instead of arguing about it.

    python eval.py

Two numbers come out:

    hit@k    was the right section anywhere in the k passages we retrieved?
             This is the one that matters. The model can read all k.
    top-1    was the right section the FIRST passage?
             A stricter proxy for whether the ranking is actually good.

Questions are written the way a customer would type them, not the way the
document is worded. An eval set that reuses the document's own phrasing tests
string matching and flatters you.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from rag import Store, build_store

HERE = Path(__file__).parent

# (question, the section heading that should answer it)
EVAL: list[tuple[str, str]] = [
    ("how long do I have to send something back?",        "Returns and refunds"),
    ("when will I actually get my money?",                "Returns and refunds"),
    ("can I send back something I already opened?",       "Returns and refunds"),
    ("is delivery free?",                                 "Delivery"),
    ("what's the quickest you can get it to me?",         "Delivery"),
    ("do you send to Jersey?",                            "Delivery"),
    ("how do I change the email on my account?",          "Accounts"),
    ("I'm locked out and can't sign in",                  "Accounts"),
    ("when does the shop shut?",                          "Warehouse and collection"),
    ("can I come and pick my order up myself?",           "Warehouse and collection"),
    ("my parcel turned up smashed",                       "Damaged or missing items"),
    ("tracking says delivered but there's nothing here",  "Damaged or missing items"),
]

_HEADING = re.compile(r"^#{2,6}\s+(.+)$", re.MULTILINE)


def sections(doc: str) -> list[tuple[str, str]]:
    """[(heading, flattened section text)], in document order."""
    out, marks = [], list(_HEADING.finditer(doc))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(doc)
        out.append((m.group(1).strip(), " ".join(doc[m.start():end].split())))
    return out


def section_of(chunk_text: str, doc: str) -> str:
    """Which section does this chunk come from?

    Found by locating the chunk inside each section's text, not by reading a
    leading heading off the chunk — continuation chunks have no heading, and
    those are exactly the ones that go wrong. A chunk that straddles a boundary
    is credited to the section it starts in.
    """
    probe = " ".join(chunk_text.split()[:8])
    for head, body in sections(doc):
        if probe in body:
            return head
    return "?"


def score(store: Store, doc: str, k: int = 4, verbose: bool = False) -> tuple[float, float]:
    hits = top1 = 0
    for question, want in EVAL:
        got = [section_of(c.text, doc) for _, c in store.search(question, k=k)]
        ok, first = want in got, got[:1] == [want]
        hits += ok
        top1 += first
        if verbose:
            flag = "ok  " if ok else "MISS"
            print(f"  {flag} {question:52} -> {got[0]}")
    n = len(EVAL)
    return hits / n, top1 / n


if __name__ == "__main__":
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    doc = (HERE / "docs" / "handbook.md").read_text(encoding="utf-8")
    store = build_store(HERE / "docs")
    hit, t1 = score(store, doc, k=k, verbose=True)
    print(f"\n  {len(EVAL)} questions, k={k}")
    print(f"  hit@{k}  {hit:.0%}")
    print(f"  top-1  {t1:.0%}")
