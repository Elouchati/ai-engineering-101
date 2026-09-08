"""Streamlit front end for the RAG chatbot.

Thin on purpose — everything interesting is in rag.py, which has no Streamlit
import so you can drive it from a plain REPL.

    python ingest.py            # build the index once
    streamlit run app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag import Store, ask

load_dotenv()

STORE_PATH = Path(__file__).parent / "store.pkl"

st.set_page_config(page_title="Ask the handbook", page_icon="📚", layout="centered")

st.title("📚 Ask the handbook")
st.caption(
    "Answers come only from the indexed documents. If something isn't in them, "
    "it says so instead of guessing."
)

if not STORE_PATH.exists():
    st.error("No index found. Run `python ingest.py` first.")
    st.stop()
if not os.environ.get("API_KEY"):
    st.error("No API_KEY found. Copy .env.example to .env and add your key.")
    st.stop()

store = Store.load(STORE_PATH)
st.caption(f"{len(store.chunks)} chunks indexed from "
           f"{len({c.source for c in store.chunks})} document(s).")

question = st.text_input(
    "Question",
    placeholder="e.g. how long do I have to return something?",
)

# Showing k in the UI is a small thing that pays off: it makes the single most
# common tuning knob visible instead of buried in a constant.
k = st.slider("Passages to retrieve (k)", 1, 8, 4)

if question and st.button("Ask", type="primary"):
    with st.spinner("Retrieving and answering…"):
        try:
            answer = ask(store, question, k=k)
        except Exception as e:                       # noqa: BLE001 — shown to user
            st.error(f"The call failed: {e}")
            st.stop()

    st.divider()

    # An honest refusal is a success, not an error. Styling it as a warning
    # rather than a failure is the difference between users trusting the thing
    # and users assuming it's broken.
    if answer.grounded:
        st.markdown(f"### {answer.text}")
    else:
        st.warning(answer.text)
        st.caption(
            "This is the system working correctly — the answer genuinely isn't "
            "in the indexed documents."
        )

    st.subheader("Retrieved passages")
    st.caption("What the answer was actually allowed to read.")
    for i, (score, chunk) in enumerate(answer.hits, start=1):
        with st.expander(f"[{i}]  {chunk.source}  ·  similarity {score:.3f}", expanded=i == 1):
            # st.text, not st.write: a chunk often starts with the "##" heading
            # it was split on, and st.write would render that as a title. Plain
            # text is also more honest — this is the exact string the model saw.
            st.text(chunk.text)
