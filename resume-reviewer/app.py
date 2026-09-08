"""Streamlit front end for the resume reviewer.

Deliberately thin. Everything interesting lives in reviewer.py so it can be
run and debugged without a browser; this file only does upload, layout and
error states.

Run locally:   streamlit run app.py
"""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from reviewer import review_resume, Review

load_dotenv()

st.set_page_config(page_title="Resume Reviewer", page_icon="📄", layout="centered")


# ── PDF text extraction ───────────────────────────────────────────────
def extract_text(uploaded) -> str:
    """Pull plain text out of an uploaded PDF.

    pypdf returns an empty string for scanned/image PDFs rather than raising,
    which is the single most common failure here — a candidate exports their
    resume from a design tool as an image and gets a blank review. So the
    emptiness is checked by the caller and surfaced, not swallowed.
    """
    from pypdf import PdfReader

    reader = PdfReader(uploaded)
    parts = [(page.extract_text() or "") for page in reader.pages]
    return "\n".join(parts).strip()


# ── UI ────────────────────────────────────────────────────────────────
st.title("📄 Resume Reviewer")
st.caption(
    "Upload a resume, get structured feedback. Optionally paste a job ad to see "
    "how much of its vocabulary your resume actually covers."
)

if not os.environ.get("API_KEY"):
    st.error("No API_KEY found. Copy .env.example to .env and add your key.")
    st.stop()

uploaded = st.file_uploader("Resume (PDF)", type=["pdf"])
job_ad = st.text_area(
    "Target job ad (optional)",
    placeholder="Paste the job description here to get keyword coverage…",
    height=140,
)

if uploaded and st.button("Review it", type="primary"):
    with st.spinner("Reading the PDF…"):
        try:
            text = extract_text(uploaded)
        except Exception as e:                      # noqa: BLE001 - shown to user
            st.error(f"Could not read that PDF: {e}")
            st.stop()

    # The failure mode worth handling explicitly. An image-only PDF parses
    # "successfully" into nothing, and sending nothing to the model wastes a
    # call and returns a confident review of an empty document.
    if len(text) < 200:
        st.error(
            "Almost no text came out of that PDF — it's probably a scan or an "
            "image export. Save it as a text-based PDF and try again."
        )
        st.stop()

    with st.spinner("Reviewing…"):
        try:
            review: Review = review_resume(text, job_ad)
        except Exception as e:                      # noqa: BLE001
            st.error(f"The review call failed: {e}")
            st.stop()

    # ── results ──────────────────────────────────────────────────────
    st.divider()

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Overall", f"{review.overall_score}/100")
        if review.keyword_coverage is not None:
            st.metric("Keyword coverage", f"{review.keyword_coverage:.0%}")
    with col2:
        st.write(review.summary)

    if review.strengths:
        st.subheader("Strengths")
        for s in review.strengths:
            st.markdown(f"- {s}")

    if review.findings:
        st.subheader("What to fix")
        order = {"high": 0, "medium": 1, "low": 2}
        for f in sorted(review.findings, key=lambda x: order.get(x.severity, 3)):
            icon = {"high": "🔴", "medium": "🟠"}.get(f.severity, "🟡")
            with st.expander(f"{icon}  {f.section} — {f.problem}", expanded=f.severity == "high"):
                st.markdown(f"**Fix:** {f.fix}")

    if review.missing_keywords:
        st.subheader("In the job ad, not in your resume")
        st.caption(
            "Not an ATS score — just vocabulary from the ad that doesn't appear "
            "in your resume. Only add terms that are genuinely true of you."
        )
        st.write(", ".join(f"`{k}`" for k in review.missing_keywords))

    with st.expander("Raw JSON"):
        st.json(review.to_dict())
