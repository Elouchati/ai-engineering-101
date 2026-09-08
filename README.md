# AI Engineering 101

Code and resources for the [**AI Engineering 101**](https://www.youtube.com/@techinonemin1) playlist — a ten-part path from prompting to agents, with no hype and no PhD required.

📺 **Watch the playlist:** https://www.youtube.com/@techinonemin1

---

## What's here

| | |
|---|---|
| [`resume-reviewer/`](resume-reviewer) | **Episode 4** — a working AI app that reviews a resume and returns structured feedback. ~150 lines. |
| [`rag-chatbot/`](rag-chatbot) | **Episodes 6 & 7** — RAG over your own documents, plus the eval set every number in episode 7 came out of. |
| [`resources/`](resources) | **Episode 2** — the prompting patterns cheat sheet (PDF, free, no email required). |

---

## The playlist

**Foundations**
1. What Is an AI Engineer? (It's Not Prompt Engineering)
2. 5 Prompting Patterns That Actually Work — [cheat sheet](resources/prompting-patterns-cheatsheet.pdf)
3. Tokens, Context Windows & Cost

**Build**
4. Build an AI Resume Reviewer — [code](resume-reviewer)

**Retrieval**
5. What Is RAG? Explained With a Library Analogy
6. I Built a RAG Chatbot That Admits When It Doesn't Know — [code](rag-chatbot)
7. 5 RAG Mistakes That Make Your Chatbot Useless — [eval set](rag-chatbot/eval.py)

**Agents**
8. What Is an AI Agent? (Cutting Through the Hype)
9. Build an AI Agent That Uses Real Tools
10. Why AI Agents Fail in Production

---

## Quick start — the resume reviewer

```bash
cd resume-reviewer
pip install -r requirements.txt
cp .env.example .env          # add your key
streamlit run app.py
```

Works with any OpenAI-compatible chat completions endpoint. For OpenAI itself,
leave `API_BASE` blank; for anything else — Groq, Together, OpenRouter, a local
server — point `API_BASE` at its URL.

---

## Quick start — the RAG chatbot

```bash
cd rag-chatbot
pip install -r requirements.txt
cp .env.example .env          # add your key
python ingest.py              # builds the index from docs/
streamlit run app.py
```

Embeddings run locally by default — ~90 MB downloaded once, then free and
offline. Only the final answer needs a key.

---

## A note on the "ATS score"

The resume reviewer deliberately does **not** give you an ATS score, and
episode 4 explains why at length. Real applicant tracking systems are keyword
matchers and every vendor differs, so a number a language model invents and
labels "ATS score" is fiction — and people rewrite their resume based on it.

It computes keyword coverage against a specific job ad instead: fifteen lines,
no model call, and you can check the arithmetic by hand.

*When you can compute something exactly, don't ask a model to guess it.*

---

## Licence

MIT. Fork it, break it, make it yours.
