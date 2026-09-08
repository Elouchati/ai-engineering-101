# Resume Reviewer

Upload a resume as a PDF, get structured feedback: a score, what's working, and
a list of specific fixes. Optionally paste a job ad to see which of its terms
your resume never mentions.

Built in **AI Engineering 101, episode 4**. It's the first real app in the
playlist, and it exists to show that the patterns from episode 2 are enough to
build something people would actually use.

```
┌──────────┐   ┌───────────┐   ┌──────────────┐   ┌────────┐
│  PDF     │ → │  extract  │ → │ system +     │ → │ JSON   │
│  upload  │   │  text     │   │ few-shot +   │   │ → UI   │
└──────────┘   └───────────┘   │ schema       │   └────────┘
                               └──────────────┘
```

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env          # then add your key
streamlit run app.py
```

Works with any OpenAI-compatible chat completions endpoint. For OpenAI itself,
leave `API_BASE` blank. For anything else — Groq, Together, OpenRouter, a local
server — set `API_BASE` to its URL and `MODEL` to a model it serves.

## What's where

| File | What it holds |
|---|---|
| `reviewer.py` | The AI half — prompt, schema, parsing, keyword coverage. No Streamlit import, so you can run it from a plain REPL. |
| `app.py` | Upload, layout, error states. Deliberately thin. |

## The four patterns, labelled

Episode 2 covered five prompting patterns. Four of them are in `reviewer.py`,
each used exactly once so they're easy to find:

- **System prompt** — `SYSTEM`. Rules live there, the resume goes in the user
  message. That separation is what stops a resume containing "ignore previous
  instructions" from redirecting the review.
- **Few-shot** — `EXAMPLES`. Two, not six: they ship on every call, so they're
  the shortest pair that still carries the judgement.
- **Structured output** — `Review` / `Finding` plus `response_format`. The UI
  reads `review.overall_score`; it never parses prose.
- **Negative example** — the second entry in `EXAMPLES` is deliberately garbage
  input, teaching the model what to do with an unreadable PDF rather than only
  what a good review looks like.

## Two decisions worth explaining

**There is no "ATS score", on purpose.** Real applicant tracking systems are
keyword matchers and every vendor differs, so a number a language model invents
and labels "ATS score" is fiction — and candidates change their resume based on
it. What *is* computable is how much of a specific job ad's vocabulary appears
in the resume. That's arithmetic, it's checkable, and `keyword_coverage()` does
it in fifteen lines without calling a model at all.

**Image-only PDFs are handled explicitly.** `pypdf` returns an empty string for
a scanned resume rather than raising, which is the most common failure here —
someone exports from a design tool as an image and gets a confident review of
nothing. `app.py` checks for it before spending a call.

## Cost

One review is roughly 1,500–2,500 input tokens (the examples dominate) and
under 500 output. `max_tokens` is capped and usage is printed on every call —
both habits from episode 3.

## Deploy

Streamlit Community Cloud is the fastest path: push to GitHub, point it at the
repo, add `API_KEY` as a secret. `.env` is gitignored; don't commit your key.

## Ideas if you want to extend it

- Cache the system prompt and examples — they never change
- Add a second pass that rewrites the three weakest bullets
- Swap the schema and examples and it reviews cover letters instead
