"""The AI half of the resume reviewer.

Kept in its own module, away from any Streamlit import, for one reason: this is
the part with the interesting bugs, and you want to be able to run it from a
plain Python prompt without spinning up a web server to see what it returns.

Every pattern from episode 2 shows up here exactly once, and they are labelled:

    system prompt       SYSTEM, below — the rules, separated from the task
    few-shot examples   EXAMPLES — judgement, shown rather than described
    structured output   Review / Finding — a schema, so the app reads fields
    "say I don't know"  the missing_sections rule in SYSTEM

There is deliberately no clever phrasing anywhere in the prompt.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict

# ── the schema ────────────────────────────────────────────────────────
# This is pattern 3 from episode 2, and it is what makes this an app rather
# than a chat. The UI reads `review.overall_score`; it never parses prose.


@dataclass
class Finding:
    """One specific, actionable observation about the resume."""
    severity: str          # "high" | "medium" | "low"
    section: str           # which part of the resume it refers to
    problem: str           # what is wrong, in one sentence
    fix: str               # what to do about it, concretely


@dataclass
class Review:
    overall_score: int             # 0-100, defined in SYSTEM so it is not vibes
    summary: str
    strengths: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    keyword_coverage: float | None = None   # 0-1, only when a job ad is supplied
    missing_keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# The JSON shape we ask the model for. Written out longhand rather than
# generated from the dataclass so that what the model is told and what the code
# expects are visibly the same thing in one screen.
SCHEMA = {
    "overall_score": "integer 0-100",
    "summary": "string, at most 2 sentences",
    "strengths": ["string"],
    "findings": [
        {
            "severity": '"high" | "medium" | "low"',
            "section": "string",
            "problem": "string, one sentence",
            "fix": "string, one concrete action",
        }
    ],
}


# ── the system prompt ─────────────────────────────────────────────────
# Pattern 4. Rules live here; the resume itself goes in the user message. That
# separation is what stops a resume containing the words "ignore previous
# instructions" from redirecting the review.
SYSTEM = """You review resumes for software and data roles.

Scoring, so the number means the same thing every time:
- 90-100: ready to send, no material weaknesses
- 70-89: solid, a few fixable weaknesses
- 50-69: real problems that will cost interviews
- below 50: needs rewriting before it is sent anywhere

Rules:
- Be specific. "Add metrics" is useless. "The Stripe migration bullet has no
  numbers - say how many transactions or how much latency dropped" is useful.
- Every finding must name a section and give one concrete action.
- If a standard section is missing entirely, say so as a finding rather than
  inventing content for it.
- If the resume text looks truncated or garbled, say that in the summary and
  score conservatively. Do not guess at what the missing part said.
- Never invent employers, dates, or achievements that are not in the text.

Return only JSON matching the requested shape. No prose outside the JSON."""


# ── few-shot examples ─────────────────────────────────────────────────
# Pattern 1. Two examples, not six: they ship on every call, so they are the
# shortest pair that still carries the judgement we want. Note the second one
# is deliberately a *bad* input, teaching the model what to do with garbage
# rather than only what a good review looks like (pattern 5).
EXAMPLES = [
    (
        "Senior Backend Engineer, Acme (2020-2024). Worked on the payments team. "
        "Responsible for the API. Used Python and Postgres.",
        {
            "overall_score": 58,
            "summary": "Relevant experience, but written as duties rather than "
                       "results. No evidence of scale or impact.",
            "strengths": ["Clear, relevant job title and stack"],
            "findings": [
                {
                    "severity": "high",
                    "section": "Experience - Acme",
                    "problem": "Every bullet describes responsibility, not outcome.",
                    "fix": "Rewrite each bullet as action + result + number, e.g. "
                           "'Cut payment API p99 latency from 800ms to 120ms'.",
                },
                {
                    "severity": "medium",
                    "section": "Overall",
                    "problem": "No indication of team size, traffic or scale.",
                    "fix": "Add one quantifier per role: requests/day, team size, or revenue touched.",
                },
            ],
        },
    ),
    (
        "%PDF-1.4 %âãÏÓ 1 0 obj << /Type /Catalog",
        {
            "overall_score": 0,
            "summary": "The extracted text is not readable resume content - the "
                       "PDF appears to be image-based or failed to parse.",
            "strengths": [],
            "findings": [
                {
                    "severity": "high",
                    "section": "Document",
                    "problem": "No readable text could be extracted from the file.",
                    "fix": "Export the resume as a text-based PDF rather than a scan "
                           "or image, then upload again.",
                }
            ],
        },
    ),
]


# ── keyword coverage ──────────────────────────────────────────────────
# Deliberately NOT asked of the model.
#
# The brief for this episode said "ATS score". Real applicant tracking systems
# are keyword matchers and every vendor differs, so a number a language model
# invents and calls an ATS score is made up - and candidates act on it. What is
# genuinely computable is how much of a specific job ad's vocabulary appears in
# the resume. That is arithmetic, it is checkable, and it is honest.

# Dots and hyphens are allowed INSIDE a token (node.js, ci-cd, .net) but not
# trailing, which was the bug: "postgres." never matched "postgres", so a
# term that was present in the resume got reported as missing.
_WORD = re.compile(r"[a-z0-9+#]+(?:[.\-][a-z0-9+#]+)*")

_STOP = {
    "the", "and", "for", "with", "you", "our", "are", "will", "have", "has", "this",
    "that", "your", "who", "all", "any", "can", "not", "from", "their", "them",
    "role", "team", "work", "working", "years", "year", "experience", "strong",
    "ability", "including", "using", "used", "use", "well", "such", "into", "across",
    "within", "about", "would", "should", "must", "plus", "etc", "job", "candidate",
    "we", "need", "looking", "join", "help", "build", "building", "engineer",
    "developer", "someone", "who'll", "what", "where", "when", "how",
}


def keyword_coverage(resume_text: str, job_ad: str) -> tuple[float, list[str]]:
    """Fraction of the job ad's meaningful vocabulary present in the resume.

    Returns (coverage 0-1, up to 15 missing terms). Not an ATS score - see the
    note above. Both sides are lowercased and stripped of a small stop list;
    anything cleverer than that starts to over-promise.
    """
    ad_words = {w for w in _WORD.findall(job_ad.lower()) if len(w) > 1 and w not in _STOP}
    if not ad_words:
        return 0.0, []
    resume_words = set(_WORD.findall(resume_text.lower()))
    missing = sorted(ad_words - resume_words)
    coverage = 1 - (len(missing) / len(ad_words))
    return round(coverage, 3), missing[:15]


# ── the call ──────────────────────────────────────────────────────────

def _build_messages(resume_text: str, job_ad: str = "") -> list[dict]:
    """System + few-shot + task, in the order the API expects."""
    messages = [{"role": "system", "content": SYSTEM}]
    for sample_resume, sample_review in EXAMPLES:
        messages.append({"role": "user", "content": f"RESUME:\n{sample_resume}"})
        messages.append({"role": "assistant", "content": json.dumps(sample_review)})

    task = f"RESUME:\n{resume_text}"
    if job_ad.strip():
        task += f"\n\nTARGET JOB AD:\n{job_ad}"
    task += (
        "\n\nReturn JSON with exactly this shape:\n"
        + json.dumps(SCHEMA, indent=2)
    )
    messages.append({"role": "user", "content": task})
    return messages


def review_resume(resume_text: str, job_ad: str = "", *, model: str | None = None) -> Review:
    """Send the resume for review and return a parsed Review.

    Provider-agnostic on purpose: it talks to any OpenAI-compatible chat
    completions endpoint, so switching provider is two environment variables
    rather than a rewrite. Set API_BASE, API_KEY and MODEL in .env.
    """
    from openai import OpenAI  # imported late so the module loads without the SDK

    client = OpenAI(
        api_key=os.environ["API_KEY"],
        base_url=os.environ.get("API_BASE") or None,
    )

    resp = client.chat.completions.create(
        model=model or os.environ.get("MODEL", "gpt-4o-mini"),
        messages=_build_messages(resume_text, job_ad),
        response_format={"type": "json_object"},   # structured output, natively
        temperature=0.2,
        max_tokens=1200,                            # episode 3: always cap output
    )

    # Episode 3 again: log usage on every call. One line, and it is the
    # difference between a surprise and a dashboard.
    if resp.usage:
        print(
            f"[tokens] in={resp.usage.prompt_tokens} "
            f"out={resp.usage.completion_tokens}"
        )

    return _parse(resp.choices[0].message.content, resume_text, job_ad)


def _parse(raw: str, resume_text: str, job_ad: str) -> Review:
    """Turn the model's JSON into a Review, tolerating small deviations."""
    data = json.loads(raw)

    findings = [
        Finding(
            severity=str(f.get("severity", "medium")).lower(),
            section=str(f.get("section", "Overall")),
            problem=str(f.get("problem", "")),
            fix=str(f.get("fix", "")),
        )
        for f in data.get("findings", [])
    ]

    # Clamp rather than trust: the score drives the UI, and a model returning
    # 105 should not paint a progress bar off the end of the card.
    score = int(data.get("overall_score", 0))
    score = max(0, min(100, score))

    review = Review(
        overall_score=score,
        summary=str(data.get("summary", "")),
        strengths=[str(s) for s in data.get("strengths", [])],
        findings=findings,
    )

    if job_ad.strip():
        review.keyword_coverage, review.missing_keywords = keyword_coverage(
            resume_text, job_ad
        )
    return review
