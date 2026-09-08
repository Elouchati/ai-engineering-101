"""Read traces.jsonl and say what actually happened.

This is the whole of episode 10's argument in one file. Given only a final
answer, three of the four failure modes below look like success. Given the
trace, every one of them is obvious in a single line of output.

    python inspect_runs.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

TRACES = Path(__file__).parent / "traces.jsonl"


def summarise(record: dict) -> dict:
    steps = [e for e in record["events"] if e["kind"] == "step"]
    answer = next((e for e in record["events"] if e["kind"] == "answer"), None)
    stopped = next((e for e in record["events"] if e["kind"] == "stopped"), None)

    # A repeated (tool, args) pair is the signature of a stuck loop. It is the
    # single most useful thing to compute from a trace and almost nobody does.
    calls = Counter((e["tool"], json.dumps(e["args"], sort_keys=True)) for e in steps)
    repeats = sum(n - 1 for n in calls.values() if n > 1)

    failed = [e for e in steps if e["failed"]]
    # Two distinct shapes hide inside "hallucinated tool call", and they have
    # different fixes: a tool that does not exist (your registry, or the
    # prompt), versus the right tool called with the wrong argument names
    # (your schema).
    phantom = sum(1 for e in failed if "no such tool" in e["result"])
    badargs = sum(1 for e in failed if "bad arguments" in e["result"])
    # An error on the very last step, followed by an answer, means the agent
    # answered over the top of a failure. That is the dangerous one.
    answered_over = bool(answer and steps and steps[-1]["failed"])

    return {
        "run": record["run_id"],
        "steps": len(steps),
        "tokens": sum(e["prompt_tokens"] for e in steps) +
                  (answer["prompt_tokens"] if answer else 0),
        "errors": len(failed),
        "phantom": phantom,
        "badargs": badargs,
        "repeats": repeats,
        "outcome": "answered" if answer else (stopped or {}).get("reason", "?"),
        "answered_over_error": answered_over,
        "goal": record["goal"],
    }


def verdict(s: dict) -> str:
    # Ordered most-diagnostic first: a stuck loop that also answers over an
    # error is a stuck loop, and saying so is more useful.
    if s["repeats"] >= 3:
        return "STUCK LOOP - same call repeated"
    if s["phantom"]:
        return "PHANTOM TOOL - called a tool that does not exist"
    if s["answered_over_error"]:
        return "SILENT FAILURE - answered over an error"
    if s["outcome"] == "max_steps":
        return "HIT THE LIMIT - no answer"
    if s["badargs"]:
        return f"recovered from {s['errors']} bad-argument error(s)"
    if s["errors"]:
        return f"recovered from {s['errors']} error(s)"
    return "clean"


if __name__ == "__main__":
    if not TRACES.exists():
        raise SystemExit("no traces.jsonl yet — run the agent first")

    rows = [summarise(json.loads(line)) for line in
            TRACES.read_text(encoding="utf-8").splitlines() if line.strip()]

    print(f"{'run':>9}  {'steps':>5}  {'tokens':>7}  {'err':>3}  {'rpt':>3}  "
          f"{'outcome':<10}  verdict")
    print("-" * 92)
    for s in rows:
        print(f"{s['run']:>9}  {s['steps']:>5}  {s['tokens']:>7,}  {s['errors']:>3}  "
              f"{s['repeats']:>3}  {s['outcome']:<10}  {verdict(s)}")

    bad = [s for s in rows if verdict(s) not in ("clean",)
           and not verdict(s).startswith("recovered")]
    print(f"\n{len(rows)} runs, {len(bad)} needing attention, "
          f"{sum(s['tokens'] for s in rows):,} prompt tokens total")
