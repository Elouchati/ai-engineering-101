"""Tracing for the agent, in about sixty lines.

Episode 10's argument is that you cannot debug an agent from its final answer,
because the answer is the one part of the run that always looks fine. What you
need is the sequence: what it called, what came back, how much it cost, and
where it went round in a circle.

Every run appends one JSON object per line to traces.jsonl. That's it — no
service, no SDK, no vendor. If you later want a hosted tool, you already have
the data model and you'll know what you're buying.

    from trace import Tracer
    run("...", tracer=Tracer("traces.jsonl"))
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Tracer:
    path: str | Path = "traces.jsonl"
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    started: float = field(default_factory=time.time)
    events: list[dict] = field(default_factory=list)

    def event(self, kind: str, **fields) -> None:
        self.events.append({
            "run_id": self.run_id,
            "t": round(time.time() - self.started, 3),
            "kind": kind,
            **fields,
        })

    # ── the three things worth recording ──────────────────────────────
    def step(self, n: int, tool: str, args: dict, result: object,
             prompt_tokens: int, ms: float) -> None:
        failed = isinstance(result, dict) and "error" in result
        self.event("step", n=n, tool=tool, args=args, prompt_tokens=prompt_tokens,
                   ms=round(ms, 1), failed=failed,
                   # The full result can be enormous. Keep a prefix — enough to
                   # tell what happened, small enough to keep forever.
                   result=json.dumps(result)[:400])

    def answer(self, n: int, text: str, prompt_tokens: int) -> None:
        self.event("answer", n=n, prompt_tokens=prompt_tokens, text=(text or "")[:600])

    def stopped(self, reason: str, n: int) -> None:
        self.event("stopped", reason=reason, n=n)

    def flush(self, goal: str) -> str:
        """Write the run and return its id."""
        record = {
            "run_id": self.run_id,
            "goal": goal,
            "seconds": round(time.time() - self.started, 2),
            "events": self.events,
        }
        p = Path(self.path)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return self.run_id
