"""The agent loop. This is the whole mechanism from episode 8, in about forty
lines of actual code.

    observe  ->  the transcript so far
    think    ->  one model call, which either answers or asks for a tool
    act      ->  run that tool, append the result, go again

Three things in here are worth more than the rest of the file:

**MAX_STEPS is not optional.** Nothing in the mechanism guarantees the loop
stops. A model that keeps calling a tool that keeps failing will keep going
until something else stops it, and that something else should be you rather
than your bill.

**Tool errors are appended, not raised.** A tool that raises ends the run. A
tool whose error goes back into the transcript gives the model a chance to
correct itself, and that is the only reason the loop is worth having.

**The transcript is re-sent every single step.** You can watch it happen in the
token counts this prints. That is why an n-step loop costs about n-squared over
two times a single call, not n times.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

from tools import TOOLS, SCHEMAS

# Overridable so the guard can be demonstrated, not just described.
MAX_STEPS = int(os.environ.get("MAX_STEPS", 8))

SYSTEM = """You are a customer support agent for Northwind Supply.

Use the tools to find out what is actually true before you answer. Do not guess
dates, and do not assume an order is a single parcel — check every tracking
number the order gives you.

When you have the answer, state it plainly in two or three sentences. If a tool
returns an error, read it and try to recover rather than giving up."""


@dataclass
class Step:
    """One turn of the loop, kept so the run can be printed or tested."""
    n: int
    tool: str | None = None
    args: dict = field(default_factory=dict)
    result: object = None
    text: str | None = None
    prompt_tokens: int = 0

    @property
    def failed(self) -> bool:
        return isinstance(self.result, dict) and "error" in self.result


@dataclass
class Run:
    goal: str
    steps: list[Step]
    answer: str | None
    stopped: str            # "answered" | "max_steps"

    @property
    def tool_calls(self) -> int:
        return sum(1 for s in self.steps if s.tool)

    @property
    def recoveries(self) -> int:
        """Steps that failed but were not the last thing the agent did."""
        return sum(1 for s in self.steps[:-1] if s.failed)


def _client():
    from openai import OpenAI
    return OpenAI(api_key=os.environ["API_KEY"],
                  base_url=os.environ.get("API_BASE") or None)


def run(goal: str, verbose: bool = True, tracer=None) -> Run:
    client = _client()
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": goal},
    ]
    steps: list[Step] = []

    for n in range(1, MAX_STEPS + 1):
        # ── think ──────────────────────────────────────────────────────
        t0 = time.time()
        resp = client.chat.completions.create(
            model=os.environ.get("MODEL", "gpt-4o-mini"),
            messages=messages,
            tools=SCHEMAS,
            temperature=0,
        )
        ms = (time.time() - t0) * 1000
        msg = resp.choices[0].message
        used = resp.usage.prompt_tokens if resp.usage else 0

        # ── answered? then we're done ─────────────────────────────────
        if not msg.tool_calls:
            steps.append(Step(n=n, text=msg.content, prompt_tokens=used))
            if verbose:
                print(f"\n[{n}] answer  ({used} prompt tokens)\n    {msg.content}")
            if tracer:
                tracer.answer(n, msg.content, used)
                tracer.flush(goal)
            return Run(goal, steps, msg.content, "answered")

        # The assistant turn has to go back in verbatim, tool calls and all —
        # leave it out and the next request is malformed.
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [{
                "id": tc.id, "type": "function",
                "function": {"name": tc.function.name,
                             "arguments": tc.function.arguments},
            } for tc in msg.tool_calls],
        })

        # ── act ────────────────────────────────────────────────────────
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            fn = TOOLS.get(name, (None, None))[0]
            if fn is None:
                result = {"error": f"no such tool {name!r}",
                          "hint": f"available: {', '.join(TOOLS)}"}
            else:
                try:
                    result = fn(**args)
                except TypeError as e:
                    # Wrong arguments are the model's mistake to fix, not a crash.
                    result = {"error": f"bad arguments for {name}: {e}"}

            step = Step(n=n, tool=name, args=args, result=result, prompt_tokens=used)
            steps.append(step)
            if verbose:
                flag = "  ERROR" if step.failed else ""
                print(f"[{n}] {name}({', '.join(f'{k}={v!r}' for k, v in args.items())})"
                      f"  ({used} prompt tokens){flag}")
                print(f"    -> {json.dumps(result)[:150]}")

            if tracer:
                tracer.step(n, name, args, result, used, ms)
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": json.dumps(result)})

    # ── the limit did its job ─────────────────────────────────────────
    if verbose:
        print(f"\n[!] stopped at MAX_STEPS={MAX_STEPS} without an answer")
    if tracer:
        tracer.stopped("max_steps", MAX_STEPS)
        tracer.flush(goal)
    return Run(goal, steps, None, "max_steps")
