# A support agent, with real tools and a real trace

**Episodes 9 and 10** of [AI Engineering 101](https://www.youtube.com/@techinonemin1).

An agent is a model in a loop, choosing its own next action from a set of tools,
until a goal is met or it gives up. This is that, in about 150 lines — plus
about 120 more that record what it did, because you cannot debug an agent from
its final answer.

---

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env          # add your key
python run.py
```

```
[1] get_order(order_id='A-4471')                    (116 prompt tokens)
[2] track_parcel(tracking_number='NW7712004')       (177 prompt tokens)  ERROR
[3] track_parcel(tracking_number='NW7712004GB')     (207 prompt tokens)
[4] track_parcel(tracking_number='NW7712119GB')     (234 prompt tokens)
[5] today()                                         (262 prompt tokens)
[6] answer                                          (268 prompt tokens)
```

Works with any OpenAI-compatible chat completions endpoint that supports tool
calling. Leave `API_BASE` blank for OpenAI itself.

---

## What's in each file

| | |
|---|---|
| `agent.py` | The loop. About forty lines do the work. |
| `tools.py` | Three tools, and the schemas the model actually sees. |
| `data.py` | A small order book. Order `A-4471` ships as two parcels — that's the interesting case. |
| `trace.py` | **Episode 10** — one line of JSON per run. No service, no vendor. |
| `inspect_runs.py` | **Episode 10** — reads the traces back and says what went wrong. |
| `run.py` | A command line. |

---

## The four things that matter

**A tool is a function plus a description, and the model never sees your code.**
So the description is the interface. Two rules most examples get wrong: say what
comes *back*, not just what goes in (`get_order` says it returns a **list** of
tracking numbers, which is the only reason the agent checks both parcels); and
put constraints in there too (`track_parcel` says it does not accept a list).

**Fail as data, never as an exception.** A tool that raises ends the run. A tool
that returns `{"error": ..., "hint": ...}` becomes an observation the model can
read and recover from. That single choice is the difference between an agent
that fixes its own mistake and one that just stops.

**The assistant's own turn goes back into the transcript verbatim**, tool calls
and all. Leave it out and you have a tool result with nothing to attach it to,
and the next request is rejected as malformed. This will cost you an afternoon.

**`MAX_STEPS` is not optional.** Nothing in the mechanism guarantees the loop
stops. Try it:

```bash
MAX_STEPS=3 python run.py     # stops after 3, no answer — the guard working
```

---

## Tracing — episode 10

```python
from trace import Tracer
run(goal, tracer=Tracer("traces.jsonl"))
```

```bash
python inspect_runs.py
```

```
      run  steps   tokens  err  rpt  outcome     verdict
 8eaf680d      5    1,264    1    0  answered    recovered from 1 error(s)
 7d0c59ff      8    1,696    8    7  max_steps   STUCK LOOP - same call repeated
 c3b4e89b      3      694    2    1  answered    PHANTOM TOOL - called a tool that does not exist
 9932da34      5    1,200    1    0  answered    recovered from 1 bad-argument error(s)
 a70d6f52      2      476    1    0  answered    SILENT FAILURE - answered over an error
```

Those are five real runs of this agent. **Four returned a confident answer. Two
of those four were wrong.** The only run that visibly failed — the one that hit
the step limit — was the safest of the five, because it refused to answer.

From the final message alone you cannot tell them apart. From the trace it takes
one glance.

The two columns doing the work:

- **`rpt`** — how many times the same tool was called with exactly the same
  arguments. The stuck run shows 7; everything else shows 0 or 1. This is the
  single most useful number you can compute from a trace, and almost nobody
  computes it.
- **answered-after-a-failure** — an answer whose immediately preceding step
  errored. That catches both the phantom tool and the silent failure.

### What to alert on

1. Repeated identical calls — a stuck loop, burning money while it's stuck.
2. An answer straight after a failed step — the highest-value alert here.
3. Runs that hit the step limit. Individually fine; a rising *rate* is not.
4. Tokens per run at **p95**, not the mean. Agents don't get expensive on
   average, they get expensive in the tail.

---

## Making it yours

The loop doesn't change. Swap the tools — search a codebase and read files,
query a database and send an email, read a calendar and book a room.

Expect almost all of your debugging to be in the tool descriptions, not in the
loop and not in the model. When the agent keeps picking the wrong tool, read the
description back and ask whether *you* could have followed it.

---

## Licence

MIT.
