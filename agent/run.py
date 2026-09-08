"""Run the agent from the command line.

    python run.py
    python run.py "is order A-4460 here yet?"
"""
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from agent import run   # noqa: E402  — after load_dotenv, on purpose

GOAL = ("Order A-4471 — when is it arriving, and will all of it be here "
        "before Friday?")

if __name__ == "__main__":
    goal = " ".join(sys.argv[1:]) or GOAL
    print(f"goal: {goal}\n" + "-" * 72)
    r = run(goal)
    print("-" * 72)
    print(f"{r.tool_calls} tool calls, {r.recoveries} recovered error(s), "
          f"stopped: {r.stopped}")
