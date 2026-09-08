"""The tools, and the schemas the model actually sees.

Episode 8's line was: a tool is a function plus a written description, and the
model never sees your code — only the description. So the description is the
interface. These are written the way you would write them for a colleague who
cannot read the implementation, because that is exactly the situation.

Two rules this file follows and most tutorials don't:

**Say what comes back, not just what goes in.** "Look up an order" tells the
model nothing about whether it will get one tracking number or four. The
description below says a list, which is what makes the agent check both parcels
in the demo instead of stopping at the first.

**Fail as data, never as an exception.** A tool that raises kills the loop. A
tool that returns {"error": ...} becomes an observation, and the model gets a
chance to fix its own mistake — which is the whole point of running a loop
rather than a chain.
"""

from __future__ import annotations

from data import ORDERS, PARCELS, TODAY


def get_order(order_id: str) -> dict:
    order = ORDERS.get(order_id.strip().upper())
    if order is None:
        return {"error": f"no order with id {order_id!r}",
                "hint": "order ids look like A-4471"}
    return order


def track_parcel(tracking_number: str) -> dict:
    parcel = PARCELS.get(tracking_number.strip().upper())
    if parcel is None:
        return {"error": f"no parcel with tracking number {tracking_number!r}",
                "hint": "tracking numbers come from get_order's 'parcels' list"}
    return parcel


def today() -> dict:
    return {"date": TODAY}


# The registry: name -> (function, schema). The schema is what gets sent.
TOOLS = {
    "get_order": (
        get_order,
        {
            "type": "function",
            "function": {
                "name": "get_order",
                "description": (
                    "Look up one order by its ID. Returns the customer, the date "
                    "it was placed, the status, the items, and a LIST of tracking "
                    "numbers under 'parcels' — an order may ship as more than one "
                    "parcel, each arriving on a different day."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string",
                                     "description": "e.g. A-4471"},
                    },
                    "required": ["order_id"],
                },
            },
        },
    ),
    "track_parcel": (
        track_parcel,
        {
            "type": "function",
            "function": {
                "name": "track_parcel",
                "description": (
                    "Track ONE parcel by its tracking number. Returns the carrier, "
                    "the current status, the estimated delivery date as 'eta', and "
                    "the last scan location. Call it once per tracking number — it "
                    "does not accept a list."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tracking_number": {"type": "string",
                                            "description": "e.g. NW7712004GB"},
                    },
                    "required": ["tracking_number"],
                },
            },
        },
    ),
    "today": (
        today,
        {
            "type": "function",
            "function": {
                "name": "today",
                "description": "Today's date, as YYYY-MM-DD. Takes no arguments.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ),
}

SCHEMAS = [schema for _, schema in TOOLS.values()]
