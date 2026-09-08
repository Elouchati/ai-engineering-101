"""A tiny order book. Real data, real lookups — the tools below are not stubs.

Order A-4471 is the interesting one: it shipped as two parcels, which is the
case episode 8 used to explain why an agent beats a pipeline here. Nothing in
the agent knows that split shipments exist.
"""

from __future__ import annotations

TODAY = "2026-03-10"          # a Tuesday, fixed so the demo is reproducible

ORDERS: dict[str, dict] = {
    "A-4471": {
        "customer": "R. Achour",
        "placed": "2026-03-06",
        "status": "shipped",
        "items": [
            {"sku": "CBL-2M-USBC", "name": "USB-C cable, 2m", "qty": 2},
            {"sku": "HUB-7PT",     "name": "7-port powered hub", "qty": 1},
        ],
        # Two parcels. A pipeline that reads the first tracking number and stops
        # gets a true answer to a question nobody asked.
        "parcels": ["NW7712004GB", "NW7712119GB"],
    },
    "A-4460": {
        "customer": "L. Diallo",
        "placed": "2026-03-02",
        "status": "delivered",
        "items": [{"sku": "PSU-90W", "name": "90W power supply", "qty": 1}],
        "parcels": ["NW7711884GB"],
    },
}

PARCELS: dict[str, dict] = {
    "NW7712004GB": {"carrier": "Northwind Express", "status": "in transit",
                    "eta": "2026-03-12", "last_scan": "Birmingham hub"},
    "NW7712119GB": {"carrier": "Northwind Express", "status": "in transit",
                    "eta": "2026-03-16", "last_scan": "awaiting collection"},
    "NW7711884GB": {"carrier": "Northwind Express", "status": "delivered",
                    "eta": "2026-03-04", "last_scan": "delivered, front porch"},
}
