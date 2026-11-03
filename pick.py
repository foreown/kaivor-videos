#!/usr/bin/env python3
"""Pick the spec for a date (default: today in Europe/Oslo).

Usage: python3 pick.py [YYYY-MM-DD] > spec.json
Prints the chosen spec as JSON (includes "caption" and "hashtags" for the post).
"""
import json, os, sys, datetime
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(HERE, "topics.json")))
day = (datetime.date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1
       else datetime.datetime.now(ZoneInfo("Europe/Oslo")).date())
key = day.isoformat()
if key in data["overrides"]:
    spec = data["overrides"][key]
else:
    n = (day - datetime.date.fromisoformat(data["start_date"])).days
    spec = data["rotation"][n % len(data["rotation"])]
spec = dict(spec, date=key)
print(json.dumps(spec, ensure_ascii=False, indent=2))
