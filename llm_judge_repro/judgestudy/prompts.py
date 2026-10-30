"""Prompt rendering and strict verdict parsing.

Canonical frame: every system pair has a canonical system A (alphabetically first name) and B.
order == "AB": canonical A is shown first (as "Assistant A").  order == "BA": canonical B is shown first.
"""
import json, re
from functools import lru_cache

_TAG = re.compile(r"\[\[\s*(A|B|C)\s*\]\]")


@lru_cache(maxsize=4)
def _template(path):
    return open(path).read().rstrip("\n")


def render(payload, order, cfg):
    """payload: dict (or JSON string) holding canonical fields; returns chat messages."""
    p = json.loads(payload) if isinstance(payload, str) else payload
    swap = order == "BA"
    if p["kind"] == "single":
        a, b = (p["resp_B"], p["resp_A"]) if swap else (p["resp_A"], p["resp_B"])
        txt = (_template(cfg["prompts"]["single"])
               .replace("{question}", p["question"]).replace("{answer_a}", a).replace("{answer_b}", b))
    else:
        a1, a2 = (p["a1_B"], p["a2_B"]) if swap else (p["a1_A"], p["a2_A"])
        b1, b2 = (p["a1_A"], p["a2_A"]) if swap else (p["a1_B"], p["a2_B"])
        txt = (_template(cfg["prompts"]["multi"])
               .replace("{question_1}", p["q1"]).replace("{question_2}", p["q2"])
               .replace("{answer_a_1}", a1).replace("{answer_a_2}", a2)
               .replace("{answer_b_1}", b1).replace("{answer_b_2}", b2))
    return [{"role": "user", "content": txt}]


def parse_shown(text):
    """Last [[A]] / [[B]] / [[C]] tag in the output, or None if invalid."""
    m = _TAG.findall(text or "")
    return m[-1] if m else None


def to_canonical(shown, order):
    """Map the label shown to the judge back to the canonical frame: 'A','B','T' or 'X' (invalid)."""
    if shown is None:
        return "X"
    if shown == "C":
        return "T"
    if order == "AB":
        return shown
    return "B" if shown == "A" else "A"


SCORE = {"A": 1.0, "T": 0.5, "B": 0.0, "X": 0.5}   # invalid verdicts count as ties (pre-specified, rate reported)
