"""Offline stand-in for the model API (OpenAI-compatible response shape). For testing the plumbing ONLY: its 'decisions' are pseudo-random,
so nothing produced with it says anything about real models."""
import hashlib
import json


def fake_transport(payload):
    user = payload["messages"][-1]["content"]
    h = int(hashlib.sha256(user.encode()).hexdigest(), 16)
    q = [2.0, 2.0, 2.0, 3.0, 6.0][h % 5]
    msgs = [dict(to="all", text="let's hold quota 2 together")] if h % 7 == 0 else []
    rep = None
    if "LENIENCY NOW OPEN" in payload["messages"][0]["content"] and h % 11 == 0:
        rep = dict(claims=[dict(about=(h % 4) + 1, round=1, fragment="let's hold quota 2")])
    text = json.dumps(dict(quantity=q, messages=msgs, report=rep))
    n = (len(payload["messages"][0]["content"]) + len(user)) // 4
    return dict(choices=[dict(message=dict(content=text))], usage=dict(total_tokens=n + 40))
