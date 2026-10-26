"""Save/load episode logs as JSON. JSON turns the int keys of per-agent dicts into strings; load_log turns them back."""
import json


def _fix(o):
    if isinstance(o, dict):
        return {(int(k) if isinstance(k, str) and k.lstrip("-").isdigit() else k): _fix(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_fix(x) for x in o]
    return o


def dumps(log):
    return json.dumps(log, default=lambda x: sorted(x) if isinstance(x, set) else str(x))


def loads(s):
    log = _fix(json.loads(s))
    return log
