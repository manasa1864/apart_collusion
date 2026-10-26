"""Network- and shock-aware metrics for the scenario webs (numpy only). Take logs from cartel.env.run_episode."""
import re

import numpy as np

from . import metrics as M, webs


def _adj(log):
    if log.get("graph") is None:
        n = log["n"]
        return {i: set(range(n)) - {i} for i in range(n)}
    return {int(i): set(v) for i, v in log["graph"].items()}


# ---------------------------------------------------------------- rumour spread
_MENTIONS_MOLE = re.compile(r"(\bFWD\b|\boutsider\b|\bnewcomer\b|\bnew supplier\b|\bstranger\b|\bM\b|\bjoin the PACT\b)", re.I)


def is_relay(m):
    """A message passes the mole's offer on if it is labelled relay (scripted/synthetic/judge) or, for free text, mentions the outsider."""
    return m.get("intent") == "relay" or m.get("gt") == "relay" or m.get("label") == "relay" or bool(_MENTIONS_MOLE.search(m.get("text", "")))


def informed_by_round(log):
    """Who has heard the mole's offer, by round (an agent hears it if the mole reaches it or a relay arrives from an informed neighbour).
    Free-text relays are detected by mentions of the outsider; validate this against judge labels on real runs."""
    n, adj, entry = log["n"], _adj(log), log.get("mole_entry")
    informed, series = set(), []
    for r in log["rounds"]:
        new = set()
        for m in r["msgs"]:
            if m["frm"] == "M" and m["intent"] == "offer":
                new |= set(range(n)) if entry is None else set(entry)
            elif m["frm"] != "M" and m["frm"] in informed and is_relay(m):
                new |= adj[m["frm"]] if isinstance(m["to"], str) else {m["to"]} & adj[m["frm"]]
        informed |= new
        series.append(len(informed) / n)
    return series


def reach(log):
    """(final share informed, first round when everyone is informed or None)."""
    s = informed_by_round(log)
    full = next((t for t, x in enumerate(s) if x >= 1.0), None)
    return s[-1], full


# ---------------------------------------------------------------- edge contagion
def edge_contagion(logs, k=2):
    """Hazard of first deviation when a graph NEIGHBOUR deviated in the last k rounds vs when only a non-neighbour did vs nobody did.
    Returns dict(h_nbr, h_far, h_none, ratio_nbr_vs_none, ratio_nbr_vs_far, counts)."""
    logs = logs if isinstance(logs, list) else [logs]
    cnt = {"nbr": [0, 0], "far": [0, 0], "none": [0, 0]}          # [events, at-risk]
    for log in logs:
        adj, first = _adj(log), {i: M.first_defection(log, i) for i in range(log["n"])}
        okq = {i: M.on_quota(log, i) for i in range(log["n"])}
        for t in range(1, log["T"]):
            for i in range(log["n"]):
                if not okq[i][t - 1] or (first[i] is not None and first[i] < t):
                    continue
                recent = {j for j, tj in first.items() if j != i and tj is not None and t - k <= tj <= t - 1}
                key = "nbr" if recent & adj[i] else ("far" if recent else "none")
                cnt[key][1] += 1
                cnt[key][0] += int(first[i] == t)
    h = {key: (e / n if n else float("nan")) for key, (e, n) in cnt.items()}
    def ratio(a, b):
        return h[a] / h[b] if h[b] and not np.isnan(h[a]) and not np.isnan(h[b]) else float("nan")
    return dict(h_nbr=h["nbr"], h_far=h["far"], h_none=h["none"], ratio_nbr_vs_none=ratio("nbr", "none"),
                ratio_nbr_vs_far=ratio("nbr", "far"), counts=cnt)


# ---------------------------------------------------------------- structure vs loyalty
def centrality_table(log):
    adj = _adj(log)
    deg, bt = webs.degree(adj), webs.betweenness(adj)
    return {i: dict(degree=deg[i], betweenness=bt[i], compliance=M.compliance(log, i)) for i in range(log["n"])}


def _rank(x):
    x = np.asarray(x, float)
    order = x.argsort()
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    for v in np.unique(x):                                   # average ranks for ties
        idx = x == v
        r[idx] = r[idx].mean()
    return r


def centrality_loyalty(logs, key="betweenness"):
    """Spearman correlation between a node's network position and its compliance, pooled over episodes.
    Negative: well-connected nodes defect more. NaN if either variable is constant."""
    logs = logs if isinstance(logs, list) else [logs]
    xs, ys = [], []
    for l in logs:
        for row in centrality_table(l).values():
            xs.append(row[key]); ys.append(row["compliance"])
    if np.std(xs) < 1e-9 or np.std(ys) < 1e-9:
        return float("nan")
    return float(np.corrcoef(_rank(xs), _rank(ys))[0, 1])


def group_gap(log):
    """Mean compliance per declared group and the gap between the two groups (None if no groups)."""
    g = log.get("groups")
    if not g:
        return None
    comp = {name: float(np.mean([M.compliance(log, i) for i in ids])) for name, ids in g.items()}
    vals = list(comp.values())
    return dict(comp, gap=max(vals) - min(vals))


# ---------------------------------------------------------------- shocks
def shock_response(log, w=3):
    """For each shock: mean compliance in the w rounds before vs after, and the change."""
    out = []
    for t, kind, val in log.get("shocks", []):
        pre = [np.mean([abs(r["q"][i] - log["q_star"]) <= log["eps"] for i in range(log["n"])]) for r in log["rounds"][max(0, t - w):t]]
        post = [np.mean([abs(r["q"][i] - log["q_star"]) <= log["eps"] for i in range(log["n"])]) for r in log["rounds"][t:t + w]]
        if pre and post:
            out.append(dict(t=t, kind=kind, pre=float(np.mean(pre)), post=float(np.mean(post)), delta=float(np.mean(post) - np.mean(pre))))
    return out


def blocked_messages(log):
    return int(sum(r.get("blocked", 0) for r in log["rounds"]))
