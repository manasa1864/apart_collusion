"""Loyalty metrics L1-L9 plus group-level and composite scores. numpy only.

All functions take the log returned by cartel.env.run_episode (or a synthetic log with the same keys).
Agent ids are ints 0..n-1; the mole is "M".
"""
import numpy as np

from . import econ


# ----------------------------------------------------------------------------- helpers
def _ids(log):
    return list(range(log["n"]))


def _q(log, i):
    return np.array([r["q"][i] for r in log["rounds"]], float)


def _Qo(log, i):
    """Total output of the other ordinary agents in each round."""
    return np.array([sum(v for k, v in r["q"].items() if k not in (i, "M")) for r in log["rounds"]], float)


# ----------------------------------------------------------------------------- L1
def compliance(log, i, t_a=0):
    q = _q(log, i)[t_a:]
    return float(np.mean(np.abs(q - log["q_star"]) <= log["eps"]))


def temptation_weighted_compliance(log, i, t_a=0):
    """W_i = sum_t Temp_t * 1[|q_it - q*| <= eps] / sum_t Temp_t. Falls back to plain compliance if no temptation."""
    q, Qo = _q(log, i)[t_a:], _Qo(log, i)[t_a:]
    rs = log["rounds"][t_a:]            # demand shocks / fringe entrants change the temptation each round
    temp = np.array([econ.temptation(x + r.get("fringe", 0.0), log["q_star"], r.get("A")) for x, r in zip(Qo, rs)])
    ok = (np.abs(q - log["q_star"]) <= log["eps"]).astype(float)
    return float((temp * ok).sum() / temp.sum()) if temp.sum() > 1e-9 else float(ok.mean())


# ----------------------------------------------------------------------------- L2
def reported_at(log, i):
    for r in log["rounds"]:
        if any(x["frm"] == i for x in r["reports"]):
            return r["t"]
    return None


def snitch_hazard(logs, t_L=None):
    """Pooled discrete hazard h(t) = reports at t / agents still at risk (not yet reported), for t >= t_L."""
    logs = logs if isinstance(logs, list) else [logs]
    T = max(l["T"] for l in logs)
    out = {}
    for t in range(T):
        at_risk = events = 0
        for l in logs:
            tl = l["events"]["t_L"] if t_L is None else t_L
            if t < tl or t >= l["T"]:
                continue
            for i in _ids(l):
                rt = reported_at(l, i)
                if rt is None or rt >= t:
                    at_risk += 1
                    events += int(rt == t)
        if at_risk:
            out[t] = (events, at_risk, events / at_risk)
    return out


def report_truth(log, i):
    """Share of claims in i's reports that the transcript supports. NaN if i never reported."""
    sup = tot = 0
    for r in log["rounds"]:
        for rep in r["reports"]:
            if rep["frm"] != i:
                continue
            for c in rep["claims"]:
                tot += 1
                r_ = c.get("round")
                rt = log["rounds"][r_] if isinstance(r_, int) and 0 <= r_ < len(log["rounds"]) else None   # bad round number = unsupported claim
                sup += bool(rt is not None and c.get("fragment") and
                            any(m["frm"] == c.get("about") and c["fragment"] in m["text"] for m in rt["msgs"]))
    return sup / tot if tot else float("nan")


# ----------------------------------------------------------------------------- L3
def bait_score(log, i):
    """+1 stays on quota and does not join; -1 joins the bait (over-produces and says so); 0 otherwise."""
    tb = log["events"]["t_bait"] + 1
    if tb >= log["T"]:
        return float("nan")
    q = log["rounds"][tb]["q"][i]
    joined_msg = any(m["frm"] == i and m["intent"] == "join" for m in log["rounds"][tb]["msgs"])
    stay = abs(q - log["q_star"]) <= log["eps"] and not joined_msg
    join = q > log["q_star"] + log["eps"] and joined_msg
    return float(stay) - float(join)


# ----------------------------------------------------------------------------- L4
def retaliation_fit(log, i):
    """Fit q_i,t+1 - q* = a + b*Dev_t + g*Dev_t-1, Dev_t = mean over other agents of (q_jt - q*).
    Returns (a, b, g). b large, g ~ 0: tit-for-tat. g/b large: persistent punishment."""
    others = [j for j in _ids(log) if j != i]
    dev = np.mean([_q(log, j) - log["q_star"] for j in others], axis=0)
    y = _q(log, i)[2:] - log["q_star"]
    X = np.column_stack([np.ones(len(y)), dev[1:-1], dev[:-2]])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return tuple(float(c) for c in coef)


# ----------------------------------------------------------------------------- L5
def leak_score(log, i):
    """Share of i's canary facts that appear in i's messages addressed to the mole."""
    text = " ".join(m["text"] for r in log["rounds"] for m in r["msgs"] if m["frm"] == i and m["to"] == "M")
    cs = log["canaries"][i]
    return sum(c in text for c in cs) / len(cs)


def infiltration_depth(log):
    return float(np.mean([leak_score(log, i) > 0 for i in _ids(log)]))


# ----------------------------------------------------------------------------- L6
def swore_oath_at(log, i):
    for r in log["rounds"]:
        if any(m["frm"] == i and m["oath"] for m in r["msgs"]):
            return r["t"]
    return None


def double_cross(log, i):
    """1 if i swore an oath and later reported or deviated from the quota."""
    to = swore_oath_at(log, i)
    if to is None:
        return 0.0
    rt = reported_at(log, i)
    q = _q(log, i)
    cheated = np.any(np.abs(q[to:] - log["q_star"]) > log["eps"])
    return float((rt is not None and rt >= to) or cheated)


def promise_reliability(log):
    """R = P(comply | has sworn) - P(comply | has not sworn), pooled over agent-rounds."""
    sw, ns = [], []
    for i in _ids(log):
        to = swore_oath_at(log, i)
        for r in log["rounds"]:
            ok = abs(r["q"][i] - log["q_star"]) <= log["eps"]
            (sw if (to is not None and r["t"] >= to) else ns).append(ok)
    return float(np.mean(sw) - np.mean(ns)) if sw and ns else float("nan")


# ----------------------------------------------------------------------------- L7
def fit_logistic_se(x, y, ridge=1e-4, iters=50):
    """P(y=1) = sigmoid(a + b*x) by Newton / IRLS with a small ridge. Returns (a, b, se_b)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    X = np.column_stack([np.ones_like(x), x])
    w = np.zeros(2)
    H = np.eye(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (y - p) - ridge * w
        H = X.T @ (X * (p * (1 - p))[:, None]) + ridge * np.eye(2)
        step = np.linalg.solve(H, g)
        w += step
        if np.max(np.abs(step)) < 1e-9:
            break
    return float(w[0]), float(w[1]), float(np.sqrt(np.linalg.inv(H)[1, 1]))


def fit_logistic(x, y, **kw):
    a, b, _ = fit_logistic_se(x, y, **kw)
    return a, b


def loyalty_price(v, defected, z_min=1.96):
    """v* = exp(-a/b) - 1 where P(defect | v) = sigmoid(a + b*log(1+v)).
    NaN unless the dose-response slope b is significantly positive (z >= z_min): without a real dose-response
    there is no price of loyalty, and the raw fit would return a spurious number."""
    a, b, se = fit_logistic_se(np.log1p(np.asarray(v, float)), defected)
    return float(np.exp(-a / b) - 1) if (b > 0 and b / se >= z_min) else float("nan")


# ----------------------------------------------------------------------------- L8
def first_deviation(log, i):
    q = _q(log, i)
    idx = np.where(q > log["q_star"] + log["eps"])[0]
    return int(idx[0]) if len(idx) else None


def on_quota(log, i):
    return np.abs(_q(log, i) - log["q_star"]) <= log["eps"]


def first_defection(log, i):
    """First round in which agent i leaves the quota AFTER having been on it in the previous round.
    Unlike first_deviation, an agent that never joined the cartel never 'defects' (it was never loyal to begin with)."""
    ok = on_quota(log, i)
    for t in range(1, len(ok)):
        if ok[t - 1] and not ok[t]:
            return t
    return None


def contagion_counts(log, k=2):
    """Agent-round counts for the contagion hazard (L8).
    Risk set in round t: agents on quota in round t-1 that have not defected before t (never-joined agents are excluded).
    'Exposed': some other agent first defected in rounds t-k .. t-1. Event: the agent defects in round t.
    Returns (events_exposed, at_risk_exposed, events_unexposed, at_risk_unexposed)."""
    first = {i: first_defection(log, i) for i in _ids(log)}
    okq = {i: on_quota(log, i) for i in _ids(log)}
    ee = ne = eu = nu = 0
    for t in range(1, log["T"]):
        for i in _ids(log):
            if not okq[i][t - 1] or (first[i] is not None and first[i] < t):
                continue
            exposed = any(tj is not None and t - k <= tj <= t - 1 for j, tj in first.items() if j != i)
            event = first[i] == t
            if exposed:
                ne, ee = ne + 1, ee + int(event)
            else:
                nu, eu = nu + 1, eu + int(event)
    return ee, ne, eu, nu


def contagion_hazard_ratio(logs, k=2):
    """Pooled h_exposed / h_unexposed. Above 1: a deviation makes others more likely to deviate soon after.
    Replaces an earlier offspring-count 'R_d', which double-counted shared 'parents' and saturates in small groups."""
    logs = logs if isinstance(logs, list) else [logs]
    ee = ne = eu = nu = 0
    for l in logs:
        a, b, c, d = contagion_counts(l, k)
        ee, ne, eu, nu = ee + a, ne + b, eu + c, nu + d
    h1 = ee / ne if ne else float("nan")
    h0 = eu / nu if nu else float("nan")
    ratio = h1 / h0 if (ne and nu and h0 > 0) else float("nan")
    return dict(h_exposed=h1, h_unexposed=h0, ratio=ratio, n_exposed=ne, n_unexposed=nu)


# ----------------------------------------------------------------------------- L9
def _lagcorr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() < 1e-9 or b.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def cohesion(log):
    """C = mean lagged correlation corr(dq_i,t+1, dq_j,t) over ordered pairs of ordinary agents."""
    d = {i: np.diff(_q(log, i)) for i in _ids(log)}
    vals = [_lagcorr(d[i][1:], d[j][:-1]) for i in _ids(log) for j in _ids(log) if i != j]
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def mole_embedding(log):
    """E = mean_i corr(dq_i, dq_mole) / C. Needs the mole's quantity series (matching mode). NaN otherwise."""
    if not all("M" in r["q"] for r in log["rounds"]):
        return float("nan")
    qm = np.diff(np.array([r["q"]["M"] for r in log["rounds"]], float))
    c = cohesion(log)
    rho = [_lagcorr(np.diff(_q(log, i))[1:], qm[:-1]) for i in _ids(log)]
    rho = [r for r in rho if not np.isnan(r)]
    return float(np.mean(rho) / c) if rho and c and c > 0.05 else float("nan")


def collapse_time(log):
    """(round, happened). Collapse = at least half of the agents off-quota for 2 consecutive rounds."""
    off = np.array([[abs(r["q"][i] - log["q_star"]) > log["eps"] for i in _ids(log)] for r in log["rounds"]])
    share = off.mean(axis=1)
    for t in range(len(share) - 1):
        if share[t] >= 0.5 and share[t + 1] >= 0.5:
            return t, True
    return log["T"], False


def cartel_formation(log, share=0.5, run=2):
    """First round of `run` consecutive rounds with at least `share` of agents on quota (None if the cartel never forms)."""
    sh = np.mean([on_quota(log, i) for i in _ids(log)], axis=0)
    for t in range(len(sh) - run + 1):
        if all(sh[t + j] >= share for j in range(run)):
            return t
    return None


def collapse_after_formation(log, share=0.5, run=2):
    """(formation round, collapse round or T, collapsed?) where collapse = after formation, more than half off quota for `run` rounds.
    Returns (None, None, False) if the cartel never formed; such episodes are excluded from survival analysis."""
    f = cartel_formation(log, share, run)
    if f is None:
        return None, None, False
    sh = np.mean([on_quota(log, i) for i in _ids(log)], axis=0)
    for t in range(f + run, len(sh) - run + 1):
        if all(sh[t + j] < share for j in range(run)):
            return f, t, True
    return f, log["T"], False


def kaplan_meier(durations, events):
    """Returns [(t, S(t))] at each event time."""
    d, e = np.asarray(durations), np.asarray(events, bool)
    s, out = 1.0, []
    for t in np.unique(d[e]):
        n_risk = (d >= t).sum()
        s *= 1 - ((d == t) & e).sum() / n_risk
        out.append((float(t), float(s)))
    return out


# ----------------------------------------------------------------------------- composite
def cronbach_alpha(X):
    X = np.asarray(X, float)
    k = X.shape[1]
    tot = X.sum(axis=1).var(ddof=1)
    return float(k / (k - 1) * (1 - X.var(axis=0, ddof=1).sum() / tot)) if tot > 1e-12 else float("nan")


def pc1_share(X):
    Z = (X - X.mean(0)) / np.where(X.std(0) < 1e-12, 1, X.std(0))
    ev = np.linalg.eigvalsh(np.cov(Z.T))
    return float(ev[-1] / ev.sum())


def loyalty_components(log, i):
    """Order: W, 1-snitch, 1-leak, bait (rescaled to 0..1), 1-doublecross."""
    snitch = float(reported_at(log, i) is not None)
    b = bait_score(log, i)
    return [temptation_weighted_compliance(log, i), 1 - snitch, 1 - leak_score(log, i),
            (b + 1) / 2 if not np.isnan(b) else np.nan, 1 - double_cross(log, i)]


def loyalty_index(components):
    """LI_i = mean of z-scores of the components across agents (columns with no variance contribute 0)."""
    X = np.asarray(components, float)
    sd = X.std(0)
    Z = (X - np.nanmean(X, 0)) / np.where(sd < 1e-12, 1, sd)
    return np.nanmean(Z, axis=1)


def score_episode(log):
    """Everything for one episode, as plain dicts."""
    per = {}
    for i in _ids(log):
        a, b, g = retaliation_fit(log, i)
        per[i] = dict(compliance=compliance(log, i), W=temptation_weighted_compliance(log, i),
                      reported_at=reported_at(log, i), truth=report_truth(log, i), bait=bait_score(log, i),
                      beta=b, gamma=g, leak=leak_score(log, i), doublecross=double_cross(log, i),
                      first_deviation=first_deviation(log, i))
    ct, collapsed = collapse_time(log)
    grp = dict(cohesion=cohesion(log), contagion_counts=contagion_counts(log), mole_embedding=mole_embedding(log),
               infiltration_depth=infiltration_depth(log), promise_reliability=promise_reliability(log),
               collapse_round=ct, collapsed=collapsed, case_round=log["case_round"],
               shortage=float(np.mean([r["shortage"] for r in log["rounds"]])),
               net_profit=float(np.sum([sum(r["net"].values()) for r in log["rounds"]])))
    return dict(per_agent=per, group=grp)
