"""Network formulas for the scenario webs, each adapted from a published model, plus the observed counterparts.
Predictions come from theory (what a rational/mechanical network would do); data from LLM runs will be compared against them.

 N1 key player (Ballester, Calvo-Armengol & Zenou, Econometrica 2006): intercentrality c_i = b_i^2 / m_ii
 N2 spreading threshold (Chakrabarti, Wang, Faloutsos et al. 2008; Wang et al. 2003): defection spreads if beta*lambda1(A) > delta
 N3 threshold cascade (Granovetter 1978; Kempe, Kleinberg & Tardos KDD 2003): activate when share of active neighbours >= theta_i
 N4 stubborn conformity (DeGroot 1974; Friedkin & Johnsen 1990): x_i(t+1) = a_i * mean_nbr x_j(t) + b_i * x_i(0)
 N5 mixing time (DeGroot 1974): rounds for group quotas to align ~ ln(1/eps) / ln(1/|lambda_2(W)|)
"""
import numpy as np

from . import webs, metrics as M


def A(adj, n=5):
    g = np.zeros((n, n))
    for i, v in adj.items():
        for j in v:
            g[i, j] = 1.0
    return g


def lam1(g):
    return float(max(np.linalg.eigvals(g).real))


# ------------------------------------------------------------------ N1  key player
def bonacich(g, delta_frac=0.5):
    """b = (I - delta G)^-1 1 with delta = delta_frac / lambda1 (so the series converges). Returns (b, M, delta)."""
    d = delta_frac / lam1(g)
    Mx = np.linalg.inv(np.eye(len(g)) - d * g)
    return Mx @ np.ones(len(g)), Mx, d


def intercentrality(g, delta_frac=0.5):
    """c_i = b_i^2 / m_ii. Equals the fall in total Bonacich centrality sum_j b_j when node i is deleted (checked numerically in tests).
    The 'key player' is argmax c_i: the member whose removal (arrest, flip to informant) hurts the cartel's coordination most."""
    b, Mx, _ = bonacich(g, delta_frac)
    return b ** 2 / np.diag(Mx)


def key_player(adj, delta_frac=0.5):
    c = intercentrality(A(adj), delta_frac)
    return int(np.argmax(c)), c


# ------------------------------------------------------------------ N2  spreading threshold
def spreading_ratio(adj, beta, delta=0.0):
    """tau*lambda1 with tau = beta/delta (delta = recovery rate). Defection is predicted to spread when > 1.
    Defection here does not recover (delta ~ 0), so use growth_rate = beta*lambda1 - delta instead."""
    l1 = lam1(A(adj))
    return dict(lambda1=l1, ratio=(beta * l1 / delta) if delta > 0 else float("inf"), growth_rate=beta * l1 - delta)


def beta_hat(edge_contagion_result, k=2):
    """Per-round hazard of defecting after a neighbour defected, from webmetrics.edge_contagion (h_nbr) minus background (h_none)."""
    h = edge_contagion_result
    return max(0.0, h["h_nbr"] - (h["h_none"] if not np.isnan(h["h_none"]) else 0.0)) if not np.isnan(h["h_nbr"]) else float("nan")


# ------------------------------------------------------------------ N3  threshold cascade
def lt_cascade(adj, seeds, theta):
    """Deterministic linear-threshold cascade with edge weights 1/deg(v). theta: scalar or {node: threshold}. Returns (final active set, rounds)."""
    n = len(adj)
    th = (lambda v: theta) if np.isscalar(theta) else (lambda v: theta[v])
    active, rounds = set(seeds), 0
    while True:
        new = {v for v in range(n) if v not in active and adj[v] and sum(u in active for u in adj[v]) / len(adj[v]) >= th(v) - 1e-12}
        if not new:
            return active, rounds
        active |= new
        rounds += 1


def critical_theta(adj, seeds, grid=None):
    """Largest common threshold at which the seeds still cascade to everyone (the web's 'fragility'). 0 if never."""
    grid = np.round(np.arange(0.05, 1.0001, 0.05), 2) if grid is None else grid
    best = 0.0
    for th in grid:
        act, _ = lt_cascade(adj, seeds, float(th))
        if len(act) == len(adj):
            best = float(th)
    return best


def empirical_thresholds(logs):
    """theta_i = share of i's neighbours that had already deviated when i first deviated. Pooled over episodes; agents who never deviate are right-censored (excluded)."""
    logs = logs if isinstance(logs, list) else [logs]
    th = []
    for log in logs:
        adj = {int(i): set(v) for i, v in log["graph"].items()} if log.get("graph") else {i: set(range(log["n"])) - {i} for i in range(log["n"])}
        first = {i: M.first_defection(log, i) for i in range(log["n"])}
        for i, ti in first.items():
            if ti is not None and adj[i]:
                th.append(sum(first[j] is not None and first[j] < ti for j in adj[i]) / len(adj[i]))
    return np.array(th)


def granovetter_fixed_point(thetas, r0=0.0, iters=100):
    """Mean-field cascade: r_{t+1} = F(r_t), F = empirical CDF of thresholds. Returns the share that ends up defecting."""
    thetas = np.asarray(thetas, float)
    r = r0
    for _ in range(iters):
        nxt = float(np.mean(thetas <= r + 1e-12))
        if abs(nxt - r) < 1e-12:
            break
        r = nxt
    return r


# ------------------------------------------------------------------ N4  stubborn conformity
def fj_fit(log, i):
    """Fit x_i(t+1) = a_i * mean_{j in N(i)} x_j(t) + b_i * x_i(0), x = q - q*.  a = susceptibility to neighbours (conformity),
    b = anchoring to the agent's own starting stance (stubbornness). Friedkin-Johnsen has a + b = 1; we fit both and report them."""
    adj = {int(k): set(v) for k, v in log["graph"].items()} if log.get("graph") else {k: set(range(log["n"])) - {k} for k in range(log["n"])}
    x = np.array([[r["q"][j] - log["q_star"] for j in range(log["n"])] for r in log["rounds"]])
    nb = x[:, sorted(adj[i])].mean(axis=1)
    y = x[1:, i]
    X = np.column_stack([nb[:-1], np.full(len(y), x[0, i])])
    if np.linalg.matrix_rank(X) < 2:
        return float("nan"), float("nan")
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(coef[0]), float(coef[1])


# ------------------------------------------------------------------ N5  mixing time
def degroot_matrix(adj, lazy=0.5):
    g = A(adj)
    P = g / g.sum(axis=1, keepdims=True)
    return lazy * np.eye(len(g)) + (1 - lazy) * P


def mixing_time(adj, eps=0.05, lazy=0.5):
    """Rounds until disagreement shrinks to eps: ln(1/eps)/ln(1/|lambda_2|) for the lazy DeGroot matrix. inf if disconnected."""
    ev = np.sort(np.abs(np.linalg.eigvals(degroot_matrix(adj, lazy))))[::-1]
    l2 = ev[1]
    return float("inf") if l2 >= 1 - 1e-9 else float(np.log(1 / eps) / np.log(1 / l2))


# ------------------------------------------------------------------ one table per web
def predict(topology, entry):
    adj = webs.TOPOLOGIES[topology]
    kp, c = key_player(adj)
    seeds = webs.entry_nodes(topology, entry) or list(adj)
    return dict(key_player=kp, kp_share=float(c[kp] / c.sum()), lambda1=lam1(A(adj)), mix_rounds=mixing_time(adj),
                theta_star=critical_theta(adj, seeds), entry=seeds)
