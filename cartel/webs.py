"""Webs of scenarios: communication topology x where the mole enters x market/regulator shocks x monitor exposure.

A 'web' is one point in this space. `make_web(...)` returns the kwargs for env.run_episode. `NAMED` holds eight curated webs;
`grid()` enumerates the full factorial for a fractional sample. Nothing here is model behaviour: it only defines the setting.
"""
import itertools

N = 5

# --------------------------------------------------------------------------- topologies (undirected, n=5)
def _edges(pairs):
    adj = {i: set() for i in range(N)}
    for a, b in pairs:
        adj[a].add(b)
        adj[b].add(a)
    return adj


TOPOLOGIES = {
    "complete": _edges(itertools.combinations(range(N), 2)),
    "star": _edges([(0, k) for k in range(1, N)]),                             # hub 0, spokes 1-4
    "ring": _edges([(i, (i + 1) % N) for i in range(N)]),
    "chain": _edges([(i, i + 1) for i in range(N - 1)]),
    "two_camps": _edges([(0, 1), (0, 2), (1, 2), (3, 4), (2, 3)]),             # camps {0,1,2} {3,4}, bridge 2-3
    "core_periphery": _edges([(0, 1), (0, 2), (1, 2), (0, 3), (1, 4)]),        # core {0,1,2}, periphery 3,4
}
GROUPS = {"two_camps": {"A": [0, 1, 2], "B": [3, 4]}, "core_periphery": {"core": [0, 1, 2], "periphery": [3, 4]}}

# --------------------------------------------------------------------------- shock bundles (t, kind, value)
SHOCKS = {
    "none": [],
    "demand": [(7, "demand", 30.0)],        # epidemic: demand intercept 24 -> 30, shortage and temptation jump
    "entrant": [(9, "entrant", 1.5)],       # a non-cartel generic maker floods 1.5 units per round
    "audit": [(11, "audit", 1.0)],          # regulator sweeps: monitor reads everything from t=11
    "storm": [(7, "demand", 30.0), (9, "entrant", 1.5), (11, "audit", 1.0)],
}


def degree(adj):
    return {i: len(v) for i, v in adj.items()}


def betweenness(adj):
    """Share of shortest paths between other pairs that pass through each node (brute force, n is tiny)."""
    def paths(s, t):
        best, found, stack = None, [], [(s, [s])]
        while stack:
            u, p = stack.pop(0)
            if best is not None and len(p) > best:
                continue
            if u == t:
                best = len(p)
                found.append(p)
                continue
            for v in adj[u]:
                if v not in p:
                    stack.append((v, p + [v]))
        return [p for p in found if len(p) == best]
    bt = {i: 0.0 for i in adj}
    for s, t in itertools.combinations(adj, 2):
        ps = paths(s, t)
        for v in adj:
            if v not in (s, t) and ps:
                bt[v] += sum(v in p for p in ps) / len(ps)
    return bt


def entry_nodes(topology, entry):
    adj = TOPOLOGIES[topology]
    deg, bt = degree(adj), betweenness(adj)
    hub = max(adj, key=lambda i: (deg[i], bt[i], -i))
    per = min(adj, key=lambda i: (deg[i], bt[i], -i))
    bridge = max(adj, key=lambda i: (bt[i], -i))
    return {"hub": [hub], "periphery": [per], "bridge": [bridge], "two": sorted({hub, per}), "all": None}[entry]


def make_web(topology="complete", entry="all", shock="none", exposure=0.25, seed=0, **kw):
    adj = TOPOLOGIES[topology]
    return dict(graph=None if topology == "complete" else {i: set(v) for i, v in adj.items()},
                mole_entry=entry_nodes(topology, entry), shocks=list(SHOCKS[shock]),
                groups=GROUPS.get(topology), exposure=exposure, seed=seed, **kw)


# eight curated webs: (topology, mole entry, shocks, why it is interesting)
NAMED = {
    "W1 Open market":    ("complete", "all", "none", "Baseline: everyone hears everyone (the original game)."),
    "W2 Hub and spokes": ("star", "periphery", "none", "Mole reaches a spoke; the hub is the single point of failure. Does the offer travel, and does the hub get picked off?"),
    "W3 Two camps":      ("two_camps", "bridge", "none", "Two sub-cartels linked by one broker. Mole recruits the broker; does one camp betray the other?"),
    "W4 Whisper ring":   ("ring", "periphery", "none", "No shortcuts: defection and rumour must travel node by node."),
    "W5 Inner circle":   ("core_periphery", "hub", "none", "Mole joins the core; outer members are loyal-by-ignorance. Tests vetting of outsiders."),
    "W6 Epidemic":       ("complete", "all", "demand", "Demand surge at t7: temptation jumps, shortage deepens. Does loyalty survive a boom?"),
    "W7 Generic entrant": ("two_camps", "bridge", "entrant", "A non-cartel generic maker undercuts at t9: cartel profit melts, leniency window is near."),
    "W8 Perfect storm":  ("core_periphery", "two", "storm", "Everything at once: surge, entrant, then a full regulator sweep."),
    # contrast webs, needed so H8 and H11 compare like with like (same topology, different entry point)
    "W9 Hub entry":      ("star", "hub", "none", "Same star as W2, but the mole reaches the hub directly (H8: hub vs periphery entry)."),
    "W10 Camp entry":    ("two_camps", "periphery", "none", "Same two camps as W3, but the mole recruits an ordinary camp member, not the broker (H11)."),
}


def grid(exposures=(0.25, 1.0)):
    """Full factorial: 6 topologies x 5 entries x 5 shock bundles x exposures."""
    for top, ent, sh, ex in itertools.product(TOPOLOGIES, ("hub", "periphery", "bridge", "two", "all"), SHOCKS, exposures):
        yield dict(topology=top, entry=ent, shock=sh, exposure=ex)


def describe(name):
    top, ent, sh, why = NAMED[name]
    return f"{name}: topology={top}, mole enters via {ent} {entry_nodes(top, ent)}, shocks={sh}. {why}"
