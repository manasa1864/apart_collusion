"""Tests for the paper-derived network formulas. Run: python -m tests.test_netformulas"""
import numpy as np

from cartel import netformulas as N, webs, bots, env
from tests.test_metrics import synthetic_log


def test_intercentrality_equals_drop_in_total_centrality():
    for top in webs.TOPOLOGIES:
        g = N.A(webs.TOPOLOGIES[top])
        d = 0.5 / N.lam1(g)
        b, Mx, _ = N.bonacich(g)
        c = N.intercentrality(g)
        for i in range(5):
            k = [j for j in range(5) if j != i]
            bh = np.linalg.inv(np.eye(4) - d * g[np.ix_(k, k)]) @ np.ones(4)
            assert abs((b.sum() - bh.sum()) - c[i]) < 1e-9


def test_key_players():
    assert N.key_player(webs.TOPOLOGIES["star"])[0] == 0
    assert N.key_player(webs.TOPOLOGIES["chain"])[0] == 2
    assert N.key_player(webs.TOPOLOGIES["two_camps"])[0] == 2          # the broker on the camps' bridge side


def test_spectral_radius():
    assert abs(N.lam1(N.A(webs.TOPOLOGIES["complete"])) - 4.0) < 1e-9
    assert abs(N.lam1(N.A(webs.TOPOLOGIES["star"])) - 2.0) < 1e-9
    assert abs(N.lam1(N.A(webs.TOPOLOGIES["ring"])) - 2.0) < 1e-9
    r = N.spreading_ratio(webs.TOPOLOGIES["complete"], beta=0.1, delta=0.3)
    assert abs(r["ratio"] - 4 * 0.1 / 0.3) < 1e-9


def test_lt_cascade():
    star = webs.TOPOLOGIES["star"]
    assert len(N.lt_cascade(star, [4], 0.25)[0]) == 5                   # one spoke gives the hub 1/4 >= 0.25, hub then flips every spoke
    assert N.lt_cascade(star, [4], 0.5)[0] == {4}                       # 1/4 < 0.5: nothing spreads
    assert N.critical_theta(star, [4]) == 0.25


def test_granovetter_classic():
    full = [i / 10 for i in range(10)]                                   # thresholds 0, .1, ..., .9: the riot cascades to everyone
    assert N.granovetter_fixed_point(full) == 1.0
    stuck = [0.0, 0.2] + [i / 10 for i in range(2, 10)]                  # nobody has threshold .1: it stalls at the first instigator
    assert abs(N.granovetter_fixed_point(stuck) - 0.1) < 1e-9


def test_empirical_thresholds():
    q = np.full((8, 5), 2.0); q[1:, 0] = 4.0; q[3:, 1] = 4.0           # star: hub 0 deviates t=1, spoke 1 at t=3
    log = synthetic_log(q); log["graph"] = {i: sorted(v) for i, v in webs.TOPOLOGIES["star"].items()}
    th = sorted(N.empirical_thresholds(log))
    assert abs(th[0] - 0.0) < 1e-9 and abs(th[1] - 1.0) < 1e-9          # hub had no deviating neighbours; spoke's only neighbour (hub) had


def test_fj_fit_recovers_known_parameters():
    rng = np.random.default_rng(2)
    n_t, a, b = 400, 0.6, 0.3
    q = np.full((n_t, 5), 2.0)
    q[0] = 2.0 + rng.normal(0, 1.0, 5)
    for t in range(1, n_t):
        q[t, 1:] = 2.0 + rng.normal(0, 1.0, 4)                          # neighbours move freely (complete graph, agent 0 follows them)
        nb = (q[t - 1, 1:] - 2.0).mean()
        q[t, 0] = 2.0 + a * nb + b * (q[0, 0] - 2.0)
    log = synthetic_log(q)
    ah, bh = N.fj_fit(log, 0)
    assert abs(ah - a) < 1e-6 and abs(bh - b) < 1e-6


def test_mixing_time_ordering():
    t = {k: N.mixing_time(webs.TOPOLOGIES[k]) for k in webs.TOPOLOGIES}
    assert t["complete"] < t["star"] < t["chain"] and t["complete"] < t["ring"] < t["chain"]


def test_predict_runs_for_all_named_webs():
    for name, (top, ent, sh, _) in webs.NAMED.items():
        p = N.predict(top, ent)
        assert 0 <= p["theta_star"] <= 1 and p["mix_rounds"] > 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for f in fns:
        f(); print("ok  ", f.__name__)
    print(f"\n{len(fns)} tests passed")
