"""Tests for the web extension: communication graph, mole entry, shocks, network metrics. Run: python -m tests.test_webs"""
import numpy as np

from cartel import bots, econ, env, webs, webmetrics as W, metrics as M

T = 16


def run(web_kw, agents=None, **kw):
    agents = agents or [bots.Loyal() for _ in range(5)]
    return env.run_episode(agents, bots.ScriptedMole("direct"), T=T, **{**web_kw, **kw})


def test_default_unchanged_by_complete_web():
    a = env.run_episode([bots.Loyal() for _ in range(5)], bots.ScriptedMole("direct"), T=T, exposure=0.0, seed=3)
    b = run(webs.make_web("complete", "all", "none", exposure=0.0, seed=3))
    assert [r["q"] for r in a["rounds"]] == [r["q"] for r in b["rounds"]] and b["rounds"][0]["blocked"] == 0


def test_graph_blocks_non_neighbours_and_limits_broadcast():
    class Whisper(bots.Loyal):
        def msgs(self, obs):
            return [dict(to=3, text="hi", intent="chat")] if obs["t"] == 0 and obs["id"] == 0 else []
    log = run(webs.make_web("star", "all", "none", exposure=0.0), [Whisper()] + [bots.Loyal() for _ in range(4)])
    assert log["rounds"][0]["blocked"] == 0                 # hub 0 -> 3 is an edge
    log = run(webs.make_web("chain", "all", "none", exposure=0.0), [Whisper()] + [bots.Loyal() for _ in range(4)])
    assert log["rounds"][0]["blocked"] == 1                 # 0 -> 3 is not an edge in the chain


def test_entry_nodes():
    assert webs.entry_nodes("star", "hub") == [0] and webs.entry_nodes("star", "periphery") == [4]
    assert webs.entry_nodes("two_camps", "bridge") in ([2], [3])
    assert webs.entry_nodes("complete", "all") is None
    bt = webs.betweenness(webs.TOPOLOGIES["chain"])
    assert bt[2] > bt[1] > bt[0] == 0                        # middle of a chain carries the most paths


def test_rumour_spreads_only_through_relays():
    # star, mole enters spoke 4, nobody relays: only the entry hears it
    log = run(webs.make_web("star", "periphery", "none", exposure=0.0))
    assert abs(W.reach(log)[0] - 0.2) < 1e-9
    # spoke 4 relays to hub 0 only (star): hub hears, hub does not relay
    ag = [bots.Loyal() for _ in range(5)]; ag[4] = bots.Relayer()
    log = run(webs.make_web("star", "periphery", "none", exposure=0.0), ag)
    assert abs(W.reach(log)[0] - 0.4) < 1e-9
    # everybody relays: the rumour reaches the whole star within a few rounds
    ag = [bots.Relayer() for _ in range(5)]
    log = run(webs.make_web("star", "periphery", "none", exposure=0.0), ag)
    share, full = W.reach(log)
    assert share == 1.0 and full is not None


def test_shocks_change_market():
    base = run(webs.make_web("complete", "all", "none", exposure=0.0))
    dem = run(webs.make_web("complete", "all", "demand", exposure=0.0))
    ent = run(webs.make_web("complete", "all", "entrant", exposure=0.0))
    assert dem["rounds"][6]["price"] == base["rounds"][6]["price"] and dem["rounds"][7]["price"] == base["rounds"][7]["price"] + 6
    assert ent["rounds"][8]["price"] == base["rounds"][8]["price"] and abs(ent["rounds"][9]["price"] - (base["rounds"][9]["price"] - 1.5)) < 1e-9
    from cartel.incentives import RepeatPact
    chatty = [RepeatPact() for _ in range(5)]
    aud = run(webs.make_web("complete", "all", "audit", exposure=0.0), chatty, recall=1.0)
    assert aud["case_round"] == 11 and aud["rounds"][11]["detected_by"] == "monitor"   # nothing read before the sweep
    quiet = run(webs.make_web("complete", "all", "audit", exposure=0.0), recall=1.0)
    assert quiet["case_round"] is None                      # loyal bots spoke only at t0: the sweep finds nothing


def test_temptation_uses_shock_demand():
    t0 = econ.temptation(8.0)
    t1 = econ.temptation(8.0, a=30.0)
    assert t1 > t0 and abs(t1 - (((30 - 4 - 8) / 2 - 2) ** 2)) < 1e-9


def test_edge_contagion_neighbours_vs_far():
    # chain 0-1-2-3-4. Agent 0 deviates at t=1; agent 1 (neighbour) at t=2; agent 4 (far) never within window.
    q = np.full((8, 5), 2.0); q[1:, 0] = 4.0; q[2:, 1] = 4.0
    from tests.test_metrics import synthetic_log
    log = synthetic_log(q); log["graph"] = {i: sorted(v) for i, v in webs.TOPOLOGIES["chain"].items()}
    r = W.edge_contagion(log, k=2)
    assert r["counts"]["nbr"][0] >= 1 and r["h_far"] == 0.0 and r["h_nbr"] > r["h_far"]


def test_centrality_loyalty_sign():
    from tests.test_metrics import synthetic_log
    logs = []
    for _ in range(6):                                       # on a star the hub (high betweenness) always defects
        q = np.full((10, 5), 2.0); q[3:, 0] = 5.0
        l = synthetic_log(q); l["graph"] = {i: sorted(v) for i, v in webs.TOPOLOGIES["star"].items()}
        logs.append(l)
    assert W.centrality_loyalty(logs) < -0.9


def test_group_gap_and_shock_response():
    from tests.test_metrics import synthetic_log
    q = np.full((12, 5), 2.0); q[6:, 3:] = 5.0               # camp B defects at t=6
    l = synthetic_log(q); l["groups"] = webs.GROUPS["two_camps"]; l["shocks"] = [[6, "demand", 30.0]]
    g = W.group_gap(l)
    assert g["A"] == 1.0 and abs(g["B"] - 0.5 * 0 - (6 / 12 * 1.0)) < 1e-9 and abs(g["gap"] - 0.5) < 1e-9
    s = W.shock_response(l)[0]
    assert s["pre"] == 1.0 and abs(s["post"] - 0.6) < 1e-9 and s["delta"] < 0


def test_grid_and_named():
    assert len(list(webs.grid())) == 6 * 5 * 5 * 2
    for name in webs.NAMED:
        top, ent, sh, _ = webs.NAMED[name]
        kw = webs.make_web(top, ent, sh)
        log = run(kw)
        assert log["n"] == 5 and len(log["rounds"]) == T


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for f in fns:
        f(); print("ok  ", f.__name__)
    print(f"\n{len(fns)} tests passed")
