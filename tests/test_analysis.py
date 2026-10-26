"""Tests for the new mechanics (leniency switch, outside option, expulsion), the statistics helpers, the synthetic agent and the
analysis pipeline end to end. Run: python -m tests.test_analysis"""
import csv
import os
import tempfile
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from cartel import bots, econ, env, stats as S


# ------------------------------------------------------------------ environment additions
def test_leniency_off_means_no_reports_accepted():
    log = env.run_episode([bots.Snitch()] + [bots.Loyal() for _ in range(4)], bots.ScriptedMole("direct"), T=16, exposure=0.0, leniency=False)
    assert log["case_round"] is None and all(not r["reports"] for r in log["rounds"]) and log["leniency_on"] is False


def test_outside_option_pays_bonus_only_at_competitive_quantity():
    class Taker(bots.Loyal):
        def q(self, obs):
            return econ.Q_NASH if obs.get("outside_offer") else econ.Q_STAR
    log = env.run_episode([Taker()] + [bots.Loyal() for _ in range(4)], bots.ScriptedMole("none"), T=10, exposure=0.0, outside_v=1.0)
    to = log["events"]["t_offer"]
    assert log["rounds"][to - 1]["bonus"][0] == 0.0 and abs(log["rounds"][to]["bonus"][0] - econ.DPI) < 1e-9
    assert log["rounds"][to]["bonus"][1] == 0.0                       # loyal members on quota get nothing
    assert abs(econ.DPI - (20 - (20 / 6) ** 2)) < 1e-9


def test_majority_vote_expels_mole_and_cuts_messages():
    class Voter(bots.Loyal):
        def act(self, obs):
            a = super().act(obs)
            if obs["t"] >= 4:
                a["vote_expel"] = "M"
            return a
    mole = bots.ScriptedMole("direct")
    log = env.run_episode([Voter(), Voter(), Voter(), bots.Loyal(), bots.Leaker()], mole, T=16, exposure=0.0)
    assert log["rounds"][4]["expelled_now"] == ["M"] and "M" in log["rounds"][-1]["expelled"]
    assert not any(m["frm"] == "M" for r in log["rounds"][5:] for m in r["msgs"])      # mole silenced after expulsion
    from cartel import metrics as M
    assert M.leak_score(log, 4) == 0.0                                                  # the probe never reached the leaker


def test_minority_vote_does_not_expel():
    class Voter(bots.Loyal):
        def act(self, obs):
            return dict(super().act(obs), vote_expel=1)
    log = env.run_episode([Voter(), Voter()] + [bots.Loyal() for _ in range(3)], bots.ScriptedMole("none"), T=6, exposure=0.0)
    assert log["rounds"][-1]["expelled"] == []


def test_no_mole_cell_has_no_mole_messages():
    log = env.run_episode([bots.Loyal() for _ in range(5)], bots.ScriptedMole("none"), T=16, exposure=0.0)
    assert not any(m["frm"] == "M" for r in log["rounds"] for m in r["msgs"])


# ------------------------------------------------------------------ statistics
def test_holm():
    adj = S.holm([0.01, 0.04, 0.03, np.nan])
    assert np.allclose(adj[:3], [0.03, 0.06, 0.06]) and np.isnan(adj[3])


def test_diff_test_and_mean_ci():
    r = np.random.default_rng(0)
    a, b = r.normal(1, 1, 200), r.normal(0, 1, 200)
    d = S.diff_test(a, b)
    assert d["p"] < 0.01 and d["lo"] < 1 < d["hi"]
    m, lo, hi, n = S.mean_ci(np.ones(10))
    assert m == lo == hi == 1 and n == 10


def test_km_and_logrank():
    t, s, lo, hi = S.km([1, 2, 3, 4], [1, 1, 0, 1])
    assert np.allclose(s, [1, 0.75, 0.5, 0.0])
    chi2, df, p = S.logrank([1, 2, 3, 4, 10, 11, 12, 13], [1] * 8, ["a"] * 4 + ["b"] * 4)
    assert df == 1 and p < 0.01
    _, _, p_same = S.logrank([1, 2, 3, 4, 1, 2, 3, 4], [1] * 8, ["a"] * 4 + ["b"] * 4)
    assert p_same > 0.9


def test_logistic_recovers_parameters():
    r = np.random.default_rng(1)
    x = r.normal(size=20000)
    y = (r.random(20000) < 1 / (1 + np.exp(-(-0.5 + 2 * x)))).astype(float)
    a, b, sa, sb, _ = S.logistic(x, y)
    assert abs(a + 0.5) < 0.06 and abs(b - 2) < 0.08


def test_kappa_and_wilson():
    k, labs, cm = S.cohen_kappa(list("aabb"), list("aabb"))
    assert k == 1.0 and cm.trace() == 4
    k2, _, _ = S.cohen_kappa(list("aaaa" "bbbb"), list("abab" "abab"))
    assert abs(k2) < 1e-9
    p, lo, hi = S.wilson(5, 10)
    assert p == 0.5 and 0.2 < lo < 0.3 and 0.7 < hi < 0.8


def test_cluster_bootstrap_resamples_episodes():
    groups = [np.full(5, i % 2, float) for i in range(40)]
    est, lo, hi, bs = S.cluster_bootstrap(groups, lambda G: np.concatenate(G).mean(), B=300)
    assert abs(est - 0.5) < 1e-9 and lo < 0.5 < hi and hi - lo > 0.15     # width reflects 40 clusters, not 200 agents


# ------------------------------------------------------------------ synthetic agent + full pipeline
def _sim(tmp, seeds=6, design="all"):
    import run_cartel
    a = run_cartel.parser().parse_args(["--backend", "sim", "--design", design, "--webs", "all", "--seeds", str(seeds), "--out", tmp])
    a.reasoning_effort = None
    assert run_cartel.run(a) == "done"
    return os.path.join(tmp, "results.jsonl")


def test_plan_covers_every_planned_cell():
    import run_cartel
    a = run_cartel.parser().parse_args(["--design", "all", "--webs", "all", "--seeds", "1"])
    plan = run_cartel.episode_plan(a)
    cells = {p["cell"] for p in plan if p["design"] == "cells"}
    assert cells == {"S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "PC"}
    assert {p["v"] for p in plan if p["design"] == "dose"} == {0.0, 0.25, 0.5, 1.0, 2.0, 4.0}
    assert len({p["web"] for p in plan if p["design"] == "webs"}) == 10
    s0 = next(p for p in plan if p["cell"] == "S0")
    assert s0["mole"] == "none" and s0["leniency"] is False
    assert len({p["episode_id"] for p in plan}) == len(plan)                # ids unique


def test_sim_agent_is_deterministic():
    from cartel.simagent import SimAgent
    adj = {i: set(range(5)) - {i} for i in range(5)}
    l1 = env.run_episode([SimAgent(i, adj, "e") for i in range(5)], bots.ScriptedMole("direct"), T=16, seed=3)
    l2 = env.run_episode([SimAgent(i, adj, "e") for i in range(5)], bots.ScriptedMole("direct"), T=16, seed=3)
    assert [r["q"] for r in l1["rounds"]] == [r["q"] for r in l2["rounds"]]


def test_pipeline_end_to_end_on_synthetic_data():
    from cartel import analysis as A
    tmp = tempfile.mkdtemp()
    D = A.build(A.load(_sim(tmp)))
    assert D["src"] == "ground_truth" and len(D["ep"]) == 6 * 35
    H = A.test_all(D)
    assert set(["H1", "H2", "H3a", "H3b", "H4", "H5", "H6", "H7", "H8", "H9", "H10", "H11", "H12", "H13", "H14", "H15", "H16"]) <= set(H.H)
    assert not (H.claim == "error").any(), H[H.claim == "error"].note.tolist()
    assert H.loc[H.H == "H1b", "estimate"].iloc[0] > 0.2                       # planted: direct offers recruit more
    assert H.loc[H.H == "H3b", "estimate"].iloc[0] > 0.3                       # planted: coded talk when told watched


def test_report_runs_and_writes_everything():
    import subprocess
    import sys
    tmp = tempfile.mkdtemp()
    res = _sim(tmp, seeds=4)
    out = os.path.join(tmp, "an")
    r = subprocess.run([sys.executable, "analyze_cartel.py", "--results", res, "--out", out, "--no-power"], capture_output=True, text=True,
                       cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert r.returncode == 0, r.stderr[-2000:]
    figs = [f for f in os.listdir(os.path.join(out, "figures")) if f.endswith(".png")]
    assert len(figs) >= 17, figs
    for t in ("T1_hypotheses", "T2_cells", "T3_webs", "T4_loyalty", "T5_data_quality", "T6_recovery"):
        for ext in ("csv", "md", "tex"):
            assert os.path.exists(os.path.join(out, "tables", f"{t}.{ext}")), (t, ext)
    rep = open(os.path.join(out, "report.md"), encoding="utf-8").read()
    assert "SYNTHETIC DATA" in rep and "F05_price_of_loyalty" in rep


def test_judge_sample_and_agreement():
    from cartel import judge
    tmp = tempfile.mkdtemp()
    res = _sim(tmp, seeds=1, design="cells")
    hand = os.path.join(tmp, "hand.csv")
    judge.sample(res, None, 20, hand)
    rows = list(csv.DictReader(open(hand, encoding="utf-8-sig")))
    assert len(rows) == 20 and len({r["keyword"] for r in rows}) >= 3           # stratified: several label types
    for r in rows:
        r["human"] = r["keyword"]
    with open(hand, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    out = judge.agree(hand)
    assert out["keyword"]["kappa"] == 1.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for f in fns:
        f(); print("ok  ", f.__name__)
    print(f"\n{len(fns)} tests passed")
