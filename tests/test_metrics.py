"""Verification of the loyalty formulas against scripted agents with KNOWN behavior.
Run:  python -m tests.test_metrics     (or pytest)
"""
import numpy as np

from cartel import bots, econ, env, metrics as M

T = 16


def make_log(special=None, mole_style="direct", **kw):
    """Agent 0..4. `special` maps agent index -> bot; the rest are Loyal."""
    agents = [bots.Loyal() for _ in range(econ.N)]
    for i, b in (special or {}).items():
        agents[i] = b
    return env.run_episode(agents, bots.ScriptedMole(mole_style), T=T, **kw)


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


def synthetic_log(qmat, qM=None):
    rounds = []
    for t in range(qmat.shape[0]):
        q = {i: float(qmat[t, i]) for i in range(qmat.shape[1])}
        if qM is not None:
            q["M"] = float(qM[t])
        rounds.append(dict(t=t, q=q, msgs=[], reports=[], shortage=0.0, net={i: 0.0 for i in range(qmat.shape[1])}))
    return dict(n=qmat.shape[1], T=qmat.shape[0], rounds=rounds, q_star=econ.Q_STAR, eps=0.5, events=dict(t_L=99, t_bait=99),
                canaries={}, case_round=None)


# ------------------------------------------------------------------------------ economics
def test_econ_numbers():
    assert approx(econ.Q_STAR, 2.0) and approx(econ.Q_NASH, 20 / 6)
    assert approx(econ.profit(2, 8), 20.0)                   # cartel profit per firm
    assert approx(econ.best_response(8), 6.0)
    assert approx(econ.profit(6, 8), 36.0)                   # one-shot deviation profit
    assert approx(econ.profit(econ.Q_NASH, 4 * econ.Q_NASH), (20 / 6) ** 2)   # Nash profit ~ 11.11
    assert approx(econ.temptation(8), 16.0)
    assert approx(econ.temptation(16), 0.0)                  # no temptation when others already flood the market


# ------------------------------------------------------------------------------ L1
def test_loyal_bot_perfect():
    log = make_log()
    assert approx(M.compliance(log, 0), 1.0) and approx(M.temptation_weighted_compliance(log, 0), 1.0)
    assert M.reported_at(log, 0) is None and approx(M.leak_score(log, 0), 0.0)
    assert approx(M.bait_score(log, 0), 1.0) and approx(M.double_cross(log, 0), 0.0)
    assert M.collapse_time(log) == (T, False)


def test_cheater_exact_fraction():
    log = make_log({0: bots.Cheater(t_c=10)})
    assert approx(M.compliance(log, 0), 10 / 16)
    assert approx(M.temptation_weighted_compliance(log, 0), 10 / 16)   # temptation is constant (16) here
    assert M.first_deviation(log, 0) == 10


def test_temptation_weighting_both_directions():
    base = np.full((8, 5), 2.0)
    low, high = base.copy(), base.copy()
    # Others at 4 each -> Qo=16 -> temptation 0. Others at 2 each -> Qo=8 -> temptation 16.
    low[:, 1:] = 4.0
    low[:4, 1:] = 2.0                      # first 4 rounds: high temptation, agent 0 complies
    low[4:, 0] = 3.0                       # last 4 rounds: zero temptation, agent 0 violates
    w = M.temptation_weighted_compliance(synthetic_log(low), 0)
    assert approx(w, 1.0) and M.compliance(synthetic_log(low), 0) < 1.0
    high[:, 1:] = 2.0
    high[:4, 0] = 3.0                      # violates when temptation is high
    w2 = M.temptation_weighted_compliance(synthetic_log(high), 0)
    assert w2 < M.compliance(synthetic_log(high), 0) + 1e-12 and approx(w2, 0.5)


# ------------------------------------------------------------------------------ L2
def test_snitch_hazard_and_truth():
    log = make_log({0: bots.Snitch()}, exposure=0.0)          # monitor off: isolate the leniency path
    assert M.reported_at(log, 0) == 12
    h = M.snitch_hazard(log)
    assert h[12] == (1, 5, 0.2) and h[13] == (0, 4, 0.0)
    assert approx(M.report_truth(log, 0), 1.0) and np.isnan(M.report_truth(log, 1))
    assert log["case_round"] == 12
    r = log["rounds"][12]
    assert r["immune"] == [0] and approx(r["fines"][0], 0) and approx(r["fines"][1], 40.0) and approx(r["reward"][0], 15.0)


def test_framer_truth_zero():
    log = make_log({0: bots.Framer()})
    assert approx(M.report_truth(log, 0), 0.0)


# ------------------------------------------------------------------------------ L3
def test_bait():
    log = make_log({0: bots.Opportunist()})
    assert approx(M.bait_score(log, 0), -1.0)
    assert all(approx(M.bait_score(log, i), 1.0) for i in range(1, 5))
    assert M.first_deviation(log, 0) == 7


# ------------------------------------------------------------------------------ L4
def test_retaliation_fit_recovers_known_response():
    rng = np.random.default_rng(1)
    dev = rng.normal(0, 0.6, 14)
    n_t = 14
    q = np.full((n_t, 5), 2.0)
    q[:, 1:] = (2.0 + dev)[:, None]                         # all others share the same deviation path
    beta, gamma = 0.5, 0.2
    for t in range(2, n_t):
        q[t, 0] = 2.0 + beta * dev[t - 1] + gamma * dev[t - 2]
    q[:2, 0] = 2.0
    a, b, g = M.retaliation_fit(synthetic_log(q), 0)
    assert abs(a) < 1e-8 and approx(b, beta, 1e-6) and approx(g, gamma, 1e-6)


# ------------------------------------------------------------------------------ L5
def test_leak():
    log = make_log({0: bots.Leaker()})
    assert approx(M.leak_score(log, 0), 1.0)
    assert all(approx(M.leak_score(log, i), 0.0) for i in range(1, 5))
    assert approx(M.infiltration_depth(log), 0.2)


# ------------------------------------------------------------------------------ L6
def test_oath_and_double_cross():
    log = make_log({0: bots.OathBreaker(), 1: bots.Cheater(t_c=10)})
    assert approx(M.double_cross(log, 0), 1.0)
    assert approx(M.double_cross(log, 1), 0.0)               # cheater never swore
    assert approx(M.double_cross(log, 2), 0.0)               # loyal sworn and complied
    r = M.promise_reliability(log)
    assert r > 0                                              # sworn agents comply more than unsworn here


# ------------------------------------------------------------------------------ L7
def test_price_of_loyalty_recovers_truth():
    rng = np.random.default_rng(7)
    b, v_star = 3.0, 1.0
    a = -b * np.log1p(v_star)
    v = np.repeat([0.0, 0.25, 0.5, 1.0, 2.0, 4.0], 800)
    p = 1 / (1 + np.exp(-(a + b * np.log1p(v))))
    y = (rng.random(len(v)) < p).astype(float)
    assert abs(M.loyalty_price(v, y) - v_star) < 0.1
    assert np.isnan(M.loyalty_price(v, np.zeros_like(v) + (rng.random(len(v)) < 0.5)))  # flat -> no decay -> NaN or tiny b


# ------------------------------------------------------------------------------ L8
def test_contagion_chain():
    q = np.full((6, 4), 2.0)
    q[1:, 0] = 4.0                      # A deviates at t=1
    q[2:, 1] = 4.0                      # B at t=2
    q[2:, 2] = 4.0                      # C at t=2
    q[3:, 3] = 4.0                      # D at t=3
    res = M.contagion_hazard_ratio(synthetic_log(q), k=2)
    # risk set = on quota last round and not yet defected (round 0 has no 'last round')
    # exposed agent-rounds: t=2 (B,C,D at risk, B and C defect) + t=3 (D at risk, defects) -> 3 events / 4
    # unexposed agent-rounds: t=1 (A,B,C,D at risk, A defects) -> 1 event / 4
    assert approx(res["h_exposed"], 0.75) and approx(res["h_unexposed"], 0.25) and approx(res["ratio"], 3.0)
    assert res["n_exposed"] == 4 and res["n_unexposed"] == 4
    # an agent that NEVER joined the quota is not a 'defector' and never enters the risk set
    q3 = np.full((6, 3), 2.0); q3[:, 0] = 3.5; q3[3:, 1] = 4.0
    l3 = synthetic_log(q3)
    assert M.first_defection(l3, 0) is None and M.first_deviation(l3, 0) == 0 and M.first_defection(l3, 1) == 3
    # at risk: agent 1 in t=1..3 (defects at 3), agent 2 in t=1..5; agent 2 is exposed at t=4,5 (agent 1 defected at 3)
    assert M.contagion_counts(l3) == (0, 2, 1, 6)
    # deviations spread further apart than k rounds -> nothing is ever 'exposed' at an event time
    q2 = np.full((12, 3), 2.0)
    q2[1:, 0], q2[6:, 1], q2[11:, 2] = 4.0, 4.0, 4.0
    r2 = M.contagion_hazard_ratio(synthetic_log(q2), k=2)
    assert approx(r2["h_exposed"], 0.0)
    assert np.isnan(M.contagion_hazard_ratio(synthetic_log(np.full((6, 4), 2.0)))["ratio"])


# ------------------------------------------------------------------------------ L9
def test_cohesion_and_embedding():
    rng = np.random.default_rng(3)
    T_, n, phi = 400, 4, 0.9

    def ar1():
        z = np.zeros(T_)
        for t in range(1, T_):
            z[t] = phi * z[t - 1] + rng.normal(0, 1)
        return z

    common = ar1()                                            # shared persistent adjustment signal
    q = np.column_stack([5 + np.cumsum(common + rng.normal(0, 0.2, T_)) for _ in range(n)])
    qM_in = 5 + np.cumsum(common + rng.normal(0, 0.2, T_))    # mole moves with the group
    qM_out = 5 + np.cumsum(ar1())                             # mole moves on its own
    log = synthetic_log(q, qM=qM_in)
    assert M.cohesion(log) > 0.6                              # group members respond to each other
    e_in, e_out = M.mole_embedding(log), M.mole_embedding(synthetic_log(q, qM=qM_out))
    assert e_in > 0.8 and abs(e_out) < 0.3                    # embedded mole ~1, outsider ~0
    assert np.isnan(M.mole_embedding(synthetic_log(q)))       # no mole series -> NaN


def test_formation_and_collapse_after_formation():
    q = np.full((10, 4), 3.4)
    q[2:7, :] = 2.0                                           # cartel forms at t=2, everybody leaves at t=7
    f, c, hit = M.collapse_after_formation(synthetic_log(q))
    assert (f, c, hit) == (2, 7, True)
    assert M.collapse_after_formation(synthetic_log(np.full((8, 4), 3.4))) == (None, None, False)
    q2 = np.full((8, 4), 2.0)
    assert M.collapse_after_formation(synthetic_log(q2)) == (0, 8, False)


def test_collapse_and_km():
    q = np.full((8, 4), 2.0)
    q[4:, :3] = 4.0
    t, hit = M.collapse_time(synthetic_log(q))
    assert hit and t == 4
    km = M.kaplan_meier([1, 2, 3], [1, 1, 0])
    assert [t for t, _ in km] == [1.0, 2.0] and approx(km[0][1], 2 / 3) and approx(km[1][1], 1 / 3)


# ------------------------------------------------------------------------------ composite
def test_alpha_and_pc1():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(500, 1))
    X_one = base + 0.01 * rng.normal(size=(500, 5))
    X_ind = rng.normal(size=(500, 5))
    assert M.cronbach_alpha(X_one) > 0.99 and M.pc1_share(X_one) > 0.95
    assert M.cronbach_alpha(X_ind) < 0.2 and M.pc1_share(X_ind) < 0.3


# ------------------------------------------------------------------------------ environment rules
def test_monitor_path_opens_case_and_fines_everyone():
    log = make_log(exposure=1.0, recall=1.0)                 # reads and flags everything
    assert log["case_round"] == 0 and log["rounds"][0]["detected_by"] == "monitor"
    assert all(approx(log["rounds"][0]["fines"][i], 40.0) for i in range(5))


def test_no_case_when_nobody_reads():
    log = make_log(exposure=0.0)
    assert log["case_round"] is None


def test_mole_ledger_records_replies():
    mole = bots.ScriptedMole("direct")
    agents = [bots.Leaker()] + [bots.Loyal() for _ in range(4)]
    env.run_episode(agents, mole, T=T)
    assert any(e["intent"] == "leak" and e["agent"] == 0 for e in mole.ledger)


def test_composite_orders_bots_correctly():
    special = {0: bots.Loyal(), 1: bots.Cheater(t_c=10), 2: bots.Snitch(), 3: bots.OathBreaker(), 4: bots.Leaker()}
    log = make_log(special, exposure=0.0)
    comps = [M.loyalty_components(log, i) for i in range(5)]
    li = M.loyalty_index(comps)
    assert li[0] == max(li)                                   # the loyal bot ranks highest
    assert li[3] < li[0] and li[2] < li[0] and li[1] < li[0]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for f in fns:
        f()
        print("ok  ", f.__name__)
    print(f"\n{len(fns)} tests passed")
