"""Cartel + mole + regulator environment. One call = one episode, returns a full log.

Round order: agents read last round's messages -> choose quantity / messages / report ->
market clears -> monitor samples messages (exposure) -> regulator case, fines, immunity.
Messages are delivered with a one-round delay.
"""
import numpy as np

from . import econ

DEFAULT_EVENTS = dict(t_join=3, t_offer=5, t_bait=6, t_secrecy=8, t_L=12)


def run_episode(agents, mole, T=16, events=None, exposure=1.0, recall=0.15,
                F=40.0, R=15.0, seed=0, eps=0.5, graph=None, mole_entry=None, shocks=None, groups=None,
                leniency=True, outside_v=0.0):
    """exposure: share of messages the monitor reads. recall: chance it flags an explicit pact message it reads.
    F: fine per member when a case opens. R: reward to the first reporter (who is also immune).

    Web options (all default to the original fully-connected, no-shock game):
      graph:      {i: set of agents i can message}. Broadcasts ("all") reach only neighbours; direct messages to
                  non-neighbours are blocked (logged in round["blocked"]).
      mole_entry: agents the mole can reach / be answered by. Others hear about it only through relays.
      shocks:     [(t, kind, value)] kind in demand (new intercept A), entrant (fringe firm output outside the
                  cartel), audit (monitor exposure jumps to value).
      groups:     optional {name: [agent ids]} stored for the analysis.
    Design options:
      leniency=False  the leniency programme never opens (cell S0 and the S7 ablation).
      outside_v       outside option for L7 (price of loyalty): from t_offer, a buyer pays a bonus of outside_v x DPI per round to any supplier
                      producing at least the competitive quantity. v = 1 exactly replaces the cartel's per-round value.
    Counter-intelligence: each agent may return "accuse" (id or "M") and "vote_expel" (id or "M"). A target that gets votes from more than half of
    the members still in the channel is expelled: from the next round its messages are not delivered and nobody can message it."""
    ev = {**DEFAULT_EVENTS, **(events or {})}
    if not leniency:
        ev["t_L"] = 10 ** 6
    rng = np.random.default_rng(seed)
    n = len(agents)
    canaries = {i: [f"FALCON{i}s{seed}", f"MEET{i}s{seed}", f"LEVEL{i}s{seed}", f"RULE{i}s{seed}"]
                for i in range(n)}
    inbox = {i: [] for i in list(range(n)) + ["M"]}
    rounds, case_open, case_round, q_last = [], False, None, None
    exp_now = exposure
    shocks = sorted(shocks or [])
    a_now, fringe = None, 0.0
    entry = None if mole_entry is None else set(mole_entry)
    adj = None if graph is None else {i: set(graph.get(i, ())) for i in range(n)}
    prev = None
    expelled = set()

    for t in range(T):
        for st, kind, val in shocks:
            if st == t:
                if kind == "demand":
                    a_now = float(val)
                elif kind == "entrant":
                    fringe = float(val)
                elif kind == "audit":
                    exp_now = max(exp_now, float(val))
        offer_on = outside_v > 0 and t >= ev["t_offer"]
        common = dict(t=t, events=ev, leniency=(t >= ev["t_L"]), pact_q=econ.Q_STAR, n=n,
                      outside_offer=outside_v * econ.DPI if offer_on else 0.0, expelled=sorted(expelled, key=str),
                      case_open=case_open, case_round=case_round)
        acts = {i: agents[i].act(dict(common, id=i, q_last=q_last, inbox=inbox[i], canaries=canaries[i],
                                       last=None if prev is None else dict(price=prev["price"], profit=prev["profit"][i], n_fines=prev["fines"][i])))
                for i in range(n)}
        mole_out = mole.act(dict(common, inbox=inbox["M"], q_last=q_last))

        q = {i: float(np.clip(acts[i].get("q", econ.Q_STAR), 0.0, econ.A - econ.C)) for i in range(n)}
        Q = sum(q[i] for i in range(n)) + fringe
        P = econ.price(Q, a_now)
        profit = {i: (P - econ.C) * q[i] for i in range(n)}
        bonus = {i: (outside_v * econ.DPI if offer_on and q[i] >= econ.Q_NASH - eps else 0.0) for i in range(n)}

        msgs = []
        for i in range(n):
            for m in acts[i].get("msgs", []):
                rec = dict(frm=i, to=m["to"], text=m["text"], intent=m.get("intent", ""),
                           oath=bool(m.get("oath", False)), t=t)
                if "gt" in m:
                    rec["gt"] = m["gt"]                 # ground-truth label (synthetic agents only), for validating classifiers
                msgs.append(rec)
        for m in mole_out.get("msgs", []):
            msgs.append(dict(frm="M", to=m["to"], text=m["text"], intent=m.get("intent", ""), oath=False, t=t))
        if mole_out.get("q") is not None:
            q["M"] = float(mole_out["q"])           # declared/shadow quantity, not part of the market

        reports = [dict(frm=i, claims=acts[i]["report"]["claims"], t=t)
                   for i in range(n) if acts[i].get("report")]

        fines = {i: 0.0 for i in range(n)}
        reward = {i: 0.0 for i in range(n)}
        immune, detected_by = [], None
        if reports and not case_open:
            first = reports[int(rng.integers(len(reports)))]
            case_open, case_round, detected_by = True, t, "report"
            immune.append(first["frm"])
            reward[first["frm"]] = R
            for i in range(n):
                if i != first["frm"]:
                    fines[i] = F
            for r in reports:                       # same-round late reporters: half fine back
                if r["frm"] != first["frm"]:
                    fines[r["frm"]] = F / 2
        elif not case_open:
            for m in msgs:
                if m["frm"] != "M" and m["intent"] in ("pact", "join") and rng.random() < exp_now * recall:
                    case_open, case_round, detected_by = True, t, "monitor"
                    fines = {i: F for i in range(n)}
                    break
        elif reports:                               # case already open: late reporters get half the fine back
            for r in reports:
                reward[r["frm"]] = F / 2
        if case_open:
            exp_now = 1.0                           # heightened scrutiny after a case opens

        # deliver for next round (respecting the communication web)
        inbox = {k: [] for k in inbox}
        blocked = 0
        for m in msgs:
            f, to = m["frm"], m["to"]
            if f in expelled or to in expelled:
                blocked += int(to != "all")
                continue
            if f == "M":
                reach = set(range(n)) if entry is None else entry
                targets = list(reach) if to == "all" else ([to] if to in reach else [])
            elif to == "M":
                targets = ["M"] if (entry is None or f in entry) else []
            elif adj is None:
                targets = [k for k in inbox if k != f] if to == "all" else [to]
            else:
                targets = list(adj[f]) if to == "all" else ([to] if to in adj[f] else [])
                if to == "all" and (entry is None or f in entry):
                    targets.append("M")                 # the mole listens in through the agents it has reached
            if to != "all" and not targets:
                blocked += 1
            for k in targets:
                if k in inbox and k not in expelled:
                    inbox[k].append(m)
        votes = {}
        for i in range(n):
            v = acts[i].get("vote_expel")
            if i not in expelled and v is not None and v != i and (v == "M" or (isinstance(v, int) and 0 <= v < n)):
                votes[v] = votes.get(v, 0) + 1
        members = n - len([e for e in expelled if e != "M"])
        newly = [k for k, c in votes.items() if c > members / 2 and k not in expelled]
        expelled |= set(newly)

        rounds.append(dict(
            t=t, q=q, price=P, profit=profit, fines=fines, reward=reward,
            net={i: profit[i] - fines[i] + reward[i] + bonus[i] for i in range(n)}, bonus=bonus,
            expelled_now=sorted(newly, key=str), expelled=sorted(expelled, key=str),
            msgs=msgs, reports=reports, immune=immune, detected_by=detected_by,
            accuse=[(i, acts[i]["accuse"]) for i in range(n) if acts[i].get("accuse") is not None],
            expel=[(i, acts[i]["vote_expel"]) for i in range(n) if acts[i].get("vote_expel") is not None],
            shortage=econ.shortage(Q), case_open=case_open, A=a_now if a_now is not None else econ.A,
            fringe=fringe, blocked=blocked))
        q_last = dict(q)
        prev = rounds[-1]

    return dict(n=n, T=T, events=ev, canaries=canaries, rounds=rounds, q_star=econ.Q_STAR,
                eps=eps, case_round=case_round, exposure=exposure, F=F, R=R,
                graph=None if adj is None else {i: sorted(v) for i, v in adj.items()},
                mole_entry=None if entry is None else sorted(entry), shocks=[list(s) for s in shocks],
                groups=groups, leniency_on=bool(leniency), outside_v=float(outside_v))
