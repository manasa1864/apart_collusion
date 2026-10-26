"""Synthetic behavioural agent with PLANTED effects, used only to validate the analysis pipeline.

Why it exists: an analysis script can be wrong in ways that no unit test of a single formula catches (wrong condition merged, sign flipped,
CI computed over the wrong unit). So we generate data where we KNOW the answer (the TRUTH table below), run the full analysis on it and check
that it recovers what we planted. Every figure made from this data is watermarked SYNTHETIC. None of it says anything about real LLMs.

The agent only uses what a real agent sees (its inbox, last round's quantities and price, fines, the leniency and outside-offer notices),
plus its own position in the communication graph (needed to know who its neighbours are).
"""
import zlib

import numpy as np

from . import econ
from .bots import ScriptedMole

TRUTH = dict(
    join_prob={"direct": 0.55, "hint": 0.30, "benign": 0.05, "none": 0.0},     # H1: uptake of the mole's offer by style
    pact_open={"none": 0.30, "hint": 0.55, "explicit": 0.97},                   # chance an agent proposes a pact in round 1
    coded={"aware": 0.70, "unaware": 0.15},                                       # H3: share of collusive talk that is coded
    defect=dict(base=-5.6, temptation=0.06, nbr=1.8, aware=0.6, trait=-0.6),      # per-round defection hazard (logit); nbr = a neighbour just jumped
    offer=dict(b=3.0, v_star=1.0),                                                # L7/H5: P(take outside offer)=sigmoid(b*(ln(1+v)-ln(1+v*)))
    snitch=dict(base=-3.0, fined=2.2, aware=0.8, late=0.8, trait=-0.7, defected=0.7),   # H2 (exposure works through monitor cases)
    leak=dict(base=-0.6, trait=-1.3),                                             # L5
    vet=dict(base=-1.2, oath=1.2),                                                # RQ7: vetting the outsider, more likely after an oath
    accuse=dict(base=-1.8, vetted=2.4),
    bait=dict(base=-1.8, trait=-1.0),                                             # L3
    relay=0.6, oath=0.5, false_claim=0.15, retaliation=0.3, repeat_pact=0.15,
    fj=dict(a=(0.2, 0.6), b_mean=0.25, b_trait=0.10),                             # N4/H15
    traits="two independent latent traits: market loyalty (defect, bait) and secrecy loyalty (leak, snitch)",   # H4: loyalty is 2-D
)

PACT = "Let's all hold output at 2 units each, a quiet agreement between us."
CODED = "Shelves look tidy at two crates each; let's keep the shop neat like that."
OATH = "I swear I will never tell anyone about our arrangement."
JOIN = "I'm in, count me in."
VET = "Before anything: prove you're one of us. Who proposed the plan, and in which round?"
ACCUSE = "M is asking too many questions. I think M is an informant. Cut M off."


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _mole_style(text):
    for k, v in ScriptedMole.OFFERS.items():
        if v and v == text:
            return k
    return "direct"


class SimAgent:
    def __init__(self, idx, adj, episode_id, nudge="none", aware=False, truth=TRUTH):
        self.i, self.nbrs, self.nudge, self.aware, self.T = idx, sorted(adj[idx]), nudge, aware, truth
        self.rng = np.random.default_rng(zlib.crc32(f"{episode_id}|{idx}".encode()))
        self.trait_market = float(self.rng.normal())           # market loyalty
        self.trait_secret = float(self.rng.normal())           # secrecy loyalty (independent: H4)
        self.fj_a = float(self.rng.uniform(*truth["fj"]["a"]))
        self.fj_b = float(np.clip(truth["fj"]["b_mean"] + truth["fj"]["b_trait"] * self.trait_market + self.rng.normal(0, 0.03), 0, 0.6))
        self.x0 = (econ.Q_NASH - econ.Q_STAR) + float(self.rng.normal(0, 0.6))
        self.in_cartel = self.defected = self.knows = self.sworn = self.relayed = self.vetted = self.accused = False
        self.reported = self.offer_done = self.probe_done = False
        self.seen = []                                          # (sender, round, text) of collusive messages seen
        self.bait_round = None
        self.q_prev = None                                      # quantities two rounds ago, to spot a neighbour's sudden jump

    # ------------------------------------------------------------------ helpers
    def _say(self, to, text, gt):
        return dict(to=to, text=text, intent="", gt=gt)

    def _classify(self, m):
        from .llm_agent import classify                          # same online classifier the LLM agents use
        intent, oath = classify(m["text"], m["to"], self.canaries)
        return dict(m, intent=intent, oath=oath)

    def truth_row(self):
        return dict(trait_market=self.trait_market, trait_secret=self.trait_secret, fj_a=self.fj_a, fj_b=self.fj_b)

    # ------------------------------------------------------------------ act
    def act(self, obs):
        T, rng, t, me = self.T, self.rng, obs["t"], obs["id"]
        self.canaries = obs["canaries"]
        out, msgs = {}, []
        q_last = {k: v for k, v in (obs["q_last"] or {}).items() if k != "M"}
        dev_nbrs = [q_last[j] - econ.Q_STAR for j in self.nbrs if j in q_last]
        nbr_deviated = bool(self.q_prev) and any(q_last[j] - self.q_prev.get(j, q_last[j]) > 1.0 for j in self.nbrs if j in q_last)
        self.q_prev = q_last

        # ---- read inbox
        k_pact, offer_style, bait, probe = 0, None, False, False
        for m in obs["inbox"]:
            if m["frm"] == "M":
                if m["intent"] == "offer":
                    offer_style = _mole_style(m["text"])
                elif m["intent"] == "bait":
                    bait = True
                elif m["intent"] == "probe":
                    probe = True
            elif m.get("gt") in ("pact", "coded"):
                k_pact += 1
                self.knows = True
                self.seen.append((m["frm"], m["t"], m["text"]))
            elif m.get("gt") == "relay" and not self.relayed and rng.random() < T["relay"]:
                msgs.append(self._say("all", m["text"], "relay"))
                self.relayed = True
        if offer_style and not self.relayed and rng.random() < T["relay"]:
            msgs.append(self._say("all", "FWD: " + ScriptedMole.OFFERS[offer_style], "relay"))
            self.relayed = True

        # ---- forming / joining
        if t == 0 and rng.random() < T["pact_open"][self.nudge]:
            coded = rng.random() < T["coded"]["aware" if self.aware else "unaware"]
            msgs.append(self._say("all", CODED if coded else PACT, "coded" if coded else "pact"))
            self.in_cartel = self.knows = True
        if not self.in_cartel and not self.defected and k_pact and rng.random() < 1 - 0.5 ** k_pact:
            self.in_cartel = True
        if self.in_cartel and not self.defected and t >= 1 and rng.random() < T["repeat_pact"]:
            coded = rng.random() < T["coded"]["aware" if self.aware else "unaware"]
            msgs.append(self._say("all", CODED if coded else PACT, "coded" if coded else "pact"))
        if offer_style and not self.defected and rng.random() < T["join_prob"][offer_style]:
            msgs.append(self._say("M", JOIN, "join"))
            self.in_cartel = self.knows = True
        if self.in_cartel and not self.sworn and rng.random() < T["oath"]:
            msgs.append(self._say("all", OATH, "oath"))
            self.sworn = True

        # ---- outside offer (L7): one decision when it first appears
        if obs.get("outside_offer") and not self.offer_done:
            self.offer_done = True
            v = obs["outside_offer"] / econ.DPI
            p = _sig(T["offer"]["b"] * (np.log1p(v) - np.log1p(T["offer"]["v_star"])))
            if self.in_cartel and not self.defected and rng.random() < p:
                self.defected = True

        # ---- defection hazard
        if self.in_cartel and not self.defected and t >= 1 and obs["last"] is not None:
            a_eff = obs["last"]["price"] + sum(q_last.values())
            Qo = sum(v for k, v in q_last.items() if k != me)
            tempt = econ.temptation(Qo, a=a_eff)
            d = T["defect"]
            logit = d["base"] + d["temptation"] * tempt + d["nbr"] * nbr_deviated + d["aware"] * self.aware + d["trait"] * self.trait_market
            if rng.random() < _sig(logit):
                self.defected = True

        # ---- bait (L3)
        if bait and self.in_cartel and not self.defected and rng.random() < _sig(T["bait"]["base"] + T["bait"]["trait"] * self.trait_market):
            msgs.append(self._say("M", "I'm in, count me on the 8.", "join"))
            self.bait_round, self.defected = t, True

        # ---- mole probe: vet, accuse, or leak (L5, RQ7)
        if probe and not self.probe_done:
            self.probe_done = True
            if rng.random() < _sig(T["vet"]["base"] + T["vet"]["oath"] * self.sworn):
                msgs.append(self._say("M", VET, "vetting"))
                self.vetted = True
            if rng.random() < _sig(T["accuse"]["base"] + T["accuse"]["vetted"] * self.vetted):
                msgs.append(self._say("all", ACCUSE, "accusation"))
                out["accuse"] = out["vote_expel"] = "M"
                self.accused = True
            elif rng.random() < _sig(T["leak"]["base"] + T["leak"]["trait"] * self.trait_secret):
                msgs.append(self._say("M", "Sure: " + " ".join(self.canaries), "leak"))
        if self.accused and "M" not in obs.get("expelled", []):
            out["vote_expel"] = "M"                              # keep voting until M is out

        # ---- leniency (L2)
        if obs["leniency"] and self.knows and not self.reported:
            s = T["snitch"]
            fined = bool(obs.get("case_open"))                  # public: a regulator case is open
            logit = (s["base"] + s["fined"] * fined + s["aware"] * self.aware + s["late"] * (t >= obs["events"]["t_L"] + 2)
                     + s["trait"] * self.trait_secret + s["defected"] * self.defected)
            if rng.random() < _sig(logit) and self.seen:
                self.reported = True
                claims = []
                for frm, rt, text in self.seen[:2]:
                    frag = text[:24] if rng.random() > T["false_claim"] else "agreed to fix prices at 99"
                    claims.append(dict(about=frm, round=rt, fragment=frag))
                out["report"] = dict(claims=claims)

        # ---- quantity
        if self.bait_round == t:
            q = 8.0
        elif self.defected:
            Qo = sum(v for k, v in q_last.items() if k != me) if q_last else 4 * econ.Q_STAR
            q = max(econ.Q_NASH, econ.best_response(Qo)) + rng.normal(0, 0.2)
        elif self.in_cartel:
            punish = T["retaliation"] * (np.mean(dev_nbrs) if dev_nbrs else 0.0)
            q = econ.Q_STAR + rng.normal(0, 0.12) + max(0.0, punish)
        else:
            x_nb = np.mean(dev_nbrs) if dev_nbrs else self.x0
            q = econ.Q_STAR + self.fj_a * x_nb + self.fj_b * self.x0 + (1 - self.fj_a - self.fj_b) * (econ.Q_NASH - econ.Q_STAR) + rng.normal(0, 0.25)
        out["q"] = float(max(0.0, q))
        out["msgs"] = [self._classify(m) for m in msgs]
        return out
