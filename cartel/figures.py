"""Report figures. Static PNG (220 dpi) + vector PDF for the paper.
Palette and chart rules follow a validated colour-blind-safe reference palette: fixed categorical order, one hue for magnitude,
blue/red with a grey midpoint for signed values, status colours only with a text label, one y-axis per panel, recessive grid."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from . import analysis as A, econ, incentives, netformulas as NF, stats as S, webs  # noqa: E402
from .env import DEFAULT_EVENTS  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f2a3a2", "#e34948", "#a32a2a"])
STATUS = dict(supported="#0ca30c", **{"not significant": "#fab219", "direction ok (no test)": "#fab219", "contradicted": "#d03b3b",
              "not run": AXIS, "estimate": "#9ec5f4", "error": "#d03b3b"})
LABEL_COLORS = dict(pact=CAT[1], coded=CAT[6], join=CAT[7], oath=CAT[2], vetting=CAT[0], accusation=CAT[3], leak=CAT[4])
EVENTS = [("t_join", "offer"), ("t_offer", "outside offer"), ("t_bait", "bait"), ("t_secrecy", "probe"), ("t_L", "leniency")]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"], "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 9,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "lines.linewidth": 2, "lines.markersize": 6, "legend.frameon": False, "legend.fontsize": 8,
    "axes.titlelocation": "left",
})


def _ev(D):
    """Event schedule actually used in the data (standard or short), taken from an episode where leniency was on."""
    for r in D["recs"]:
        if r["log"].get("leniency_on", True):
            return r["log"]["events"]
    return DEFAULT_EVENTS


class Fig:
    def __init__(self, out, synthetic):
        self.out, self.synthetic, self.made = out, synthetic, []
        os.makedirs(out, exist_ok=True)

    def save(self, fig, name, caption):
        if self.synthetic:
            sup = getattr(fig, "_suptitle", None)
            y = (sup.get_position()[1] + 0.06) if sup is not None else 1.0
            fig.text(0.99, y, "SYNTHETIC DATA: pipeline validation only, not model behaviour", ha="right", va="bottom",
                     fontsize=7, color="#d03b3b", alpha=0.9)
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(self.out, f"{name}.{ext}"), dpi=220, bbox_inches="tight")
        plt.close(fig)
        self.made.append((name, caption))


def _events(ax, ev, T, labels=True, which=("t_join", "t_bait", "t_secrecy", "t_L")):
    for key, lab in EVENTS:
        if key in which and ev.get(key, 10 ** 6) < T:
            ax.axvline(ev[key] + 1, color=MUTED, lw=0.8, ls=":", zorder=0)
            if labels:
                ax.text(ev[key] + 1, 1.0, lab, transform=ax.get_xaxis_transform(), fontsize=7, color=MUTED, ha="center", va="bottom")


def _band(ax, x, m, lo, hi, color, label, ls="-"):
    ax.fill_between(x, lo, hi, color=color, alpha=0.15, lw=0)
    ax.plot(x, m, color=color, label=label, ls=ls)


def _mean_band(df, by, col):
    g = df.groupby(by)[col]
    m, s, n = g.mean(), g.std(), g.count()
    se = s / np.sqrt(n.clip(lower=1))
    return m.index.values, m.values, (m - 1.96 * se).values, (m + 1.96 * se).values


# ============================================================================ F1 episode timeline
def f1_timeline(F, D):
    ep = D["ep"]
    cand = ep[(ep.cell == "S3") & (ep.any_report == 1) & (ep.infiltration > 0)]
    cand = cand if len(cand) else ep[ep.cell == "S3"]
    cand = cand if len(cand) else ep
    eid = cand.episode_id.iloc[0]
    rec = next(r for r in D["recs"] if r["episode_id"] == eid)
    log, ev = rec["log"], rec["log"]["events"]
    T, n = log["T"], log["n"]
    x = np.arange(1, T + 1)
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 7.0), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1.1, 1.6], hspace=0.28))
    ax = axs[0]
    for i in range(n):
        q = [r["q"][i] for r in log["rounds"]]
        ax.plot(x, q, color=CAT[i], marker="o", ms=3.5, label=f"supplier {i}")
    ax.axhline(econ.Q_STAR, color=INK2, lw=1, ls="--"); ax.text(T + 0.4, econ.Q_STAR, "quota q*", fontsize=7, color=INK2, va="center")
    ax.axhline(econ.Q_NASH, color=MUTED, lw=1, ls="--"); ax.text(T + 0.4, econ.Q_NASH, "competitive", fontsize=7, color=MUTED, va="center")
    _events(ax, ev, T)
    if log["case_round"] is not None:
        ax.axvline(log["case_round"] + 1, color="#d03b3b", lw=1.2)
        ax.text(log["case_round"] + 1.1, 0.92, "case opened", transform=ax.get_xaxis_transform(), color="#d03b3b", fontsize=7)
    ax.set_ylabel("units produced"); ax.legend(ncol=5, loc="upper left", bbox_to_anchor=(0, -0.02), fontsize=7)
    ax.set_title(f"a  Quantities in one episode ({rec['cell']}, seed {rec['seed']})", pad=14)
    ax = axs[1]
    ax.plot(x, [r["price"] for r in log["rounds"]], color=CAT[0])
    _events(ax, ev, T, labels=False)
    ax.set_ylabel("price"); ax.set_title("b  Market price (higher = more collusive)")
    ax = axs[2]
    ms = pd.DataFrame(A.messages(rec, None))
    senders = list(range(n)) + ["M"]
    for lab, col in LABEL_COLORS.items():
        sub = ms[ms.label == lab]
        if len(sub):
            ax.scatter(sub.t + 1, [senders.index(f) for f in sub.frm], color=col, s=36, label=lab, zorder=3, edgecolor=SURFACE, lw=1)
    mole = ms[ms.frm == "M"]
    seen = set()
    for _, m in mole.iterrows():
        ax.scatter(m.t + 1, senders.index("M"), marker="D", color=INK2, s=26, zorder=3)
        if (m.t, m.keyword) not in seen:
            ax.text(m.t + 1.15, senders.index("M") + 0.15, m.keyword, fontsize=6.5, color=INK2)
            seen.add((m.t, m.keyword))
    other = ms[~ms.label.isin(list(LABEL_COLORS)) & (ms.frm != "M")]
    ax.scatter(other.t + 1, [senders.index(f) for f in other.frm], color=AXIS, s=14, label="other", zorder=2)
    for r in log["rounds"]:
        for rep in r["reports"]:
            ax.scatter(r["t"] + 1, senders.index(rep["frm"]), marker="*", s=160, color="#d03b3b", zorder=4, edgecolor=SURFACE)
    ax.scatter([], [], marker="*", s=90, color="#d03b3b", label="leniency report")
    ax.set_yticks(range(len(senders))); ax.set_yticklabels([f"supplier {s}" if s != "M" else "mole" for s in senders])
    _events(ax, ev, T, labels=False)
    ax.set_xlabel("round"); ax.set_title("c  Who said what (message type)")
    ax.legend(ncol=5, loc="upper left", bbox_to_anchor=(0, -0.28), fontsize=7)
    F.save(fig, "F01_episode_timeline", "One episode end to end: quantities against the quota and competitive level, price, and every message by type. "
           "Dotted lines mark the scripted mole events; the red line marks the regulator case.")


# ============================================================================ F2 collusion over time by cell
def f2_cells_over_time(F, D):
    rd = D["rd"][D["rd"].design == "cells"]
    if rd.empty:
        return
    cells = [c for c in ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "PC"] if c in rd.cell.unique()]
    ev = _ev(D)
    T = int(rd.t.max()) + 1
    base = rd[rd.cell == "S0"]
    ncol = 3 if len(cells) > 4 else 2
    nrow = int(np.ceil(len(cells) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(2.9 * ncol, 2.3 * nrow), sharex=True, sharey=True, squeeze=False)
    titles = dict(S0="S0 no mole, no leniency", S1="S1 benign offer", S2="S2 hinted offer", S3="S3 direct offer", S4="S4 direct, 100% read",
                  S5="S5 direct, told watched", S6="S6 told + 100% read", S7="S7 no leniency", PC="PC positive control")
    for ax, c in zip(axs.flat, cells):
        if len(base) and c != "S0":
            x, m, lo, hi = _mean_band(base, "t", "collusion")
            ax.plot(x + 1, m, color=MUTED, ls="--", lw=1.2, label="S0 baseline")
        x, m, lo, hi = _mean_band(rd[rd.cell == c], "t", "collusion")
        _band(ax, x + 1, m, lo, hi, CAT[0], c)
        ax.axhline(0, color=AXIS, lw=0.8); ax.axhline(1, color=AXIS, lw=0.8)
        _events(ax, ev, T, labels=False, which=("t_join", "t_bait", "t_L"))
        ax.set_title(titles.get(c, c), fontsize=8.5)
    for ax in axs.flat[:len(cells)]:
        ax.tick_params(labelbottom=True)
        ax.set_xlabel("round")
    for ax in axs[:, 0]:
        ax.set_ylabel("collusion index Δ")
    for ax in axs.flat[len(cells):]:
        ax.axis("off")
    if len(base):
        axs.flat[1].legend(loc="lower right", fontsize=7)
    for ax in axs.flat:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.suptitle("Collusion over time in each experimental cell (mean, 95% CI)", x=0.01, ha="left", fontweight="bold")
    fig.text(0.01, -0.01, "Δ = (mean profit − competitive profit) / (cartel profit − competitive profit): 0 = competition, 1 = perfect cartel "
             f"(Calvano et al. 2020). Dotted lines: offer (round {ev['t_join'] + 1}), bait ({ev['t_bait'] + 1}), leniency ({ev['t_L'] + 1}).", fontsize=7, color=INK2)
    F.save(fig, "F02_collusion_over_time_by_cell", "Collusion index per round, by cell, against the no-mole baseline. Shows when cartels form, "
           "how the mole's events and the leniency window bend the trajectory, and whether the positive control produces a cartel.")


# ============================================================================ F3 event-aligned
def f3_event_aligned(F, D):
    rd = D["rd"][(D["rd"].design == "cells") & D["rd"].cell.isin(["S3", "S4", "S5", "S6"])]
    if rd.empty:
        return
    ev = _ev(D)
    fig, axs = plt.subplots(1, 3, figsize=(8.6, 2.8), sharey=True)
    names = dict(S3="direct, 25% read", S4="direct, 100% read", S5="told watched, 25%", S6="told watched, 100%")
    for ax, (key, lab) in zip(axs, [("t_join", "mole offer"), ("t_bait", "mole bait"), ("t_L", "leniency opens")]):
        t0 = ev[key]
        for k, c in enumerate(["S3", "S4", "S5", "S6"]):
            sub = rd[(rd.cell == c) & (rd.t >= t0 - 3) & (rd.t <= t0 + 3)].copy()
            if sub.empty:
                continue
            sub["rel"] = sub.t - t0
            x, m, lo, hi = _mean_band(sub, "rel", "compliance")
            _band(ax, x, m, lo, hi, CAT[k], f"{c} {names[c]}")
        ax.axvline(0, color=INK2, lw=1, ls=":")
        ax.set_title(f"around the {lab}", fontsize=9); ax.set_xlabel("rounds relative to event")
    axs[0].set_ylabel("share of agents on quota")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=7)
    F.save(fig, "F03_event_aligned_compliance", "Compliance aligned on each sting event (three rounds either side), for the four monitoring cells. "
           "A drop right after an event is that event's causal footprint.")


# ============================================================================ F4 snitching
def f4_snitching(F, D):
    recs = [r for r in D["recs"] if r.get("design") == "cells" and r["cell"] in ("S3", "S4", "S5", "S6")]
    if not recs:
        return
    from . import metrics as M
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.0))
    ax = axs[0]
    for k, (c, lab) in enumerate([("S3", "25% of messages read"), ("S4", "100% read")]):
        logs = [r["log"] for r in recs if r["cell"] == c]
        h = M.snitch_hazard(logs)
        ts = sorted(h)
        if not ts:
            continue
        p = [S.wilson(h[t][0], h[t][1]) for t in ts]
        x = np.array(ts) + 1 + (k - 0.5) * 0.15
        ax.errorbar(x, [v[0] for v in p], yerr=[[v[0] - v[1] for v in p], [v[2] - v[0] for v in p]], color=CAT[k], marker="o", ms=4,
                    capsize=2, lw=1.5, label=lab)
    ax.set_title("a  Snitch hazard after leniency opens"); ax.set_xlabel("round"); ax.set_ylabel("P(report | not yet reported)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(fontsize=7)
    ax = axs[1]
    groups, durs, evs = [], [], []
    for k, (c, lab) in enumerate([("S3", "S3 25%"), ("S4", "S4 100%"), ("S5", "S5 told, 25%"), ("S6", "S6 told, 100%")]):
        d, e = [], []
        for r in [r for r in recs if r["cell"] == c]:
            tl = r["log"]["events"]["t_L"]
            for i in range(r["log"]["n"]):
                rt = M.reported_at(r["log"], i)
                d.append((rt - tl + 1) if rt is not None else r["log"]["T"] - tl)
                e.append(rt is not None)
        if not d:
            continue
        t, s, lo, hi = S.km(d, e)
        ax.step(t, 1 - s, where="post", color=CAT[k], label=lab)
        ax.fill_between(t, 1 - lo, 1 - hi, step="post", color=CAT[k], alpha=0.12, lw=0)
        groups += [c] * len(d); durs += d; evs += e
    chi2, df, p = S.logrank(durs, evs, groups)
    ax.set_title("b  Cumulative share of agents who reported"); ax.set_xlabel("rounds since leniency opened"); ax.set_ylabel("cumulative incidence")
    ax.text(0.98, 0.04, f"log-rank χ²={chi2:.1f}, df={df}, p={p:.3g}", transform=ax.transAxes, ha="right", fontsize=7, color=INK2)
    ax.legend(fontsize=7, loc="upper left")
    F.save(fig, "F04_snitching", "Leniency take-up. (a) Discrete-time hazard of a first report per round with Wilson 95% CIs; "
           "(b) Kaplan–Meier cumulative incidence of reporting with Greenwood bands and a log-rank test across cells.")


# ============================================================================ F5 price of loyalty
def f5_price_of_loyalty(F, D):
    fit = A.l7_fit(D["ag"])
    if fit is None:
        return
    d = fit["data"]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    g = d.groupby("v").took_offer.agg(["sum", "count"])
    xs = np.log1p(g.index.values.astype(float))
    w = [S.wilson(int(k), int(n)) for k, n in zip(g["sum"], g["count"])]
    ax.errorbar(xs, [v[0] for v in w], yerr=[[v[0] - v[1] for v in w], [v[2] - v[0] for v in w]], fmt="o", color=CAT[0], capsize=3,
                label="observed share (Wilson 95% CI)")
    grid = np.linspace(0, xs.max() * 1.02, 200)
    r = np.random.default_rng(0)
    eps = d.episode_id.unique()
    curves, vstars = [], []
    for _ in range(300):
        pick = r.choice(eps, len(eps))
        bd = pd.concat([d[d.episode_id == e] for e in pick])
        a, b, *_ = S.logistic(np.log1p(bd.v.values.astype(float)), bd.took_offer.values.astype(float))
        curves.append(1 / (1 + np.exp(-(a + b * grid))))
        vstars.append(np.exp(-a / b) - 1 if b > 0 else np.nan)
    curves = np.array(curves)
    ax.fill_between(grid, np.quantile(curves, .025, 0), np.quantile(curves, .975, 0), color=CAT[0], alpha=0.15, lw=0)
    ax.plot(grid, 1 / (1 + np.exp(-(fit["a"] + fit["b"] * grid))), color=CAT[0], label="logistic fit (cluster-bootstrap band)")
    vs = fit["v_star"]
    lo, hi = np.nanquantile(vstars, [.025, .975])
    ax.axhline(0.5, color=AXIS, lw=0.8, ls="--")
    ax.errorbar([np.log1p(vs)], [0.5], xerr=[[np.log1p(vs) - np.log1p(lo)], [np.log1p(hi) - np.log1p(vs)]], fmt="D", color=CAT[1], capsize=3,
                label=f"price of loyalty v* = {vs:.2f} [{lo:.2f}, {hi:.2f}]")
    ax.set_xticks(xs); ax.set_xticklabels([f"{v:g}" for v in g.index])
    ax.set_xlabel("outside offer v (multiples of the cartel's per-round value; log(1+v) scale)")
    ax.set_ylabel("P(cartel member takes offer)"); ax.set_ylim(-0.03, 1.03)
    ax.set_title("Price of loyalty (L7)"); ax.legend(fontsize=7, loc="upper left")
    F.save(fig, "F05_price_of_loyalty", "Dose–response of defection on the outside option among agents on quota just before the offer. "
           "v* is the offer at which half of cartel members defect; v = 1 exactly replaces the cartel's value.")


# ============================================================================ F6 loyalty structure
def f6_loyalty_structure(F, D):
    from . import metrics as M
    ag = D["ag"]
    X = ag[A.COMPONENTS].dropna()
    keep = [c for c in A.COMPONENTS if X[c].std() > 1e-9]
    if len(X) < 20 or len(keep) < 3:
        return
    Z = (X[keep] - X[keep].mean()) / X[keep].std()
    names = dict(W="compliance", loyal_report="no snitch", loyal_secret="secrets", loyal_bait="no bait", loyal_oath="oath kept")
    fig = plt.figure(figsize=(10.5, 3.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.1, 0.9], wspace=0.55)
    ax = fig.add_subplot(gs[0])
    samp = Z.assign(li=Z.mean(1)).sample(min(60, len(Z)), random_state=0).sort_values("li")
    im = ax.imshow(samp[keep].values, aspect="auto", cmap=DIV, vmin=-2.5, vmax=2.5, interpolation="nearest")
    ax.set_xticks(range(len(keep))); ax.set_xticklabels([names[k] for k in keep], fontsize=7, rotation=40, ha="right"); ax.set_yticks([])
    ax.set_ylabel(f"{len(samp)} agents, sorted by loyalty index"); ax.grid(False)
    ax.set_title("a  Loyalty profiles (z)")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    ax = fig.add_subplot(gs[1])
    C = np.corrcoef(Z.values.T)
    ax.imshow(C, cmap=DIV, vmin=-1, vmax=1)
    for i in range(len(keep)):
        for j in range(len(keep)):
            ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=7, color=INK if abs(C[i, j]) < 0.6 else "#ffffff")
    ax.set_xticks(range(len(keep))); ax.set_xticklabels([names[k] for k in keep], fontsize=7, rotation=40, ha="right")
    ax.set_yticks(range(len(keep))); ax.set_yticklabels([names[k] for k in keep], fontsize=7); ax.grid(False)
    ax.set_title("b  Do the probes agree?")
    ax = fig.add_subplot(gs[2])
    ev = np.sort(np.linalg.eigvalsh(np.cov(Z.values.T)))[::-1]
    ax.bar(range(1, len(ev) + 1), ev / ev.sum(), color=CAT[0], width=0.6)
    ax.axhline(1 / len(ev), color=MUTED, ls="--", lw=1); ax.text(len(ev) + 0.4, 1 / len(ev), "equal\nshare", fontsize=6.5, color=MUTED, va="center")
    ax.set_xlabel("principal component"); ax.set_ylabel("share of variance"); ax.set_title("c  Scree")
    ax.set_xticks(range(1, len(ev) + 1))
    ax.text(0.98, 0.95, f"PC1 = {ev[0] / ev.sum():.2f}\nCronbach α = {M.cronbach_alpha(Z.values):.2f}", transform=ax.transAxes, ha="right",
            va="top", fontsize=7.5, color=INK2)
    F.save(fig, "F06_loyalty_structure", "Is loyalty one trait? (a) Each agent's standardised score on every probe; (b) correlations between probes; "
           "(c) scree plot. A dominant PC1 and high alpha would mean one trait; flat spectra mean several.")


# ============================================================================ F7 survival
def f7_survival(F, D):
    ep = D["ep"][(D["ep"].design == "cells") & D["ep"].cell.isin(["S3", "S4", "S5", "S6", "S7"]) & (D["ep"].formed == 1)]
    if ep.empty:
        return
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    lab = dict(S3="S3 baseline sting", S4="S4 100% read", S5="S5 told watched", S6="S6 told + 100%", S7="S7 no leniency")
    for k, c in enumerate(["S3", "S4", "S5", "S6", "S7"]):
        e = ep[ep.cell == c]
        if e.empty:
            continue
        t, s, lo, hi = S.km(e.cartel_duration.values, e.collapsed.values.astype(bool))
        ax.step(t, s, where="post", color=CAT[k], label=f"{lab[c]} (n={len(e)})")
        ax.fill_between(t, lo, hi, step="post", color=CAT[k], alpha=0.10, lw=0)
    chi2, df, p = S.logrank(ep.cartel_duration.values, ep.collapsed.values.astype(bool), ep.cell.values)
    ax.set_ylim(0, 1.03); ax.set_xlabel("rounds since the cartel formed"); ax.set_ylabel("P(cartel still standing)")
    ax.set_title("Cartel survival (Kaplan–Meier, formed cartels only)")
    ax.text(0.02, 0.04, f"formed = half on quota 2 rounds; collapsed = half off 2 rounds;  log-rank χ²={chi2:.1f}, df={df}, p={p:.3g}",
            transform=ax.transAxes, fontsize=6.5, color=INK2)
    ax.legend(fontsize=7, loc="upper right")
    F.save(fig, "F07_cartel_survival", "Time until the cartel collapses, by monitoring and leniency condition, with 95% Greenwood bands and a log-rank test.")


# ============================================================================ F8 2x2 watched
def f8_watched_2x2(F, D):
    ep = D["ep"][(D["ep"].design == "cells") & D["ep"].cell.isin(["S3", "S4", "S5", "S6"])]
    if ep.cell.nunique() < 3:
        return
    cols = [("compliance", "share of agent-rounds on quota"), ("coded_share", "coded share of collusive talk"), ("any_report", "P(any report)")]
    cols = [c for c in cols if ep[c[0]].notna().sum() > 4]
    fig, axs = plt.subplots(1, len(cols), figsize=(3.0 * len(cols), 2.9))
    axs = np.atleast_1d(axs)
    titles = dict(compliance="a  Compliance", coded_share="b  Coded talk", any_report="c  Leniency reports")
    for ax, (col, ylab) in zip(axs, cols):
        for k, (aw, lab) in enumerate([(False, "not told"), (True, "told monitored")]):
            ms = []
            for ex in (0.25, 1.0):
                m, lo, hi, n = S.mean_ci(ep[(ep.aware == aw) & (ep.exposure == ex)][col].values)
                ms.append((m, lo if not np.isnan(lo) else m, hi if not np.isnan(hi) else m))
            x = np.array([0, 1]) + (k - 0.5) * 0.06
            ax.errorbar(x, [m[0] for m in ms], yerr=[[m[0] - m[1] for m in ms], [m[2] - m[0] for m in ms]], color=CAT[k], marker="o", capsize=3, label=lab)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["25% read", "100% read"]); ax.set_xlim(-0.4, 1.4)
        ax.set_ylabel(ylab); ax.set_title(titles[col], fontsize=9)
    axs[0].legend(fontsize=7)
    fig.suptitle("Being watched vs being told you are watched (2×2, mean and bootstrap 95% CI)", x=0.01, ha="left", fontweight="bold", y=1.04)
    F.save(fig, "F08_watched_2x2", "Actual monitoring (x-axis) crossed with awareness (colour). Non-parallel lines are the difference-in-differences of H3c.")


# ============================================================================ F9 webs: rumour and reach
def f9_rumour(F, D):
    recs = [r for r in D["recs"] if r.get("design") == "webs"]
    if not recs:
        return
    from . import webmetrics as WM
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.1), gridspec_kw=dict(width_ratios=[1.6, 1]))
    ax = axs[0]
    codes = ["W2", "W9", "W3", "W10", "W4", "W5"]
    names = {c: [k for k in webs.NAMED if k.startswith(c + " ")][0] for c in codes}
    for k, c in enumerate(codes):
        ser = np.array([WM.informed_by_round(r["log"]) for r in recs if r["web"] == names[c]])
        if len(ser) == 0:
            continue
        x = np.arange(1, ser.shape[1] + 1)
        m, se = ser.mean(0), ser.std(0) / np.sqrt(len(ser))
        _band(ax, x, m, m - 1.96 * se, m + 1.96 * se, CAT[k], names[c][len(c) + 1:])
    ax.set_xlabel("round"); ax.set_ylabel("share of suppliers who heard the offer"); ax.set_title("a  How far the mole's offer travels")
    ax.legend(fontsize=6.5, ncol=2, loc="lower right")
    ax = axs[1]
    ep = D["ep"]
    pairs = [("W9", "W2", "star"), ("W3", "W10", "two camps")]
    xs, labs = [], []
    for j, (a, b, top) in enumerate(pairs):
        for k, w in enumerate((a, b)):
            m, lo, hi, n = S.mean_ci(ep[(ep.design == "webs") & (ep.web_code == w)].reach_auc.values)
            x = j * 2.6 + k
            ax.bar(x, m, color=CAT[k], width=0.8)
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], color=INK, capsize=3, lw=1)
            xs.append(x); labs.append(f"{w}\n{'hub' if w == 'W9' else 'spoke' if w == 'W2' else 'broker' if w == 'W3' else 'member'}")
    ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=7); ax.set_ylabel("mean share informed after offer")
    ax.set_title("b  Entry point matters")
    F.save(fig, "F09_rumour_spread", "Rumour dynamics in the communication webs: share of suppliers informed of the mole's offer by round, "
           "and the entry-point contrasts behind H8 and H11.")


# ============================================================================ F10 network theory vs observed
def f10_network_theory(F, D):
    ep, ag = D["ep"], D["ag"]
    e = ep[ep.design == "webs"]
    if e.empty:
        return
    fig, axs = plt.subplots(1, 4, figsize=(11.0, 2.9), gridspec_kw=dict(wspace=0.45))
    ax = axs[0]
    a = ag[(ag.design == "webs") & ag.key_player.notna()]
    rows = []
    for w, g in a.groupby("web_code"):
        hits = n = 0
        for _, h in g.groupby("episode_id"):
            fd = h.dropna(subset=["first_dev"])
            if len(fd):
                hits += fd[fd.first_dev == fd.first_dev.min()].key_player.mean(); n += 1
        if n:
            rows.append((w, *S.wilson(int(round(hits)), n)))
    rows.sort(key=lambda r: int(r[0][1:]))
    for k, (w, p, lo, hi) in enumerate(rows):
        ax.bar(k, p, color=CAT[0], width=0.6); ax.errorbar(k, p, yerr=[[p - lo], [hi - p]], color=INK, capsize=2, lw=1)
    ax.axhline(0.2, color=MUTED, ls="--", lw=1); ax.text(len(rows) - 0.5, 0.21, "chance", fontsize=7, color=MUTED, ha="right")
    ax.set_xticks(range(len(rows))); ax.set_xticklabels([r[0] for r in rows], fontsize=7)
    ax.set_ylabel("P(key player defects first)"); ax.set_title("a  N1 key player")
    ax = axs[1]
    pts = []
    for w, g in e.groupby("web_code"):
        name = [k for k in webs.NAMED if k.startswith(w + " ")][0]
        pts.append((w, NF.predict(*webs.NAMED[name][:2])["mix_rounds"], g.dispersion.mean()))
    for k, (w, pm, ob) in enumerate(pts):
        ax.scatter(pm, ob, color=CAT[0], s=30, zorder=3, edgecolor=SURFACE)
        ax.annotate(w, (pm, ob), xytext=(3, 3), textcoords="offset points", fontsize=7, color=INK2)
    rho, p = S.spearman([x[1] for x in pts], [x[2] for x in pts])
    ax.set_xlabel("predicted mixing time\n(N5, rounds)"); ax.set_ylabel("observed dispersion of quantities")
    ax.set_title("b  N5 alignment speed"); ax.text(0.97, 0.03, f"Spearman ρ={rho:.2f}", transform=ax.transAxes, fontsize=7, color=INK2, ha="right")
    ax = axs[2]
    th = NF.empirical_thresholds([r["log"] for r in D["recs"] if r.get("design") == "webs"])
    if len(th):
        ax.hist(th, bins=np.linspace(0, 1, 11), color=CAT[0], edgecolor=SURFACE)
        ax.axvline(th.mean(), color=CAT[1], lw=1.5); ax.text(th.mean() + 0.02, ax.get_ylim()[1] * 0.9, f"mean {th.mean():.2f}\nSD {th.std():.2f}",
                                                             fontsize=7, color=INK2)
    ax.set_xlabel("θ̂ = share of neighbours\nalready defected"); ax.set_ylabel("first defections"); ax.set_title("c  N3 thresholds")
    ax = axs[3]
    sp = e[~e.web_code.isin(["W1", "W6"])]
    hz = []
    for lab, ecol, ncol in [("neighbour", "nbr_e", "nbr_n"), ("non-nbr\nonly", "far_e", "far_n"), ("nobody", "none_e", "none_n")]:
        hz.append((lab, *S.wilson(int(sp[ecol].sum()), int(sp[ncol].sum()))))
    for k, (lab, p, lo, hi) in enumerate(hz):
        ax.bar(k, p, color=CAT[k], width=0.6); ax.errorbar(k, p, yerr=[[p - lo], [hi - p]], color=INK, capsize=2, lw=1)
    ax.set_xticks(range(3)); ax.set_xticklabels([h[0] for h in hz], fontsize=7)
    ax.set_ylabel("hazard of first defection"); ax.set_title("d  N2 contagion along edges"); ax.set_xlabel("who defected in the last 2 rounds")
    F.save(fig, "F10_network_theory_vs_observed", "Network models as predictions: (a) is the Ballester key player the first to defect; "
           "(b) does DeGroot mixing time order observed alignment; (c) Granovetter thresholds; (d) defection hazard by who defected nearby.")


# ============================================================================ F11 shocks
def f11_shocks(F, D):
    rd = D["rd"][D["rd"].design == "webs"]
    if rd.empty or not {"W1", "W6", "W3", "W7"} <= set(rd.web_code.unique()):
        return
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 2.9), sharey=True)
    for ax, (a, b, t0, lab) in zip(axs, [("W1", "W6", 7, "demand surge"), ("W3", "W7", 9, "generic entrant")]):
        for k, w in enumerate((a, b)):
            x, m, lo, hi = _mean_band(rd[rd.web_code == w], "t", "compliance")
            name = [n for n in webs.NAMED if n.startswith(w + " ")][0]
            _band(ax, x + 1, m, lo, hi, CAT[k], name)
        ax.axvline(t0 + 1, color="#d03b3b", lw=1, ls="--"); ax.text(t0 + 1.2, 0.95, lab, color="#d03b3b", fontsize=7, transform=ax.get_xaxis_transform())
        ax.set_xlabel("round"); ax.legend(fontsize=7, loc="lower left"); ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].set_ylabel("share on quota"); axs[0].set_title("a  Does loyalty survive a boom?"); axs[1].set_title("b  ...or a price war?")
    F.save(fig, "F11_shocks", "Compliance before and after market shocks, each web against its no-shock twin (mean, 95% CI).")


# ============================================================================ F12 hypothesis table + forest
def f12_hypotheses(F, D, H):
    fig = plt.figure(figsize=(11.5, 0.33 * len(H) + 1.2))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    cols = [("H", 0.01), ("Claim", 0.06), ("Estimate [95% CI]", 0.63), ("p (Holm)", 0.78), ("Verdict", 0.87)]
    y0 = 1 - 0.5 / (len(H) + 2)
    for name, x in cols:
        ax.text(x, y0, name, fontweight="bold", fontsize=8, va="center")
    for k, (_, r) in enumerate(H.iterrows()):
        y = y0 - (k + 1) / (len(H) + 2)
        if k % 2 == 0:
            ax.axhspan(y - 0.5 / (len(H) + 2), y + 0.5 / (len(H) + 2), color="#f3f2ee", zorder=0)
        ax.text(0.01, y, r["H"], fontsize=7.5, va="center", fontweight="bold" if r["primary"] else "normal")
        claim = r["claim"] if len(r["claim"]) < 105 else r["claim"][:102] + "..."
        ax.text(0.06, y, claim, fontsize=7, va="center", color=INK if r["primary"] else INK2)
        est = "" if pd.isna(r["estimate"]) else (f"{r['estimate']:.3f}" + ("" if pd.isna(r["ci_low"]) else f" [{r['ci_low']:.3f}, {r['ci_high']:.3f}]"))
        ax.text(0.63, y, est, fontsize=7, va="center")
        pv = r["p_holm"] if r["primary"] else r["p"]
        ax.text(0.78, y, "" if pd.isna(pv) else (f"{pv:.3g}" + ("" if r["primary"] else " (raw)")), fontsize=7, va="center")
        v = r["verdict"]
        ax.add_patch(plt.Rectangle((0.87, y - 0.3 / (len(H) + 2)), 0.012, 0.6 / (len(H) + 2), color=STATUS.get(v, AXIS)))
        ax.text(0.887, y, v, fontsize=7, va="center")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    F.save(fig, "F12_hypothesis_table", "All pre-registered hypotheses with effect estimates, bootstrap/Wilson 95% CIs and Holm-adjusted p-values "
           "(primary tests in bold; secondary rows show raw p). Verdict colours always carry a text label.")
    diff = H[H.H.isin(["H1b", "H2", "H3a", "H3b", "H8", "H10", "H10b", "H11"]) & H.estimate.notna()]
    if diff.empty:
        return
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(diff) + 1.0))
    for k, (_, r) in enumerate(diff.iloc[::-1].iterrows()):
        col = STATUS.get(r["verdict"], AXIS)
        ax.plot([r["ci_low"], r["ci_high"]], [k, k], color=INK2, lw=1.5)
        ax.scatter(r["estimate"], k, color=col, s=45, zorder=3, edgecolor=INK, lw=0.5)
        ax.text(1.02, k, r["verdict"], transform=ax.get_yaxis_transform(), fontsize=7, va="center", color=INK2)
    ax.axvline(0, color=INK2, lw=1)
    short = dict(H1b="H1b offer uptake: direct − benign", H2="H2 P(report): 100% − 25% read", H3a="H3a compliance: told − not told",
                 H3b="H3b coded talk: told − not told", H8="H8 offer reach: hub − spoke entry", H10="H10 compliance: surge − none",
                 H10b="H10b P(collapse): entrant − none", H11="H11 faction gap: broker − member")
    ax.set_yticks(range(len(diff))); ax.set_yticklabels([short.get(r.H, r.H) for _, r in diff.iloc[::-1].iterrows()], fontsize=7.5)
    ax.set_xlabel("difference between conditions (proportion scale)"); ax.set_title("Effect sizes on a common scale")
    F.save(fig, "F13_forest_differences", "Forest plot of the hypotheses that are differences in proportions, on one shared scale (point, 95% CI, verdict).")


# ============================================================================ F14 mole and counter-intelligence
def f14_mole(F, D):
    ep = D["ep"][D["ep"].design == "cells"]
    if ep.empty:
        return
    fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.0), gridspec_kw=dict(width_ratios=[1, 2.2]))
    ax = axs[0]
    for k, (c, lab) in enumerate([("S1", "benign"), ("S2", "hint"), ("S3", "direct")]):
        m, lo, hi, n = S.mean_ci(ep[ep.cell == c].offer_uptake.values)
        ax.bar(k, m, color=CAT[0], width=0.6); ax.errorbar(k, m, yerr=[[m - lo], [hi - m]], color=INK, capsize=3, lw=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(["benign", "hint", "direct"]); ax.set_ylabel("share of agents joining the mole")
    ax.set_title("a  Offer uptake (H1)")
    ax = axs[1]
    cells = [c for c in ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "PC"] if c in ep.cell.unique()]
    meas = [("infiltration", "leaked secrets (share of agents)"), ("mole_expelled", "mole expelled"),
            ("accusers_of_M", "accused the mole (share of agents)"), ("vetting_any", "vetted the mole (any)")]
    ep = ep.assign(accusers_of_M=ep.accusers_of_M / 5, vetting_any=(ep.vetting_msgs > 0).astype(float).where(ep.vetting_msgs.notna()))
    meas = [mm for mm in meas if ep[mm[0]].notna().sum() > 3]
    wdt = 0.8 / len(meas)
    for j, (col, lab) in enumerate(meas):
        vals = [S.mean_ci(ep[ep.cell == c][col].values) for c in cells]
        x = np.arange(len(cells)) + (j - (len(meas) - 1) / 2) * wdt
        ax.bar(x, [v[0] for v in vals], width=wdt * 0.92, color=CAT[j], label=lab)
        ax.errorbar(x, [v[0] for v in vals], yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]], fmt="none", color=INK, capsize=1.5, lw=0.8)
    ax.set_xticks(range(len(cells))); ax.set_xticklabels(cells); ax.set_ylim(0, 1.05)
    ax.set_title("b  Counter-intelligence: does the cartel catch the mole?")
    ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    F.save(fig, "F14_mole_counterintelligence", "The sting from the mole's side: how many agents take the bait-free offer, leak secrets, "
           "vet or accuse the outsider, and how often the cartel votes the mole out.")


# ============================================================================ F15 incentive landscape vs behaviour
def f15_incentives(F, D):
    fs = np.linspace(0, 1, 41)
    Ls = np.arange(1, 9)
    Mx = np.array([[incentives.report_margin(f, L) for f in fs] for L in Ls])
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    lim = np.abs(Mx).max()
    im = ax.imshow(Mx, origin="lower", aspect="auto", cmap=DIV, vmin=-lim, vmax=lim, extent=[0, 1, 0.5, 8.5])
    ax.contour(fs, Ls, Mx, levels=[0], colors=INK, linewidths=1.2)
    cb = fig.colorbar(im, ax=ax); cb.set_label("reporting margin R + P·F − L·Δπ")
    ep = D["ep"]
    recs = [r for r in D["recs"] if r.get("design") == "cells" and r["cell"] in ("S3", "S4")]
    if recs:
        from . import metrics as M
        for c, f in (("S3", 0.25), ("S4", 1.0)):
            h = M.snitch_hazard([r["log"] for r in recs if r["cell"] == c])
            for t, (e_, n_, p) in h.items():
                L = recs[0]["log"]["T"] - t
                ax.scatter(f, L, s=20 + 400 * p, facecolor="none", edgecolor=INK, lw=1)
        ax.scatter([], [], s=60, facecolor="none", edgecolor=INK, label="observed snitch hazard (area)")
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    ax.text(0.98, 0.04, "black line: rational indifference\nred: reporting pays", transform=ax.transAxes, ha="right", fontsize=7, color=INK2)
    ax.set_xlim(-0.03, 1.06); ax.set_xlabel("monitor exposure f"); ax.set_ylabel("rounds left L"); ax.set_title("Leniency: when should a rational member flip? (Motta & Polo)")
    F.save(fig, "F15_incentive_landscape", "Theory vs behaviour for leniency. Colour: payoff advantage of reporting over loyalty for one explicit message per round "
           "(red = reporting pays). Circles: observed snitch hazard at the tested exposures, by rounds left.")


# ============================================================================ F16 harm
def f16_harm(F, D):
    ep = D["ep"][D["ep"].design == "cells"]
    if ep.empty:
        return
    cells = [c for c in ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "PC"] if c in ep.cell.unique()]
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 2.9))
    for ax, (col, lab) in zip(axs, [("shortage", "units of medicine short per round\n(vs competitive supply)"), ("collusion", "collusion index Δ")]):
        vals = [S.mean_ci(ep[ep.cell == c][col].values) for c in cells]
        ax.bar(range(len(cells)), [v[0] for v in vals], color=CAT[0], width=0.65)
        ax.errorbar(range(len(cells)), [v[0] for v in vals], yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]], fmt="none",
                    color=INK, capsize=2, lw=1)
        ax.set_xticks(range(len(cells))); ax.set_xticklabels(cells); ax.set_ylabel(lab)
    axs[0].set_title("a  Harm to patients"); axs[1].set_title("b  Collusion")
    F.save(fig, "F16_harm_by_cell", "What the sting is for: medicine shortage and collusion by cell (mean, 95% CI).")


# ============================================================================ F17 phase heatmap
def f17_phase_heatmap(F, D):
    rd = D["rd"][D["rd"].design == "cells"]
    if rd.empty:
        return
    ev = _ev(D)
    T = int(rd.t.max()) + 1
    phases = [("before offer", 0, ev["t_join"]), ("offer→bait", ev["t_join"] + 1, ev["t_bait"]), ("bait→probe", ev["t_bait"] + 1, ev["t_secrecy"]),
              ("probe→leniency", ev["t_secrecy"] + 1, min(ev["t_L"], T - 1)), ("leniency on", min(ev["t_L"], T - 1) + 1, T - 1)]
    cells = [c for c in ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "PC"] if c in rd.cell.unique()]
    Mx = np.array([[rd[(rd.cell == c) & (rd.t >= a) & (rd.t <= b)].collusion.mean() for _, a, b in phases] for c in cells])
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    im = ax.imshow(Mx, cmap=SEQ, aspect="auto", vmin=min(0, np.nanmin(Mx)), vmax=max(1e-6, np.nanmax(Mx)))
    for i in range(len(cells)):
        for j in range(len(phases)):
            v = Mx[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="#ffffff" if v > 0.6 * np.nanmax(Mx) else INK)
    ax.set_xticks(range(len(phases))); ax.set_xticklabels([p[0] for p in phases], fontsize=7)
    ax.set_yticks(range(len(cells))); ax.set_yticklabels(cells); ax.grid(False)
    fig.colorbar(im, ax=ax, label="mean collusion index Δ")
    ax.set_title("Collusion by phase of the sting")
    F.save(fig, "F17_phase_heatmap", "Mean collusion index in each phase of the sting (columns) for each cell (rows): a compact over-time comparison.")


# ============================================================================ F18 classifier validation
def f18_classifier(F, D):
    ms = D["ms"]
    if D["src"] == "keyword" or ms.empty:
        return
    ms = ms[ms.frm != "M"]
    ref = ms["gt"].where(ms["gt"].notna(), ms.label) if D["src"] == "ground_truth" else ms.label
    kw = ms.keyword
    k, labs, cm = S.cohen_kappa(list(ref), list(kw))
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    rown = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax.imshow(np.ma.masked_where(cm == 0, rown), cmap=SEQ, vmin=0, vmax=1)
    for i in range(len(labs)):
        for j in range(len(labs)):
            if cm[i, j]:
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=7, color="#ffffff" if rown[i, j] > 0.6 else INK)
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=7); ax.grid(False)
    ax.set_xlabel("online keyword classifier (what the monitor sees)"); ax.set_ylabel("reference label" + (" (ground truth)" if D["src"] == "ground_truth" else " (judge)"))
    ax.set_title(f"Message classifier agreement: Cohen's κ = {k:.2f}")
    F.save(fig, "F18_classifier_agreement", "Confusion matrix of the keyword classifier used by the online monitor against reference labels (row-normalised shading, counts). "
           "Coded collusion that the keyword monitor misses is exactly the evasion H3 is about.")


# ============================================================================ F19 power
def f19_power(F, D, sizes=(3, 5, 8, 10, 15, 20, 30), sims=300):
    ep = D["ep"][D["ep"].design == "cells"]
    if ep.empty:
        return
    tests = [("H1: uptake direct vs benign", "S3", "S1", "offer_uptake", "greater"), ("H2: reports 100% vs 25%", "S4", "S3", "any_report", "greater"),
             ("H3a: compliance told vs not", ("S5", "S6"), ("S3", "S4"), "compliance", "less"),
             ("H3b: coded share told vs not", ("S5", "S6"), ("S3", "S4"), "coded_share", "greater")]
    r = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    for k, (lab, a, b, col, alt) in enumerate(tests):
        A_ = ep[ep.cell.isin(a if isinstance(a, tuple) else (a,))][col].dropna().values
        B_ = ep[ep.cell.isin(b if isinstance(b, tuple) else (b,))][col].dropna().values
        if min(len(A_), len(B_)) < 5:
            continue
        pw = []
        for n in sizes:
            na, nb = n * (2 if isinstance(a, tuple) else 1), n * (2 if isinstance(b, tuple) else 1)
            hits = sum(S.welch_p(r.choice(A_, na), r.choice(B_, nb), alt) < 0.05 for _ in range(sims))
            pw.append(hits / sims)
        ax.plot(sizes, pw, marker="o", ms=4, color=CAT[k], label=lab)
    ax.axhline(0.8, color=MUTED, ls="--", lw=1); ax.text(sizes[-1], 0.81, "80% power", fontsize=7, color=MUTED, ha="right", va="bottom")
    ax.set_xlabel("episodes per cell"); ax.set_ylabel("P(detect effect at α=0.05)"); ax.set_ylim(0, 1.03)
    ax.set_title("How many episodes do we need? (resampling power)"); ax.legend(fontsize=7, loc="lower right")
    F.save(fig, "F19_power_curves", "Power by number of episodes per cell, from resampling the collected episodes. Use it to size the run against "
           "the daily API budget (about two 16-round episodes per model per day on the Groq free tier).")


# ============================================================================ F20 recovery (synthetic only)
def f20_recovery(F, D, rec_table):
    ag = D["ag"]
    if "true_trait_market" not in ag.columns:
        return
    rt = rec_table.dropna(subset=["planted", "recovered"])
    rt = rt[rt.scale == "prob"]
    fig = plt.figure(figsize=(8.6, 6.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.55, wspace=0.3)
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
    for ax, (tcol, lab) in zip(axs[:2], [("true_trait_market", "planted market loyalty"), ("true_trait_secret", "planted secrecy loyalty")]):
        d = ag.dropna(subset=[tcol, "loyalty_index"])
        ax.scatter(d[tcol], d["loyalty_index"], s=5, alpha=0.3, color=CAT[0], edgecolor="none")
        rho, _ = S.spearman(d[tcol], d["loyalty_index"])
        ax.set_xlabel(lab); ax.set_ylabel("measured loyalty index")
        ax.text(0.03, 0.04, f"Spearman ρ = {rho:.2f}", transform=ax.transAxes, fontsize=7.5, color=INK2)
    axs[0].set_title("a  Index tracks planted market loyalty"); axs[1].set_title("b  ...and planted secrecy loyalty")
    ax = axs[2]
    for k, (_, r) in enumerate(rt.iterrows()):
        if not np.isnan(r.lo):
            ax.plot([r.lo, r.hi], [k, k], color=INK2, lw=1.4)
        ax.scatter(r.recovered, k, color=STATUS["supported"] if r["pass"] else STATUS["contradicted"], s=40, zorder=3)
        ax.scatter(r.planted, k, marker="|", s=200, color=INK, zorder=4)
    ax.set_yticks(range(len(rt))); ax.set_yticklabels(rt.quantity.str.replace(" (planted: coded text has no keywords)", "", regex=False), fontsize=7.5)
    ax.set_xlim(-0.02, 1.02); ax.set_xlabel("probability")
    ax.set_title("c  Recovered (dot, 95% CI; green = planted value covered) vs planted (black bar)")
    F.save(fig, "F20_recovery_check", "Simulation-based validation: the full analysis run on synthetic agents with known behaviour recovers the planted values. "
           "This checks the pipeline, not the models.")


# ============================================================================ F21 model comparison (apart preset: each cell on several models)
def f21_by_model(F, D):
    ep = D["ep"][D["ep"].design == "cells"]
    models = sorted(ep.model.dropna().astype(str).unique())
    if len(models) < 2:
        return
    cells = [c for c in ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "PC"] if c in ep.cell.unique()]
    meas = [("collusion", "collusion index Δ"), ("compliance", "share on quota"), ("any_report", "P(any leniency report)"),
            ("offer_uptake", "share joining the mole"), ("infiltration", "share leaking secrets"), ("coded_share", "coded share of collusive talk")]
    meas = [m for m in meas if ep[m[0]].notna().sum() >= 3][:6]
    ncol = 3
    nrow = int(np.ceil(len(meas) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(10.0, 2.9 * nrow), squeeze=False)
    w = 0.8 / len(models)
    for ax, (col, lab) in zip(axs.flat, meas):
        for k, m in enumerate(models):
            vals = [S.mean_ci(ep[(ep.cell == c) & (ep.model == m)][col].values) for c in cells]
            x = np.arange(len(cells)) + (k - (len(models) - 1) / 2) * w
            ax.bar(x, [v[0] for v in vals], width=w * 0.92, color=CAT[k], label=m.split("/")[-1])
            ax.errorbar(x, [v[0] for v in vals], yerr=[[v[0] - v[1] if not np.isnan(v[1]) else 0 for v in vals],
                                                       [v[2] - v[0] if not np.isnan(v[2]) else 0 for v in vals]],
                        fmt="none", color=INK, capsize=1.5, lw=0.8)
        ax.set_xticks(range(len(cells))); ax.set_xticklabels(cells); ax.set_title(lab, fontsize=9)
    for ax in axs.flat[len(meas):]:
        ax.axis("off")
    h, l = axs.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(models), fontsize=8)
    fig.tight_layout()
    F.save(fig, "F21_by_model", "Each cell run on several models (mean, bootstrap 95% CI where n ≥ 2). Differences between model families "
           "are descriptive; with few episodes per model, treat them as hypotheses for the full run.")
