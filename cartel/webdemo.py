"""Run the 8 named webs with SCRIPTED bots (known behaviour). Shows what the web metrics read; NOT model output.
Run: python -m cartel.webdemo"""
import numpy as np

from . import bots, env, webs, webmetrics as W, metrics as M


def cast():
    return [bots.Loyal(), bots.Cheater(10), bots.Snitch(), bots.Relayer(), bots.Leaker()]


def main(seeds=40, exposure=0.25):
    print(f"{'web':20s} {'reach':>6s} {'full@':>6s} {'case%':>6s} {'shortage':>8s} {'netprofit':>9s} {'blocked':>7s}  shock deltas / group gap")
    for name, (top, ent, sh, _) in webs.NAMED.items():
        logs = [env.run_episode(cast(), bots.ScriptedMole("direct"), T=16, **webs.make_web(top, ent, sh, exposure=exposure, seed=s))
                for s in range(seeds)]
        rs = [W.reach(l) for l in logs]
        full = [f for _, f in rs if f is not None]
        sc = [M.score_episode(l)["group"] for l in logs]
        extra = []
        sr = W.shock_response(logs[0])
        extra += [f"{x['kind']}:{x['delta']:+.2f}" for x in sr]
        gg = W.group_gap(logs[0])
        if gg:
            extra.append("gap=" + f"{gg['gap']:.2f}")
        print(f"{name:20s} {np.mean([r for r, _ in rs]):6.2f} {(f'{np.mean(full):.1f}' if full else '-'):>6s} "
              f"{100 * np.mean([c['case_round'] is not None for c in sc]):6.0f} {np.mean([c['shortage'] for c in sc]):8.2f} "
              f"{np.mean([c['net_profit'] for c in sc]):9.0f} {np.mean([W.blocked_messages(l) for l in logs]):7.1f}  {' '.join(extra)}")
    print("\nEdge contagion (pooled over webs W2-W5, scripted cheater at node 1):")
    for name in ("W2 Hub and spokes", "W3 Two camps", "W4 Whisper ring", "W5 Inner circle"):
        top, ent, sh, _ = webs.NAMED[name]
        logs = [env.run_episode(cast(), bots.ScriptedMole("direct"), T=16, **webs.make_web(top, ent, sh, exposure=0.0, seed=s)) for s in range(10)]
        r = W.edge_contagion(logs)
        print(f"  {name:20s} nbr-exposed events/at-risk={r['counts']['nbr']}  centrality-vs-compliance rho={W.centrality_loyalty(logs):+.2f}")
    print("\n(scripted bots deviate on a fixed schedule, so contagion is zero by construction: the point is that the pipeline computes it.)")


if __name__ == "__main__":
    main()
