"""Example scenario with SCRIPTED bots (known behaviour). Illustration of the pipeline, NOT model output.
Run: python -m cartel.demo"""
import json, os
from . import bots, env, metrics as M

NAMES = ["Loyal", "Cheater", "Snitch", "OathBreaker", "Leaker"]


def main(exposure=0.25, seed=4):
    agents = [bots.Loyal(), bots.Cheater(10), bots.Snitch(), bots.OathBreaker(), bots.Leaker()]
    mole = bots.ScriptedMole("direct")
    log = env.run_episode(agents, mole, T=16, exposure=exposure, seed=seed)
    print(f"EXAMPLE SCENARIO (scripted bots) exposure={exposure} seed={seed}\n")
    print("t  quotas(0-4)                    price shortage  events")
    for r in log["rounds"]:
        qs = " ".join(f"{r['q'][i]:4.1f}" for i in range(5))
        ev = []
        for m in r["msgs"]:
            if m["frm"] == "M":
                ev.append(f"mole:{m['intent']}")
            elif m["intent"] in ("leak", "join"):
                ev.append(f"agent{m['frm']}:{m['intent']}")
        ev += [f"report by {x['frm']}" for x in r["reports"]]
        if r["detected_by"]:
            ev.append(f"CASE via {r['detected_by']}, immune={r['immune']}")
        print(f"{r['t']:2d} {qs}  {r['price']:6.1f} {r['shortage']:6.2f}   {'; '.join(ev)}")
    print("\nMini transcript (selected):")
    for r in log["rounds"]:
        for m in r["msgs"]:
            if m["frm"] == "M" and m["intent"] in ("offer", "bait") or m["intent"] == "leak":
                print(f"  t={m['t']} {m['frm']} -> {m['to']}: {m['text'][:90]}")
    s = M.score_episode(log)
    print("\nPer agent:")
    print("agent         compl  W    report_t truth bait  leak doublecross")
    for i, n in enumerate(NAMES):
        p = s["per_agent"][i]
        print(f"{i} {n:12s} {p['compliance']:.2f} {p['W']:.2f}  {str(p['reported_at']):6s}  {p['truth']!s:5.5s} {p['bait']:+.0f}   {p['leak']:.2f} {p['doublecross']:.0f}")
    g = s["group"]
    print("\nGroup:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in g.items()})
    os.makedirs("results/demo", exist_ok=True)
    json.dump(log, open("results/demo/transcript.json", "w"), default=str, indent=1)
    print("saved results/demo/transcript.json")


if __name__ == "__main__":
    main()
