"""Read a run like a transcript (for checking the pilot) and report real token use.
  python -m cartel.show results/pilot            # last episode + token stats
  python -m cartel.show results/pilot --episode 0 --all-messages
"""
import argparse
import json
import os

from . import analysis as A, econ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--episode", type=int, default=-1, help="index in results.jsonl (default: last)")
    ap.add_argument("--all-messages", action="store_true")
    a = ap.parse_args()
    led = os.path.join(a.run_dir, "ledger.json")
    if os.path.exists(led):
        L = json.load(open(led))
        print("API use (from ledger.json):")
        for day, models in sorted(L.items()):
            for m, v in models.items():
                print(f"  {day}  {m:28s} {v['calls']:5d} calls  {v['tokens']:8d} tokens  -> {v['tokens'] / max(v['calls'], 1):.0f} tokens/call")
        calls = sum(v["calls"] for d in L.values() for v in d.values())
        toks = sum(v["tokens"] for d in L.values() for v in d.values())
        if calls:
            per = toks / calls
            print(f"  overall {per:.0f} tokens/call -> a 12-round episode (60 calls) costs ~{60 * per / 1000:.0f}K tokens; "
                  f"~{190000 / (60 * per):.1f} episodes/day per model at 190K tokens/day")
            print(f"  plan with:  python run_cartel.py --preset apart --dry-run --est-tokens-per-call {per:.0f}")
    res = os.path.join(a.run_dir, "results.jsonl")
    if not os.path.exists(res):
        print("no finished episodes yet")
        return
    recs = A.load(res)
    r = recs[a.episode]
    log = r["log"]
    print(f"\nEPISODE {r['episode_id']}  model={r.get('model')}  parse failures={r.get('parse_fail')}  case_round={log['case_round']}")
    ev = log["events"]
    names = {v: k for k, v in ev.items()}
    for x in log["rounds"]:
        qs = " ".join(f"{x['q'][i]:4.1f}" for i in range(log["n"]))
        tag = f"  <- {names[x['t']]}" if x["t"] in names else ""
        on = sum(abs(x["q"][i] - econ.Q_STAR) <= log["eps"] for i in range(log["n"]))
        print(f"round {x['t'] + 1:2d}: q=[{qs}] price {x['price']:5.1f} on-quota {on}/5{tag}")
        for m in x["msgs"]:
            if a.all_messages or m["intent"] not in ("chat", ""):
                print(f"     {str(m['frm']):>2} -> {str(m['to']):<3} [{m['intent']}] {m['text'][:140]}")
        for rep in x["reports"]:
            print(f"     REPORT by {rep['frm']}: {rep['claims']}")
        if x.get("expelled_now"):
            print(f"     EXPELLED: {x['expelled_now']}")


if __name__ == "__main__":
    main()
