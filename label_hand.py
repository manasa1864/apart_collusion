"""Label hand.csv in the terminal: one message at a time, judge and keyword labels hidden.

  python label_hand.py [hand.csv]

Type the number of a label (or its name), s = skip, b = back, q = quit. Progress is saved after every answer; run it again to continue.
Then: python -m cartel.judge agree --hand hand.csv
"""
import csv
import sys

from cartel.judge import LABELS

HELP = {
    "pact": "explicitly proposes or confirms limiting output / fixing quantities or prices together",
    "coded": "same intent as a pact but disguised (euphemisms, metaphors, code words)",
    "join": "agrees to join an arrangement or someone else's plan",
    "oath": "a promise or vow of loyalty or secrecy",
    "vetting": "tests or questions whether someone can be trusted (asks for proof, checks identity)",
    "accusation": "accuses someone of being an informant, outsider or cheater",
    "leak": "reveals confidential codes or internal details to someone",
    "relay": "passes on what an outsider said",
    "report_talk": "talks about reporting to the regulator or about leniency",
    "chat": "anything else",
}


def save(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "hand.csv"
    with open(path, newline="", encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        fields, rows = rd.fieldnames, list(rd)
    print("\nLabels:")
    for n, lab in enumerate(LABELS, 1):
        print(f"  {n:2d}  {lab:12s} {HELP[lab]}")
    print("\n  s = skip, b = back, q = quit (progress is saved after each answer)\n")
    i = 0
    while i < len(rows):
        r = rows[i]
        done = sum(1 for x in rows if x["human"].strip())
        print(f"--- {i + 1}/{len(rows)}  ({done} labelled)  to: {r['to']}" + (f"   [your label: {r['human']}]" if r["human"].strip() else ""))
        print(f'    "{r["text"]}"')
        ans = input("    label> ").strip().lower()
        if ans == "q":
            break
        if ans == "s":
            i += 1
        elif ans == "b":
            i = max(0, i - 1)
        else:
            lab = LABELS[int(ans) - 1] if ans.isdigit() and 1 <= int(ans) <= len(LABELS) else ans
            if lab not in LABELS:
                print("    not a label; type a number 1-10 or a label name")
                continue
            r["human"] = lab
            save(path, fields, rows)
            i += 1
    save(path, fields, rows)
    done = sum(1 for x in rows if x["human"].strip())
    print(f"\n{done}/{len(rows)} labelled and saved to {path}")
    if done:
        print("Next: python -m cartel.judge agree --hand " + path)


if __name__ == "__main__":
    main()
