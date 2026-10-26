#!/usr/bin/env python3
"""Full analysis of a cartel-sting run: hypotheses H1-H16, 20 figures, tables (CSV/Markdown/LaTeX) and report.md.

  python analyze_cartel.py --results results/groq/results.jsonl --out analysis/groq [--labels results/groq/labels.jsonl]
  python analyze_cartel.py --results results/sim/results.jsonl --out analysis/sim          # synthetic: adds the recovery check

Works on partial data: anything that was not collected yet shows as 'not run' and its figure is skipped.
"""
import argparse
import os
import time
import warnings

warnings.filterwarnings("ignore")

from cartel import analysis as A, figures as FG, report as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--labels", default=None, help="judge labels (python -m cartel.judge label ...)")
    ap.add_argument("--out", default="analysis/run")
    ap.add_argument("--no-power", action="store_true", help="skip the (slower) power curves")
    a = ap.parse_args()
    t0 = time.time()
    recs = A.load(a.results)
    D = A.build(recs, A.load_labels(a.labels))
    synthetic = any(r.get("backend") in ("sim", "fake") for r in recs)
    print(f"loaded {len(recs)} episodes ({'SYNTHETIC' if synthetic else 'model'} data, labels: {D['src']}) in {time.time() - t0:.1f}s")
    H = A.test_all(D)
    figdir, tabdir = os.path.join(a.out, "figures"), os.path.join(a.out, "tables")
    F = FG.Fig(figdir, synthetic)
    steps = [FG.f1_timeline, FG.f2_cells_over_time, FG.f3_event_aligned, FG.f4_snitching, FG.f5_price_of_loyalty, FG.f6_loyalty_structure,
             FG.f7_survival, FG.f8_watched_2x2, FG.f9_rumour, FG.f10_network_theory, FG.f11_shocks, lambda F, D: FG.f12_hypotheses(F, D, H[H.verdict != "not run"]),
             FG.f14_mole, FG.f15_incentives, FG.f16_harm, FG.f17_phase_heatmap, FG.f18_classifier, FG.f21_by_model]
    if not a.no_power:
        steps.append(FG.f19_power)
    for f in steps:
        try:
            f(F, D)
        except Exception as e:                                 # noqa: BLE001  one broken figure must not kill the report
            print(f"  figure {getattr(f, '__name__', 'f12')} skipped: {type(e).__name__}: {e}")
    rec = None
    if synthetic and "true_trait_market" in D["ag"].columns:
        rec = R.recovery(D, H)
        FG.f20_recovery(F, D, rec)
    tables = []
    for name, df, cap in [("T1_hypotheses", R.hypothesis_table(H), "Pre-registered hypotheses: estimates, 95% CIs, raw and Holm p, verdicts."),
                          ("T2_cells", R.cell_table(D["ep"]), "Outcomes by experimental cell (mean [95% CI])."),
                          ("T3_webs", R.web_table(D["ep"]), "Outcomes by communication web with the network-theory predictions (N1-N5)."),
                          ("T7_by_model", R.model_table(D["ep"]), "Outcomes by model and cell (mean [95% CI])."),
                          ("T4_loyalty", R.loyalty_table(D["ag"]), "Loyalty components, their link to the index, Cronbach alpha and PC1 share."),
                          ("T5_data_quality", R.quality_table(D, os.path.join(os.path.dirname(a.results), "ledger.json")),
                           "Data quality and API usage.")] + ([("T6_recovery", rec, "Synthetic recovery check: planted vs recovered.")] if rec is not None else []):
        if df is not None and len(df):
            R.write_table(df, tabdir, name, cap)
            tables.append((name, cap))
    D["ep"].to_csv(os.path.join(tabdir, "episodes.csv"), index=False)
    D["ag"].to_csv(os.path.join(tabdir, "agents.csv"), index=False)
    D["rd"].to_csv(os.path.join(tabdir, "rounds.csv"), index=False)
    R.build_report(a.out, D, H, F.made, tables, synthetic, rec)
    print(f"{len(F.made)} figures, {len(tables)} tables -> {a.out}/report.md  ({time.time() - t0:.0f}s)")
    print(H[["H", "estimate", "ci_low", "ci_high", "p_holm", "verdict"]].round(3).to_string(index=False))
    if rec is not None:
        print(f"\nrecovery: {int(rec['pass'].sum())}/{len(rec)} checks pass")
        print(rec[["quantity", "planted", "recovered", "lo", "hi", "pass"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
