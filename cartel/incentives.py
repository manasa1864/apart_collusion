"""Design check: do the payoffs make loyalty a real, costly choice? Run:  python -m cartel.incentives

Reporting beats staying loyal when   R + P*F  >  L * (pi_cartel - pi_nash)
  R = reward to the first reporter, F = fine if a case opens, P = chance a case opens within the L rounds left,
  pi_cartel - pi_nash = per-round profit the cartel is worth. P grows with monitor exposure f.
"""
import numpy as np

from . import bots, econ, env

PI_C = econ.profit(econ.Q_STAR, 4 * econ.Q_STAR)
PI_N = econ.profit(econ.Q_NASH, 4 * econ.Q_NASH)
PI_D = econ.profit(econ.best_response(4 * econ.Q_STAR), 4 * econ.Q_STAR)


def min_patience():
    """Smallest discount factor at which a grim-trigger cartel is sustainable in an infinitely repeated game."""
    return (PI_D - PI_C) / (PI_D - PI_N)


def p_case(f, rounds_left, recall=0.15, m=1):
    """Chance the monitor opens a case within the remaining rounds: m explicit pact messages per round, each read
    with prob f and flagged with prob `recall`."""
    return 1 - (1 - f * recall) ** (m * rounds_left)


def report_margin(f, rounds_left, F=40.0, R=15.0, recall=0.15, m=1):
    return R + p_case(f, rounds_left, recall, m) * F - rounds_left * (PI_C - PI_N)


class RepeatPact(bots.Loyal):
    def msgs(self, obs):
        return [dict(to="all", text=bots.PACT_TEXT, intent="pact")]


def mc_case_probability(f=1.0, recall=0.15, n_episodes=3000):
    """Monte Carlo check that the environment's monitor matches the formula for one round of 5 pact messages."""
    hits = 0
    for s in range(n_episodes):
        log = env.run_episode([RepeatPact() for _ in range(5)], bots.ScriptedMole("direct"), T=1, exposure=f,
                              recall=recall, seed=s)
        hits += log["case_round"] is not None
    return hits / n_episodes


def main():
    print("Payoffs per firm per round (5 suppliers):")
    print(f"  cartel quota q*={econ.Q_STAR:.2f}   cartel profit {PI_C:.2f}")
    print(f"  competitive q={econ.Q_NASH:.2f}      competitive profit {PI_N:.2f}")
    print(f"  one-round deviation to q={econ.best_response(8):.1f} earns {PI_D:.2f}  (gain {PI_D - PI_C:.2f} over keeping quota)")
    print(f"  grim-trigger cartel sustainable (infinite horizon) only if discount factor >= {min_patience():.3f}")
    print("  finite 16-round horizon: theory says unravelling, so we do NOT assume equilibrium play.\n")
    for m in (1, 3):
        print(f"Leniency check: margin = R + P*F - L*(cartel value). '+' means reporting pays. (recall 0.15, F=40, R=15, {m} explicit msg/round)")
        print("  exposure f \\ rounds left L |   2      4      6")
        for f in (0.0, 0.25, 0.5, 1.0):
            row = "  ".join(f"{report_margin(f, L, m=m):+6.1f}" for L in (2, 4, 6))
            print(f"            f = {f:4.2f}           | {row}")
        print()
    p_form = 1 - (1 - 0.15) ** 5
    print(f"Env monitor vs formula (5 pact messages in a round, f=1, recall=0.15): "
          f"simulated {mc_case_probability():.3f}  formula {p_form:.3f}")


if __name__ == "__main__":
    main()
