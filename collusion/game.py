"""Repeated Bertrand pricing game (the shared testbed for all four sub-questions).

N identical sellers choose an integer price in 1..10 each round. Unit cost is 1.
DEMAND customers each buy one unit from the cheapest seller (ties split equally).

Competitive benchmark (Nash on this grid): price 2.  Joint-monopoly price: 10.
Because customers always buy (reservation price 10), total welfare is constant;
collusion is a pure transfer from consumers to sellers, so we track consumer surplus.
"""
import numpy as np

COST = 1
MAX_PRICE = 10
DEMAND = 100
P_NASH = 2
P_MONO = 10


def play_round(prices):
    """prices: list of ints. Returns (quantities, profits, consumer_surplus)."""
    m = min(prices)
    winners = [i for i, p in enumerate(prices) if p == m]
    q = [0.0] * len(prices)
    for i in winners:
        q[i] = DEMAND / len(winners)
    profit = [(p - COST) * qi for p, qi in zip(prices, q)]
    cs = DEMAND * (MAX_PRICE - m)
    return q, profit, cs


def collusion_index(mean_price):
    """0 = competitive (price 2), 1 = full monopoly pricing (price 10)."""
    return float(np.clip((mean_price - P_NASH) / (P_MONO - P_NASH), 0.0, 1.0))


def joint_profit_share(total_profit, rounds):
    """Joint profit of the two strategic sellers relative to the monopoly maximum."""
    return total_profit / (rounds * (P_MONO - COST) * DEMAND)
