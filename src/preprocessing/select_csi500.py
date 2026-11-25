"""Select 30 CSI 500 stocks at December 2018 formation.

Run with `python select_csi500.py` in an environment with pandas installed.
By default, selection uses the formation snapshots and preceding 504 close dates.
Use `--survival-screen` to compare with a screen based on later prices.
"""

import argparse
import random
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"
FORMATION_DATE = pd.Timestamp("2018-12-31")
TEST_START = pd.Timestamp("2019-01-02")
TEST_END = pd.Timestamp("2024-08-30")
LOOKBACK_SESSIONS = 504
MAX_UNTRADABLE_SHARE = 0.05
STOCKS_PER_TIER = 10
SEED = 4020
TIERS = ("Small", "Mid", "Large")


def flagged_session_counts(filename, sessions):
    pairs = pd.read_pickle(DATA_DIR / filename).index
    dates = pairs.get_level_values("date")
    window_pairs = pairs[dates.isin(sessions)].drop_duplicates()
    return window_pairs.get_level_values("asset").value_counts()


def select_stocks(survival_screen=None):
    membership = pd.read_pickle(DATA_DIR / "csi500_mask_monthly.pkl")
    snapshot = membership.loc[FORMATION_DATE]
    members = snapshot[snapshot].index

    prices = pd.read_pickle(DATA_DIR / "daily_adj_close_post_20260630.pkl")
    history = (
        prices.loc[prices.index <= FORMATION_DATE]
        .sort_index()
        .tail(LOOKBACK_SESSIONS)
        .reindex(columns=members)
    )
    sessions = history.index
    if len(sessions) != LOOKBACK_SESSIONS or not sessions.is_unique:
        raise ValueError("Expected 504 distinct trading sessions at formation.")

    blacklist_counts = flagged_session_counts("BLACKLIST.pkl", sessions)
    untradable_counts = flagged_session_counts("UNTRADABLE.pkl", sessions)
    after_blacklist = members.difference(blacklist_counts.index)
    untradable_share = (
        untradable_counts.reindex(after_blacklist, fill_value=0) / len(sessions)
    )
    after_untradable = untradable_share[
        untradable_share <= MAX_UNTRADABLE_SHARE
    ].index
    complete_history = history[after_untradable].notna().all()
    eligible = complete_history[complete_history].index
    counts = {
        "CSI 500 members": len(members),
        "After blacklist screen": len(after_blacklist),
        "After untradability screen": len(after_untradable),
        "After historical price screen": len(eligible),
    }

    if survival_screen is not None:
        test_prices = prices.loc[TEST_START:TEST_END, eligible]
        if survival_screen == "end-price":
            survives = test_prices.loc[TEST_END].notna()
        else:
            survives = test_prices.notna().all()
        eligible = survives[survives].index
        counts["After test-period survival screen"] = len(eligible)

    cap = pd.read_pickle(DATA_DIR / "mcap.pkl").loc[FORMATION_DATE].reindex(eligible)
    if cap.isna().any():
        raise ValueError("Some eligible stocks lack a formation-date market cap.")
    ranked = (
        cap.rename_axis("ticker")
        .rename("market_cap")
        .reset_index()
        .sort_values(["market_cap", "ticker"])
    )
    base_size, remainder = divmod(len(ranked), len(TIERS))
    if base_size < STOCKS_PER_TIER:
        raise ValueError("Not enough eligible stocks for 10 picks per tier.")

    rng = random.Random(SEED)
    picks = {}
    tier_sizes = {}
    start = 0
    for index, label in enumerate(TIERS):
        size = base_size + int(index < remainder)
        population = ranked["ticker"].iloc[start : start + size].tolist()
        draw = set(rng.sample(population, STOCKS_PER_TIER))
        picks[label] = [ticker for ticker in population if ticker in draw]
        tier_sizes[label] = size
        start += size

    return sessions, counts, tier_sizes, picks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--survival-screen",
        choices=("end-price", "complete-prices"),
        help="Reproduce the comparison using future price availability.",
    )
    args = parser.parse_args()
    sessions, counts, tier_sizes, picks = select_stocks(args.survival_screen)
    print(f"Formation: {FORMATION_DATE.date()}")
    print(f"History: {sessions[0].date()} to {sessions[-1].date()} ({len(sessions)} sessions)")
    print(f"Seed: {SEED} (Python random.Random)")
    if args.survival_screen:
        print(f"Survival screen: {args.survival_screen}, {TEST_START.date()} to {TEST_END.date()}")
    for label, count in counts.items():
        print(f"{label}: {count}")
    for tier, tickers in picks.items():
        print(f"{tier} ({len(tickers)} of {tier_sizes[tier]}): {', '.join(tickers)}")


if __name__ == "__main__":
    main()
