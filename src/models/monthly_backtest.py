"""Evaluate portfolio weights reset on the first test trading day of each month."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from risk_model import PROJECT, read_returns


def monthly_backtest(returns, weights):
    months = returns.groupby(returns.index.to_period("M"), sort=False)
    rebalance_dates = pd.DatetimeIndex(
        [period.index[0] for _, period in months], name="rebalance_date"
    )
    if not weights.index.equals(rebalance_dates) or not weights.columns.equals(returns.columns):
        raise ValueError("Weights must match the first test trading day of each month and ordered tickers.")
    if not np.allclose(weights.sum(axis=1), 1, rtol=0, atol=1e-8):
        raise ValueError("Weights must sum to one at each rebalance.")

    records = []
    for (_, period), (date, allocation) in zip(months, weights.iterrows(), strict=True):
        # Fixed holdings between rebalances let asset weights drift with returns.
        asset_returns = (1 + period).prod() - 1
        records.append({
            "rebalance_date": date,
            "period_end": period.index[-1],
            "trading_days": len(period),
            "portfolio_return": float(allocation @ asset_returns),
        })

    results = pd.DataFrame(records).set_index("rebalance_date")
    results["cumulative_return"] = (1 + results["portfolio_return"]).cumprod() - 1
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-returns", type=Path, default=PROJECT / "data/processed/test_returns.csv")
    parser.add_argument("--weights", type=Path, help="CSV of rebalance dates and portfolio weights. Defaults to equal weights.")
    parser.add_argument("--out", type=Path, default=PROJECT / "data/backtest/monthly_returns.csv")
    args = parser.parse_args()

    returns = read_returns(args.test_returns)
    if args.weights is None:
        dates = returns.index[~returns.index.to_period("M").duplicated()]
        weights = pd.DataFrame(1 / len(returns.columns), index=dates, columns=returns.columns)
    else:
        weights = read_returns(args.weights)
    results = monthly_backtest(returns, weights)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out)


if __name__ == "__main__":
    main()
