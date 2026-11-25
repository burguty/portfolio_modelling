"""Produce daily covariance forecasts aligned to the return model's target dates."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


PROJECT = Path(__file__).resolve().parents[1]
WINDOWS = (63, 126, 252)
HALF_LIVES = (10, 21, 42, 63)
MODELS = ("sample", "ledoit_wolf", "ewma")


def read_returns(path):
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    if (
        frame.empty
        or not isinstance(frame.index, pd.DatetimeIndex)
        or frame.index.hasnans
        or not frame.index.is_unique
        or not frame.index.is_monotonic_increasing
        or not np.isfinite(frame.to_numpy(dtype=float)).all()
    ):
        raise ValueError(f"Expected finite data and unique, sorted dates in {path}")
    return frame


def forecast(returns, dates, estimator, window, half_life=None):
    positions = returns.index.get_indexer(dates)
    if (positions < window).any():
        raise ValueError("A forecast target is missing or has insufficient return history.")
    values = returns.to_numpy()
    covariances = []
    if estimator == "ewma":
        weights = 2.0 ** (-np.arange(window - 1, -1, -1) / half_life)
        weights /= weights.sum()
        correction = 1 - weights @ weights

    for position in positions:
        # The target-date return is never part of the estimation window.
        history = values[position - window:position]
        if estimator == "sample":
            matrix = np.cov(history, rowvar=False, ddof=1)
        elif estimator == "ledoit_wolf":
            matrix = LedoitWolf().fit(history).covariance_
        else:
            centered = history - weights @ history
            matrix = (centered.T * weights) @ centered / correction
        covariances.append((matrix + matrix.T) / 2)

    return (
        np.stack(covariances),
        returns.index[positions - window],
        returns.index[positions - 1],
    )


def score(matrices, realized):
    # Compare all models under the same zero-mean assumption for scoring.
    cholesky = np.linalg.cholesky(matrices)
    whitened = np.linalg.solve(cholesky, realized[..., None])[..., 0]
    logdet = 2 * np.log(np.diagonal(cholesky, axis1=1, axis2=2)).sum(axis=1)

    eigenvalues = np.linalg.eigvalsh(matrices)
    # The average entry equals w.T @ covariance @ w for equal weights.
    variance = matrices.mean(axis=(1, 2))
    squared_return = realized.mean(axis=1) ** 2
    return pd.DataFrame({
        "gaussian_loss": (logdet + (whitened ** 2).sum(axis=1)) / realized.shape[1],
        "equal_weight_qlike": np.log(variance) + squared_return / variance,
        "equal_weight_variance": variance,
        "equal_weight_squared_return": squared_return,
        "condition_number": eigenvalues[:, -1] / eigenvalues[:, 0],
        "minimum_eigenvalue": eigenvalues[:, 0],
    })


def select_parameters(returns, validation_dates):
    candidates = [
        (model, window, None)
        for model in MODELS[:2]
        for window in WINDOWS
    ] + [("ewma", max(WINDOWS), half_life) for half_life in HALF_LIVES]
    records = []
    realized = returns.loc[validation_dates].to_numpy()
    for model, window, half_life in candidates:
        matrices, _, _ = forecast(returns, validation_dates, model, window, half_life)
        metrics = score(matrices, realized)
        records.append({
            "model": model,
            "window": window,
            "half_life": half_life,
            "mean_gaussian_loss": metrics["gaussian_loss"].mean(),
        })
    scores = pd.DataFrame(records)
    winners = scores.groupby("model", sort=False)["mean_gaussian_loss"].idxmin()
    scores["selected_for_test"] = scores.index.isin(winners)
    selected = [candidates[index] for index in winners]
    preferred = scores.loc[winners].sort_values("mean_gaussian_loss").iloc[0]["model"]
    return scores, selected, preferred


def plot_diagnostics(diagnostics, path):
    fig, (volatility, conditioning) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True, layout="constrained"
    )
    labels = dict(zip(MODELS, ("Sample covariance", "Ledoit–Wolf", "EWMA")))
    for model, rows in diagnostics.groupby("model", sort=False):
        volatility.plot(rows["target_date"], np.sqrt(rows["equal_weight_variance"]),
                        label=labels[model], linewidth=1)
        conditioning.plot(rows["target_date"], rows["condition_number"], linewidth=1)
    volatility.set(title="Equal-weight portfolio risk forecast", ylabel="Daily volatility")
    volatility.yaxis.set_major_formatter(PercentFormatter(1))
    volatility.legend(frameon=False, ncol=3)
    conditioning.set(ylabel="Condition number", yscale="log", xlabel="Target date")
    for axis in (volatility, conditioning):
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=PROJECT / "data/processed")
    parser.add_argument("--predictions", type=Path,
                        default=PROJECT / "data/predictions/baseline_oos_predictions.csv")
    parser.add_argument("--out", type=Path, default=PROJECT / "data/risk")
    args = parser.parse_args()

    returns = read_returns(args.processed / "daily_returns.csv")
    validation = read_returns(args.processed / "valid_returns.csv")
    test = read_returns(args.processed / "test_returns.csv")
    predictions = read_returns(args.predictions)
    if (
        not returns.columns.equals(predictions.columns)
        or not returns.columns.equals(validation.columns)
        or not returns.columns.equals(test.columns)
        or not validation.index.isin(returns.index).all()
        or not predictions.index.isin(test.index).all()
        or validation.index[-1] >= test.index[0]
    ):
        raise ValueError("Return inputs and predictions must share ordered tickers and valid splits.")

    validation_scores, selected, preferred = select_parameters(returns, validation.index)
    dates = predictions.index
    realized = returns.loc[dates].to_numpy()
    forecasts, reports = {}, []
    for model, window, half_life in selected:
        matrices, start, end = forecast(returns, dates, model, window, half_life)
        forecasts[model] = matrices
        reports.append(score(matrices, realized).assign(
            model=model, target_date=dates, history_start=start, history_end=end,
        ))

    diagnostics = pd.concat(reports, ignore_index=True)
    summary = diagnostics.groupby("model", sort=False).agg(
        forecast_days=("target_date", "size"),
        mean_gaussian_loss=("gaussian_loss", "mean"),
        mean_equal_weight_qlike=("equal_weight_qlike", "mean"),
        mean_predicted_variance=("equal_weight_variance", "mean"),
        mean_realized_squared_return=("equal_weight_squared_return", "mean"),
        median_condition_number=("condition_number", "median"),
        minimum_eigenvalue=("minimum_eigenvalue", "min"),
    )
    summary["preferred_on_validation"] = summary.index == preferred

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out / "covariance_forecasts.npz",
        dates=dates.to_numpy(dtype="datetime64[D]"),
        tickers=returns.columns.to_numpy(dtype=str),
        preferred_model=preferred,
        **forecasts,
    )
    validation_scores.to_csv(args.out / "validation_scores.csv", index=False)
    diagnostics.to_csv(args.out / "forecast_diagnostics.csv", index=False)
    summary.to_csv(args.out / "test_summary.csv")
    plot_diagnostics(diagnostics, args.out / "risk_diagnostics.png")
    print(f"Validation-selected estimator: {preferred}")
    print(summary.to_string())


if __name__ == "__main__":
    main()
