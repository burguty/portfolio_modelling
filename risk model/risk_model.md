# Daily covariance forecasts

This implements Part 2 on page 9 of the STA4020 project guideline for the return model's fixed 30-stock CSI 500 universe. Run from the team repository root with NumPy, pandas, scikit-learn, and Matplotlib installed.

```bash
python "risk model/risk_model.py"
```

## Estimation and validation

Each covariance forecasts one trading day's decimal returns. For target date (t), the estimation window ends on the preceding trading day. At the start of validation and testing, the window can draw on the previous split's observed returns. The forecast dates and ticker order come from the saved return predictions. The resulting test period runs from January 4, 2019 through August 30, 2024.

- **Sample covariance** uses rolling de-meaned returns and divides by the window length minus one.
- **Ledoit–Wolf** shrinks the rolling empirical covariance toward a scaled identity. The estimator calculates shrinkage from each historical window.
- **EWMA covariance** uses a 252-day rolling window. Observations one half-life older get half the weight. The estimator subtracts the weighted mean and divides the weighted cross-products by one minus the sum of squared normalized weights.

The 2018 validation split selects among 63, 126, and 252 trading days for sample and Ledoit–Wolf covariance, and among 10, 21, 42, and 63 trading days for the EWMA half-life. Lower mean Gaussian covariance loss determines the parameters for each model and the preferred model. Parameters are then kept fixed during testing while covariance estimates update daily.

## Evaluation and portfolio handoff

The Gaussian loss per asset combines the log determinant and the squared Mahalanobis length of the realized return vector under a common zero-mean scoring assumption. Equal-weight QLIKE compares forecast variance with the squared realized equal-weight return, which is a noisy single-day variance proxy. Lower scores are better within each metric. The condition number tracks numerical stability and the minimum eigenvalue checks positive definiteness.

Outputs are written to `data/risk/`. The archive stores daily 30-by-30 matrices for all three models in squared decimal-return units. The diagnostic CSVs give validation scores, forecast-by-forecast estimation dates and losses, and the test summary. The figure shows forecast equal-weight volatility and matrix conditioning over time.

```python
import numpy as np

with np.load("data/risk/covariance_forecasts.npz") as risk:
    dates = risk["dates"]
    tickers = risk["tickers"]
    daily_covariances = risk[risk["preferred_model"].item()]
```

Use matching target dates and ticker order for the expected-return vectors and covariance matrices. The portfolio's daily variance for weights `w` is `w @ daily_covariances[i] @ w`.
