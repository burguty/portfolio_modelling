import pandas as pd
import numpy as np
from pathlib import Path

SMALL_CAP = ["600169.SH", "000049.SZ", "600428.SH", "002064.SZ", "000501.SZ", "603000.SH", "000712.SZ", "600580.SH", "600017.SH", "603355.SH"]
MID_CAP = ["600138.SH", "000718.SZ", "002444.SZ", "600597.SH", "002285.SZ", "600729.SH", "000062.SZ", "002463.SZ", "002603.SZ", "603883.SH"]
LARGE_CAP = ["600718.SH", "600507.SH", "002583.SZ", "601718.SH", "600835.SH", "002268.SZ", "002092.SZ", "600098.SH", "002340.SZ", "300001.SZ"]

ASSETS = SMALL_CAP + MID_CAP + LARGE_CAP

DATA = Path("./data")
RAW_DATA = DATA / "raw"
NUM_ASSETS = len(ASSETS)

# Prices and returns
daily_prices = pd.read_pickle(RAW_DATA / "daily_adj_close_post_20260630.pkl")
monthly_returns = pd.read_pickle(RAW_DATA / "monthly_returns.pkl")
prices = pd.read_hdf(RAW_DATA / "price.h5", key="price")

# Trading characteristics
daily_amount = pd.read_pickle(RAW_DATA / "daily_amount.pkl")
daily_turnover = pd.read_pickle(RAW_DATA / "daily_turnover.pkl")
daily_volume = pd.read_pickle(RAW_DATA / "daily_volume.pkl")

# Company characteristics
mcap = pd.read_pickle(RAW_DATA / "mcap.pkl")
industry = pd.read_pickle(RAW_DATA / "ind_code.pkl")

# Factor returns
ff5 = pd.read_csv(RAW_DATA / "FF5.csv", index_col=0, parse_dates=True)
hxz = pd.read_csv(RAW_DATA / "HXZ.csv", index_col=0, parse_dates=True)

# Risk-free rates
risk_free = pd.read_csv(RAW_DATA / "risk_free.csv")
shibor = pd.read_pickle(RAW_DATA / "shibor_rates.pkl")

# Stock selection and trading constraints
blacklist = pd.read_pickle(RAW_DATA / "BLACKLIST.pkl")
untradable = pd.read_pickle(RAW_DATA / "UNTRADABLE.pkl")
csi500_mask = pd.read_pickle(RAW_DATA / "csi500_mask_monthly.pkl")


INIT_DATE = pd.Timestamp("2015-05-13")
LAST_DATE = pd.Timestamp("2024-08-30")

TRAIN_START = INIT_DATE
TRAIN_END = pd.Timestamp("2017-12-31")

VAL_START = pd.Timestamp("2018-01-01")
VAL_END = pd.Timestamp("2018-12-31")

TEST_START = pd.Timestamp("2019-01-01")
TEST_END = LAST_DATE

daily_prices = daily_prices.loc[:, ASSETS].copy()
daily_prices.index = pd.to_datetime(daily_prices.index)
daily_prices.index.name = "date"
daily_prices.columns.name = "asset"
daily_prices = daily_prices.sort_index().loc[INIT_DATE:LAST_DATE]
daily_returns = daily_prices.pct_change(fill_method=None).iloc[1:]

train_returns = daily_returns.loc[TRAIN_START:TRAIN_END]
valid_returns = daily_returns.loc[VAL_START:VAL_END]
test_returns = daily_returns.loc[TEST_START:TEST_END]

for name, df in [
    ("Train", train_returns),
    ("Valid", valid_returns),
    ("Test", test_returns)
]:
    print(
        f"{name:12s} | "
        f"{df.index.min().date()} - {df.index.max().date()} | "
        f"{df.shape}"
    )

PROCESSED_DATA = DATA / "processed"
PROCESSED_DATA.mkdir(parents=True, exist_ok=True)

# Check data quality
assert np.isfinite(daily_returns.to_numpy()).all()
assert daily_returns.columns.tolist() == ASSETS

# Save processed data
daily_prices.to_csv(PROCESSED_DATA / "daily_prices.csv")
daily_returns.to_csv(PROCESSED_DATA / "daily_returns.csv")

train_returns.to_csv(PROCESSED_DATA / "train_returns.csv")
valid_returns.to_csv(PROCESSED_DATA / "valid_returns.csv")
test_returns.to_csv(PROCESSED_DATA / "test_returns.csv")

print(f"Processed data saved to {PROCESSED_DATA}")


