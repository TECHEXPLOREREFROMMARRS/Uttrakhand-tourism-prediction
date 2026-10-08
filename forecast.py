"""
Tourist Demand Forecasting for Uttarakhand (+ Himachal Pradesh for comparison)
B.Tech Final Year Project - core pipeline

Run:  py forecast.py                              (Uttarakhand, default)
      py forecast.py --region himachal_pradesh    (Himachal Pradesh)

Input CSV needs only two columns:
    date      -> first day of month, e.g. 2019-05-01
    visitors  -> total tourist visits in that month (domestic + foreign)

Event features (Kumbh, COVID, Char Dham season, Kanwar) are generated
automatically from the date, so you don't have to add them by hand.
"""

import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. Calendar / event features
# ---------------------------------------------------------------------------
# Haridwar Kumbh windows (Ardh Kumbh 2016, Kumbh 2021 [COVID-limited], Kumbh 2027)
KUMBH_MONTHS = (
    [f"2016-{m:02d}" for m in range(1, 5)]
    + [f"2021-{m:02d}" for m in range(3, 5)]
    + [f"2027-{m:02d}" for m in range(1, 5)]
)
# Main COVID disruption months (national lockdown + second wave)
COVID_MONTHS = (
    [f"2020-{m:02d}" for m in range(4, 13)]
    + [f"2021-{m:02d}" for m in range(4, 7)]
)
POST_COVID_START = "2022-01-01"

# ---------------------------------------------------------------------------
# Region settings: the SAME models run for both states, only the event
# features differ. Kumbh is a Haridwar (Uttarakhand) event, so Himachal does
# not use it; Himachal instead gets its own summer and snow-season dummies.
# ---------------------------------------------------------------------------
REGIONS = {
    "uttarakhand": {
        "name": "Uttarakhand",
        "sample_data": "data/sample_monthly_visits.csv",                       # synthetic
        "estimated_data": "data/estimated/uttarakhand_monthly_estimated.csv",  # official annual totals
        "out_dir": "outputs",
        "exog": ["kumbh", "covid", "post_covid"],                 # SARIMAX regressors
        "ml_features": ["month_sin", "month_cos", "year", "kumbh", "covid", "post_covid",
                        "chardham_season", "kanwar", "lag_12"],   # GBM inputs
    },
    "himachal_pradesh": {
        "name": "Himachal Pradesh",
        "sample_data": "data/sample_monthly_visits_hp.csv",
        "estimated_data": "data/estimated/himachal_pradesh_monthly_estimated.csv",
        "out_dir": "outputs/himachal_pradesh",
        "exog": ["covid", "post_covid"],
        "ml_features": ["month_sin", "month_cos", "year", "covid", "post_covid",
                        "summer_peak", "snow_season", "lag_12"],
    },
}
DEFAULT_REGION = "uttarakhand"
EXOG_COLS = REGIONS[DEFAULT_REGION]["exog"]


def add_features(df: pd.DataFrame, region: str = DEFAULT_REGION) -> pd.DataFrame:
    """Add event dummies and calendar features from the date column."""
    df = df.copy()
    ym = df["date"].dt.strftime("%Y-%m")
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    # Kumbh only affects Uttarakhand (Haridwar); always 0 for other states
    df["kumbh"] = (ym.isin(KUMBH_MONTHS) & (region == "uttarakhand")).astype(int)
    df["covid"] = ym.isin(COVID_MONTHS).astype(int)
    df["chardham_season"] = df["month"].between(5, 10).astype(int)
    df["kanwar"] = (df["month"] == 7).astype(int)
    # Himachal: summer escape from the plains (May-Jun) and snow/New-Year (Dec-Jan)
    df["summer_peak"] = df["month"].between(5, 6).astype(int)
    df["snow_season"] = df["month"].isin([12, 1]).astype(int)
    # Structural break: post-COVID footfall settled at a much higher level
    # (better roads, Char Dham registration counting, travel boom)
    df["post_covid"] = (df["date"] >= POST_COVID_START).astype(int)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def load_data(path, region: str = DEFAULT_REGION) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
    if df["visitors"].isna().any():
        df["visitors"] = df["visitors"].interpolate()
    return add_features(df, region)


def future_frame(last_date: pd.Timestamp, months: int,
                 region: str = DEFAULT_REGION) -> pd.DataFrame:
    dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=months, freq="MS")
    return add_features(pd.DataFrame({"date": dates}), region)


# ---------------------------------------------------------------------------
# 2. Models
# ---------------------------------------------------------------------------
def seasonal_naive(train: pd.DataFrame, horizon: pd.DataFrame) -> np.ndarray:
    """Baseline: same month last year (skipping COVID months)."""
    clean = train[train["covid"] == 0]
    last_by_month = clean.groupby("month")["visitors"].last()
    return horizon["month"].map(last_by_month).values


def sarimax_model(train: pd.DataFrame, horizon: pd.DataFrame, return_sims=False,
                  exog_cols=EXOG_COLS):
    """SARIMAX on log(visitors) with event dummies (e.g. Kumbh, COVID) as regressors."""
    y = np.log(train["visitors"].values)
    model = SARIMAX(
        y,
        exog=train[exog_cols].values,
        order=(1, 0, 1),
        seasonal_order=(0, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)
    fc = fit.get_forecast(steps=len(horizon), exog=horizon[exog_cols].values)
    mean = np.exp(fc.predicted_mean)
    ci = np.exp(fc.conf_int(alpha=0.2))  # 80% interval
    if not return_sims:
        return mean, ci[:, 0], ci[:, 1]
    sims = fit.simulate(nsimulations=len(horizon), repetitions=1000, anchor="end",
                        exog=horizon[exog_cols].values)  # Monte-Carlo: interval may vary slightly per run
    sims = np.exp(np.asarray(sims).reshape(len(horizon), -1))
    return mean, ci[:, 0], ci[:, 1], sims


ML_FEATURES = REGIONS[DEFAULT_REGION]["ml_features"]


def gbm_model(train: pd.DataFrame, horizon: pd.DataFrame,
              features=ML_FEATURES) -> np.ndarray:
    """Gradient Boosting with a 12-month lag, forecast recursively."""
    hist = train[["date", "visitors"] + [c for c in train.columns
                                         if c not in ("date", "visitors")]].copy()
    hist["lag_12"] = hist["visitors"].shift(12)
    # Replace COVID-affected lags with the value from 2 years back
    hist["lag_12"] = np.where(hist["covid"].shift(12) == 1,
                              hist["visitors"].shift(24), hist["lag_12"])
    tr = hist.dropna(subset=["lag_12"])
    tr = tr[tr["covid"] == 0]  # learn "normal" behaviour

    gbm = GradientBoostingRegressor(n_estimators=400, max_depth=3,
                                    learning_rate=0.05, subsample=0.9,
                                    random_state=42)
    gbm.fit(tr[features], np.log(tr["visitors"]))

    series = dict(zip(hist["date"], hist["visitors"]))
    covid_flag = dict(zip(hist["date"], hist["covid"]))
    preds = []
    for _, row in horizon.iterrows():
        d12 = row["date"] - pd.DateOffset(years=1)
        d24 = row["date"] - pd.DateOffset(years=2)
        lag = series.get(d24) if covid_flag.get(d12, 0) == 1 else series.get(d12)
        x = row.copy()
        x["lag_12"] = lag
        p = float(np.exp(gbm.predict(pd.DataFrame([x[features]]))[0]))
        preds.append(p)
        series[row["date"]] = p
        covid_flag[row["date"]] = row["covid"]
    return np.array(preds)


# ---------------------------------------------------------------------------
# 3. Evaluation
# ---------------------------------------------------------------------------
def metrics(actual, pred) -> dict:
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    err = actual - pred
    return {
        "MAE (lakh)": np.mean(np.abs(err)) / 1e5,
        "RMSE (lakh)": np.sqrt(np.mean(err ** 2)) / 1e5,
        "MAPE (%)": np.mean(np.abs(err / actual)) * 100,
    }


def region_features(region: str):
    """Return (SARIMAX exog columns, GBM feature list) for a region."""
    cfg = REGIONS[region]
    return cfg["exog"], cfg["ml_features"]


def backtest(df: pd.DataFrame, test_months: int = 24, region: str = DEFAULT_REGION):
    exog, feats = region_features(region)
    train, test = df.iloc[:-test_months], df.iloc[-test_months:]
    preds = {
        "Seasonal Naive": seasonal_naive(train, test),
        "SARIMAX": sarimax_model(train, test, exog_cols=exog)[0],
        "Gradient Boosting": gbm_model(train, test, features=feats),
    }
    preds["Ensemble (SARIMAX+GBM)"] = (preds["SARIMAX"] + preds["Gradient Boosting"]) / 2
    table = pd.DataFrame({name: metrics(test["visitors"], p) for name, p in preds.items()}).T
    return train, test, preds, table.round(2)


def forecast_future(df: pd.DataFrame, months: int = 36, region: str = DEFAULT_REGION):
    exog, feats = region_features(region)
    fut = future_frame(df["date"].iloc[-1], months, region)
    sar, lo, hi, sims = sarimax_model(df, fut, return_sims=True, exog_cols=exog)
    gbm = gbm_model(df, fut, features=feats)
    out = fut[["date", "kumbh"]].copy()
    out["sarimax"] = sar
    out["gbm"] = gbm
    out["ensemble"] = (sar + gbm) / 2
    out["lower_80"] = lo
    out["upper_80"] = hi
    return out, sims


def annual_summary(fc: pd.DataFrame, sims: np.ndarray) -> pd.DataFrame:
    """Annual totals in crore. Interval comes from summing simulated paths,
    which is correct (summing monthly bounds would overstate the range)."""
    years = fc["date"].dt.year.values
    rows = []
    for y in np.unique(years):
        mask = years == y
        if mask.sum() < 12:
            continue
        path_totals = sims[mask].sum(axis=0) / 1e7
        rows.append({"year": y,
                     "forecast_crore": fc.loc[mask, "ensemble"].sum() / 1e7,
                     "lower_80": np.percentile(path_totals, 10),
                     "upper_80": np.percentile(path_totals, 90)})
    return pd.DataFrame(rows).set_index("year").round(2)


# ---------------------------------------------------------------------------
# 4. Plots
# ---------------------------------------------------------------------------
def plot_backtest(train, test, preds, path):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(train["date"], train["visitors"] / 1e5, color="grey", label="Train (actual)")
    ax.plot(test["date"], test["visitors"] / 1e5, color="black", lw=2, label="Test (actual)")
    for name, p in preds.items():
        ax.plot(test["date"], p / 1e5, "--", label=name)
    ax.set_title("Backtest: model predictions vs actual (test period)")
    ax.set_ylabel("Visitors (lakh / month)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_forecast(df, fc, path, region_name="Uttarakhand"):
    fig, ax = plt.subplots(figsize=(12, 5))
    recent = df[df["date"] >= df["date"].iloc[-1] - pd.DateOffset(years=4)]
    ax.plot(recent["date"], recent["visitors"] / 1e5, color="black", label="Actual")
    ax.plot(fc["date"], fc["ensemble"] / 1e5, color="tab:blue", lw=2, label="Forecast (ensemble)")
    ax.fill_between(fc["date"], fc["lower_80"] / 1e5, fc["upper_80"] / 1e5,
                    color="tab:blue", alpha=0.15, label="80% interval (SARIMAX)")
    k = fc[fc["kumbh"] == 1]
    if len(k):
        ax.axvspan(k["date"].min(), k["date"].max() + pd.offsets.MonthEnd(1),
                   color="orange", alpha=0.15, label="Haridwar Kumbh 2027")
    ax.set_title(f"{region_name} monthly tourist demand forecast")
    ax.set_ylabel("Visitors (lakh / month)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default=DEFAULT_REGION, choices=list(REGIONS))
    ap.add_argument("--dataset", default="synthetic", choices=["synthetic", "estimated"],
                    help="synthetic sample, or monthly estimate from official annual totals")
    ap.add_argument("--data", default=None, help="CSV path (overrides --dataset)")
    ap.add_argument("--test_months", type=int, default=24)
    ap.add_argument("--horizon", type=int, default=36)
    ap.add_argument("--out", default=None, help="output folder (default depends on region)")
    args = ap.parse_args()
    cfg = REGIONS[args.region]
    args.data = args.data or cfg["sample_data" if args.dataset == "synthetic" else "estimated_data"]
    if not args.out:
        args.out = cfg["out_dir"] if args.dataset == "synthetic"             else os.path.join("outputs", "estimated", args.region)
    os.makedirs(args.out, exist_ok=True)

    df = load_data(args.data, args.region)
    print(f"[{cfg['name']}] Loaded {len(df)} months: "
          f"{df['date'].min():%b %Y} to {df['date'].max():%b %Y}\n")

    train, test, preds, table = backtest(df, args.test_months, args.region)
    print("Backtest results (lower is better):")
    print(table.to_string(), "\n")
    plot_backtest(train, test, preds, os.path.join(args.out, "backtest.png"))

    fc, sims = forecast_future(df, args.horizon, args.region)
    fc.to_csv(os.path.join(args.out, "forecast.csv"), index=False)
    plot_forecast(df, fc, os.path.join(args.out, "forecast.png"), cfg["name"])

    annual = annual_summary(fc, sims)
    annual.to_csv(os.path.join(args.out, "annual_forecast.csv"))
    print("Annual forecast (crore visitors, 80% interval from SARIMAX simulation):")
    print(annual.to_string())
    print(f"\nSaved plots and forecast.csv to '{args.out}/'")


if __name__ == "__main__":
    main()
