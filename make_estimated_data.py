"""
Builds ESTIMATED monthly series from OFFICIAL annual totals:
    data/estimated/official_annual_totals.csv   (input, every number cited)
 -> data/estimated/uttarakhand_monthly_estimated.csv
 -> data/estimated/himachal_pradesh_monthly_estimated.csv

What is real and what is assumed
    REAL     : the annual total for each year (domestic + foreign tourist
               visits, Ministry of Tourism - India Tourism Statistics /
               India Tourism Data Compendium). Each monthly file adds up to
               exactly these totals.
    ASSUMED  : how each year's total is split across the 12 months
               (an average seasonal profile + COVID / Kumbh adjustments).
               Official monthly data is not freely published; replace these
               files with real monthly data (indiastat / RTI) when you have it.

Method (temporal disaggregation, pro-rata to an indicator profile):
    monthly = annual_total x weight_m / sum(weights)
    weight_m = average seasonal share of month m x event factor
No random noise is added, so the result is fully reproducible.

Run:  py make_estimated_data.py
"""
import os

import numpy as np
import pandas as pd

ANNUAL_FILE = "data/estimated/official_annual_totals.csv"

# Average share of the year's visitors in each month (Jan..Dec).
# ASSUMPTION based on the known peaks of each state (same profiles as the
# synthetic generator). Replace with shares computed from real monthly data.
PROFILE = {
    # Peaks: May-Jun (Char Dham opening + holidays), Jul (Kanwar), Oct
    "uttarakhand": [.060, .055, .070, .080, .130, .130, .090, .060, .070, .100, .080, .075],
    # Peaks: May-Jun (summer), Dec-Jan (snow / New Year); monsoon dip Jul-Aug
    "himachal_pradesh": [.085, .060, .065, .080, .120, .125, .065, .055, .065, .085, .080, .115],
}


def event_factors(region: str, year: int) -> np.ndarray:
    """Multipliers for known shocks inside a year (1.0 = normal month)."""
    f = np.ones(12)
    if year == 2020:
        f[3:12] *= 0.15            # national lockdown from late March 2020
    if year == 2021:
        if region == "uttarakhand":
            f[2:4] *= 1.15         # Haridwar Kumbh (Mar-Apr 2021, restricted)
        f[3:6] *= 0.25             # second wave (Apr-Jun 2021)
    return f


def build(region: str, annual: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in annual[annual["region"] == region].sort_values("year").iterrows():
        total = r["domestic_visits"] + r["foreign_visits"]
        w = np.array(PROFILE[region]) * event_factors(region, int(r["year"]))
        monthly = total * w / w.sum()
        monthly = np.round(monthly).astype(int)
        monthly[-1] += total - monthly.sum()      # fix rounding so the year adds up exactly
        for m in range(12):
            rows.append({"date": f"{int(r['year'])}-{m + 1:02d}-01", "visitors": int(monthly[m])})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    annual = pd.read_csv(ANNUAL_FILE)
    os.makedirs("data/estimated", exist_ok=True)
    for region in PROFILE:
        out = build(region, annual)
        path = f"data/estimated/{region}_monthly_estimated.csv"
        out.to_csv(path, index=False)
        yearly = out.assign(year=out["date"].str[:4]).groupby("year")["visitors"].sum() / 1e7
        print(f"Wrote {path}  ({len(out)} months)  annual totals in crore:")
        print("   " + "  ".join(f"{y}: {v:.2f}" for y, v in yearly.items()))
    print("\nANNUAL totals are official (Ministry of Tourism); the MONTHLY split is an estimate.")
