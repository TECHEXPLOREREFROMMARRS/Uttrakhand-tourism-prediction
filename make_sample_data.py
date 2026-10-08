"""
Creates SYNTHETIC monthly data for testing the pipeline ONLY:
    data/sample_monthly_visits.csv      -> Uttarakhand
    data/sample_monthly_visits_hp.csv   -> Himachal Pradesh

Uttarakhand: annual totals for 2022-2025 match the Uttarakhand Tourism
Department's published figures. Annual totals for 2015-2021 and ALL monthly
splits are approximate placeholders.

Himachal Pradesh: EVERY number (annual totals and monthly splits) is a
placeholder. Replace with official HP Tourism Department figures before
reporting anything.

Replace these files with real monthly data (Tourism Depts / indiastat / RTI)
before you report results.
"""
import os
import numpy as np
import pandas as pd

# ---------------- Uttarakhand ----------------
ANNUAL_CRORE = {
    2015: 2.95, 2016: 3.20, 2017: 3.47, 2018: 3.68, 2019: 3.90,   # placeholders
    2020: 0.85, 2021: 2.00,                                          # placeholders
    2022: 5.40, 2023: 5.96, 2024: 5.96, 2025: 6.03,                  # official totals
}
# Typical share of annual visits by month (peaks: May-Jun, Oct)
MONTH_SHARE = np.array([.060, .055, .070, .080, .130, .130,
                        .090, .060, .070, .100, .080, .075])

# ---------------- Himachal Pradesh (ALL PLACEHOLDERS) ----------------
ANNUAL_CRORE_HP = {
    2015: 1.8, 2016: 1.9, 2017: 2.0, 2018: 1.6, 2019: 1.7,
    2020: 0.3, 2021: 0.6,
    2022: 1.5, 2023: 1.6, 2024: 1.8, 2025: 1.9,
}
# Peaks: May-Jun (summer escape from the plains), Dec-Jan (snow, New Year);
# dip in Jul-Aug (monsoon landslides)
MONTH_SHARE_HP = np.array([.085, .060, .065, .080, .120, .125,
                           .065, .055, .065, .085, .080, .115])


def make_series(annual, month_share, seed, kumbh_2016=False, kumbh_2021=False):
    """Split each annual total into 12 months using the share profile plus
    a little random noise, and apply COVID (and Kumbh) shocks."""
    rng = np.random.default_rng(seed)
    rows = []
    for year, total in annual.items():
        w = month_share * rng.normal(1, 0.04, 12)
        if year == 2016 and kumbh_2016:
            w[0:4] *= 1.35           # Ardh Kumbh
        if year == 2020:
            w[3:12] *= 0.15          # lockdown
        if year == 2021:
            if kumbh_2021:
                w[2:4] *= 1.15       # restricted Kumbh
            w[3:6] *= 0.25           # second wave
        w = w / w.sum()
        for m in range(12):
            rows.append({"date": f"{year}-{m + 1:02d}-01",
                         "visitors": int(round(total * 1e7 * w[m]))})
    return pd.DataFrame(rows)


os.makedirs("data", exist_ok=True)
make_series(ANNUAL_CRORE, MONTH_SHARE, seed=7, kumbh_2016=True, kumbh_2021=True) \
    .to_csv("data/sample_monthly_visits.csv", index=False)
print("Wrote data/sample_monthly_visits.csv (SYNTHETIC - replace with real data)")

make_series(ANNUAL_CRORE_HP, MONTH_SHARE_HP, seed=11) \
    .to_csv("data/sample_monthly_visits_hp.csv", index=False)
print("Wrote data/sample_monthly_visits_hp.csv (SYNTHETIC placeholders - replace with real HP data)")
