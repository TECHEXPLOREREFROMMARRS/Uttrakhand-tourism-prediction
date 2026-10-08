"""
Tourist hotspots for Uttarakhand and Himachal Pradesh, and a simple
TOP-DOWN method to split a state-level monthly forecast into hotspots.

How the split works (top-down disaggregation):
    hotspot visitors in month m =
        state forecast in month m
        x  weight of hotspot  x  seasonal profile of hotspot in month m  x  event boost
        /  (sum of the same product over all hotspots + "rest of the state")

So the hotspots always add up to (less than) the state total, the state
forecast stays the single source of truth, and each hotspot gets its own
seasonality (e.g. Kedarnath is closed in winter, Manali peaks in snow season).

WARNING: latitude/longitude are real locations, but the WEIGHTS and seasonal
PROFILES below are illustrative PLACEHOLDERS, not official statistics.
Replace them with district/destination-wise shares from the Tourism
Departments (or indiastat) before reporting results.

Run:  py hotspots.py    -> writes outputs/hotspot_forecast.csv for both states
"""
import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Seasonal profiles: relative pull of a place in each month (Jan..Dec).
#    They are normalised to an average of 1 below, so only the SHAPE matters.
# ---------------------------------------------------------------------------
PROFILES = {
    # Year-round Ganga pilgrimage towns; July spike = Kanwar Yatra
    "ganga_pilgrim": [1.0, 1.0, 1.1, 1.1, 1.2, 1.2, 1.6, 0.8, 0.8, 1.0, 1.0, 0.9],
    # Char Dham shrines: closed ~Nov-Apr, peaks at opening (May-Jun) and Oct
    "chardham":      [0, 0, 0, 0.2, 2.0, 2.0, 0.8, 0.6, 1.0, 1.6, 0.3, 0],
    # Hill stations: summer peak, small autumn bump
    "hill_station":  [0.8, 0.7, 0.8, 1.0, 1.6, 1.6, 0.8, 0.6, 0.8, 1.2, 1.0, 1.1],
    # Hill stations that also get snow / New-Year crowds
    "hill_snow":     [1.3, 0.9, 0.8, 0.9, 1.5, 1.5, 0.7, 0.6, 0.8, 1.1, 0.9, 1.4],
    # Ski / snow destinations
    "snow":          [2.0, 1.8, 1.0, 0.5, 0.6, 0.6, 0.3, 0.3, 0.4, 0.6, 0.8, 1.6],
    # National park: core zones closed in monsoon
    "wildlife":      [1.2, 1.2, 1.2, 1.2, 1.3, 1.1, 0.1, 0.1, 0.3, 1.0, 1.3, 1.3],
    # High-altitude desert: road open roughly Jun-Oct only
    "high_altitude": [0, 0, 0, 0.2, 0.8, 2.0, 2.2, 2.0, 1.6, 0.8, 0.2, 0],
    # Shakti-peeth temples: Navratri (Mar-Apr, Sep-Oct) and Shravan (Aug)
    "temple":        [1.0, 0.9, 1.3, 1.4, 1.0, 1.0, 0.9, 1.1, 1.0, 1.3, 1.0, 0.9],
}

# ---------------------------------------------------------------------------
# 2. Hotspots.  weight = rough share of the state's annual visitors (PLACEHOLDER).
#    Whatever is left (1 - sum of weights) is "rest of the state" (not mapped).
#    kumbh_boost: extra pull during Kumbh months (Haridwar only). 1.6 is chosen so
#    that the extra Kumbh crowd the STATE model predicts goes to Haridwar while the
#    other hotspots stay about the same (a bigger value would "steal" from them).
# ---------------------------------------------------------------------------
HOTSPOTS = pd.DataFrame([
    # region,            hotspot,                  district,          lat,     lon,     weight, profile,        kumbh_boost
    ("uttarakhand",      "Haridwar",               "Haridwar",        29.9457, 78.1642, 0.30, "ganga_pilgrim", 1.6),
    ("uttarakhand",      "Rishikesh",              "Dehradun",        30.0869, 78.2676, 0.12, "ganga_pilgrim", 1.0),
    ("uttarakhand",      "Dehradun",               "Dehradun",        30.3165, 78.0322, 0.07, "hill_station",  1.0),
    ("uttarakhand",      "Mussoorie",              "Dehradun",        30.4598, 78.0644, 0.07, "hill_snow",     1.0),
    ("uttarakhand",      "Nainital",               "Nainital",        29.3919, 79.4542, 0.08, "hill_station",  1.0),
    ("uttarakhand",      "Jim Corbett (Ramnagar)", "Nainital",        29.3947, 79.1260, 0.03, "wildlife",      1.0),
    ("uttarakhand",      "Kedarnath",              "Rudraprayag",     30.7352, 79.0669, 0.04, "chardham",      1.0),
    ("uttarakhand",      "Badrinath",              "Chamoli",         30.7433, 79.4938, 0.04, "chardham",      1.0),
    ("uttarakhand",      "Gangotri",               "Uttarkashi",      30.9947, 78.9398, 0.015, "chardham",     1.0),
    ("uttarakhand",      "Yamunotri",              "Uttarkashi",      31.0140, 78.4600, 0.012, "chardham",     1.0),
    ("uttarakhand",      "Auli",                   "Chamoli",         30.5286, 79.5664, 0.005, "snow",         1.0),
    ("himachal_pradesh", "Shimla",                 "Shimla",          31.1048, 77.1734, 0.20, "hill_snow",     1.0),
    ("himachal_pradesh", "Manali",                 "Kullu",           32.2396, 77.1887, 0.17, "hill_snow",     1.0),
    ("himachal_pradesh", "Kullu",                  "Kullu",           31.9579, 77.1095, 0.06, "hill_station",  1.0),
    ("himachal_pradesh", "Kasol",                  "Kullu",           32.0100, 77.3150, 0.03, "hill_station",  1.0),
    ("himachal_pradesh", "Dharamshala-McLeodganj", "Kangra",          32.2190, 76.3234, 0.11, "hill_station",  1.0),
    ("himachal_pradesh", "Dalhousie",              "Chamba",          32.5387, 75.9710, 0.04, "hill_snow",     1.0),
    ("himachal_pradesh", "Kasauli",                "Solan",           30.8987, 76.9653, 0.04, "hill_station",  1.0),
    ("himachal_pradesh", "Naina Devi",             "Bilaspur",        31.3069, 76.5367, 0.06, "temple",        1.0),
    ("himachal_pradesh", "Spiti (Kaza)",           "Lahaul & Spiti",  32.2276, 78.0710, 0.015, "high_altitude", 1.0),
], columns=["region", "hotspot", "district", "lat", "lon", "weight", "profile", "kumbh_boost"])


def _profile_matrix(spots: pd.DataFrame) -> np.ndarray:
    """(n_hotspots x 12) array of seasonal profiles, each scaled to mean 1."""
    mat = np.array([PROFILES[p] for p in spots["profile"]], dtype=float)
    return mat / mat.mean(axis=1, keepdims=True)


def distribute(monthly: pd.DataFrame, region: str) -> pd.DataFrame:
    """Split a state's monthly series into hotspots (top-down).

    monthly: DataFrame with columns date, visitors (and optionally kumbh)
    returns: long DataFrame -> date, region, hotspot, district, lat, lon, visitors
    """
    spots = HOTSPOTS[HOTSPOTS["region"] == region].reset_index(drop=True)
    prof = _profile_matrix(spots)
    rest_weight = max(0.0, 1.0 - spots["weight"].sum())   # unmapped rest of state

    rows = []
    for _, r in monthly.iterrows():
        m = r["date"].month - 1
        kumbh = bool(r.get("kumbh", 0))
        boost = np.where(kumbh, spots["kumbh_boost"].values, 1.0)
        pull = spots["weight"].values * prof[:, m] * boost   # each hotspot's pull this month
        share = pull / (pull.sum() + rest_weight)            # rest of state has flat profile
        for i, s in spots.iterrows():
            rows.append({"date": r["date"], "region": region, "hotspot": s["hotspot"],
                         "district": s["district"], "lat": s["lat"], "lon": s["lon"],
                         "visitors": r["visitors"] * share[i]})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import forecast as F

    frames = []
    for region, cfg in F.REGIONS.items():
        df = F.load_data(cfg["sample_data"], region)
        fc, _ = F.forecast_future(df, 36, region)
        fc_series = fc.rename(columns={"ensemble": "visitors"})[["date", "visitors", "kumbh"]]
        frames.append(distribute(fc_series, region))
    out = pd.concat(frames)
    out["year"] = out["date"].dt.year
    annual = (out.groupby(["region", "hotspot", "year"])["visitors"].sum() / 1e5) \
        .unstack("year").round(1)
    os.makedirs("outputs", exist_ok=True)
    out.to_csv("outputs/hotspot_forecast.csv", index=False)
    print("Hotspot forecast (lakh visitors / year)  -- SYNTHETIC / PLACEHOLDER SHARES")
    print(annual.to_string())
    print("\nSaved outputs/hotspot_forecast.csv")
