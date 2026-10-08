"""
Streamlit dashboard for the Uttarakhand tourist demand forecasting project
(with Himachal Pradesh as a comparison state).
Run:  py -m streamlit run app.py

Layout
    header      title + data-status badge
    filter row  state, forecast horizon, backtest window, data source
    tabs        Overview | Hotspots | Model evaluation | Seasonality | Data
Charts use Altair (ships with Streamlit) so every chart has hover tooltips.
"""
import io

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

import forecast as F
import hotspots as H

st.set_page_config(page_title="Tourism Demand Forecast", page_icon="📈", layout="wide")

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
REGION_NAMES = {k: v["name"] for k, v in F.REGIONS.items()}

# ---------------------------------------------------------------------------
# Colours. One colour per STATE everywhere (map, bars, lines), so a reader
# learns "blue = Uttarakhand, orange = Himachal" once. Palette was checked for
# colour-blind safety; dark mode uses its own, slightly shifted steps.
# ---------------------------------------------------------------------------
DARK = st.context.theme.type == "dark"
if DARK:
    C = {"series": ["#3987e5", "#d95926", "#199e70", "#c98500"],
         "ink": "#ffffff", "muted": "#898781", "surface": [26, 26, 25]}
else:
    C = {"series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
         "ink": "#0b0b0b", "muted": "#898781", "surface": [252, 252, 251]}
STATE_COLOR = {"uttarakhand": C["series"][0], "himachal_pradesh": C["series"][1]}


def hex_rgb(h):
    return [int(h[i:i + 2], 16) for i in (1, 3, 5)]


# ---------------------------------------------------------------------------
# Light-touch styling: KPI cards, tighter page, status badge
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1440px;}
[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,0.22); border-radius: 8px; padding: 12px 16px;
    min-height: 118px;
}
[data-testid="stMetricLabel"] p {font-size: 0.8rem;}
.dash-eyebrow {font-size: 0.78rem; letter-spacing: .06em; text-transform: uppercase;
               opacity: .65; margin-bottom: 2px;}
.dash-title {font-size: 1.75rem; font-weight: 700; line-height: 1.2; margin: 0 0 .75rem 0;}
.badge {display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: .78rem;
        font-weight: 600; border: 1px solid rgba(250,178,25,.6);
        background: rgba(250,178,25,.14);}
.badge-ok {border-color: rgba(12,163,12,.6); background: rgba(12,163,12,.12);}
.badge-info {border-color: rgba(42,120,214,.6); background: rgba(42,120,214,.12);}
.legend-dot {display:inline-block; width:10px; height:10px; border-radius:50%;
             margin: 0 6px 0 14px; vertical-align: middle;}
.small-note {font-size: .8rem; opacity: .7;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Filter row (one row of controls that scopes everything below it)
# ---------------------------------------------------------------------------
head_l, head_r = st.columns([3, 1], vertical_alignment="bottom")

f1, f2, f3, f4 = st.columns([2, 1, 1, 1], vertical_alignment="bottom")
region = f1.segmented_control("State", list(REGION_NAMES), default="uttarakhand",
                              format_func=REGION_NAMES.get) or "uttarakhand"  # if user un-clicks
horizon = f2.selectbox("Forecast horizon", [12, 24, 36, 48], index=2,
                       format_func=lambda m: f"{m} months")
test_months = f3.selectbox("Backtest window", [12, 18, 24, 30, 36], index=2,
                           format_func=lambda m: f"Last {m} months")
DATASETS = {
    "estimated": "Official annual totals, monthly split estimated (2017-2024)",
    "sample": "Synthetic sample (pipeline test, 2015-2025)",
}
with f4.popover("Data source", width="stretch"):
    dataset = st.radio("Dataset", list(DATASETS), format_func=DATASETS.get)
    st.caption("Estimated: yearly totals are official Ministry of Tourism figures "
               "(see data/estimated/official_annual_totals.csv); how each year is split "
               "into months is an assumption.")
    uploaded = st.file_uploader(f"Or upload a monthly CSV for {REGION_NAMES[region]}",
                                type="csv", help="Two columns: date (YYYY-MM-01), visitors")

with head_l:
    st.markdown('<div class="dash-eyebrow">Uttarakhand · Himachal Pradesh</div>'
                '<div class="dash-title">Tourist Demand Forecasting</div>', unsafe_allow_html=True)
with head_r:
    if uploaded:
        st.markdown(f'<div style="text-align:right"><span class="badge badge-ok">'
                    f'Uploaded data: {REGION_NAMES[region]}</span></div>', unsafe_allow_html=True)
    elif dataset == "estimated":
        st.markdown('<div style="text-align:right"><span class="badge badge-info">'
                    'Official annual totals · monthly split estimated</span></div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div style="text-align:right"><span class="badge">⚠ Synthetic data · '
                    'pipeline test only</span></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data + models (cached so changing tabs doesn't retrain)
# ---------------------------------------------------------------------------
def get_df(r):
    """Uploaded data for the selected state, otherwise the chosen dataset."""
    if uploaded and r == region:
        uploaded.seek(0)
        return F.load_data(uploaded, r)
    return F.load_data(F.REGIONS[r][f"{dataset}_data"], r)


@st.cache_data(show_spinner="Training models...")
def run(df_json, test_months, horizon, region):
    data = pd.read_json(io.StringIO(df_json))
    data["date"] = pd.to_datetime(data["date"])
    data = F.add_features(data[["date", "visitors"]], region)
    train, test, preds, table = F.backtest(data, test_months, region)
    fc, sims = F.forecast_future(data, horizon, region)
    annual = F.annual_summary(fc, sims)
    return train, test, preds, table, fc, annual


def run_region(r):
    df_r = get_df(r)
    return df_r, run(df_r[["date", "visitors"]].to_json(date_format="iso"),
                     test_months, horizon, r)


@st.cache_data(show_spinner="Splitting state forecasts into hotspots...")
def hotspot_table(series_json, region):
    """Actual + forecast monthly series for one state -> per-hotspot visitors."""
    s = pd.read_json(io.StringIO(series_json))
    s["date"] = pd.to_datetime(s["date"])
    out = H.distribute(s, region)
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    return out


df, (train, test, preds, table, fc, annual) = run_region(region)
state_color = STATE_COLOR[region]

st.write("")
t_over, t_map, t_eval, t_season, t_data = st.tabs(
    ["Overview", "Hotspots", "Model evaluation", "Seasonality", "Data"])

# ===========================================================================
# OVERVIEW
# ===========================================================================
with t_over:
    last_year = int(df["date"].dt.year.max())
    next_year = last_year + 1
    last_total = df.loc[df["date"].dt.year == last_year, "visitors"].sum()
    nxt = fc[fc["date"].dt.year == next_year]
    next_total = nxt["ensemble"].sum()
    peak = nxt.loc[nxt["ensemble"].idxmax()]
    best = table["MAPE (%)"].idxmin()

    k1, k2, k3, k4 = st.columns(4)
    prev_total = df.loc[df["date"].dt.year == last_year - 1, "visitors"].sum()
    k1.metric(f"Visitors {last_year} (actual)", f"{last_total / 1e7:.2f} cr",
              f"{(last_total / prev_total - 1) * 100:+.1f}% vs {last_year - 1}")
    k2.metric(f"Forecast {next_year}", f"{next_total / 1e7:.2f} cr",
              f"{(next_total / last_total - 1) * 100:+.1f}% vs {last_year}")
    k3.metric(f"Peak month {next_year} · {peak['date']:%B}",
              f"{peak['ensemble'] / 1e5:.1f} lakh")
    k4.metric(f"Best model · {best}", f"{table.loc[best, 'MAPE (%)']:.2f}% MAPE",
              help=f"Mean absolute % error on the last {test_months} months (held out)")

    left, right = st.columns([3, 1.15], gap="large")
    with left:
        st.markdown(f"**Monthly visitors: actual and forecast** "
                    f"<span class='small-note'>(lakh / month, shaded = 80% interval)</span>",
                    unsafe_allow_html=True)
        hist = df[df["date"] >= "2019-01-01"][["date", "visitors"]]
        # Join the forecast line to the last actual point so there is no gap
        bridge = pd.DataFrame({"date": [hist["date"].iloc[-1]],
                               "forecast": [hist["visitors"].iloc[-1]]})
        wide = pd.concat([
            hist.rename(columns={"visitors": "actual"}),
            bridge,
            fc[["date", "ensemble", "lower_80", "upper_80"]]
              .rename(columns={"ensemble": "forecast", "lower_80": "lower", "upper_80": "upper"}),
        ], ignore_index=True).groupby("date", as_index=False).first()
        for c in ["actual", "forecast", "lower", "upper"]:
            wide[c] = wide[c] / 1e5
        long = wide.melt("date", ["actual", "forecast"], var_name="series",
                         value_name="lakh").dropna()
        long["series"] = long["series"].map({"actual": "Actual", "forecast": "Forecast"})

        x = alt.X("date:T", title=None, axis=alt.Axis(format="%Y", tickCount="year"))
        hover = alt.selection_point(nearest=True, on="pointerover", fields=["date"],
                                    empty=False)
        layers = []
        k = fc[fc["kumbh"] == 1]
        if len(k):
            kumbh = pd.DataFrame({"start": [k["date"].min()],
                                  "end": [k["date"].max() + pd.offsets.MonthEnd(1)],
                                  "label": ["Haridwar Kumbh"]})
            layers += [
                alt.Chart(kumbh).mark_rect(opacity=0.08, color=C["muted"])
                   .encode(x="start:T", x2="end:T"),
                alt.Chart(kumbh).mark_text(align="left", dx=4, dy=8, fontSize=11,
                                           color=C["muted"], baseline="top")
                   .encode(x="start:T", y=alt.value(0), text="label:N"),
            ]
        layers += [
            alt.Chart(wide).mark_area(opacity=0.16, color=state_color)
               .encode(x=x, y="lower:Q", y2="upper:Q"),
            alt.Chart(long).mark_line(strokeWidth=2).encode(
                x=x,
                y=alt.Y("lakh:Q", title="Visitors (lakh)"),
                color=alt.Color("series:N", title=None,
                                scale=alt.Scale(domain=["Actual", "Forecast"],
                                                range=[C["ink"], state_color]),
                                legend=alt.Legend(orient="top-left", direction="horizontal"))),
            alt.Chart(wide).mark_rule(color=C["muted"]).encode(
                x=x,
                opacity=alt.condition(hover, alt.value(0.7), alt.value(0)),
                tooltip=[alt.Tooltip("date:T", title="Month", format="%b %Y"),
                         alt.Tooltip("actual:Q", title="Actual (lakh)", format=",.1f"),
                         alt.Tooltip("forecast:Q", title="Forecast (lakh)", format=",.1f"),
                         alt.Tooltip("lower:Q", title="80% low", format=",.1f"),
                         alt.Tooltip("upper:Q", title="80% high", format=",.1f")],
            ).add_params(hover),
            alt.Chart(long).mark_point(filled=True, size=50).encode(
                x=x, y="lakh:Q",
                color=alt.Color("series:N", legend=None,
                                scale=alt.Scale(domain=["Actual", "Forecast"],
                                                range=[C["ink"], state_color])),
                opacity=alt.condition(hover, alt.value(1), alt.value(0))),
        ]
        st.altair_chart(alt.layer(*layers).properties(height=380), width="stretch")

    with right:
        st.markdown("**Annual totals** <span class='small-note'>(crore)</span>",
                    unsafe_allow_html=True)
        yearly = pd.DataFrame({
            "Year": [last_year] + list(annual.index),
            "Type": ["Actual"] + ["Forecast"] * len(annual),
            "Visitors": [last_total / 1e7] + list(annual["forecast_crore"]),
            "80% range": ["–"] + [f"{lo:.2f} – {hi:.2f}"
                                  for lo, hi in zip(annual["lower_80"], annual["upper_80"])],
        })
        yearly["Change"] = [("–" if pd.isna(c) else f"{c:+.1f}%")
                            for c in yearly["Visitors"].pct_change() * 100]
        st.dataframe(yearly, hide_index=True, width="stretch",
                     column_config={
                         "Year": st.column_config.NumberColumn(format="%d"),
                         "Visitors": st.column_config.NumberColumn(format="%.2f"),
                         "Change": st.column_config.TextColumn("vs prev."),
                     })
        st.caption("Forecast = mean of SARIMAX and Gradient Boosting. "
                   "Interval from 1,000 simulated SARIMAX paths.")

# ===========================================================================
# HOTSPOTS (2-D map of both states)
# ===========================================================================
with t_map:
    frames = []
    for r in F.REGIONS:
        df_r, (_, _, _, _, fc_r, _) = run_region(r)
        series = pd.concat([
            df_r[["date", "visitors", "kumbh"]].assign(kind="actual"),
            fc_r.rename(columns={"ensemble": "visitors"})[["date", "visitors", "kumbh"]]
                .assign(kind="forecast"),
        ], ignore_index=True)   # fresh 0..n row numbers (to_json needs unique index)
        hs = hotspot_table(series.to_json(date_format="iso"), r)
        hs["kind"] = hs["date"].map(dict(zip(series["date"], series["kind"])))
        frames.append(hs)
    spots = pd.concat(frames, ignore_index=True)

    # Only offer whole years that both states have (12 months each)
    counts = spots.groupby(["region", "year"])["date"].nunique().unstack(0)
    years = sorted(counts[(counts == 12).all(axis=1)].index)
    last_actual = int(spots.loc[spots["kind"] == "actual", "year"].max())

    h1, h2, h3 = st.columns([2, 1, 1], vertical_alignment="bottom")
    year = h1.select_slider("Year", years, value=min(last_actual + 1, years[-1]),
                            format_func=lambda y: f"{y} (F)" if y > last_actual else str(y))
    period = h2.selectbox("Period", ["Whole year"] + MONTHS)
    base_year = h3.selectbox("Compare with", [y for y in years if y <= last_actual][::-1])

    def period_total(y):
        sel = spots[spots["year"] == y]
        if period != "Whole year":
            sel = sel[sel["month"] == MONTHS.index(period) + 1]
        return (sel.groupby(["region", "hotspot", "district", "lat", "lon"], as_index=False)
                ["visitors"].sum())

    view = period_total(year)
    base = period_total(base_year)[["hotspot", "visitors"]].rename(columns={"visitors": "base"})
    view = view.merge(base, on="hotspot")
    # Growth is undefined where the base is 0 (e.g. Kedarnath in January)
    view["growth_pct"] = np.where(view["base"] > 0,
                                  (view["visitors"] / view["base"].where(view["base"] > 0) - 1) * 100,
                                  np.nan)
    view["state"] = view["region"].map(REGION_NAMES)
    view["lakh"] = view["visitors"] / 1e5
    period_txt = "full year" if period == "Whole year" else period
    tag = " (forecast)" if year > last_actual else ""

    kk = st.columns(3)
    for col, r in zip(kk, F.REGIONS):
        v = view[view["region"] == r]
        col.metric(f"{REGION_NAMES[r]} · mapped hotspots, {period_txt} {year}{tag}",
                   f"{v['lakh'].sum():,.1f} lakh",
                   f"{(v['visitors'].sum() / v['base'].sum() - 1) * 100:+.1f}% vs {base_year}")
    top = view.sort_values("visitors", ascending=False).iloc[0]
    kk[2].metric(f"Busiest hotspot · {top['lakh']:,.1f} lakh ({top['state']})", top["hotspot"])

    m_left, m_right = st.columns([1.6, 1], gap="large")
    with m_left:
        st.markdown(
            f"<b>Hotspot map</b> <span class='small-note'>circle area ∝ visitors</span>"
            f"<span class='legend-dot' style='background:{STATE_COLOR['uttarakhand']}'></span>Uttarakhand"
            f"<span class='legend-dot' style='background:{STATE_COLOR['himachal_pradesh']}'></span>"
            f"Himachal Pradesh", unsafe_allow_html=True)
        mv = view.copy()
        # Circle AREA proportional to visitors -> radius ~ sqrt(visitors). Largest = 20 km.
        mv["radius"] = 20_000 * np.sqrt(mv["visitors"] / mv["visitors"].max())
        mv["fill"] = mv["region"].map(lambda r: hex_rgb(STATE_COLOR[r]) + [190])
        mv["lakh_txt"] = mv["lakh"].map("{:,.1f}".format)
        mv["growth_txt"] = mv["growth_pct"].map(lambda g: "–" if pd.isna(g) else f"{g:+.1f}%")
        # Label only the busiest hotspot in each district (others: hover tooltip)
        labels = (mv[mv["visitors"] > 0].sort_values("visitors", ascending=False)
                  .drop_duplicates("district")).copy()
        # Put each label just outside its circle (~890 m per pixel at this zoom);
        # a few go on the LEFT where a neighbour sits on the right.
        left = labels["hotspot"].isin(["Haridwar", "Kedarnath", "Kasauli", "Naina Devi",
                                       "Dharamshala-McLeodganj"])
        gap = (6 + labels["radius"] / 890).round()
        labels["anchor"] = np.where(left, "end", "start")
        labels["offset"] = [[-g, 0] if l else [g, 0] for g, l in zip(gap, left)]
        state_labels = pd.DataFrame({"name": ["HIMACHAL PRADESH", "UTTARAKHAND"],
                                     "lat": [31.75, 29.75], "lon": [76.55, 79.75]})
        layers = [
            pdk.Layer("ScatterplotLayer", data=mv, get_position="[lon, lat]",
                      get_radius="radius", get_fill_color="fill",
                      stroked=True, get_line_color=C["surface"], line_width_min_pixels=1.5,
                      radius_min_pixels=3, pickable=True, auto_highlight=True),
            pdk.Layer("TextLayer", data=labels, get_position="[lon, lat]",
                      get_text="hotspot", get_size=12, get_color=hex_rgb(C["ink"]),
                      font_family="'Segoe UI, Arial, sans-serif'", font_weight=600,
                      get_text_anchor="anchor", get_alignment_baseline="'center'",
                      get_pixel_offset="offset"),
            pdk.Layer("TextLayer", data=state_labels, get_position="[lon, lat]",
                      get_text="name", get_size=13, get_color=hex_rgb(C["muted"]),
                      font_family="'Segoe UI, Arial, sans-serif'", font_weight=600),
        ]
        st.pydeck_chart(pdk.Deck(
            layers=layers,
            map_provider="carto",
            map_style=(pdk.map_styles.CARTO_DARK_NO_LABELS if DARK
                       else pdk.map_styles.CARTO_LIGHT_NO_LABELS),
            initial_view_state=pdk.ViewState(latitude=30.95, longitude=77.85, zoom=6.25,
                                             pitch=0, bearing=0),
            tooltip={"html": "<b>{hotspot}</b><br/>{district}, {state}<br/>"
                             "Visitors: <b>{lakh_txt} lakh</b><br/>vs " + str(base_year) +
                             ": {growth_txt}",
                     "style": {"fontSize": "12px"}},
        ), height=520)
        st.caption("Basemap © CARTO © OpenStreetMap contributors (needs internet). "
                   "Locations are real. State totals come from the selected dataset; "
                   "the split into hotspots uses placeholder shares.")

    with m_right:
        st.markdown(f"**Ranking · {period_txt} {year}{tag}** "
                    "<span class='small-note'>(lakh)</span>", unsafe_allow_html=True)
        bars = alt.Chart(view).mark_bar(cornerRadiusEnd=3, height=13).encode(
            y=alt.Y("hotspot:N", sort="-x", title=None,
                    axis=alt.Axis(labelOverlap=False, labelLimit=170)),
            x=alt.X("lakh:Q", title=None),
            color=alt.Color("state:N", title=None,
                            scale=alt.Scale(domain=list(REGION_NAMES.values()),
                                            range=list(STATE_COLOR.values())),
                            legend=None),
            tooltip=[alt.Tooltip("hotspot:N", title="Hotspot"),
                     alt.Tooltip("state:N", title="State"),
                     alt.Tooltip("lakh:Q", title="Visitors (lakh)", format=",.1f"),
                     alt.Tooltip("growth_pct:Q", title=f"vs {base_year} (%)", format="+.1f")],
        ).properties(height=24 * len(view))
        st.altair_chart(bars, width="stretch")

    with st.expander("Hotspot table"):
        st.dataframe(
            view.sort_values("visitors", ascending=False)
                [["state", "hotspot", "district", "lakh", "growth_pct"]],
            hide_index=True, width="stretch",
            column_config={
                "state": "State", "hotspot": "Hotspot", "district": "District",
                "lakh": st.column_config.NumberColumn("Visitors (lakh)", format="%.1f"),
                "growth_pct": st.column_config.NumberColumn(f"vs {base_year}", format="%+.1f%%"),
            })

# ===========================================================================
# MODEL EVALUATION
# ===========================================================================
with t_eval:
    e_left, e_right = st.columns([1.25, 2], gap="large")
    with e_left:
        st.markdown(f"**Backtest: last {test_months} months held out** "
                    "<span class='small-note'>(lower is better)</span>", unsafe_allow_html=True)
        tbl = table.sort_values("MAPE (%)").reset_index(names="Model")
        tbl.insert(0, "Rank", range(1, len(tbl) + 1))
        tbl = tbl[["Rank", "Model", "MAPE (%)", "MAE (lakh)", "RMSE (lakh)"]]
        st.dataframe(tbl, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.2f")
                                    for c in ["MAE (lakh)", "RMSE (lakh)", "MAPE (%)"]})
        st.caption("MAE / RMSE: average error in lakh visitors per month. "
                   "MAPE: average % error. Seasonal naive = same month last year.")
    with e_right:
        st.markdown("**Predictions vs actual on the test period** "
                    "<span class='small-note'>(lakh / month)</span>", unsafe_allow_html=True)
        bt = pd.DataFrame({"date": test["date"].values, "Actual": test["visitors"].values})
        for name, p in preds.items():
            bt[name] = p
        bt_long = bt.melt("date", var_name="series", value_name="v")
        bt_long["lakh"] = bt_long["v"] / 1e5
        names = ["Actual"] + list(preds)
        bt_chart = alt.Chart(bt_long).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=28)) \
            .encode(
                x=alt.X("date:T", title=None, axis=alt.Axis(format="%b %Y")),
                y=alt.Y("lakh:Q", title="Visitors (lakh)"),
                color=alt.Color("series:N", title=None, sort=names,
                                scale=alt.Scale(domain=names, range=[C["ink"]] + C["series"]),
                                legend=alt.Legend(orient="top", columns=3)),
                strokeWidth=alt.condition(alt.datum.series == "Actual",
                                          alt.value(2.5), alt.value(1.5)),
                tooltip=[alt.Tooltip("date:T", title="Month", format="%b %Y"),
                         alt.Tooltip("series:N", title="Series"),
                         alt.Tooltip("lakh:Q", title="Lakh", format=",.1f")],
            ).properties(height=360)
        st.altair_chart(bt_chart, width="stretch")

# ===========================================================================
# SEASONALITY (both states on one indexed axis: % of the year's visitors)
# ===========================================================================
with t_season:
    rows = []
    for r in F.REGIONS:
        d = get_df(r)
        d = d[d["covid"] == 0]
        prof = d.groupby("month")["visitors"].mean()
        prof = prof / prof.sum() * 100
        rows += [{"state": REGION_NAMES[r], "month": MONTHS[m - 1], "m": m, "share": v}
                 for m, v in prof.items()]
    season = pd.DataFrame(rows)
    s_left, s_right = st.columns([2, 1], gap="large")
    with s_left:
        st.markdown("**Share of annual visitors by month** "
                    "<span class='small-note'>(%, non-COVID years)</span>",
                    unsafe_allow_html=True)
        sc = alt.Chart(season).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=50)).encode(
            x=alt.X("month:N", sort=MONTHS, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("share:Q", title="% of annual visitors"),
            color=alt.Color("state:N", title=None,
                            scale=alt.Scale(domain=list(REGION_NAMES.values()),
                                            range=list(STATE_COLOR.values())),
                            legend=alt.Legend(orient="top")),
            tooltip=[alt.Tooltip("state:N", title="State"), alt.Tooltip("month:N", title="Month"),
                     alt.Tooltip("share:Q", title="% of year", format=".1f")],
        ).properties(height=340)
        st.altair_chart(sc, width="stretch")
    with s_right:
        st.markdown("**What drives the peaks**")
        st.markdown(
            "- **Uttarakhand:** May–June (Char Dham opening + school holidays), "
            "July (Kanwar Yatra), October (second Char Dham phase).\n"
            "- **Himachal Pradesh:** May–June (summer escape from the plains), "
            "December–January (snow, New Year); monsoon dip in July–August.")
        st.caption("Showing both states as a share of their own annual total puts "
                   "them on one comparable scale even though Uttarakhand is ~3× larger.")

# ===========================================================================
# DATA
# ===========================================================================
with t_data:
    d_left, d_right = st.columns([2, 1], gap="large")
    with d_left:
        st.markdown(f"**Input data: {REGION_NAMES[region]}** "
                    f"<span class='small-note'>{len(df)} months, "
                    f"{df['date'].min():%b %Y} – {df['date'].max():%b %Y}</span>",
                    unsafe_allow_html=True)
        st.dataframe(df[["date", "visitors"] + F.REGIONS[region]["exog"]],
                     hide_index=True, width="stretch", height=420,
                     column_config={"date": st.column_config.DateColumn("Month", format="MMM YYYY"),
                                    "visitors": st.column_config.NumberColumn("Visitors",
                                                                              format="localized")})
    with d_right:
        st.markdown("**Downloads**")
        st.download_button("Monthly forecast (CSV)", fc.to_csv(index=False),
                           f"{region}_forecast.csv", "text/csv", width="stretch")
        st.download_button("Hotspot forecast, all months (CSV)",
                           spots.drop(columns=["lat", "lon"]).to_csv(index=False),
                           "hotspot_forecast.csv", "text/csv", width="stretch")
        st.download_button("Backtest metrics (CSV)", table.to_csv(),
                           f"{region}_backtest.csv", "text/csv", width="stretch")
        st.markdown("**Event features used**")
        st.caption(", ".join(F.REGIONS[region]["exog"]) + " (SARIMAX) · "
                   + ", ".join(f for f in F.REGIONS[region]["ml_features"]) + " (GBM)")
