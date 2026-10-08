# Tourist Demand Forecasting for Uttarakhand
*B.Tech Final Year Project — Time-series forecasting with event-aware models*

## 1. Problem statement
Uttarakhand received over 6 crore tourists and pilgrims in 2025, but demand is extremely
seasonal (Char Dham peaks in May–June and October) and is hit by shocks such as the
Haridwar Kumbh, monsoon disasters and COVID. Hotels, transport, the Char Dham
administration and disaster-management teams need month-wise demand forecasts
to plan capacity. This project builds and compares forecasting models that predict
monthly tourist arrivals 1–3 years ahead, with uncertainty ranges.

## 2. Objectives
1. Collect and clean monthly tourist-arrival data for Uttarakhand.
2. Model seasonality, trend and special events (Kumbh, COVID, post-COVID shift).
3. Compare a statistical model (SARIMAX) with a machine-learning model (Gradient Boosting) against a baseline.
4. Forecast 2026–2028 demand, including the 2027 Haridwar Kumbh, with 80% intervals.
5. Deliver an interactive dashboard (Streamlit) for decision-makers.

## 3. Project structure
```
forecast.py            core pipeline: features, models, backtest, forecast, plots
app.py                 Streamlit dashboard
make_sample_data.py    generates SYNTHETIC test data for UK + HP (replace with real data!)
make_estimated_data.py monthly ESTIMATE from official annual totals (data/estimated/)
hotspots.py            hotspot list (lat/lon) + top-down split of state forecast into hotspots
data/                  put your real CSV here (columns: date, visitors)
outputs/               plots + forecast CSVs are written here
```

## 4. How to run
```
pip install -r requirements.txt
python make_sample_data.py          # only for testing the pipeline
python forecast.py --data data/sample_monthly_visits.csv
python forecast.py --region himachal_pradesh   # comparison state
python hotspots.py                  # per-hotspot forecast -> outputs/hotspot_forecast.csv
streamlit run app.py                # dashboard
```

## 5. ⚠️ Data — the most important step
Three kinds of data, kept in separate places:

| Dataset | Files | What is real |
|---|---|---|
| **Estimated** (default in the dashboard) | `data/estimated/*_monthly_estimated.csv` | Annual totals 2017–2024 are official Ministry of Tourism figures (domestic + foreign visits), each cited in `data/estimated/official_annual_totals.csv`. The **monthly split is an assumption** (average seasonal profile + COVID/Kumbh factors), made by `make_estimated_data.py`. |
| **Synthetic** | `data/sample_monthly_visits*.csv` | Nothing — for testing the pipeline only. |
| **Real monthly** (to collect) | `data/real/` | Everything — replaces the estimate when available. |

Run `py forecast.py --dataset estimated` (add `--region himachal_pradesh` for HP).
Note: the Ministry's figures differ from some UTDB press figures (e.g. Uttarakhand 2023:
5.77 cr in Compendium 2025 vs 5.96 cr quoted by UTDB). Use one source consistently.

`data/sample_monthly_visits.csv` is SYNTHETIC. Only the 2022–2025 annual totals match
official figures (5.40, 5.96, 5.96, 6.03 crore); everything else is a placeholder.
**Do not report results from it.** Collect real data from:
- Uttarakhand Tourism Development Board / Tourism Dept (district-wise monthly arrivals)
- Ministry of Tourism — *India Tourism Statistics* (annual, state-wise)
- indiastat.com (district-wise monthly tables; your college library may have access)
- Char Dham daily darshan counts (Badrinath-Kedarnath Temple Committee, Garhwal Commissioner press releases)
- An RTI application to the Tourism Dept if monthly data isn't published
Aim for at least 8–10 years of monthly data (2014 onward).

## 6. Methodology
| Step | What | Why |
|---|---|---|
| Preprocessing | Monthly aggregation, interpolation, log transform | Stabilises variance |
| Feature engineering | Kumbh dummy, COVID dummy, post-COVID level shift, Char Dham season, Kanwar month, cyclic month, 12-month lag | Captures events and seasonality |
| Baseline | Seasonal naive (same month last year) | Any model must beat this |
| Model 1 | SARIMAX(1,0,1)(0,1,1)₁₂ on log visitors with exogenous dummies | Classical, interpretable, gives intervals |
| Model 2 | Gradient Boosting Regressor, recursive multi-step | Captures non-linear effects |
| Ensemble | Average of Model 1 + 2 | Usually more robust |
| Evaluation | Hold out last 24 months; MAE, RMSE, MAPE | Honest out-of-sample test |
| Uncertainty | 1,000 simulated SARIMAX paths → 80% interval | Annual range for planning |

**Key insight to discuss in your report:** without the `post_covid` level-shift regressor,
SARIMAX extrapolated the 2021→2022 rebound as a permanent trend and its MAPE jumped
from ~4% to ~20% on test data. Handling this structural break is a strong talking point in the viva.

## 7. Suggested extensions (to score higher)
- **Facebook Prophet** or **LSTM** as additional models in the comparison table.
- **District-level models** (Haridwar, Dehradun, Nainital, Rudraprayag, Chamoli) — hierarchical forecasting.
- **Extra exogenous features:** monthly rainfall (IMD), Google Trends for "Kedarnath"/"Char Dham", Char Dham registrations (a leading indicator), hotel occupancy.
- **Carrying-capacity alert:** flag months where forecast exceeds shrine/road capacity.
- **Hyperparameter search:** grid-search SARIMA orders by AIC, time-series cross-validation (rolling origin).

## 8. Report chapter outline
1. Introduction (tourism in Uttarakhand, motivation, problem statement)
2. Literature review (ARIMA/SARIMA in tourism, ML/LSTM tourism forecasting, event effects)
3. Data collection & exploratory analysis (trend, seasonality decomposition, ACF/PACF)
4. Methodology (features, models, evaluation design)
5. Implementation (architecture, code, dashboard screenshots)
6. Results & discussion (comparison table, forecast plots, Kumbh 2027 scenario)
7. Conclusion, limitations (data quality, counting methods, disasters), future work

## 9. Suggested timeline (one semester)
| Weeks | Task |
|---|---|
| 1–3 | Literature review, data collection, RTI if needed |
| 4–5 | EDA, cleaning, seasonal decomposition |
| 6–9 | Models, backtesting, tuning, extensions |
| 10–11 | Dashboard, final forecasts |
| 12–14 | Report, presentation, viva prep |
