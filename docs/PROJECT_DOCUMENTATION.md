# Uttarakhand Tourist Demand Forecasting — Project Documentation

*B.Tech final-year project · documentation as of 8 October 2026*

This document explains the whole project so that someone who has never seen it can
understand, run and extend it: what problem it solves, how the code is organised, where the
data comes from, which features are engineered and why, how the models work, how they are
evaluated, and what the results mean.

---

## Contents

1. [Project overview](#1-project-overview)
2. [System architecture](#2-system-architecture)
3. [How to run it](#3-how-to-run-it)
4. [Datasets](#4-datasets)
5. [Feature engineering](#5-feature-engineering)
6. [Models](#6-models)
7. [Evaluation method](#7-evaluation-method)
8. [Results](#8-results)
9. [Hotspot forecasts (spatial split)](#9-hotspot-forecasts-spatial-split)
10. [Dashboard](#10-dashboard)
11. [Limitations and next steps](#11-limitations-and-next-steps)
12. [Glossary](#12-glossary)
13. [Sources](#13-sources)

---

## 1. Project overview

The project forecasts **monthly tourist visits to Uttarakhand 1–3 years ahead** and compares
statistical and machine-learning models. On the current dataset, **SARIMAX is the best model
for Uttarakhand, with a 3.2% average error** on two years of data it had not seen. It forecasts
about **6.19 crore visits in 2025** and **6.41 crore in 2027**, the Haridwar Kumbh year.

### The problem

Tourism in Uttarakhand is very seasonal and is hit by shocks:

- **Seasonality:** the Char Dham shrines open in May; visits peak in May–June and again in
  October. The Kanwar Yatra brings a spike in July.
- **Events:** the Haridwar Kumbh (next one 14 Jan – 20 Apr 2027) adds very large crowds.
- **Shocks:** COVID-19 lockdowns collapsed demand in 2020 and the second wave hit April–June 2021.

Hotels, transport operators, the Char Dham administration and disaster-management teams need
**month-wise demand estimates** to plan capacity. This project builds those estimates.

### What the system does

1. Reads a monthly series of tourist visits (one CSV with two columns: `date`, `visitors`).
2. Automatically creates calendar and event features from the date.
3. Trains four models and tests each on the last 24 months, which it does not see in training.
4. Forecasts the next 36 months, with an 80% uncertainty range for each year.
5. Splits each state's forecast across major destinations ("hotspots").
6. Shows everything in an interactive dashboard, including a map of both states.

**Himachal Pradesh** is included as a comparison state. The same pipeline runs on it,
with event features suited to Himachal.

### Key results at a glance

Estimated dataset, test period January 2023 – December 2024:

| Item | Uttarakhand | Himachal Pradesh |
| --- | --- | --- |
| Best model | SARIMAX | Seasonal naive |
| Best test error (MAPE) | 3.18% | 11.17% |
| Last actual year (2024) | 6.10 crore visits | 1.81 crore visits |
| Forecast 2025 | 6.19 crore | 1.77 crore |

> **Important:** the yearly totals are official government figures, but the **monthly split is
> an estimate** (section 4). Read every result in this document with that in mind.

---

## 2. System architecture

The project is five Python files around one core module, `forecast.py`. Every other file either
feeds it data or uses its results.

```
  DATA SOURCES                    forecast.py PIPELINE                 OUTPUTS
 ┌───────────────────────────┐   ┌──────────────────────────────┐    ┌─────────────────────────┐
 │ official_annual_totals.csv│   │ 1. load_data + add_features  │    │ hotspots.py             │
 │            │              │   │              │               │ ─▶ │ state forecast → 20     │
 │            ▼              │   │              ▼               │    │ hotspots                │
 │ make_estimated_data.py ───┼─▶ │ 2. backtest (4 models,       │    └───────────┬─────────────┘
 │  (estimated monthly)      │   │    last 24 months held out)  │                ▼
 │                           │   │              │               │    ┌─────────────────────────┐
 │ make_sample_data.py ──────┼─▶ │              ▼               │ ─▶ │ app.py dashboard        │
 │  (synthetic test data)    │   │ 3. forecast_future           │    │ 5 tabs + 2D map         │
 │                           │   │    (36 months, ensemble)     │    └─────────────────────────┘
 │ Uploaded CSV (real data) ─┼─▶ │              │               │    ┌─────────────────────────┐
 │                           │   │              ▼               │ ─▶ │ outputs/ folder         │
 └───────────────────────────┘   │ 4. annual_summary            │    │ forecast CSVs + plots   │
                                 │    (80% range, 1,000 paths)  │    └─────────────────────────┘
                                 └──────────────────────────────┘
```

All three data sources produce the **same two-column CSV**, so the pipeline does not care where
the data came from. `hotspots.py` and `app.py` import `forecast.py` and reuse its functions
instead of copying them.

### Files

| File | What it does |
| --- | --- |
| `forecast.py` | Core pipeline: feature engineering, the four models, backtest, forecast, plots. Also runs from the command line. |
| `hotspots.py` | List of 20 destinations with real coordinates, and the method that splits a state forecast across them. |
| `app.py` | Streamlit dashboard: Overview, Hotspots (map), Model evaluation, Seasonality, Data. |
| `make_estimated_data.py` | Builds monthly estimates from official annual totals. |
| `make_sample_data.py` | Builds a synthetic dataset, used only to test the pipeline. |
| `data/estimated/official_annual_totals.csv` | Official yearly totals with the source of every number. |
| `.streamlit/config.toml` | Dashboard accent colour; hides Streamlit's developer toolbar. |
| `requirements.txt` | Python packages needed. |

### Main functions in `forecast.py`

| Function | Purpose |
| --- | --- |
| `load_data(path, region)` | Reads the CSV, sorts by date, fills small gaps, adds features. |
| `add_features(df, region)` | Creates all calendar and event features from the date (section 5). |
| `seasonal_naive`, `sarimax_model`, `gbm_model` | The models (section 6). |
| `backtest(df, test_months, region)` | Hides the last N months, trains every model, scores it. |
| `forecast_future(df, months, region)` | Trains on all data and forecasts ahead. |
| `annual_summary(fc, sims)` | Yearly totals with an 80% range from simulated paths. |
| `REGIONS` (settings) | Per-state settings: file paths and which features each model uses. |

---

## 3. How to run it

### Requirements

- Windows with **Python 3.10 or newer** (tested on 3.12.3).
- This machine also has an old, broken Python 2.7. Plain `pip` points to it and fails, so
  **always use the `py` launcher**: `py -m pip …` and `py script.py`.
- In VS Code, select the Python 3.12 interpreter (Ctrl+Shift+P → "Python: Select Interpreter").

### Commands

```
py -m pip install -r requirements.txt            # one-time install
py make_estimated_data.py                        # build the estimated monthly data
py forecast.py --dataset estimated               # Uttarakhand: backtest + forecast
py forecast.py --dataset estimated --region himachal_pradesh
py hotspots.py                                   # per-hotspot forecast CSV
py -m streamlit run app.py                       # open the dashboard
```

Useful command-line options for `forecast.py`:

| Option | Default | Meaning |
| --- | --- | --- |
| `--region` | `uttarakhand` | `uttarakhand` or `himachal_pradesh` |
| `--dataset` | `synthetic` | `synthetic` or `estimated` |
| `--data` | – | Path to your own CSV (overrides `--dataset`) |
| `--test_months` | 24 | Length of the held-out test period |
| `--horizon` | 36 | Months to forecast |

Results are saved to `outputs/` (synthetic data) or `outputs/estimated/<region>/` (estimated
data): `forecast.csv`, `annual_forecast.csv`, `backtest.png`, `forecast.png`.

**Packages:** pandas, numpy, statsmodels, scikit-learn, matplotlib, streamlit. The dashboard's
charts (Altair) and map (pydeck) come with Streamlit. The map's background tiles need internet.

---

## 4. Datasets

The project keeps three kinds of data in separate places, so test data can never be mistaken
for real data.

| Dataset | Files | What is real | Use |
| --- | --- | --- | --- |
| **Estimated** (default) | `data/estimated/*_monthly_estimated.csv` | Yearly totals 2017–2024 are official. The split into months is an assumption. | Main results |
| **Synthetic** | `data/sample_monthly_visits*.csv` | Nothing; generated for testing. | Testing the code only |
| **Real monthly** | `data/real/` (to be collected) | Everything | Will replace the estimate |

### 4.1 Data format

Every dataset has the same two columns, one row per month:

```csv
date,visitors
2017-01-01,2069623
2017-02-01,1897154
2017-03-01,2414560
```

- `date` — the first day of the month.
- `visitors` — total tourist visits that month (domestic + foreign).

No other columns are needed; all features are generated from the date (section 5).

### 4.2 The official annual totals

The yearly totals come from the **Ministry of Tourism, Government of India**, through its yearly
publication *India Tourism Statistics* (renamed *India Tourism Data Compendium* in 2024). Each
number was copied from the original government PDF and checked against the growth-rate column
printed in the same table. The full list with exact table names and links is in
`data/estimated/official_annual_totals.csv`.

Total visits (domestic + foreign), in crore (1 crore = 10 million):

| Year | Uttarakhand | Himachal Pradesh | Source edition |
| --- | --- | --- | --- |
| 2017 | 3.45 | 1.96 | India Tourism Statistics 2019 |
| 2018 | 3.58 | 1.65 | MoT state table 2018 & 2019 |
| 2019 | 3.77 | 1.72 | MoT state table 2018 & 2019 |
| 2020 | 0.70 | 0.32 | India Tourism Statistics 2022 |
| 2021 | 1.94 | 0.56 | India Tourism Statistics 2022 |
| 2022 | 5.47 | 1.51 | India Tourism Data Compendium 2024 |
| 2023 | 5.77 | 1.60 | India Tourism Data Compendium 2025 |
| 2024 | 6.10 | 1.81 | India Tourism Data Compendium 2025 |

Notes on the data:

- **Why it starts in 2017:** the editions covering 2015–2016 could not be found online.
- **Why it ends in 2024:** 2025 figures are not yet published in this series.
- **Revisions:** the Ministry revised Uttarakhand 2023 from 58.14 million (2024 edition) to
  57.57 million (2025 edition). The newest figure is used.
- **Sources disagree:** the Uttarakhand Tourism Development Board quotes 5.96 crore for 2023,
  higher than the Ministry's 5.77 crore. The definitions differ, so this project uses **one
  source (the Ministry) for every year** to keep the series consistent.
- The 2022–2024 figures are published in millions to three decimals, so they are rounded to the
  nearest thousand visits.

### 4.3 How yearly totals become monthly values

Official monthly data is not freely published, so `make_estimated_data.py` spreads each year's
total across its 12 months. This is called **temporal disaggregation** (pro-rata method):

```
weight(month m) = average seasonal share(m) × event factor(m)
visitors(month m) = annual total × weight(m) / sum of the 12 weights
```

The 12 monthly values always **add up exactly to the official yearly total** (rounding is
corrected in December). No random noise is added, so the files are the same every run.

**Average seasonal share** — the assumed share of a normal year's visitors in each month (%):

| Month | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Uttarakhand | 6.0 | 5.5 | 7.0 | 8.0 | 13.0 | 13.0 | 9.0 | 6.0 | 7.0 | 10.0 | 8.0 | 7.5 |
| Himachal | 8.5 | 6.0 | 6.5 | 8.0 | 12.0 | 12.5 | 6.5 | 5.5 | 6.5 | 8.5 | 8.0 | 11.5 |

Why these shapes: Uttarakhand peaks with the Char Dham opening and summer holidays (May–June),
Kanwar (July) and the second Char Dham phase (October). Himachal peaks in summer (May–June) and
in the snow and New Year season (December–January), with a monsoon dip in July–August.

**Event factors** — adjustments for known shocks:

| Year | Months | Factor | Reason |
| --- | --- | --- | --- |
| 2020 | Apr–Dec | × 0.15 | National lockdown and restrictions |
| 2021 | Mar–Apr (Uttarakhand only) | × 1.15 | Haridwar Kumbh 2021 (restricted) |
| 2021 | Apr–Jun | × 0.25 | Second COVID wave |

### 4.4 The synthetic dataset

`make_sample_data.py` produces a fully synthetic 2015–2025 series for testing the code. It uses
the same seasonal shapes plus 4% random noise (fixed seed, so it is reproducible). **No result
from it should be reported as real.**

### 4.5 Getting real monthly data

The best source of real monthly data is **indiastat** (district-wise monthly visits for both
states; paid, often available through a college library). Other options are the state tourism
departments and an **RTI application** for month-wise Char Dham darshan counts. Once collected,
the real file uses the same two-column format and the rest of the project works unchanged.

---

## 5. Feature engineering

All features are created by `add_features()` from the `date` column alone. This means the
same features can be built for **future months**, which is essential: a model can only use a
feature at forecast time if its future values are known in advance.

### 5.1 Target transformation: log(visitors)

The models predict **log(visitors)**, not visitors, and convert back with `exp()` at the end.

Why:

- Seasonal swings are **proportional**: a peak month is roughly double a quiet month, whether
  the yearly total is 3 crore or 6 crore. On a log scale this becomes a constant **additive**
  difference, which linear models like SARIMAX handle well.
- It stabilises variance: big months no longer dominate the error.
- Back-transformed forecasts can never be negative.

```
y_t = log(visitors_t)        model is fit on y_t
forecast = exp(ŷ_t)          converted back to visitor counts
```

### 5.2 Feature list

| Feature | Definition | Why it is needed | Used by |
| --- | --- | --- | --- |
| `month` | 1–12 | Grouping for seasonal naive and seasonality plots | Naive, plots |
| `year` | Calendar year | Lets GBM learn the long-term growth level | GBM |
| `month_sin`, `month_cos` | sin(2π·month/12), cos(2π·month/12) | Encodes the month as a circle, so December sits next to January | GBM |
| `kumbh` | 1 in Kumbh months, else 0 | Captures the extra Kumbh crowd | SARIMAX, GBM (Uttarakhand) |
| `covid` | 1 in COVID-hit months, else 0 | Stops the models learning lockdown months as "normal" | SARIMAX, GBM, Naive |
| `post_covid` | 0 before Jan 2022, 1 from Jan 2022 | Structural break: demand settled at a much higher level after COVID | SARIMAX, GBM |
| `chardham_season` | 1 in May–October | The Char Dham yatra season | GBM (Uttarakhand) |
| `kanwar` | 1 in July | Kanwar Yatra spike | GBM (Uttarakhand) |
| `summer_peak` | 1 in May–June | Summer escape from the plains | GBM (Himachal) |
| `snow_season` | 1 in December–January | Snow and New Year tourism | GBM (Himachal) |
| `lag_12` | Visitors in the same month last year | Strongest single predictor of a month's demand | GBM |

### 5.3 Event windows

| Event | Months flagged | Notes |
| --- | --- | --- |
| Kumbh | Jan–Apr 2016 (Ardh Kumbh), Mar–Apr 2021, Jan–Apr 2027 | Uttarakhand only; Haridwar is in Uttarakhand |
| COVID | Apr–Dec 2020, Apr–Jun 2021 | National lockdown + second wave |
| Post-COVID | Jan 2022 onwards | A permanent level shift, not a one-off event |

### 5.4 Why the `post_covid` feature matters

Uttarakhand's visits jumped from about 3.8 crore before COVID (2019) to 5.5–6 crore from 2022.
Without a feature for this, SARIMAX treats the 2021→2022 rebound as a **trend** and keeps
extending it, so it over-forecasts badly. A dummy variable that switches on in January 2022
tells the model "the level changed once" instead.

During development on the synthetic test data, adding `post_covid` cut SARIMAX's test error
from **about 20% to about 4%**. Handling this **structural break** is one of the project's main
findings.

### 5.5 Why the 12-month lag has a COVID fix

`lag_12` normally takes the value from the same month last year. If that month was COVID-hit
(e.g. predicting May 2021 from May 2020), the lag would be misleadingly tiny. In that case the
code uses the value from **two years back** instead:

```
lag_12 = visitors(t − 24)   if month t − 12 was a COVID month
lag_12 = visitors(t − 12)   otherwise
```

### 5.6 Per-state feature sets

Both states use the same models; only the event features differ (set in `REGIONS`):

| | Uttarakhand | Himachal Pradesh |
| --- | --- | --- |
| SARIMAX regressors | `kumbh`, `covid`, `post_covid` | `covid`, `post_covid` |
| GBM features | month_sin, month_cos, year, kumbh, covid, post_covid, chardham_season, kanwar, lag_12 | month_sin, month_cos, year, covid, post_covid, summer_peak, snow_season, lag_12 |

---

## 6. Models

Four models are compared, from simplest to most complex. A forecasting project needs a simple
baseline: a complex model is only useful if it beats it.

### 6.1 Seasonal naive (baseline)

**Forecast = the same month last year.** COVID months are skipped, so the model uses the last
normal value of that month.

It needs no training and captures seasonality perfectly, but ignores growth and events. Any
other model must beat it to be worth using.

### 6.2 SARIMAX (statistical model)

**SARIMAX** = Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors.
It models the series as a combination of its own recent past, its value one season (12 months)
ago, recent forecast errors, and external event variables.

Settings: **SARIMAX(1,0,1)(0,1,1)₁₂** on log(visitors), with the event dummies as regressors.

| Part | Value | Meaning |
| --- | --- | --- |
| p = 1 | AR(1) | This month depends on last month |
| d = 0 | – | No ordinary differencing |
| q = 1 | MA(1) | Corrects using last month's error |
| P = 0, D = 1, Q = 1, s = 12 | Seasonal | Compares each month with the same month last year (seasonal difference) and corrects with last year's seasonal error |
| Exogenous | `kumbh`, `covid`, `post_covid` | Each gets a coefficient: its effect on log(visitors) |

Because the target is in logs, a regressor's coefficient β means the event multiplies visitors
by about e^β. For example, β = 0.25 means about 28% more visitors in those months.

**Why it was chosen:** it is the standard model for monthly seasonal data, it is
interpretable, and it produces proper uncertainty ranges.

**80% ranges for yearly totals.** The code simulates **1,000 possible future paths** from the
fitted model, adds up each path's 12 months, and takes the 10th and 90th percentiles of those
yearly totals. This is more correct than adding up the monthly upper bounds, which would assume
every month is extreme at once and overstate the range. Because the simulation is random, the
range can shift slightly between runs.

### 6.3 Gradient Boosting (machine-learning model)

A **Gradient Boosting Regressor** (scikit-learn) builds 400 small decision trees one after
another; each new tree corrects the errors of the trees before it.

| Setting | Value | Why |
| --- | --- | --- |
| `n_estimators` | 400 | Number of trees |
| `max_depth` | 3 | Small trees, to avoid memorising the data |
| `learning_rate` | 0.05 | Each tree makes a small correction |
| `subsample` | 0.9 | Each tree sees 90% of rows, which reduces overfitting |
| `random_state` | 42 | Same result every run |

Training details:

- Trained on log(visitors) with the features in section 5.6.
- **COVID months are removed from training** so the model learns normal behaviour.
- **Recursive forecasting:** to forecast 36 months, the model predicts month 1, then uses that
  prediction as the `lag_12` input when it reaches the same month a year later, and so on.

**Known weakness:** tree models cannot predict values outside the range seen in training. When
demand reaches new record highs (as in 2023–24), the model caps its forecast near the old
maximum. This explains its high error on Uttarakhand (section 8).

### 6.4 Ensemble

**Ensemble = average of SARIMAX and Gradient Boosting.** Averaging two different kinds of
model often cancels their individual errors. The ensemble is used for the final forecast
numbers; the uncertainty range comes from SARIMAX.

---

## 7. Evaluation method

### 7.1 Backtest (hold-out test)

The last **24 months** of data are hidden from every model. Each model is trained on the
earlier data, forecasts those 24 months, and is scored against what actually happened. This
simulates using the model in real life: you only ever know the past.

```
Estimated data:   Jan 2017 ─────── training ─────── Dec 2022 │ Jan 2023 ── test ── Dec 2024
```

### 7.2 Metrics

| Metric | Formula | Meaning |
| --- | --- | --- |
| **MAE** | mean of \|actual − forecast\| | Average error, in lakh visitors per month |
| **RMSE** | √(mean of (actual − forecast)²) | Like MAE but punishes big misses more |
| **MAPE** | mean of \|actual − forecast\| / actual × 100 | Average error as a percentage; easiest to compare across states |

Lower is better for all three. The dashboard ranks models by MAPE.

---

## 8. Results

### 8.1 Backtest results — estimated dataset

**Uttarakhand** (test period Jan 2023 – Dec 2024):

| Rank | Model | MAPE (%) | MAE (lakh) | RMSE (lakh) |
| --- | --- | --- | --- | --- |
| 1 | SARIMAX | 3.18 | 1.41 | 1.62 |
| 2 | Seasonal naive | 7.76 | 3.87 | 4.27 |
| 3 | Ensemble (SARIMAX + GBM) | 9.57 | 4.61 | 5.33 |
| 4 | Gradient Boosting | 20.84 | 9.73 | 10.96 |

**Himachal Pradesh** (same period):

| Rank | Model | MAPE (%) | MAE (lakh) | RMSE (lakh) |
| --- | --- | --- | --- | --- |
| 1 | Seasonal naive | 11.17 | 1.64 | 1.93 |
| 2 | Gradient Boosting | 14.16 | 2.10 | 2.59 |
| 3 | Ensemble (SARIMAX + GBM) | 15.59 | 2.49 | 3.14 |
| 4 | SARIMAX | 26.94 | 3.88 | 4.28 |

### 8.2 How to read these results

- **SARIMAX wins clearly for Uttarakhand.** With the `post_covid` level shift, it tracks the
  steady post-COVID growth well.
- **Gradient Boosting fails for Uttarakhand** because 2023–24 were record years, higher than
  anything in its training data, and tree models cannot forecast above their training range.
  This also drags down the ensemble.
- **SARIMAX does badly for Himachal.** With training ending in December 2022, it has only one
  post-COVID year (2022) to estimate the level shift from. Himachal also had an unusual
  2017→2018 drop (1.96 → 1.65 crore) that the model cannot explain.
- **The simple baseline wins for Himachal.** This is a valid and useful finding: with short,
  noisy data, complex models do not automatically help.

### 8.3 Forecasts — estimated dataset

Yearly totals in crore, ensemble forecast with 80% range from SARIMAX simulation:

| Year | Uttarakhand | 80% range | Himachal Pradesh | 80% range |
| --- | --- | --- | --- | --- |
| 2025 | 6.19 | 5.03 – 7.63 | 1.77 | 1.38 – 2.15 |
| 2026 | 6.21 | 4.53 – 7.93 | 1.78 | 1.14 – 2.47 |
| 2027 | 6.41 | 4.67 – 8.42 | 1.77 | 1.03 – 2.56 |

The Uttarakhand rise in 2027 comes from the Kumbh feature (January–April 2027). The ranges
widen each year because uncertainty grows the further ahead the forecast goes.

### 8.4 Results on the synthetic dataset (pipeline test only)

For reference only. Test period Jan 2024 – Dec 2025, Uttarakhand: SARIMAX 3.83%, seasonal
naive 4.60%, ensemble 5.34%, Gradient Boosting 8.07%. These numbers only show that the code
works; they say nothing about real tourism.

---

## 9. Hotspot forecasts (spatial split)

`hotspots.py` splits each state's monthly forecast into **20 major destinations** (11 in
Uttarakhand, 9 in Himachal). This is a **top-down** method: the state forecast stays the single
source of truth, and the hotspots never add up to more than it.

### 9.1 Method

For each month m, every hotspot h gets a "pull":

```
pull(h, m)  = weight(h) × seasonal profile(h, m) × Kumbh boost(h, m)
share(h, m) = pull(h, m) / (sum of all hotspots' pulls + rest-of-state weight)
visitors(h, m) = state forecast(m) × share(h, m)
```

- **weight** — rough share of the state's yearly visitors (placeholder values).
- **seasonal profile** — each kind of place has its own monthly pattern, scaled to average 1.
- **rest of state** — whatever weight is left over (1 − sum of weights) goes to unmapped places,
  with a flat profile.
- **Kumbh boost** — Haridwar's pull is multiplied by 1.6 in Kumbh months. The value was chosen so
  that the extra Kumbh crowd the state model predicts goes to Haridwar while other hotspots stay
  about the same. In February 2027 Haridwar rises about 65% while nearby places rise about 3%.

### 9.2 Seasonal profiles

| Profile | Pattern | Used for |
| --- | --- | --- |
| `ganga_pilgrim` | Year-round, July (Kanwar) spike | Haridwar, Rishikesh |
| `chardham` | Closed Nov–Apr; peaks May–Jun and Oct | Kedarnath, Badrinath, Gangotri, Yamunotri |
| `hill_station` | Summer peak, small autumn bump | Dehradun, Nainital, Kullu, Kasol, Dharamshala, Kasauli |
| `hill_snow` | Summer peak + winter snow / New Year | Mussoorie, Shimla, Manali, Dalhousie |
| `snow` | Winter (Dec–Feb) | Auli |
| `wildlife` | Closed in monsoon (Jul–Sep) | Jim Corbett |
| `high_altitude` | Open roughly Jun–Oct only | Spiti (Kaza) |
| `temple` | Navratri (Mar–Apr, Sep–Oct), Shravan (Aug) | Naina Devi |

### 9.3 Hotspot list

| State | Hotspot | District | Weight | Profile |
| --- | --- | --- | --- | --- |
| Uttarakhand | Haridwar | Haridwar | 0.30 | ganga_pilgrim (Kumbh boost 1.6) |
| Uttarakhand | Rishikesh | Dehradun | 0.12 | ganga_pilgrim |
| Uttarakhand | Nainital | Nainital | 0.08 | hill_station |
| Uttarakhand | Dehradun | Dehradun | 0.07 | hill_station |
| Uttarakhand | Mussoorie | Dehradun | 0.07 | hill_snow |
| Uttarakhand | Kedarnath | Rudraprayag | 0.04 | chardham |
| Uttarakhand | Badrinath | Chamoli | 0.04 | chardham |
| Uttarakhand | Jim Corbett (Ramnagar) | Nainital | 0.03 | wildlife |
| Uttarakhand | Gangotri | Uttarkashi | 0.015 | chardham |
| Uttarakhand | Yamunotri | Uttarkashi | 0.012 | chardham |
| Uttarakhand | Auli | Chamoli | 0.005 | snow |
| Himachal | Shimla | Shimla | 0.20 | hill_snow |
| Himachal | Manali | Kullu | 0.17 | hill_snow |
| Himachal | Dharamshala-McLeodganj | Kangra | 0.11 | hill_station |
| Himachal | Kullu | Kullu | 0.06 | hill_station |
| Himachal | Naina Devi | Bilaspur | 0.06 | temple |
| Himachal | Dalhousie | Chamba | 0.04 | hill_snow |
| Himachal | Kasauli | Solan | 0.04 | hill_station |
| Himachal | Kasol | Kullu | 0.03 | hill_station |
| Himachal | Spiti (Kaza) | Lahaul & Spiti | 0.015 | high_altitude |

> The **coordinates are real**, but the **weights and profiles are placeholders**, not official
> statistics. Replace them with district-wise shares from indiastat or the tourism departments.

---

## 10. Dashboard

`app.py` is a Streamlit web app (run `py -m streamlit run app.py`; it opens in the browser).

**Controls (top row, apply to every tab):** state toggle (Uttarakhand / Himachal Pradesh),
forecast horizon (12–48 months), backtest window (12–36 months), and a "Data source" menu to
choose the estimated or synthetic dataset or upload your own CSV. A badge in the header always
shows which kind of data is in use.

| Tab | Shows |
| --- | --- |
| **Overview** | KPI cards (last year's visitors, next year's forecast and % change, peak month, best model's MAPE), the actual + forecast chart with the 80% band and the Kumbh period shaded, and a table of yearly totals. |
| **Hotspots** | A 2D map of both states. Circle **area** is proportional to visitors (blue = Uttarakhand, orange = Himachal). Next to it, a ranking bar chart. Filters: year, whole year or one month, comparison year. |
| **Model evaluation** | Models ranked by MAPE, and their test-period predictions plotted against the actual values. |
| **Seasonality** | Both states' monthly pattern as % of their own yearly total, so they are comparable on one scale. |
| **Data** | The input table and CSV downloads (forecast, hotspot forecast, backtest scores). |

**Design choices:**

- One colour per state everywhere.
- The colours were checked for colour-blind safety.
- Every chart has hover tooltips.
- Map labels are placed beside circles so they do not overlap.
- Models are cached, so switching tabs does not retrain them.

---

## 11. Limitations and next steps

### Limitations

1. **Monthly values are estimated.** Only yearly totals are official. The backtest therefore
   mainly tests how well each model follows the yearly level, not real month-to-month behaviour.
2. **Short history.** 96 months (2017–2024), including two COVID-disrupted years. Only 2022 is
   available to learn the post-COVID level before the test period starts.
3. **Hotspot shares are placeholders.** Hotspot numbers are illustrative until real
   district-wise data is added.
4. **Sources disagree.** State tourism boards and the Ministry publish different totals. The
   project uses the Ministry series consistently.
5. **SARIMAX order is fixed**, not selected automatically.

### Next steps

1. Collect real monthly data (indiastat, RTI) and place it in `data/real/`.
2. Organise the code into `data/`, `src/`, `outputs/`, `notebooks/`.
3. Add an exploratory analysis notebook: trend, STL decomposition, ACF/PACF plots.
4. Add optional models: Prophet and LSTM.
5. Use rolling-origin cross-validation, and choose the SARIMA order by AIC.
6. Add external features: rainfall (IMD), Google Trends ("Kedarnath", "Char Dham"), Char Dham
   registrations as a leading indicator.
7. Produce district-level forecasts from real district data instead of placeholder shares.

---

## 12. Glossary

| Term | Meaning |
| --- | --- |
| Crore / lakh | 1 crore = 10,000,000 (10 million); 1 lakh = 100,000 |
| DTV / FTV | Domestic / Foreign Tourist Visits (Ministry of Tourism terms) |
| Backtest | Testing a model on past data it was not trained on |
| Hold-out period | The final months hidden from the model for testing |
| Exogenous regressor | An outside variable fed into a model (e.g. the Kumbh dummy) |
| Dummy variable | A 0/1 feature marking whether something applies in a month |
| Structural break | A permanent change in a series' level (e.g. after COVID) |
| Seasonal differencing | Comparing each month with the same month a year earlier |
| Recursive forecasting | Feeding a model's own predictions back in as inputs for later months |
| Temporal disaggregation | Splitting yearly totals into monthly values |
| Top-down disaggregation | Splitting a total (state) into parts (hotspots) by shares |
| Ensemble | Combining several models' forecasts, here by averaging |
| MAPE | Mean Absolute Percentage Error |
| 80% interval | A range expected to contain the true value 8 times out of 10 |

---

## 13. Sources

Official yearly totals (Ministry of Tourism, Government of India):

- India Tourism Data Compendium 2025, Table 4.1.2 (2023, 2024) —
  https://data.tourism.gov.in/mrd/Uploads/tourism_data/India%20Tourism%20Data%20Compendium%202025_1.pdf
- India Tourism Data Compendium 2024, Table 4.1.2 (2022) —
  https://www.data.tourism.gov.in/mrd/Uploads/tourism_data/India%20Tourism%20Data%20Compendium%202024.pdf
- India Tourism Statistics 2022, Table 5.1.2 (2020, 2021) —
  https://static.pib.gov.in/WriteReadData/userfiles/IndiaTourismStatistics2022English.pdf
- State/UT-wise Domestic and Foreign Tourist Visits during 2018 & 2019 (2018, 2019; archived copy) —
  https://web.archive.org/web/2021/https://tourism.gov.in/sites/default/files/2020-08/Figures.pdf
- India Tourism Statistics 2019 (2017; archived copy) —
  https://web.archive.org/web/2021/https://tourism.gov.in/sites/default/files/2020-04/India%20Tourism%20Statistics%202019.pdf

Hotspot coordinates are standard geographic locations. Basemap in the dashboard:
© CARTO, © OpenStreetMap contributors.
