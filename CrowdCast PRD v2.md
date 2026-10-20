# CrowdCast: Taxi Demand Forecasting System (PRD v2)

Deep Learning Lab project · LSTM vs GRU · multi-horizon quantile forecasting · Streamlit dashboard

## 1. Overview

CrowdCast forecasts taxi pickups for each of 15 to 20 NYC taxi zones for the next 1 to 6 hours, with an uncertainty range, and converts the forecast into a fleet requirement a dispatcher can act on. An LSTM and a GRU are trained under an identical protocol and compared against simple seasonal baselines. The deep learning experiment is the core of the project; the dashboard exists to make its output usable.

**Illustrative example only (real numbers come from the trained models).** For a busy zone at 6 pm, the model predicts a median of 48 pickups next hour, with an 80% range of 38 to 61. With an average busy time of 22 minutes per pickup and 0.7 target utilization, the expected requirement is 26 taxis and the safe-plan requirement is 32. If the operator has 28 taxis available, the dashboard shows a surplus of 2 against the expected case and a shortage of 4 against the safe plan.

## 2. Problem statement

Taxi demand varies by zone and hour. Without reliable forecasts, operators put too few taxis where demand is high and too many where it is low. A single-number forecast is also risky to plan around, because it hides how uncertain the prediction is. CrowdCast provides a short-horizon forecast with a range, and turns both into an understandable fleet estimate.

## 3. Objectives

- Forecast hourly pickups for the next 6 hours in each selected zone.
- Train one shared LSTM and one shared GRU across all selected zones using a learned zone embedding.
- Output a 10th, 50th and 90th percentile forecast for each horizon step, trained with a quantile (pinball) loss.
- Compare both models against two seasonal baselines on chronologically later, unseen data.
- Translate the forecast into a fleet requirement (expected and safe-plan) and a fleet gap.
- Present everything in a simple three-page Streamlit dashboard.

## 4. Target users

- Taxi fleet operators and dispatch or operations teams, who use the fleet estimate and gap.
- Transportation planners interested in demand patterns, who use the zone comparison and peak analysis.

## 5. Functional requirements

| ID | Feature | Requirement |
| --- | --- | --- |
| FR-01 | Data processing | Load TLC Parquet files, clean records, aggregate pickups by hour and zone, fill missing hours with zero. |
| FR-02 | Zone selection | Select 15 to 20 zones stratified by total volume (high, medium, low), driven by a config file. |
| FR-03 | Windowing and splits | Build 168-hour input windows with 6-hour targets. Split chronologically into train, validation and test. Fit scalers on train only. |
| FR-04 | Models | LSTM and GRU with zone embedding and 18 outputs (6 hours × 3 quantiles). |
| FR-05 | Training | Identical training protocol for both models, early stopping on validation loss, 3 seeds each, saved checkpoints and metrics. |
| FR-06 | Baselines | Same hour yesterday (lag 24) and same hour last week (lag 168). |
| FR-07 | Evaluation | MAE, RMSE, WAPE, peak-hour MAE, interval coverage and width, training time, parameter count, all reported per horizon step. |
| FR-08 | Fleet estimation | Compute required taxis from the forecast, with busy time derived from trip data and configurable utilization. |
| FR-09 | Fleet comparison | Accept the available taxi count and show the gap against both the expected and safe-plan requirement. |
| FR-10 | Dashboard | Three pages: forecast dashboard, zone comparison, model evaluation. |
| FR-11 | Reproducibility | One config file, fixed seeds, and a separate command-line script for each pipeline stage. |

## 6. Deep learning methodology

### 6.1 Pipeline

Raw TLC trips → cleaning → hourly pickups per zone → feature construction → sliding windows → chronological split → train LSTM and GRU → evaluate against baselines → save checkpoints and metrics → dashboard loads saved artifacts (no training at app time).

### 6.2 Inputs and features

- Per timestep (168 steps): log1p of pickup count, standardized per zone using training statistics only; sin and cos of hour of day; sin and cos of day of week; weekend flag.
- Per sample: zone ID, passed through a learned embedding (about 8 dimensions) and concatenated with the recurrent output before the output head.

### 6.3 Architecture

- Recurrent encoder: LSTM or GRU, 1 to 2 layers, hidden size 64, dropout 0.2. Both models use identical hyperparameters, so any difference comes from the cell type. Parameter counts will differ (GRU has fewer) and are reported.
- Output head: linear layer from (hidden + zone embedding) to 18 values, reshaped to 6 horizons × 3 quantiles (0.1, 0.5, 0.9).
- Non-crossing quantiles: sort the three quantile outputs at inference so P10 ≤ P50 ≤ P90.
- Direct multi-horizon prediction: all 6 hours are predicted at once, not recursively.

### 6.4 Loss

Average pinball loss over the 3 quantiles and 6 horizons. Predictions are inverse-transformed (expm1) and clipped at zero before metrics are computed.

### 6.5 Training protocol

Adam, learning rate 1e-3, batch size 256, up to 50 epochs, early stopping on validation loss with patience 5, 3 random seeds per model. Training must be feasible on CPU; a free GPU (such as Colab) is optional.

## 7. Dataset

NYC TLC Yellow Taxi trip records (Parquet) plus the taxi zone lookup table, from the official page: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

- Use at least 6 consecutive recent months. Split by time (about 70% train, 15% validation, 15% test), never randomly.
- Cleaning: drop trips with pickup timestamps outside the file's month, missing zone IDs, zero or negative durations, and implausibly long durations (for example over 3 hours).
- Zone selection: rank zones by total pickups, then pick from the high, medium and low tiers (for example a mix like airports, Midtown and quieter residential zones). Exclude zones where most hours are zero.
- Limitation: completed trips are a proxy for demand and miss unmet requests. With only a few months of data, yearly seasonality and holidays cannot be learned. Available taxi count is entered manually.

## 8. Model evaluation

All metrics are computed on the held-out test period, per horizon step (+1h to +6h) and averaged.

| Metric | Purpose |
| --- | --- |
| MAE (median forecast) | Average absolute error in pickups. |
| RMSE | Penalizes large errors. |
| WAPE | Total absolute error divided by total actual pickups; comparable across busy and quiet zones. |
| Peak-hour MAE | Error on hours at or above each zone's 90th percentile of training demand. |
| Interval coverage and width | Share of actual values inside the P10 to P90 range (target about 80%) and the average range width. |
| Training time and parameters | Computational cost of each model. |

**Baselines:** B1 predicts the value from the same hour on the previous day; B2 predicts the value from the same hour on the previous week.

**Experiments:**

- E1 (main): LSTM vs GRU, 168-hour lookback, 3 seeds, results as mean ± standard deviation.
- E2: lookback comparison, 24 hours vs 168 hours, for the better model type.
- E3 (optional): with and without time features.

The report states whichever model performs better, including if neither beats the best baseline. That is a valid finding and must be analysed, not hidden.

## 9. Fleet requirement calculation

N = ceil( D × T / (60 × U) )

- N: estimated taxis required.
- D: predicted pickups in the hour. Use the median (P50) for the expected case and the P90 for the safe plan.
- T: average taxi busy minutes per pickup. Computed from training data as the average trip duration for that zone and hour of day, plus a configurable allowance (default 5 minutes) for reaching the next passenger.
- U: target utilization as a decimal (default 0.7, configurable).

Fleet gap = required taxis − available taxis. A positive gap means a potential shortage. This is an approximate planning calculation under stated assumptions, not an operational guarantee.

## 10. User interface

One Streamlit app with three pages. There is no live data feed, so the current time is chosen by the user from the held-out test period and the app must say so clearly.

- **Forecast dashboard:** choose zone and current time; shows the last 48 hours of actual pickups and the 6-hour forecast with a P10 to P90 band; fleet table per hour with expected requirement, safe-plan requirement, available taxis and gap; configurable T allowance and U.
- **Zone comparison:** select several zones and compare predicted demand and requirement over the next 6 hours.
- **Model evaluation:** metrics tables for LSTM, GRU and baselines; error by horizon step; predicted vs actual overlay for a chosen zone and period; interval coverage.

Stretch (only if time remains): a map or heatmap of predicted shortage by zone.

## 11. Technology stack and structure

| Component | Technology |
| --- | --- |
| Language | Python 3.11 |
| Data processing | Pandas, NumPy, PyArrow |
| Deep learning | PyTorch |
| Visualization | Plotly |
| Interface | Streamlit |
| Config and tests | YAML config, pytest |

No separate backend, database, Docker deployment or cloud infrastructure is required for this version. Suggested repository layout: config/, data/ (raw and processed, not committed), src/ (data, features, models, train, evaluate, fleet), scripts/ (one per stage), app/ (Streamlit), artifacts/ (checkpoints, metrics), tests/, docs/.

## 12. Scope

**Included:** data cleaning and hourly aggregation; shared LSTM and GRU with quantile outputs; baselines and evaluation; fleet estimate and gap; three-page dashboard.

**Out of scope:** live GPS tracking and dispatch; automatic taxi repositioning; real-time ride-request ingestion; weather or event data; citywide optimization.

## 13. Success criteria

1. The full pipeline runs end to end from the config file with one command per stage.
2. No data leakage: scalers fit on train only, and every test timestamp is later than every training timestamp (checked by automated tests).
3. Both models produce 6-hour P10, P50 and P90 forecasts for all selected zones.
4. Results for LSTM, GRU and both baselines are reported on the test set with mean ± standard deviation over 3 seeds.
5. P10 to P90 interval coverage lies between 70% and 90%.
6. The dashboard runs from saved artifacts and shows forecast, range and fleet gap for any selected zone.
7. Target (not a pass or fail condition): at least one model beats the best baseline on +1h MAE. If not, the report explains why.

## 14. Risks and assumptions

- Strong seasonal baselines may be hard to beat; the report treats this honestly.
- Short data span limits seasonality learning; holidays in the test period may appear as outliers.
- Shared-model training can favour busy zones; WAPE and per-zone MAE are reported to check this.
- Fleet formula ignores spatial movement of taxis between zones.

## 15. Build milestones (to be detailed in the task plan)

- M0: repository setup, config, environment.
- M1: data ingestion, cleaning, hourly aggregation.
- M2: features, windowing, chronological split, leakage tests.
- M3: baselines and evaluation harness.
- M4: LSTM and GRU training with quantile loss.
- M5: experiments E1 to E3 and result tables.
- M6: fleet module and Streamlit dashboard.
- M7: report, documentation and final validation.

## 16. Changes from v1

- Added multi-step 6-hour forecasting and quantile outputs for uncertainty, and a fleet estimate based on both expected and safe-plan demand.
- One shared model with zone embedding instead of per-zone models; 168-hour lookback with a 24-hour comparison.
- Added a second baseline, WAPE, interval coverage, seeds, a validation split and a defined peak-hour rule.
- Busy time T is derived from trip data; the dashboard uses a selectable test-period time because there is no live feed; removed the tracking parameter from the dataset link.
