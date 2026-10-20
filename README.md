# CrowdCast

CrowdCast forecasts NYC Yellow Taxi pickups for selected zones one to six
hours ahead. It compares a shared LSTM and GRU with seasonal baselines,
produces P10/P50/P90 forecasts, and converts the forecast into expected and
safe-plan fleet requirements.

The dashboard uses saved artifacts from the historical test period. It does
not use a live feed or train models when the app starts.

## Requirements

- Windows, macOS, or Linux
- Python 3.11
- `uv`
- The dependencies pinned in [requirements.txt](requirements.txt)

The project is configured to use eight consecutive months of 2026 Yellow Taxi
data: January through August.

## Environment setup with uv

From the repository root:

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

Run project commands through the environment's Python executable:

```powershell
.venv\Scripts\python.exe --version
```

On macOS or Linux, use `.venv/bin/python` instead.

## Raw data

Place these files in `data/raw/`:

```text
yellow_tripdata_2026-01.parquet
yellow_tripdata_2026-02.parquet
yellow_tripdata_2026-03.parquet
yellow_tripdata_2026-04.parquet
yellow_tripdata_2026-05.parquet
yellow_tripdata_2026-06.parquet
yellow_tripdata_2026-07.parquet
yellow_tripdata_2026-08.parquet
taxi_zone_lookup.csv
```

Download the Yellow Taxi Parquet files and the taxi-zone lookup from the
official [NYC TLC Trip Record Data page](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).

Raw data is intentionally ignored by Git. Do not commit it.

## Configuration

- `config/default.yaml` contains the full eight-month experiment settings.
- `config/smoke.yaml` overrides the defaults for a reduced run using one
  month, six zones, two epochs, and one seed.

Configuration is loaded and merged by `src.config.load_config`.

## Pipeline order

Each stage reads its inputs from disk and writes its outputs to disk:

```text
raw TLC files
  → ingest
  → hourly processed data
  → feature construction and windows
  → LSTM/GRU training
  → evaluation and baselines
  → Streamlit dashboard
```

### 1. Ingest and aggregate

```powershell
.venv\Scripts\python.exe scripts\ingest.py --config config/default.yaml
```

This cleans trip records, aggregates hourly pickups by zone, fills missing
hours with zero pickups, and writes `data/processed/hourly_all.parquet`.

### 2. Build features and windows

```powershell
.venv\Scripts\python.exe scripts\build_features.py --config config/default.yaml
```

This selects zones, creates chronological splits, fits train-only statistics,
computes busy-time information, and writes feature windows.

### 3. Train a model

```powershell
.venv\Scripts\python.exe scripts/train.py --config config/default.yaml --model lstm --seed 0
```

Use `--model gru` for the GRU. The training stage saves checkpoints, histories,
and summaries under `artifacts/`.

### 4. Evaluate

```powershell
.venv\Scripts\python.exe scripts/evaluate.py --config config/default.yaml --experiment all
```

Evaluation produces baseline predictions, model metrics, aggregated results,
and the validation-selected application model.

### 5. Run the dashboard

```powershell
.venv\Scripts\python.exe -m streamlit run app/main.py
```

The app reads saved artifacts only. It does not train models.

## Smoke configuration

Use the smoke configuration when a quick pipeline check is required:

```powershell
.venv\Scripts\python.exe scripts/ingest.py --config config/smoke.yaml
.venv\Scripts\python.exe scripts/build_features.py --config config/smoke.yaml
.venv\Scripts\python.exe scripts/train.py --config config/smoke.yaml --model lstm --seed 0
.venv\Scripts\python.exe scripts/evaluate.py --config config/smoke.yaml --experiment E1
```

The smoke run is for validation and debugging. Its metrics are not the final
experiment results.

## Tests

Run the complete test suite with:

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Run a focused suite, for example:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_ingest.py -v
```

## Reproducibility and scientific safeguards

- Splits are chronological, never random.
- Scaling statistics, busy-time values, zone selection, and peak thresholds
  use training data only.
- Validation data controls early stopping and application-model selection.
- The test set remains untouched until final evaluation.
- LSTM and GRU use the same data, features, hyperparameters, and seeds.
- Predictions are inverse-transformed and clipped at zero before metrics.
- Quantiles are ordered so P10 ≤ P50 ≤ P90 at inference.
- Results are reported honestly, including when a baseline wins.

## Project structure

```text
config/       YAML configuration
data/raw/     downloaded TLC files; ignored by Git
data/processed/ generated tables and windows; ignored by Git
src/          reusable pipeline modules
scripts/      thin command-line stage wrappers
app/          Streamlit dashboard
artifacts/    checkpoints and metrics; ignored by Git
tests/        pytest tests
docs/reports/ task reports
```
