# ARCHITECTURE.md: CrowdCast

This document fixes every name, shape, schema and interface so the coding agent never has to guess. If the PRD and this file conflict, ask the user. If something here is missing, ask; do not invent.

## 1. Data flow

```
data/raw/*.parquet + taxi_zone_lookup.csv
  -> scripts/ingest.py          -> data/processed/hourly_all.parquet
  -> scripts/build_features.py  -> data/processed/{hourly_selected.parquet, splits.json,
                                    zone_stats.json, busy_time.parquet,
                                    windows_L{L}_{train,val,test}.npz}
  -> scripts/train.py           -> artifacts/{model}_L{L}_s{seed}/{model.pt, history.json, train_summary.json}
  -> scripts/evaluate.py        -> artifacts/{tag}/pred_{val,test}.npz, artifacts/baselines/pred_{val,test}.npz,
                                    artifacts/metrics/*.json, artifacts/results.json, artifacts/app_model.json
  -> streamlit run app/main.py  (reads saved files only; never trains)
```

Every stage reads from disk and writes to disk, so any stage can be re-run alone.

## 2. Repository layout and modules

```
config/default.yaml          full run
config/smoke.yaml            tiny fast run for smoke tests (1 month, 6 zones, 2 epochs, 1 seed)
src/config.py                load_config(path) -> dict; set_seed(seed) -> None
src/data/ingest.py           load_trips, clean_trips, aggregate_hourly
src/data/zones.py            select_zones
src/features/splits.py       compute_splits
src/features/scaling.py      fit_zone_stats, apply_scaling, compute_busy_time
src/features/windows.py      build_windows
src/models/rnn.py            QuantileRNN
src/models/losses.py         pinball_loss
src/train/trainer.py         train_model
src/evaluate/baselines.py    lag_baseline
src/evaluate/metrics.py      mae, rmse, wape, peak_mae, coverage, interval_width
src/evaluate/run_eval.py     evaluate_run, aggregate_results, pick_app_model
src/fleet/fleet.py           required_taxis, fleet_table
scripts/ingest.py  build_features.py  train.py  evaluate.py     (thin CLI wrappers)
app/main.py  app/pages/{1_Forecast.py, 2_Zone_Comparison.py, 3_Model_Evaluation.py}  app/loaders.py
tests/test_*.py
```

## 3. Config: `config/default.yaml`

All values below are defaults. Code reads every path, threshold and hyperparameter from here.

```yaml
paths:
  raw_dir: data/raw
  processed_dir: data/processed
  artifacts_dir: artifacts
data:
  months: ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06"]  # adjust if files unavailable
  min_duration_min: 1
  max_duration_min: 180
zones:
  n_high: 6
  n_medium: 6
  n_low: 6
  min_nonzero_hour_fraction: 0.5      # exclude zones with mostly empty hours (train period)
split:
  fractions: [0.70, 0.15, 0.15]       # train, val, test, chronological
windows:
  lookback: 168
  horizon: 6
  quantiles: [0.1, 0.5, 0.9]
model:
  hidden_size: 64
  num_layers: 2
  dropout: 0.2
  zone_emb_dim: 8
train:
  lr: 0.001
  batch_size: 256
  max_epochs: 50
  patience: 5
  seeds: [0, 1, 2]
  device: auto                        # cuda if available, else cpu
evaluation:
  peak_percentile: 90                 # of train-period pickups, per zone
fleet:
  deadhead_min: 5
  utilization: 0.7
experiments:
  lookbacks: [24, 168]                # E2
```

## 4. Data schemas

### 4.1 Raw inputs

- `data/raw/yellow_tripdata_YYYY-MM.parquet`. Columns used: `tpep_pickup_datetime`, `tpep_dropoff_datetime`, `PULocationID`. All other columns are ignored.
- `data/raw/taxi_zone_lookup.csv`. Columns: `LocationID`, `Borough`, `Zone`, `service_zone`.
- Timestamps are local New York time without a timezone. Floor to the hour. Daylight saving changes may cause one missing or duplicated hour; missing hours are filled with zero pickups (see 4.2). Do not convert time zones.

### 4.2 `hourly_all.parquet` (all zones)

| Column | Type | Meaning |
| --- | --- | --- |
| zone_id | int16 | TLC LocationID |
| hour_ts | datetime64[ns] | Hour start, naive local time |
| pickups | int32 | Number of cleaned trips starting that hour |
| mean_duration_min | float32 | Mean trip duration that hour; NaN when pickups = 0 |

The table is complete: every zone has a row for every hour between the first and last hour in the data. Missing hours are filled with `pickups = 0`.

Cleaning rules (applied in `clean_trips`, with counts logged in a report dict): drop rows with a missing pickup time, dropoff time or zone ID; drop pickups outside the months in the config; drop durations below `min_duration_min` or above `max_duration_min`; drop zone IDs not in the lookup file.

### 4.3 Split rule (`splits.json`)

- Timeline = sorted unique hours in the data. Boundaries from `split.fractions`, saved as `{train: [start, end], val: [start, end], test: [start, end]}` (ISO strings, inclusive).
- A sample belongs to a split only if **all of its target hours lie inside that split's range**. Its input window may reach back into earlier splits (that is only past data). Samples whose targets straddle a boundary are discarded. Per zone, the expected sample counts are `n_train - L - H + 1` (train, the first `L` hours are needed as input), `n_val - H + 1` (val) and `n_test - H + 1` (test), where `n_*` is the number of hours in the split.
- Scaler statistics, busy times, peak thresholds and zone selection use the **train period only**.

### 4.4 Zone selection

Rank zones by total train-period pickups. Drop zones whose fraction of non-zero train hours is below `min_nonzero_hour_fraction`. Split the remainder into three equal-sized volume tiers (high, medium, low) and take `n_high`, `n_medium`, `n_low` zones spread evenly across each tier. Log the chosen zone IDs and names. The result is `hourly_selected.parquet`: same schema as 4.2, restricted to chosen zones.

### 4.5 `zone_stats.json`

```json
{
  "zone_ids": [132, 161, ...],
  "zone_index": {"132": 0, "161": 1, ...},
  "stats": {"132": {"mean_log": 5.1, "std_log": 0.9, "peak_threshold": 410.0}, ...}
}
```

`mean_log` and `std_log` are the mean and standard deviation of `log1p(pickups)` over the train period for that zone. `peak_threshold` is the `peak_percentile` of raw train pickups for that zone. If `std_log` is 0, raise an error.

### 4.6 `busy_time.parquet`

Columns: `zone_id`, `hour_of_day`, `mean_duration_min`. Computed from the train period, weighted by pickups. If a zone and hour has no train pickups, fall back to the zone's overall mean, then to the global mean.

## 5. Features, windows and tensor shapes

Per input timestep (F = 6 features, in this order):

1. `z_pickups` = (log1p(pickups) - mean_log) / std_log, using that zone's train statistics
2. `sin_hod`, 3. `cos_hod` (hour of day, period 24)
4. `sin_dow`, 5. `cos_dow` (day of week, period 7)
6. `is_weekend` (0 or 1)

`windows_L{L}_{split}.npz` keys, with N samples, lookback L, horizon H = 6, Q = 3:

| Key | Shape | Type | Meaning |
| --- | --- | --- | --- |
| X | (N, L, 6) | float32 | Input features |
| zone_idx | (N,) | int64 | Index into `zone_index` |
| y_scaled | (N, H) | float32 | Targets in scaled log space (same transform as feature 1) |
| y_raw | (N, H) | float32 | Targets as raw pickup counts |
| origin_ts | (N,) | datetime64[ns] | Last input hour (forecast origin) |
| zone_id | (N,) | int16 | TLC LocationID |

Target hour h (1-based) is `origin_ts + h hours`. Samples are ordered by (zone_id, origin_ts).

## 6. Model interface

```python
# src/models/rnn.py
class QuantileRNN(nn.Module):
    def __init__(self, cell: str, n_features: int, n_zones: int, emb_dim: int,
                 hidden_size: int, num_layers: int, dropout: float,
                 horizon: int, n_quantiles: int): ...
    def forward(self, x: Tensor, zone_idx: Tensor) -> Tensor:
        """x: (B, L, F), zone_idx: (B,) -> (B, H, Q), scaled space, unsorted."""
    def predict_quantiles(self, x, zone_idx) -> Tensor:
        """Same shape, sorted along the last dimension so P10 <= P50 <= P90."""
```

- `cell` is `"lstm"` or `"gru"`; everything else is identical between them.
- The encoder uses the last timestep of the top layer. It is concatenated with the zone embedding and passed through one `Linear(hidden_size + emb_dim, H * Q)`, then reshaped to (B, H, Q).
- `pinball_loss(pred: (B,H,Q), target: (B,H), quantiles: (Q,)) -> scalar` in `src/models/losses.py`: mean over batch, horizons and quantiles.

## 7. Training outputs

`scripts/train.py --model {lstm,gru} --seed S [--lookback L]`. Run tag: `{model}_L{L}_s{seed}`. Writes `artifacts/{tag}/`:

- `model.pt`: dict with `state_dict`, `config` (the full config used), `zone_index`, `n_features`, `lookback`.
- `history.json`: `{"train_loss": [...], "val_loss": [...]}` per epoch.
- `train_summary.json`: `{"tag", "model", "seed", "lookback", "best_epoch", "best_val_loss", "train_time_sec", "n_params"}`.

The checkpoint with the lowest validation loss is saved. Training uses the train split only; the validation split is used for early stopping only.

## 8. Evaluation outputs

### 8.1 Predictions

`artifacts/{tag}/pred_{val,test}.npz`: `origin_ts (N,)`, `zone_id (N,)`, `y_true (N,H)` raw counts, `q_pred (N,H,Q)` raw counts (sorted, expm1, clipped at 0).

`artifacts/baselines/pred_{val,test}.npz`: same keys except `q_pred` is replaced by `b1 (N,H)` (same hour yesterday: value at `origin_ts + h - 24`) and `b2 (N,H)` (same hour last week: `origin_ts + h - 168`). Both always refer to hours already observed at the origin, since h <= 6.

### 8.2 Metrics

For models, metrics use the P50 for MAE, RMSE, WAPE and peak MAE; for baselines they use the point forecast. Definitions:

- `mae`, `rmse`: on raw counts.
- `wape` = sum(|error|) / sum(actual).
- `peak_mae`: MAE over samples whose actual value is at or above that zone's `peak_threshold`.
- `coverage`: fraction of actual values within [P10, P90] (models only). `width`: mean of P90 - P10 (models only).

`artifacts/metrics/{tag}_{split}.json`:

```json
{"tag": "gru_L168_s0", "split": "test",
 "per_horizon": {"1": {"mae": 0, "rmse": 0, "wape": 0, "peak_mae": 0, "coverage": 0, "width": 0}, "...": {}},
 "overall": {"mae": 0, "rmse": 0, "wape": 0, "peak_mae": 0, "coverage": 0, "width": 0},
 "per_zone": {"132": {"mae": 0, "wape": 0}},
 "train_time_sec": 0, "n_params": 0}
```

`artifacts/results.json` aggregates across seeds: `{experiment: {model_or_baseline: {split: {metric: {"mean": x, "std": y}}}}}`, with experiments `E1` (lookback 168), `E2` (lookback 24 vs 168) and optionally `E3`. A CSV table of E1 is also written for the report.

### 8.3 App model choice

`artifacts/app_model.json`: `{"tag": "...", "selected_by": "val_mae"}`. The tag is the run with the lowest **validation** overall MAE. The test set is never used to choose it.

## 9. Fleet module

```python
# src/fleet/fleet.py
def required_taxis(pickups: float, busy_min: float, utilization: float) -> int:
    """ceil(D * T / (60 * U)). Raise ValueError if pickups < 0, busy_min <= 0, or utilization not in (0, 1]."""

def fleet_table(origin_ts, zone_id, q_pred: np.ndarray, busy_time: pd.DataFrame,
                available: int, deadhead_min: float, utilization: float) -> pd.DataFrame:
    """One row per horizon hour h = 1..H. Columns:
    horizon, hour_ts, p10, p50, p90, busy_min, n_expected, n_safe, available,
    gap_expected, gap_safe, status."""
```

- `busy_min` = `mean_duration_min` for that zone and target hour of day, plus `deadhead_min`.
- `n_expected` uses P50; `n_safe` uses P90. `gap_* = n_* - available` (positive = shortage).
- `status`: `"shortage"` if `gap_expected > 0`; `"risk"` if `gap_safe > 0 >= gap_expected`; otherwise `"ok"`.

## 10. App contract

- `app/loaders.py` is the only place that reads files: config, `app_model.json`, the chosen run's `pred_test.npz`, `hourly_selected.parquet`, `zone_stats.json`, `busy_time.parquet`, `results.json`, metrics files, and the zone lookup names. Loaders use `st.cache_data`.
- The app shows saved test-period predictions; it does not run the model. "Current time" is chosen from the test-period `origin_ts` values, and the UI states that no live data is used.
- Forecast page: zone dropdown, origin time selector, last 48 hours of actuals from `hourly_selected.parquet`, a 6-hour forecast with a P10 to P90 band, and a fleet table from `fleet_table` with inputs for available taxis, deadhead minutes and utilization.
- Zone comparison page: multi-select zones, P50 and the fleet requirement for the next 6 hours.
- Model evaluation page: `results.json` tables, error by horizon, predicted vs actual overlay, coverage.
- If a required artifact is missing, show a clear message naming the script to run; never crash with a traceback.

## 11. Script interfaces

All scripts take `--config config/default.yaml` (default path) and exit with a non-zero status on failure.

- `ingest.py`: runs load, clean and aggregate; writes `hourly_all.parquet`; prints the cleaning report.
- `build_features.py [--lookback L]`: runs steps 4.3 to 4.6 and window building; writes everything in section 1 for that lookback (repeat for each lookback in `experiments.lookbacks`).
- `train.py --model M --seed S [--lookback L]`.
- `evaluate.py [--experiment E1|E2|E3|all]`: predictions, baselines, metrics, `results.json`, `app_model.json`.

## 12. Tests

| File | Must verify |
| --- | --- |
| test_ingest.py | Cleaning rules drop the right rows; hourly table is complete with zero fill; mean duration is NaN when pickups = 0. |
| test_splits.py | Chronological order; every target hour of every sample lies inside its own split's range; the target-hour ranges of train, val and test are disjoint and ordered; per-zone sample counts match section 4.3. |
| test_scaling.py | Statistics come from the train period only (changing test data does not change them); `std_log = 0` raises. |
| test_windows.py | Shapes match section 5; target alignment is `origin_ts + h`; sample counts are consistent; no NaN. |
| test_models.py | `forward` output shape (B, H, Q); `predict_quantiles` is sorted; LSTM and GRU differ only in the cell. |
| test_losses.py | Pinball loss is zero for a perfect prediction; asymmetry is correct for q = 0.1 vs 0.9; a hand-computed example matches. |
| test_baselines.py | Lag offsets are correct and only use hours observed at the origin. |
| test_metrics.py | Hand-computed MAE, RMSE, WAPE, peak MAE and coverage match. |
| test_fleet.py | The PRD example (D = 48, T = 22, U = 0.7 gives 26; D = 61 gives 32); invalid inputs raise; status rule. |

## 13. Conventions

- Zone IDs are TLC `LocationID` integers everywhere on disk; the model uses `zone_index` internally.
- Hour timestamps are naive local time floored to the hour.
- Raw pickup counts are `int`/`float32` on disk; scaled values are only used inside windows and training.
- All random operations take their seed from the config or the command line.
- Keep files small and functions pure where possible; side effects (reading and writing files) live in scripts and loaders.
