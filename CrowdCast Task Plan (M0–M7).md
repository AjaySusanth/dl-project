# CrowdCast Task Plan (M0–M7)

Oct 20, 2026 · @ajay

## M0 — Repository setup, config, environment

**Goal:** A runnable repo skeleton with working imports and smoke-test config before any data code is written.

### Tasks

| ID | Task | Inputs | Outputs | Done when |
| --- | --- | --- | --- | --- |
| M0-T1 | Scaffold directory tree | PRD §11, ARCH §2 | All directories created; `.gitignore` covers `data/`, `artifacts/`, `__pycache__` | `tree` matches ARCH §2 exactly |
| M0-T2 | Write `config/default.yaml` | ARCH §3 | `config/default.yaml` with all keys in §3 | `load_config` returns a dict with every key from §3 |
| M0-T3 | Write `config/smoke.yaml` | ARCH §3 comment | `smoke.yaml` overrides: 1 month, 6 zones (2+2+2), 2 epochs, 1 seed | Smoke run completes in < 2 min on CPU |
| M0-T4 | Implement `src/config.py` | ARCH §2 | `load_config(path) -> dict`, `set_seed(seed) -> None` | `load_config` merges YAML; `set_seed` sets Python/NumPy/PyTorch seeds |
| M0-T5 | Pin `requirements.txt` | PRD §11 stack | `requirements.txt` with exact versions | `pip install -r requirements.txt` exits 0 on a clean env |
| M0-T6 | Add `README.md` quickstart | All scripts | One-command-per-stage run order documented | Reader can go from raw data to app with copy-paste |

### Validation to run after M0

```bash
python -c "from src.config import load_config, set_seed; cfg = load_config('config/default.yaml'); assert 'paths' in cfg"
python -c "from src.config import set_seed; set_seed(0)"
```

## M1 — Data ingestion, cleaning, hourly aggregation

**Goal:** Produce `data/processed/hourly_all.parquet` — a complete, clean hourly pickup table for all zones — and validate it before any feature work begins.

### Tasks

| ID | Task | Inputs | Outputs | Done when |
| --- | --- | --- | --- | --- |
| M1-T1 | Implement `src/data/ingest.py` — `load_trips` | Raw parquet files; config `paths.raw_dir`, `data.months` | DataFrame with columns from ARCH §4.1 | Loads only the months in config; warns on missing files |
| M1-T2 | Implement `clean_trips` | Raw DataFrame; config thresholds | Cleaned DataFrame; cleaning report dict (counts per rule) | All five cleaning rules in ARCH §4.2 applied; report printed |
| M1-T3 | Implement `aggregate_hourly` | Cleaned DataFrame | `hourly_all.parquet` with schema from ARCH §4.2 | Every zone has a row for every hour; missing hours filled with `pickups=0`, `mean_duration_min=NaN` |
| M1-T4 | Write `scripts/ingest.py` CLI | `src/data/ingest.py` | Thin wrapper: `--config` flag, calls the three functions, prints cleaning report, exits non-zero on failure | `python scripts/ingest.py --config config/smoke.yaml` exits 0 |
| M1-T5 | Write `tests/test_ingest.py` | ARCH §12 test spec | Passing pytest tests | See validation gate V1 below |

### Validation gate V1 — after M1

Run `pytest tests/test_ingest.py -v` — all must pass:

- **Clean rules:** each rule drops exactly the rows it should; a synthetic DataFrame with one bad row per rule confirms each count.
- **Completeness:** for a 2-zone, 48-hour synthetic input, `hourly_all` has exactly 96 rows; no hour gap; no zone gap.
- **Zero fill:** a zone with no pickups in an hour has `pickups=0` and `mean_duration_min=NaN`.
- **Duration NaN:** `mean_duration_min` is `NaN` when and only when `pickups=0`.

Manual spot-check after running on real data:

```bash
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/hourly_all.parquet')
print(df.dtypes)
print(df.isnull().sum())
print('Zones:', df.zone_id.nunique(), '| Hours per zone:', df.groupby('zone_id').size().unique())
assert df['pickups'].min() >= 0
"
```

## M2 — Features, windowing, chronological split, leakage tests

**Goal:** Produce all `.npz` window files and supporting artifacts (`splits.json`, `zone_stats.json`, `busy_time.parquet`) with zero data leakage, verified by automated tests.

### Tasks

| ID | Task | Inputs | Outputs | Done when |
| --- | --- | --- | --- | --- |
| M2-T1 | Implement `src/data/zones.py` — `select_zones` | `hourly_all.parquet`; config `zones.*` | `hourly_selected.parquet` (ARCH §4.4 schema) | Zone count = `n_high + n_medium + n_low`; `min_nonzero_hour_fraction` filter applied; IDs and names logged |
| M2-T2 | Implement `src/features/splits.py` — `compute_splits` | `hourly_selected.parquet`; config `split.fractions` | `splits.json` (ARCH §4.3) | Boundaries ISO strings; fractions sum to 1.0; train end < val start < test start |
| M2-T3 | Implement `src/features/scaling.py` — `fit_zone_stats`, `apply_scaling`, `compute_busy_time` | `hourly_selected.parquet`; `splits.json` | `zone_stats.json` (ARCH §4.5); `busy_time.parquet` (ARCH §4.6) | Stats from train period only; `std_log=0` raises; fallback chain in §4.6 implemented |
| M2-T4 | Implement `src/features/windows.py` — `build_windows` | `hourly_selected.parquet`; `zone_stats.json`; `splits.json`; config `windows.*` | `windows_L{L}_{train,val,test}.npz` with keys from ARCH §5 | Shapes match §5; `origin_ts + h` alignment correct; no NaN |
| M2-T5 | Write `scripts/build_features.py` CLI | All above modules | Thin CLI: `--config`, optional `--lookback L`; runs M2-T1 through M2-T4; repeats for each lookback in `experiments.lookbacks` | `python scripts/build_features.py --config config/smoke.yaml` exits 0 |
| M2-T6 | Write `tests/test_splits.py`, `test_scaling.py`, `test_windows.py` | ARCH §12 | Passing pytest suite | See validation gate V2 below |

### Validation gate V2 — after M2

Run `pytest tests/test_splits.py tests/test_scaling.py tests/test_windows.py -v` — all must pass:

**Leakage checks (non-negotiable):**

- Every target hour in every train sample lies strictly inside the train range; same for val and test.
- Target-hour ranges of train, val, test are disjoint and in chronological order.
- `zone_stats` computed on train-only data: modify test-period pickups in a copy of the DataFrame, recompute stats — values must not change.
- `busy_time` uses only train-period rows.

**Shape and alignment checks:**

- `X.shape == (N, L, 6)`, `y_scaled.shape == (N, H)`, `y_raw.shape == (N, H)`, `zone_idx.shape == (N,)`.
- For sample `i`: `origin_ts[i] + timedelta(hours=h)` == the h-th target hour (1-indexed).
- Per-zone sample counts match the formula in ARCH §4.3.
- No NaN anywhere in `X` or `y_scaled`.

**Scaling checks:**

- `std_log = 0` raises `ValueError`.
- Re-scaling a known value round-trips to within float32 tolerance.

Manual spot-check:

```bash
python -c "
import numpy as np, json
d = np.load('data/processed/windows_L168_train.npz', allow_pickle=True)
print({k: d[k].shape for k in d})
assert not np.isnan(d['X']).any()
"
```

## M3 — Baselines and evaluation harness

**Goal:** A fully functional evaluation harness that can produce metrics for any prediction array in the correct format, plus both seasonal baselines — so model results in M4/M5 drop straight into the same pipeline.

### Tasks

| ID | Task | Inputs | Outputs | Done when |
| --- | --- | --- | --- | --- |
| M3-T1 | Implement `src/evaluate/metrics.py` | ARCH §8.2 definitions | `mae`, `rmse`, `wape`, `peak_mae`, `coverage`, `interval_width` functions | All six functions pass `test_metrics.py` hand-computed examples |
| M3-T2 | Implement `src/evaluate/baselines.py` — `lag_baseline` | `.npz` windows; `hourly_selected.parquet` | `b1 (N,H)` lag-24 and `b2 (N,H)` lag-168 arrays | Lag offsets correct; only hours before `origin_ts` used (ARCH §8.1) |
| M3-T3 | Implement `src/evaluate/run_eval.py` — `evaluate_run`, `aggregate_results`, `pick_app_model` | Prediction `.npz`; `zone_stats.json`; `splits.json` | Per-horizon, overall, per-zone metrics JSON (ARCH §8.2 schema); `results.json`; `app_model.json` | Schema matches ARCH §8.2 exactly; `pick_app_model` uses val MAE only |
| M3-T4 | Write `scripts/evaluate.py` CLI | All eval modules | `--experiment E1\|E2\|E3\|all` flag; writes all artifacts from ARCH §8; exits non-zero on missing predictions | `python scripts/evaluate.py --experiment E1` with stub predictions exits 0 |
| M3-T5 | Write `tests/test_baselines.py`, `test_metrics.py` | ARCH §12 | Passing pytest suite | See validation gate V3 below |

### Validation gate V3 — after M3

Run `pytest tests/test_baselines.py tests/test_metrics.py -v`:

**Baseline correctness:**

- B1 for horizon h uses the value at `origin_ts + h - 24`; B2 uses `origin_ts + h - 168`.
- For every sample, both lag hours are strictly before `origin_ts` (no leakage from the future).
- Synthetic hourly table with known values produces exact expected lag values.

**Metrics correctness (hand-computed):**

- `mae([10, 20], [12, 18]) == 2.0`
- `wape` = sum(|error|) / sum(actual), verified against manual calc.
- `peak_mae` filters correctly on the `peak_threshold` per zone.
- `coverage`: fraction inside \[P10, P90\], verified with a 5-sample synthetic example.
- `interval_width`: mean of P90 − P10.

**Baseline sanity check on real data (run once after M2 data is ready):**

```bash
python scripts/evaluate.py --experiment E1  # predictions will be stubs until M4
# after M4: B1 and B2 val MAE should be plausible (roughly 10–40 pickups for busy zones)
```

## M4 — LSTM and GRU training with quantile loss

**Goal:** Both model variants train end-to-end, produce correctly shaped quantile outputs, save checkpoints and history, and pass pre-training sanity checks before a full run.

### Tasks

| ID | Task | Inputs | Outputs | Done when |
| --- | --- | --- | --- | --- |
| M4-T1 | Implement `src/models/losses.py` — `pinball_loss` | ARCH §6 | `pinball_loss(pred, target, quantiles) -> scalar` | Zero for a perfect prediction; asymmetry correct; hand-computed example matches (see V4) |
| M4-T2 | Implement `src/models/rnn.py` — `QuantileRNN` | ARCH §6 | LSTM and GRU variants; `forward` and `predict_quantiles` | `forward` output `(B,H,Q)`; `predict_quantiles` sorted; LSTM and GRU differ only in cell |
| M4-T3 | Implement `src/train/trainer.py` — `train_model` | ARCH §7 | Trains with Adam, early stopping, saves best checkpoint to `artifacts/{tag}/` | `model.pt`, `history.json`, `train_summary.json` written; checkpoint is the best-val-loss epoch |
| M4-T4 | Write `scripts/train.py` CLI | `src/train/trainer.py` | `--model {lstm,gru}`, `--seed S`, optional `--lookback L`; reads config; exits non-zero on error | `python scripts/train.py --model lstm --seed 0` exits 0 and writes artifacts |
| M4-T5 | Write `tests/test_models.py`, `test_losses.py` | ARCH §12 | Passing pytest suite | See validation gate V4 below |

### Validation gate V4 — before and after full training

**Pre-training checks (run before committing to a full multi-seed run):**

```python
# 1. Overfit one batch — loss must decrease to near-zero within 200 steps
# Run manually with a single batch of ~32 samples and lr=0.01
# If loss does not fall below 0.05 in 200 steps, architecture or loss has a bug.

# 2. Quantile ordering
from src.models.rnn import QuantileRNN
import torch
model = QuantileRNN('lstm', n_features=6, n_zones=18, emb_dim=8,
                    hidden_size=64, num_layers=2, dropout=0.0, horizon=6, n_quantiles=3)
x = torch.randn(4, 168, 6); z = torch.randint(0, 18, (4,))
out = model.predict_quantiles(x, z)  # (4, 6, 3)
assert (out[..., 1] >= out[..., 0]).all()  # P50 >= P10
assert (out[..., 2] >= out[..., 1]).all()  # P90 >= P50
```

**Post-training checks (run `pytest tests/test_models.py tests/test_losses.py -v`):**

- `forward` output shape `(B, H, Q)` for both cell types.
- `predict_quantiles` is sorted along last dim.
- LSTM and GRU have different parameter counts; both are reported in `train_summary.json`.
- `pinball_loss` is zero for a perfect prediction.
- Asymmetry: over-prediction penalised more at q=0.1; under-prediction penalised more at q=0.9.
- Hand-computed example: for q=0.5 the pinball loss equals MAE/2.

**After training all 6 runs (lstm×3 seeds, gru×3 seeds):**

- All 6 `artifacts/*/model.pt` files exist.
- Val loss in `history.json` decreases over the first 5 epochs (sanity; not a hard requirement).
- No run exits with a NaN loss (check `train_summary.json` `best_val_loss` is finite).

## M5 — Experiments E1–E3 and result tables

**Goal:** Run all experiments, aggregate across seeds, produce `results.json` and the CSV table, and confirm that the best model is selected on val MAE (never test MAE).

### Tasks

| ID | Task | Inputs | Outputs | Done when |
| --- | --- | --- | --- | --- |
| M5-T1 | Run E1 — LSTM vs GRU, L=168, 3 seeds each | `windows_L168_*.npz`; config | 6 trained runs; `pred_{val,test}.npz` for each; baseline predictions | All 6 `artifacts/*/model.pt` exist; baselines written |
| M5-T2 | Run E2 — lookback 24 vs 168 for the better model type | `windows_L24_*.npz` and `windows_L168_*.npz` | 6 more trained runs (3 seeds × 2 lookbacks); predictions | Requires `build_features.py --lookback 24` to have been run |
| M5-T3 | Run E3 (optional) — with/without time features | Modify `build_features.py` to zero out time features | 6 more runs; clearly labelled artifacts | E3 tag = `{model}_L168_notime_s{seed}` |
| M5-T4 | Run `evaluate.py --experiment all` | All predictions | `results.json`; `metrics/*.json`; `app_model.json`; E1 CSV table | Schema matches ARCH §8.2; `app_model.json` selected by val MAE |
| M5-T5 | Verify `app_model.json` selection | `results.json`; all `metrics/*.json` | Confirmed tag | See validation gate V5 |

### Validation gate V5 — after M5

**Model selection integrity (critical):**

- `app_model.json` tag matches the run with the lowest overall MAE on the **validation** set.
- Confirm manually: sort `metrics/*_val.json` by `overall.mae`; the minimum matches `app_model.json`.
- The test set was never used to choose the model.

**Result sanity checks:**

```bash
python -c "
import json
r = json.load(open('artifacts/results.json'))
# E1 must have lstm and gru entries
assert 'lstm' in r['E1'] and 'gru' in r['E1']
# Coverage should be between 0.7 and 0.9 for well-trained models
for model in ['lstm', 'gru']:
    cov = r['E1'][model]['test']['coverage']['mean']
    print(f'{model} test coverage: {cov:.3f}')
    assert 0.5 < cov < 1.0, f'Coverage out of plausible range: {cov}'
"
```

**Cross-check baselines vs models (expectation, not a hard gate):**

- B2 (weekly lag) is expected to be stronger than B1 (daily lag) on most metrics — if not, check the lag offset implementation.
- If both models underperform B2, this is a valid result; it must be stated in the report, not hidden.

## M6 — Fleet module and Streamlit dashboard

**Goal:** A working three-page app that loads saved artifacts, shows forecasts with uncertainty bands, and lets a dispatcher check fleet gap — with no training code and a clear "test-period data only" label.

### Tasks

| ID | Task | Inputs | Outputs | Done when |
| --- | --- | --- | --- | --- |
| M6-T1 | Implement `src/fleet/fleet.py` — `required_taxis`, `fleet_table` | ARCH §9 | Both functions with correct column schema | PRD example (D=48,T=22,U=0.7 → 26; D=61 → 32); invalid inputs raise `ValueError`; status rules correct |
| M6-T2 | Write `tests/test_fleet.py` | ARCH §12 | Passing pytest | See V6 below |
| M6-T3 | Implement `app/loaders.py` | ARCH §10 | All loaders with `@st.cache_data`; clear error message (not traceback) if file missing | Every loader tested with a missing file returns a user-facing string, not an exception |
| M6-T4 | Build `app/pages/1_Forecast.py` | ARCH §10 | Zone dropdown, origin-time selector, 48-hr actuals chart, 6-hr forecast band, fleet table, T/U inputs | Renders from saved test-period predictions; "no live data" notice visible |
| M6-T5 | Build `app/pages/2_Zone_Comparison.py` | ARCH §10 | Multi-zone select, P50 and fleet requirement for next 6 hours | Side-by-side comparison across selected zones |
| M6-T6 | Build `app/pages/3_Model_Evaluation.py` | ARCH §10 | `results.json` tables, error-by-horizon chart, predicted vs actual overlay, coverage stat | All charts use Plotly |
| M6-T7 | Wire `app/main.py` | All pages | Page routing; config and artifact path from `load_config` | `streamlit run app/main.py` starts without errors |

### Validation gate V6 — after M6

**Fleet unit tests (`pytest tests/test_fleet.py -v`):**

- `required_taxis(48, 22, 0.7) == 26` and `required_taxis(61, 22, 0.7) == 32` (PRD §9 examples).
- `pickups < 0` raises `ValueError`.
- `busy_min <= 0` raises `ValueError`.
- `utilization = 0` raises `ValueError`; `utilization = 1.0` is valid.
- Status: `gap_expected > 0` → `"shortage"`; `gap_safe > 0 >= gap_expected` → `"risk"`; else `"ok"`.

**Dashboard smoke test:**

```bash
# Start the app and check for errors in the terminal
streamlit run app/main.py &
sleep 5
# Visit http://localhost:8501 — all three pages must load without traceback
# Remove a required artifact and confirm the error message names the script to run
```

**App contract checks:**

- The app displays "no live data" / "test-period only" copy on the Forecast page.
- Removing `artifacts/app_model.json` shows a clear error message (not a Python traceback).
- The fleet table contains all columns from ARCH §9: `horizon, hour_ts, p10, p50, p90, busy_min, n_expected, n_safe, available, gap_expected, gap_safe, status`.

## M7 — Report, documentation, and final validation

**Goal:** The full pipeline runs cleanly end-to-end from config; the report contains honest results for all experiments; every success criterion from PRD §13 is met or explicitly noted.

### Tasks

| ID | Task | Inputs | Outputs | Done when |
| --- | --- | --- | --- | --- |
| M7-T1 | Full end-to-end smoke run | `config/smoke.yaml` | All stages complete; all artifacts written | `bash run_all.sh --config config/smoke.yaml` exits 0 |
| M7-T2 | Full end-to-end default run | `config/default.yaml` | Production artifacts in `artifacts/` | Same script exits 0 with 6-month data |
| M7-T3 | Run complete test suite | All test files | All tests pass | `pytest tests/ -v --tb=short` exits 0 with 0 failures |
| M7-T4 | Write lab report sections | `results.json`; metrics CSVs | Report draft | Covers: methodology, E1/E2/(E3) results, honest conclusion if baselines win, limitations |
| M7-T5 | Final README and docstrings | All source files | Every public function has a one-line docstring; README reflects final commands | A new reader can run the pipeline from the README alone |

### Validation gate V7 — end-to-end and success criteria

**PRD §13 success criteria checklist:**

| Criterion | Check |
| --- | --- |
| SC-1: Pipeline runs end-to-end | `bash run_all.sh --config config/default.yaml` exits 0 |
| SC-2: No data leakage | `pytest tests/test_splits.py tests/test_scaling.py` — 0 failures |
| SC-3: Both models produce 6-hour P10/P50/P90 | `pred_test.npz` for all 6 runs has `q_pred` shape `(N,6,3)` |
| SC-4: Results reported as mean ± std, 3 seeds | `results.json` E1 entries have `mean` and `std` for all metrics |
| SC-5: Coverage 70–90% | `results.json` E1 `coverage.mean` in \[0.70, 0.90\] for at least one model |
| SC-6: Dashboard runs from saved artifacts | `streamlit run app/main.py` serves all 3 pages without error |
| SC-7: Report states outcome honestly | Report section explicitly names the better model or states baselines win |

**Final full-suite test run:**

```bash
pytest tests/ -v --tb=short 2>&1 | tail -20
# Expected: all green, 0 errors
```

**Coverage check (SC-5):**

```bash
python -c "
import json
r = json.load(open('artifacts/results.json'))
for model in ['lstm', 'gru']:
    cov = r['E1'][model]['test']['coverage']['mean']
    print(f'{model} coverage: {cov:.3f} — {\"PASS\" if 0.70 <= cov <= 0.90 else \"REVIEW\"}')
"
```
