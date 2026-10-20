# CrowdCast Validation Gates

These gates define the evidence required before a task or milestone is
accepted. They supplement `AGENTS.md`, the PRD, `ARCHITECTURE.md`, and
the M0--M7 task plan; they do not replace them.

## How to use this document

-   Run the gate at the milestone specified below and record the actual
    commands, exit codes, and relevant output in the task report.
-   A gate is **PASS** only when its required checks pass with real
    evidence. Use **FAIL** when a required check fails, **BLOCKED** when
    it cannot be run because a prerequisite is missing, and **REVIEW**
    for a non-blocking diagnostic or expectation.
-   Never fabricate output or mark a check as passed because the code
    looks correct.
-   If a check cannot be run, state why, what remains unverified, and
    what is needed to run it.
-   Do not proceed to the next milestone until the current milestone
    gate passes and the user approves the milestone, as required by
    `AGENTS.md`.
-   Synthetic fixtures are required for deterministic edge-case tests.
    Real-data spot checks supplement, but do not replace, unit tests.
-   Use the exact schemas, interfaces, config keys, paths, shapes, and
    task definitions in `ARCHITECTURE.md` and the task plan. If a
    specification conflicts or is ambiguous, stop and ask rather than
    silently choosing.

## Gate summary

  -----------------------------------------------------------------------
  Gate                    When                    Main evidence
  ----------------------- ----------------------- -----------------------
  V0                      After M0                Repository layout,
                                                  config loading, seed
                                                  setup, dependency
                                                  install, smoke
                                                  configuration

  V1                      After M1                Cleaning rules,
                                                  complete hourly zone
                                                  grid, zero fill, output
                                                  schema

  V2                      After M2                Chronological split,
                                                  train-only statistics,
                                                  window
                                                  shapes/alignment, no
                                                  leakage

  V3                      After M3                Correct lag baselines
                                                  and hand-verified
                                                  metrics

  V4                      Before and after M4     Pinball loss, model
                          training                output shape/order,
                                                  overfit sanity check,
                                                  six reproducible runs

  V5                      After M5                Experiment result
                                                  schema, validation-only
                                                  model selection, honest
                                                  metrics

  V6                      After M6                Fleet calculation
                                                  tests, dashboard page
                                                  smoke test, app
                                                  contract

  V7                      After M7                End-to-end pipeline,
                                                  full tests, PRD success
                                                  criteria and final
                                                  report
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## V0 --- Repository setup, config, and environment (M0)

### Required checks

1.  Repository layout matches `ARCHITECTURE.md` §2.
2.  `.gitignore` excludes `data/`, `artifacts/`, and `__pycache__`;
    raw/processed data and large checkpoints are not committed.
3.  `config/default.yaml` contains every required key defined in
    `ARCHITECTURE.md` §3.
4.  `config/smoke.yaml` overrides the full-run configuration for one
    month, six zones (two per volume tier), two epochs, and one seed.
5.  `load_config` loads and merges configuration correctly.
6.  `set_seed` seeds Python, NumPy, and PyTorch.
7.  Pinned dependencies install successfully in a clean Python 3.11
    environment.
8.  README lists the stage commands in execution order.

### Commands from the task plan

``` bash
python -c "from src.config import load_config, set_seed; cfg = load_config('config/default.yaml'); assert 'paths' in cfg"
python -c "from src.config import set_seed; set_seed(0)"
```

Also run the clean-environment dependency installation check:

``` bash
python -m pip install -r requirements.txt
```

### Acceptance

-   Both config commands exit with status 0.
-   The default config contains all architecture-defined keys.
-   Smoke config values match the specified reduced run.
-   Dependency installation exits 0 in the intended environment.
-   Directory structure and ignore rules match the architecture.
-   Do not claim the full pipeline works at M0; only setup and
    smoke-test prerequisites are being validated.

------------------------------------------------------------------------

## V1 --- Data ingestion, cleaning, and hourly aggregation (M1)

### Automated tests

Run:

``` bash
pytest tests/test_ingest.py -v
```

All checks must pass:

-   Each cleaning rule drops exactly the expected rows in a synthetic
    DataFrame with one bad row per rule.
-   A synthetic input with two zones and 48 hours produces exactly 96
    rows.
-   The hourly grid contains no missing hour for any included zone.
-   An hour with no pickups has `pickups = 0` and
    `mean_duration_min = NaN`.
-   `mean_duration_min` is NaN if and only if `pickups = 0`, according
    to the task plan's stated contract.
-   The output columns and dtypes match `ARCHITECTURE.md` §4.2.
-   The ingestion CLI returns a non-zero exit code on failure and warns
    about missing configured monthly files.

### Real-data spot check

After the configured raw files are available and ingestion has run:

``` bash
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/hourly_all.parquet')
print(df.dtypes)
print(df.isnull().sum())
print('Zones:', df.zone_id.nunique(), '| Hours per zone:', df.groupby('zone_id').size().unique())
assert df['pickups'].min() >= 0
"
```

Record the actual output, number of rows, zones, date range, and
cleaning counts. Do not invent or hardcode expected real-data counts
before inspecting the downloaded files.

### Acceptance

-   `pytest tests/test_ingest.py -v` passes.
-   `python scripts/ingest.py --config config/smoke.yaml` exits 0 when
    smoke data prerequisites are present.
-   The output Parquet schema matches the architecture and the real-data
    spot check is documented.
-   Missing source files are surfaced clearly; they must not silently
    produce a misleadingly complete dataset.

------------------------------------------------------------------------

## V2 --- Features, chronological split, and leakage prevention (M2)

Run:

``` bash
pytest tests/test_splits.py tests/test_scaling.py tests/test_windows.py -v
```

### Leakage checks --- non-negotiable

-   Every target hour for a train sample lies strictly inside the train
    range; likewise for validation and test.
-   Target-hour ranges for train, validation, and test are disjoint and
    chronologically ordered.
-   Changing test-period pickup values and recomputing statistics does
    not change train-fitted `zone_stats`.
-   `busy_time` is computed from training-period rows only.
-   Scalers and all derived statistics are fitted using training data
    only.
-   The test set is not used for tuning, early stopping, model
    selection, or threshold selection.

### Shape and alignment checks

-   `X.shape == (N, L, 6)`.
-   `y_scaled.shape == (N, H)`.
-   `y_raw.shape == (N, H)`.
-   `zone_idx.shape == (N,)`.
-   For sample `i`, `origin_ts[i] + timedelta(hours=h)` aligns with the
    h-th target hour, where `h` is 1-indexed.
-   Per-zone sample counts match the formula in `ARCHITECTURE.md` §4.3.
-   No NaN occurs in `X` or `y_scaled`.
-   Split fractions sum to 1.0 and saved boundaries are valid ISO
    strings.

### Scaling checks

-   `std_log = 0` raises `ValueError`.
-   A known value round-trips through scaling and inverse scaling within
    float32 tolerance.
-   The documented fallback chain for busy-time statistics is
    implemented as specified in `ARCHITECTURE.md` §4.6.

### Manual artifact check

``` bash
python -c "
import numpy as np
d = np.load('data/processed/windows_L168_train.npz', allow_pickle=True)
print({k: d[k].shape for k in d})
assert not np.isnan(d['X']).any()
assert not np.isnan(d['y_scaled']).any()
"
```

### Acceptance

All automated checks pass, shapes and timestamp alignment match the
architecture, and train-only statistic behavior is demonstrated by a
test. Any leakage failure blocks progression.

------------------------------------------------------------------------

## V3 --- Baselines and evaluation metrics (M3)

Run:

``` bash
pytest tests/test_baselines.py tests/test_metrics.py -v
```

### Baseline correctness

-   B1 at horizon `h` uses the observation at
    `origin_ts + h - 24 hours`.
-   B2 at horizon `h` uses the observation at
    `origin_ts + h - 168 hours`.
-   Both lag timestamps are strictly earlier than `origin_ts`; no future
    observation is used.
-   A synthetic hourly table with known values yields exact expected lag
    predictions.
-   Baseline predictions align with the same windows and targets used
    for model evaluation.

### Metric correctness

-   `mae([10, 20], [12, 18]) == 2.0`.
-   WAPE equals `sum(abs(error)) / sum(actual)` for a hand-calculated
    example; zero-denominator behavior must be explicit and tested.
-   `peak_mae` filters observations using the relevant zone's
    training-derived `peak_threshold`.
-   Coverage equals the fraction of actual values within `[P10, P90]` in
    a five-sample synthetic example.
-   Interval width equals the mean of `P90 - P10`.
-   RMSE matches a hand-computed example.
-   Metric output schemas match `ARCHITECTURE.md` §8.2.
-   App model selection uses validation MAE only, never test MAE.

### Diagnostic

The task plan suggests checking real-data baseline MAE for plausibility
(roughly 10--40 pickups for busy zones). This is a **REVIEW
diagnostic**, not a universal pass/fail threshold. Demand varies by zone
and period; investigate unexpected values rather than altering results
to fit this range.

### Acceptance

All unit tests pass; hand calculations and synthetic lag examples agree
exactly within defined numeric tolerance; missing prediction inputs
cause a clear non-zero failure rather than silently generating
misleading metrics.

------------------------------------------------------------------------

## V4 --- Quantile loss, LSTM/GRU model, and training (M4)

### Pre-training sanity checks

Before the full multi-seed run, perform the task-plan one-batch overfit
test:

-   Train on a single batch of about 32 samples with learning rate
    `0.01`.
-   Loss should fall below `0.05` within 200 steps, as specified in the
    task plan.
-   If it does not, inspect the model, target scaling, loss, optimizer,
    and training loop before starting the full run.

Check quantile ordering using the architecture-defined model interface:

``` python
from src.models.rnn import QuantileRNN
import torch

model = QuantileRNN(
    'lstm', n_features=6, n_zones=18, emb_dim=8,
    hidden_size=64, num_layers=2, dropout=0.0,
    horizon=6, n_quantiles=3
)
x = torch.randn(4, 168, 6)
z = torch.randint(0, 18, (4,))
out = model.predict_quantiles(x, z)
assert out.shape == (4, 6, 3)
assert (out[..., 1] >= out[..., 0]).all()
assert (out[..., 2] >= out[..., 1]).all()
```

Use the actual constructor contract if it is updated by an approved
change; do not silently diverge from `ARCHITECTURE.md`.

### Automated checks

Run:

``` bash
pytest tests/test_models.py tests/test_losses.py -v
```

Required checks:

-   Both LSTM and GRU `forward` outputs have shape `(B, H, Q)`.
-   `predict_quantiles` is sorted along the quantile dimension: P10 ≤
    P50 ≤ P90.
-   LSTM and GRU differ only in recurrent cell type under the shared
    configuration.
-   Parameter counts differ and are recorded in `train_summary.json`.
-   Pinball loss is zero for a perfect prediction.
-   Pinball-loss asymmetry is correct: over-prediction is penalized more
    at q=0.1, under-prediction more at q=0.9.
-   For q=0.5, the pinball loss matches MAE/2 for the hand-computed
    example.
-   Loss computation supports the six horizons and quantiles
    `[0.1, 0.5, 0.9]`.

### After all E1 runs

-   All six model checkpoints exist: LSTM × three seeds and GRU × three
    seeds.
-   Each run writes `model.pt`, `history.json`, and
    `train_summary.json`.
-   `best_val_loss` is finite; no run exits with NaN loss.
-   The selected checkpoint is the best validation-loss epoch, not the
    final epoch by default.
-   Record actual training duration, seed, parameter count, best epoch,
    and validation loss.
-   The task plan says validation loss decreasing over the first five
    epochs is a sanity check, not a hard requirement.

### Acceptance

Unit tests and pre-training checks pass; all six E1 runs finish and save
required artifacts with finite losses. If the one-batch test fails, do
not proceed to full training until investigated.

------------------------------------------------------------------------

## V5 --- Experiments, aggregation, and model selection (M5)

### Model selection integrity --- critical

-   `artifacts/app_model.json` identifies the run with the lowest
    overall MAE on the **validation** set.
-   Manually sort `artifacts/metrics/*_val.json` by `overall.mae` and
    confirm the selected tag is the minimum.
-   Confirm test metrics were not used to select a model, tune
    hyperparameters, choose thresholds, or choose the lookback.
-   E1 includes both LSTM and GRU with three seeds each and the
    configured 168-hour lookback.
-   E2 compares 24-hour and 168-hour lookbacks for the better model
    type, according to the task plan.
-   E3 is optional and must be labelled clearly if run.
-   Aggregate seed results as mean ± standard deviation; preserve
    individual-seed results.
-   Results and metric schemas match `ARCHITECTURE.md` §8.2.

### Result sanity check

``` bash
python -c "
import json
r = json.load(open('artifacts/results.json'))
assert 'lstm' in r['E1'] and 'gru' in r['E1']
for model in ['lstm', 'gru']:
    cov = r['E1'][model]['test']['coverage']['mean']
    print(f'{model} test coverage: {cov:.3f}')
    assert 0.5 < cov < 1.0, f'Coverage out of plausible range: {cov}'
"
```

The `0.5 < coverage < 1.0` assertion is the task plan's broad **sanity
check**, not the final project success criterion. V7 separately checks
the PRD target range.

### Baseline comparison

-   Compare LSTM, GRU, daily-lag B1, and weekly-lag B2 on the same
    held-out test windows.
-   B2 is expected to outperform B1 on most metrics; this is a
    diagnostic expectation, not a hard gate.
-   If both neural models underperform the strongest baseline, report
    that honestly and investigate possible causes without hiding or
    cherry-picking results.
-   The target that at least one model beats the best baseline on +1h
    MAE is not a pass/fail condition in the PRD.

### Acceptance

Validation-only selection is verified; experiment outputs and schemas
are complete; seed-level and aggregated results are preserved; results
are reported honestly, including if a baseline wins.

------------------------------------------------------------------------

## V6 --- Fleet module and Streamlit dashboard (M6)

### Fleet unit tests

Run:

``` bash
pytest tests/test_fleet.py -v
```

Required checks:

-   `required_taxis(48, 22, 0.7) == 26`.
-   `required_taxis(61, 22, 0.7) == 32`.
-   Negative pickups raise `ValueError`.
-   `busy_min <= 0` raises `ValueError`.
-   `utilization = 0` raises `ValueError`; `utilization = 1.0` is valid.
-   Status logic:
    -   `gap_expected > 0` → `"shortage"`.
    -   `gap_safe > 0 >= gap_expected` → `"risk"`.
    -   Otherwise → `"ok"`.
-   Fleet output includes all architecture-defined columns:
    `horizon, hour_ts, p10, p50, p90, busy_min, n_expected, n_safe, available, gap_expected, gap_safe, status`.

### Dashboard smoke test

Start the app:

``` bash
streamlit run app/main.py
```

Confirm all three pages load without traceback:

1.  Forecast.
2.  Zone Comparison.
3.  Model Evaluation.

Also verify: - Forecast page clearly states there is **no live data**
and the selectable time is from the held-out test period. - Removing or
renaming `artifacts/app_model.json` results in a clear user-facing
message naming the appropriate command/script to generate the artifact,
not a raw Python traceback. - The app reads saved artifacts and does not
train models at app startup. - All charts use Plotly. - The app supports
selecting zones and test-period timestamps as specified in the PRD.

The task plan suggests a terminal smoke check using
`streamlit run app/main.py &` and `sleep 5`; if using it, also check the
process output and actually load the pages in a browser. A process
starting is not, by itself, proof that every page works.

### Acceptance

Fleet tests pass, all three pages are verified, required columns and
explanatory copy are present, and missing-artifact handling is
user-friendly.

------------------------------------------------------------------------

## V7 --- End-to-end pipeline and final project acceptance (M7)

### Pipeline and test suite

Run the smoke pipeline:

``` bash
bash run_all.sh --config config/smoke.yaml
```

Run the configured full pipeline when the six-month raw dataset is
present:

``` bash
bash run_all.sh --config config/default.yaml
```

Run the complete test suite:

``` bash
pytest tests/ -v --tb=short
```

The task plan also suggests:

``` bash
pytest tests/ -v --tb=short 2>&1 | tail -20
```

Record the full command, exit status, test count, failures, and relevant
output. Do not use `tail` as the only retained evidence if it hides
details needed to understand a failure.

### PRD success criteria checklist

  -----------------------------------------------------------------------------------------------------
  ID                      Criterion               Required evidence
  ----------------------- ----------------------- -----------------------------------------------------
  SC-1                    Full pipeline runs      `bash run_all.sh --config config/default.yaml` exits
                          end-to-end              0

  SC-2                    No data leakage         `pytest tests/test_splits.py tests/test_scaling.py`
                                                  passes; chronological and train-only checks are
                                                  present

  SC-3                    Both models produce     `q_pred` in test prediction files has shape
                          six-hour P10/P50/P90    `(N, 6, 3)` for all six E1 runs
                          forecasts               

  SC-4                    Results are reported    E1 `results.json` entries include mean and standard
                          over three seeds        deviation for metrics

  SC-5                    P10--P90 interval       At least one model's E1 test `coverage.mean` is
                          coverage target         within `[0.70, 0.90]`, as the PRD specifies

  SC-6                    Dashboard runs from     All three Streamlit pages load without error
                          saved artifacts         

  SC-7                    Honest conclusion       Report identifies the better model or states that the
                                                  baselines win, and documents limitations
  -----------------------------------------------------------------------------------------------------

For SC-5, if neither model meets the target, do not alter or hide the
result. Mark the criterion as unmet, investigate, and explain the
finding. The PRD requires honest reporting; it does not justify
manipulating results to force a pass.

### Final coverage check

``` bash
python -c "
import json
r = json.load(open('artifacts/results.json'))
for model in ['lstm', 'gru']:
    cov = r['E1'][model]['test']['coverage']['mean']
    status = 'PASS' if 0.70 <= cov <= 0.90 else 'REVIEW'
    print(f'{model} coverage: {cov:.3f} — {status}')
"
```

### Final review

-   Pipeline works from the documented commands.
-   Full tests pass, or every failure is explicitly reported and
    resolved/accepted by the user.
-   Artifacts and results are traceable to configuration and seed.
-   No raw data, processed data, or large model artifacts are committed.
-   README lets a new reader reproduce the pipeline.
-   Lab report covers methodology, E1/E2/(E3) results, limitations, and
    honest comparison with baselines.
-   All PRD success criteria are marked **PASS**, **NOT MET**, or **NOT
    APPLICABLE** with evidence and rationale.
-   The task report is complete and the user approves the milestone.

------------------------------------------------------------------------

## Failure escalation and stop conditions

Follow `AGENTS.md` §10:

-   If the specification is ambiguous or documents conflict, stop and
    ask.
-   If a validation gate fails twice after a fix attempt, stop and ask
    rather than continuing indefinitely.
-   Ask before adding a dependency, changing zone selection, changing
    split fractions or metrics, or deviating from the PRD.
-   Never delete, skip, weaken, or silently xfail a test to make the
    gate pass.
-   Never continue to the next milestone before the current gate is
    accepted and the user approves.
