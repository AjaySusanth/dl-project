# AGENTS.md: CrowdCast

Standing instructions for any coding agent working in this repository. Read this file fully before every session.

## 1. Project in one paragraph

CrowdCast forecasts NYC taxi pickups for 15 to 20 zones, 1 to 6 hours ahead, with P10/P50/P90 quantile outputs. One shared LSTM and one shared GRU (with a zone embedding) are trained under an identical protocol and compared against two seasonal baselines. A Streamlit dashboard turns forecasts into fleet requirements and a fleet gap. The deep learning experiment is the core; the dashboard is secondary.

## 2. Source of truth

If documents disagree, the higher one wins. If a referenced document does not exist yet, ask; never invent its contents.

1. `docs/PRD.md`: what to build and why.
2. `docs/ARCHITECTURE.md`: schemas, shapes, config keys, module interfaces, file names.
3. `docs/TASK_PLAN.md`: the ordered tasks (M0 to M7) and their definitions of done.
4. `docs/VALIDATION_GATES.md`: checks that must pass before moving on.
5. `docs/templates/TASK_REPORT.md`: the report written after every task.
6. `docs/EXPERIMENT_LOG.md`: where experiment results are recorded.

## 3. Environment

- Python 3.11, a virtual environment in `.venv`, dependencies pinned in `requirements.txt`.
- Core libraries: pandas, numpy, pyarrow, torch, plotly, streamlit, pyyaml, pytest.
- Training must run on CPU. Use a GPU only if available, and never require one.
- Do not add a dependency without approval. If one is needed, stop and ask.

## 4. Commands

Exact names are defined in `docs/ARCHITECTURE.md`. The intended shape is:

```
python scripts/ingest.py --config config/default.yaml
python scripts/build_features.py --config config/default.yaml
python scripts/train.py --config config/default.yaml --model lstm --seed 0
python scripts/evaluate.py --config config/default.yaml
streamlit run app/main.py
pytest -q
```

Every stage is runnable on its own, reads its inputs from disk, and writes its outputs to disk.

## 5. Repository layout

```
config/       YAML config (single source of every hyperparameter and path)
data/raw/     downloaded TLC files, read-only, never committed
data/processed/  generated, never committed
src/          data, features, models, train, evaluate, fleet
scripts/      one thin command-line entry point per stage
app/          Streamlit app (loads saved artifacts only, never trains)
artifacts/    checkpoints and metrics JSON, never committed unless small
tests/        pytest tests
docs/         the documents listed in section 2
```

## 6. How to work: one task at a time

1. Pick the next task in `docs/TASK_PLAN.md`. Do not start a later task early.
2. Restate the task in two or three lines and list the files you expect to touch.
3. Implement only that task. No refactors, renames or features outside it.
4. Run the validation named in the task and in `docs/VALIDATION_GATES.md`. Paste the real output.
5. Write the task report (section 9) and commit.
6. At the end of every milestone, stop and wait for the user's approval before starting the next one.

## 7. Deep learning rules that are never broken

- **No leakage.** Split chronologically into train, validation and test. Fit scalers and any statistics (zone means, busy times, peak thresholds) on the train period only.
- **The test set is untouchable** for tuning, early stopping, model selection or threshold choice. Early stopping uses the validation set only.
- **Fair comparison.** LSTM and GRU use identical data, features, hyperparameters, seeds and training loop. Only the recurrent cell differs.
- **Seeds.** Seed Python, NumPy and PyTorch everywhere, and record the seed with every result.
- **Quantiles.** Pinball loss over q = 0.1, 0.5, 0.9 and 6 horizons. At inference, sort quantiles so P10 <= P50 <= P90.
- **Transforms.** Inverse-transform predictions (expm1), clip at zero, and only then compute metrics.
- **Baselines.** Same hour yesterday (lag 24) and same hour last week (lag 168), computed on the same test windows as the models.
- **Honest reporting.** Report real numbers as produced. Never fabricate, round favourably, cherry-pick seeds, or hide a result where a baseline wins.

## 8. Code standards

- Type hints on all functions. Docstrings state tensor shapes where tensors appear.
- No magic numbers: every hyperparameter, path and threshold comes from the config.
- Small, single-purpose functions in `src/`. Scripts only parse arguments and call `src/`.
- No logic that lives only in a notebook.
- Use `logging`, not stray `print`. Fail loudly with clear error messages on bad data.
- Every module that transforms data or computes a metric has at least one pytest test, including the edge cases (missing hours, empty zones, zero counts).
- Never delete, skip or weaken a test to make it pass. Fix the cause.

## 9. Explanation after every task

After each task, write a report using `docs/templates/TASK_REPORT.md` and show a short version in chat:

- **What I did and why**, in plain language a student can follow.
- **Files changed.**
- **Validation run**, with the actual command and its output.
- **Result**, including numbers where relevant.
- **Deviations** from the PRD, architecture or plan, and the reason.
- **Open questions or risks.**
- **Next task.**

The agent MUST follow `LEARNING_PROTOCOL.md` before and after
every task. A task is not complete until its implementation,
validation, repository report, and student-facing explanation
are complete.

## 10. Stop and ask when

- The spec is ambiguous or two documents conflict.
- A validation gate fails twice after a fix attempt.
- A new dependency, a new zone choice, or a change to the split or metrics seems necessary.
- Anything would deviate from the PRD, or touch the out-of-scope list.

## 11. Out of scope

Live GPS or dispatch, automatic taxi repositioning, real-time ride-request ingestion, weather or event data, a separate backend or database, Docker or cloud deployment. Do not add these, even as stubs.

## 12. Git

- One branch for the project, small commits, one per task.
- Commit message format: `M2-T3: add train-only scaler and leakage test`.
- Never commit raw data, processed data or large artifacts.

## 13. Definition of done (every task)

- The task's acceptance criteria are met.
- All tests pass (`pytest -q`) and the relevant validation gate passes.
- The task report is written and committed.
- No unrelated files were changed.
