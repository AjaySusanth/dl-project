# CrowdCast Task Report Template

> Copy this template to `docs/reports/<TASK-ID>.md` (or the report
> location approved for this repository) after each task. Replace every
> bracketed placeholder. Do not leave unsupported claims or invented
> validation output.

## 1. Task identification

-   **Task ID:** \[e.g., M0-T1\]
-   **Task title:** \[short title\]
-   **Milestone:** \[M0--M7\]
-   **Date:** \[YYYY-MM-DD\]
-   **Branch:** \[branch name\]
-   **Commit:** \[commit hash, after commit\]
-   **Status:** \[PASS / FAIL / BLOCKED / REVIEW\]

## 2. Pre-coding briefing

### What I planned to do

\[Restate the task in 2--3 plain-language sentences.\]

### Why this task is needed

\[Explain the problem it solves and how it fits into the architecture.\]

### Expected files

-   `[path]` --- \[expected change and reason\]

### Acceptance criteria

-   [ ] \[Criterion taken from the task plan or architecture\]
-   [ ] \[Criterion taken from the task plan or architecture\]

### Risks or questions identified before coding

\[State none if there were none. If a blocking ambiguity exists, do not
implement until clarified.\]

## 3. What I implemented

\[Explain what changed and why, in language a student can follow.
Distinguish actual implementation from planned work.\]

### File-by-file explanation

#### `[path/to/file]`

-   **Responsibility:** \[What this file owns.\]
-   **Changes:** \[What was added or modified.\]
-   **How it works:** \[Explain the important functions, inputs,
    outputs, and control flow.\]
-   **Why this design:** \[Explain the rationale and
    alternatives/trade-offs where relevant.\]
-   **Connections:** \[What calls this file or what it calls.\]

\[Repeat for each changed file. List deleted or renamed files
explicitly; do not omit them.\]

## 4. Architecture and data flow

\[Explain how this task's code fits into the overall CrowdCast pipeline.
Trace the relevant inputs → transformations → outputs. If tensors are
involved, state their shapes and meanings.\]

## 5. Key technical decisions

  -----------------------------------------------------------------------
  Decision          Reason            Alternative       Trade-off /
                                      considered        consequence
  ----------------- ----------------- ----------------- -----------------
  \[Decision\]      \[Reason\]        \[Alternative or  \[Trade-off\]
                                      "not              
                                      considered"\]     

  -----------------------------------------------------------------------

\[Do not invent alternatives that were not actually considered; label
your own post-hoc analysis as such.\]

## 6. Deep learning / data-science explanation (when applicable)

-   **Concepts used:** \[Define the relevant concepts simply.\]
-   **Inputs and targets:** \[Data columns, features, labels/targets, or
    tensor shapes.\]
-   **Math or loss function:** \[Explain relevant equations and variable
    meanings.\]
-   **Training/evaluation implications:** \[Explain how this affects
    learning or measured performance.\]
-   **Leakage and reproducibility safeguards:** \[Explain what prevents
    leakage and how seeds/config are handled.\]

For non-ML tasks, state "Not applicable to this task" and explain the
relevant software-engineering concepts instead.

## 7. Validation evidence

### Commands executed

  ----------------------------------------------------------------------------------
  Command                         Exit code Result                  Evidence
  ------------------- --------------------- ----------------------- ----------------
  `[exact command]`          \[0/non-zero\] \[PASS/FAIL/BLOCKED\]   \[actual output
                                                                    excerpt or log
                                                                    path\]

  ----------------------------------------------------------------------------------

### Automated tests

-   **Command:** `[exact pytest command]`
-   **Result:** \[actual passed/failed/skipped counts\]
-   **Failures:** \[exact failures, or "None" if verified\]

### Manual checks

-   \[Check performed and observed result.\]
-   \[If not performed, say "Not run" and why.\]

### Gate result

-   **Gate:** \[V0--V7, or "No milestone gate triggered by this task"\]
-   **Status:** \[PASS / FAIL / BLOCKED / REVIEW\]
-   **Reason:** \[Evidence-based explanation.\]

> Only report commands that were actually run. Do not write "all tests
> pass" unless the full relevant suite was run and passed. If a command
> could not run, record the blocker and do not imply success.

## 8. Results and outputs

-   **Artifacts created/updated:** \[paths\]
-   **Important measured values:** \[real values with units; or "Not
    applicable."\]
-   **Observed behavior:** \[what was verified\]
-   **Reproducibility information:** \[config path, seed, data period,
    command, model tag, as relevant\]

## 9. Deviations from specifications

-   **PRD deviations:** \[None, or describe each with reason and
    approval status.\]
-   **Architecture deviations:** \[None, or describe each.\]
-   **Task-plan deviations:** \[None, or describe each.\]
-   **Dependency changes:** \[None, or list the dependency and
    approval.\]
-   **Unrelated changes:** \[Must be none; if any occurred, explain and
    request review.\]

Never silently reconcile conflicting documents. Link the issue and stop
for user clarification when required.

## 10. Risks, limitations, and open questions

-   \[Known risks or limitations.\]
-   \[Questions requiring user decision.\]
-   \[Unverified assumptions.\]
-   \[If none, say so explicitly.\]

## 11. Learning checklist

After reviewing this task, the student should be able to explain:

-   [ ] What problem this code solves.
-   [ ] What the main functions/classes do.
-   [ ] What enters and leaves the module.
-   [ ] Why the main design decision was made.
-   [ ] How the code is tested and what the test proves.
-   [ ] \[Task-specific concept, formula, tensor shape, metric, or edge
    case.\]

### Explain it simply

\[Give a short explanation as if teaching a student seeing this concept
for the first time.\]

### Questions to check understanding

1.  \[Conceptual question.\]
2.  \[Question about an implementation choice.\]
3.  \[Question about a test, edge case, or limitation.\]

Do not make the task's completion depend on the student already knowing
the answer; use these questions to guide learning and discussion.

## 12. Next task

-   **Next task in the plan:** \[exact task ID and title\]
-   **Why it is next:** \[dependency/sequence\]
-   **Prerequisites:** \[what must be true first\]
-   **Approval required:** \[Yes/No; milestone boundaries require user
    approval.\]

## 13. Completion checklist

-   [ ] Only the assigned task was implemented.
-   [ ] Acceptance criteria were checked.
-   [ ] Relevant tests and validation gate were run and recorded
    honestly.
-   [ ] No tests were deleted, skipped, weakened, or falsely reported.
-   [ ] File-by-file explanation is complete.
-   [ ] Learning checklist is complete.
-   [ ] Deviations and risks are documented.
-   [ ] README/config/docs were updated if required by the task.
-   [ ] Task report is ready to commit.
-   [ ] Commit follows `AGENTS.md` format:
    `<TASK-ID>: <imperative summary>`.
-   [ ] No raw data, processed data, or large artifacts were committed.
-   [ ] If this ends a milestone, development is paused pending user
    approval.

------------------------------------------------------------------------

## Short chat summary (required after every task)

After writing the full report, present this concise summary in chat:

**Task:** \[ID --- title\]\
**What changed:** \[plain-language summary\]\
**Files:** \[paths\]\
**Validation:** \[exact commands and real outcomes\]\
**Result:** \[measured values or observed behavior\]\
**Learning point:** \[most important concept and why it matters\]\
**Deviations/risks:** \[none, or details\]\
**Next task:** \[ID --- title\]\
**Approval needed:** \[yes/no\]

The chat summary complements the full report; it does not replace it.
