# LEARNING_PROTOCOL.md --- CrowdCast

## 1. Purpose

CrowdCast is both a software project and a learning project. The coding
agent must not only implement the task correctly; it must help the
student understand what was built, how it works, why the implementation
was chosen, and how it fits into the wider system.

**A task is not complete until its implementation, validation, and
learning report are complete.** A passing test suite alone is not
enough.

This protocol supplements `AGENTS.md`. It does not replace the PRD,
architecture, task plan, validation gates, or task-report template.

## 2. Source-of-truth and precedence

Follow the document precedence defined in `AGENTS.md`:

1.  `docs/PRD.md`
2.  `docs/ARCHITECTURE.md`
3.  `docs/TASK_PLAN.md`
4.  `docs/VALIDATION_GATES.md`
5.  `docs/templates/TASK_REPORT.md`
6.  `docs/EXPERIMENT_LOG.md`

This document defines the required teaching and explanation workflow. It
does not authorize changes that conflict with the sources above.

-   Never invent missing requirements, test results, metrics, files, or
    implementation details.
-   If a requirement is unclear or documents conflict, stop and ask the
    user.
-   If a proposed implementation deviates from the documented design,
    explain the deviation and its consequences, and obtain approval when
    required by `AGENTS.md`.
-   Keep explanations grounded in the actual code and actual command
    output.

7. `LEARNING_PROTOCOL.md`: mandatory pre-coding brief,
   post-task explanation, technical reasoning, and learning checklist.

## 3. The required task lifecycle

For every task in `TASK_PLAN.md`, follow the stages below in order. The
pre-coding briefing and post-coding report must be visible in chat. The
post-coding report must also be saved to the location required by
`AGENTS.md`.

### Stage A --- Task briefing (before editing code)

Before writing or modifying code, send a concise briefing containing:

1.  **Task ID and goal** --- name the exact task from the task plan and
    restate its intended outcome in plain language.
2.  **Why this task matters** --- explain which part of CrowdCast
    depends on it and what it enables next.
3.  **Concepts to understand** --- introduce the relevant programming,
    data-science, or deep-learning concepts at a student-friendly level.
4.  **Planned approach** --- outline the implementation steps in order.
5.  **Files expected to change** --- list the expected files and each
    file's responsibility.
6.  **Key decisions and rationale** --- state important design choices
    already specified by the architecture, plus any implementation
    choices that remain open.
7.  **Validation plan** --- identify the exact tests, commands, and
    acceptance criteria that will be used.
8.  **Scope boundaries** --- state what will not be changed as part of
    this task.

Keep the briefing short enough to read before coding, but detailed
enough that the student can predict the shape of the implementation. Do
not ask for approval for routine work already specified by the plan. Do
ask when a genuine ambiguity or approval-required decision exists.

### Stage B --- Implementation

Implement only the current task.

-   Follow `AGENTS.md`, the architecture, and the task plan.
-   Prefer small, single-purpose functions and clear names.
-   Use type hints and docstrings, including tensor shapes where tensors
    are involved.
-   Keep configuration values in the designated configuration files
    rather than hard-coding hyperparameters, paths, or thresholds.
-   Add or update tests for the behaviour being implemented.
-   Do not perform unrelated refactors, add unrequested features, or
    silently change interfaces.
-   Do not claim a test passed unless it was actually run and passed.

### Stage C --- Validation

Run the validation specified by the task and the applicable validation
gate.

Report: - The exact command run. - The actual result and relevant
output. - Which acceptance criteria passed or failed. - Any additional
checks performed and why. - Any checks not run, with the reason.

If validation fails, explain the failure and investigate it. Do not
weaken, skip, or delete tests to obtain a green result. Follow the retry
and stop rules in `AGENTS.md`.

### Stage D --- Post-task teaching report

After implementation and validation, explain the completed work in chat
and write the report to the repository location required by `AGENTS.md`
(currently `docs/templates/TASK_REPORT.md` is the report template;
follow the repository's defined reporting convention rather than
overwriting the template itself).

Use the report structure in Section 4 below. Tailor the depth to the
task: a small configuration task needs a concise explanation; a model,
data pipeline, training, or evaluation task needs a deeper technical
walkthrough.

### Stage E --- Stop point

After the report:

-   State whether the task is complete and whether its validation gate
    passed.
-   List unresolved issues, limitations, and open questions.
-   Name the next task from `TASK_PLAN.md`, but do not begin it
    automatically if a milestone boundary requires user approval.
-   At the end of each milestone, stop and wait for the user's approval
    as required by `AGENTS.md`.

## 4. Required post-task report format

Use this structure for every task. Sections may be marked "Not
applicable" only when that is genuinely appropriate.

### 4.1 Task summary

-   **Task ID and title:**
-   **Outcome:** What now works or exists?
-   **Why it matters:** What does this enable in CrowdCast?

Explain in plain language first. Avoid opening with a wall of code or
unexplained terminology.

### 4.2 What changed, file by file

For every file created or modified, include:

  File   Responsibility   What changed   Why it changed
  ------ ---------------- -------------- ----------------

Then explain the important implementation details for each file. Do not
merely repeat the filename or provide a vague statement such as
"implemented logic."

For source files, describe: - The purpose of the module. - The
responsibility of each important class or function. - The inputs,
outputs, and important data structures. - The main steps in the logic,
in execution order. - How errors and edge cases are handled. - Which
other modules call it or depend on its output.

For tests, describe: - The behaviour each test protects. - The important
test inputs and expected outputs. - The bug or regression the test would
catch.

For configuration or documentation, describe: - The setting or contract
introduced or changed. - Which code reads or relies on it. - Why the
value or wording is appropriate under the project specification.

### 4.3 Code walkthrough

Walk through the important logic in a logical order, using small code
excerpts only where they make the explanation clearer.

Explain: - What the code is doing. - Why that step is necessary. - What
the variables and shapes represent. - What would happen if the step were
removed or changed. - Any non-obvious Python, pandas, NumPy, PyTorch, or
Streamlit behaviour.

Do not explain every punctuation mark or every trivial line. Focus on
the logic that a student should understand and be able to explain
independently.

### 4.4 Architecture and data flow

Explain how the changes fit into the larger system:

-   What calls or uses this code?
-   What input does it receive, and where does that input originate?
-   What output or artifact does it produce?
-   Where is that output stored?
-   Which downstream task or module consumes it?

When helpful, include a compact text flow such as:

`raw input → transformation → saved artifact → downstream consumer`

Do not claim an integration works unless it has been implemented and
verified. Distinguish the current task's verified behaviour from planned
future integration.

### 4.5 Decisions and why

Document each meaningful design or implementation decision:

  -----------------------------------------------------------------------
  Decision          Why this approach Alternative(s)    Trade-off or
                                                        consequence
  ----------------- ----------------- ----------------- -----------------

  -----------------------------------------------------------------------

Include decisions about architecture, data representation, algorithms,
dependencies, interfaces, error handling, testing, performance, and
reproducibility where relevant.

Distinguish between: - **Specified decisions:** required by the PRD,
architecture, or task plan. - **Implementation decisions:** choices made
while implementing within the specification. - **Deferred decisions:**
unresolved choices that require the user's input or a later task.

Do not invent alternatives that were actually evaluated. If alternatives
were not investigated, label them as possible alternatives rather than
claiming they were considered.

### 4.6 Deep-learning and data-science explanation

Include this section whenever a task touches data preparation, feature
engineering, windowing, models, loss functions, training, evaluation,
experiments, or forecasting. For unrelated tasks, give a short note
explaining why this section is not applicable.

Cover whichever items apply:

-   **Data:** what each row or sample represents; important columns and
    data types.
-   **Features and targets:** what information the model receives and
    what it must predict.
-   **Shapes:** explain dimensions such as `(N, L, F)`, `(B, H, Q)`, and
    what each symbol means.
-   **Mathematics:** explain important equations and variables in plain
    language.
-   **Training:** explain what is learned, what the loss measures, and
    how optimization or early stopping works.
-   **Evaluation:** explain what each metric measures and how to
    interpret it.
-   **Experimental fairness:** explain how the task preserves comparable
    conditions between LSTM, GRU, and baselines where relevant.
-   **Leakage and time:** explain how chronological splits, train-only
    statistics, validation-only model selection, and untouched test data
    apply.
-   **Limitations:** explain what the implementation cannot conclude or
    predict.

For CrowdCast specifically, preserve the project's documented rules:
chronological train/validation/test splits; train-only fitting of
statistics; validation-based early stopping and model selection;
identical LSTM/GRU protocols; seeded experiments; pinball loss for
quantiles; ordered P10/P50/P90 at inference; inverse transformation and
non-negative clipping before metrics; and honest reporting even if a
baseline wins.

Do not imply that a model is accurate merely because training completes
or loss decreases. Use actual evaluation results when available.

### 4.7 Validation results

Include a compact table:

  Check   Command or method   Actual result   Status
  ------- ------------------- --------------- --------

Use statuses such as `PASS`, `FAIL`, `NOT RUN`, or `BLOCKED`. Never mark
a check `PASS` based on expectation alone.

For numeric results, show the actual value and relevant context. Do not
fabricate or selectively omit results that make the system look worse.

### 4.8 Risks, limitations, and deviations

Explicitly state: - Deviations from the PRD, architecture, task plan, or
validation gate. - Why each deviation occurred. - Whether approval was
obtained or is still needed. - Remaining technical risks and known
limitations. - Missing data, unrun experiments, or unverified
integrations.

If there were no deviations, say so. Do not silently normalize an
unapproved deviation.

### 4.9 Learning checklist

Finish with a short checklist the student can use to assess
understanding. Include 3--6 relevant prompts, for example:

-   [ ] I can explain the purpose of each changed file.
-   [ ] I can trace the data from input to output.
-   [ ] I can explain the main algorithm without reading the
    implementation line by line.
-   [ ] I can justify the key design decision and name a trade-off.
-   [ ] I can explain what the tests prove and what they do not prove.

Tailor the checklist to the task. For a model task, include relevant
shapes, loss, training, or evaluation questions.

### 4.10 Next step

State: - **Task status:** complete, incomplete, or blocked. -
**Validation gate:** passed, failed, or not applicable. - **Open
questions:** if any. - **Next planned task:** exact ID and title from
the task plan. - **Approval needed:** yes or no, and why.

Do not start the next task in the same change set unless the task plan
explicitly groups the work together.

## 5. Explanation quality rules

The goal is to teach, not merely to document that code exists.

1.  **Plain language first.** Introduce an idea intuitively, then
    provide technical details.
2.  **Define terminology.** Explain unfamiliar terms at first use; do
    not assume the student already knows the concept.
3.  **Connect code to purpose.** Explain why a step exists, not only
    what syntax it uses.
4.  **Use concrete examples.** When useful, walk through a small example
    with realistic shapes or values. Clearly label illustrative values
    as examples.
5.  **Explain consequences.** State what could break, become incorrect,
    or change if a key line or decision were different.
6.  **Separate fact from interpretation.** Clearly distinguish verified
    behaviour, design intent, expected behaviour, and illustrative
    examples.
7.  **Be honest about uncertainty.** If code or output was not inspected
    or a check was not run, say so.
8.  **Avoid overwhelming detail.** Explain all important logic, but
    group routine lines and focus on concepts that help the student
    understand the system.
9.  **Do not hide behind generated code.** The agent must be able to
    explain the code it wrote in terms of the project's requirements and
    the underlying concepts.
10. **No unsupported claims.** Do not claim production readiness,
    correctness, model quality, or successful integration without
    suitable evidence.

## 6. Learning interaction

The student may ask follow-up questions about any task report. The agent
should answer using the actual implementation and project documents.

When useful, the agent may offer a short optional comprehension check
after the explanation. Do not force a quiz or block progress on a quiz
unless the user explicitly asks for a teaching session.

If the student identifies a misunderstanding, clarify the concept using
a different example or explanation. Do not make code changes merely
because the student asks a conceptual question.

If the student requests a code change while learning, determine whether
it belongs to the current task or should be a separate task. Follow the
one-task-at-a-time rule in `AGENTS.md`.

## 7. Chat summary and repository report

After every task, provide both:

1.  **A readable chat explanation** with the task summary, the most
    important file explanations, key decisions and rationale, actual
    validation results, risks, and next task.
2.  **A durable repository report** saved according to the project's
    reporting convention.

The chat version may be shorter than the repository report, but it must
be useful on its own. Do not say "see the report" as a substitute for
explaining the work in chat.

Keep the task report template intact. If the repository later
establishes a dedicated report directory or per-task filenames, follow
that convention consistently and update this protocol only with the
user's approval.

## 8. Definition of done

A task is complete only when all applicable items are satisfied:

-   [ ] The exact task scope was briefed before coding.
-   [ ] Only task-relevant changes were made.
-   [ ] The implementation follows the source-of-truth documents.
-   [ ] Tests and required validation were actually run.
-   [ ] Actual command output and results are reported honestly.
-   [ ] Every changed file is explained.
-   [ ] The main code path and data flow are explained.
-   [ ] Meaningful decisions and their rationale are documented.
-   [ ] Deep-learning concepts are explained when relevant.
-   [ ] Deviations, limitations, and open questions are disclosed.
-   [ ] The post-task report is saved in the repository.
-   [ ] The chat explanation is provided.
-   [ ] The next task and any required approval boundary are stated.

If any applicable item is incomplete, explicitly label the task
`INCOMPLETE` or `BLOCKED`; do not describe it as done.

## 9. Required response template

Use this concise structure in chat after each task:

1.  **Task and outcome**
2.  **Files changed** --- what each file does and why it changed
3.  **How it works** --- the important code path and data flow
4.  **Key decisions** --- why this design was chosen and its trade-offs
5.  **Relevant concepts** --- deep-learning or engineering explanation
6.  **Validation** --- exact commands and actual results
7.  **Risks/deviations**
8.  **Learning checklist**
9.  **Next task and approval status**

This is the minimum chat report. Expand the sections when the task
involves complex model behaviour, data handling, training, evaluation,
or architectural changes.

------------------------------------------------------------------------

**Core rule: Build the project in a way the student can understand,
explain, validate, and defend---not merely run.**
