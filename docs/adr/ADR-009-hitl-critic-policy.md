# ADR-009. HITL Checkpoints and Critic Agent Policy

## 1. Status

Accepted

## 2. Context

The current system has minimal human-in-the-loop interaction: the interactive CLI asks for confirmation before proceeding from PLAN to ACT, and the Agent Manager can optionally ask the user to validate the parsed requirements. There is no formal checkpoint system, no structured recording of human decisions, and no dedicated critic agent to catch errors before they propagate.

TASKS.md sections 3.5 and 3.6 require formalizing HITL levels, checkpoint locations, and a Critic Agent policy.

## 3. Decision

### 3.1 HITL Levels

Three configurable levels:

| Level | Behavior |
| --- | --- |
| `off` | Zero extra human checkpoints beyond the existing interactive flow. Default for `cli run` (non-interactive mode). |
| `standard` | Checkpoints at high-risk points where early correction prevents wasted time and cost. Default for interactive mode. |
| `strict` | Checkpoints before every sensitive step. For high-stakes or compliance-sensitive runs. |

The HITL level is set per run via CLI flag, run configuration, or API parameter.

### 3.2 Checkpoint Locations

The following checkpoints are defined. Each is active only at the indicated HITL level or higher:

| Checkpoint | Active at | Trigger |
| --- | --- | --- |
| After Prompt Agent parse | `standard` | User reviews parsed requirements JSON before planning begins. |
| After plan generation | `standard` | User reviews generated plans before specialist execution. |
| Before executing a Critic-rejected plan | `standard` | User decides whether to proceed despite critic warning. |
| Before destructive workspace cleanup | `standard` | User confirms purge of previous run data. |
| Before code generation when high risk is detected | `strict` | User reviews code instructions before Operation Agent generates code. |
| Before accepting final result with conflicting verifiers | `strict` | User resolves disagreement between pre-exec and post-exec verification. |
| Before exposing web demo or public endpoint | `strict` | User confirms before any external-facing deployment step. |

### 3.3 Decision Recording

Every HITL interaction is recorded as a pair of events in the run ledger:

1. `hitl_requested` — includes checkpoint name, context presented to the user, options available, timestamp, `run_id`, `branch_id`, `agent_id`.
2. `hitl_resolved` — includes user decision (approve, reject, modify, skip), any user-provided rationale, timestamp, duration of wait.

Recorded decisions are also persisted in `exp/runs/<run_id>/analyses/hitl_decisions.jsonl`.

### 3.4 Non-Interactive Compatibility

In `cli run` mode (non-interactive), HITL defaults to `off`. If explicitly set to `standard` or `strict`, the system:

1. uses pre-configured default decisions when available (e.g., auto-approve with logging),
2. fails gracefully with a clear error message when a checkpoint requires input that cannot be automated.

This ensures that batch and CI usage is not broken by HITL configuration.

### 3.5 Critic Agent Policy

The Critic Agent is a new review role that inspects agent outputs at critical transitions. Three operating modes:

| Mode | Behavior |
| --- | --- |
| `always` | Critic reviews every critical transition (parse, plans, handoffs, code instructions, execution results). |
| `high-risk` (default) | Critic reviews only when risk signals are detected: constraint violations, unusual cost, suspicious paths, missing artifacts, metric mismatches. |
| `off` | Critic is disabled. Existing verification logic in the Agent Manager still runs. |

The Critic Agent mode is set per run via CLI flag or run configuration.

### 3.6 Critic Authority Levels

The Critic Agent can take one of four actions at each review point:

| Action | Effect |
| --- | --- |
| `pass` | No issue found. Execution continues silently. |
| `warn` | Issue logged as `critic_warned` event. Execution continues. Manager records the warning. |
| `request_hitl` | Issue escalated to human via HITL checkpoint. Execution blocks until human decides. |
| `block` | Execution is halted. Only available when HITL level is `strict`. Otherwise downgrades to `request_hitl`. |

The Agent Manager records whether it followed or overrode the Critic Agent recommendation.

### 3.7 Critic Review Scope

The Critic Agent checks for:

1. constraint violations (model, metric, time, framework preferences),
2. incorrect file paths (hardcoded absolute paths, paths outside workspace),
3. missing or unavailable packages,
4. inconsistencies between dataset and task type,
5. irrelevant metrics for the task type,
6. implausible hyperparameters,
7. hallucinated artifacts (references to files that do not exist),
8. lineage breaks (missing handoff data),
9. abnormal cost (single call exceeding a configurable threshold).

Critic reports are persisted in `exp/runs/<run_id>/analyses/critic/`.

## 4. Consequences

### 4.1 Positive

1. Early human correction prevents cascading errors and wasted cost.
2. All human decisions are auditable.
3. The Critic Agent catches common failure modes before they reach code execution.
4. Non-interactive mode is not broken by HITL or Critic configuration.
5. The system is configurable for different risk tolerances.

### 4.2 Negative

1. HITL checkpoints increase wall-clock time for interactive runs.
2. The Critic Agent adds LLM calls and cost per run.
3. The interaction between HITL levels and Critic authority requires clear documentation to avoid user confusion.
4. A new agent role increases the number of modules to maintain.

## 5. Alternatives Considered

1. Relying on the existing verification steps only — rejected because they are coarse-grained and do not cover the full range of detectable errors.
2. Making the Critic Agent a mandatory always-on component — rejected because it adds cost and latency that may not be justified for simple or low-stakes tasks.
3. Embedding critic logic directly in the Agent Manager — rejected because it violates the agent separation principle (ADR-001) and makes the Manager harder to maintain.
