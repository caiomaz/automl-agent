# ADR-008. Scheduler With Queue and Serial Fallback

## 1. Status

Accepted

## 2. Context

The current Agent Manager uses `multiprocessing.Pool` to execute plans in parallel. This works when the LLM provider is responsive, but has no mechanism to handle rate limiting (HTTP 429), provider instability, or quota exhaustion. When the provider blocks concurrent requests, the pool workers either hang or fail simultaneously, wasting tokens and time.

TASKS.md section 3.4 requires formalizing a concurrency policy with controlled fallback.

## 3. Decision

### 3.1 Default Execution Mode

Branch-level concurrency is the default. Each plan generates a branch, and branches execute their Data Agent and Model Agent reasoning in parallel, preserving the current `Pool`-based approach as the starting point.

### 3.2 Concurrency Limits

Configurable semaphores per provider and per model control the maximum number of concurrent LLM calls. Defaults are conservative (e.g., 3 concurrent calls per model). The limits are read from environment variables or run configuration and can be overridden per run.

### 3.3 Fallback Trigger

The scheduler automatically falls back to serial queue execution when any of the following conditions are detected:

1. HTTP 429 (Too Many Requests) received from the provider,
2. persistent rate-limit errors across multiple retries,
3. provider instability (repeated connection errors or timeouts within a short window).

The fallback is per-provider: if one provider is rate-limited, only branches targeting that provider switch to serial; other providers may continue concurrently.

### 3.4 Identity Preservation

The `branch_id → agent_id → handoff_id` chain defined in ADR-007 remains consistent regardless of whether execution is concurrent or serial. Switching execution mode does not change the identity or lineage of a branch.

### 3.5 Observability

The following events are recorded in the run event ledger:

1. `scheduler_fallback_to_serial` — includes provider, model, trigger reason, timestamp.
2. `scheduler_restored_concurrent` — includes provider, model, timestamp.
3. `scheduler_branch_started` — includes `branch_id`, execution mode.
4. `scheduler_branch_completed` — includes `branch_id`, execution mode, duration.

### 3.6 Worker Isolation

Cost, timing, and event data must not depend on mutable shared state inside worker processes. Workers return structured results (including cost records and timing) as serializable return values. The main process aggregates them into the run ledger and cost summary after workers complete.

## 4. Consequences

### 4.1 Positive

1. The system degrades gracefully under rate limiting instead of failing.
2. Cost and lineage data are not lost when the execution mode changes.
3. The scheduler is observable and auditable.
4. Provider-specific limits prevent exhausting shared quotas.

### 4.2 Negative

1. Serial fallback increases wall-clock time for affected branches.
2. Semaphore configuration adds operational complexity.
3. The scheduler introduces a new module or abstraction layer in the codebase.

## 5. Alternatives Considered

1. Always serial execution — rejected because it negates the benefit of multi-plan exploration and significantly increases latency for well-behaved providers.
2. Exponential backoff without mode change — rejected because sustained rate-limiting makes concurrent retries wasteful; serial execution is cheaper and more predictable.
3. User-only manual switch — rejected because users should not need to diagnose provider behavior in real time; automatic fallback is safer.
