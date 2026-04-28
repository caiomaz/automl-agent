# ADR-007. Run Namespace, Handoff Lineage, and Cost Consolidation

## 1. Status

Accepted

## 2. Context

The current repository uses a flat `agent_workspace/` layout with timestamp-based directory naming inside `exp/`. There is no formal `run_id`, no namespace isolation between runs, no persistent handoff lineage, and no auditable cost consolidation. Multiple concurrent or sequential runs can collide by writing to the same directories, and there is no way to reconstruct after the fact which agent passed what to whom.

TASKS.md sections 3.1, 3.2, and 3.3 require formalizing:

1. an operational identity model with distinct IDs,
2. a per-run namespace inside the canonical workspace,
3. a cleanup policy that prevents data loss,
4. dataset provenance tracking per retrieval mode,
5. handoff lineage between agents,
6. cost consolidation per LLM call, per model, and per run,
7. a thinking visibility policy that does not depend on provider-specific chain-of-thought.

## 3. Decision

### 3.1 Operational Identity Model

Five distinct IDs are introduced:

| ID | Format | Created by | Propagation |
| --- | --- | --- | --- |
| `trace_id` | UUIDv4 | Top-level `initiate_chat` | Flows to all child spans and events |
| `run_id` | UUIDv4 | `prepare_new_run()` at CLI or API entry | Flows to all agents, workers, and artifacts |
| `branch_id` | `{run_id}__b{index}` | Agent Manager per plan | Flows to Data Agent, Model Agent, and Operation Agent for that plan |
| `agent_id` | `{agent_type}_{run_id}_{seq}` | Each agent instance on construction | Local to the agent; recorded in events |
| `handoff_id` | UUIDv4 | Emitted at every payload transfer between agents | Recorded in handoff ledger |

### 3.2 Workspace Namespace per Run

The canonical workspace layout from ADR-002 is preserved. A `runs/<run_id>/` subtree is added inside each canonical directory:

```text
agent_workspace/
├── datasets/
│   ├── cache/          ← stable, shared across runs, keyed by URL hash
│   └── runs/<run_id>/  ← per-run dataset references and provenance
├── exp/
│   └── runs/<run_id>/  ← generated code, logs, manifests, analyses
└── trained_models/
    └── runs/<run_id>/  ← saved model artifacts
```

During the transitional period, the existing flat layout remains functional. New code writes into namespaced paths; legacy code that writes to the flat root is not broken but is gradually migrated.

### 3.3 Cleanup Policy

Three modes, configurable per run:

| Mode | Behavior |
| --- | --- |
| `preserve` (default) | No destructive cleanup. Previous run namespaces are left intact. |
| `archive` | Move the active scratch area to a timestamped backup before the new run. |
| `purge` | Wipe the active area. Requires explicit confirmation via HITL or a CLI flag. |

`datasets/cache/` is never purged by default regardless of cleanup mode.

### 3.4 Dataset Provenance

Each retrieval mode records specific provenance metadata in the run namespace:

| Mode | Provenance fields |
| --- | --- |
| `manual-upload` | `source_type`, local origin path, checksum (SHA-256), copy timestamp |
| `user-link` | `source_type`, original URL, cache path, checksum, download timestamp |
| `auto-retrieval` | `source_type`, sources attempted, source chosen, justification, search metadata, download timestamp |

Provenance is persisted as `datasets_provenance.json` inside `exp/runs/<run_id>/`.

### 3.5 Handoff Lineage

Every payload transfer between agents emits a pair of events:

1. `handoff_emitted` — recorded by the source agent,
2. `handoff_received` — recorded by the destination agent.

Each event includes:

1. `handoff_id` (UUIDv4),
2. source `agent_id`,
3. destination `agent_id`,
4. `run_id`,
5. `branch_id`,
6. timestamp,
7. payload size,
8. payload hash (SHA-256 of serialized content),
9. summary (truncated human-readable description).

Handoff events are persisted in `exp/runs/<run_id>/handoffs.jsonl`.

### 3.6 Cost Consolidation

Each LLM call records:

1. provider,
2. alias,
3. model slug,
4. phase (parsing, planning, data, model, verification, coding, revision),
5. prompt tokens,
6. completion tokens,
7. total tokens,
8. reasoning tokens (when available),
9. estimated cost (when returned by provider or computed from known pricing).

Individual records are appended to `exp/runs/<run_id>/cost_records.jsonl`.

At run finalization, a `cost_summary.json` is written with:

1. total cost,
2. total tokens,
3. breakdown by model,
4. breakdown by phase.

### 3.7 Thinking Visibility

1. Record `reasoning_tokens` from the usage response when the provider exposes them.
2. Record reasoning summaries produced by agents as structured events.
3. Persist a `reasoning_trail` as part of the run analyses.
4. Never block functionality when the provider does not expose internal thinking.

## 4. Consequences

### 4.1 Positive

1. Runs are fully isolated; no collision between concurrent or sequential runs.
2. Complete auditability of who passed what to whom and when.
3. Cost per run and per model is reliable and queryable after the fact.
4. Dataset provenance enables reproducibility and compliance.
5. The canonical workspace contract from ADR-002 is preserved.

### 4.2 Negative

1. More files per run increases disk usage.
2. Existing tooling that reads flat `exp/` must be updated to handle namespaced paths.
3. ID propagation across `multiprocessing` boundaries requires explicit serialization.

## 5. Alternatives Considered

1. Using the existing timestamp-based naming as the run identity — rejected because timestamps collide, are not UUIDs, and do not propagate across agent boundaries.
2. Storing all lineage in memory only — rejected because it is lost on crash and cannot be audited after the fact.
3. Using a database for lineage and cost — rejected as premature; JSONL files are sufficient for the current scale and keep the system dependency-free.
