# ADR-005. Ephemeral Generated Artifacts Policy

## 1. Status

Accepted

## 2. Context

The repository generates datasets, experiment outputs, scripts, models, logs, and backup runs at execution time. These artifacts are large, noisy, and often machine-specific.

Treating them as source-of-truth repository content would clutter history and confuse the distinction between handwritten source and generated runtime output.

## 3. Decision

Generated content under runtime-oriented directories such as `agent_workspace/` and `bkp/` is treated as ephemeral by default and should be ignored going forward unless a curated example is intentionally versioned.

## 4. Consequences

### 4.1 Positive

1. cleaner repository history,
2. reduced accidental binary or dataset commits,
3. clearer separation between source and outputs.

### 4.2 Negative

1. contributors must consciously preserve any example artifacts they want to share,
2. documentation must explain where artifacts go because they are not treated as permanent tracked content.

## 5. Alternatives Considered

1. tracking all generated outputs,
2. leaving ignore rules broad but undocumented.

These were rejected because they make repository hygiene and contributor intent harder to maintain.