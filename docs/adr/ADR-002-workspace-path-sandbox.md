# ADR-002. Workspace Path Sandbox

## 1. Status

Accepted

## 2. Context

Generated code runs locally and needs deterministic file locations for datasets, outputs, and serialized models. Historically, LLM-generated code tends to invent absolute paths that do not exist or violate the intended workspace boundaries.

## 3. Decision

The repository defines a canonical local workspace under `agent_workspace/` with the following subdirectories:

1. `datasets/`,
2. `exp/`,
3. `trained_models/`.

All project code and generated code must target these directories through shared helpers and path constants.

## 4. Consequences

### 4.1 Positive

1. predictable artifact layout,
2. easier debugging,
3. less path drift in generated code,
4. simpler documentation.

### 4.2 Negative

1. less freedom for ad hoc path conventions,
2. more prompt and code constraints to maintain.

## 5. Alternatives Considered

1. allowing arbitrary absolute paths,
2. allowing each generated script to define its own artifact tree.

These alternatives were rejected because they reduce reproducibility and make failures harder to inspect.