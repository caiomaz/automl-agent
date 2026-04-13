# ADR-006. Numbered Documentation And Canonical Terminology

## 1. Status

Accepted

## 2. Context

The repository historically had fragmented documentation, mixed naming, and no single index for readers or agents. This made onboarding and maintenance harder than necessary.

## 3. Decision

Human-facing documentation under `docs/` follows a numbered structure with a shared index and canonical terminology. Agent-facing repository instructions are maintained separately in `.github/copilot-instructions.md`.

Canonical terms include:

1. Agent Manager,
2. Prompt Agent,
3. Data Agent,
4. Model Agent,
5. Operation Agent,
6. Task Type,
7. Workspace,
8. Experiment Outputs,
9. Trained Models,
10. RAP,
11. Plan Decomposition,
12. Verification Loop.

## 4. Consequences

### 4.1 Positive

1. easier navigation,
2. lower ambiguity,
3. better documentation retrieval for both humans and agents,
4. clearer maintenance expectations.

### 4.2 Negative

1. documentation updates must be more disciplined,
2. contributors need to follow the numbering convention.

## 5. Alternatives Considered

1. keeping a flat, unnumbered documentation set,
2. relying only on the README,
3. documenting agent-facing and human-facing rules in the same file.

These were rejected because they do not scale well as the repository evolves.