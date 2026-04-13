# ADR-001. Multi-Agent Specialization

## 1. Status

Accepted

## 2. Context

The project covers more than model selection. It must parse natural-language requirements, reason about data sources, choose modeling strategies, and generate executable code that runs locally.

A single undifferentiated agent would have to manage conflicting roles and would make it harder to preserve clear contracts between planning, execution, and revision.

## 3. Decision

AutoML-Agent uses a specialized multi-agent structure composed of:

1. Prompt Agent,
2. Agent Manager,
3. Data Agent,
4. Model Agent,
5. Operation Agent.

The Agent Manager remains the orchestrator and is responsible for coordination, validation, and revision logic.

## 4. Consequences

### 4.1 Positive

1. clearer responsibilities,
2. better prompt specialization,
3. cleaner verification boundaries,
4. easier extension of data or model behavior.

### 4.2 Negative

1. more orchestration complexity,
2. more states to reason about,
3. more surface area for documentation and debugging.

## 5. Alternatives Considered

1. a single monolithic agent,
2. a code-only generator with no explicit planning stage.

Both were rejected because they make it harder to reason about the system and to revise failures in a structured way.