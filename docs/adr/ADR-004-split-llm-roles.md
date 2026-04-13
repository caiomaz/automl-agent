# ADR-004. Split LLM Roles For Backbone And Prompt Parsing

## 1. Status

Accepted

## 2. Context

Requirement parsing and full-pipeline planning do not have the same cost, latency, or reasoning needs. Forcing a single model to handle both roles makes it harder to optimize for speed and budget.

## 3. Decision

The repository keeps separate configuration slots for:

1. the backbone LLM used for planning, verification, and code generation,
2. the Prompt Agent LLM used for requirement parsing and prompt decomposition.

The Prompt Agent slot may also point to a local vLLM-served adapter.

## 4. Consequences

### 4.1 Positive

1. lower cost for parsing,
2. more freedom in model selection,
3. better fit for heterogeneous workflows.

### 4.2 Negative

1. more configuration to understand,
2. more documentation required around precedence and defaults.

## 5. Alternatives Considered

1. a single model for all roles,
2. hard-coding the Prompt Agent to one provider.

These were rejected because they constrain usability and experimentation.