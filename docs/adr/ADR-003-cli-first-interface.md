# ADR-003. CLI-First Interface With Optional Programmatic API

## 1. Status

Accepted

## 2. Context

The project needs a human-friendly onboarding path while still supporting notebooks and advanced experimentation. The original repository already exposed agent orchestration through Python, but the current user experience is centered on the CLI.

## 3. Decision

The default public interface is the CLI:

1. `python -m cli` for interactive use,
2. `python -m cli list-models` for model discovery,
3. `python -m cli run ...` for scripted automation.

The programmatic `AgentManager` API remains available as an advanced path.

## 4. Consequences

### 4.1 Positive

1. lower onboarding friction,
2. clearer operational workflow,
3. easier reproducible shell usage.

### 4.2 Negative

1. CLI documentation must stay current,
2. any CLI drift becomes a user-facing documentation issue.

## 5. Alternatives Considered

1. notebook-first onboarding,
2. programmatic API as the only supported interface.

These were rejected because they are harder to standardize for new users.