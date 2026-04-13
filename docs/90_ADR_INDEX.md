# 90. ADR Index

## 1. Purpose

This index tracks the architectural and governance decisions that shape AutoML-Agent.

The ADR set includes both:

1. decisions already reflected in the current implementation,
2. newly formalized rules introduced by this documentation overhaul.

## 2. ADR List

| ADR | Title | Status |
| --- | --- | --- |
| [ADR-001](adr/ADR-001-multi-agent-specialization.md) | Multi-Agent Specialization | Accepted |
| [ADR-002](adr/ADR-002-workspace-path-sandbox.md) | Workspace Path Sandbox | Accepted |
| [ADR-003](adr/ADR-003-cli-first-interface.md) | CLI-First Interface With Optional Programmatic API | Accepted |
| [ADR-004](adr/ADR-004-split-llm-roles.md) | Split LLM Roles For Backbone And Prompt Parsing | Accepted |
| [ADR-005](adr/ADR-005-ephemeral-generated-artifacts.md) | Ephemeral Generated Artifacts Policy | Accepted |
| [ADR-006](adr/ADR-006-numbered-documentation-standard.md) | Numbered Documentation And Canonical Terminology | Accepted |

## 3. How To Use These ADRs

Use the ADRs when you need to answer questions such as:

1. why are there multiple agents instead of a single one,
2. why are generated scripts forced into `agent_workspace/`,
3. why is the CLI treated as the default public interface,
4. why are Prompt Agent and backbone model roles separate,
5. why are generated outputs treated as ephemeral,
6. why is the documentation suite numbered and terminology-controlled.

## 4. Maintenance Rule

When a future change deliberately alters one of these accepted decisions, either:

1. update the affected ADR with a superseding note, or
2. add a new ADR that explicitly supersedes the older decision.