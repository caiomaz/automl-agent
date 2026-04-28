# 07. Execution Pipeline And Artifacts

## 1. End-To-End Flow

AutoML-Agent executes work in layered stages rather than going directly from prompt to code.

The practical runtime flow is:

1. parse the user request,
2. validate and summarize the requirements,
3. generate one or more candidate plans,
4. derive data and model solutions from those plans,
5. verify whether a candidate is worth implementing,
6. synthesize a single implementation instruction,
7. generate and run Python code locally,
8. verify the implementation result,
9. revise if necessary.

## 2. Phase 1: Requirement Parsing

The Prompt Agent converts a free-form request into structured JSON. The Agent Manager then checks:

1. whether the request is relevant to machine learning or AI,
2. whether the resulting requirement object is sufficient to proceed.

This is where the selected task type matters most. The task type from the CLI constrains how the downstream request is interpreted.

## 3. Phase 2: Planning

The Agent Manager generates one or more plans. If RAP is enabled, the planning stage can retrieve supporting knowledge before writing the plans.

Each plan is intended to cover the full ML lifecycle when appropriate, including:

1. data handling,
2. preprocessing,
3. model choice,
4. tuning,
5. evaluation,
6. deployment-oriented outputs.

## 4. Phase 3: Specialist Execution Across Plans

For each plan, the repository runs:

1. a Data Agent execution path,
2. a Model Agent execution path.

These plan-level executions are parallelized across the number of plans selected by the user.

The result is a structured candidate solution made from both the data and model perspectives.

## 5. Phase 4: Pre-Execution Verification

Before code is written, the Agent Manager can ask the backbone LLM whether the proposed solution appears to satisfy the user requirements.

This step helps avoid unnecessary implementation attempts when the plan is clearly not aligned.

If no candidate passes this stage and revision budget remains, the Agent Manager revises the plans and tries again.

## 6. Phase 5: Instruction Synthesis

Once a candidate solution is accepted for implementation, the Agent Manager produces a single detailed instruction set for the Operation Agent.

That instruction typically includes:

1. selected model family,
2. data split expectations,
3. preprocessing strategy,
4. tuning targets,
5. evaluation expectations,
6. artifact save locations.

## 7. Phase 6: Operation Agent Execution

The Operation Agent performs the code-first steps of the workflow:

1. merges the selected instruction with task-specific prompt templates,
2. injects environment constraints and installed-package context,
3. writes a Python file under `agent_workspace/exp/`,
4. runs that file locally as a subprocess,
5. captures stdout and stderr,
6. retries when execution fails.

## 8. Phase 7: Post-Execution Verification

After the script finishes, the Agent Manager evaluates the generated code and the observed results against the original requirements.

Possible outcomes:

1. success and end of run,
2. instruction revision followed by another implementation attempt,
3. failure after the revision budget is exhausted.

## 9. Typical Artifact Locations

| Location | Typical Content |
| --- | --- |
| `agent_workspace/datasets/` | uploaded datasets, URL downloads, cached retrievals |
| `agent_workspace/exp/` | generated scripts and experiment outputs |
| `agent_workspace/trained_models/` | serialized pipelines, checkpoints, final artifacts |

## 10. Common Artifact Types

Not every run produces the same files, but successful runs commonly create:

1. a generated Python script,
2. metrics files such as JSON or CSV,
3. artifact-specific outputs such as residuals, SHAP values, confusion matrices, or forecast logs,
4. a serialized `joblib`, `pkl`, or task-specific model file,
5. optional demo or deployment entry points.

## 11. Naming Conventions

Generated script names usually include some combination of:

1. task type,
2. timestamp or unique identifier,
3. backbone model identifier,
4. plan or feature flags such as RAP or verification.

The exact filename is runtime-specific, but the directory location remains stable under `agent_workspace/exp/`.

## 12. Real-World Example Of Retry Behavior

The reference Crab Age run showed two important behaviors:

1. the generated script initially failed because the dataset was not present locally,
2. a later retry fixed a `joblib` serialization call and completed successfully.

This is not an edge case. It is part of the intended execution-grounded design.

## 13. What To Inspect After A Run

Recommended inspection order:

1. open the generated script in ``agent_workspace/exp/runs/<run_id>/`` (or the legacy flat path if the run predates Phase 1),
2. inspect saved metrics and CSV outputs,
3. inspect the serialized model under ``agent_workspace/trained_models/runs/<run_id>/``,
4. read ``run_manifest.json`` in the run's ``exp/runs/<run_id>/`` directory for status, timestamps, and configuration,
5. verify whether the reported deployment surface actually launched.

For runs created before Phase 1, the legacy flat layout under ``agent_workspace/exp/`` is still valid.

## 14. Run Lifecycle (Phase 1)

Every run now follows a formal lifecycle managed by ``RunContext`` (``utils/run_context.py``):

1. ``prepare_new_run()`` creates a ``RunContext`` with a unique ``run_id``, provisions namespaced workspace directories, and writes an initial ``run_manifest.json``.
2. The ``AgentManager`` receives the ``RunContext`` and uses run-namespaced paths for generated scripts.
3. ``finalize_run()`` marks the context as ``completed``, ``failed``, or ``cancelled``, records the end timestamp, and updates the manifest.

The CLI (both interactive and ``run`` subcommand) creates and finalizes the ``RunContext`` automatically. Keyboard interrupts result in ``cancelled`` status.

## 14.1 Tracing And Observability (Phase 2)

Every run produces a structured audit trail under ``exp/runs/<run_id>/``:

| Artifact | Producer | Purpose |
| --- | --- | --- |
| ``events.jsonl`` | ``utils.ledger.append_event`` | Append-only event log: ``run_started``, ``run_completed``, ``run_failed``, ``run_cancelled``, ``run_cleanup_started``, ``run_cleanup_completed``, ``agent_started``, ``agent_finished``, ``llm_call_completed``, ``handoff_emitted``, ``span_started``, ``span_ended``, ``hitl_requested``, ``hitl_resolved``, ``critic_warned``, ``critic_blocked``, ``reasoning_recorded``, ``dataset_recorded``, ``artifact_written``, ``scheduler_started``, ``scheduler_completed``, ``scheduler_fallback_serial``, ``constraints_recorded``, ``tokens_saved`` |
| ``handoffs.jsonl`` | ``utils.ledger.append_handoff`` / ``emit_handoff`` | Inter-agent handoffs correlated by ``handoff_id`` |
| ``cost_records.jsonl`` | ``utils.ledger.append_cost_record`` / ``record_llm_usage`` | One record per LLM call (provider, alias, model, phase, tokens) |
| ``cost_summary.json`` | ``utils.ledger.write_cost_summary`` | Per-model and per-run aggregation written by ``finalize_run`` |
| ``terminal.log`` | ``operation_agent.execution`` | Captured stdout/stderr from generated-script subprocesses |
| ``analyses/`` | ``utils.ledger.write_analysis`` | Intermediate planning artefacts (``prompt_parse``, ``req_summary``, ``plan_N``, ``code_instruction``) |
| ``analyses/constraints.json`` | ``utils.constraints.persist_constraints`` | Phase-5 structured constraints actually applied to the run (mirror of ``run_manifest.json::constraints``) |
| ``analyses/dataset_provenance.json`` | ``utils.provenance.record_provenance`` | One entry per dataset binding with mode (``manual-upload`` / ``user-link`` / ``auto-retrieval``) and SHA-256 |
| ``analyses/reasoning/<agent>__<label>.txt`` | ``utils.ledger.write_reasoning`` | Free-form reasoning trail entries paired with a ``reasoning_recorded`` event |
| ``analyses/critic/<target>__<review_id>.json`` | ``critic_agent.run_review`` | Critic Agent report per review (target, policy, action, findings) paired with a ``critic_warned`` or ``critic_blocked`` event |
| ``cache/<key>.json`` | ``utils.run_cache.RunCache`` | Phase-8 run-scoped content cache (SHA-256 keys); cache hits emit a ``tokens_saved`` event with ``source="cache_hit"`` and the stored ``tokens_estimate`` |

Two helpers make the audit trail easy to extend:

- ``utils.tracing.span(ctx, name, source=...)`` is a context manager that emits paired ``span_started`` / ``span_ended`` events, capturing ``elapsed_ms`` and marking failures as ``status="error"`` with the exception type.
- ``utils.ledger.emit_handoff(ctx, source_agent_id=..., dest_agent_id=...)`` writes the ``handoffs.jsonl`` record AND the ``handoff_emitted`` event in one call, returning the shared ``handoff_id``.

The ``PromptAgent`` accepts an optional ``run_ctx`` (and ``workspace``) parameter; when provided, it emits the full ``agent_started`` / ``llm_call_completed`` / ``agent_finished`` lifecycle and records cost just like the other agents.

## 15. Planned Evolution

The following accepted ADRs describe changes to the execution pipeline and artifact structure that are planned but not yet implemented:

1. [ADR-007](adr/ADR-007-run-namespace-lineage-cost.md) — Run namespace, ledger, custos, analises e tracing entregues nas Fases 1 e 2. Restante: cobertura sistematica de toda a linha `Prompt -> ... -> Operation` em testes de sistema.
2. [ADR-008](adr/ADR-008-scheduler-fallback.md) — The `multiprocessing.Pool` execution will be replaced by a scheduler with automatic serial fallback. Worker isolation ensures cost and timing data are not lost on mode changes.
3. [ADR-009](adr/ADR-009-hitl-critic-policy.md) — HITL checkpoints (Phase 6) and Critic Agent reviews (Phase 7) are integrated into the pipeline. Critic reports are persisted under `analyses/critic/<target>__<review_id>.json` and the policy is selected via `--critic-policy` or `constraints.critic_policy` (default `warn`).

The current artifact layout and pipeline described above remains valid during the transitional period.

## 16. Reading Continuation

- Read [08. Task Types And Metrics](08_TASK_TYPES_AND_METRICS.md) to understand how planning targets differ by task.
- Read [11. Troubleshooting](11_TROUBLESHOOTING.md) for failure recovery.
- Read [12. Development And Testing](12_DEVELOPMENT_AND_TESTING.md) if you plan to extend the pipeline.