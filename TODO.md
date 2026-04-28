# TODO

Checklist operacional alinhado ao [TASKS.md](TASKS.md).

Objetivo deste arquivo:

1. servir como painel de acompanhamento rapido,
2. registrar o que ja foi entregue no repositorio,
3. deixar explicito o que ainda falta por fase,
4. permitir atualizacao incremental sem reescrever o plano-base.

Regra de uso:

1. o plano-fonte continua sendo o `TASKS.md`,
2. este arquivo so traduz o plano para acompanhamento executavel,
3. ao concluir uma entrega, atualizar o status da fase e marcar os itens correspondentes.

Legenda:

1. `concluida`: fase entregue de forma satisfatoria para o escopo atual.
2. `parcial`: ha entrega relevante, mas ainda existem gaps frente ao `TASKS.md`.
3. `pendente`: ainda nao iniciada.

## Resumo Geral

| Fase | Status | Observacao curta |
| --- | --- | --- |
| 0 | concluida | ADRs novos + baseline de regressao criados |
| 1 | concluida | Active-run guard, cleanup `preserve/archive/purge`, provenance por modo entregues |
| 2 | concluida | tracing/spans, handoff_emitted (incl. agent_manager), hitl/critic, reasoning trail e cost reconciliation entregues |
| 3 | concluida | `BranchScheduler` com fallback serial substituiu `multiprocessing.Pool`; eventos de scheduler no ledger |
| 4 | concluida | Cancelamento gracioso (SIGINT/SIGTERM), `--scheduler-mode`, `--max-concurrency`, subcomando `list-runs` e summary pos-run entregues |
| 5 | concluida | constraints granulares no schema, helper `persist_constraints`, novas flags na CLI e merge no manifesto + `analyses/constraints.json` |
| 6 | concluida | `utils.hitl.request_checkpoint` com politica off/standard/strict, gate interativo antes de purge, `--hitl-level` na CLI |
| 7 | concluida | Critic Agent rule-based + politica `off/warn/request_hitl/block`, reports em `analyses/critic/`, integrado a parse e plans no manager |
| 8 | concluida | token economy + roteamento por etapa + run cache; eventos `tokens_saved`; operation agent compacta erros antes de reenviar |
| 9 | parcial | subprocesso do Operation Agent ja grava em `terminal.log`; restante ainda nao |
| 10 | parcial | baseline + testes Phase 1/2/3/4/5/6 entregues; suite alta e2e ainda nao |
| 11 | parcial | docs das fases iniciais/ADRs atualizadas; sync completo ainda nao |
| 12 | pendente | matriz formal de providers ainda nao |
| 13 | pendente | monorepo `agent/backend/frontend` ainda nao |
| 14 | pendente | backend FastAPI ainda nao |
| 15 | pendente | frontend web ainda nao |
| 16 | pendente | modos unificados ainda nao |
| 17 | pendente | containerizacao e compose ainda nao |
| 18 | pendente | documentacao de deploy ainda nao |
| 19 | pendente | modernizacao LangChain/LangSmith ainda nao |
| 20 | pendente | tools operacionais com approvals ainda nao |

## Ja Entregue Ate Agora

### Base entregue no repositorio

- [x] ADR-007 criado: `docs/adr/ADR-007-run-namespace-lineage-cost.md`
- [x] ADR-008 criado: `docs/adr/ADR-008-scheduler-fallback.md`
- [x] ADR-009 criado: `docs/adr/ADR-009-hitl-critic-policy.md`
- [x] Indice de ADR atualizado em `docs/90_ADR_INDEX.md`
- [x] Suite de regressao adicionada para CLI, Agent Manager, Prompt Agent, Data/Model Agents, Operation Agent, tracing e configs
- [x] `RunContext` criado em `utils/run_context.py`
- [x] Helpers de workspace por run criados em `utils/workspace.py`
- [x] CLI integrada a `prepare_new_run(...)` e `finalize_run(...)`
- [x] Ledger local criado em `utils/ledger.py`
- [x] `events.jsonl`, `handoffs.jsonl`, `cost_records.jsonl`, `cost_summary.json` e `analyses/` por run passaram a ter suporte de escrita
- [x] `terminal.log` por run passou a ser escrito pelo fluxo de execucao do Operation Agent
- [x] Testes `tests/test_run_context.py` adicionados
- [x] Testes `tests/test_ledger.py` adicionados
- [x] Dependencias `playwright` e `html2text` adicionadas ao `requirements.txt`
- [x] Fallback robusto de retrieval web implementado em `data_agent/retriever.py` para nao depender rigidamente de Playwright
- [x] Correcao de `_manager_agent_id` no `RunContext` para compatibilidade com `__slots__`

### Observacoes de escopo

- [x] O repositorio avancou bem nas fases 0, 1 e 2, mas nem todas as tarefas dessas fases estao completas segundo o `TASKS.md`
- [x] Este TODO reflete o estado real do codigo, nao apenas a intencao da fase

## Fase 0. Preparacao, baseline e ADRs

Status: `concluida`

### Feito

- [x] Criar ADR para namespace por run, lineage de handoff, custo consolidado e politica de cleanup
- [x] Criar ADR para scheduler com fallback fila/serial
- [x] Criar ADR para HITL e Critic Agent
- [x] Congelar comportamento atual com testes de regressao antes da mudanca arquitetural
- [x] Atualizar indice de ADRs e docs de arquitetura para refletir os novos ADRs

### Ainda revisar eventualmente

- [ ] Referenciar de forma mais explicita os artefatos de `bkp/run-01/terminal.log` e `agent_workspace/exp/` na documentacao de migracao, se isso ainda fizer falta na evolucao das proximas fases

## Fase 1. Run lifecycle, namespace e limpeza controlada do workspace

Status: `concluida`

### Feito

- [x] Introduzir `RunContext` serializavel com `run_id`, `branch_id`, `agent_id`, `attempt_id`, timestamps, status, task type, backbone LLM, prompt LLM e politica de HITL
- [x] Criar helpers de path por run em `utils.workspace`
- [x] Criar `prepare_new_run(...)`
- [x] Criar `finalize_run(...)`
- [x] Remover dependencia exclusiva do `code_path` flat no fluxo principal e passar a usar `run_exp_dir(run_id)` no fluxo da CLI/manager/operation
- [x] Garantir coexistencia basica de multiplas runs sem colisao em `exp`, `datasets` e `trained_models`
- [x] Atualizar docs iniciais de workspace/artefatos para o namespace por run
- [x] Criar cobertura automatizada para `RunContext` e helpers de workspace
- [x] Validar formalmente se existe run ativa antes de nova run (`ActiveRunError`, `force=True` para override)
- [x] Implementar politica real de cleanup `preserve/archive/purge` em `utils.workspace.cleanup_workspace(...)`
- [x] Limpar area scratch/ativa de forma orquestrada antes de nova run (cleanup roda antes do provisionamento do novo `run_id`)
- [x] Manter cache de datasets remotos preservado em todas as politicas de cleanup
- [x] Emitir eventos `run_cleanup_started` e `run_cleanup_completed` no ledger da nova run
- [x] Liberar registro de run ativa em `finalize_run(...)` para permitir nova run subsequente
- [x] Integrar provenance por modo de dataset (`manual-upload`, `user-link`, `auto-retrieval`) via `utils.provenance.DatasetProvenance` + `record_provenance(...)`
- [x] Registrar checksum SHA-256 da fonte local (`compute_checksum`) e emitir evento `dataset_recorded`
- [x] Wirar `--cleanup-mode` na CLI `run` e questionar a politica no wizard interativo
- [x] Wirar registro de provenance na CLI tanto para `--data` quanto para `data_url` no wizard
- [x] Adicionar `tests/test_run_lifecycle.py` cobrindo active-run, cleanup, eventos de cleanup e provenance

### Falta

- [ ] Politica formal de fechamento de logs/locks/filas em `finalize_run(...)` quando esses recursos passarem a existir (Fase 3+)

## Fase 2. Tracing robusto, lineage, logs e custos

Status: `parcial`

### Feito

- [x] Criar manifest por run em `agent_workspace/exp/runs/<run_id>/run_manifest.json`
- [x] Criar ledger de eventos em `events.jsonl`
- [x] Criar ledger de handoffs em `handoffs.jsonl`
- [x] Criar `cost_summary.json` e `cost_records.jsonl` por run
- [x] Criar `terminal.log` por run para o subprocesso do Operation Agent
- [x] Criar `analyses/` por run
- [x] Registrar parte dos eventos minimos (`run_started`, `agent_started`, `agent_finished`, `manager_waiting`, `manager_received`, `llm_call_completed`, `artifact_written`, `run_cancelled`, `run_failed`, `run_completed`)
- [x] Registrar custo por chamada LLM com provider inferido, alias, modelo, fase e tokens
- [x] Consolidar custo por modelo e por run no final da run
- [x] Registrar handoffs Manager -> Data e Manager -> Model e retornos basicos para o manager
- [x] Persistir analises intermediarias como `prompt_parse`, `req_summary`, `plan_N`, `code_instruction`
- [x] Criar testes de ledger/custo/analises
- [x] Evoluir `utils/tracing.py` com context manager `span(...)` que emite `span_started`/`span_ended` (com `elapsed_ms` e `status`)
- [x] Instrumentar `prompt_agent/__init__.py` para emitir `agent_started`, `llm_call_completed`, `agent_finished` e custo via `record_llm_usage` quando `run_ctx` esta presente
- [x] Wirar `run_ctx` no `agent_manager` ao chamar `parser.parse(...)`
- [x] Registrar `run_cleanup_started` e `run_cleanup_completed` (entregue na Fase 1)
- [x] Registrar `handoff_emitted` como evento estruturado via `utils.ledger.emit_handoff(...)` com correlacao por `handoff_id`
- [x] Registrar `hitl_requested` e `hitl_resolved` via `append_hitl_requested` / `append_hitl_resolved`
- [x] Registrar `critic_blocked` e `critic_warned` via `append_critic_warned` / `append_critic_blocked`
- [x] Salvar reasoning trail observavel via `utils.ledger.write_reasoning(...)` em `analyses/reasoning/<agent>__<label>.txt` + evento `reasoning_recorded`
- [x] Garantir que o custo final bata com a soma das chamadas em cenarios multi-agente / multi-modelo (regressao em `tests/test_tracing_phase2.py::TestCostReconciliation`)
- [x] Adicionar `tests/test_tracing_phase2.py` cobrindo spans, handoff event, hitl/critic, reasoning, prompt-agent e cost reconciliation

### Falta

- [ ] Cobrir a linha completa `Prompt -> Manager -> Data -> Manager -> Model -> Manager -> Critic -> Manager -> Operation -> Manager` em teste de sistema (Critic↔Manager ja coberto em `tests/test_critic_manager_integration.py` apos a Fase 7; falta integracao end-to-end com Data/Model/Operation reais)

### Estabilizacao ja feita durante a fase

- [x] Corrigir `RunContext.__slots__` para suportar `_manager_agent_id`
- [x] Adicionar `playwright` ao ambiente base
- [x] Adicionar `html2text` ao ambiente base
- [x] Implementar fallback de retrieval para `requests` quando Playwright/binarios/deps do host falharem

## Fase 3. Scheduler, concorrencia controlada e fallback para fila

Status: `concluida`

### Feito

- [x] Substituir `multiprocessing.Pool` cru por `utils.scheduler.BranchScheduler` em `agent_manager` (execucao de planos e verificacao)
- [x] Preservar `branch_id` ponta a ponta via payloads explicitos do scheduler
- [x] Suportar limites por run via `max_concurrency`
- [x] Adicionar fila serial como fallback automatico via `SchedulerFallback`
- [x] Emitir no trace `scheduler_started`, `scheduler_fallback_serial` e `scheduler_completed` com `mode` e contagem de jobs
- [x] Cobrir scheduler com `tests/test_scheduler.py` (parallel/serial/fallback/branch_id)

### Falta

- [ ] Benchmark formal documentado processo vs thread vs serial para Data/Model (decisao em ADR follow-up se aparecer regressao)
- [ ] Configuracao por provider/modelo (atualmente ha apenas o cap geral `max_concurrency`)

## Fase 4. CLI lifecycle, cancelamento gracioso e experiencia pos-run

Status: `concluida`

### Feito

- [x] Mapear cancelamento basico na CLI para status `cancelled`
- [x] Mapear excecoes nao tratadas na CLI para status `failed`
- [x] Acionar `prepare_new_run(...)` antes da execucao
- [x] Acionar `finalize_run(...)` ao fim da execucao
- [x] Mostrar `run_id` na CLI
- [x] Adicionar tratamento formal de SIGINT/SIGTERM via `utils.cli_lifecycle.install_cancellation_handler` (primeiro sinal: cancelamento gracioso; segundo sinal: escalation para `KeyboardInterrupt`)
- [x] Adicionar flags `--scheduler-mode`, `--max-concurrency` e `--cleanup-mode` no `cli run`
- [x] Adicionar subcomando `python -m cli list-runs` listando manifests de runs anteriores ordenados por data
- [x] Imprimir summary pos-run (`build_post_run_summary`) com `run_id`, status, contagem de eventos, presenca de `cost_summary`/`terminal.log` e diretorio de artefatos
- [x] Cobrir cancelamento, listagem de runs e summary com `tests/test_cli_lifecycle.py`

### Falta

- [ ] Menu pos-run interativo (abrir logs, abrir artefatos, iniciar outra run, subir interface web) — depende da Fase 14/15
- [ ] Definir comportamento de `stop after model created` ou `stop after training` — depende da Fase 5/6

## Fase 5. Constraints mais granulares e persistencia das analises

Status: `concluida`

### Feito

- [x] Evoluir `prompt_agent/schema.json` para constraints mais granulares (bloco opcional `constraints` cobrindo split, seed, framework, fairness, explainability, allowed_packages, deploy_required, concurrency, hitl, critic, cleanup, token_economy)
- [x] Adicionar `--seed`, `--framework`, `--split-policy`, `--token-economy`, `--hitl-level` ao `cli run`
- [x] Persistir constraints estruturadas em `run_manifest.json` via `utils.constraints.persist_constraints`
- [x] Salvar copia em `analyses/constraints.json` por run
- [x] Emitir evento `constraints_recorded` no ledger
- [x] Cobrir normalizacao + persistencia + schema com `tests/test_constraints_phase5.py`

### Falta

- [ ] Estender o wizard interativo (`cmd_interactive`) para coletar grupos adicionais (split policy, framework preference, fairness/explainability) sem inflar o fluxo basico
- [ ] Fazer Prompt Agent inferir o bloco `constraints` quando o texto livre menciona seed/framework/split (depende de iteracao do prompt + Critic Agent da Fase 7)
- [ ] Gravar todas as analises listadas no TASKS.md (resumo do manager, racional de planos, resumo final) — `prompt_parse`, `req_summary`, `plan_N`, `code_instruction` ja sao gravadas; restantes dependem de Critic e do refator de manager

## Fase 6. HITL estrategico

Status: `concluida`

### Feito

- [x] Definir checkpoints formais de HITL em `utils.hitl.KNOWN_CHECKPOINTS` (`after_parse`, `after_plans`, `before_critic_override`, `before_code_generation_high_risk`, `before_destructive_cleanup`, `before_deploy`, `before_final_acceptance_on_conflict`)
- [x] Implementar `utils.hitl.request_checkpoint` com politica `off`/`standard`/`strict` e modos `auto`/`human`/`human-fallback`/`policy-skipped`
- [x] Registrar `hitl_requested` + `hitl_resolved` correlacionados por `hitl_id` em todos os modos
- [x] Gate interativo antes de cleanup destrutivo (`purge`) na CLI interativa
- [x] Adicionar `--hitl-level` ao `cli run` e propagar para `prepare_new_run`
- [x] Garantir compatibilidade com modo nao interativo (modo `auto` aplica o default e ainda registra o checkpoint)
- [x] Cobrir politica + interacao + fallback com `tests/test_hitl_phase6.py`

### Falta

- [ ] Wirar checkpoints `after_parse`, `after_plans` e `before_deploy` no agent_manager (parse/plans podem ser cobertos via `critic_policy=request_hitl` da Fase 7; `before_deploy` aguarda fase de deploy oficial)

## Fase 7. Critic Agent

Status: `concluida`

### Feito

- [x] `critic_agent/__init__.py` criado com `Finding`, `review_parse`, `review_plans`, `review_handoff`, `review_instruction`, `review_execution_result` e `run_review`
- [x] Politica `off/warn/request_hitl/block` resolvida pelo gate centralizado (`_decide_action`); `block` exige severidade `error`
- [x] `request_hitl` integra com `utils.hitl.request_checkpoint` e emite `critic_warned` + `hitl_requested` + `hitl_resolved`
- [x] Reports persistidos em `exp/runs/<id>/analyses/critic/<target>__<review_id>.json`
- [x] Eventos `critic_warned` / `critic_blocked` ja existentes em `utils.ledger` reutilizados
- [x] Manager chama `review_parse` apos parse e `review_plans` apos planejamento, lendo `critic_policy` das constraints (default `warn`)
- [x] CLI `--critic-policy {off,warn,request_hitl,block}` adicionado e mergeado nas constraints estruturadas
- [x] `tests/test_critic_phase7.py` cobrindo Finding, regras por target, gate de politica e persistencia (19 testes)

### Falta

- [ ] Wirar `review_handoff`, `review_instruction` e `review_execution_result` nos pontos correspondentes do manager/operation agent (atualmente expostos como API mas nao acionados pelo runtime)
- [ ] Adicionar checks adicionais (custo anormal, pacotes ausentes, hiperparametros absurdos) conforme novos modos de falha forem observados em runs reais

## Fase 8. Economia de tokens e melhorias de eficiencia

Status: `concluida`

### Feito

- [x] Roteamento de modelos por etapa via `utils/stage_routing.py` (`LLM_STAGE_<NAME>` env vars; 6 etapas conhecidas)
- [x] Stage routing aplicado nos construtores de `PromptAgent` (`prompt_parse`) e `OperationAgent` (`code_generation`)
- [x] Truncamento/compressao de payloads e logs para prompts via `utils/token_economy.py` (`truncate_payload`, `summarize_error`)
- [x] Cache run-scoped para parse/retrieval/summaries em `utils/run_cache.py` (chave SHA-256, persistencia em `exp/runs/<id>/cache/<key>.json`)
- [x] PromptAgent.parse adota `RunCache` quando `token_economy != "off"`; cache miss/hit emitem `tokens_saved`
- [x] Orcamento dinamico de `n_plans` ligado ao policy + confidence (`dynamic_n_plans`)
- [x] Registrar custo evitado via novo evento de ledger `tokens_saved`
- [x] Operation Agent compacta stderr antes de reinjetar no proximo prompt
- [x] Cobertura: 26 testes unitarios em `tests/test_token_economy_phase8.py`, 4 testes de integracao em `tests/test_token_economy_manager_integration.py` e 7 testes de wiring em `tests/test_phase8_wiring.py`

### Falta

- [ ] Adotar `RunCache` nos retrievers de Data/Model Agent quando as buscas forem deterministicas
- [ ] Estender uso explicito de `summarize_error`/`truncate_payload` em mensagens longas reusadas entre planos
- [ ] Aplicar `dynamic_n_plans` tambem ao `n_revise` quando esse loop for reativado
- [ ] Fazer rodada dedicada de review de eficiencia comparando custos com/sem token economy

## Fase 9. Terminal log ponta a ponta e reasoning trail

Status: `parcial`

### Feito

- [x] Persistir streaming do subprocesso do Operation Agent em `terminal.log`

### Falta

- [ ] Persistir tudo que hoje sai por `print_message(...)` no `terminal.log` da run
- [ ] Carimbar timestamps e canais (`stdout`/`stderr`) no log do subprocesso
- [ ] Persistir prompts humanos da CLI e respostas do usuario quando seguro
- [ ] Persistir resumos de reasoning e decisao por agente de forma sistematica
- [ ] Redigir ou mascarar segredos antes de salvar logs
- [ ] Definir nivel formal de verbosidade entre terminal log e ledger estruturado

## Fase 10. Suite de testes, regressao e cobertura alta

Status: `parcial`

### Feito

- [x] Criar testes unitarios para `RunContext`
- [x] Criar testes unitarios para helpers de workspace por run
- [x] Criar testes unitarios para ledger, custo e analises
- [x] Criar baseline de regressao para CLI
- [x] Criar baseline de regressao para Agent Manager
- [x] Criar baseline de regressao para Prompt Agent
- [x] Criar baseline de regressao para Data/Model Agents
- [x] Criar baseline de regressao para Operation Agent
- [x] Criar baseline de regressao para tracing/configs

### Falta

- [x] Criar testes para scheduler e fallback fila/serial (`tests/test_scheduler.py`)
- [ ] Criar testes para persistencia completa de `terminal.log`
- [x] Criar testes para novos modos de constraints (`tests/test_constraints_phase5.py`)
- [x] Criar testes para HITL (`tests/test_hitl_phase6.py`)
- [x] Criar regressao para os tres modos de dataset com provenance por run (coberta em `tests/test_run_lifecycle.py`)
- [x] Criar regressao para cancelamento gracioso completo (`tests/test_cli_lifecycle.py`)
- [ ] Criar integracoes com mocks de provider para lineage completo
- [ ] Fechar lacunas de cobertura em namespace, cleanup e lineage ponta a ponta

## Fase 11. Documentacao, README, CLI reference e ADRs

Status: `parcial`

### Feito

- [x] Atualizar `docs/90_ADR_INDEX.md`
- [x] Atualizar `docs/02_ARCHITECTURE_AND_AGENTS.md` com ADRs planejados
- [x] Atualizar `docs/06_WORKSPACE_AND_DATASETS.md` com namespace por run
- [x] Atualizar `docs/07_EXECUTION_PIPELINE_AND_ARTIFACTS.md` com lifecycle por run

### Falta

- [ ] Atualizar `README.md` com run lifecycle, cleanup, tracing, custos, HITL e pos-run
- [x] Atualizar `docs/05_CLI_REFERENCE.md` com novos flags e lifecycle real (cleanup, scheduler, list-runs, post-run summary, granular constraints, hitl-level)
- [x] Atualizar `docs/08_TASK_TYPES_AND_METRICS.md` com constraints mais granulares (§6.1)
- [ ] Atualizar `docs/09_LLM_CONFIGURATION.md` com roteamento por etapa e economia de tokens
- [x] Atualizar `docs/11_TROUBLESHOOTING.md` com fallback de fila, cleanup, consulta de run, HITL e constraints
- [ ] Atualizar `docs/12_DEVELOPMENT_AND_TESTING.md` com invariantes e novas suites
- [ ] Revisar `.gitignore` caso novos artefatos precisem de regra adicional

## Fase 12. Matriz de execucao e compatibilidade de providers

Status: `pendente`

### Falta

- [ ] Formalizar matriz de compatibilidade OpenRouter/OpenAI/local-first por superficie
- [ ] Tornar provider conceito de primeira classe na UX e nos contratos
- [ ] Validar contratos dos agentes nas combinacoes suportadas
- [ ] Ajustar docs/tests/CLI para coerencia de provider

## Fase 13. Refactor estrutural do repositorio para monorepo multi-surface

Status: `pendente`

### Falta

- [ ] Mover nucleo atual para `agent/`
- [ ] Definir empacotamento instalavel de `agent/`
- [ ] Ajustar imports, entrypoints, testes e notebook
- [ ] Criar `backend/`
- [ ] Criar `frontend/`
- [ ] Revisar organizacao de `tests/`

## Fase 14. Backend minimo para consumo do agente

Status: `pendente`

### Falta

- [ ] Criar `backend/` em FastAPI
- [ ] Expor endpoints minimos de health, modelos, task types, iniciar run, status e artefatos/logs
- [ ] Proteger requests com segredo compartilhado
- [ ] Garantir que o backend usa o mesmo nucleo do agente e o mesmo `RunContext`
- [ ] Garantir suporte a dataset especifico enviado pelo usuario

## Fase 15. Frontend web minimo, mobile first e separado do backend

Status: `pendente`

### Falta

- [ ] Criar `frontend/` com React/Next
- [ ] Usar `shadcn/ui`
- [ ] Implementar fluxo minimo protegido por senha
- [ ] Expor configuracao de task/model/dataset/prompt/opcoes essenciais
- [ ] Criar painel de status/resultado da run
- [ ] Garantir responsividade e MVP sem overengineering

## Fase 16. Modos de execucao e configuracao unificados

Status: `pendente`

### Falta

- [ ] Definir oficialmente os modos de execucao suportados
- [ ] Definir matriz de configuracao por modo
- [ ] Garantir coerencia de `.env` entre modos
- [ ] Garantir dataset especifico do usuario em todos os modos suportados
- [ ] Criar documentacao operacional clara por modo

## Fase 17. Containerizacao, compose e proxy de producao

Status: `pendente`

### Falta

- [ ] Criar Dockerfile para `agent/` quando aplicavel
- [ ] Criar Dockerfile para `backend/`
- [ ] Decidir estrategia de container do frontend
- [ ] Criar `compose.dev.yml`
- [ ] Criar `compose.prod.yml`
- [ ] Configurar Traefik
- [ ] Definir `.env` e secrets por ambiente

## Fase 18. Documentacao de deploy e operacao do stack web

Status: `pendente`

### Falta

- [ ] Criar `DEPLOY.md` quando a fase de plataforma amadurecer
- [ ] Documentar deploy do frontend na Vercel
- [ ] Documentar deploy de backend/agente na VPS com Traefik
- [ ] Documentar dominios, env vars, secrets, TLS e networking
- [ ] Documentar diferencas entre dev e producao

## Fase 19. Modernizacao controlada para latest do ecossistema LangChain/LangSmith

Status: `pendente`

### Falta

- [ ] Levantar acoplamentos atuais a APIs antigas
- [ ] Definir versao alvo estavel de LangChain/LangSmith
- [ ] Revisar tracing, wrappers, embeddings e retrievers
- [ ] Garantir que LangSmith complemente, e nao substitua, o ledger local
- [ ] Atualizar requirements, docs, testes e exemplos
- [ ] Revisar reducao de legado como `langchain-classic`

## Fase 20. Tools operacionais para a rede de agentes com approvals

Status: `pendente`

### Falta

- [ ] Definir quais tools fazem sentido para a rede do AutoML-Agent
- [ ] Separar tools por nivel de risco
- [ ] Criar politica de approval para instalacao, alteracao de ambiente, limpeza e acoes destrutivas
- [ ] Registrar toda invocacao de tool no trace e no ledger
- [ ] Garantir que tools nao virem bypass de constraints, Critic Agent ou HITL
- [ ] Definir limites de instalacao de dependencia em dev/producao

## Proximo Corte Recomendado

Se a ideia for seguir a ordem natural do `TASKS.md`, o proximo bloco de trabalho mais coerente e:

1. fechar os gaps restantes da Fase 1 e da Fase 2,
2. implementar a Fase 3 (scheduler com fallback),
3. depois consolidar a Fase 4 (lifecycle de CLI mais completo).

## Atualizacao Rapida por Iteracao

Ao final de cada iteracao, atualizar pelo menos:

1. o status da fase afetada,
2. os itens de `Feito`,
3. os itens de `Falta`,
4. o resumo geral no topo, se o status da fase mudou.