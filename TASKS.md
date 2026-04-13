# TASKS

## 1. Objetivo

Este arquivo consolida as tarefas discutidas nesta sessao para a evolucao do AutoML-Agent.

O foco aqui e organizar o trabalho de forma executavel, detalhada e alinhada com o estado atual do repositorio, sem perder os invariantes ja existentes:

1. a CLI continua sendo a interface principal,
2. `agent_workspace/` continua sendo o workspace canonico,
3. Prompt Agent, Agent Manager, Data Agent, Model Agent e Operation Agent continuam separados,
4. qualquer mudanca estrutural relevante exige testes, docs e ADR.

## 2. Estado atual do repositorio relevante para estas tarefas

| Tema | Estado atual | Arquivos principais | Gap atual |
| --- | --- | --- | --- |
| Workspace | Existe somente o contrato global `agent_workspace/datasets`, `agent_workspace/exp` e `agent_workspace/trained_models` | `utils/workspace.py`, `docs/06_WORKSPACE_AND_DATASETS.md` | Nao ha namespace real por run, branch ou agente |
| Execucao por planos | Cada plano roda em `multiprocessing.Pool` | `agent_manager/__init__.py`, `docs/02_ARCHITECTURE_AND_AGENTS.md`, `docs/07_EXECUTION_PIPELINE_AND_ARTIFACTS.md` | Ha paralelismo, mas nao ha isolamento forte de contexto nem lineage por branch |
| Tracing | Existem decoradores e metadata basicos | `utils/tracing.py`, `utils/__init__.py` | Nao ha `run_id`, `agent_id`, `handoff_id`, trilha completa de eventos, nem custo consolidado por run/modelo |
| Prompt Agent | Faz parse para JSON e usa um LLM proprio | `prompt_agent/__init__.py`, `prompt_agent/schema.json` | Nao ha log estruturado do parse, nem controles granulares de HITL |
| Constraints | A CLI atual so coleta modelo, metrica, alvo, treino e inferencia | `cli.py`, `experiments/constraint_prompts.py`, `prompt_agent/schema.json` | Faltam constraints granulares de reproducibilidade, recursos, deploy, criticidade e politicas operacionais |
| Data entry modes | Ha tres modos: upload/local, link e auto-retrieval | `cli.py`, `data_agent/retriever.py`, `docs/06_WORKSPACE_AND_DATASETS.md` | Falta provenance por run e politica especifica por modo dentro de um namespace isolado |
| Operation Agent | Gera script, roda subprocesso, streama stdout/stderr | `operation_agent/__init__.py`, `operation_agent/execution.py` | O terminal nao fica persistido ponta a ponta em `agent_workspace/exp` |
| Custos | Alguns agentes guardam `usage` em `self.money` | `agent_manager/__init__.py`, `data_agent/__init__.py`, `model_agent/__init__.py`, `operation_agent/__init__.py` | O consolidado atual nao e confiavel para auditoria completa por run/modelo |
| CLI lifecycle | Existe wizard interativo e `run` nao interativo | `cli.py`, `docs/05_CLI_REFERENCE.md` | Nao ha menu pos-run, cancelamento gracioso formal, limpeza orquestrada de workspace, nem retomada/consulta de runs |
| HITL | Existe apenas interatividade pontual do manager em alguns estados | `cli.py`, `agent_manager/__init__.py` | Nao ha checkpoints estrategicos, aprovacoes formais nem trilha de decisoes humanas |
| Critic / review | Existe verificacao LLM antes e depois da execucao | `agent_manager/__init__.py`, `docs/07_EXECUTION_PIPELINE_AND_ARTIFACTS.md` | Nao existe um agente critico dedicado para pegar burradas no meio do fluxo |
| Providers e runtime | O fluxo atual privilegia OpenRouter e tem caminho local para Prompt Agent via vLLM | `configs.py`, `cli.py`, `utils/__init__.py`, `README.md`, `docs/09_LLM_CONFIGURATION.md` | Falta matriz formal de compatibilidade entre OpenAI, OpenRouter e local-first sem quebrar CLI, testes e agentes |
| Topologia do repositorio | O projeto atual esta todo na raiz, sem separacao clara entre `agent`, `backend` e `frontend` | raiz do repositorio, `README.md`, `tests/` | Falta estrutura modular para produto multi-surface e deploy limpo |
| Superficies de uso | Hoje a superficie principal e CLI, com uso programatico opcional; existe notebook no repo | `cli.py`, `__main__.py`, `AutoMLAgent.ipynb`, `README.md`, `docs/05_CLI_REFERENCE.md` | Falta estrategia unificada para rodar via venv, container, web e notebook |
| Web/API | Nao existe backend HTTP nem frontend para consumo do agente | inexistente | Falta backend minimo protegido, frontend simples e contratos de API |
| Containerizacao e deploy | Nao existem Dockerfiles, composes separados ou reverse proxy de producao | inexistente | Falta padrao de empacotamento local e de VPS para `agent` + `backend` |
| Observabilidade externa | Existe wrapper basico, mas nao ha integracao madura com LangSmith nem alinhamento ao latest do ecossistema LangChain | `utils/tracing.py`, `utils/__init__.py`, `requirements.txt`, `requirements-local.txt` | Falta modernizacao planejada e padronizada de observabilidade |
| Ferramentas dos agentes | Os agentes produzem texto e codigo, mas nao possuem uma camada propria de tools com approvals operacionais | `agent_manager/__init__.py`, `operation_agent/`, `utils/` | Falta politica de tools, instalacao de dependencias em run, approvals e guardrails |
| Testes | Ha cobertura de workspace, CLI, configs e algumas integracoes | `tests/`, `pytest.ini`, `docs/12_DEVELOPMENT_AND_TESTING.md` | Nao ha suite robusta para namespace, tracing, lineage, fila, HITL, critic agent, custos e logs ponta a ponta |

## 3. Decisoes de projeto que precisam ficar travadas antes da implementacao

Estas decisoes devem ser formalizadas em ADR antes da codificacao pesada, porque afetam arquitetura, docs e testes.

### 3.1 Modelo de identidade operacional

Padronizar cinco IDs distintos:

1. `trace_id`: arvore global observavel da execucao,
2. `run_id`: identificador unico da run do usuario,
3. `branch_id`: identificador da linha de execucao de um plano/solucao,
4. `agent_id`: identificador unico da instancia do agente,
5. `handoff_id`: identificador unico de cada repasse entre agentes.

Padrao recomendado:

1. `run_id` gerado no inicio da run,
2. `branch_id` derivado por plano e mantido ate o final,
3. `agent_id` criado por instancia concreta (`prompt`, `manager`, `data`, `model`, `operation`, `critic`),
4. `handoff_id` criado sempre que um payload muda de dono.

### 3.2 Politica de limpeza do workspace antes de nova run

Ha um conflito real entre:

1. limpar o `agent_workspace` antes de uma nova run,
2. manter historico completo consultavel,
3. permitir multiplas runs isoladas.

Resolucao recomendada:

1. manter historico em namespaces imutaveis por run,
2. limpar somente a area ativa ou scratch antes da nova run,
3. oferecer uma opcao explicita de limpeza destrutiva total apenas com confirmacao/HITL.

Proposta de layout sem quebrar o contrato canonico:

```text
agent_workspace/
├── datasets/
│   ├── cache/
│   └── runs/<run_id>/
├── exp/
│   └── runs/<run_id>/
└── trained_models/
    └── runs/<run_id>/
```

Se o repositorio precisar de uma area ativa sem namespace para compatibilidade transitoria, ela deve ser tratada como alias temporario, nunca como fonte de verdade.

### 3.3 Visibilidade de thinking

Nao depender de chain-of-thought bruto do provider.

Padrao recomendado:

1. registrar prompts, outputs, resumos de decisao e reasoning summary quando o provider expuser,
2. registrar `reasoning_tokens` quando existirem,
3. persistir um `reasoning trail` auditavel por eventos,
4. nao bloquear a funcionalidade caso o provider nao entregue pensamento interno completo.

### 3.4 Politica de concorrencia e fallback

Padrao recomendado:

1. tentar concorrencia controlada por branch e por modelo,
2. limitar chamadas por provider/modelo com semaforos configuraveis,
3. ao detectar bloqueio, 429, rate limit severo ou instabilidade do provider, cair para fila serial,
4. manter a mesma linha `branch_id -> agent_id -> handoff_id` mesmo apos fallback.

### 3.5 Politica de HITL

Definir niveis formais de HITL:

1. `off`: zero checkpoints humanos extras,
2. `standard`: checkpoints nas partes mais perigosas,
3. `strict`: checkpoints antes de cada etapa sensivel.

### 3.6 Politica do Critic Agent

Definir se o agente critico:

1. roda sempre,
2. roda apenas quando ha risco alto,
3. roda apenas quando o usuario habilita,
4. pode bloquear a execucao sozinho ou apenas recomendar HITL.

### 3.7 Politica de compatibilidade de provedores e modos de execucao

Definir como o projeto se comporta oficialmente em cada combinacao abaixo:

1. OpenRouter API-first,
2. OpenAI API-first,
3. local-first com modelos locais configurados,
4. execucao por CLI,
5. execucao por container,
6. execucao via backend web,
7. execucao via notebook.

Padrao recomendado:

1. manter uma matriz de suporte explicita e testavel,
2. garantir que a escolha do provider nao altere semanticamente o comportamento dos agentes,
3. garantir que fallback entre providers seja uma capacidade explicitamente configurada,
4. garantir que os testes deixem claro o que e suportado oficialmente e o que e opcional.

### 3.8 Politica de topologia do monorepo

Definir o desenho alvo do repositorio para suportar produto multi-surface sem ambiguidade.

Padrao recomendado:

```text
automl-agent/
├── agent/
├── backend/
├── frontend/
├── docs/
├── tests/
└── TASKS.md
```

Decisoes que precisam ser travadas:

1. se `tests/` fica na raiz ou e repartido por pacote,
2. se `agent/` sera empacotado como package instalavel,
3. se `backend/` importa `agent/` via `pip install -e` ou via workspace compartilhado,
4. como o notebook referencia `agent/` sem hacks fragilizados.

### 3.9 Politica de seguranca para superficie web

Definir o baseline de seguranca do MVP web.

Padrao recomendado:

1. frontend simples e minimalista,
2. backend FastAPI minimo com autenticacao por segredo compartilhado no MVP,
3. recusa imediata no backend quando o cabecalho de autenticacao estiver incorreto,
4. separacao clara entre segredo do backend e configuracoes publicas do frontend,
5. logs e traces jamais devem vazar segredo.

### 3.10 Politica de containerizacao e deploy

Definir o baseline de runtime para dev e producao.

Padrao recomendado:

1. `compose.dev.yml` para desenvolvimento local,
2. `compose.prod.yml` para VPS com `agent` + `backend` + Traefik,
3. frontend servido separadamente em producao quando a estrategia escolhida for Vercel,
4. containers sempre opcionais em relacao ao fluxo por venv local,
5. documentar claramente o que roda em container e o que roda fora dele.

### 3.11 Politica de tools e approvals na rede de agentes

Definir se e como os agentes podem usar tools operacionais.

Padrao recomendado:

1. permitir tools controladas para diagnostico e execucao local,
2. qualquer acao destrutiva ou instalacao de dependencia exige approval,
3. diferenciar ferramentas permitidas em desenvolvimento versus producao,
4. registrar toda acao de tool no trace e no ledger,
5. nunca permitir instalacao ad hoc sem contexto, justificativa e trilha.

### 3.12 Politica de observabilidade definitiva com LangChain e LangSmith

Definir se o stack final de observabilidade sera baseado em LangSmith como camada principal.

Padrao recomendado:

1. usar LangSmith como ferramenta definitiva de trace externo,
2. manter ledger local da run como fonte de auditoria operacional,
3. atualizar a codebase para o latest estavel relevante de LangChain/LangSmith apenas apos endurecer os contratos internos,
4. tratar a atualizacao como refactor controlado, com testes e migracoes por modulo.

## 4. Fases de implementacao recomendadas

## Fase 0. Preparacao, baseline e ADRs

### Objetivo

Preparar o repositorio para mudancas grandes sem quebrar o comportamento atual no meio do caminho.

### Tarefas

1. Criar ADR para namespace por run, lineage de handoff, custo consolidado e politica de cleanup.
2. Criar ADR para scheduler com fallback fila/serial.
3. Criar ADR para HITL e Critic Agent.
4. Congelar o comportamento atual com testes de regressao antes de mexer na arquitetura.
5. Usar `bkp/run-01/terminal.log` e os artefatos reais em `agent_workspace/exp/` como referencia de migracao para novos logs e manifests.

### Arquivos principais

1. `docs/90_ADR_INDEX.md`
2. `docs/adr/`
3. `tests/`
4. `README.md`
5. `docs/02_ARCHITECTURE_AND_AGENTS.md`
6. `docs/07_EXECUTION_PIPELINE_AND_ARTIFACTS.md`

### Criterios de aceite

1. Os novos ADRs deixam claro o modelo de IDs, cleanup, lineage e fallback.
2. Existe baseline de testes para provar que a migracao nao quebra o fluxo CLI atual antes da troca maior.

## Fase 1. Run lifecycle, namespace e limpeza controlada do workspace

### Objetivo

Eliminar colisao entre runs e criar um ciclo de vida formal de run.

### Tarefas

1. Introduzir um `RunContext` serializavel com `run_id`, `branch_id`, `agent_id`, `attempt_id`, `started_at`, `status`, `task_type`, `llm_backbone`, `prompt_llm` e politica de HITL.
2. Criar helpers de path por run em `utils.workspace` sem remover o contrato canonico atual.
3. Criar rotina `prepare_new_run(...)` para:
   - validar se existe run ativa,
   - arquivar ou preservar historico,
   - limpar area scratch/ativa,
   - opcionalmente manter cache de datasets remotos,
   - emitir evento de cleanup no trace.
4. Criar rotina `finalize_run(...)` para:
   - marcar run como `completed`, `failed` ou `cancelled`,
   - fechar manifests,
   - fechar logs,
   - liberar locks/filas.
5. Remover dependencia de `code_path` baseado apenas em string montada no manager e passar a usar diretorios namespaceados por run.
6. Garantir que multiplas runs possam coexistir sem escrever nos mesmos arquivos.
7. Definir politica explicita para `datasets/cache` versus `datasets/runs/<run_id>`.

### Arquivos principais

1. `utils/workspace.py`
2. `agent_manager/__init__.py`
3. `operation_agent/__init__.py`
4. `data_agent/retriever.py`
5. `cli.py`
6. `docs/06_WORKSPACE_AND_DATASETS.md`
7. `docs/07_EXECUTION_PIPELINE_AND_ARTIFACTS.md`

### Proposta de comportamento por modo de dataset

1. Upload/manual:
   - copiar ou linkar o arquivo fornecido para o namespace da run,
   - registrar provenance `manual-upload`,
   - guardar checksum e origem local.
2. URL informada pelo usuario:
   - baixar para cache estavel por URL,
   - anexar essa referencia ao namespace da run,
   - registrar provenance `user-link`.
3. Auto-retrieval pelo agente:
   - registrar fonte tentada, fonte escolhida e justificativa,
   - salvar metadados do processo de busca,
   - registrar provenance `auto-retrieval`.

### Criterios de aceite

1. Duas runs do mesmo task type nao colidem em `exp`, `datasets` ou `trained_models`.
2. A politica de cleanup nao apaga historico consultavel por padrao.
3. O usuario consegue iniciar nova run sem carregar sujeira de uma run anterior.

## Fase 2. Tracing robusto, lineage, logs e custos

### Objetivo

Tornar cada transicao observavel, auditavel e mensuravel.

### Tarefas

1. Evoluir `utils/tracing.py` para suportar spans e eventos de orquestracao, nao apenas metadata basica.
2. Criar manifest por run em `agent_workspace/exp/runs/<run_id>/run_manifest.json`.
3. Criar ledger de eventos em `agent_workspace/exp/runs/<run_id>/events.jsonl`.
4. Criar ledger de handoffs em `agent_workspace/exp/runs/<run_id>/handoffs.jsonl`.
5. Criar `cost_summary.json` e `cost_records.jsonl` por run.
6. Criar `terminal.log` ponta a ponta por run.
7. Criar `analyses/` por run para armazenar parse, summaries, reviews, critic reports, revisoes e notas de HITL.
8. Registrar eventos minimos:
   - `run_started`
   - `run_cleanup_started`
   - `run_cleanup_completed`
   - `agent_started`
   - `agent_finished`
   - `handoff_emitted`
   - `handoff_received`
   - `manager_waiting`
   - `manager_received`
   - `llm_call_completed`
   - `artifact_written`
   - `hitl_requested`
   - `hitl_resolved`
   - `critic_blocked`
   - `critic_warned`
   - `run_cancelled`
   - `run_failed`
   - `run_completed`
9. Registrar para cada evento:
   - timestamp,
   - `trace_id`, `run_id`, `branch_id`, `agent_id`, `handoff_id`,
   - evento,
   - origem,
   - destino,
   - resumo do payload,
   - referencia do payload salvo,
   - tamanho do payload,
   - hash do payload.
10. Registrar custo por chamada LLM com:
   - provider,
   - alias,
   - slug do modelo,
   - fase,
   - prompt tokens,
   - completion tokens,
   - total tokens,
   - reasoning tokens, se houver,
   - custo estimado ou retornado pelo provider.
11. Consolidar custo por modelo e por run no final.
12. Registrar no trace quando o manager recebe algo de Prompt Agent, Data Agent, Model Agent, Critic Agent e Operation Agent.

### Arquivos principais

1. `utils/tracing.py`
2. `utils/__init__.py`
3. `agent_manager/__init__.py`
4. `prompt_agent/__init__.py`
5. `data_agent/__init__.py`
6. `model_agent/__init__.py`
7. `operation_agent/__init__.py`
8. `operation_agent/execution.py`

### Requisito explicito desta sessao que esta coberto aqui

1. `run_id` unico.
2. `agent_id` unico.
3. rastrear quem iniciou, terminou e para quem passou.
4. rastrear quando o manager esta aguardando e quando recebeu algo.
5. custo por modelo e por run.
6. salvar log do terminal ponta a ponta em `/exp`.
7. salvar analises intermediarias.
8. reasoning trail observavel sem depender de chain-of-thought bruto.

### Criterios de aceite

1. E possivel reconstruir a linha completa `Prompt -> Manager -> Data -> Manager -> Model -> Manager -> Critic -> Manager -> Operation -> Manager` para cada branch.
2. Existe um log textual bruto e um ledger estruturado por run.
3. O custo final bate com a soma das chamadas LLM registradas.

## Fase 3. Scheduler, concorrencia controlada e fallback para fila

### Objetivo

Manter paralelismo quando fizer sentido, mas com degradacao segura para fila serial quando a API bloquear.

### Tarefas

1. Substituir o uso cru de `multiprocessing.Pool` por um scheduler de branches com controle explicito.
2. Preservar `branch_id` como a identidade da linha de execucao ponta a ponta.
3. Configurar limites por provider/modelo.
4. Adicionar fila serial como fallback automatico.
5. Registrar no trace quando o sistema caiu para fila serial e por qual motivo.
6. Avaliar se a parte de Data/Model continua em processos separados, threads ou execucao serial controlada; a decisao deve sair de ADR e benchmark basico.
7. Garantir que o consolidado de custos, tempos e eventos nao dependa de estado mutavel dentro do worker process.

### Arquivos principais

1. `agent_manager/__init__.py`
2. `utils/tracing.py`
3. `utils/__init__.py`
4. possivel novo modulo `utils/scheduler.py` ou equivalente

### Criterios de aceite

1. O sistema tenta concorrencia quando habilitado.
2. Em caso de bloqueio de API, o sistema cai para fila sem corromper a run.
3. A linha `data_agent_N -> model_agent_N` continua observavel e consistente mesmo apos fallback.

## Fase 4. CLI lifecycle, cancelamento gracioso e experiencia pos-run

### Objetivo

Eliminar dependencia de `KeyboardInterrupt` como fim informal de execucao e dar ao usuario um fluxo limpo antes, durante e depois da run.

### Tarefas

1. Adicionar tratamento formal de sinais e cancelamento gracioso.
2. Mapear cancelamento do usuario para status `cancelled`, nao `failed`.
3. Antes de nova run, acionar `prepare_new_run(...)` com politica de cleanup configuravel.
4. Adicionar flags e opcoes de CLI para:
   - limpar area ativa,
   - preservar cache,
   - consultar runs anteriores,
   - escolher nivel de HITL,
   - escolher nivel do Critic Agent,
   - controlar concorrencia maxima,
   - forcar modo fila/serial,
   - reabrir menu ao final da run.
5. Adicionar menu pos-run para:
   - abrir logs da run,
   - abrir artefatos,
   - iniciar outra run,
   - subir interface web, se disponivel,
   - encerrar graciosamente.
6. Definir comportamento para `stop after model created` ou `stop after training`, caso o usuario queira terminar o pipeline mais cedo e ainda assim reabrir a CLI.
7. Garantir que qualquer encerramento fique anotado nos artefatos da run e no trace.

### Arquivos principais

1. `cli.py`
2. `__main__.py`
3. `agent_manager/__init__.py`
4. `utils/tracing.py`
5. `docs/05_CLI_REFERENCE.md`
6. `docs/11_TROUBLESHOOTING.md`

### Criterios de aceite

1. O usuario nao precisa mais usar `Ctrl+C` como caminho principal de saida.
2. O cancelamento do usuario nao aparece como erro operacional no trace.
3. A CLI oferece caminho claro para nova run, consulta e saida limpa.

## Fase 5. Constraints mais granulares e persistencia das analises

### Objetivo

Expandir o contrato de requisitos para reduzir ambiguidade e melhorar a qualidade das decisoes.

### Tarefas

1. Evoluir `prompt_agent/schema.json` para suportar constraints mais granulares.
2. Evoluir a etapa 5 da CLI para coletar novos grupos de constraints.
3. Persistir essas constraints de forma estruturada no `run_manifest.json`.
4. Salvar analises derivadas dessas constraints em `analyses/`.

### Constraints novas recomendadas

1. estrategia de split (`train/val/test`, k-fold, group split, time split),
2. seed e nivel de reproducibilidade,
3. budget de treino, inferencia, memoria e armazenamento,
4. limite de pacotes permitidos,
5. preferencia de framework (`sklearn`, `xgboost`, `lightgbm`, `pytorch` etc.),
6. exigencia de explicabilidade,
7. exigencia de fairness ou validacoes especificas,
8. exigencia de deploy/web UI,
9. exigencia de salvar artefatos especificos,
10. politica de concorrencia,
11. politica de HITL,
12. politica do Critic Agent,
13. politica de cleanup,
14. nivel de agressividade de economia de tokens.

### Analises que devem ser anotadas

1. parse estruturado do Prompt Agent,
2. resumo do manager,
3. racional de selecao de planos,
4. resumo de Data Agent,
5. resumo de Model Agent,
6. relatorio do Critic Agent,
7. resumo de revisoes,
8. decisoes humanas de HITL,
9. resumo final da run.

### Arquivos principais

1. `prompt_agent/schema.json`
2. `prompt_agent/__init__.py`
3. `cli.py`
4. `agent_manager/__init__.py`
5. `experiments/constraint_prompts.py`
6. `docs/08_TASK_TYPES_AND_METRICS.md`
7. `README.md`

### Criterios de aceite

1. As novas constraints entram no fluxo sem quebrar a CLI atual.
2. O manager, os agentes e o Critic Agent recebem contexto suficiente para detectar violacoes cedo.
3. As analises ficam persistidas por run.

## Fase 6. HITL estrategico

### Objetivo

Adicionar checkpoints humanos nas partes onde correcoes precoces evitam perda de tempo, custo e erro acumulado.

### Tarefas

1. Definir pontos formais de HITL.
2. Registrar cada checkpoint no ledger e no terminal log.
3. Persistir a decisao humana e o contexto apresentado ao humano.

### Pontos recomendados de HITL

1. apos parse do Prompt Agent,
2. apos geracao dos planos,
3. antes de executar um plano rejeitado pelo Critic Agent,
4. antes da geracao de codigo quando houver alto risco,
5. antes de limpeza destrutiva do workspace,
6. antes de subir interface web ou expor demo,
7. antes de aceitar resultado final quando houver conflito entre verificadores.

### Arquivos principais

1. `cli.py`
2. `agent_manager/__init__.py`
3. `utils/tracing.py`
4. `docs/05_CLI_REFERENCE.md`
5. `docs/07_EXECUTION_PIPELINE_AND_ARTIFACTS.md`

### Criterios de aceite

1. O usuario consegue corrigir equivocos em pontos estrategicos.
2. Toda decisao humana fica auditavel.
3. HITL nao quebra o modo nao interativo; nesse caso ele deve ser desligado ou preconfigurado.

## Fase 7. Critic Agent

### Objetivo

Criar um agente critico dedicado para detectar burradas no meio do fluxo antes que elas virem codigo ruim ou execucao desperdicada.

### Tarefas

1. Introduzir um Critic Agent ou modo equivalente de critica estruturada.
2. Fazer o Critic Agent revisar:
   - parse do prompt,
   - planos do manager,
   - handoff Data -> Model,
   - instrucao final para Operation Agent,
   - resultados de execucao suspeitos.
3. Fazer o Critic Agent checar:
   - violacao de constraints,
   - caminho de arquivo errado,
   - pacote nao instalado,
   - inconsistencias entre dataset e task type,
   - metricas irrelevantes,
   - hiperparametros absurdos,
   - alucinacao de artefatos,
   - falha de lineage,
   - custo anormal.
4. Definir se o Critic Agent apenas alerta, solicita HITL ou bloqueia.
5. Salvar critic reports em `analyses/critic/`.

### Arquivos principais

1. possivel novo modulo `critic_agent/` ou `agent_manager/critic.py`
2. `agent_manager/__init__.py`
3. `utils/tracing.py`
4. `docs/02_ARCHITECTURE_AND_AGENTS.md`
5. `docs/07_EXECUTION_PIPELINE_AND_ARTIFACTS.md`

### Criterios de aceite

1. O fluxo consegue interromper ou sinalizar solucoes obviamente ruins mais cedo.
2. O relatorio do Critic Agent e consultavel depois da run.
3. O manager sabe registrar quando seguiu ou ignorou a critica.

## Fase 8. Economia de tokens e melhorias de eficiencia

### Objetivo

Reduzir custo e latencia sem piorar a qualidade do resultado.

### Tarefas

1. Criar politica de roteamento de modelos por etapa:
   - Prompt Agent com modelo mais barato,
   - Critic Agent com modelo mais barato quando possivel,
   - backbone forte apenas para planejamento, verificacoes complexas e revisoes criticas.
2. Reusar artefatos e resumos em vez de reenviar payloads enormes.
3. Fazer cache de parse, knowledge retrieval e summaries quando a entrada for identica.
4. Reduzir repeticao de contexto entre manager, data, model e critic.
5. Introduzir orcamento dinamico de `n_plans` e `n_revise`.
6. Registrar custo evitado por cache e por fallback.
7. Definir politicas de truncamento e compressao de logs longos em prompts.
8. Adicionar uma rodada especifica de review de eficiencia apos a base de tracing estar pronta.

### Melhorias concretas recomendadas

1. nao reenviar o texto completo de `data_result` e `model_result` para todos os passos seguintes; usar resumos e referencias de artefatos,
2. nao reenviar continuamente contexto inteiro de chat quando nao for necessario,
3. usar `Prompt Agent` e `Critic Agent` mais baratos por padrao,
4. evitar chamar retrieval externo em duplicidade dentro da mesma run,
5. resumir erros de execucao antes de reenviar ao modelo,
6. reduzir `n_plans` automaticamente quando a confianca do problema for alta e o budget for apertado.

### Arquivos principais

1. `agent_manager/__init__.py`
2. `prompt_agent/__init__.py`
3. `data_agent/__init__.py`
4. `model_agent/__init__.py`
5. `operation_agent/__init__.py`
6. `utils/tracing.py`
7. `configs.py`
8. `docs/09_LLM_CONFIGURATION.md`

### Criterios de aceite

1. O custo total por run cai ou pelo menos fica previsivel.
2. Ha evidencia clara de onde os tokens estao sendo gastos.
3. O fluxo consegue usar modelos menores onde o risco for baixo.

## Fase 9. Terminal log ponta a ponta e reasoning trail

### Objetivo

Permitir consulta completa da run depois que ela termina.

### Tarefas

1. Persistir tudo que hoje vai para `print_message(...)` no `terminal.log` da run.
2. Persistir o streaming do subprocesso executado pelo Operation Agent com carimbo de tempo e canal (`stdout`/`stderr`).
3. Persistir prompts humanos da CLI e respostas dadas pelo usuario quando isso nao expuser segredo sensivel.
4. Persistir resumos de reasoning e decisao por agente.
5. Redigir ou mascarar segredos do ambiente antes de salvar logs.
6. Definir qual nivel de verbosidade entra no terminal log e qual entra apenas no ledger estruturado.

### Arquivos principais

1. `utils/__init__.py`
2. `operation_agent/execution.py`
3. `cli.py`
4. `agent_manager/__init__.py`
5. `utils/tracing.py`

### Criterios de aceite

1. Uma run finalizada pode ser auditada sem depender da memoria do terminal ao vivo.
2. Existe um `terminal.log` por run dentro de `agent_workspace/exp/...`.
3. O reasoning trail e suficientemente bom para explicar o fluxo mesmo sem chain-of-thought bruto.

## Fase 10. Suite de testes, regressao e cobertura alta

### Objetivo

Garantir que essas mudancas grandes nao quebrem o repositorio nem criem regressao silenciosa.

### Tarefas

1. Criar testes unitarios para `RunContext`, helpers de workspace e politicas de cleanup.
2. Criar testes unitarios para tracing, ledger, cost accounting e handoffs.
3. Criar testes para scheduler e fallback fila/serial.
4. Criar testes para persistencia de `terminal.log`.
5. Criar testes para os novos modos de constraints.
6. Criar testes para HITL e criticidade.
7. Criar testes de regressao do fluxo atual da CLI.
8. Criar testes de regressao para os tres modos de dataset.
9. Criar testes de regressao para cancelamento gracioso.
10. Criar integracoes com mocks de provider para lineage completo sem depender de API real.
11. Garantir que testes de integracao com APIs reais facam skip limpo quando faltarem chaves.

### Arquivos de teste recomendados

1. `tests/test_workspace_cleanup.py`
2. `tests/test_run_context.py`
3. `tests/test_tracing.py`
4. `tests/test_handoffs.py`
5. `tests/test_cost_accounting.py`
6. `tests/test_scheduler.py`
7. `tests/test_cli_lifecycle.py`
8. `tests/test_hitl.py`
9. `tests/test_critic_agent.py`
10. `tests/test_terminal_logging.py`
11. `tests/test_dataset_provenance.py`
12. `tests/test_regression_current_workflow.py`

### Casos obrigatorios de regressao

1. mesma URL reutiliza cache sem colidir com outra run,
2. upload manual entra no namespace correto,
3. auto-retrieval registra provenance,
4. `data_agent_1` nao entrega payload para `model_agent_2`,
5. fallback para fila nao perde lineage,
6. custo por run bate com soma dos custos por modelo,
7. cancelamento do usuario gera `cancelled`, nao `failed`,
8. `terminal.log` contem CLI + agentes + subprocesso,
9. cleanup padrao nao destrui historico,
10. limpeza destrutiva so acontece com confirmacao adequada.

### Criterios de aceite

1. A suite cobre os novos contratos e os caminhos antigos criticos.
2. Ha regressao explicita para os bugs e riscos discutidos nesta sessao.
3. A cobertura das novas areas criticas e alta, especialmente tracing, namespace, cleanup e lineage.

## Fase 11. Documentacao, README, CLI reference e ADRs

### Objetivo

Manter a documentacao sincronizada com o comportamento real do projeto.

### Tarefas

1. Atualizar `README.md` com o novo fluxo de run, cleanup, tracing, custos, HITL e pos-run.
2. Atualizar `docs/02_ARCHITECTURE_AND_AGENTS.md` com Critic Agent, lineage, scheduler e novos contracts.
3. Atualizar `docs/05_CLI_REFERENCE.md` com novos flags, cleanup, cancelamento e menu pos-run.
4. Atualizar `docs/06_WORKSPACE_AND_DATASETS.md` com namespace por run, cache, provenance e cleanup.
5. Atualizar `docs/07_EXECUTION_PIPELINE_AND_ARTIFACTS.md` com logs, manifests, handoffs e custos.
6. Atualizar `docs/08_TASK_TYPES_AND_METRICS.md` com constraints mais granulares.
7. Atualizar `docs/09_LLM_CONFIGURATION.md` com roteamento por etapa e economia de tokens.
8. Atualizar `docs/11_TROUBLESHOOTING.md` com fallback de fila, cancelamento, cleanup e consulta de run.
9. Atualizar `docs/12_DEVELOPMENT_AND_TESTING.md` com novos testes e invariantes.
10. Atualizar `docs/90_ADR_INDEX.md` com os ADRs novos.
11. Atualizar `.gitignore` se novos artefatos de run precisarem ser explicitamente ignorados.

### Criterios de aceite

1. Toda mudanca de comportamento publico vem acompanhada da doc correspondente.
2. A documentacao explica claramente onde o usuario encontra logs, custos, analyses e manifests.

## Fase 12. Matriz de execucao e compatibilidade de providers

### Objetivo

Garantir, de forma explicita e testavel, que o sistema funciona de maneira previsivel em OpenAI, OpenRouter e local-first, sem quebrar agentes, testes ou formas de uso.

### Tarefas

1. Formalizar uma matriz de compatibilidade com combinacoes de:
   - OpenRouter API-first,
   - OpenAI API-first,
   - local-first com modelos locais configurados,
   - Prompt Agent local versus remoto,
   - CLI, backend web e notebook.
2. Tornar a selecao de provider um conceito de primeira classe, nao um comportamento incidental de env vars.
3. Verificar se todos os agentes funcionam nas combinacoes suportadas sem divergencia de contrato.
4. Garantir que o fluxo local-first reflita corretamente as configuracoes em `.env`, docs e testes.
5. Definir quais combinacoes sao oficialmente suportadas e quais sao best effort.
6. Expandir os testes para provar que mudar provider nao quebra:
   - parse do Prompt Agent,
   - planejamento,
   - retrieval,
   - implementacao,
   - tracing,
   - custos.
7. Corrigir a UX da CLI para que OpenAI e OpenRouter sejam tratados de forma coerente quando suportados.

### Arquivos principais

1. `configs.py`
2. `cli.py`
3. `utils/__init__.py`
4. `prompt_agent/__init__.py`
5. `agent_manager/__init__.py`
6. `README.md`
7. `docs/03_SETUP_AND_ENVIRONMENT.md`
8. `docs/09_LLM_CONFIGURATION.md`
9. `tests/test_configs.py`
10. `tests/test_cli.py`
11. `tests/test_integration.py`

### Criterios de aceite

1. Existe uma matriz de suporte explicita por provider e superficie de uso.
2. Nenhum agente muda de contrato silenciosamente ao trocar provider.
3. Os testes deixam claro o que deve continuar funcionando em cada modo.

## Fase 13. Refactor estrutural do repositorio para monorepo multi-surface

### Objetivo

Preparar o repositorio para suportar `agent`, `backend` e `frontend` de forma limpa, mantendo o nucleo do AutoML-Agent isolado e reutilizavel.

### Tarefas

1. Mover todo o nucleo atual para uma pasta `agent/`.
2. Definir se `agent/` sera pacote Python instalavel e padronizar esse caminho.
3. Ajustar imports, entrypoints, testes e notebook para a nova topologia.
4. Criar pasta `backend/` para a API minima.
5. Criar pasta `frontend/` para a interface web.
6. Revisar como `tests/` se organiza apos a separacao.
7. Garantir que o repositorio continue rodando por venv local sem container e sem depender do frontend.

### Arquivos principais

1. raiz do repositorio
2. `__main__.py`
3. `cli.py`
4. `tests/`
5. `README.md`
6. `AutoMLAgent.ipynb`
7. `docs/12_DEVELOPMENT_AND_TESTING.md`

### Criterios de aceite

1. A topologia final separa claramente nucleo, backend e frontend.
2. O uso por CLI local continua funcionando apos o refactor.
3. O notebook continua utilizavel com o novo layout.

## Fase 14. Backend minimo para consumo do agente

### Objetivo

Criar uma API minima, segura e simples para expor o agente por HTTP sem reimplementar a logica central.

### Tarefas

1. Criar `backend/` em FastAPI.
2. Expor endpoints minimos para:
   - health,
   - listar modelos,
   - listar task types,
   - iniciar run,
   - consultar status de run,
   - consultar artefatos/logs de run.
3. Proteger toda requisicao com segredo compartilhado no MVP.
4. Validar o segredo no cabecalho e rejeitar imediatamente quando estiver incorreto.
5. Garantir que o backend encaminha requests ao `agent/` respeitando o mesmo `RunContext`, tracing, lineage e politicas de cleanup.
6. Garantir suporte a dataset especifico enviado pelo usuario e pedido feito em cima dele.
7. Garantir que o backend nao cria uma segunda fonte de verdade para regras de negocio; ele so orquestra o `agent/`.

### Arquivos principais

1. `backend/`
2. `agent/`
3. `.env.example`
4. `README.md`
5. `docs/05_CLI_REFERENCE.md`
6. documentacao futura de API, se criada

### Criterios de aceite

1. O backend usa o mesmo nucleo do agente, nao uma copia paralela da logica.
2. Toda requisicao protegida falha rapido quando o segredo estiver invalido.
3. Dataset especifico fornecido pelo usuario continua sendo suportado corretamente.

## Fase 15. Frontend web minimo, mobile first e separado do backend

### Objetivo

Oferecer uma interface web simples, minimalista e responsiva com a mesma praticidade essencial da CLI.

### Tarefas

1. Criar `frontend/` com React/Next.
2. Usar `shadcn/ui` como base de componentes.
3. Priorizar mobile first e responsividade web sem excesso de complexidade.
4. Implementar fluxo minimo com:
   - acesso protegido por senha,
   - selecao de task type,
   - selecao/configuracao de modelo,
   - dataset upload ou referencia de dataset,
   - prompt do usuario,
   - opcoes essenciais equivalentes a CLI,
   - painel de status/resultado da run.
5. Garantir que o frontend consome o backend protegido e nunca conhece mais segredo do que o necessario para o MVP.
6. Garantir que a interface nao desvia do escopo de MVP com overengineering.

### Arquivos principais

1. `frontend/`
2. `backend/`
3. `.env.example`
4. `README.md`
5. documentacao futura de frontend/deploy

### Criterios de aceite

1. O frontend oferece o essencial da CLI com friccao minima.
2. A UX e valida em mobile e desktop.
3. O frontend continua um cliente do backend, nao reimplementa logica do agente.

## Fase 16. Modos de execucao e configuracao unificados

### Objetivo

Documentar e padronizar todas as formas de rodar o produto sem comportamento divergente.

### Tarefas

1. Definir oficialmente os modos de execucao:
   - CLI por venv,
   - CLI por container,
   - backend por venv,
   - backend por container,
   - frontend em dev local,
   - frontend em deploy web,
   - notebook,
   - conjunto completo em compose de desenvolvimento.
2. Definir a matriz de configuracao por modo.
3. Garantir que `.env` e equivalentes sejam coerentes entre esses modos.
4. Garantir que dataset especifico passado pelo usuario funcione em todos os modos suportados.
5. Criar documentacao operacional clara sobre como iniciar, parar, depurar e consultar logs em cada modo.

### Arquivos principais

1. `README.md`
2. `docs/03_SETUP_AND_ENVIRONMENT.md`
3. `docs/04_QUICKSTART_TUTORIAL.md`
4. `docs/05_CLI_REFERENCE.md`
5. `docs/11_TROUBLESHOOTING.md`
6. `AutoMLAgent.ipynb`

### Criterios de aceite

1. O produto pode ser operado por CLI, web, container e notebook com instrucoes claras.
2. A configuracao de cada modo esta documentada e testavel.
3. O uso com dataset especifico do usuario continua sendo um caminho de primeira classe.

## Fase 17. Containerizacao, compose e proxy de producao

### Objetivo

Empacotar o sistema para desenvolvimento local e producao em VPS de forma clara e previsivel.

### Tarefas

1. Criar Dockerfile para `agent/` quando necessario como base operacional.
2. Criar Dockerfile para `backend/`.
3. Decidir se o `frontend/` tera container apenas para desenvolvimento local ou tambem para cenarios opcionais fora da Vercel.
4. Criar `compose.dev.yml` contemplando o fluxo local mais util.
5. Criar `compose.prod.yml` contemplando producao com `agent` + `backend` + Traefik na mesma rede.
6. Configurar Traefik para dominio do backend e roteamento correto.
7. Definir `.env` e secrets por ambiente.
8. Garantir que o frontend em producao consome corretamente o backend publicado.
9. Deixar explicito na documentacao quando compose e necessario e quando nao e.

### Arquivos principais

1. `compose.dev.yml`
2. `compose.prod.yml`
3. Dockerfiles futuros
4. configuracoes do Traefik
5. `.env.example`
6. documentacao de deploy

### Criterios de aceite

1. Existe um caminho claro para desenvolvimento local com containers.
2. Existe um caminho claro para VPS com backend e agente na mesma rede.
3. A producao nao depende desnecessariamente de containerizar o frontend se a estrategia escolhida for Vercel.

## Fase 18. Documentacao de deploy e operacao do stack web

### Objetivo

Consolidar em documentacao unica como publicar, operar e configurar backend, frontend e agente no stack final.

### Tarefas

1. Criar `DEPLOY.md` na raiz quando a fase de plataforma estiver madura.
2. Documentar deploy do frontend na Vercel.
3. Documentar deploy do backend e do agente na VPS com Traefik.
4. Documentar dominios, env vars, secrets, TLS e networking.
5. Documentar diferencas entre dev e producao.
6. Documentar estrategia de custos e limites quando aplicavel.

### Arquivos principais

1. `DEPLOY.md`
2. `README.md`
3. `docs/03_SETUP_AND_ENVIRONMENT.md`
4. `docs/11_TROUBLESHOOTING.md`

### Criterios de aceite

1. O fluxo de deploy e operacao cabe em um documento raiz claro.
2. O documento explica separadamente frontend, backend e agente.

## Fase 19. Modernizacao controlada para latest do ecossistema LangChain/LangSmith

### Objetivo

Adequar o projeto ao stack moderno de LangChain e LangSmith sem quebrar o que ja foi endurecido internamente.

### Tarefas

1. Levantar dependencias atuais e pontos de acoplamento a APIs antigas.
2. Definir a versao alvo estavel para LangChain, LangChain Community, LangChain Core e LangSmith.
3. Revisar `utils/tracing.py`, `utils/__init__.py`, embeddings e retrievers para o latest suportado.
4. Ajustar wrappers, callbacks, spans e metadata para aderir melhor ao LangSmith atual.
5. Garantir que a observabilidade externa complemente, e nao substitua, o ledger local.
6. Atualizar requirements, docs, testes e exemplos.
7. Revisar o uso de `langchain-classic` e dependencias transitorias, reduzindo legado quando viavel.

### Arquivos principais

1. `requirements.txt`
2. `requirements-local.txt`
3. `utils/tracing.py`
4. `utils/__init__.py`
5. `utils/embeddings.py`
6. `agent_manager/retriever.py`
7. `data_agent/retriever.py`
8. `docs/09_LLM_CONFIGURATION.md`
9. `docs/12_DEVELOPMENT_AND_TESTING.md`

### Criterios de aceite

1. O projeto fica alinhado a um alvo moderno e explicitamente suportado do ecossistema LangChain/LangSmith.
2. A modernizacao nao quebra tracing, retrieval nem testes.
3. LangSmith passa a ser a ferramenta externa principal de observabilidade, com o ledger local preservado.

## Fase 20. Tools operacionais para a rede de agentes com approvals

### Objetivo

Expandir a rede de agentes com ferramentas operacionais uteis sem abrir mao de governanca, approvals e seguranca.

### Tarefas

1. Definir quais tools fazem sentido para a rede do AutoML-Agent.
2. Separar tools por nivel de risco:
   - leitura/diagnostico,
   - execucao controlada,
   - instalacao de dependencias,
   - acoes destrutivas.
3. Criar politica de approval para:
   - instalar dependencia em run,
   - alterar ambiente,
   - limpar artefatos,
   - executar acao destrutiva.
4. Registrar no trace e no ledger toda invocacao de tool.
5. Garantir que tools nao virem bypass das constraints, do Critic Agent ou do HITL.
6. Definir se instalacao de dependencia em run pode existir apenas em dev ou tambem em producao, e com quais limites.

### Arquivos principais

1. `agent_manager/__init__.py`
2. `operation_agent/`
3. `utils/tracing.py`
4. documentacao operacional futura

### Criterios de aceite

1. A rede de agentes ganha ferramentas uteis sem perder auditabilidade.
2. Instalacao de dependencia em run nunca ocorre sem approval quando assim exigido.
3. O sistema evita crashes bobos por falta de dependencia sem abrir um buraco de seguranca.

## 5. Ordem pratica de execucao recomendada

Para reduzir risco e retrabalho, seguir esta ordem:

1. Fase 0: ADRs e baseline de regressao.
2. Fase 1: namespace por run + cleanup controlado.
3. Fase 2: tracing, lineage, terminal log e custos.
4. Fase 3: scheduler com fallback fila/serial.
5. Fase 4: cancelamento gracioso e pos-run CLI.
6. Fase 5: constraints granulares e analises persistidas.
7. Fase 6: HITL.
8. Fase 7: Critic Agent.
9. Fase 8: economia de tokens.
10. Fase 9: refinamentos de reasoning trail.
11. Fase 10: fechar suite de testes e regressao final.
12. Fase 11: sincronizacao final de docs e ADRs.
13. Fase 12: matriz de compatibilidade de providers e modos de execucao.
14. Fase 13: refactor estrutural do repositorio.
15. Fase 14: backend minimo.
16. Fase 15: frontend minimo.
17. Fase 16: unificacao dos modos de execucao e configuracao.
18. Fase 17: containerizacao, compose e proxy.
19. Fase 18: documentacao de deploy.
20. Fase 19: modernizacao controlada para latest LangChain/LangSmith.
21. Fase 20: tools operacionais com approvals.

## 6. Milestones recomendados

### Milestone A. Fundacao segura

Entregar:

1. namespace por run,
2. cleanup controlado,
3. manifest basico,
4. custo por run/modelo,
5. terminal log basico,
6. baseline de testes.

### Milestone B. Observabilidade completa

Entregar:

1. handoff lineage completo,
2. manager waiting/received events,
3. scheduler com fallback,
4. ledger estruturado,
5. reasoning trail consultavel.

### Milestone C. Controle de qualidade operacional

Entregar:

1. constraints granulares,
2. HITL estrategico,
3. Critic Agent,
4. cancelamento gracioso,
5. menu pos-run.

### Milestone D. Otimizacao e endurecimento

Entregar:

1. economia de tokens,
2. regressao completa,
3. documentacao final,
4. ADRs finais,
5. rollout controlado.

### Milestone E. Compatibilidade e topologia de produto

Entregar:

1. matriz oficial de compatibilidade OpenAI/OpenRouter/local-first,
2. reorganizacao `agent/backend/frontend`,
3. notebook e CLI preservados com o novo layout,
4. base de configuracao coerente entre superficies.

### Milestone F. Superficie web MVP

Entregar:

1. backend FastAPI minimo e protegido,
2. frontend Next/shadcn minimalista,
3. upload de dataset e pedido sobre dataset especifico,
4. fluxo web coerente com a CLI.

### Milestone G. Plataforma e operacao

Entregar:

1. `compose.dev.yml`,
2. `compose.prod.yml`,
3. Traefik para backend em producao,
4. `DEPLOY.md`,
5. documentacao dos modos de execucao.

### Milestone H. Observabilidade e ferramentas definitivas

Entregar:

1. alinhamento controlado ao latest suportado de LangChain/LangSmith,
2. LangSmith como camada externa principal de tracing,
3. tools operacionais com approvals,
4. governanca clara para instalacao de dependencias e acoes sensiveis.

## 7. Definicao de pronto

O trabalho discutido nesta sessao so deve ser considerado concluido quando todos os pontos abaixo estiverem verdadeiros:

1. uma nova run nao colide com outra run,
2. cada run possui `run_id` unico,
3. cada agente possui `agent_id` unico,
4. cada handoff possui `handoff_id` unico,
5. e possivel provar no trace quem passou o que para quem,
6. o manager registra quando esta aguardando e quando recebeu algo,
7. o custo por modelo e por run esta correto e auditavel,
8. o `terminal.log` ponta a ponta fica salvo em `agent_workspace/exp/...`,
9. a run pode terminar ou ser cancelada sem depender de `Ctrl+C` como mecanismo principal,
10. existe caminho claro para reabrir a CLI e decidir o proximo passo,
11. os tres modos de dataset estao isolados e auditaveis,
12. ha constraints granulares suficientes para reduzir ambiguidades,
13. ha analises persistidas por run,
14. HITL existe em pontos estrategicos,
15. existe Critic Agent ou equivalente com efeito real no fluxo,
16. o sistema tenta concorrencia e cai para fila quando necessario,
17. a suite de testes cobre regressao, namespace, tracing, cleanup, lineage, custos e logs,
18. existe uma matriz oficial de compatibilidade entre OpenAI, OpenRouter e local-first,
19. CLI, web, container e notebook estao documentados como formas oficiais ou explicitamente suportadas de uso,
20. o repositorio foi reorganizado de forma limpa em `agent`, `backend` e `frontend` quando essa frente for executada,
21. backend e frontend MVP funcionam com autenticacao minima correta,
22. o backend recusa imediatamente requests com segredo invalido,
23. o sistema continua permitindo fornecer um dataset especifico e fazer um pedido em cima dele,
24. existem `compose.dev.yml` e `compose.prod.yml` coerentes com a estrategia final,
25. `DEPLOY.md` explica a operacao do stack web,
26. LangSmith esta integrado como camada externa principal de observabilidade quando a fase correspondente for concluida,
27. a rede de agentes possui politica clara de tools e approvals,
28. README, docs e ADRs foram atualizados no mesmo change set.

## 8. Observacao final

Este backlog foi organizado com base no estado atual do repositorio nesta sessao. A implementacao deve priorizar migracao segura, compatibilidade controlada e feature flags sempre que uma mudanca puder afetar runs em andamento ou artefatos ja existentes.