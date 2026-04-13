# 09. LLM Configuration

## 1. Configuration Model

The repository uses a registry-based abstraction for LLM definitions. This allows the rest of the codebase to refer to logical names while still supporting raw provider model identifiers.

## 2. Alias Vs Raw Slug

AutoML-Agent accepts two kinds of model references:

1. registered aliases such as `or-glm-5`,
2. raw OpenRouter slugs such as `z-ai/glm-5.1`.

If a string contains `/`, the registry treats it as a raw OpenRouter-compatible model identifier.

## 3. Main Variables

### 3.1 `LLM_BACKBONE`

The default reasoning model for planning, verification, and code generation.

### 3.2 `LLM_PROMPT_AGENT`

The model used by the Prompt Agent for requirement parsing and prompt decomposition. This is often set to a cheaper or faster model than the backbone.

### 3.3 `OPENROUTER_MODEL`

An optional workflow-wide override. In interactive mode, this acts as a lock and forces both the backbone and the Prompt Agent to use the same overridden model.

### 3.4 `VLLM_BASE_URL`

The local endpoint used when `LLM_PROMPT_AGENT=prompt-llm`.

## 4. Provider Modes

### 4.1 OpenRouter

This is the default provider path for most users. It supports both registered aliases and raw slugs.

### 4.2 OpenAI Direct

The repository also keeps a direct OpenAI-compatible path for compatibility.

### 4.3 Local vLLM Prompt Agent

The Prompt Agent can be served locally through vLLM using the `prompt-llm` entry in the registry.

## 5. Current Alias Families

The registry currently includes alias groups for:

1. Anthropic,
2. OpenAI,
3. Google,
4. Meta,
5. DeepSeek,
6. Mistral,
7. Qwen,
8. xAI,
9. Z-AI,
10. Moonshot,
11. MiniMax,
12. Perplexity.

Use the CLI to inspect the exact current list:

```bash
python -m cli list-models
```

## 6. Choosing Models In Practice

### 6.1 Backbone Choice

Use a stronger backbone when:

1. plans are complex,
2. you need more reliable instruction synthesis,
3. you expect the Operation Agent to solve tricky local failures.

### 6.2 Prompt Agent Choice

Use a faster Prompt Agent when:

1. the request parsing step is straightforward,
2. you want lower cost,
3. you are iterating quickly on prompts.

## 7. Precedence Notes

The practical selection order depends on how you launch the workflow:

1. interactive mode exposes both model choices unless `OPENROUTER_MODEL` is set,
2. non-interactive mode can take an explicit `--llm`,
3. otherwise the CLI falls back to values from `.env`.

## 8. Recommended Defaults

For general local experimentation:

1. use a capable backbone such as `or-glm-5` or a raw Z-AI slug,
2. use a lighter Prompt Agent such as `or-gpt-5-nano` or `or-qwen3.5-flash`,
3. keep system information enabled in the CLI.

## 9. Advanced Local Prompt Agent Setup

If you want the fine-tuned local Prompt Agent path:

1. serve the model through vLLM,
2. set `LLM_PROMPT_AGENT=prompt-llm`,
3. ensure `VLLM_BASE_URL` points to the local endpoint.

This is an advanced path and is not required for the default OpenRouter-based workflow.

## 10. Reading Continuation

- Read [03. Setup And Environment](03_SETUP_AND_ENVIRONMENT.md) for environment-variable setup.
- Read [05. CLI Reference](05_CLI_REFERENCE.md) for where model selection happens interactively.
- Read [90. ADR Index](90_ADR_INDEX.md) for the decision record behind split LLM roles.