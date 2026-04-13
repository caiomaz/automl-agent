# 03. Setup And Environment

## 1. Installation Profiles

AutoML-Agent supports two installation profiles.

### 1.1 Core Profile

Use this when you want the CLI, orchestration logic, provider-backed LLM access, and lightweight local execution:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1.2 Local Heavy Profile

Use this when you also want local ML stacks such as PyTorch, local vision or audio libraries, or optional local Prompt Agent infrastructure:

```bash
pip install -r requirements-local.txt
```

## 2. Minimum Environment Variables

Copy the example file first:

```bash
cp .env.example .env
```

At minimum, the default CLI workflow requires:

```bash
OPENROUTER_API_KEY=your-key-here
```

The repository reads `.env` through `python-dotenv` and reloads it before CLI execution.

## 3. Environment Variable Reference

| Variable | Required | Purpose |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | yes for default workflow | Main LLM provider key |
| `OPENAI_API_KEY` | optional | Direct OpenAI compatibility path |
| `HF_API_KEY` | optional | Hugging Face dataset/model listing and downloads |
| `KAGGLE_API_TOKEN` | optional | Kaggle dataset access |
| `SEARCHAPI_API_KEY` | optional | Search provider support |
| `TAVILY_API_KEY` | optional | Web search support |
| `LLM_BACKBONE` | recommended | Default backbone alias or raw OpenRouter slug |
| `LLM_PROMPT_AGENT` | recommended | Default Prompt Agent alias or slug |
| `OPENROUTER_MODEL` | optional | Workflow-wide override for all agents |
| `VLLM_BASE_URL` | optional | Local vLLM endpoint for Prompt Agent mode |
| `LANGCHAIN_TRACING_V2` | optional | Enable tracing |
| `LANGCHAIN_API_KEY` | optional | LangSmith API key |
| `LANGCHAIN_PROJECT` | optional | LangSmith project name |

## 4. Choosing Backbone Models

The repository supports two styles of model selection:

1. registered aliases such as `or-glm-5` or `or-claude-sonnet`,
2. raw OpenRouter slugs such as `z-ai/glm-5.1` or `qwen/qwen3.5-flash-02-23`.

The default interactive wizard lets you choose both a backbone LLM and a lighter Prompt Agent LLM separately.

## 5. Optional Local Prompt Agent Via vLLM

The repository still supports a local Prompt Agent served through vLLM. This is an advanced setup and not the default onboarding path.

Typical shape:

```bash
HF_TOKEN="your-hf-token" \
CUDA_VISIBLE_DEVICES="0,1,2,3" \
python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --enable-lora \
  --lora-modules prompt-llama=./adapter/adapter-mixtral/ \
  --tensor-parallel-size 4
```

Then configure:

```bash
LLM_PROMPT_AGENT=prompt-llm
VLLM_BASE_URL=http://localhost:8000/v1
```

Use this only if you specifically want the local adapter path. For most users, the provider-backed Prompt Agent path is simpler.

## 6. Verifying The Setup

### 6.1 Check Available Models

```bash
python -m cli list-models
```

### 6.2 Start The Interactive CLI

```bash
python -m cli
```

You should see:

1. the CLI banner,
2. API key validation,
3. the six-step configuration wizard.

### 6.3 Run A Minimal Non-Interactive Command

```bash
python -m cli run \
  --task tabular_regression \
  --prompt "Predict crab age from both categorical and numerical features in the uploaded Crab Age Dataset" \
  --n-plans 1 \
  --n-revise 1
```

## 7. Machine-Specific Adaptation

The CLI can collect system information so generated code only relies on installed packages and visible hardware. Keep this enabled unless you have a specific reason to turn it off.

This is especially useful when:

1. your machine does not have a CUDA-capable GPU,
2. you only installed `requirements.txt`,
3. you want the Operation Agent to avoid heavy dependencies that are unavailable locally.

## 8. Security And Credential Notes

1. Keep `.env` untracked.
2. Keep `.env.example` as the template of record.
3. Prefer short-lived or scoped provider tokens when possible.
4. Treat generated code as local executable content and inspect artifacts when running on sensitive machines.

## 9. Next Reading

- Read [04. Quickstart Tutorial](04_QUICKSTART_TUTORIAL.md) for the fastest successful first run.
- Read [09. LLM Configuration](09_LLM_CONFIGURATION.md) for deeper model-selection details.
- Read [11. Troubleshooting](11_TROUBLESHOOTING.md) if the initial startup fails.