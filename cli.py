#!/usr/bin/env python3
"""AutoML-Agent CLI — interactive command-line interface.

Usage:
    python -m cli                   # interactive mode
    python -m cli list-models       # show available LLM aliases
    python -m cli run --task ... --prompt ...
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Force reload .env, even if already loaded elsewhere (e.g., by configs.py import)
load_dotenv(override=True)

# ── ANSI helpers ──────────────────────────────────────────────────────────────

BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"


def _header():
    print(f"""
{CYAN}{BOLD}╔══════════════════════════════════════════════════════════════╗
║                   🤖  AutoML-Agent  CLI                      ║
║          Multi-Agent LLM Framework for Full-Pipeline AutoML  ║
╚══════════════════════════════════════════════════════════════╝{RESET}
""")


def _sep():
    print(f"{DIM}{'─' * 62}{RESET}")


def _step(current: int, total: int, label: str):
    tag = f"  {current}/{total} — {label}  "
    w = 62
    dashes = w - len(tag)
    left = dashes // 2
    right = dashes - left
    print(f"\n{CYAN}{'─' * left}{BOLD}{tag}{RESET}{CYAN}{'─' * right}{RESET}\n")


# ── Model listing ────────────────────────────────────────────────────────────

def cmd_list_models():
    from configs import AVAILABLE_LLMs
    _header()
    print(f"{BOLD}Pre-registered LLM aliases:{RESET}\n")
    print(f"  {'Alias':<22} {'Model slug':<42}")
    _sep()
    for alias in AVAILABLE_LLMs.list():
        cfg = AVAILABLE_LLMs.get(alias)
        provider = "local" if cfg.base_url and "localhost" in cfg.base_url else "openrouter"
        if not cfg.base_url:
            provider = "openai"
        marker = f"{DIM}({provider}){RESET}"
        print(f"  {GREEN}{alias:<22}{RESET} {cfg.model:<42} {marker}")
    print()
    backbone = os.getenv("LLM_BACKBONE", "or-glm-5")
    prompt = os.getenv("LLM_PROMPT_AGENT", "prompt-llm")
    override = os.getenv("OPENROUTER_MODEL", "")
    print(f"{BOLD}Current defaults:{RESET}")
    print(f"  LLM_BACKBONE      = {CYAN}{backbone}{RESET}")
    print(f"  LLM_PROMPT_AGENT  = {CYAN}{prompt}{RESET}")
    if override:
        print(f"  OPENROUTER_MODEL  = {YELLOW}{override}{RESET}  {DIM}← overrides LLM_BACKBONE{RESET}")
    print()
    print(f"{BOLD}Tip:{RESET} You can use {GREEN}any OpenRouter slug directly{RESET} — no need to pre-register.")
    print(f"  Set {CYAN}LLM_BACKBONE=anthropic/claude-opus-4.6{RESET} in .env and it just works.")
    print(f"  Browse models: {CYAN}https://openrouter.ai/models{RESET}")
    print()


# ── Interactive prompt helpers ───────────────────────────────────────────────

def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{CYAN}{default}{RESET}]" if default else ""
    try:
        val = input(f"{BOLD}{prompt}{suffix}: {RESET}").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)
    return val or default


def _choose(prompt: str, options: list[str], default: str = "") -> str:
    print(f"\n{BOLD}{prompt}{RESET}")
    for i, opt in enumerate(options, 1):
        marker = f" {YELLOW}← default{RESET}" if opt == default else ""
        print(f"  {DIM}{i}.{RESET} {opt}{marker}")
    while True:
        raw = _ask("Choice (number or name)", default)
        if raw in options:
            return raw
        # Allow default value to pass through (e.g., raw OpenRouter slug like 'z-ai/glm-5.1')
        if default and raw == default:
            return raw
        # Allow any raw OpenRouter slug format (contains '/')
        if "/" in raw:
            return raw
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx]
        except ValueError:
            pass
        print(f"  {RED}Invalid choice, try again.{RESET}")


def _discover_datasets() -> list[str]:
    from utils.workspace import DATASETS_DIR
    ws = DATASETS_DIR
    if not ws.exists():
        return []
    return sorted(str(p) for p in ws.iterdir() if p.is_dir() or p.is_file())


# ── Interactive mode ─────────────────────────────────────────────────────────

def cmd_interactive():
    from dotenv import load_dotenv
    load_dotenv(override=True)  # Ensure .env is fresh

    import importlib
    import configs
    importlib.reload(configs)  # Rebuild registry with fresh .env values
    from configs import AVAILABLE_LLMs, TASK_METRICS

    _header()

    # API key check
    key = os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        print(f"{RED}✗ OPENROUTER_API_KEY not set in .env{RESET}")
        print(f"  Get your key at: {CYAN}https://openrouter.ai/keys{RESET}")
        sys.exit(1)
    print(f"{GREEN}✓{RESET} OpenRouter API key configured")
    _sep()

    # ── Step 1/6 — LLMs ─────────────────────────────────────────────────────
    _step(1, 6, "LLMs")
    override = os.getenv("OPENROUTER_MODEL", "")
    backbone_default = os.getenv("LLM_BACKBONE", "or-glm-5")
    # Exclude synthetic entries from user-facing list
    or_models = [a for a in AVAILABLE_LLMs.list() if a.startswith("or-") and a != "or-custom"]

    if override:
        llm = "or-custom"
        cfg = AVAILABLE_LLMs.get(llm)
        prompt_llm = "or-custom"
        prompt_cfg = cfg
        print(f"{YELLOW}⚡ OPENROUTER_MODEL override active:{RESET} {BOLD}{override}{RESET} ({cfg.model})")
        print(f"{DIM}  Both backbone and Prompt Agent are locked to this model.")
        print(f"  Unset OPENROUTER_MODEL in .env to choose them freely.{RESET}\n")
        os.environ["LLM_PROMPT_AGENT"] = prompt_llm
    else:
        print(f"{DIM}  Backbone: drives planning, synthesis, verification, and code generation.")
        print(f"  You can type any alias from the list, or any raw OpenRouter slug (e.g. anthropic/claude-opus-4.6).{RESET}")
        llm = _choose("LLM backbone:", or_models, backbone_default)
        cfg = AVAILABLE_LLMs.get(llm)
        print(f"\n  {GREEN}→{RESET} Backbone: {BOLD}{cfg.model}{RESET}\n")

        print(f"{DIM}  Prompt Agent: parses your description into structured requirements.")
        print(f"  A faster/cheaper model is usually sufficient here.{RESET}")
        prompt_llm_default = os.getenv("LLM_PROMPT_AGENT", llm)
        prompt_llm = _choose("Prompt Agent LLM:", or_models, prompt_llm_default)
        try:
            prompt_cfg = AVAILABLE_LLMs.get(prompt_llm)
            print(f"\n  {GREEN}→{RESET} Prompt Agent: {BOLD}{prompt_cfg.model}{RESET}\n")
        except Exception:
            prompt_cfg = None
            print(f"\n  {GREEN}→{RESET} Prompt Agent: {BOLD}{prompt_llm}{RESET}\n")
        os.environ["LLM_PROMPT_AGENT"] = prompt_llm
    _sep()

    # ── Step 2/6 — Task type ─────────────────────────────────────────────────
    _step(2, 6, "Task type")
    print(f"{DIM}  Determines which metrics and prompt templates are used.{RESET}")
    tasks = list(TASK_METRICS.keys())
    task = _choose("Task type:", tasks)
    print(f"\n  {GREEN}→{RESET} {BOLD}{task}{RESET}  {DIM}(primary metric: {TASK_METRICS[task]}){RESET}\n")
    _sep()

    # ── Step 3/6 — Dataset ───────────────────────────────────────────────────
    _step(3, 6, "Dataset")
    print(f"{DIM}  Provide a local path, a remote URL, or leave both blank.")
    print(f"  If blank, the agent will attempt to locate and download a suitable dataset.{RESET}\n")
    datasets = _discover_datasets()
    data_path = ""
    if datasets:
        print(f"{BOLD}  Previously downloaded datasets:{RESET}")
        for i, d in enumerate(datasets, 1):
            print(f"    {DIM}{i}.{RESET} {d}")
        data_path = _ask("\n  Local path (number, path, or Enter to skip)", "")
        if data_path.isdigit() and 1 <= int(data_path) <= len(datasets):
            data_path = datasets[int(data_path) - 1]
    else:
        data_path = _ask("  Local path (file or directory, or Enter to skip)", "")

    data_url = ""
    if not data_path:
        data_url = _ask(f"  Remote URL {DIM}(Kaggle competition, HuggingFace dataset, direct download — or Enter to skip){RESET}", "")
        if data_url:
            print(f"\n  {GREEN}→{RESET} Will download to agent_workspace/datasets/ before launch.")
        else:
            print(f"\n  {YELLOW}⚠{RESET} No dataset specified — the agent will attempt to retrieve one automatically.")
    else:
        print(f"\n  {GREEN}→{RESET} Dataset: {data_path}")
    _sep()

    # ── Step 4/6 — Task description ──────────────────────────────────────────
    _step(4, 6, "Task description")
    _task_examples = {
        "image_classification": "Classify butterfly species from the uploaded dataset of butterfly wing images",
        "text_classification": "Classify e-commerce product descriptions into Electronics, Household, Books, and Clothing categories",
        "tabular_classification": "Classify banana quality as Good or Bad based on size, weight, sweetness, softness, harvest time, ripeness, and acidity",
        "tabular_regression": "Predict crab age from both categorical and numerical features in the uploaded Crab Age Dataset",
        "tabular_clustering": "Group smoker status into two clusters based on numerical health features, evaluated against the smoking label",
        "node_classification": "Predict paper category for each node in the Cora citation graph dataset",
        "ts_forecasting": "Forecast future weather observations (96 steps, 21 dims) from past sequences of the same size",
    }
    example = _task_examples.get(task, "Describe your machine learning task clearly")
    print(f"{DIM}  Describe what you want to achieve. The more context you give, the better the generated plans.")
    print(f"  Example: '{example}'{RESET}\n")
    user_prompt = _ask("  Prompt")
    if not user_prompt:
        print(f"\n{RED}✗ A task description is required.{RESET}")
        sys.exit(1)
    _sep()

    # ── Step 5/6 — Constraints (optional) ───────────────────────────────────
    _step(5, 6, "Constraints  (optional)")
    print(f"{DIM}  Constraints are injected explicitly into every plan generated by the agent.")
    print(f"  Skip all to run in free-form mode — the agent will choose methods and targets freely.{RESET}\n")
    _metric_defaults = {
        "image_classification": "accuracy",
        "text_classification": "accuracy",
        "tabular_classification": "F1",
        "tabular_regression": "RMSLE",
        "tabular_clustering": "Rand index",
        "node_classification": "accuracy",
        "ts_forecasting": "RMSLE",
    }
    c_model = _ask("  Preferred model/algorithm (e.g. XGBoost, LightGBM, ResNet-50, LSTM)", "")
    c_perf_name = ""
    c_perf_val = ""
    if _ask("  Set a performance target? (y/n)", "n").lower().startswith("y"):
        c_perf_name = _ask("    Metric name", _metric_defaults.get(task, ""))
        c_perf_val = _ask("    Target value (e.g. 0.95 for accuracy, 0.1 for RMSLE)", "")
    c_train_time = _ask("  Max training time (e.g. '30 minutes', '2 hours')", "unlimited").strip()
    if c_train_time.lower() == "unlimited":
        c_train_time = ""
    c_inference = _ask("  Max inference time per sample (e.g. '5 ms', '100 ms')", "unlimited").strip()
    if c_inference.lower() == "unlimited":
        c_inference = ""

    system_info_str = ""
    print(f"\n  {DIM}System info: let the agent know your installed packages and hardware{RESET}")
    print(f"  {DIM}so it generates code that uses only what is available locally.{RESET}")
    if _ask("  Include system info? (y/n)", "y").lower().startswith("y"):
        print(f"  {CYAN}⟳{RESET}  Collecting system information...", end="", flush=True)
        from utils.sysinfo import collect_system_info, format_for_agent
        system_info_str = format_for_agent(collect_system_info())
        print(f"\r  {GREEN}✓{RESET}  System information collected.          ")
    _sep()

    # Build structured constraints dict and enriched prompt
    constraints_dict = {}
    constraints_parts = []
    if c_model:
        constraints_dict["model"] = c_model
        constraints_parts.append(f"Use {c_model} as the model/algorithm.")
    if c_perf_name and c_perf_val:
        constraints_dict["perf_metric"] = c_perf_name
        constraints_dict["perf_value"] = c_perf_val
        constraints_parts.append(f"Achieve at least {c_perf_val} {c_perf_name}.")
    elif c_perf_name:
        constraints_dict["perf_metric"] = c_perf_name
        constraints_parts.append(f"Optimize for {c_perf_name}.")
    if c_train_time:
        constraints_dict["max_train_time"] = c_train_time
        constraints_parts.append(f"Training time must not exceed {c_train_time}.")
    if c_inference:
        constraints_dict["max_inference_time"] = c_inference
        constraints_parts.append(f"Inference time per sample must be under {c_inference}.")
    if data_path:
        constraints_parts.append(f"The dataset has been uploaded at: {data_path}")
    elif data_url:
        constraints_parts.append(f"The dataset can be downloaded from: {data_url}")

    prompt_type = "constraint" if constraints_dict else "free"
    if constraints_parts:
        user_prompt = user_prompt.rstrip(".") + ". " + " ".join(constraints_parts)

    # ── Step 6/6 — Advanced options ──────────────────────────────────────────
    _step(6, 6, "Advanced options")
    print(f"{DIM}  Press Enter to accept defaults shown in brackets.{RESET}\n")
    n_plans_raw = _ask("  Number of plans [more = broader search, slower]", "3")
    n_plans = int(n_plans_raw)
    n_revise_raw = _ask("  Max revisions   [retries when no plan passes validation; 0 = disabled]", "3")
    n_revise = int(n_revise_raw)
    rap_input = _ask("  RAP — fetch papers/examples to guide planning? (y/n)", "y")
    rap = rap_input.lower().startswith("y")
    _sep()

    # ── Confirm & launch ─────────────────────────────────────────────────────
    try:
        prompt_model_str = f" ({prompt_cfg.model})" if prompt_cfg else ""
    except Exception:
        prompt_model_str = ""

    print(f"\n{BOLD}Configuration summary{RESET}\n")
    print(f"  {'Backbone LLM':<18}: {CYAN}{llm}{RESET} ({cfg.model})")
    print(f"  {'Prompt Agent':<18}: {CYAN}{prompt_llm}{RESET}{prompt_model_str}")
    print(f"  {'Task':<18}: {BOLD}{task}{RESET}  {DIM}metric: {TASK_METRICS[task]}{RESET}")
    print(f"  {'Prompt type':<18}: {YELLOW if prompt_type == 'constraint' else DIM}{prompt_type}{RESET}")
    if data_url:
        print(f"  {'Dataset':<18}: {CYAN}URL{RESET} → {data_url}")
    elif data_path:
        print(f"  {'Dataset':<18}: {data_path}")
    else:
        print(f"  {'Dataset':<18}: {DIM}(auto-retrieve){RESET}")
    if constraints_dict:
        print(f"  {'Constraints':<18}:")
        for c in constraints_parts:
            if not c.startswith("The dataset"):
                print(f"    {GREEN}•{RESET} {c}")
    print(f"  {'Plans':<18}: {n_plans}")
    print(f"  {'Revisions':<18}: {n_revise}")
    print(f"  {'RAP':<18}: {'enabled' if rap else 'disabled'}")
    print(f"  {'System info':<18}: {'enabled' if system_info_str else f'{DIM}disabled{RESET}'}")
    print()

    confirm = _ask("Launch? (y/n)", "y")
    if not confirm.lower().startswith("y"):
        print("Aborted.")
        sys.exit(0)

    print(f"\n{GREEN}{BOLD}🚀 Launching AutoML-Agent...{RESET}\n")
    _sep()

    # Download dataset from URL if provided
    if data_url and not data_path:
        from data_agent.retriever import retrieve_download
        from utils.workspace import WORKSPACE_DIR, ensure_workspace
        ensure_workspace()
        print(f"\n  {CYAN}⬇{RESET}  Downloading dataset from URL...")
        downloaded = retrieve_download(url=data_url, name=task + "_dataset", workspace=WORKSPACE_DIR)
        if downloaded:
            data_path = downloaded
            print(f"  {GREEN}→{RESET} Dataset saved to: {data_path}")
        else:
            print(f"  {YELLOW}⚠{RESET} Download failed — agent will attempt auto-retrieval.")

    # Resolve data_path to list if it's a directory
    from glob import glob as g
    if data_path and os.path.isdir(data_path):
        data_path = g(data_path + "/*")

    from agent_manager import AgentManager
    manager = AgentManager(
        task=task,
        llm=llm,
        interactive=False,
        data_path=data_path or None,
        n_plans=n_plans,
        n_revise=n_revise,
        rap=rap,
        constraints=constraints_dict or None,
        system_info=system_info_str or None,
    )
    manager.initiate_chat(user_prompt)


# ── Non-interactive run ──────────────────────────────────────────────────────

def __collect_sysinfo_if(enabled: bool) -> str | None:
    if not enabled:
        return None
    from utils.sysinfo import collect_system_info, format_for_agent
    return format_for_agent(collect_system_info()) or None


def cmd_run(args):
    _header()
    from agent_manager import AgentManager
    from glob import glob as g

    data_path = args.data
    if data_path and os.path.isdir(data_path):
        data_path = g(data_path + "/*")

    llm = args.llm or os.getenv("LLM_BACKBONE", "or-glm-5")

    from configs import AVAILABLE_LLMs
    cfg = AVAILABLE_LLMs.get(llm)

    # Build structured constraints dict and enriched prompt
    prompt = args.prompt
    constraints_dict = {}
    constraints_parts = []
    if args.model:
        constraints_dict["model"] = args.model
        constraints_parts.append(f"Use {args.model} as the model/algorithm.")
    if args.perf_metric and args.perf_value:
        constraints_dict["perf_metric"] = args.perf_metric
        constraints_dict["perf_value"] = args.perf_value
        constraints_parts.append(f"Achieve at least {args.perf_value} {args.perf_metric}.")
    elif args.perf_metric:
        constraints_dict["perf_metric"] = args.perf_metric
        constraints_parts.append(f"Optimize for {args.perf_metric}.")
    if args.max_train_time and args.max_train_time.lower() != "unlimited":
        constraints_dict["max_train_time"] = args.max_train_time
        constraints_parts.append(f"Training time must not exceed {args.max_train_time}.")
    if args.max_inference_time and args.max_inference_time.lower() != "unlimited":
        constraints_dict["max_inference_time"] = args.max_inference_time
        constraints_parts.append(f"Inference time per sample must be under {args.max_inference_time}.")
    if data_path:
        dp = data_path if isinstance(data_path, str) else ", ".join(data_path)
        constraints_parts.append(f"The dataset has been uploaded at: {dp}")

    prompt_type = "constraint" if constraints_dict else "free"
    if constraints_parts:
        prompt = prompt.rstrip(".") + ". " + " ".join(constraints_parts)

    print(f"{GREEN}→{RESET} LLM: {BOLD}{cfg.model}{RESET}")
    print(f"{GREEN}→{RESET} Task: {BOLD}{args.task}{RESET}")
    print(f"{GREEN}→{RESET} Type: {BOLD}{prompt_type}{RESET}")
    print(f"{GREEN}→{RESET} Prompt: {prompt[:120]}...")
    _sep()
    print()

    manager = AgentManager(
        task=args.task,
        llm=llm,
        interactive=False,
        data_path=data_path,
        n_plans=args.n_plans,
        n_revise=args.n_revise,
        rap=not args.no_rap,
        constraints=constraints_dict or None,
        system_info=__collect_sysinfo_if(args.system_info),
    )
    manager.initiate_chat(prompt)


# ── Entrypoint ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="automl-agent",
        description="AutoML-Agent CLI — Multi-Agent LLM Framework",
    )
    sub = parser.add_subparsers(dest="command")

    # list-models
    sub.add_parser("list-models", help="Show available LLM aliases")

    # run (non-interactive)
    run_p = sub.add_parser("run", help="Run with arguments (non-interactive)")
    run_p.add_argument("--llm", type=str, help="LLM alias (default: LLM_BACKBONE env)")
    run_p.add_argument("--task", type=str, required=True, help="Task type")
    run_p.add_argument("--prompt", type=str, required=True, help="User prompt")
    run_p.add_argument("--data", type=str, default=None, help="Path to dataset")
    run_p.add_argument("--n-plans", type=int, default=3, help="Number of plans")
    run_p.add_argument("--n-revise", type=int, default=3, help="Max revisions")
    run_p.add_argument("--no-rap", action="store_true", help="Disable RAP")
    # Constraint flags
    run_p.add_argument("--model", type=str, default=None, help="Preferred model/algorithm (e.g. XGBoost, ResNet-50)")
    run_p.add_argument("--perf-metric", type=str, default=None, help="Performance metric name (e.g. accuracy, F1, RMSLE)")
    run_p.add_argument("--perf-value", type=str, default=None, help="Performance target value (e.g. 0.95)")
    run_p.add_argument("--max-train-time", type=str, default=None, help="Max training time (e.g. '30 minutes')")
    run_p.add_argument("--max-inference-time", type=str, default=None, help="Max inference time per sample (e.g. '5 ms')")
    run_p.add_argument("--system-info", action="store_true", default=True, help="Collect and pass system info to agent (default: on)")
    run_p.add_argument("--no-system-info", action="store_false", dest="system_info", help="Disable system info collection")

    args = parser.parse_args()

    if args.command == "list-models":
        cmd_list_models()
    elif args.command == "run":
        cmd_run(args)
    else:
        # Default: interactive mode
        cmd_interactive()


if __name__ == "__main__":
    main()
