import os
import shutil
import uuid

from configs import AVAILABLE_LLMs
from utils import print_message, get_client
from utils.workspace import ensure_workspace, WORKSPACE_DIR, DATASETS_DIR, MODELS_DIR, EXP_DIR, run_exp_dir, run_models_dir
from operation_agent.execution import execute_script
from utils.ledger import append_event, record_llm_usage, write_analysis

# agent_profile = """You are a helpful assistant."""

# agent_profile = """You are a helpful assistant. You have the following main responsibilities to complete.
# 1. Write accurate Python codes to retrieve/load the given dataset from the corresponding source.
# 2. Write effective Python codes to preprocess the retrieved dataset.
# 3. Write precise Python codes to retrieve/load the given model and optimize it with the suggested hyperparameters.
# 4. Write efficient Python codes to train/finetune the retrieved model.
# 5. Write suitable Python codes to prepare the trained model for deployment. This step may include model compression and conversion according to the target inference platform.
# 6. Write Python codes to build the web application demo using the Gradio library.
# 7. Run the model evaluation using the given Python functions and summarize the results for validation againts the user's requirements.
# """

# agent_profile = """You are an MLOps engineer of an automated machine learning project (AutoML) that can implement the optimal solution for production-level deployment, given any datasets and models. You have the following main responsibilities to complete.
# 1. Write accurate Python codes to retrieve/load the given dataset from the corresponding source.
# 2. Write effective Python codes to preprocess the retrieved dataset.
# 3. Write precise Python codes to retrieve/load the given model and optimize it with the suggested hyperparameters.
# 4. Write efficient Python codes to train/finetune the retrieved model.
# 5. Write suitable Python codes to prepare the trained model for deployment. This step may include model compression and conversion according to the target inference platform.
# 6. Write Python codes to build the web application demo using the Gradio library.
# 7. Run the model evaluation using the given Python functions and summarize the results for validation againts the user's requirements.
# """

# agent_profile = """You an experienced MLOps engineer of an automated machine learning project (AutoML) that can implement the optimal solution for production-level deployment, given any datasets and models. You have the following main responsibilities to complete.
# 1. Write accurate Python codes to retrieve/load the given dataset from the corresponding source.
# 2. Write effective Python codes to preprocess the retrieved dataset.
# 3. Write precise Python codes to retrieve/load the given model and optimize it with the suggested hyperparameters.
# 4. Write efficient Python codes to train/finetune the retrieved model.
# 5. Write suitable Python codes to prepare the trained model for deployment. This step may include model compression and conversion according to the target inference platform.
# 6. Write Python codes to build the web application demo using the Gradio library.
# 7. Run the model evaluation using the given Python functions and summarize the results for validation againts the user's requirements.
# """

agent_profile = """You are the world's best MLOps engineer of an automated machine learning project (AutoML) that can implement the optimal solution for production-level deployment, given any datasets and models. You have the following main responsibilities to complete.
1. Write accurate Python codes to retrieve/load the given dataset from the corresponding source.
2. Write effective Python codes to preprocess the retrieved dataset.
3. Write precise Python codes to retrieve/load the given model and optimize it with the suggested hyperparameters.
4. Write efficient Python codes to train/finetune the retrieved model.
5. Write suitable Python codes to prepare the trained model for deployment. This step may include model compression and conversion according to the target inference platform.
6. Write Python codes to build the web application demo using the Gradio library.
7. Run the model evaluation using the given Python functions and summarize the results for validation againts the user's requirements.
"""


class OperationAgent:
    def __init__(self, user_requirements, llm, code_path, device=0, system_info="", run_ctx=None, token_economy="off"):
        # setup Farm Manager
        self.agent_type = "operation"
        # Phase 8: per-stage routing \u2014 LLM_STAGE_CODE_GENERATION overrides
        # the backbone for code synthesis. Falls back to the constructor
        # argument when no override is configured.
        try:
            from utils.stage_routing import resolve_stage_alias
            llm = resolve_stage_alias("code_generation", default=llm)
        except Exception:
            pass
        self.llm = llm
        self.model = AVAILABLE_LLMs[llm]["model"]
        self.experiment_logs = []
        self.user_requirements = user_requirements
        self.run_ctx = run_ctx  # Phase 1: optional RunContext for namespaced runs
        # Phase 8: token-economy policy controls error/payload compaction.
        self.token_economy = token_economy or "off"
        if run_ctx is not None:
            self.root_path = str(run_exp_dir(run_ctx.run_id))
            self.agent_id = f"operation_{run_ctx.run_id[:8]}_{uuid.uuid4().hex[:8]}"
            run_ctx.agent_id = self.agent_id
        else:
            self.root_path = str(EXP_DIR)  # absolute path to generated scripts directory
            self.agent_id = None
        self.code_path = code_path
        self.device = device
        self.system_info = system_info or ""
        self.money = {}
        ensure_workspace()  # guarantee all workspace dirs exist before any script runs
        if run_ctx is not None:
            append_event(
                run_ctx, "agent_started",
                source="operation",
                payload_summary="operation_agent initializing",
            )

    def self_validation(self, filename, log_path=None):
        rcode, log = execute_script(filename, device=self.device, log_path=log_path)
        return rcode, log

    def implement_solution(self, code_instructions, full_pipeline=True, code="", n_attempts=5):
        print_message(
            self.agent_type,
            f"I am implementing the following instruction:\n\r{code_instructions}",
        )

        log = "Nothing. This is your first attempt."
        error_logs = []
        code = code  # if a template/skeleton code is provided
        iteration = 0
        completion = None
        action_result = ""
        rcode = -1
        while iteration < n_attempts:
            try:
                if full_pipeline:
                    pipeline = (
                        "entire machine learning pipeline (from data retrieval to model deployment via Gradio). "
                        "The main() function MUST end by calling iface.launch() (or app.launch()) so the Gradio web "
                        "app is actually started. Do NOT comment out the launch call."
                    )
                else:
                    pipeline = "modeling pipeline (from data retrieval to model saving)"
                sysinfo_block = self.system_info if self.system_info else "(no system info provided — use common ML libraries)"

                # List datasets already present on disk so the LLM uses exact filenames
                _ds_listing = ""
                if DATASETS_DIR.exists():
                    _entries = []
                    for _d in sorted(DATASETS_DIR.iterdir()):
                        if _d.is_dir() and any(_d.iterdir()):
                            _files = [f.name for f in _d.iterdir() if f.is_file()]
                            _entries.append(f'  "{_d}" → files: {", ".join(_files)}')
                        elif _d.is_file():
                            _entries.append(f'  "{_d}"')
                    if _entries:
                        _ds_listing = "\n                Datasets already downloaded locally (use these exact paths — do NOT re-download):\n" + "\n".join(_entries)

                exec_prompt = f"""Carefully read the following instructions to write Python code for {self.user_requirements["problem"]["downstream_task"]} task.
                {code_instructions}
                
                # Previously Written Code
                ```python
                {code}
                ```
                
                # Error from the Previously Written Code
                {log}

                # Execution Environment
                The code will run locally on the following machine. Use ONLY libraries that are listed as installed.
                Do NOT install or import packages that are not listed below.
                {sysinfo_block}

                CRITICAL — File path rules (violating these causes PermissionError):
                - NEVER use absolute paths like /app/, /data/, /home/, /tmp/ etc.
                - ALL file operations MUST use the workspace directories below:
                  * Datasets: "{DATASETS_DIR}"
                  * Trained models: "{MODELS_DIR}"
                  * Experiment outputs: "{EXP_DIR}"
                - If the plan mentions paths like /app/data/... or /app/models/..., IGNORE those paths and use the workspace directories above instead.
                - Create subdirectories inside these workspace dirs as needed using os.makedirs(..., exist_ok=True).
                {_ds_listing}

                Note that you need to write the python code for the {pipeline}. Start the python code with "```python".
                Please ensure the completeness of the code so that it can be run without additional modifications.
                If there is any error from the previous attempt, please carefully fix it first."""

                messages = [
                    {"role": "system", "content": agent_profile},
                    {"role": "user", "content": exec_prompt},
                ]
                res = get_client(self.llm).chat.completions.create(
                    model=self.model, messages=messages, temperature=0.3
                )
                raw_completion = res.choices[0].message.content.strip()
                completion = raw_completion.split("```python")[1].split("```")[0]
                self.money[f'Operation_Coding_{iteration}'] = res.usage.to_dict(mode='json')
                record_llm_usage(
                    self.run_ctx, res,
                    alias=self.llm,
                    model_slug=self.model,
                    phase=f"coding_iteration_{iteration}",
                )
                if self.run_ctx is not None:
                    append_event(
                        self.run_ctx, "llm_call_completed",
                        source="operation",
                        payload_summary=f"code generation attempt {iteration}",
                        payload_size=res.usage.total_tokens,
                    )

                if not completion.strip(" \n"):
                    continue

                filename = f"{self.root_path}{self.code_path}.py"
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, "wt") as file:
                    file.write(completion)
                code = completion

                if self.run_ctx is not None:
                    append_event(
                        self.run_ctx, "artifact_written",
                        source="operation",
                        payload_ref=filename,
                        payload_summary="generated ML script",
                        payload_text=completion,
                    )

                _log_path = None
                if self.run_ctx is not None:
                    _log_path = str(run_exp_dir(self.run_ctx.run_id) / "terminal.log")
                rcode, log = self.self_validation(filename, log_path=_log_path)
                if rcode == 0:
                    action_result = log
                    break
                else:
                    # Phase 8: compact long stderr blobs before reinjection.
                    try:
                        from utils.token_economy import summarize_error, record_tokens_saved
                        compact, info = summarize_error(log, policy=self.token_economy)
                        if info.get("truncated") and self.run_ctx is not None:
                            saved_chars = max(0, len(log) - len(compact))
                            record_tokens_saved(
                                self.run_ctx,
                                source="error_summarization",
                                saved_tokens=saved_chars // 4,  # rough chars→tokens
                                stage="code_generation",
                                iteration=iteration,
                            )
                        log = compact
                    except Exception:
                        pass
                    error_logs.append(log)
                    action_result = log
                    print_message(self.agent_type, f"I got this error (itr #{iteration}): {log}")
                    iteration += 1                    
                    # break
            except Exception as e:
                iteration += 1
                print_message(self.agent_type, f"===== Retry: {iteration} =====")
                print_message(self.agent_type, f"Executioin error occurs: {e}")
            continue
        if not completion:
            completion = ""

        if self.run_ctx is not None:
            write_analysis(
                self.run_ctx,
                "code_instruction",
                code_instructions,
            )
            append_event(
                self.run_ctx, "agent_finished",
                source="operation",
                payload_summary=f"rcode={rcode} after {iteration} iterations",
            )

        print_message(
            self.agent_type,
            f"I executed the given plan and got the follow results:\n\n{action_result}",
        )
        return {"rcode": rcode, "action_result": action_result, "code": completion, "error_logs": error_logs}
