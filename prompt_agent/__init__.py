import json, re, os

from configs import AVAILABLE_LLMs
from utils import print_message, get_client

json_specification = json.load(
    open(f"{os.getcwd()}/prompt_agent/WizardLAMP/template_schema.json")
)

# agent_profile = f"""You are a helpful assistant.
# # #JSON SPECIFICATION SCHEMA#
# ```json
# {json_specification}
# ```
# """

# agent_profile = f"""You are a helpful assistant.
# Your task is to parse the user's requirement into a valid JSON format using the JSON specification schema as your reference. Your response must exactly follow the given JSON schema and be based only on the user's instruction.
# Make sure that your answer contains only the JSON response without any comment or explanation because it can cause parsing errors.

# #JSON SPECIFICATION SCHEMA#
# ```json
# {json_specification}
# ```

# Your response must begin with "```json" or "{{" and end with "```" or "}}", respectively.
# """

agent_profile = f"""You are an assistant project manager in the AutoML development team. 
Your task is to parse the user's requirement into a valid JSON format using the JSON specification schema as your reference. Your response must exactly follow the given JSON schema and be based only on the user's instruction. 
Make sure that your answer contains only the JSON response without any comment or explanation because it can cause parsing errors.

#JSON SPECIFICATION SCHEMA#
```json
{json_specification}
```

Your response must begin with "```json" or "{{" and end with "```" or "}}", respectively.
"""

# agent_profile = f"""You are an experienced assistant project manager in the AutoML development team.
# Your task is to parse the user's requirement into a valid JSON format using the JSON specification schema as your reference. Your response must exactly follow the given JSON schema and be based only on the user's instruction.
# Make sure that your answer contains only the JSON response without any comment or explanation because it can cause parsing errors.

# #JSON SPECIFICATION SCHEMA#
# ```json
# {json_specification}
# ```

# Your response must begin with "```json" or "{{" and end with "```" or "}}", respectively.
# """

# agent_profile = f"""You are the world's best assistant project manager in the AutoML development team.
# Your task is to parse the user's requirement into a valid JSON format using the JSON specification schema as your reference. Your response must exactly follow the given JSON schema and be based only on the user's instruction.
# Make sure that your answer contains only the JSON response without any comment or explanation because it can cause parsing errors.

# #JSON SPECIFICATION SCHEMA#
# ```json
# {json_specification}
# ```

# Your response must begin with "```json" or "{{" and end with "```" or "}}", respectively.
# """


class PromptAgent:
    def __init__(self, llm: str = "prompt-llm"):
        self.agent_type = "prompt"
        # Phase 8: per-stage routing — when LLM_STAGE_PROMPT_PARSE is set,
        # it overrides the alias passed in. Falls back to the constructor
        # argument when no override is configured.
        try:
            from utils.stage_routing import resolve_stage_alias
            llm = resolve_stage_alias("prompt_parse", default=llm)
        except Exception:
            pass
        self.llm = llm
        self.client = get_client(llm)
        self.model = AVAILABLE_LLMs[llm]["model"]

    def parse_openai(self, instruction, return_json=False, task: str = None,
                     run_ctx=None, workspace=None):
        print_message(
            self.agent_type, "I am analyzing your request 🔍. Please wait for a moment."
        )
        if run_ctx is not None:
            try:
                from utils.ledger import append_event, record_llm_usage
                append_event(run_ctx, "agent_started", source="prompt-agent",
                             workspace=workspace)
            except Exception:
                pass
        task_constraint = f"\n        IMPORTANT: The downstream_task field MUST be set to \"{task}\" exactly as written. Do NOT infer a different task type from the instruction." if task else ""
        prompt = f"""Please carefully parse the following #Instruction#.{task_constraint}
        Your response can only begin with "```json" or "{{" and end with "```" or "}}" without saying any word or explain.
        
        #Instruction#
        {instruction}
        
        #Valid JSON Response#
        """
        client = get_client(self.llm)
        res = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": agent_profile},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.01,
        )
        if run_ctx is not None:
            try:
                from utils.ledger import append_event, record_llm_usage
                record_llm_usage(run_ctx, res, alias=self.llm,
                                 model_slug=self.model, phase="prompt_parse",
                                 workspace=workspace)
                append_event(run_ctx, "llm_call_completed", source="prompt-agent",
                             workspace=workspace, alias=self.llm,
                             model_slug=self.model, phase="prompt_parse")
            except Exception:
                pass

        if return_json:
            content = res.choices[0].message.content.strip()
            try:
                content = json.loads(content)
                if run_ctx is not None:
                    try:
                        from utils.ledger import append_event
                        append_event(run_ctx, "agent_finished", source="prompt-agent",
                                     workspace=workspace, status="ok")
                    except Exception:
                        pass
                return content
            except Exception as e:
                content = json.loads(content.split("\n\n")[0].strip())
                if run_ctx is not None:
                    try:
                        from utils.ledger import append_event
                        append_event(run_ctx, "agent_finished", source="prompt-agent",
                                     workspace=workspace, status="ok")
                    except Exception:
                        pass
                return content
        else:
            if run_ctx is not None:
                try:
                    from utils.ledger import append_event
                    append_event(run_ctx, "agent_finished", source="prompt-agent",
                                 workspace=workspace, status="ok")
                except Exception:
                    pass
            return res.choices[0].message.content.strip()

    def parse(self, instruction, return_json=False, task: str = None,
              run_ctx=None, workspace=None, token_economy: str = "off"):
        print_message(
            self.agent_type, "I am analyzing your request 🔍. Please wait for a moment."
        )
        if run_ctx is not None:
            try:
                from utils.ledger import append_event
                append_event(run_ctx, "agent_started", source="prompt-agent",
                             workspace=workspace)
            except Exception:
                pass
        task_constraint = f"\n        IMPORTANT: The downstream_task field MUST be set to \"{task}\" exactly as written. Do NOT infer a different task type from the instruction." if task else ""
        prompt = f"""Please carefully parse the following #Instruction#.{task_constraint}
        Your response can only begin with "```json" or "{{" and end with "```" or "}}" without saying any word or explain.
        
        #Instruction#
        {instruction}
        
        #Valid JSON Response#
        """

        # Phase 8: opt-in run-scoped cache for parse outputs. Identical
        # (model, task, instruction) triples reuse the previous JSON parse
        # without spending tokens. Disabled when token_economy="off".
        _cache = None
        _cache_key = None
        if (
            return_json
            and run_ctx is not None
            and (token_economy or "off") != "off"
        ):
            try:
                from utils.run_cache import RunCache, make_key
                _cache = RunCache(run_ctx, workspace=workspace)
                _cache_key = make_key(
                    "prompt_parse", self.model, str(task or ""), instruction
                )
                cached = _cache.get(_cache_key, stage="prompt_parse")
                if cached is not None:
                    if run_ctx is not None:
                        try:
                            from utils.ledger import append_event
                            append_event(run_ctx, "agent_finished",
                                         source="prompt-agent",
                                         workspace=workspace, status="ok",
                                         cache_hit=True)
                        except Exception:
                            pass
                    return cached
            except Exception:
                _cache = None

        res = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": agent_profile + "\n" + prompt},
            ],
            temperature=0.01,
        )
        if run_ctx is not None:
            try:
                from utils.ledger import append_event, record_llm_usage
                record_llm_usage(run_ctx, res, alias=self.llm,
                                 model_slug=self.model, phase="prompt_parse",
                                 workspace=workspace)
                append_event(run_ctx, "llm_call_completed", source="prompt-agent",
                             workspace=workspace, alias=self.llm,
                             model_slug=self.model, phase="prompt_parse")
            except Exception:
                pass

        def _finish_event():
            if run_ctx is not None:
                try:
                    from utils.ledger import append_event
                    append_event(run_ctx, "agent_finished", source="prompt-agent",
                                 workspace=workspace, status="ok")
                except Exception:
                    pass

        if return_json:
            content = res.choices[0].message.content.strip()
            pattern = r"^```(?:\w+)?\s*\n(.*?)(?=^```)```"
            results = re.findall(pattern, content, re.DOTALL | re.MULTILINE)
            if len(results) > 0:
                content = results[0].strip()
            try:
                content = json.loads(content)
                if _cache is not None and _cache_key is not None:
                    try:
                        _tokens = 0
                        if getattr(res, "usage", None) is not None:
                            _tokens = int(getattr(res.usage, "total_tokens", 0) or 0)
                        _cache.set(_cache_key, content, tokens_estimate=_tokens)
                    except Exception:
                        pass
                _finish_event()
                return content
            except Exception as e:
                content = json.loads(content.split("\n\n")[0].strip())
                if _cache is not None and _cache_key is not None:
                    try:
                        _tokens = 0
                        if getattr(res, "usage", None) is not None:
                            _tokens = int(getattr(res.usage, "total_tokens", 0) or 0)
                        _cache.set(_cache_key, content, tokens_estimate=_tokens)
                    except Exception:
                        pass
                _finish_event()
                return content
        else:
            _finish_event()
            return res.choices[0].message.content.strip()
