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
        self.llm = llm
        self.client = get_client(llm)
        self.model = AVAILABLE_LLMs[llm]["model"]

    def parse_openai(self, instruction, return_json=False, task: str = None):
        print_message(
            self.agent_type, "I am analyzing your request 🔍. Please wait for a moment."
        )
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

        if return_json:
            content = res.choices[0].message.content.strip()
            try:
                content = json.loads(content)
                return content
            except Exception as e:
                content = json.loads(content.split("\n\n")[0].strip())
                return content
        else:
            return res.choices[0].message.content.strip()

    def parse(self, instruction, return_json=False, task: str = None):
        print_message(
            self.agent_type, "I am analyzing your request 🔍. Please wait for a moment."
        )
        task_constraint = f"\n        IMPORTANT: The downstream_task field MUST be set to \"{task}\" exactly as written. Do NOT infer a different task type from the instruction." if task else ""
        prompt = f"""Please carefully parse the following #Instruction#.{task_constraint}
        Your response can only begin with "```json" or "{{" and end with "```" or "}}" without saying any word or explain.
        
        #Instruction#
        {instruction}
        
        #Valid JSON Response#
        """
        res = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": agent_profile + "\n" + prompt},
            ],
            temperature=0.01,
        )

        if return_json:
            content = res.choices[0].message.content.strip()
            pattern = r"^```(?:\w+)?\s*\n(.*?)(?=^```)```"
            results = re.findall(pattern, content, re.DOTALL | re.MULTILINE)
            if len(results) > 0:
                content = results[0].strip()
            try:
                content = json.loads(content)
                return content
            except Exception as e:
                content = json.loads(content.split("\n\n")[0].strip())
                return content
        else:
            return res.choices[0].message.content.strip()
