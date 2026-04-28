import uuid

from configs import AVAILABLE_LLMs
from data_agent import retriever
from utils import print_message, get_client
from utils.tracing import traceable as _traceable, set_run_metadata as _set_run_metadata
from utils.ledger import append_event, append_handoff, record_llm_usage


# agent_profile = """You are a helpful assistant."""

# agent_profile = """You are a helpful assistant. You have the following main responsibilities to complete.
# 1. Retrieve a dataset from the user or search for the dataset based on the user instruction.
# 2. Perform data preprocessing based on the user instruction or best practice based on the given tasks.
# 3. Perform data augmentation as neccesary.
# 4. Extract useful information and underlying characteristics of the dataset."""

# agent_profile = """You are a data scientist of an automated machine learning project (AutoML) that can find the most relevant datasets,run useful preprocessing, perform suitable data augmentation, and make meaningful visulaization to comprehensively understand the data based on the user requirements. You have the following main responsibilities to complete.
# 1. Retrieve a dataset from the user or search for the dataset based on the user instruction.
# 2. Perform data preprocessing based on the user instruction or best practice based on the given tasks.
# 3. Perform data augmentation as neccesary.
# 4. Extract useful information and underlying characteristics of the dataset."""

# agent_profile = """You are an experienced data scientist of an automated machine learning project (AutoML) that can find the most relevant datasets,run useful preprocessing, perform suitable data augmentation, and make meaningful visulaization to comprehensively understand the data based on the user requirements. You have the following main responsibilities to complete.
# 1. Retrieve a dataset from the user or search for the dataset based on the user instruction.
# 2. Perform data preprocessing based on the user instruction or best practice based on the given tasks.
# 3. Perform data augmentation as neccesary.
# 4. Extract useful information and underlying characteristics of the dataset."""

agent_profile = """You are the world's best data scientist of an automated machine learning project (AutoML) that can find the most relevant datasets,run useful preprocessing, perform suitable data augmentation, and make meaningful visulaization to comprehensively understand the data based on the user requirements. You have the following main responsibilities to complete.
1. Retrieve a dataset from the user or search for the dataset based on the user instruction.
2. Perform data preprocessing based on the user instruction or best practice based on the given tasks.
3. Perform data augmentation as neccesary.
4. Extract useful information and underlying characteristics of the dataset."""


class DataAgent:
    def __init__(self, user_requirements, llm="qwen", rap=True, decomp=True, run_ctx=None, branch_id=None):
        self.agent_type = "data"
        self.llm = llm
        self.model = AVAILABLE_LLMs[llm]["model"]
        self.user_requirements = user_requirements
        self.rap = rap
        self.decomp = decomp
        self.money = {}
        self.run_ctx = run_ctx
        if run_ctx is not None:
            if branch_id is not None:
                run_ctx.branch_id = branch_id
            self.agent_id = f"data_{run_ctx.run_id[:8]}_{uuid.uuid4().hex[:8]}"
            run_ctx.agent_id = self.agent_id
            append_event(
                run_ctx, "agent_started",
                source="data",
                payload_summary="data_agent initializing",
            )
        else:
            self.agent_id = None

    @_traceable(name="data_agent_understand_plan", run_type="chain")
    def understand_plan(self, plan):
        _set_run_metadata(task=getattr(self, 'task', None), llm=self.llm, model=self.model)
        summary_prompt = f"""As a proficient data scientist, summarize the following plan given by the senior AutoML project manager according to the user's requirements and your expertise in data science.
        
        # User's Requirements
        ```json
        {self.user_requirements}
        ```
        
        # Project Plan
        {plan}
        
        The summary of the plan should enable you to fulfill your responsibilities as the answers to the following questions by focusing on the data manipulation and analysis.
        1. How to retrieve or collect the dataset(s)?
        2. How to preprocess the retrieved dataset(s)?
        3. How to efficiently augment the dataset(s)?
        4. How to extract and understand the underlying characteristics of the dataset(s)?
        
        Note that you should not perform data visualization because you cannot see it. Make sure that another data scientist can exectly reproduce the results based on your summary."""

        messages = [
            {"role": "system", "content": agent_profile},
            {"role": "user", "content": summary_prompt},
        ]

        retry = 0
        while retry < 10:
            try:
                res = get_client(self.llm).chat.completions.create(
                    model=self.model, messages=messages, temperature=0.3
                )
                break
            except Exception as e:
                print_message("system", e)
                retry += 1
                continue

        data_plan = res.choices[0].message.content.strip()
        self.money[f"Data_Plan_Decomposition"] = res.usage.to_dict(mode="json")
        record_llm_usage(
            self.run_ctx, res,
            alias=self.llm,
            model_slug=self.model,
            phase="data_plan_decomposition",
        )
        if self.run_ctx is not None:
            append_event(
                self.run_ctx, "llm_call_completed",
                source="data",
                payload_summary="data plan decomposition",
                payload_size=res.usage.total_tokens,
            )
        _set_run_metadata(understand_plan_tokens=self.money.get("Data_Plan_Decomposition"))
        return data_plan

    @_traceable(name="data_agent_execute_plan", run_type="chain")
    def execute_plan(self, plan, data_path, pid):
        _set_run_metadata(llm=self.llm, model=self.model, pid=pid, rap=str(self.rap), decomp=str(self.decomp))
        print_message(self.agent_type, "I am working with the given plan!", pid)

        handoff_id = str(uuid.uuid4())
        if self.run_ctx is not None:
            append_handoff(
                self.run_ctx, handoff_id,
                source_agent_id=getattr(self.run_ctx, "_manager_agent_id", "manager"),
                dest_agent_id=self.agent_id or "data",
                direction="received",
                payload_summary="data execution plan",
                payload_text=plan,
            )
        if self.decomp:
            data_plan = self.understand_plan(plan)
        else:
            data_plan = plan

        available_sources = retriever.retrieve_datasets(
            self.user_requirements, data_path, get_client(self.llm), self.model
        )

        # Check whether the given source is accessible before running the execution --> reduce FileNotFound error

        # modality-based extraction ?

        exec_prompt = f"""As a proficient data scientist, your task is to explain **detailed** steps for data manipulation and analysis parts by executing the following machine learning development plan.
        
        # Plan
        {data_plan}
        
        # Potential Source of Dataset
        {available_sources}
        
        Make sure that your explanation follows these instructions:
        - All of your explanation must be self-contained without using any placeholder to ensure that other data scientists can exactly reproduce all the steps, but do not include any code.
        - Include how and where to retrieve or collect the data.
        - Include how to preprocess the data and which tools or libraries are used for the preprocessing.
        - Include how to do the data augmentation with details and names.
        - Include how to extract and understand the characteristics of the data.
        - Include reasons why each step in your explanations is essential to effectively complete the plan.        
        Note that you should not perform data visualization because you cannot see it. Make sure to focus only on the data part as it is your expertise. Do not conduct or perform anything regarding modeling or training.
        After complete the explanations, explicitly specify the (expected) outcomes and results both quantitative and qualitative of your explanations."""

        messages = [
            {"role": "system", "content": agent_profile},
            {"role": "user", "content": exec_prompt},
        ]

        retry = 0
        while retry < 10:
            try:
                res = get_client(self.llm).chat.completions.create(
                    model=self.model, messages=messages, temperature=0.3
                )
                break
            except Exception as e:
                print_message("system", e)
                retry += 1
                continue

        # Data LLaMA summarizes the given plan for optimizing data relevant processes
        action_result = res.choices[0].message.content.strip()
        self.money[f"Data_Plan_Execution_{pid}"] = res.usage.to_dict(mode="json")
        record_llm_usage(
            self.run_ctx, res,
            alias=self.llm,
            model_slug=self.model,
            phase=f"data_plan_execution_{pid}",
        )
        if self.run_ctx is not None:
            append_event(
                self.run_ctx, "llm_call_completed",
                source="data",
                payload_summary=f"data plan execution pid={pid}",
                payload_size=res.usage.total_tokens,
            )
            append_event(
                self.run_ctx, "agent_finished",
                source="data",
                payload_summary=f"data execution done pid={pid}",
            )
        _set_run_metadata(execute_plan_tokens=self.money)

        print_message(self.agent_type, "I have done with my execution!", pid)
        return action_result
