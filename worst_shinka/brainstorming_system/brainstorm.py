import ast
import os
import json
import re
import yaml
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from openai import OpenAI
from worst_shinka.llm.client import get_client_llm
from .evaluation_adapter import BrainstormingEvaluationAdapter
from worst_shinka.judge import Judge
from typing import Callable
from datetime import datetime


client: OpenAI = get_client_llm()
# TODO refine prompts (Kalina)

@dataclass
class BrainstormResult:
    proposal_1: str
    proposal_2: str
    debate_history: List[Dict[str, str]]



# algorithm.py's maintained public interface (see initial_model/algorithm.py) - every
# generated proposal must implement exactly these top-level functions, or training
# crashes deep inside RL_training/train.py instead of failing fast here.
REQUIRED_ALGORITHM_FUNCTIONS = ("get_epsilon", "select_action", "select_opponent_action", "update_model")

ALGORITHM_INTERFACE_SPEC = """
The code must be a single, self-contained Python module that implements EXACTLY this public interface (same names and parameters as the parent code - do not rename, remove, or omit any of them, and do not add a class wrapper):
  - get_epsilon(episode_index: int, hyperparameters: dict) -> float
  - select_action(model, observation, epsilon: float, num_actions: int) -> int
  - select_opponent_action(model, observation, num_actions: int) -> int
  - update_model(model, state, action, reward, next_state, done, hyperparameters) -> None
The `hyperparameters`/config dict is loaded verbatim from the generation's fixed config.yaml and is NOT extensible - only reference keys that are explicitly listed as available below; inventing a new key crashes training with a KeyError.
`model`, `observation`, and the Q-values returned by `model.predict(...)` are opaque objects whose concrete implementation you have NOT been shown - only use operations already demonstrated by the parent code (e.g. `.argmax()` on the predict result, `is None`/`is not None` on `model`). Do NOT add `assert`/`isinstance`/type-check statements, truthiness checks (`if q_values:`), or any other comparison about their type, shape, or value beyond `is None` on `model` - such assumptions are frequently wrong (e.g. a Q-value array is never a plain bool/list and can't be used in a truthiness check) and crash training.
Do NOT wrap calls to `model.predict(...)` or any other logic in `try`/`except` to log a warning and fall back to a random/default action - errors must propagate, not be silently hidden; the reference implementation has no try/except anywhere. If you are unsure an operation is safe, don't use it - stick to what the parent code already demonstrates.
Output raw Python source only - no markdown code fences, no explanatory text outside the code.
"""

SYS_PROMPT_BRAINSTRORMER_TEMPLATE = """
You are {role}. Focus on {metrics}.
"""

B1_MSG_TEMPLATE = """
{context}
Propose an improved algorithmic strategy combining the best of these parents.
""" + ALGORITHM_INTERFACE_SPEC

B2_MSG_TEMPLATE = """
{context}
Propose a distinct alternative strategy combining the best of these parents."
""" + ALGORITHM_INTERFACE_SPEC

SYS_PROMPT_CODE_CRITIC = """
You are a Harsh Code Critic.

"""

SYS_PROMPT_REFINE = """
Refine your approach based on the critique.
"""

REFINE_IDEA_MSG = """
Your Idea:
{idea}

Critique:
{critique}
"""

CODE_OUTPUT_SYS_PROMPT_TEMPLATE = "Output clean Python code for Proposal {n} based on final consensus.\n" + ALGORITHM_INTERFACE_SPEC


_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def _extract_code(text: str) -> str:
    """Strip a markdown code fence LLMs commonly wrap output in despite being told not to.

    Takes the content of the first ```/```python fenced block found anywhere in `text`;
    falls back to the raw (stripped) text if no fence is present.
    """
    match = _CODE_FENCE_RE.search(text)
    return match.group(1).strip() if match else text.strip()


def _missing_algorithm_functions(code: str) -> List[str]:
    """Names from REQUIRED_ALGORITHM_FUNCTIONS not defined at module level in `code`."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return list(REQUIRED_ALGORITHM_FUNCTIONS)
    defined = {
        node.name for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    return [name for name in REQUIRED_ALGORITHM_FUNCTIONS if name not in defined]


def _unknown_hyperparameter_keys(code: str, allowed_keys: List[str]) -> List[str]:
    """String-literal keys used as code["key"]/hyperparameters["key"] that aren't in allowed_keys.

    Best-effort static check (skipped if allowed_keys is empty, e.g. config couldn't be read) -
    catches the common case of an LLM inventing a config key that doesn't exist in config.yaml,
    which would otherwise crash training deep inside RL_training/train.py with a KeyError.
    """
    if not allowed_keys:
        return []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    allowed = set(allowed_keys)
    unknown = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id in ("hyperparameters", "config")
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
            and node.slice.value not in allowed
        ):
            unknown.add(node.slice.value)
    return sorted(unknown)


def _has_assert_statements(code: str) -> bool:
    """Whether the proposal contains any `assert` statement.

    The reference implementation (initial_model/algorithm.py) has none - an LLM adding one is
    inventing an unverified assumption about an opaque object (model/observation/q_values) whose
    concrete type it was never shown, which crashes training with an AssertionError when wrong.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False  # already caught by _missing_algorithm_functions
    return any(isinstance(node, ast.Assert) for node in ast.walk(tree))


def _has_try_except(code: str) -> bool:
    """Whether the proposal contains any `try`/`except` block.

    The reference implementation (initial_model/algorithm.py) has none. An LLM adding one
    typically wraps a call it isn't confident about (e.g. model.predict(...)) and silently
    falls back to a random/default action on any error, which doesn't crash training but
    quietly degrades it to near-random play - much harder to notice than an outright crash.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False  # already caught by _missing_algorithm_functions
    return any(isinstance(node, ast.Try) for node in ast.walk(tree))


class BrainstormingPipeline:
    def __init__(self, model_a: str, model_b: str,config_path, max_debate_rounds: int = 2):
        self.model_a = model_a
        self.model_b = model_b
        self.max_debate_rounds = max_debate_rounds
        self.path = config_path
        self.cost_usd = 0.0
        self.cost_available = False

    def _record_response_cost(self, response) -> None:
        usage = getattr(response, "usage", None)
        cost = getattr(usage, "cost", None) if usage is not None else None
        if cost is None and isinstance(usage, dict):
            cost = usage.get("cost")
        try:
            if cost is not None:
                self.cost_usd += float(cost)
                self.cost_available = True
        except (TypeError, ValueError):
            pass

    def _call_llm(self, model: str, system_prompt: str, user_prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        self._record_response_cost(response)
        return response.choices[0].message.content or ""
    def save_debate_to_json(self, debate_history: list[dict], filepath):
        directory = os.path.dirname(filepath)
            
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {
            "timestamp": datetime.now().isoformat(),
            "total_turns": len(debate_history),
            "history": debate_history
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4, ensure_ascii=False)
            
        print(f"Debate history successfully saved to {filepath}")
    def append_turn_to_jsonl(self, turn_data: dict, filepath: str = "debate_stream.jsonl"):
        directory = os.path.dirname(filepath)
    
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {
            "timestamp": datetime.now().isoformat(),
            "content": turn_data
        }
        
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    def run_brainstorming(
        self,
        parents_data: List[Dict],
        attempt: int,
        judge_rejection_reason: Optional[str] = None,
        hyperparameter_keys: Optional[List[str]] = None,
    ) -> BrainstormResult:
        context = f"Parent Code Candidates & Performance:\n{json.dumps(parents_data, indent=2)}\n"
        if hyperparameter_keys:
            context += (
                f"\nThe hyperparameters/config dict available to get_epsilon/update_model contains "
                f"EXACTLY these keys, nothing else: {hyperparameter_keys}. Do not invent, rename, or "
                "assume any other keys - accessing a missing key crashes training."
            )
        if judge_rejection_reason:
            context += f"\nCRITICAL: Previous proposals were REJECTED by Judge for: {judge_rejection_reason}. Address this!"

        debate_history = []
        path = os.path.join(self.path, f"debate_history/debate_check_{attempt}.json")
        b1_prompt = B1_MSG_TEMPLATE.format(context=context)
        idea_1 = self._call_llm(
            self.model_a,
            SYS_PROMPT_BRAINSTRORMER_TEMPLATE.format(role='Expert Brainstormer 1', metrics='algorithmic efficiency'),
            b1_prompt,
            )
        self.append_turn_to_jsonl(idea_1, path)
        b2_prompt = B2_MSG_TEMPLATE.format(context=context)
        idea_2 = self._call_llm(
            self.model_b,
            SYS_PROMPT_BRAINSTRORMER_TEMPLATE.format(role='Expert Brainstormer 2', metrics='code simplicity and robust edge-case handling'),
            b2_prompt)
        self.append_turn_to_jsonl(idea_2, path)
        debate_history.extend([{"agent": "Brainstormer 1", "content": idea_1}, {"agent": "Brainstormer 2", "content": idea_2}])

        for round_num in range(self.max_debate_rounds):
            critic_prompt = f"Parent Data:\n{context}\n\nProposed Ideas:\n1: {idea_1}\n2: {idea_2}\nIdentify trade-offs, potential bugs, or performance bottlenecks in both."
            critique = self._call_llm(self.model_a, SYS_PROMPT_CODE_CRITIC, critic_prompt) # we choose model a - to be think through
            debate_history.append({"agent": f"Critic (Round {round_num+1})", "content": critique})

            idea_1 = self._call_llm(self.model_a, SYS_PROMPT_REFINE, REFINE_IDEA_MSG.format(idea=idea_1, critique=critique))
            idea_2 = self._call_llm(self.model_b, SYS_PROMPT_REFINE, REFINE_IDEA_MSG.format(idea=idea_2, critique=critique))

            # consider the case when they found agreement faster than self.max_debate_rounds
            # break the loop.

        prop1_code = _extract_code(self._call_llm(self.model_a, CODE_OUTPUT_SYS_PROMPT_TEMPLATE.format(n=1), idea_1))
        prop2_code = _extract_code(self._call_llm(self.model_b, CODE_OUTPUT_SYS_PROMPT_TEMPLATE.format(n=2), idea_2))
        self.save_debate_to_json(debate_history, os.path.join(self.path, f"debate_history/debate_{attempt}.json"))
        return BrainstormResult(proposal_1=prop1_code, proposal_2=prop2_code, debate_history=debate_history)

class EvolutionWorkflow:
    def __init__(self, models: List[str], 
                gen_id: int, 
                history_path: str,
                train_config_path: str,
                train_function: Callable,
                max_debate_rounds: int = 2,
                workers: int = 1):
        self.brainstormer = BrainstormingPipeline(models[0], models[1], history_path, max_debate_rounds)
        judge = Judge(
            train_function=train_function,
            history_path=os.path.join(history_path, "history_judge"),
            workers=workers,
        )
        self.judge = BrainstormingEvaluationAdapter(judge, train_function, workers=workers)
        self.gen_id = gen_id
        self.train_config_path = train_config_path
        try:
            with open(train_config_path) as f:
                self.hyperparameter_keys = sorted((yaml.safe_load(f) or {}).keys())
        except OSError:
            self.hyperparameter_keys = []

    @property
    def cost_usd(self) -> float | None:
        return self.brainstormer.cost_usd if self.brainstormer.cost_available else None

    def execute_crossover(self, parents_data: List[Dict], max_judge_retries: int = 3):
        rejection_reason = None

        for attempt in range(max_judge_retries):
            proposals = self.brainstormer.run_brainstorming(
                parents_data, attempt, rejection_reason, hyperparameter_keys=self.hyperparameter_keys
            )

            missing_1 = _missing_algorithm_functions(proposals.proposal_1)
            missing_2 = _missing_algorithm_functions(proposals.proposal_2)
            unknown_1 = _unknown_hyperparameter_keys(proposals.proposal_1, self.hyperparameter_keys)
            unknown_2 = _unknown_hyperparameter_keys(proposals.proposal_2, self.hyperparameter_keys)
            has_assert_1 = _has_assert_statements(proposals.proposal_1)
            has_assert_2 = _has_assert_statements(proposals.proposal_2)
            has_try_1 = _has_try_except(proposals.proposal_1)
            has_try_2 = _has_try_except(proposals.proposal_2)
            if (
                missing_1 or missing_2 or unknown_1 or unknown_2
                or has_assert_1 or has_assert_2 or has_try_1 or has_try_2
            ):
                print(
                    f"Looping back. Proposal 1 missing {missing_1}, unknown keys {unknown_1}, "
                    f"has assert {has_assert_1}, has try/except {has_try_1}; "
                    f"Proposal 2 missing {missing_2}, unknown keys {unknown_2}, "
                    f"has assert {has_assert_2}, has try/except {has_try_2}"
                )
                rejection_reason = (
                    "Generated code did not implement the required algorithm.py interface, referenced "
                    "hyperparameter keys that don't exist in config.yaml, or contained a forbidden assert/try-except statement. "
                    f"Proposal 1 is missing functions: {missing_1 or 'nothing'}, unknown keys used: {unknown_1 or 'none'}, "
                    f"contains assert: {has_assert_1}, contains try/except: {has_try_1}. "
                    f"Proposal 2 is missing functions: {missing_2 or 'nothing'}, unknown keys used: {unknown_2 or 'none'}, "
                    f"contains assert: {has_assert_2}, contains try/except: {has_try_2}. "
                    f"Every proposal MUST define exactly these top-level functions: {', '.join(REQUIRED_ALGORITHM_FUNCTIONS)}. "
                    f"The only available hyperparameter keys are: {self.hyperparameter_keys}. "
                    "Remove any assert/isinstance/type-check statements or try/except blocks about model, observation, "
                    "or Q-values - you were not shown their concrete implementation, and errors must propagate, not be hidden."
                )
                continue

            result = self.judge.evaluate(
                proposal_1=proposals.proposal_1, 
                proposal_2=proposals.proposal_2,
                config_path=self.train_config_path,
                gen_id=self.gen_id
            )
            
            if result["winner"]:
                if result.get("training_error"):
                    print(f"Success! {result['training_error']}")
                else:
                    print(f"Success! Judge selected proposal")
                result["winner_code"] = (proposals.proposal_1 if result["winner_id"] == "proposal_1" else proposals.proposal_2)
                return result

            if result.get("training_error"):
                print(f"Looping back. {result['training_error']}")
                rejection_reason = result["training_error"]
                continue

            print(f"Looping back. Metrics {result['metrics']}")
            rejection_reason = f"Lack of sufficient winner. Metrics of both models: {result['metrics']}"

        print("Failed to satisfy Judge within max retries; continuing with the next generation.")
        return None