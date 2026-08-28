import ast
import logging
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
log = logging.getLogger(__name__)
# TODO refine prompts (Kalina)

@dataclass
class BrainstormResult:
    proposal_1: str
    proposal_2: str
    debate_history: List[Dict[str, str]]
    config_1: Optional[Dict] = None
    config_2: Optional[Dict] = None



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
The `hyperparameters`/config dict comes from your own proposed config.yaml (see below), not a fixed schema - you may retune existing keys' values and/or introduce new keys your algorithm needs. The only hard rule is self-consistency: every key your code reads via `hyperparameters[...]`/`config[...]` MUST be declared in the config.yaml you output for this same proposal, or training crashes with a KeyError.
`model`, `observation`, and the Q-values returned by `model.predict(...)` are opaque objects whose concrete implementation you have NOT been shown - only use operations already demonstrated by the parent code (e.g. `.argmax()` on the predict result, `is None`/`is not None` on `model`). Do NOT add `assert`/`isinstance`/type-check statements, truthiness checks (`if q_values:`), or any other comparison about their type, shape, or value beyond `is None` on `model` - such assumptions are frequently wrong (e.g. a Q-value array is never a plain bool/list and can't be used in a truthiness check) and crash training.
Do NOT wrap calls to `model.predict(...)` or any other logic in `try`/`except` to log a warning and fall back to a random/default action - errors must propagate, not be silently hidden; the reference implementation has no try/except anywhere. If you are unsure an operation is safe, don't use it - stick to what the parent code already demonstrates.
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

CODE_OUTPUT_SYS_PROMPT_TEMPLATE = (
    "Output clean Python code for Proposal {n} based on final consensus.\n"
    + ALGORITHM_INTERFACE_SPEC
    + """
Your response must contain exactly these two sections, in this order, and nothing else - no prose
before, between, or after them:

### ALGORITHM.PY
```python
<the complete algorithm.py source, following every rule above>
```

### CONFIG.YAML
```yaml
<the complete config.yaml content: a flat mapping of every hyperparameter key the code above reads
via hyperparameters[...]/config[...] - keep unchanged parent values as-is, retune what you changed,
and add any new keys you introduced>
```
"""
)


_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
_YAML_FENCE_RE = re.compile(r"```ya?ml\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_GENERIC_FENCE_RE = re.compile(r"```[a-zA-Z]*\s*\n(.*?)```", re.DOTALL)


def _extract_code(text: str) -> str:
    """Strip a markdown code fence LLMs commonly wrap output in despite being told not to.

    Takes the content of the first ```/```python fenced block found anywhere in `text`;
    falls back to the raw (stripped) text if no fence is present.
    """
    match = _CODE_FENCE_RE.search(text)
    return match.group(1).strip() if match else text.strip()


def _extract_yaml_block(text: str) -> Optional[Dict]:
    """Parse the config.yaml fenced block in `text` into a dict, or None if absent/invalid.

    Prefers a fence explicitly tagged ```yaml/```yml (case-insensitive). Falls back to the
    *last* generically-fenced block in `text` when there are at least two fenced blocks and
    none is yaml-tagged - the expected shape is "```python ... ``` then ```<something> ... ```",
    so the second block is the config even if the model dropped/misspelled the language tag.
    Returns None (rather than raising) on no candidate fence, a YAML parse error, or a parsed
    value that isn't a mapping - callers treat any of these as "no config was proposed" and
    reject the proposal, same as a missing required algorithm function.
    """
    match = _YAML_FENCE_RE.search(text)
    if match:
        candidate = match.group(1)
    else:
        blocks = _GENERIC_FENCE_RE.findall(text)
        if len(blocks) < 2:
            return None
        candidate = blocks[-1]
    try:
        parsed = yaml.safe_load(candidate)
    except yaml.YAMLError:
        return None
    return parsed if isinstance(parsed, dict) else None


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

        log.debug("Debate history saved to %s", filepath)
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
    ) -> BrainstormResult:
        context = f"Parent Code Candidates & Performance:\n{json.dumps(parents_data, indent=2)}\n"
        context += (
            "\nEach parent above includes its own config.yaml (its hyperparameters dict). You may reuse, "
            "retune, or extend these hyperparameters in your proposal - the config.yaml you output must "
            "declare every key your code reads via hyperparameters[...]/config[...]."
        )
        if judge_rejection_reason:
            context += f"\nCRITICAL: Previous proposals were REJECTED by Judge for: {judge_rejection_reason}. Address this!"

        log.info(
            "Brainstorming attempt %s: %s parent(s), models %s / %s%s",
            attempt, len(parents_data), self.model_a, self.model_b,
            " (retrying after judge rejection)" if judge_rejection_reason else "",
        )

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
        log.debug("Both brainstormers proposed an initial idea for attempt %s", attempt)

        for round_num in range(self.max_debate_rounds):
            critic_prompt = f"Parent Data:\n{context}\n\nProposed Ideas:\n1: {idea_1}\n2: {idea_2}\nIdentify trade-offs, potential bugs, or performance bottlenecks in both."
            critique = self._call_llm(self.model_a, SYS_PROMPT_CODE_CRITIC, critic_prompt) # we choose model a - to be think through
            debate_history.append({"agent": f"Critic (Round {round_num+1})", "content": critique})

            idea_1 = self._call_llm(self.model_a, SYS_PROMPT_REFINE, REFINE_IDEA_MSG.format(idea=idea_1, critique=critique))
            idea_2 = self._call_llm(self.model_b, SYS_PROMPT_REFINE, REFINE_IDEA_MSG.format(idea=idea_2, critique=critique))
            log.debug("Critique round %s/%s complete for attempt %s", round_num + 1, self.max_debate_rounds, attempt)

            # consider the case when they found agreement faster than self.max_debate_rounds
            # break the loop.

        output_1 = self._call_llm(self.model_a, CODE_OUTPUT_SYS_PROMPT_TEMPLATE.format(n=1), idea_1)
        output_2 = self._call_llm(self.model_b, CODE_OUTPUT_SYS_PROMPT_TEMPLATE.format(n=2), idea_2)
        prop1_code = _extract_code(output_1)
        prop2_code = _extract_code(output_2)
        config_1 = _extract_yaml_block(output_1)
        config_2 = _extract_yaml_block(output_2)
        log.info(
            "Attempt %s: both proposals generated (config parsed: proposal_1=%s, proposal_2=%s)",
            attempt, config_1 is not None, config_2 is not None,
        )
        self.save_debate_to_json(debate_history, os.path.join(self.path, f"debate_history/debate_{attempt}.json"))
        return BrainstormResult(
            proposal_1=prop1_code,
            proposal_2=prop2_code,
            debate_history=debate_history,
            config_1=config_1,
            config_2=config_2,
        )

class EvolutionWorkflow:
    def __init__(self, models: List[str],
                gen_id: int,
                history_path: str,
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

    @property
    def cost_usd(self) -> float | None:
        return self.brainstormer.cost_usd if self.brainstormer.cost_available else None

    def execute_crossover(self, parents_data: List[Dict], max_judge_retries: int = 3):
        rejection_reason = None

        for attempt in range(max_judge_retries):
            proposals = self.brainstormer.run_brainstorming(
                parents_data, attempt, rejection_reason
            )

            missing_config_1 = proposals.config_1 is None
            missing_config_2 = proposals.config_2 is None
            keys_1 = list(proposals.config_1.keys()) if proposals.config_1 else []
            keys_2 = list(proposals.config_2.keys()) if proposals.config_2 else []

            missing_1 = _missing_algorithm_functions(proposals.proposal_1)
            missing_2 = _missing_algorithm_functions(proposals.proposal_2)
            unknown_1 = _unknown_hyperparameter_keys(proposals.proposal_1, keys_1)
            unknown_2 = _unknown_hyperparameter_keys(proposals.proposal_2, keys_2)
            has_assert_1 = _has_assert_statements(proposals.proposal_1)
            has_assert_2 = _has_assert_statements(proposals.proposal_2)
            has_try_1 = _has_try_except(proposals.proposal_1)
            has_try_2 = _has_try_except(proposals.proposal_2)
            if (
                missing_1 or missing_2 or unknown_1 or unknown_2
                or has_assert_1 or has_assert_2 or has_try_1 or has_try_2
                or missing_config_1 or missing_config_2
            ):
                log.warning(
                    "Attempt %s rejected by static checks. Proposal 1 missing %s, unknown keys %s, "
                    "has assert %s, has try/except %s, missing config %s; "
                    "Proposal 2 missing %s, unknown keys %s, has assert %s, has try/except %s, missing config %s",
                    attempt, missing_1, unknown_1, has_assert_1, has_try_1, missing_config_1,
                    missing_2, unknown_2, has_assert_2, has_try_2, missing_config_2,
                )
                rejection_reason = (
                    "Generated code did not implement the required algorithm.py interface, referenced "
                    "hyperparameter keys that don't exist in its own config.yaml, contained a forbidden "
                    "assert/try-except statement, or was missing a valid ```yaml config.yaml block entirely. "
                    f"Proposal 1 is missing functions: {missing_1 or 'nothing'}, unknown keys used: {unknown_1 or 'none'}, "
                    f"contains assert: {has_assert_1}, contains try/except: {has_try_1}, missing config: {missing_config_1}. "
                    f"Proposal 2 is missing functions: {missing_2 or 'nothing'}, unknown keys used: {unknown_2 or 'none'}, "
                    f"contains assert: {has_assert_2}, contains try/except: {has_try_2}, missing config: {missing_config_2}. "
                    f"Every proposal MUST define exactly these top-level functions: {', '.join(REQUIRED_ALGORITHM_FUNCTIONS)}. "
                    "Every proposal MUST include a ```yaml fenced config.yaml block that declares every "
                    "hyperparameter key its code references. "
                    "Remove any assert/isinstance/type-check statements or try/except blocks about model, observation, "
                    "or Q-values - you were not shown their concrete implementation, and errors must propagate, not be hidden."
                )
                continue

            result = self.judge.evaluate(
                proposal_1=proposals.proposal_1,
                proposal_2=proposals.proposal_2,
                config_1=proposals.config_1,
                config_2=proposals.config_2,
                gen_id=self.gen_id
            )

            if result["winner"]:
                if result.get("training_error"):
                    log.info("Attempt %s succeeded (with a training issue): %s", attempt, result["training_error"])
                else:
                    log.info("Attempt %s succeeded: Judge selected %s", attempt, result["winner_id"])
                is_winner_1 = result["winner_id"] == "proposal_1"
                result["winner_code"] = proposals.proposal_1 if is_winner_1 else proposals.proposal_2
                result["winner_config"] = proposals.config_1 if is_winner_1 else proposals.config_2
                return result

            if result.get("training_error"):
                log.warning("Attempt %s: %s. Retrying.", attempt, result["training_error"])
                rejection_reason = result["training_error"]
                continue

            log.info("Attempt %s: Judge found no clear winner. Metrics: %s", attempt, result["metrics"])
            rejection_reason = f"Lack of sufficient winner. Metrics of both models: {result['metrics']}"

        log.error("Failed to satisfy Judge within %s attempts; continuing with the next generation.", max_judge_retries)
        return None