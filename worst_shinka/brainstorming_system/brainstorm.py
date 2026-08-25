import os
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from openai import OpenAI
from llm.client import get_client_llm
from .evaluation_adapter import BrainstormingEvaluationAdapter

client: OpenAI = get_client_llm()
# TODO refine prompts (Kalina)

@dataclass
class BrainstormResult:
    proposal_1: str
    proposal_2: str
    debate_history: List[Dict[str, str]]

class BrainstormingPipeline:
    def __init__(self, model_a: str, model_b: str, gen_id: int, config_path: str, max_debate_rounds: int = 2):
        self.model_a = model_a
        self.model_b = model_b
        self.max_debate_rounds = max_debate_rounds
        self.gen_id = gen_id
        self.config_path = config_path
    def _call_llm(self, model: str, system_prompt: str, user_prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content or ""

    def run_brainstorming(
        self, 
        parents_data: List[Dict], 
        judge_rejection_reason: Optional[str] = None
    ) -> BrainstormResult:
        context = f"Parent Code Candidates & Performance:\n{json.dumps(parents_data, indent=2)}\n"
        if judge_rejection_reason:
            context += f"\nCRITICAL: Previous proposals were REJECTED by Judge for: {judge_rejection_reason}. Address this!"

        debate_history = []

        b1_prompt = f"{context}\nPropose an improved algorithmic strategy combining the best of these parents."
        idea_1 = self._call_llm(self.model_a, "You are Expert Brainstormer 1. Focus on algorithmic efficiency.", b1_prompt)
        
        b2_prompt = f"{context}\nPropose a distinct alternative strategy combining the best of these parents."
        idea_2 = self._call_llm(self.model_b, "You are Expert Brainstormer 2. Focus on code simplicity and robust edge-case handling.", b2_prompt)
        
        debate_history.extend([{"agent": "Brainstormer 1", "content": idea_1}, {"agent": "Brainstormer 2", "content": idea_2}])

        for round_num in range(self.max_debate_rounds):
            critic_prompt = f"Parent Data:\n{context}\n\nProposed Ideas:\n1: {idea_1}\n2: {idea_2}\nIdentify trade-offs, potential bugs, or performance bottlenecks in both."
            critique = self._call_llm(self.model_a, "You are a Harsh Code Critic.", critic_prompt) # we choose model a - to be think through
            debate_history.append({"agent": f"Critic (Round {round_num+1})", "content": critique})

            idea_1 = self._call_llm(self.model_a, "Refine your approach based on the critique.", f"Your Idea:\n{idea_1}\nCritique:\n{critique}")
            idea_2 = self._call_llm(self.model_b, "Refine your approach based on the critique.", f"Your Idea:\n{idea_2}\nCritique:\n{critique}")

        prop1_code = self._call_llm(self.model_a, "Output clean Python code for Proposal 1 based on final consensus.", idea_1)
        prop2_code = self._call_llm(self.model_b, "Output clean Python code for Proposal 2 based on final consensus.", idea_2)

        return BrainstormResult(proposal_1=prop1_code, proposal_2=prop2_code, debate_history=debate_history)

class EvolutionWorkflow:
    def __init__(self, brainstormer: BrainstormingPipeline, judge: BrainstormingEvaluationAdapter):
        self.brainstormer = brainstormer
        self.judge = judge

    def execute_crossover(self, parents_data: List[Dict], max_judge_retries: int = 3):
        rejection_reason = None
        
        for _ in range(max_judge_retries):
            proposals = self.brainstormer.run_brainstorming(parents_data, rejection_reason)
            
            result = self.judge.evaluate(
                proposal_1=proposals.proposal_1, 
                proposal_2=proposals.proposal_2,
                config_path=self.brainstormer.config_path,
                gen_id=self.brainstormer.gen_id
            )
            
            if result["winner"]:
                print(f"Success! Judge selected proposal")
                return result
            
            print(f"Looping back. Metrics {result["metrics"]}")
            rejection_reason = f"Lack of sufficient winner. Metrics of both models: {result["metrics"]}"

        raise RuntimeError("Failed to satisfy Judge within max retries.")