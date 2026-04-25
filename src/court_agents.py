import os
import logging
from pathlib import Path
from typing import Dict, Any
from llm_client import create_client

logger = logging.getLogger(__name__)

def load_prompt(filename: str) -> str:
    prompt_path = Path(__file__).parent.parent / "prompts" / filename
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()

class CourtArenaAgents:
    def __init__(self, pros_model: str, def_model: str, eval_model: str, judge_model: str):
        self.pros_client = create_client(model_name=pros_model, temperature=0.7, max_tokens=2048)
        self.def_client = create_client(model_name=def_model, temperature=0.7, max_tokens=2048)
        self.eval_client = create_client(model_name=eval_model, temperature=0.2, max_tokens=1024)
        self.judge_client = create_client(model_name=judge_model, temperature=0.5, max_tokens=2048)
        
        self.pros_model_name = pros_model
        self.def_model_name = def_model
        self.eval_model_name = eval_model
        self.judge_model_name = judge_model
        
        # Load templates
        self.pros_template = load_prompt("1_prosecution.txt")
        self.def_template = load_prompt("2_defense.txt")
        self.eval_template = load_prompt("evaluator.txt")
        self.judge_template = load_prompt("3_judge.txt")
        self.fairness_clause = load_prompt("fairness_clause.txt")

    async def generate_prosecution(self, case: Dict[str, Any]) -> str:
        prompt = self.pros_template.format(
            case_prompt=case.get("prompt", ""),
            case_facts=case.get("facts", "")
        )
        messages = [{"role": "user", "content": prompt}]
        response = await self.pros_client.chat(messages=messages)
        return response.content

    async def evaluate_argument(self, case: Dict[str, Any], argument_text: str, prev_argument_text: str = "None") -> str:
        prompt = self.eval_template.format(
            case_facts=case.get("facts", ""),
            prev_argument_text=prev_argument_text,
            argument_text=argument_text
        )
        messages = [{"role": "user", "content": prompt}]
        response = await self.eval_client.chat(messages=messages)
        return response.content

    async def generate_defense(self, case: Dict[str, Any], prosecution_arg: str) -> str:
        prompt = self.def_template.format(
            case_facts=case.get("facts", ""),
            prosecution_opening_arg=prosecution_arg,
            fairness_clause=self.fairness_clause
        )
        messages = [{"role": "user", "content": prompt}]
        response = await self.def_client.chat(messages=messages)
        return response.content

    async def generate_judge_ruling(self, case: Dict[str, Any], pros_arg: str, pros_eval: str, def_arg: str, def_eval: str) -> str:
        prompt = self.judge_template.format(
            case_facts=case.get("facts", ""),
            prosecution_opening_arg=pros_arg,
            prosecution_evaluation=pros_eval,
            defense_opening_arg=def_arg,
            defense_evaluation=def_eval,
            fairness_clause=self.fairness_clause
        )
        messages = [
            {"role": "system", "content": "You are a fair and impartial judge."}, 
            {"role": "user", "content": prompt}
        ]
        response = await self.judge_client.chat(messages=messages)
        return response.content