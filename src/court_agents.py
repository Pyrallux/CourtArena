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
        self.pros_rebuttal_template = load_prompt("4_prosecution_rebuttal.txt")
        self.def_rebuttal_template = load_prompt("5_defense_rebuttal.txt")
        self.second_judge_template = load_prompt("6_second_judgement.txt")
        self.pros_closing_template = load_prompt("7_prosecution_closing.txt")
        self.def_closing_template = load_prompt("8_defense_closing.txt")
        self.final_judge_template = load_prompt("9_final_judgement.txt")
        self.rebuttal_eval_template = load_prompt("rebutal_evaluator.txt")
        self.closing_eval_template = load_prompt("closing_arg_evaluator.txt")
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

    async def generate_prosecution_rebuttal(self, case: Dict[str, Any], defense_opening_arg: str) -> str:
        prompt = self.pros_rebuttal_template.format(
            case_prompt=case.get("prompt", ""),
            case_facts=case.get("facts", ""),
            defense_opening_arg=defense_opening_arg
        )
        messages = [{"role": "user", "content": prompt}]
        response = await self.pros_client.chat(messages=messages)
        return response.content

    async def evaluate_rebuttal(self, case: Dict[str, Any], rebuttal_text: str, prev_argument_text: str = "None", opposing_argument_text: str = "None") -> str:
        prompt = self.rebuttal_eval_template
        prompt = prompt.replace("{case_facts}", case.get("facts", ""))
        prompt = prompt.replace("{prev_argument_text}", prev_argument_text)
        prompt = prompt.replace("{opposing_argument_text}", opposing_argument_text)
        prompt += f"\n\nRebuttal to Evaluate:\n{rebuttal_text}"
        messages = [{"role": "user", "content": prompt}]
        response = await self.eval_client.chat(messages=messages)
        return response.content

    async def generate_defense_rebuttal(self, case: Dict[str, Any], prosecution_rebuttal: str) -> str:
        prompt = self.def_rebuttal_template.format(
            case_facts=case.get("facts", ""),
            prosecution_rebuttal=prosecution_rebuttal,
            fairness_clause=self.fairness_clause
        )
        messages = [{"role": "user", "content": prompt}]
        response = await self.def_client.chat(messages=messages)
        return response.content

    async def generate_second_judge_ruling(self, case: Dict[str, Any], pros_rebuttal: str, pros_rebuttal_eval: str, def_rebuttal: str, def_rebuttal_eval: str) -> str:
        prompt = self.second_judge_template.format(
            case_facts=case.get("facts", ""),
            prosecution_rebuttal=pros_rebuttal,
            prosecution_rebuttal_evaluation=pros_rebuttal_eval,
            defense_rebuttal=def_rebuttal,
            defense_rebuttal_evaluation=def_rebuttal_eval,
            fairness_clause=self.fairness_clause
        )
        messages = [
            {"role": "system", "content": "You are a fair and impartial judge."}, 
            {"role": "user", "content": prompt}
        ]
        response = await self.judge_client.chat(messages=messages)
        return response.content

    async def generate_prosecution_closing(self, case: Dict[str, Any], defense_rebuttal: str) -> str:
        prompt = self.pros_closing_template.format(
            case_prompt=case.get("prompt", ""),
            case_facts=case.get("facts", ""),
            defense_rebuttal=defense_rebuttal
        )
        messages = [{"role": "user", "content": prompt}]
        response = await self.pros_client.chat(messages=messages)
        return response.content

    async def evaluate_closing(self, case: Dict[str, Any], closing_text: str, prev_argument_text: str = "None", rebuttal_text: str = "None") -> str:
        prompt = self.closing_eval_template
        prompt = prompt.replace("{case_facts}", case.get("facts", ""))
        prompt = prompt.replace("{prev_argument_text}", prev_argument_text)
        prompt = prompt.replace("{rebuttal_text}", rebuttal_text)
        prompt += f"\n\nClosing Argument to Evaluate:\n{closing_text}"
        messages = [{"role": "user", "content": prompt}]
        response = await self.eval_client.chat(messages=messages)
        return response.content

    async def generate_defense_closing(self, case: Dict[str, Any], prosecution_closing_statement: str) -> str:
        prompt = self.def_closing_template.format(
            case_facts=case.get("facts", ""),
            prosecution_closing_statement=prosecution_closing_statement,
            fairness_clause=self.fairness_clause
        )
        messages = [{"role": "user", "content": prompt}]
        response = await self.def_client.chat(messages=messages)
        return response.content

    async def generate_final_judge_ruling(self, case: Dict[str, Any], pros_closing: str, pros_closing_eval: str, def_closing: str, def_closing_eval: str) -> str:
        prompt = self.final_judge_template.format(
            case_facts=case.get("facts", ""),
            prosecution_closing_statement=pros_closing,
            prosecution_closing_evaluation=pros_closing_eval,
            defense_closing_statement=def_closing,
            defense_closing_evaluation=def_closing_eval,
            fairness_clause=self.fairness_clause
        )
        messages = [
            {"role": "system", "content": "You are a fair and impartial judge."}, 
            {"role": "user", "content": prompt}
        ]
        response = await self.judge_client.chat(messages=messages)
        return response.content
