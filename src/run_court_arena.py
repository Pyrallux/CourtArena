import os
import json
import asyncio
import logging
import argparse
import random
from pathlib import Path
import yaml

from court_agents import CourtArenaAgents

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

def get_agent_models():
    config_path = Path(__file__).parent / 'model_config.yaml'
    default_fallback = "glm-5.1:cloud"
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            agents = config.get('arena_agents', {})
            return (
                agents.get('prosecution', default_fallback),
                agents.get('defense', default_fallback),
                agents.get('evaluator', default_fallback),
                agents.get('judge', default_fallback)
            )
    except Exception as e:
        logger.warning(f"Could not read agent models from model_config.yaml: {e}")
        return default_fallback, default_fallback, default_fallback, default_fallback

async def run_arena_on_case(case: dict, agents: CourtArenaAgents) -> dict:
    case_name = f"Case {case.get('id')}"
    logger.info(f"--- Starting CourtArena for Case: {case_name} ---")

    # Set up logging for this specific run
    logs_dir = Path(__file__).parent.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = logs_dir / f"arena_log_case_{case.get('id')}.txt"
    
    with open(log_file_path, "w", encoding="utf-8") as lf:
        
        def write_log(stage, model_used, content):
            lf.write(f"\n{'='*60}\n")
            lf.write(f"STAGE: {stage}\n")
            lf.write(f"MODEL: {model_used}\n")
            lf.write(f"{'='*60}\n")
            lf.write(str(content) + "\n")
            
        lf.write(f"COURT ARENA LOG - {case_name}\n")
        lf.write(f"FACTS:\n{case.get('facts', '')}\n")
        
        # 1 - Prompt Prosecution
        logger.info(f"Step 1: Generating Prosecution Argument...")
        pros_arg = await agents.generate_prosecution(case)
        write_log("1. Prosecution Argument", agents.pros_model_name, pros_arg)
        
        # 1.5 - Evaluate Prosecution (no previous arguments)
        logger.info(f"Step 1.5: Evaluating Prosecution Argument...")
        pros_eval = await agents.evaluate_argument(case, pros_arg, prev_argument_text="None")
        write_log("1.5. Evaluator (Prosecution)", agents.eval_model_name, pros_eval)
        
        # 2 - Prompt Defense
        logger.info(f"Step 2: Generating Defense Argument...")
        def_arg = await agents.generate_defense(case, pros_arg)
        write_log("2. Defense Argument", agents.def_model_name, def_arg)
        
        # 2.5 - Evaluate Defense (includes past arguments)
        logger.info(f"Step 2.5: Evaluating Defense Argument...")
        def_eval = await agents.evaluate_argument(case, def_arg, prev_argument_text=f"Prosecution's Argument:\n{pros_arg}")
        write_log("2.5. Evaluator (Defense)", agents.eval_model_name, def_eval)
        
        # 3 - Prompt Judge
        logger.info(f"Step 3: Generating Judge Ruling...")
        judge_ruling = await agents.generate_judge_ruling(
            case, pros_arg, pros_eval, def_arg, def_eval
        )
        write_log("3. Judge Preliminary Ruling", agents.judge_model_name, judge_ruling)

        # 3.5 - Evaluate Judge (include argument history)
        logger.info(f"Step 3.5: Evaluating Judge Ruling...")
        judgement_history = (
            f"Prosecution's Argument:\n{pros_arg}\n\n"
            f"Defense's Argument:\n{def_arg}\n"
        )
        judge_eval = await agents.evaluate_argument(case, judge_ruling, prev_argument_text=judgement_history)
        write_log("3.5. Evaluator (Judge)", agents.eval_model_name, judge_eval)

    logger.info(f"--- Completed CourtArena for Case: {case_name} ---")
    
    return {
        "case_id": case.get("id"),
        "case_name": case_name,
        "prosecution_argument": pros_arg,
        "prosecution_evaluation": pros_eval,
        "defense_argument": def_arg,
        "defense_evaluation": def_eval,
        "judge_ruling": judge_ruling,
        "judge_evaluation": judge_eval,
        "log_file": str(log_file_path)
    }

async def main():
    parser = argparse.ArgumentParser(description="Run CourtArena Multi-Agent Evaluation")
    parser.add_argument("--max-cases", type=int, default=3, help="Max cases to evaluate as a batch")
    args = parser.parse_args()

    pros_model, def_model, eval_model, judge_model = get_agent_models()

    logger.info(f"Using Prosecution model: {pros_model}")
    logger.info(f"Using Defense model: {def_model}")
    logger.info(f"Using Evaluator model: {eval_model}")
    logger.info(f"Using Judge model: {judge_model}")

    case_data_path = Path(__file__).parent.parent / "case_data" / "sampled_cases.json"
    if not case_data_path.exists():
        logger.error(f"Case data not found at {case_data_path}. Please run get_cases.py first.")
        return

    with open(case_data_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    max_cases = min(len(cases), args.max_cases)
    logger.info(f"Loaded {len(cases)} total cases limit. Randomly sampling {max_cases} instances.")
    cases_subset = random.sample(cases, max_cases)

    agents = CourtArenaAgents(pros_model, def_model, eval_model, judge_model)
    
    results = []
    for case in cases_subset:
        try:
            result = await run_arena_on_case(case, agents)
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing case {case.get('name')}: {e}")

    # Step 7 (Pipeline hook): Save outcomes for next-stage analysis
    out_dir = Path(__file__).parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "arena_results.json"
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Saved CourtArena agentic pipeline decisions to {out_file}")

if __name__ == "__main__":
    asyncio.run(main())