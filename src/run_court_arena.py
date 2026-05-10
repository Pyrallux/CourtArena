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

NUM_ROUNDS = 3

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

    rounds = []

    with open(log_file_path, "w", encoding="utf-8") as lf:
        
        def write_log(stage, model_used, content):
            lf.write(f"\n{'='*60}\n")
            lf.write(f"STAGE: {stage}\n")
            lf.write(f"MODEL: {model_used}\n")
            lf.write(f"{'='*60}\n")
            lf.write(str(content) + "\n")
            
        lf.write(f"COURT ARENA LOG - {case_name}\n")
        lf.write(f"FACTS:\n{case.get('facts', '')}\n")

        prev_judge_ruling = None
        prev_pros_arg = None
        prev_def_arg = None

        for round_num in range(1, NUM_ROUNDS + 1):
            lf.write(f"\n{'#'*60}\n")
            lf.write(f"ROUND {round_num} of {NUM_ROUNDS}\n")
            lf.write(f"{'#'*60}\n")
            logger.info(f"=== Round {round_num}/{NUM_ROUNDS} ===")

            # Append the previous judge ruling to facts so agents have full context.
            round_case = dict(case)
            if prev_judge_ruling is not None:
                round_case["facts"] = (
                    case.get("facts", "") +
                    f"\n\n--- Judge's Ruling from Round {round_num - 1} ---\n{prev_judge_ruling}"
                )

            prefix = f"R{round_num}."

            if round_num == 1:
                # 1 - Prompt Prosecution
                logger.info(f"Step {prefix}1: Generating Prosecution Opening Argument...")
                pros_arg = await agents.generate_prosecution(round_case)
                write_log(f"{prefix}1. Prosecution Opening Argument", agents.pros_model_name, pros_arg)
                
                # 1.5 - Evaluate Prosecution (no previous arguments)
                logger.info(f"Step {prefix}1.5: Evaluating Prosecution Opening Argument...")
                pros_eval = await agents.evaluate_argument(round_case, pros_arg, prev_argument_text="None")
                write_log(f"{prefix}1.5. Evaluator (Prosecution Opening)", agents.eval_model_name, pros_eval)
                
                # 2 - Prompt Defense
                logger.info(f"Step {prefix}2: Generating Defense Opening Argument...")
                def_arg = await agents.generate_defense(round_case, pros_arg)
                write_log(f"{prefix}2. Defense Opening Argument", agents.def_model_name, def_arg)
                
                # 2.5 - Evaluate Defense (includes past arguments)
                logger.info(f"Step {prefix}2.5: Evaluating Defense Opening Argument...")
                def_eval = await agents.evaluate_argument(round_case, def_arg, prev_argument_text=f"Prosecution's Opening Argument:\n{pros_arg}")
                write_log(f"{prefix}2.5. Evaluator (Defense Opening)", agents.eval_model_name, def_eval)
                
                # 3 - Prompt Judge
                logger.info(f"Step {prefix}3: Generating Preliminary Judge Ruling...")
                judge_ruling = await agents.generate_judge_ruling(
                    round_case, pros_arg, pros_eval, def_arg, def_eval
                )
                write_log(f"{prefix}3. Preliminary Judge Ruling", agents.judge_model_name, judge_ruling)

            elif round_num == 2:
                prior_openings = (
                    f"Prosecution's Opening Argument:\n{prev_pros_arg}\n\n"
                    f"Defense's Opening Argument:\n{prev_def_arg}\n"
                )

                # 1 - Prompt Prosecution
                logger.info(f"Step {prefix}1: Generating Prosecution Rebuttal...")
                pros_arg = await agents.generate_prosecution_rebuttal(round_case, prev_def_arg)
                write_log(f"{prefix}1. Prosecution Rebuttal", agents.pros_model_name, pros_arg)
                
                # 1.5 - Evaluate Prosecution (with previous arguments)
                logger.info(f"Step {prefix}1.5: Evaluating Prosecution Rebuttal...")
                pros_eval = await agents.evaluate_rebuttal(
                    round_case,
                    pros_arg,
                    prev_argument_text=prior_openings,
                    opposing_argument_text=f"Defense's Opening Argument:\n{prev_def_arg}"
                )
                write_log(f"{prefix}1.5. Evaluator (Prosecution Rebuttal)", agents.eval_model_name, pros_eval)
                
                # 2 - Prompt Defense
                logger.info(f"Step {prefix}2: Generating Defense Rebuttal...")
                def_arg = await agents.generate_defense_rebuttal(round_case, pros_arg)
                write_log(f"{prefix}2. Defense Rebuttal", agents.def_model_name, def_arg)
                
                # 2.5 - Evaluate Defense (includes past arguments)
                logger.info(f"Step {prefix}2.5: Evaluating Defense Rebuttal...")
                def_eval = await agents.evaluate_rebuttal(
                    round_case,
                    def_arg,
                    prev_argument_text=f"{prior_openings}\nProsecution's Rebuttal:\n{pros_arg}",
                    opposing_argument_text=f"Prosecution's Rebuttal:\n{pros_arg}"
                )
                write_log(f"{prefix}2.5. Evaluator (Defense Rebuttal)", agents.eval_model_name, def_eval)
                
                # 3 - Prompt Judge
                logger.info(f"Step {prefix}3: Generating Second Judge Ruling...")
                judge_ruling = await agents.generate_second_judge_ruling(
                    round_case, pros_arg, pros_eval, def_arg, def_eval
                )
                write_log(f"{prefix}3. Second Judge Ruling", agents.judge_model_name, judge_ruling)

            else:
                prior_arguments = (
                    f"Prosecution's Opening Argument:\n{rounds[0].get('prosecution_argument', '')}\n\n"
                    f"Defense's Opening Argument:\n{rounds[0].get('defense_argument', '')}\n"
                )
                prior_rebuttals = (
                    f"Prosecution's Rebuttal:\n{prev_pros_arg}\n\n"
                    f"Defense's Rebuttal:\n{prev_def_arg}\n"
                )

                # 1 - Prompt Prosecution
                logger.info(f"Step {prefix}1: Generating Prosecution Closing Argument...")
                pros_arg = await agents.generate_prosecution_closing(round_case, prev_def_arg)
                write_log(f"{prefix}1. Prosecution Closing Argument", agents.pros_model_name, pros_arg)
                
                # 1.5 - Evaluate Prosecution (with previous arguments)
                logger.info(f"Step {prefix}1.5: Evaluating Prosecution Closing Argument...")
                pros_eval = await agents.evaluate_closing(
                    round_case,
                    pros_arg,
                    prev_argument_text=prior_arguments,
                    rebuttal_text=prior_rebuttals
                )
                write_log(f"{prefix}1.5. Evaluator (Prosecution Closing)", agents.eval_model_name, pros_eval)
                
                # 2 - Prompt Defense
                logger.info(f"Step {prefix}2: Generating Defense Closing Argument...")
                def_arg = await agents.generate_defense_closing(round_case, pros_arg)
                write_log(f"{prefix}2. Defense Closing Argument", agents.def_model_name, def_arg)
                
                # 2.5 - Evaluate Defense (includes past arguments)
                logger.info(f"Step {prefix}2.5: Evaluating Defense Closing Argument...")
                def_eval = await agents.evaluate_closing(
                    round_case,
                    def_arg,
                    prev_argument_text=f"{prior_arguments}\nProsecution's Closing Argument:\n{pros_arg}",
                    rebuttal_text=prior_rebuttals
                )
                write_log(f"{prefix}2.5. Evaluator (Defense Closing)", agents.eval_model_name, def_eval)
                
                # 3 - Prompt Judge
                logger.info(f"Step {prefix}3: Generating Final Judge Ruling...")
                judge_ruling = await agents.generate_final_judge_ruling(
                    round_case, pros_arg, pros_eval, def_arg, def_eval
                )
                write_log(f"{prefix}3. Final Judge Ruling", agents.judge_model_name, judge_ruling)

            # 3.5 - Evaluate Judge (include argument history)
            logger.info(f"Step {prefix}3.5: Evaluating Judge Ruling...")
            judgement_history = (
                f"Prosecution's Argument:\n{pros_arg}\n\n"
                f"Defense's Argument:\n{def_arg}\n"
            )
            judge_eval = await agents.evaluate_argument(round_case, judge_ruling, prev_argument_text=judgement_history)
            write_log(f"{prefix}3.5. Evaluator (Judge)", agents.eval_model_name, judge_eval)

            rounds.append({
                "round": round_num,
                "prosecution_argument": pros_arg,
                "prosecution_evaluation": pros_eval,
                "defense_argument": def_arg,
                "defense_evaluation": def_eval,
                "judge_ruling": judge_ruling,
                "judge_evaluation": judge_eval,
            })

            prev_judge_ruling = judge_ruling
            prev_pros_arg = pros_arg
            prev_def_arg = def_arg

    logger.info(f"--- Completed CourtArena for Case: {case_name} ---")
    
    return {
        "case_id": case.get("id"),
        "case_name": case_name,
        "rounds": rounds,
        "log_file": str(log_file_path)
    }

def _parse_num_cases(value: str):
    """Parse num-cases argument: either an integer or 'all'."""
    if value.lower() == "all":
        return "all"
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"num-cases must be an integer or 'all', got '{value}'")

async def main():
    parser = argparse.ArgumentParser(description="Run CourtArena Multi-Agent Evaluation")
    parser.add_argument("--num-cases", type=_parse_num_cases, default=3, help="Number of cases to evaluate as a batch (integer or 'all')")
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

    if args.num_cases == "all":
        max_cases = len(cases)
    else:
        max_cases = min(len(cases), args.num_cases)
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