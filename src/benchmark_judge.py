#!/usr/bin/env python3
"""
Benchmark CourtArena's judge_ruling against CourtReasoner's gold-standard
analysis-report.md for each case using an LLM-as-judge.

Reads:
  - results/arena_results.json   (per-case judge_ruling produced by run_court_arena)
  - case_data/sampled_cases.json (per-case gold_answer added by get_courtreasoner_cases)

Writes:
  - results/judge_benchmark.csv  (per-case scores)
And prints aggregate stats (% outcome match, mean reasoning alignment).
"""

import argparse
import asyncio
import csv
import json
import logging
import re
import statistics
import sys
from pathlib import Path

import yaml

from llm_client import create_client

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


def load_prompt_template() -> str:
    prompt_path = Path(__file__).parent.parent / "prompts" / "judge_benchmark.txt"
    return prompt_path.read_text(encoding="utf-8")


def parse_benchmark_response(text: str):
    """Pull outcome_match (yes/no) and reasoning_alignment (0-4) out of the
    benchmark response. Returns (outcome_match, reasoning_alignment) or None
    if either is missing.
    """
    if not text:
        return None

    outcome_pattern = r'["\']?Outcome Match["\']?\s*[:,]\s*["\']?(yes|no)["\']?'
    align_pattern = r'["\']?Reasoning Alignment["\']?\s*[:,]\s*["\']?(\d+)["\']?'

    outcome_match = re.search(outcome_pattern, text, re.IGNORECASE)
    align_match = re.search(align_pattern, text, re.IGNORECASE)

    if not outcome_match or not align_match:
        return None

    outcome = outcome_match.group(1).lower()
    alignment = int(align_match.group(1))
    if alignment < 0 or alignment > 4:
        return None
    return outcome, alignment


def get_benchmark_model(default: str = "glm-4.7:cloud") -> str:
    config_path = Path(__file__).parent / 'model_config.yaml'
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
        agents = config.get('arena_agents', {}) or {}
        return agents.get('benchmark_judge') or agents.get('evaluator', default)
    except Exception as e:
        logger.warning(f"Could not read model_config.yaml ({e}); using {default}")
        return default


async def benchmark_case(client, template: str, case_facts: str, gold_answer: str, judge_ruling: str) -> str:
    prompt = template.format(
        case_facts=case_facts,
        gold_answer=gold_answer,
        judge_ruling=judge_ruling,
    )
    response = await client.chat(messages=[{"role": "user", "content": prompt}])
    return response.content


async def main():
    parser = argparse.ArgumentParser(description="Benchmark CourtArena judge against CourtReasoner gold-standard")
    parser.add_argument("--arena-results", default="results/arena_results.json")
    parser.add_argument("--cases", default="case_data/sampled_cases.json")
    parser.add_argument("--output", default="results/judge_benchmark.csv")
    parser.add_argument("--model", default=None, help="Override benchmark model (default: model_config evaluator)")
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    arena_path = Path(args.arena_results)
    cases_path = Path(args.cases)
    out_path = Path(args.output)
    if not arena_path.is_absolute():
        arena_path = repo_root / arena_path
    if not cases_path.is_absolute():
        cases_path = repo_root / cases_path
    if not out_path.is_absolute():
        out_path = repo_root / out_path

    if not arena_path.exists():
        logger.error(f"Arena results not found: {arena_path}. Run run_court_arena.py first.")
        sys.exit(1)
    if not cases_path.exists():
        logger.error(f"Sampled cases not found: {cases_path}. Run get_courtreasoner_cases.py first.")
        sys.exit(1)

    arena_results = json.loads(arena_path.read_text(encoding="utf-8"))
    sampled_cases = json.loads(cases_path.read_text(encoding="utf-8"))
    cases_by_id = {str(c.get("id")): c for c in sampled_cases}

    model_name = args.model or get_benchmark_model()
    logger.info(f"Using benchmark judge model: {model_name}")
    client = create_client(model_name=model_name, temperature=0.2, max_tokens=1024)

    template = load_prompt_template()

    rows = []
    skipped = 0
    for result in arena_results:
        case_id = str(result.get("case_id"))
        judge_ruling = result.get("judge_ruling", "")
        case = cases_by_id.get(case_id)
        if case is None:
            logger.warning(f"Case {case_id} not in sampled_cases.json; skipping")
            skipped += 1
            continue
        gold_answer = case.get("gold_answer", "")
        if not gold_answer:
            logger.warning(f"Case {case_id} has no gold_answer; skipping")
            skipped += 1
            continue
        if not judge_ruling:
            logger.warning(f"Case {case_id} has no judge_ruling; skipping")
            skipped += 1
            continue

        logger.info(f"Benchmarking case {case_id}...")
        try:
            response_text = await benchmark_case(
                client, template, case.get("facts", ""), gold_answer, judge_ruling
            )
        except Exception as e:
            logger.error(f"Case {case_id}: benchmark call failed ({e})")
            skipped += 1
            continue

        parsed = parse_benchmark_response(response_text)
        if parsed is None:
            logger.warning(f"Case {case_id}: could not parse benchmark scores")
            skipped += 1
            continue

        outcome, alignment = parsed
        rows.append({
            "case_id": case_id,
            "outcome_match": outcome,
            "reasoning_alignment": alignment,
            "rationale": response_text.strip(),
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["case_id", "outcome_match", "reasoning_alignment", "rationale"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Wrote {len(rows)} rows to {out_path} ({skipped} skipped)")

    if rows:
        match_rate = sum(1 for r in rows if r["outcome_match"] == "yes") / len(rows)
        mean_align = statistics.mean(r["reasoning_alignment"] for r in rows)
        print()
        print(f"Cases benchmarked:        {len(rows)}")
        print(f"Outcome match rate:       {match_rate * 100:.1f}%")
        print(f"Mean reasoning alignment: {mean_align:.2f} / 4")
    else:
        print("No cases were benchmarked.")


if __name__ == "__main__":
    asyncio.run(main())
