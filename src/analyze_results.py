#!/usr/bin/env python3
"""
Parse evaluator scores out of arena_results.json and write a per-evaluation
CSV plus a printed summary table of mean scores per role.
"""

import argparse
import csv
import json
import logging
import re
import statistics
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

AXES = [
    "Citation Relevance",
    "Constraints Extraction",
    "Argument Validity per Constraint",
]
REBUTTAL_AXES = ["Rebuttal Responsiveness", "Rebuttal Strength / Persuasiveness"]
FINAL_AXES = ["Issue Prioritization and Synthesis", "Closing Persuasiveness"]

ROLES = ["prosecution", "defense", "judge"]
ROUNDS = [1, 2, 3]


def extract_scores(eval_text, round, is_judge):
    """Pull the three rubric scores from an evaluator response.

    The evaluator prompt asks for a json block but the format is loose, so
    we just regex each axis name to the next digit. Returns a dict with all
    three axes, or None if any are missing.
    """
    if not eval_text:
        return None

    axes = AXES.copy()
    if not is_judge:
        if round == 2:
            axes += REBUTTAL_AXES
        elif round == 3:
            axes += FINAL_AXES

    scores = {}
    for axis in axes:
        # Allow either ':' or ',' between key and value (the prompt has a typo
        # that produces ',' for the third axis), and quoted or bare numbers.
        pattern = rf'["\']?{re.escape(axis)}["\']?\s*[:,]\s*["\']?(\d+)["\']?'
        match = re.search(pattern, eval_text, re.IGNORECASE)
        if match:
            scores[axis] = int(match.group(1))

    if len(scores) != len(axes):
        return None
    return scores


def parse_results(results):
    rows = []
    skipped = 0
    for case in results:
        case_id = case.get("case_id")
        rounds = case.get("rounds", [])
        for round in rounds:
            for role in ROLES:
                eval_text = round.get(f"{role}_evaluation", "")
                scores = extract_scores(
                    eval_text, round.get("round"), is_judge=(role == "judge")
                )
                if scores is None:
                    logger.warning(
                        f"Could not parse scores for case {case_id} ({role})"
                    )
                    skipped += 1
                    continue
                rows.append(
                    {
                        "case_id": case_id,
                        "role": role,
                        "citation_relevance": scores["Citation Relevance"],
                        "constraints_extraction": scores["Constraints Extraction"],
                        "argument_validity": scores["Argument Validity per Constraint"],
                        "rebuttal_responsiveness": scores.get(
                            "Rebuttal Responsiveness", None
                        ),
                        "rebuttal_strength": scores.get(
                            "Rebuttal Strength / Persuasiveness", None
                        ),
                        "issue_prioritization": scores.get(
                            "Issue Prioritization and Synthesis", None
                        ),
                        "closing_persuasiveness": scores.get(
                            "Closing Persuasiveness", None
                        ),
                    }
                )
    return rows, skipped


def write_csv(rows, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "role",
        "citation_relevance",
        "constraints_extraction",
        "argument_validity",
        "rebuttal_responsiveness",
        "rebuttal_strength",
        "issue_prioritization",
        "closing_persuasiveness",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows):
    if not rows:
        print("No scores parsed.")
        return

    def mean_or_none(values):
        present = [value for value in values if value is not None]
        return statistics.mean(present) if present else None

    print()
    print(
        f"{'Role':<12}{'Citation':>12}{'Constraints':>14}{'Validity':>12}{'Rebuttal Resp.':>20}{'Rebuttal Str.':>18}{'Issue Prior.':>20}{'Closing Pers.':>20}{'N':>6}"
    )
    print("-" * 136)
    for role in ROLES:
        role_rows = [r for r in rows if r["role"] == role]
        if not role_rows:
            print(
                f"{role:<12}{'-':>12}{'-':>14}{'-':>12}{'-':>20}{'-':>18}{'-':>20}{'-':>20}{0:>6}"
            )
            continue
        cit = statistics.mean(r["citation_relevance"] for r in role_rows)
        con = statistics.mean(r["constraints_extraction"] for r in role_rows)
        val = statistics.mean(r["argument_validity"] for r in role_rows)
        # For partial metrics, only average over rows where they were present (i.e. rounds 2 and 3 for non-judges)
        reb_res = mean_or_none(r["rebuttal_responsiveness"] for r in role_rows)
        reb_str = mean_or_none(r["rebuttal_strength"] for r in role_rows)
        issue = mean_or_none(r["issue_prioritization"] for r in role_rows)
        closing = mean_or_none(r["closing_persuasiveness"] for r in role_rows)
        print(
            f"{role:<12}"
            f"{cit:>12.2f}"
            f"{con:>14.2f}"
            f"{val:>12.2f}"
            f"{(f'{reb_res:.2f}' if reb_res is not None else '-'):>20}"
            f"{(f'{reb_str:.2f}' if reb_str is not None else '-'):>18}"
            f"{(f'{issue:.2f}' if issue is not None else '-'):>20}"
            f"{(f'{closing:.2f}' if closing is not None else '-'):>20}"
            f"{len(role_rows):>6}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate CourtArena evaluator scores"
    )
    parser.add_argument(
        "--input",
        default="results/arena_results.json",
        help="Path to arena_results.json",
    )
    parser.add_argument(
        "--output",
        default="results/scores.csv",
        help="Path to write per-evaluation CSV",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    in_path = Path(args.input)
    if not in_path.is_absolute():
        in_path = repo_root / in_path
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = repo_root / out_path

    if not in_path.exists():
        logger.error(f"Input not found: {in_path}")
        sys.exit(1)

    with open(in_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    rows, skipped = parse_results(results)
    write_csv(rows, out_path)

    logger.info(f"Wrote {len(rows)} rows to {out_path} ({skipped} skipped)")
    print_summary(rows)


if __name__ == "__main__":
    main()
