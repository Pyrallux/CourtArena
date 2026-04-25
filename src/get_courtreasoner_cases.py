#!/usr/bin/env python3
"""
CourtReasoner Case Downloader

Downloads all `question.txt` files from the yale-nlp/CourtReasoner GitHub repository.
Stores them sequentially in the case_data directory and creates a `sampled_cases.json` 
to maintain compatibility with the CourtArena benchmark system.
"""

import sys
import json
import logging
import argparse
import httpx
import re
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# ============================================================================
# Parsing
# ============================================================================

def parse_case_text(raw_text: str):
    """
    Parses a case text by splitting strictly on '****'.
    Returns a dictionary with 'prompt' and 'facts', or None if the format is invalid.
    """
    # Split by **** which separates prompt from case text
    asterisk_split = re.split(r'\n+\s*(?:\*\s*){3,}\n+', raw_text, maxsplit=1)
    
    if len(asterisk_split) == 2:
        return {
            "prompt": asterisk_split[0].strip(),
            "facts": asterisk_split[1].strip()
        }
    else:
        # Exclude cases that do not follow this explicitly required format
        return None

# ============================================================================
# Logging
# ============================================================================

def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CourtReasoner Case Downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
                       
    args = parser.parse_args()

    global logger
    logger = setup_logging(args.verbose)

    logger.info("=" * 60)
    logger.info("CourtReasoner Case Downloader")
    logger.info("=" * 60)

    out_dir = Path(__file__).parent.parent / "case_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    error_dir = out_dir / "errors"
    if error_dir.exists():
        shutil.rmtree(error_dir)
    error_dir.mkdir(parents=True, exist_ok=True)
    
    api_url = "https://api.github.com/repos/yale-nlp/CourtReasoner/git/trees/main?recursive=1"
    
    cases = []
    
    with httpx.Client(timeout=60.0) as client:
        logger.info("Fetching repository tree from GitHub...")
        try:
            response = client.get(api_url)
            response.raise_for_status()
            tree = response.json().get("tree", [])
            
            question_files = [item["path"] for item in tree if item["path"].endswith("question.txt")]
            logger.info(f"Found {len(question_files)} question.txt files to download.")
            
            # Track correctly parsed cases
            valid_cases_count = 0
            error_cases_count = 0
            
            for idx, path in enumerate(question_files):
                raw_url = f"https://raw.githubusercontent.com/yale-nlp/CourtReasoner/main/{path}"
                logger.info(f"Downloading case {idx + 1}/{len(question_files)}...")
                
                resp = client.get(raw_url)
                resp.raise_for_status()
                text = resp.text
                
                parsed_data = parse_case_text(text)
                if not parsed_data:
                    error_cases_count += 1
                    logger.warning(f"Skipping case {idx + 1} - missing '****' separator. Saving to errors as error_case_{error_cases_count}.txt")
                    error_file = error_dir / f"error_case_{error_cases_count}.txt"
                    error_file.write_text(text, encoding="utf-8")
                    continue
                
                case_obj = {
                    "id": str(valid_cases_count),
                    "prompt": parsed_data["prompt"],
                    "facts": parsed_data["facts"]
                }
                
                cases.append(case_obj)
                valid_cases_count += 1
                
        except Exception as e:
            logger.error(f"Failed to fetch or process files from GitHub: {e}")

    # Output to sampled_cases.json for pipeline compatibility
    out_file = out_dir / "sampled_cases.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)
    
    logger.info("\n" + "=" * 60)
    logger.info(f"FINAL DATASET SIZE: {len(cases)} cases downloaded")
    logger.info(f"JSON DATASET SAVED TO: {out_file.resolve()}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()