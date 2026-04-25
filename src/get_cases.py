#!/usr/bin/env python3
"""
CourtListener Case Sampler

Samples full cases from the CourtListener API based on jurisdiction and time buckets.
Filters cases by reasoning keywords and minimum word count.
Follows project conventions for logging and environment configuration.
"""

import os
import sys
import time
import json
import random
import logging
import argparse
import httpx
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

BASE_URL = "https://www.courtlistener.com/api/rest/v4"

# CourtListener commonly uses court string identifiers or IDs (e.g. 'cal', 'ny', 'tex', 'mass')
COURTS = ["cal", "ny", "tex", "mass"] 

TIME_BUCKETS = [
    ("1900-01-01", "1950-01-01"),
    ("1950-01-01", "1990-01-01"),
    ("1990-01-01", "2010-01-01"),
    ("2010-01-01", "2020-01-01"),
]

SAMPLES_PER_BUCKET = 5
MIN_WORDS = 2000

REASONING_KEYWORDS = [
    "we hold",
    "we find",
    "the court finds",
    "because",
    "therefore",
    "in light of",
    "it follows that",
    "we conclude",
]


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
# CourtListener Client
# ============================================================================

class CourtListenerClient:
    def __init__(self, token: str):
        self.token = token
        self.headers = {"Authorization": f"Token {self.token}"}
        self.session = httpx.Client(headers=self.headers, timeout=60.0)

    def fetch_clusters(self, court: str, start_date: str, end_date: str, page: int = 1) -> Dict[str, Any]:
        """Fetch a page of case clusters from CourtListener."""
        url = f"{BASE_URL}/clusters/"
        
        # CourtListener filtering uses query parameters such as docket__court__id, date_filed__gte
        params = {
            "docket__court__id": court,
            "date_filed__gte": start_date,
            "date_filed__lte": end_date,
            "page": page
        }
        
        response = self.session.get(url, params=params)
        if response.status_code == 404:
            return {"results": [], "count": 0}
        response.raise_for_status()
        return response.json()

    def fetch_opinion(self, opinion_url: str) -> Optional[Dict[str, Any]]:
        """Fetch a specific opinion data object from its API URL."""
        if not opinion_url.startswith("http"):
            # Prepend base domain if only path is provided
            domain = BASE_URL.split("/api")[0]
            opinion_url = f"{domain}{opinion_url}"
        
        response = self.session.get(opinion_url)
        if response.status_code == 200:
            return response.json()
        return None


# ============================================================================
# Sampler Logic
# ============================================================================

def extract_text(opinion_data: Dict[str, Any]) -> str:
    """Extract text from a CourtListener opinion."""
    if not opinion_data:
        return ""
        
    # CL opinions store text in `plain_text` or `html_with_citations`/`html`
    text = opinion_data.get("plain_text") or opinion_data.get("html_with_citations") or opinion_data.get("html") or ""
    return str(text)

def has_reasoning(text: str) -> bool:
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in REASONING_KEYWORDS)

def is_valid_opinion(text: str) -> bool:
    if not text:
        return False
        
    word_count = len(text.split())
    if word_count < MIN_WORDS:
        return False
        
    if not has_reasoning(text):
        return False
        
    return True

def stratified_sample_filtered(client: CourtListenerClient) -> List[Dict[str, Any]]:
    dataset = []

    for court in COURTS:
        for start, end in TIME_BUCKETS:
            logger.info(f"\nSampling Court: {court} | {start} → {end}")

            try:
                # Issue getting the total count directly via count=on to avoid pagination issues 
                total_url = f"{BASE_URL}/clusters/"
                total_params = {
                    "docket__court__id": court,
                    "date_filed__gte": start,
                    "date_filed__lte": end,
                    "count": "on"
                }
                total_resp = client.session.get(total_url, params=total_params)
                if total_resp.status_code == 404:
                    total = 0
                else:
                    total_resp.raise_for_status()
                    total = int(total_resp.json().get("count", 0))
            except Exception as e:
                logger.error(f"  ✗ Failed to fetch count for {court}: {e}")
                continue
                
            if total == 0:
                logger.warning(f"  ! No cases found for {court} in this date range.")
                continue
                
            # CL paginates results (assuming ~20 per page). Cap large searches for random pick pools
            max_pages = min(total // 20, 100) 

            collected = 0
            attempts = 0
            max_attempts = SAMPLES_PER_BUCKET * 10

            while collected < SAMPLES_PER_BUCKET and attempts < max_attempts:
                page = random.randint(1, max_pages if max_pages > 0 else 1)
                attempts += 1
                
                try:
                    page_data = client.fetch_clusters(court, start, end, page=page)
                except Exception as e:
                    logger.debug(f"Failed to fetch page {page}: {e}")
                    time.sleep(1)
                    continue

                results = page_data.get("results", [])
                if not results:
                    continue

                cluster = random.choice(results)
                
                # Fetch opinions linked to this cluster
                opinions_urls = cluster.get("sub_opinions", [])
                if not opinions_urls:
                    continue
                    
                # Evaluate the first available opinion linked to this cluster
                opinion_url = opinions_urls[0]
                try:
                    opinion_data = client.fetch_opinion(opinion_url)
                except Exception as e:
                    logger.debug(f"Failed to fetch opinion {opinion_url}: {e}")
                    continue

                text = extract_text(opinion_data)

                if is_valid_opinion(text):
                    word_count = len(text.split())
                    
                    dataset.append({
                        "id": cluster.get("id"),
                        "name": cluster.get("case_name"),
                        "date": cluster.get("date_filed"),
                        "jurisdiction": court,
                        "word_count": word_count,
                        "url": cluster.get("absolute_url"),
                        "text": text[:5000],  # truncate text for storage/memory constraints
                    })

                    collected += 1
                    logger.info(f"  ✓ Added: {cluster.get('case_name', 'Unknown')} ({word_count} words)")

                # Respect API rate limits globally
                time.sleep(0.5)

    return dataset


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CourtListener Case Sampler",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
                       
    args = parser.parse_args()

    global logger
    logger = setup_logging(args.verbose)

    logger.info("=" * 60)
    logger.info("CourtListener Case Sampler")
    logger.info("=" * 60)

    token = os.environ.get("COURTLISTENER_TOKEN")
    if not token:
        logger.error("COURTLISTENER_TOKEN not found in environment or .env file.")
        logger.error("Please add COURTLISTENER_TOKEN=your_token to your .env file.")
        sys.exit(1)
        
    logger.info("Client configured with API token.")

    client = CourtListenerClient(token)
    
    logger.info("Starting stratified sampling...")
    cases = stratified_sample_filtered(client)
    
    out_dir = Path(__file__).parent.parent / "case_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "sampled_cases.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)
    
    logger.info("\n" + "=" * 60)
    logger.info(f"FINAL DATASET SIZE: {len(cases)} cases")
    logger.info(f"SAVED TO: {out_file.resolve()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()