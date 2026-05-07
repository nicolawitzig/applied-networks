#!/usr/bin/env python3
"""
Classify r/ethz posts and comments by topic and tonality using a local Ollama LLM.
Mirrors the manual coding scheme from Volk et al. (2024): topic × tonality per item.

Input:  flat CSV from scrape_subreddit_content.py
Output: same CSV with topic and tonality columns appended, written incrementally so a
        partial run is not lost if interrupted.

Usage:
  python classify_content.py --input results/ethz_2024-01-01_2026-01-01_flat.csv
  python classify_content.py --input results/ethz_flat.csv --model mistral --host http://localhost:11434
"""

import argparse
import csv
import json
import os
import re
import sys
from typing import Dict, Iterator, List, Set, Tuple

import requests

VALID_TOPICS     = {"academic", "organizational", "other"}   # categories from Volk et al.
VALID_TONALITIES = {"positive", "neutral", "negative"}       # categories from Volk et al.
SKIP_TEXTS       = {"[deleted]", "[removed]"}                # Reddit tombstones — no signal
TEXT_LIMIT       = 500    # chars sent to the LLM; enough for topic/tone, keeps inference fast

PROMPT = """\
Classify this Reddit post or comment from r/ethz (ETH Zurich).

Text:
{text}

Reply with ONLY a JSON object on one line, no extra text:
{{"topic": "<academic|organizational|other>", "tonality": "<positive|neutral|negative>"}}

Definitions:
  academic       — research, teaching, exams, courses, theses, professors, science
  organizational — admin, jobs, events, deadlines, university news, infrastructure, housing
  other          — personal, off-topic, humor, greetings, unrelated to university
  positive       — praise, enthusiasm, gratitude, encouragement
  neutral        — factual, informational, question, neutral observation
  negative       — criticism, complaint, frustration, sarcasm
"""


def _build_prompt(text: str) -> str:
    """Truncate text and inject into classification prompt."""
    truncated = text[:TEXT_LIMIT] + ("…" if len(text) > TEXT_LIMIT else "")
    return PROMPT.format(text=truncated)


def _call_ollama(host: str, model: str, prompt: str) -> str:
    """POST one prompt to Ollama and return the raw text response."""
    resp = requests.post(
        f"{host}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def _parse_response(raw: str) -> Tuple[str, str]:
    """
    Extract topic and tonality from the LLM's JSON output.
    Asserts valid values — halts immediately on malformed or out-of-vocabulary responses.
    """
    match = re.search(r'\{[^}]+\}', raw)
    assert match, f"No JSON found in LLM response: {raw!r}"
    data = json.loads(match.group())
    topic    = data.get("topic",    "").strip().lower()
    tonality = data.get("tonality", "").strip().lower()
    assert topic    in VALID_TOPICS,     f"Unexpected topic {topic!r} in: {raw!r}"
    assert tonality in VALID_TONALITIES, f"Unexpected tonality {tonality!r} in: {raw!r}"
    return topic, tonality


def load_items(path: str) -> List[dict]:
    """Load all rows from the flat CSV; skip deleted/empty text."""
    items: List[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text = row["text"].strip()
            if text and text not in SKIP_TEXTS and len(text) >= 10:
                items.append(row)
    assert items, f"No usable rows in {path}"
    return items


def _already_classified(path: str) -> Set[str]:
    """Read IDs already present in a partial output file — enables resuming interrupted runs."""
    if not os.path.exists(path):
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {row["id"] for row in csv.DictReader(f) if row.get("id")}


def classify_items(
    items: List[dict],
    skip_ids: Set[str],
    host: str,
    model: str,
) -> Iterator[Tuple[dict, str, str]]:
    """Yield (item, topic, tonality) for each unclassified item via Ollama."""
    pending = [it for it in items if it["id"] not in skip_ids]
    n_total  = len(items)
    n_skip   = len(items) - len(pending)
    if n_skip:
        print(f"Resuming: skipping {n_skip} already-classified rows")

    for i, item in enumerate(pending, 1):
        prompt           = _build_prompt(item["text"])
        raw              = _call_ollama(host, model, prompt)
        topic, tonality  = _parse_response(raw)
        done             = n_skip + i
        print(f"  [{done}/{n_total}]  {item['type']:<8}  topic={topic:<15}  tone={tonality}", flush=True)
        yield item, topic, tonality


def _print_summary(out_path: str) -> None:
    """Print topic × tonality frequency table from the completed output CSV."""
    counts: Dict[Tuple[str, str], int] = {}
    with open(out_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row.get("topic", "?"), row.get("tonality", "?"))
            counts[key] = counts.get(key, 0) + 1

    print("\nClassification summary:")
    print(f"  {'topic':<16}  {'tonality':<12}  count")
    print(f"  {'-'*16}  {'-'*12}  -----")
    for (topic, tone), n in sorted(counts.items()):
        print(f"  {topic:<16}  {tone:<12}  {n}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Classify subreddit posts and comments by topic and tonality via Ollama.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input results/ethz_2024-01-01_2026-01-01_flat.csv
  %(prog)s --input results/ethz_flat.csv --model mistral
  %(prog)s --input results/ethz_flat.csv --model llama3.2 --host http://localhost:11434
        """,
    )
    parser.add_argument("--input",      required=True, help="Flat or clustered CSV to classify")
    parser.add_argument("--output-dir", default="results", help="Output directory (default: results/)")
    parser.add_argument("--model",      default="llama3.2",
                        help="Ollama model name (default: llama3.2). Must be pulled first via 'ollama pull <model>'.")
    parser.add_argument("--host",       default="http://localhost:11434",
                        help="Ollama API host (default: http://localhost:11434)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    stem     = os.path.splitext(os.path.basename(args.input))[0]
    out_path = os.path.join(args.output_dir, f"{stem}_classified.csv")

    items    = load_items(args.input)
    skip_ids = _already_classified(out_path)    # IDs already written in a previous run
    print(f"Loaded {len(items)} items from {args.input}")
    print(f"Model: {args.model}  |  Host: {args.host}")
    print(f"Output: {out_path}\n")

    # Determine output fieldnames — preserve any extra columns (e.g. cluster_id) from input
    extra_fields = [k for k in items[0].keys() if k not in ("topic", "tonality")]
    fieldnames   = extra_fields + ["topic", "tonality"]

    write_header = not os.path.exists(out_path)    # append if resuming, create fresh otherwise
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for item, topic, tonality in classify_items(items, skip_ids, args.host, args.model):
            writer.writerow({**item, "topic": topic, "tonality": tonality})
            f.flush()    # incremental flush — partial results survive a crash or keyboard interrupt

    _print_summary(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
