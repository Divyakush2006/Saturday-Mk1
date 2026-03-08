"""
Saturday MK1 Dataset Preparation — MoE Training Data Pipeline
==============================================================
Downloads, processes, and structures training data for Saturday's
8 Mixture-of-Experts model. Supports HuggingFace datasets and
local JSONL files.

Usage:
    # Process all experts
    python scripts/prepare_datasets.py --all

    # Process specific expert
    python scripts/prepare_datasets.py --expert 1

    # Dry run (test pipeline without writing)
    python scripts/prepare_datasets.py --dry-run --all

    # Process local JSONL files
    python scripts/prepare_datasets.py --input-dir ./raw_data --all
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("saturday-datasets")


# ═══════════════════════════════════════════
# EXPERT CONFIGURATIONS
# ═══════════════════════════════════════════

EXPERT_DOMAINS = {
    0: {
        "name": "Backend & APIs",
        "hf_datasets": [
            "bigcode/the-stack-v2-dedup",
        ],
        "filter_langs": ["python", "javascript", "typescript", "java", "go"],
        "keywords": ["flask", "django", "fastapi", "express", "spring", "endpoint", "route", "api"],
    },
    1: {
        "name": "Security & Auth",
        "hf_datasets": [
            "bigcode/the-stack-v2-dedup",
        ],
        "filter_langs": ["python", "javascript", "java"],
        "keywords": ["jwt", "oauth", "encrypt", "hash", "owasp", "vulnerability", "auth", "security"],
    },
    2: {
        "name": "Data & ML",
        "hf_datasets": [
            "bigcode/the-stack-v2-dedup",
        ],
        "filter_langs": ["python", "sql", "r"],
        "keywords": ["pandas", "numpy", "tensorflow", "pytorch", "sql", "pipeline", "model", "dataset"],
    },
    3: {
        "name": "DevOps & Infra",
        "hf_datasets": [
            "bigcode/the-stack-v2-dedup",
        ],
        "filter_langs": ["python", "shell", "yaml"],
        "keywords": ["docker", "kubernetes", "terraform", "ci", "cd", "deploy", "pipeline", "container"],
    },
    4: {
        "name": "Frontend & UI",
        "hf_datasets": [
            "bigcode/the-stack-v2-dedup",
        ],
        "filter_langs": ["javascript", "typescript", "html", "css"],
        "keywords": ["react", "vue", "angular", "component", "dom", "css", "html", "frontend"],
    },
    5: {
        "name": "Systems & Performance",
        "hf_datasets": [
            "bigcode/the-stack-v2-dedup",
        ],
        "filter_langs": ["python", "go", "rust", "c", "cpp"],
        "keywords": ["async", "thread", "memory", "cache", "performance", "socket", "protocol", "concurrent"],
    },
    6: {
        "name": "Testing & QA",
        "hf_datasets": [
            "bigcode/the-stack-v2-dedup",
        ],
        "filter_langs": ["python", "javascript", "java"],
        "keywords": ["test", "assert", "mock", "coverage", "debug", "log", "monitor", "fixture"],
    },
    7: {
        "name": "Architecture & Design",
        "hf_datasets": [
            "bigcode/the-stack-v2-dedup",
        ],
        "filter_langs": ["python", "java", "typescript"],
        "keywords": ["pattern", "solid", "microservice", "refactor", "design", "architecture", "clean"],
    },
}


# ═══════════════════════════════════════════
# DATASET PROCESSOR
# ═══════════════════════════════════════════

class DatasetProcessor:
    """
    Processes and structures datasets for Saturday MoE training.
    """

    def __init__(self, output_dir: str = "./training_data", dry_run: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dry_run = dry_run
        self.stats = {
            "total_loaded": 0,
            "total_processed": 0,
            "total_written": 0,
            "by_expert": {i: 0 for i in range(8)},
        }

    def process_expert(
        self,
        expert_id: int,
        input_dir: Optional[str] = None,
        max_samples: int = 10000,
    ) -> dict:
        """
        Process training data for a specific expert.

        Args:
            expert_id: Expert ID (0-7)
            input_dir: Directory with JSONL files (None = try HuggingFace)
            max_samples: Maximum samples to process

        Returns:
            Processing statistics
        """
        if expert_id not in EXPERT_DOMAINS:
            log.error(f"Unknown expert ID: {expert_id}")
            return {"error": f"Unknown expert {expert_id}"}

        config = EXPERT_DOMAINS[expert_id]
        log.info(f"Processing Expert {expert_id}: {config['name']}")
        log.info(f"  Keywords: {config['keywords'][:5]}")
        log.info(f"  Languages: {config['filter_langs']}")

        records = []

        # Try loading from local files first
        if input_dir:
            records = self._load_local_data(input_dir, config)
        else:
            # Try HuggingFace datasets
            records = self._load_from_huggingface(config, max_samples)

        if not records:
            log.warning(f"  No data found for Expert {expert_id}")
            return {"expert_id": expert_id, "loaded": 0, "written": 0}

        log.info(f"  Loaded {len(records)} raw records")
        self.stats["total_loaded"] += len(records)

        # Filter by keywords
        filtered = self._filter_by_keywords(records, config["keywords"])
        log.info(f"  After keyword filter: {len(filtered)}")

        # Score quality
        scored = self._score_records(filtered)
        log.info(f"  After quality filter: {len(scored)}")

        # Convert to ChatML format
        chatml = self._to_chatml(scored, expert_id)
        log.info(f"  ChatML formatted: {len(chatml)}")

        # Limit to max_samples
        chatml = chatml[:max_samples]

        # Write output
        if not self.dry_run:
            output_file = self._write_expert_data(expert_id, chatml)
            log.info(f"  Written to: {output_file}")
        else:
            log.info(f"  DRY RUN — would write {len(chatml)} records")

        self.stats["total_processed"] += len(chatml)
        self.stats["total_written"] += len(chatml)
        self.stats["by_expert"][expert_id] = len(chatml)

        return {
            "expert_id": expert_id,
            "name": config["name"],
            "loaded": len(records),
            "filtered": len(filtered),
            "scored": len(scored),
            "written": len(chatml),
        }

    def process_all_experts(
        self, input_dir: Optional[str] = None, max_samples: int = 10000
    ) -> dict:
        """Process data for all 8 experts."""
        results = {}
        for expert_id in range(8):
            results[expert_id] = self.process_expert(expert_id, input_dir, max_samples)
        return results

    # ── Internal Methods ──

    def _load_local_data(self, input_dir: str, config: dict) -> list[dict]:
        """Load data from local JSONL files."""
        records = []
        input_path = Path(input_dir)

        if not input_path.exists():
            log.warning(f"  Input directory not found: {input_path}")
            return records

        for jsonl_file in input_path.glob("*.jsonl"):
            try:
                with open(jsonl_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                records.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
            except OSError as e:
                log.warning(f"  Error reading {jsonl_file}: {e}")

        return records

    def _load_from_huggingface(self, config: dict, max_samples: int) -> list[dict]:
        """Attempt to load from HuggingFace datasets."""
        try:
            from datasets import load_dataset
        except ImportError:
            log.info("  HuggingFace datasets not installed — use --input-dir for local data")
            log.info("  Install: pip install datasets")
            return []

        records = []
        for ds_name in config.get("hf_datasets", []):
            try:
                log.info(f"  Loading from HuggingFace: {ds_name}")
                ds = load_dataset(ds_name, split="train", streaming=True)

                count = 0
                for item in ds:
                    if count >= max_samples:
                        break

                    lang = item.get("lang", item.get("language", ""))
                    if lang and lang.lower() not in config["filter_langs"]:
                        continue

                    content = item.get("content", item.get("text", item.get("code", "")))
                    if content and len(content) > 100:
                        records.append({
                            "text": content,
                            "language": lang,
                            "source": ds_name,
                        })
                        count += 1

                log.info(f"  Loaded {count} records from {ds_name}")

            except Exception as e:
                log.warning(f"  Failed to load {ds_name}: {e}")

        return records

    def _filter_by_keywords(self, records: list[dict], keywords: list[str]) -> list[dict]:
        """Filter records by keyword presence."""
        filtered = []
        for r in records:
            content = json.dumps(r).lower()
            matches = sum(1 for kw in keywords if kw in content)
            if matches >= 1:  # At least 1 keyword match
                r.setdefault("metadata", {})["keyword_matches"] = matches
                filtered.append(r)
        return filtered

    def _score_records(self, records: list[dict], min_score: float = 0.4) -> list[dict]:
        """Score records for quality."""
        scored = []
        for r in records:
            content = json.dumps(r)
            score = 0.0

            if "```" in content or "def " in content or "class " in content:
                score += 0.3
            if len(content) > 500:
                score += 0.2
            if "import " in content or "from " in content:
                score += 0.1
            if "try:" in content or "except" in content or "catch" in content:
                score += 0.1
            if '"""' in content or "'''" in content:
                score += 0.1
            if "raise " in content or "throw " in content:
                score += 0.1
            if "->" in content or ": str" in content or ": int" in content:
                score += 0.1

            if score >= min_score:
                r.setdefault("metadata", {})["quality_score"] = round(score, 3)
                scored.append(r)

        return scored

    def _to_chatml(self, records: list[dict], expert_id: int) -> list[dict]:
        """Convert records to ChatML JSONL format."""
        chatml = []
        for r in records:
            # Extract the main content
            text = r.get("text", r.get("content", ""))
            prompt = r.get("prompt", "")

            if not text and not prompt:
                # Try messages format
                messages = r.get("messages", [])
                if messages:
                    chatml.append({
                        "messages": messages,
                        "metadata": {
                            **r.get("metadata", {}),
                            "expert_id": expert_id,
                            "source": "passthrough",
                        },
                    })
                continue

            if not prompt:
                prompt = f"Analyze and explain the following {r.get('language', 'code')} code, then improve it for production use."

            chatml.append({
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are Saturday MK1, an enterprise-grade coding AI. "
                            "Provide production-ready, secure, well-documented solutions."
                        ),
                    },
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": text},
                ],
                "metadata": {
                    **r.get("metadata", {}),
                    "expert_id": expert_id,
                    "source": r.get("source", "local"),
                },
            })

        return chatml

    def _write_expert_data(self, expert_id: int, records: list[dict]) -> Path:
        """Write processed data for an expert."""
        expert_dir = self.output_dir / f"expert_{expert_id}"
        expert_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = expert_dir / f"train_{timestamp}.jsonl"

        with open(output_file, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return output_file


# ═══════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Saturday MK1 Dataset Preparation — MoE Training Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--all", action="store_true", help="Process all 8 experts")
    parser.add_argument("--expert", type=int, choices=range(8), help="Process specific expert (0-7)")
    parser.add_argument("--input-dir", type=str, help="Directory with raw JSONL files")
    parser.add_argument("--output-dir", type=str, default="./training_data", help="Output directory")
    parser.add_argument("--max-samples", type=int, default=10000, help="Max samples per expert")
    parser.add_argument("--dry-run", action="store_true", help="Process without writing output")

    args = parser.parse_args()

    if not args.all and args.expert is None:
        parser.print_help()
        print("\nError: Specify --all or --expert <id>")
        sys.exit(1)

    log.info("=" * 60)
    log.info("SATURDAY MK1 DATASET PREPARATION")
    log.info("=" * 60)

    processor = DatasetProcessor(output_dir=args.output_dir, dry_run=args.dry_run)

    if args.all:
        results = processor.process_all_experts(
            input_dir=args.input_dir, max_samples=args.max_samples
        )
    else:
        results = {
            args.expert: processor.process_expert(
                args.expert, input_dir=args.input_dir, max_samples=args.max_samples
            )
        }

    # Summary
    log.info("\n" + "=" * 60)
    log.info("PREPARATION COMPLETE")
    log.info("=" * 60)

    for expert_id, result in results.items():
        name = EXPERT_DOMAINS.get(expert_id, {}).get("name", f"Expert {expert_id}")
        written = result.get("written", 0)
        log.info(f"  Expert {expert_id} ({name}): {written} samples")

    log.info(f"\n  Total processed: {processor.stats['total_processed']}")
    log.info(f"  Output dir: {args.output_dir}")

    if args.dry_run:
        log.info("  Mode: DRY RUN (no files written)")

    log.info("=" * 60)


if __name__ == "__main__":
    main()
