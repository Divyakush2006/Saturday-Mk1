"""
Saturday MK1 Distillation Pipeline — Collect Training Data from Expert Models
=========================================================================
Generates diverse coding tasks, sends them to a target model API,
captures high-quality input/output pairs in ChatML JSONL format for SFT.

Usage:
    # Generate 50 samples using Claude API
    python scripts/distill_from_api.py --provider anthropic --count 50

    # Generate from OpenAI-compatible endpoint
    python scripts/distill_from_api.py --provider openai --model gpt-4o --count 100

    # Generate from local model (Ollama, vLLM, etc.)
    python scripts/distill_from_api.py --provider local --base-url http://localhost:8000/v1 --count 200

    # Resume from checkpoint
    python scripts/distill_from_api.py --provider anthropic --count 500 --resume

    # Dry run (generate tasks only, no API calls)
    python scripts/distill_from_api.py --dry-run --count 10
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("saturday-distill")


# ═══════════════════════════════════════════
# DATA TYPES
# ═══════════════════════════════════════════

@dataclass
class TaskItem:
    """A single coding task to send to the expert model."""
    task_id: str
    category: str
    expert_id: int
    task_type: str
    prompt: str
    difficulty: str
    tags: list[str] = field(default_factory=list)


@dataclass
class DistilledSample:
    """A collected input/output pair for training."""
    task_id: str
    category: str
    expert_id: int
    task_type: str
    difficulty: str
    tags: list[str]
    prompt: str
    response: str
    quality_score: float
    token_count_prompt: int
    token_count_response: int
    model_used: str
    latency_seconds: float
    collected_at: str


@dataclass
class QualityMetrics:
    """Quality assessment of a model response."""
    has_code: bool
    has_explanation: bool
    has_error_handling: bool
    code_blocks_count: int
    response_length: int
    has_type_hints: bool
    has_docstrings: bool
    has_imports: bool
    completeness_score: float  # 0-1
    overall_score: float  # 0-1


# ═══════════════════════════════════════════
# SYSTEM PROMPT
# ═══════════════════════════════════════════

SATURDAY_SYSTEM_PROMPT = (
    "You are Saturday MK1, an enterprise-grade coding AI assistant built for "
    "Fortune 500 companies. You write production-ready, secure, well-documented "
    "code that follows industry best practices.\n\n"
    "RULES:\n"
    "1. Always reason step-by-step before writing code\n"
    "2. Consider edge cases, error handling, and security implications\n"
    "3. Use proper type hints, docstrings, and logging\n"
    "4. Follow SOLID principles and clean architecture\n"
    "5. Never leave security vulnerabilities (SQLi, XSS, SSRF, etc.)\n"
    "6. Code must be production-ready — not prototypes\n"
    "7. Include comprehensive error handling with specific exception types\n"
    "8. Use async patterns where appropriate for I/O-bound operations\n"
    "9. Follow the principle of least privilege\n"
    "10. Provide clear explanations of design decisions"
)


# ═══════════════════════════════════════════
# QUALITY FILTER
# ═══════════════════════════════════════════

class QualityFilter:
    """
    Scores model outputs on coding quality metrics.
    Rejects responses below the threshold.
    """

    def __init__(self, min_score: float = 0.6):
        self.min_score = min_score
        self.stats = {"total": 0, "passed": 0, "rejected": 0}

    def evaluate(self, prompt: str, response: str) -> QualityMetrics:
        """Evaluate a response's quality across multiple dimensions."""
        self.stats["total"] += 1

        # Extract code blocks
        code_blocks = re.findall(r"```[\w]*\n(.*?)```", response, re.DOTALL)
        all_code = "\n".join(code_blocks)

        has_code = len(code_blocks) > 0
        has_explanation = len(response) > len(all_code) + 100 if has_code else len(response) > 200
        has_error_handling = any(
            kw in all_code for kw in ["try:", "except", "raise ", "Error(", "catch", "throw"]
        )
        has_type_hints = any(
            kw in all_code for kw in ["-> ", ": str", ": int", ": list", ": dict", ": bool",
                                       ": Optional", ": float", "List[", "Dict[", "Tuple["]
        )
        has_docstrings = '"""' in all_code or "'''" in all_code or "/**" in all_code
        has_imports = any(
            kw in all_code for kw in ["import ", "from ", "require(", "const "]
        )

        # Completeness scoring
        scores = []

        # Code presence (weighted heavily for coding tasks)
        scores.append(0.25 if has_code else 0.0)

        # Explanation quality
        scores.append(0.15 if has_explanation else 0.0)

        # Error handling
        scores.append(0.15 if has_error_handling else 0.05)

        # Type hints / documentation
        scores.append(0.10 if has_type_hints else 0.0)
        scores.append(0.10 if has_docstrings else 0.0)

        # Response length (penalize very short or absurdly long)
        resp_len = len(response)
        if resp_len < 200:
            scores.append(0.0)
        elif resp_len < 500:
            scores.append(0.05)
        elif resp_len < 5000:
            scores.append(0.15)
        elif resp_len < 20000:
            scores.append(0.10)
        else:
            scores.append(0.05)  # too verbose

        # Code block count (multiple blocks suggest structured response)
        if len(code_blocks) >= 3:
            scores.append(0.10)
        elif len(code_blocks) >= 1:
            scores.append(0.05)
        else:
            scores.append(0.0)

        completeness = min(sum(scores) / 0.85, 1.0)  # normalize to 0-1
        overall = round(completeness, 3)

        metrics = QualityMetrics(
            has_code=has_code,
            has_explanation=has_explanation,
            has_error_handling=has_error_handling,
            code_blocks_count=len(code_blocks),
            response_length=resp_len,
            has_type_hints=has_type_hints,
            has_docstrings=has_docstrings,
            has_imports=has_imports,
            completeness_score=completeness,
            overall_score=overall,
        )

        if overall >= self.min_score:
            self.stats["passed"] += 1
        else:
            self.stats["rejected"] += 1

        return metrics

    def passes(self, metrics: QualityMetrics) -> bool:
        """Check if metrics meet the minimum quality threshold."""
        return metrics.overall_score >= self.min_score

    def get_stats(self) -> dict:
        """Return filtering statistics."""
        return {
            **self.stats,
            "pass_rate": (
                round(self.stats["passed"] / self.stats["total"] * 100, 1)
                if self.stats["total"] > 0 else 0
            ),
        }


# ═══════════════════════════════════════════
# TASK GENERATOR
# ═══════════════════════════════════════════

class TaskGenerator:
    """
    Generates diverse coding tasks from templates.
    Supports loading from task_templates.json and dynamic variation.
    """

    def __init__(self, templates_path: Optional[str] = None):
        self.templates_path = templates_path or str(
            Path(__file__).parent / "task_templates.json"
        )
        self.templates = self._load_templates()
        self._used_ids: set[str] = set()

    def _load_templates(self) -> dict:
        """Load task templates from JSON file."""
        path = Path(self.templates_path)
        if not path.exists():
            log.warning(f"Templates file not found: {path}")
            return {"categories": {}}

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        total = sum(
            len(cat.get("templates", []))
            for cat in data.get("categories", {}).values()
        )
        log.info(f"Loaded {total} task templates from {path.name}")
        return data

    def generate_tasks(
        self,
        count: int,
        categories: Optional[list[str]] = None,
        difficulties: Optional[list[str]] = None,
    ) -> list[TaskItem]:
        """
        Generate a list of coding tasks.

        Args:
            count: Number of tasks to generate
            categories: Filter to specific categories (None = all)
            difficulties: Filter to specific difficulties (None = all)

        Returns:
            List of TaskItem objects
        """
        all_templates = []

        for cat_name, cat_data in self.templates.get("categories", {}).items():
            if categories and cat_name not in categories:
                continue

            expert_id = cat_data.get("expert_id", 0)

            for tmpl in cat_data.get("templates", []):
                if difficulties and tmpl.get("difficulty") not in difficulties:
                    continue

                all_templates.append(
                    TaskItem(
                        task_id=tmpl["id"],
                        category=cat_name,
                        expert_id=expert_id,
                        task_type=tmpl.get("type", "general"),
                        prompt=tmpl["prompt"],
                        difficulty=tmpl.get("difficulty", "medium"),
                        tags=tmpl.get("tags", []),
                    )
                )

        if not all_templates:
            log.error("No templates matched the filter criteria")
            return []

        # Cycle through templates if count > available
        tasks = []
        idx = 0
        while len(tasks) < count:
            tmpl = all_templates[idx % len(all_templates)]
            variation = idx // len(all_templates)

            task = TaskItem(
                task_id=f"{tmpl.task_id}_v{variation}" if variation > 0 else tmpl.task_id,
                category=tmpl.category,
                expert_id=tmpl.expert_id,
                task_type=tmpl.task_type,
                prompt=tmpl.prompt,
                difficulty=tmpl.difficulty,
                tags=tmpl.tags,
            )

            if task.task_id not in self._used_ids:
                self._used_ids.add(task.task_id)
                tasks.append(task)
            idx += 1

            # Safety valve
            if idx > count * 3:
                break

        log.info(
            f"Generated {len(tasks)} tasks "
            f"(categories: {categories or 'all'}, difficulties: {difficulties or 'all'})"
        )
        return tasks


# ═══════════════════════════════════════════
# MODEL PROVIDERS
# ═══════════════════════════════════════════

class ModelProvider:
    """Base class for model API providers."""

    def __init__(self, model: str, **kwargs):
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> tuple[str, float]:
        """
        Send a prompt to the model and return (response_text, latency_seconds).
        Must be implemented by subclasses.
        """
        raise NotImplementedError


class AnthropicProvider(ModelProvider):
    """Provider for Anthropic Claude API."""

    def __init__(self, model: str = "claude-opus-4-20250514", **kwargs):
        super().__init__(model, **kwargs)
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            log.warning("ANTHROPIC_API_KEY not set — API calls will fail")

    def generate(self, system_prompt: str, user_prompt: str) -> tuple[str, float]:
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")

        client = anthropic.Anthropic(api_key=self.api_key)

        start = time.time()
        response = client.messages.create(
            model=self.model,
            max_tokens=8192,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        latency = time.time() - start

        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text

        return text, latency


class OpenAIProvider(ModelProvider):
    """Provider for OpenAI-compatible APIs (OpenAI, vLLM, Ollama, etc.)."""

    def __init__(
        self,
        model: str = "gpt-4o",
        base_url: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.api_key = os.environ.get("OPENAI_API_KEY", "sk-placeholder")
        self.base_url = base_url

    def generate(self, system_prompt: str, user_prompt: str) -> tuple[str, float]:
        try:
            import openai
        except ImportError:
            raise ImportError("pip install openai")

        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        client = openai.OpenAI(**client_kwargs)

        start = time.time()
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=8192,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
        )
        latency = time.time() - start

        text = response.choices[0].message.content or ""
        return text, latency


class LocalProvider(ModelProvider):
    """Provider for local models via OpenAI-compatible endpoints."""

    def __init__(
        self,
        model: str = "qwen3.5",
        base_url: str = "http://localhost:8000/v1",
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.base_url = base_url
        self._openai = OpenAIProvider(
            model=model,
            base_url=base_url,
        )

    def generate(self, system_prompt: str, user_prompt: str) -> tuple[str, float]:
        return self._openai.generate(system_prompt, user_prompt)


class DryRunProvider(ModelProvider):
    """Mock provider for testing — returns synthetic responses."""

    def __init__(self, model: str = "dry-run", **kwargs):
        super().__init__(model, **kwargs)

    def generate(self, system_prompt: str, user_prompt: str) -> tuple[str, float]:
        # Generate a realistic-looking mock response
        mock = (
            "## Analysis\n\n"
            "Let me break this down step by step.\n\n"
            "### Step 1: Understanding the Problem\n\n"
            "The core issue here involves proper error handling and security.\n\n"
            "### Step 2: Implementation\n\n"
            "```python\n"
            'import logging\n'
            'from typing import Optional\n\n'
            'logger = logging.getLogger(__name__)\n\n\n'
            'class SecureHandler:\n'
            '    """Production-ready handler with security controls."""\n\n'
            '    def __init__(self, config: dict) -> None:\n'
            '        self.config = config\n'
            '        self._validated = False\n\n'
            '    def process(self, data: dict) -> dict:\n'
            '        """Process input with validation and error handling."""\n'
            '        try:\n'
            '            self._validate(data)\n'
            '            result = self._transform(data)\n'
            '            return {"status": "success", "data": result}\n'
            '        except ValueError as e:\n'
            '            logger.error(f"Validation failed: {e}")\n'
            '            raise\n'
            '        except Exception as e:\n'
            '            logger.critical(f"Unexpected error: {e}")\n'
            '            raise RuntimeError("Processing failed") from e\n'
            "```\n\n"
            "### Step 3: Security Considerations\n\n"
            "- Input validation prevents injection attacks\n"
            "- Type hints ensure correctness at dev time\n"
            "- Specific exception types avoid swallowing errors\n"
            "- Logging captures audit trail without exposing sensitive data\n"
        )
        time.sleep(0.1)  # Simulate latency
        return mock, 0.1


def get_provider(
    provider_name: str,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> ModelProvider:
    """Factory to create model providers."""
    providers = {
        "anthropic": lambda: AnthropicProvider(
            model=model or "claude-opus-4-20250514",
        ),
        "openai": lambda: OpenAIProvider(
            model=model or "gpt-4o",
            base_url=base_url,
        ),
        "local": lambda: LocalProvider(
            model=model or "qwen3.5",
            base_url=base_url or "http://localhost:8000/v1",
        ),
        "dry-run": lambda: DryRunProvider(model="dry-run"),
    }

    if provider_name not in providers:
        raise ValueError(
            f"Unknown provider '{provider_name}'. "
            f"Available: {list(providers.keys())}"
        )

    return providers[provider_name]()


# ═══════════════════════════════════════════
# DISTILLATION COLLECTOR
# ═══════════════════════════════════════════

class DistillationCollector:
    """
    Orchestrates the distillation pipeline:
    1. Generate tasks → 2. Send to model → 3. Filter quality → 4. Save JSONL
    """

    def __init__(
        self,
        provider: ModelProvider,
        output_dir: str = "./distillation_data",
        quality_threshold: float = 0.6,
        system_prompt: Optional[str] = None,
    ):
        self.provider = provider
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.quality_filter = QualityFilter(min_score=quality_threshold)
        self.system_prompt = system_prompt or SATURDAY_SYSTEM_PROMPT

        # Checkpoint tracking
        self.checkpoint_file = self.output_dir / ".checkpoint.json"
        self.completed_ids: set[str] = set()
        self._load_checkpoint()

        # Stats
        self.stats = {
            "total_tasks": 0,
            "api_calls": 0,
            "samples_collected": 0,
            "samples_rejected": 0,
            "total_prompt_tokens": 0,
            "total_response_tokens": 0,
            "total_latency": 0.0,
            "errors": 0,
        }

    def _load_checkpoint(self):
        """Load checkpoint to support resume."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, "r") as f:
                data = json.load(f)
                self.completed_ids = set(data.get("completed_ids", []))
            log.info(f"Resumed from checkpoint: {len(self.completed_ids)} tasks already done")

    def _save_checkpoint(self):
        """Save checkpoint for resume support."""
        with open(self.checkpoint_file, "w") as f:
            json.dump(
                {"completed_ids": list(self.completed_ids), "saved_at": datetime.now(timezone.utc).isoformat()},
                f,
            )

    def _estimate_tokens(self, text: str) -> int:
        """Rough token count estimation (4 chars ≈ 1 token)."""
        return max(1, len(text) // 4)

    def collect(
        self,
        tasks: list[TaskItem],
        batch_size: int = 10,
        delay_between: float = 0.5,
    ) -> Path:
        """
        Run the distillation pipeline.

        Args:
            tasks: List of tasks to process
            batch_size: Save checkpoint every N tasks
            delay_between: Seconds between API calls (rate limiting)

        Returns:
            Path to the output JSONL file
        """
        self.stats["total_tasks"] = len(tasks)

        # Filter out already-completed tasks
        pending = [t for t in tasks if t.task_id not in self.completed_ids]
        if len(pending) < len(tasks):
            log.info(f"Skipping {len(tasks) - len(pending)} already-completed tasks (resume mode)")

        # Output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"distilled_{timestamp}.jsonl"

        log.info(f"Starting distillation: {len(pending)} tasks → {output_file.name}")
        log.info(f"Provider: {self.provider.model} | Quality threshold: {self.quality_filter.min_score}")

        with open(output_file, "a", encoding="utf-8") as f:
            for i, task in enumerate(pending):
                try:
                    # Call the model
                    log.info(
                        f"[{i+1}/{len(pending)}] {task.category}/{task.task_type} "
                        f"({task.difficulty}) — {task.task_id}"
                    )

                    self.stats["api_calls"] += 1
                    response_text, latency = self.provider.generate(
                        self.system_prompt, task.prompt
                    )

                    # Quality check
                    metrics = self.quality_filter.evaluate(task.prompt, response_text)

                    prompt_tokens = self._estimate_tokens(task.prompt)
                    response_tokens = self._estimate_tokens(response_text)
                    self.stats["total_prompt_tokens"] += prompt_tokens
                    self.stats["total_response_tokens"] += response_tokens
                    self.stats["total_latency"] += latency

                    if self.quality_filter.passes(metrics):
                        # Save as ChatML format
                        sample = {
                            "messages": [
                                {"role": "system", "content": self.system_prompt},
                                {"role": "user", "content": task.prompt},
                                {"role": "assistant", "content": response_text},
                            ],
                            "metadata": {
                                "task_id": task.task_id,
                                "category": task.category,
                                "expert_id": task.expert_id,
                                "task_type": task.task_type,
                                "difficulty": task.difficulty,
                                "tags": task.tags,
                                "quality_score": metrics.overall_score,
                                "quality_details": asdict(metrics),
                                "model": self.provider.model,
                                "latency_seconds": round(latency, 2),
                                "prompt_tokens": prompt_tokens,
                                "response_tokens": response_tokens,
                                "collected_at": datetime.now(timezone.utc).isoformat(),
                                "source": "distilled",
                            },
                        }

                        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        f.flush()
                        self.stats["samples_collected"] += 1

                        log.info(
                            f"  ✅ Quality: {metrics.overall_score:.2f} | "
                            f"Code blocks: {metrics.code_blocks_count} | "
                            f"Tokens: {response_tokens} | "
                            f"Latency: {latency:.1f}s"
                        )
                    else:
                        self.stats["samples_rejected"] += 1
                        log.warning(
                            f"  ❌ Rejected (score: {metrics.overall_score:.2f} < "
                            f"{self.quality_filter.min_score})"
                        )

                    # Mark as complete
                    self.completed_ids.add(task.task_id)

                    # Checkpoint
                    if (i + 1) % batch_size == 0:
                        self._save_checkpoint()
                        self._log_progress(i + 1, len(pending))

                    # Rate limiting
                    if delay_between > 0 and i < len(pending) - 1:
                        time.sleep(delay_between)

                except KeyboardInterrupt:
                    log.warning("Interrupted — saving checkpoint")
                    self._save_checkpoint()
                    break
                except Exception as e:
                    self.stats["errors"] += 1
                    log.error(f"  💥 Error on {task.task_id}: {e}")
                    continue

        # Final checkpoint
        self._save_checkpoint()

        # Save run summary
        self._save_summary(output_file)

        return output_file

    def _log_progress(self, done: int, total: int):
        """Log progress update."""
        pct = done / total * 100
        avg_latency = (
            self.stats["total_latency"] / self.stats["api_calls"]
            if self.stats["api_calls"] > 0 else 0
        )
        eta_seconds = avg_latency * (total - done)
        eta_minutes = eta_seconds / 60

        log.info(
            f"  📊 Progress: {done}/{total} ({pct:.0f}%) | "
            f"Collected: {self.stats['samples_collected']} | "
            f"Rejected: {self.stats['samples_rejected']} | "
            f"ETA: {eta_minutes:.1f}min"
        )

    def _save_summary(self, output_file: Path):
        """Save run summary to JSON."""
        filter_stats = self.quality_filter.get_stats()

        summary = {
            "run_completed_at": datetime.now(timezone.utc).isoformat(),
            "output_file": str(output_file),
            "provider": self.provider.model,
            "stats": self.stats,
            "quality_filter": filter_stats,
            "avg_latency_seconds": round(
                self.stats["total_latency"] / max(1, self.stats["api_calls"]), 2
            ),
            "total_tokens_used": (
                self.stats["total_prompt_tokens"] + self.stats["total_response_tokens"]
            ),
            "estimated_cost_usd": self._estimate_cost(),
        }

        summary_file = output_file.with_suffix(".summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        log.info(f"\n{'='*60}")
        log.info(f"DISTILLATION COMPLETE")
        log.info(f"{'='*60}")
        log.info(f"  Samples collected: {self.stats['samples_collected']}")
        log.info(f"  Samples rejected:  {self.stats['samples_rejected']}")
        log.info(f"  Errors:            {self.stats['errors']}")
        log.info(f"  Pass rate:         {filter_stats['pass_rate']}%")
        log.info(f"  Avg latency:       {summary['avg_latency_seconds']}s")
        log.info(f"  Total tokens:      {summary['total_tokens_used']:,}")
        log.info(f"  Estimated cost:    ${summary['estimated_cost_usd']:.2f}")
        log.info(f"  Output:            {output_file}")
        log.info(f"  Summary:           {summary_file}")
        log.info(f"{'='*60}")

    def _estimate_cost(self) -> float:
        """Estimate API cost based on token usage."""
        model = self.provider.model.lower()

        # Pricing per million tokens (input, output)
        pricing = {
            "claude-opus": (5.0, 25.0),
            "claude-sonnet": (3.0, 15.0),
            "gpt-4o": (2.5, 10.0),
            "gpt-4": (10.0, 30.0),
            "gpt-3.5": (0.5, 1.5),
        }

        input_price, output_price = 0, 0
        for key, (ip, op) in pricing.items():
            if key in model:
                input_price, output_price = ip, op
                break

        cost = (
            self.stats["total_prompt_tokens"] / 1_000_000 * input_price
            + self.stats["total_response_tokens"] / 1_000_000 * output_price
        )
        return round(cost, 4)


# ═══════════════════════════════════════════
# DATASET MERGER
# ═══════════════════════════════════════════

class DatasetMerger:
    """
    Merges distilled data with existing MoE training data.
    Converts all data to unified ChatML JSONL format.
    """

    def __init__(self, training_data_dir: str = "./training_data"):
        self.training_data_dir = Path(training_data_dir)

    def convert_moe_to_chatml(self, expert_dir: str) -> list[dict]:
        """
        Convert existing MoE JSONL chunks to ChatML format.

        Existing format: {"text": "...", "prompt": "...", "quality_score": ..., ...}
        Target format: {"messages": [...], "metadata": {...}}
        """
        expert_path = self.training_data_dir / expert_dir
        samples = []

        for chunk_file in sorted(expert_path.glob("chunk_*.jsonl")):
            with open(chunk_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        chatml = {
                            "messages": [
                                {"role": "system", "content": SATURDAY_SYSTEM_PROMPT},
                                {"role": "user", "content": item.get("prompt", "")},
                                {"role": "assistant", "content": item.get("text", "")},
                            ],
                            "metadata": {
                                "source": "moe_training_data",
                                "expert": expert_dir,
                                "quality_score": item.get("quality_score", 0.95),
                                "language": item.get("language", "python"),
                            },
                        }
                        samples.append(chatml)
                    except json.JSONDecodeError:
                        continue

        log.info(f"Converted {len(samples)} samples from {expert_dir}")
        return samples

    def merge(
        self,
        distilled_files: list[str],
        output_file: str = "merged_training_data.jsonl",
        include_moe: bool = True,
        moe_experts: Optional[list[str]] = None,
    ) -> Path:
        """
        Merge distilled data with existing MoE training data.

        Args:
            distilled_files: Paths to distilled JSONL files
            output_file: Output merged file name
            include_moe: Whether to include existing MoE training data
            moe_experts: Which expert directories to include (None = all)

        Returns:
            Path to merged JSONL file
        """
        output_path = self.training_data_dir / output_file
        total = 0

        with open(output_path, "w", encoding="utf-8") as out:
            # Add distilled data
            for df in distilled_files:
                df_path = Path(df)
                if not df_path.exists():
                    log.warning(f"Distilled file not found: {df}")
                    continue

                with open(df_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            out.write(line + "\n")
                            total += 1

                log.info(f"Added distilled data from {df_path.name}")

            # Add existing MoE data
            if include_moe:
                experts = moe_experts or [
                    d.name for d in self.training_data_dir.iterdir()
                    if d.is_dir() and (d / "chunk_0000.jsonl").exists()
                ]

                for expert in experts:
                    samples = self.convert_moe_to_chatml(expert)
                    for sample in samples:
                        out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        total += 1

        log.info(f"Merged dataset: {total:,} samples → {output_path}")
        return output_path


# ═══════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Saturday MK1 Distillation Pipeline — Collect training data from expert models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "local", "dry-run"],
        default="dry-run",
        help="Model provider to use (default: dry-run)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model name (e.g., claude-opus-4-20250514, gpt-4o, qwen3.5)",
    )
    parser.add_argument(
        "--base-url", type=str, default=None,
        help="Base URL for OpenAI-compatible endpoints",
    )
    parser.add_argument(
        "--count", type=int, default=10,
        help="Number of samples to collect (default: 10)",
    )
    parser.add_argument(
        "--categories", nargs="+", default=None,
        help="Filter to specific categories (e.g., security enterprise debug)",
    )
    parser.add_argument(
        "--difficulties", nargs="+", default=None,
        help="Filter to specific difficulties (easy, medium, hard, expert)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./distillation_data",
        help="Output directory for collected data",
    )
    parser.add_argument(
        "--quality-threshold", type=float, default=0.6,
        help="Minimum quality score to keep a sample (0-1, default: 0.6)",
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Delay between API calls in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from previous checkpoint",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Dry run mode — uses mock responses",
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Merge distilled data with existing MoE training data",
    )
    parser.add_argument(
        "--merge-files", nargs="+", default=None,
        help="Paths to distilled JSONL files to merge",
    )

    args = parser.parse_args()

    # Force dry-run provider if --dry-run flag set
    if args.dry_run:
        args.provider = "dry-run"

    log.info("=" * 60)
    log.info("SATURDAY MK1 DISTILLATION PIPELINE")
    log.info("=" * 60)

    # Handle merge mode
    if args.merge:
        if not args.merge_files:
            log.error("--merge requires --merge-files with paths to distilled JSONL files")
            sys.exit(1)

        merger = DatasetMerger(training_data_dir="./training_data")
        output = merger.merge(
            distilled_files=args.merge_files,
            include_moe=True,
        )
        log.info(f"Merged dataset saved to: {output}")
        return

    # Create provider
    provider = get_provider(
        args.provider,
        model=args.model,
        base_url=args.base_url,
    )
    log.info(f"Provider: {args.provider} ({provider.model})")

    # Generate tasks
    generator = TaskGenerator()
    tasks = generator.generate_tasks(
        count=args.count,
        categories=args.categories,
        difficulties=args.difficulties,
    )

    if not tasks:
        log.error("No tasks generated — check filters")
        sys.exit(1)

    # Run collection
    collector = DistillationCollector(
        provider=provider,
        output_dir=args.output_dir,
        quality_threshold=args.quality_threshold,
    )

    output_file = collector.collect(
        tasks=tasks,
        delay_between=args.delay,
    )

    log.info(f"\nOutput saved to: {output_file}")


if __name__ == "__main__":
    main()
