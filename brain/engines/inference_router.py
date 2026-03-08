"""
Saturday Inference Router — 4-Tier Intelligent Task Classification
=================================================================
Multi-signal classification with cost-optimized routing for maximum
AI performance at minimum compute cost.

Tiers:
  Lightning (T1): Trivial tasks — naming, formatting, simple completions
  Fast (T2):      Standard tasks — refactoring, testing, documentation
  Balanced (T3):  Complex tasks — architecture, multi-file changes
  Deep (T4):      Critical tasks — security review, production deployment

Also includes MoE (Mixture of Experts) routing to specialist models.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class RoutingResult:
    """Result of routing a task through the inference router."""
    tier: int
    tier_name: str
    confidence: float
    signals: list[str] = field(default_factory=list)
    expert_model: str = ""    # for MoE routing
    estimated_tokens: int = 0
    cost_estimate: float = 0.0
    reasoning: str = ""


@dataclass
class RoutingStats:
    """Tracking stats for routing decisions."""
    total_routed: int = 0
    by_tier: dict = field(default_factory=lambda: {1: 0, 2: 0, 3: 0, 4: 0})
    by_expert: dict = field(default_factory=dict)
    avg_confidence: float = 0.0
    estimated_savings: float = 0.0


class InferenceRouter:
    """
    4-tier intelligent routing with MoE expert selection.

    Usage:
        router = InferenceRouter()
        result = router.route("Review this function for SQL injection", code)
        print(result.tier_name)  # "Deep (T4)"
        print(result.expert_model)  # "security_expert"
    """

    TIER_NAMES = {1: "Lightning", 2: "Fast", 3: "Balanced", 4: "Deep"}
    TIER_COSTS = {1: 0.001, 2: 0.005, 3: 0.02, 4: 0.08}  # cost per 1K tokens

    # Expert models for MoE routing
    EXPERTS = {
        "code_gen": {"keywords": ["write", "create", "generate", "build", "implement", "add"],
                    "weight": 1.0},
        "code_review": {"keywords": ["review", "check", "analyze", "audit", "quality"],
                       "weight": 1.0},
        "security": {"keywords": ["security", "vulnerability", "injection", "xss", "auth", "encrypt",
                                  "hack", "breach", "threat", "attack", "pentest"],
                    "weight": 1.5},
        "debug": {"keywords": ["fix", "bug", "error", "crash", "exception", "broken", "failing",
                               "debug", "issue", "problem", "not working"],
                 "weight": 1.0},
        "refactor": {"keywords": ["refactor", "clean", "optimize", "restructure", "improve",
                                  "simplify", "extract", "rename"],
                    "weight": 1.0},
        "architecture": {"keywords": ["architect", "design", "pattern", "structure", "system",
                                      "microservice", "scalab", "infra"],
                        "weight": 1.2},
        "testing": {"keywords": ["test", "spec", "assert", "mock", "coverage", "tdd",
                                "unit test", "integration"],
                   "weight": 1.0},
        "docs": {"keywords": ["document", "readme", "api doc", "comment", "explain", "docstring"],
                "weight": 0.8},
    }

    def __init__(self):
        self.stats = RoutingStats()
        self._tier_signals = self._build_tier_signals()

    def route(self, task: str, code: str = "") -> RoutingResult:
        """
        Route a task to the optimal tier and expert.

        Multi-signal classification:
        1. Keyword analysis (task description)
        2. Complexity estimation (code structure)
        3. Safety criticality (security/production)
        4. Expert matching (MoE)
        """
        task_lower = task.lower()
        combined = f"{task} {code}".lower()
        signals = []

        # Signal 1: Keyword-based tier estimation
        tier_scores = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        for tier, tier_data in self._tier_signals.items():
            for kw in tier_data["keywords"]:
                if kw in task_lower:
                    tier_scores[tier] += tier_data["weight"]
                    signals.append(f"keyword '{kw}' → T{tier}")

        # Signal 2: Code complexity
        if code:
            lines = len(code.split("\n"))
            funcs = len(re.findall(r"(?:def |function |func )", code))
            classes = len(re.findall(r"(?:class |struct |interface )", code))

            if lines > 500 or funcs > 15 or classes > 5:
                tier_scores[4] += 2.0
                signals.append(f"large code ({lines}L, {funcs}fn, {classes}cls) → T4")
            elif lines > 200 or funcs > 8:
                tier_scores[3] += 2.0
                signals.append(f"medium code ({lines}L, {funcs}fn) → T3")
            elif lines > 50:
                tier_scores[2] += 1.0
                signals.append(f"moderate code ({lines}L) → T2")
            else:
                tier_scores[1] += 1.0
                signals.append(f"small code ({lines}L) → T1")

        # Signal 3: Safety criticality
        safety_keywords = ["security", "auth", "production", "deploy", "migration",
                          "delete", "drop", "vulnerability", "credential", "secret"]
        safety_hits = sum(1 for kw in safety_keywords if kw in combined)
        if safety_hits >= 2:
            tier_scores[4] += 3.0
            signals.append(f"safety-critical ({safety_hits} indicators) → T4")
        elif safety_hits == 1:
            tier_scores[3] += 1.5
            signals.append(f"safety-relevant → T3")

        # Signal 4: Multi-file indicator
        multi_file_kw = ["across files", "project-wide", "all files", "multiple",
                         "codebase", "everywhere", "global"]
        if any(kw in task_lower for kw in multi_file_kw):
            tier_scores[3] += 2.0
            signals.append("multi-file scope → T3")

        # Select best tier
        best_tier = max(tier_scores, key=tier_scores.get)
        if tier_scores[best_tier] == 0:
            best_tier = 2  # default to Fast

        # Calculate confidence
        total_score = sum(tier_scores.values()) or 1
        confidence = round(tier_scores[best_tier] / total_score, 2)

        # Expert routing (MoE)
        expert = self._match_expert(task_lower)

        # Token estimation
        tokens = self._estimate_tokens(task, code)

        # Cost
        cost = tokens / 1000 * self.TIER_COSTS.get(best_tier, 0.02)

        # Update stats
        self.stats.total_routed += 1
        self.stats.by_tier[best_tier] = self.stats.by_tier.get(best_tier, 0) + 1
        if expert:
            self.stats.by_expert[expert] = self.stats.by_expert.get(expert, 0) + 1
        savings = (self.TIER_COSTS[4] - self.TIER_COSTS.get(best_tier, 0.02)) * tokens / 1000
        self.stats.estimated_savings += max(0, savings)

        return RoutingResult(
            tier=best_tier,
            tier_name=f"{self.TIER_NAMES[best_tier]} (T{best_tier})",
            confidence=confidence,
            signals=signals,
            expert_model=expert,
            estimated_tokens=tokens,
            cost_estimate=round(cost, 4),
            reasoning=f"Routed to T{best_tier} ({self.TIER_NAMES[best_tier]}) "
                     f"with {confidence:.0%} confidence. Expert: {expert or 'general'}.",
        )

    def get_stats(self) -> dict:
        """Get routing statistics and cost savings."""
        return {
            "total_routed": self.stats.total_routed,
            "tier_distribution": {self.TIER_NAMES.get(k, k): v
                                  for k, v in self.stats.by_tier.items()},
            "expert_usage": self.stats.by_expert,
            "estimated_savings": f"${self.stats.estimated_savings:.2f}",
        }

    def _match_expert(self, task_lower: str) -> str:
        """Match task to the best expert model."""
        best_expert = ""
        best_score = 0.0

        for expert, data in self.EXPERTS.items():
            score = sum(data["weight"] for kw in data["keywords"] if kw in task_lower)
            if score > best_score:
                best_score = score
                best_expert = expert

        return best_expert

    def _estimate_tokens(self, task: str, code: str) -> int:
        """Estimate token count for the task."""
        text = f"{task} {code}"
        # Rough estimate: 1 token ≈ 4 characters
        return max(100, len(text) // 4)

    def _build_tier_signals(self) -> dict:
        """Build keyword signals for each tier."""
        return {
            1: {"keywords": ["rename", "format", "typo", "comment", "simple",
                             "variable name", "import", "indent", "spacing"],
                "weight": 1.5},
            2: {"keywords": ["refactor", "test", "document", "clean up",
                             "add function", "update", "modify", "change",
                             "convert", "extract", "move"],
                "weight": 1.0},
            3: {"keywords": ["design", "architect", "implement feature",
                             "integrate", "multi-file", "restructure",
                             "optimize", "performance", "scale", "complex"],
                "weight": 1.0},
            4: {"keywords": ["security review", "audit", "production",
                             "deploy", "migration", "compliance", "pentest",
                             "threat model", "data breach", "critical",
                             "enterprise", "zero-day"],
                "weight": 1.5},
        }
