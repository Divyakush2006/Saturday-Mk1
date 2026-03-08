"""
Saturday Memory Orchestrator — Unified Memory Coordination Layer
================================================================
Coordinates Working Memory ↔ Context State ↔ Knowledge Base.

Responsibilities:
  - Automatic tier promotion (episodic → semantic when reinforced)
  - Automatic tier demotion (semantic → episodic when stale)
  - Memory budget management across all layers
  - Proactive context pre-fetching based on query patterns
  - Unified search across all memory layers
  - Pre-output hallucination verification
"""

from datetime import datetime, timezone
from typing import Optional

from .knowledge_base import (
    KnowledgeBase, KnowledgeItem, MEMORY_TIERS, TIER_HALF_LIVES_DAYS,
    VerificationResult, HallucinationVerdict,
)
from .context_state import (
    ContextStateEngine, WorkingMemory, TemporalReasoner,
    DecisionExplorer, InstructionAnchor,
)


# Tier promotion/demotion ladder
TIER_ORDER = ["episodic", "semantic", "procedural", "strategic", "institutional"]
PROMOTION_THRESHOLD_ACCESSES = 5      # promote after N accesses
PROMOTION_THRESHOLD_REINFORCEMENTS = 3
DEMOTION_CONFIDENCE_THRESHOLD = 0.25  # demote when confidence drops below this


class MemoryOrchestrator:
    """Unified coordinator for all Saturday memory layers.

    Layer 1: Working Memory (token-budget L1 cache)
    Layer 2: Context State (anchors, decisions, drift detection)
    Layer 3: Knowledge Base (5-tier persistent + all V2 innovations)

    Usage:
        from brain.engines.knowledge_base import KnowledgeBase
        from brain.engines.context_state import ContextStateEngine, WorkingMemory
        from brain.engines.memory_orchestrator import MemoryOrchestrator

        kb = KnowledgeBase("./memory")
        ctx = ContextStateEngine()
        wm = WorkingMemory(token_budget=8192)

        memory = MemoryOrchestrator(kb, ctx, wm)

        # Store with automatic tier assignment
        memory.remember("Use RS256 for JWT tokens", source="arch_review", tier="auto")

        # Unified recall across all layers
        result = memory.recall("JWT configuration")

        # Verify before LLM outputs anything
        verdict = memory.verify_before_output("We use HS256 for JWT")
    """

    def __init__(self, knowledge_base: KnowledgeBase,
                 context_engine: ContextStateEngine,
                 working_memory: WorkingMemory = None):
        self.kb = knowledge_base
        self.ctx = context_engine
        self.wm = working_memory or WorkingMemory(token_budget=8192)
        self.temporal = TemporalReasoner()
        self.explorer = DecisionExplorer()
        self._query_history: list[str] = []

    # ── Core Memory Operations ──

    def remember(self, content: str, source: str = "",
                 domain: str = "general", tier: str = "auto",
                 tags: list[str] = None, title: str = "") -> str:
        """Store knowledge with automatic tier assignment if tier='auto'.

        Auto-tier logic:
          - Short, session-specific → episodic
          - General concepts/decisions → semantic
          - How-to patterns → procedural
          - Architecture rules (high-impact) → strategic
        """
        if tier == "auto":
            tier = self._auto_classify_tier(content, domain)

        if not title:
            # Auto-generate title from first meaningful chunk
            title = content[:80].split("\n")[0].strip()
            if len(title) > 60:
                title = title[:57] + "..."

        item = KnowledgeItem(
            item_id="",
            domain=domain,
            title=title,
            content=content,
            tier=tier,
            tags=tags or [],
            source=source,
            source_type="memory_orchestrator",
        )
        item_id = self.kb.add(item)

        # Also inject into working memory for immediate availability
        wm_priority = MEMORY_TIERS.get(tier, 1) * 2.0
        self.wm.inject(content, priority=wm_priority, ttl=15, source=source)

        return item_id

    def recall(self, query: str, context: dict = None,
               limit: int = 10) -> dict:
        """Unified search across all memory layers.

        Returns results from:
        1. Working Memory (instant, already in context)
        2. Context State (active anchors + decisions)
        3. Knowledge Base (BM25+ search across all tiers)
        """
        context = context or {}
        self._query_history.append(query)

        # Layer 1: Working Memory
        wm_context = self.wm.get_prompt_context()

        # Layer 2: Context State
        ctx_summary = self.ctx.get_context_summary()
        active_anchors = [a.instruction for a in self.ctx.get_active_anchors()]
        active_decisions = [
            f"{d.domain}: {d.decision}"
            for d in self.ctx.get_decisions(active_only=True)
        ]

        # Layer 2b: Temporal facts
        temporal_facts = [
            f.content for f in self.temporal.get_current_facts(
                domain=context.get("domain"))
        ]

        # Layer 3: Knowledge Base search
        domain_filter = context.get("domain")
        kb_results = self.kb.search(query, domain=domain_filter, limit=limit)

        return {
            "query": query,
            "working_memory": wm_context,
            "context_summary": ctx_summary,
            "active_anchors": active_anchors,
            "active_decisions": active_decisions,
            "temporal_facts": temporal_facts,
            "knowledge_items": [
                {
                    "id": item.item_id,
                    "title": item.title,
                    "content": item.content,
                    "tier": item.tier,
                    "confidence": item.confidence,
                    "domain": item.domain,
                }
                for item in kb_results
            ],
            "total_results": len(kb_results),
        }

    # ── Tier Promotion / Demotion ──

    def promote(self, item_id: str) -> bool:
        """Promote an item to a higher memory tier."""
        item = self.kb.items.get(item_id)
        if not item:
            return False
        current_idx = TIER_ORDER.index(item.tier) if item.tier in TIER_ORDER else -1
        if current_idx < 0 or current_idx >= len(TIER_ORDER) - 1:
            return False
        item.tier = TIER_ORDER[current_idx + 1]
        item.updated_at = datetime.now(timezone.utc).isoformat()
        self.kb._save()
        return True

    def demote(self, item_id: str) -> bool:
        """Demote an item to a lower memory tier."""
        item = self.kb.items.get(item_id)
        if not item:
            return False
        current_idx = TIER_ORDER.index(item.tier) if item.tier in TIER_ORDER else -1
        if current_idx <= 0:
            return False
        item.tier = TIER_ORDER[current_idx - 1]
        item.updated_at = datetime.now(timezone.utc).isoformat()
        self.kb._save()
        return True

    def auto_manage_tiers(self) -> dict:
        """Automatically promote/demote items based on usage and confidence.

        Returns summary of promotions and demotions performed.
        """
        promotions = []
        demotions = []

        for item_id, item in self.kb.items.items():
            if item.superseded_by:
                continue

            # Promotion: frequently accessed + high reinforcement
            if (item.access_count >= PROMOTION_THRESHOLD_ACCESSES and
                item.reinforcement_count >= PROMOTION_THRESHOLD_REINFORCEMENTS and
                item.confidence >= 0.8):
                current_idx = TIER_ORDER.index(item.tier) if item.tier in TIER_ORDER else -1
                if 0 <= current_idx < len(TIER_ORDER) - 1:
                    old_tier = item.tier
                    if self.promote(item_id):
                        promotions.append({
                            "item_id": item_id,
                            "from": old_tier,
                            "to": item.tier,
                        })

            # Demotion: low confidence
            elif item.confidence < DEMOTION_CONFIDENCE_THRESHOLD:
                current_idx = TIER_ORDER.index(item.tier) if item.tier in TIER_ORDER else -1
                if current_idx > 0:
                    old_tier = item.tier
                    if self.demote(item_id):
                        demotions.append({
                            "item_id": item_id,
                            "from": old_tier,
                            "to": item.tier,
                        })

        return {
            "promotions": promotions,
            "demotions": demotions,
            "total_changes": len(promotions) + len(demotions),
        }

    # ── Pre-fetch ──

    def prefetch(self, query_hint: str) -> int:
        """Proactively load relevant knowledge into working memory.

        Uses the query hint to anticipate what context will be needed.
        Returns number of items pre-loaded.
        """
        results = self.kb.search(query_hint, limit=5, min_confidence=0.5)
        loaded = 0
        for item in results:
            priority = MEMORY_TIERS.get(item.tier, 1) * 1.5
            context_str = f"[{item.domain}] {item.title}: {item.content[:200]}"
            if self.wm.inject(context_str, priority=priority, ttl=8,
                              source=f"prefetch:{item.item_id}"):
                loaded += 1
        return loaded

    # ── Anti-Hallucination Gate ──

    def verify_before_output(self, claim: str,
                             domain: str = None) -> dict:
        """Run 4-stage anti-hallucination pipeline before outputting a claim.

        This is the PRIMARY hallucination prevention mechanism.
        Should be called before the LLM outputs any factual statement.
        """
        result = self.kb.verify_claim(claim, domain=domain)

        # Inject verification result into working memory
        if result.verdict == HallucinationVerdict.CONTRADICTED.value:
            self.wm.inject(
                f"⚠️ BLOCKED CLAIM: '{claim[:100]}' — CONTRADICTED by knowledge base",
                priority=10.0, ttl=20, source="anti_hallucination",
            )
        elif result.verdict == HallucinationVerdict.UNCERTAIN.value:
            self.wm.inject(
                f"⚡ UNCERTAIN CLAIM: '{claim[:80]}' — exercise caution",
                priority=8.0, ttl=10, source="anti_hallucination",
            )

        return {
            "claim": claim,
            "verdict": result.verdict,
            "is_verified": result.is_verified,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "stages_passed": result.stages_passed,
            "stages_failed": result.stages_failed,
            "supporting_count": len(result.supporting_items),
            "contradicting_count": len(result.contradicting_items),
            "sources": result.sources,
        }

    # ── Turn Management ──

    def on_turn_start(self, query: str):
        """Called at the start of each interaction turn.

        - Prefetches relevant context
        - Updates coherence tracking
        - Applies confidence decay
        """
        self.prefetch(query)
        self.ctx.update_coherence()
        self.temporal.cleanup_expired()

    def on_turn_end(self):
        """Called at the end of each interaction turn.

        - Ticks working memory TTLs
        - Checks for drift alerts
        """
        self.wm.tick()
        alerts = self.ctx.check_drift()
        for alert in alerts:
            if alert.severity == "critical":
                self.wm.inject(
                    f"⚠️ DRIFT ALERT: {alert.message}",
                    priority=9.5, ttl=10, source="drift_detection",
                )

    # ── Budget Report ──

    def get_memory_budget_report(self) -> dict:
        """Get comprehensive memory usage across all layers."""
        kb_stats = self.kb.get_stats()
        wm_usage = self.wm.get_usage()
        coherence = self.ctx.get_coherence_report()

        return {
            "working_memory": wm_usage,
            "context_state": coherence,
            "knowledge_base": {
                "total_items": kb_stats.total_items,
                "items_by_tier": kb_stats.items_by_tier,
                "avg_confidence": kb_stats.avg_confidence,
                "stale_items": kb_stats.stale_items,
                "causal_edges": kb_stats.causal_edges,
                "merkle_root": kb_stats.merkle_root,
            },
            "temporal_facts": {
                "total": len(self.temporal.facts),
                "current": len(self.temporal.get_current_facts()),
                "expired": len(self.temporal.get_expired_facts()),
            },
            "decision_explorer": {
                "active_branches": len(self.explorer.get_active_branches()),
                "committed": len(self.explorer.get_committed_branches()),
            },
        }

    # ── Internal ──

    def _auto_classify_tier(self, content: str, domain: str) -> str:
        """Classify content into the appropriate memory tier."""
        content_lower = content.lower()
        length = len(content)

        # Strategic indicators
        strategic_keywords = {
            "architecture", "design principle", "system constraint",
            "never", "always", "must", "critical rule", "invariant",
            "non-negotiable", "security policy",
        }
        if any(kw in content_lower for kw in strategic_keywords):
            return "strategic"

        # Procedural indicators
        procedural_keywords = {
            "how to", "step 1", "workflow", "recipe", "procedure",
            "convention", "pattern", "template", "follow these",
        }
        if any(kw in content_lower for kw in procedural_keywords):
            return "procedural"

        # Episodic: short, session-specific
        if length < 100 or domain == "session":
            return "episodic"

        # Default to semantic
        return "semantic"
