"""
Saturday Memory Architecture V2 — Comprehensive Test Suite
===========================================================
Tests all 7 knowledge base innovations + 3 context state upgrades +
1 memory orchestrator, ensuring zero-defect memory architecture.
"""

import json
import os
import shutil
import tempfile
import pytest
from datetime import datetime, timezone, timedelta

# ── Import all components under test ──
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from brain.engines.knowledge_base import (
    KnowledgeBase, KnowledgeItem, KnowledgeBase as KB,
    BM25PlusEngine, MerkleIntegrityLayer, BayesianConfidence,
    CausalKnowledgeGraph, CausalEdge, AntiHallucinationPipeline,
    WriteAheadLog, ComplianceLayer, HallucinationVerdict,
    VerificationResult, MerkleProof, IntegrityReport, AuditEntry,
    MEMORY_TIERS,
)
from brain.engines.context_state import (
    WorkingMemory, WorkingMemorySlot,
    TemporalReasoner, TemporalFact,
    DecisionExplorer, DecisionBranch,
)
from brain.engines.memory_orchestrator import MemoryOrchestrator
from brain.engines.context_state import ContextStateEngine


# ══════════════════════════════════════════════════════════════
#  Fixtures
# ══════════════════════════════════════════════════════════════

@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="saturday_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def kb(tmp_dir):
    return KnowledgeBase(os.path.join(tmp_dir, "kb"))


@pytest.fixture
def bm25():
    return BM25PlusEngine()


@pytest.fixture
def merkle():
    return MerkleIntegrityLayer()


@pytest.fixture
def causal():
    return CausalKnowledgeGraph()


# ══════════════════════════════════════════════════════════════
#  Test BM25+ Search Engine
# ══════════════════════════════════════════════════════════════

class TestBM25PlusEngine:
    """Test field-weighted BM25+ ranking with proximity boost."""

    def test_basic_search(self, bm25):
        bm25.index_document("d1", {
            "title": "JWT Authentication",
            "content": "Use RS256 signing algorithm for all JSON Web Tokens",
            "tags": "jwt auth security",
        })
        bm25.index_document("d2", {
            "title": "Database Schema",
            "content": "PostgreSQL schema design with normalized tables",
            "tags": "database postgres",
        })
        results = bm25.search("JWT authentication")
        assert len(results) > 0
        assert results[0][0] == "d1"

    def test_title_matches_rank_higher(self, bm25):
        bm25.index_document("d1", {
            "title": "Unrelated",
            "content": "JWT authentication with secure tokens",
            "tags": "",
        })
        bm25.index_document("d2", {
            "title": "JWT Authentication Guide",
            "content": "Simple guide to tokens",
            "tags": "",
        })
        results = bm25.search("JWT authentication")
        # Title match should rank higher due to 3x weight
        assert results[0][0] == "d2"

    def test_empty_query_returns_empty(self, bm25):
        bm25.index_document("d1", {"title": "Test", "content": "test", "tags": ""})
        assert bm25.search("") == []
        assert bm25.search("the is a") == []  # all stopwords

    def test_empty_index_returns_empty(self, bm25):
        assert bm25.search("anything") == []

    def test_document_removal(self, bm25):
        bm25.index_document("d1", {"title": "Test One", "content": "alpha beta", "tags": ""})
        bm25.index_document("d2", {"title": "Test Two", "content": "gamma delta", "tags": ""})
        bm25.remove_document("d1")
        results = bm25.search("alpha beta")
        assert all(r[0] != "d1" for r in results)

    def test_long_doc_doesnt_dominate(self, bm25):
        """BM25+ term saturation: long docs shouldn't unfairly dominate."""
        bm25.index_document("short", {
            "title": "JWT Config",
            "content": "Use RS256 for JWT",
            "tags": "jwt",
        })
        bm25.index_document("long", {
            "title": "Everything",
            "content": "JWT " * 200 + "unrelated content " * 500,
            "tags": "",
        })
        results = bm25.search("JWT config RS256")
        assert results[0][0] == "short"

    def test_proximity_boost(self, bm25):
        """Terms appearing near each other should score higher."""
        bm25.index_document("close", {
            "title": "API",
            "content": "Use JWT authentication tokens for secure access",
            "tags": "",
        })
        bm25.index_document("far", {
            "title": "API",
            "content": ("JWT is a standard. " * 20 +
                        "Authentication is important for security."),
            "tags": "",
        })
        results = bm25.search("JWT authentication")
        assert results[0][0] == "close"


# ══════════════════════════════════════════════════════════════
#  Test Merkle Tree Integrity
# ══════════════════════════════════════════════════════════════

class TestMerkleIntegrity:
    """Test SHA-256 cryptographic tamper detection."""

    def test_build_and_verify(self, merkle):
        items = {
            "a": KnowledgeItem(item_id="a", domain="test", title="A", content="Alpha"),
            "b": KnowledgeItem(item_id="b", domain="test", title="B", content="Beta"),
        }
        root = merkle.build_tree(items)
        assert root and len(root) == 64  # SHA-256 hex

        report = merkle.verify_integrity(items)
        assert report.is_clean
        assert report.verified_items == 2

    def test_tamper_detection(self, merkle):
        items = {
            "a": KnowledgeItem(item_id="a", domain="test", title="A", content="Alpha"),
            "b": KnowledgeItem(item_id="b", domain="test", title="B", content="Beta"),
        }
        root1 = merkle.build_tree(items)

        # Tamper with item
        items["a"].content = "TAMPERED"
        report = merkle.verify_integrity(items)
        assert not report.is_clean
        assert "a" in report.tampered_items

    def test_root_hash_changes_on_tamper(self, merkle):
        items = {"a": KnowledgeItem(item_id="a", domain="test", title="A", content="X")}
        root1 = merkle.build_tree(items)
        items["a"].content = "Y"
        root2 = merkle.build_tree(items)
        assert root1 != root2

    def test_merkle_proof(self, merkle):
        items = {
            "a": KnowledgeItem(item_id="a", domain="test", title="A", content="Alpha"),
            "b": KnowledgeItem(item_id="b", domain="test", title="B", content="Beta"),
            "c": KnowledgeItem(item_id="c", domain="test", title="C", content="Gamma"),
            "d": KnowledgeItem(item_id="d", domain="test", title="D", content="Delta"),
        }
        merkle.build_tree(items)
        proof = merkle.generate_proof("b")
        assert proof.verified
        assert merkle.verify_proof(proof)

    def test_empty_tree(self, merkle):
        root = merkle.build_tree({})
        assert root  # Should still produce a hash


# ══════════════════════════════════════════════════════════════
#  Test Bayesian Confidence
# ══════════════════════════════════════════════════════════════

class TestBayesianConfidence:
    """Test Bayesian belief revision + temporal decay."""

    def test_supporting_evidence_increases_confidence(self):
        prior = 0.6
        posterior = BayesianConfidence.update_belief(prior, 0.8, True)
        assert posterior > prior

    def test_contradicting_evidence_decreases_confidence(self):
        prior = 0.8
        posterior = BayesianConfidence.update_belief(prior, 0.8, False)
        assert posterior < prior

    def test_temporal_decay(self):
        item = KnowledgeItem(
            item_id="x", domain="test", title="Test", content="test",
            tier="episodic", confidence=1.0,
            updated_at=(datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
        )
        decayed = BayesianConfidence.apply_temporal_decay(item)
        assert decayed < 1.0
        # Episodic has 7-day half-life, so after 30 days significant decay
        assert decayed < 0.15

    def test_institutional_decays_slowly(self):
        item = KnowledgeItem(
            item_id="x", domain="test", title="Test", content="test",
            tier="institutional", confidence=1.0,
            updated_at=(datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
        )
        decayed = BayesianConfidence.apply_temporal_decay(item)
        # Institutional has 1825-day half-life, so 30 days = minimal decay
        assert decayed > 0.95

    def test_reinforcement_extends_halflife(self):
        base_item = KnowledgeItem(
            item_id="x", domain="test", title="Test", content="test",
            tier="episodic", confidence=1.0,
            updated_at=(datetime.now(timezone.utc) - timedelta(days=14)).isoformat(),
            reinforcement_count=0,
        )
        reinforced_item = KnowledgeItem(
            item_id="y", domain="test", title="Test", content="test",
            tier="episodic", confidence=1.0,
            updated_at=(datetime.now(timezone.utc) - timedelta(days=14)).isoformat(),
            reinforcement_count=10,
        )
        decay_base = BayesianConfidence.apply_temporal_decay(base_item)
        decay_reinforced = BayesianConfidence.apply_temporal_decay(reinforced_item)
        assert decay_reinforced > decay_base

    def test_outcome_calibration(self):
        conf = 0.7
        correct = BayesianConfidence.calibrate_from_outcome(conf, True)
        wrong = BayesianConfidence.calibrate_from_outcome(conf, False)
        assert correct > conf
        assert wrong < conf


# ══════════════════════════════════════════════════════════════
#  Test Causal Knowledge Graph
# ══════════════════════════════════════════════════════════════

class TestCausalGraph:
    """Test typed directed edge relationships."""

    def test_add_and_get_edges(self, causal):
        e = CausalEdge(source_id="a", target_id="b", relation="causes")
        assert causal.add_edge(e)
        edges = causal.get_edges("a", direction="outgoing")
        assert len(edges) == 1
        assert edges[0].relation == "causes"

    def test_invalid_relation_rejected(self, causal):
        e = CausalEdge(source_id="a", target_id="b", relation="loves")
        assert not causal.add_edge(e)

    def test_impact_analysis(self, causal):
        causal.add_edge(CausalEdge(source_id="a", target_id="b", relation="causes"))
        causal.add_edge(CausalEdge(source_id="b", target_id="c", relation="causes"))
        impact = causal.impact_analysis("a")
        assert "b" in impact["directly_affected"]
        assert "c" in impact["indirectly_affected"]

    def test_reasoning_path(self, causal):
        causal.add_edge(CausalEdge(source_id="a", target_id="b", relation="causes"))
        causal.add_edge(CausalEdge(source_id="b", target_id="c", relation="enables"))
        path = causal.find_reasoning_path("a", "c")
        assert len(path) == 2

    def test_cycle_detection(self, causal):
        causal.add_edge(CausalEdge(source_id="a", target_id="b", relation="causes"))
        causal.add_edge(CausalEdge(source_id="b", target_id="c", relation="causes"))
        causal.add_edge(CausalEdge(source_id="c", target_id="a", relation="causes"))
        cycles = causal.detect_cycles()
        assert len(cycles) > 0

    def test_serialization(self, causal):
        causal.add_edge(CausalEdge(source_id="a", target_id="b", relation="causes"))
        data = causal.to_dict()
        new_graph = CausalKnowledgeGraph()
        new_graph.from_dict(data)
        assert len(new_graph.edges) == 1


# ══════════════════════════════════════════════════════════════
#  Test Anti-Hallucination Pipeline
# ══════════════════════════════════════════════════════════════

class TestAntiHallucinationPipeline:
    """Test 4-stage verification pipeline."""

    def test_verified_claim(self, kb):
        kb.add(KnowledgeItem(
            item_id="jwt_001", domain="auth",
            title="JWT Algorithm",
            content="Use RS256 signing algorithm for all JWT tokens",
            tier="strategic", source="arch_review",
            tags=["jwt", "auth"],
        ))
        result = kb.verify_claim("Use RS256 for JWT tokens")
        assert result.verdict == "verified"
        assert result.is_verified
        assert len(result.supporting_items) > 0

    def test_contradicted_claim(self, kb):
        kb.add(KnowledgeItem(
            item_id="jwt_001", domain="auth",
            title="JWT Algorithm Preference",
            content="Always use RS256 and never use HS256 for JWT signing",
            tier="strategic", source="security_policy",
            tags=["jwt"],
        ))
        result = kb.verify_claim("Use HS256 for JWT signing")
        # Should detect contradiction via negation pairs
        assert result.verdict in ("contradicted", "uncertain")

    def test_insufficient_evidence(self, kb):
        result = kb.verify_claim("Use GraphQL for all API endpoints")
        assert result.verdict == "insufficient_evidence"
        assert not result.is_verified

    def test_stages_tracked(self, kb):
        kb.add(KnowledgeItem(
            item_id="test_001", domain="test",
            title="Testing Pattern",
            content="Always write unit tests before integration tests",
            tier="procedural", source="coding_standards",
            tags=["testing"],
        ))
        result = kb.verify_claim("Write unit tests first")
        assert len(result.stages_passed) + len(result.stages_failed) == 4


# ══════════════════════════════════════════════════════════════
#  Test Write-Ahead Log
# ══════════════════════════════════════════════════════════════

class TestWriteAheadLog:
    """Test crash recovery via WAL."""

    def test_log_and_replay(self, tmp_dir):
        wal_path = os.path.join(tmp_dir, "test.wal")
        wal = WriteAheadLog(wal_path)

        wal.log_operation("add", "item_001", {"content": "test data"})
        wal.log_operation("add", "item_002", {"content": "more data"})

        # Simulate crash: create new WAL from same file
        wal2 = WriteAheadLog(wal_path)
        ops = wal2.replay()
        assert len(ops) == 2
        assert ops[0]["id"] == "item_001"

    def test_checkpoint_clears_wal(self, tmp_dir):
        wal_path = os.path.join(tmp_dir, "test.wal")
        wal = WriteAheadLog(wal_path)

        wal.log_operation("add", "item_001", {"content": "test"})
        wal.checkpoint()

        wal2 = WriteAheadLog(wal_path)
        ops = wal2.replay()
        assert len(ops) == 0

    def test_pending_count(self, tmp_dir):
        wal = WriteAheadLog(os.path.join(tmp_dir, "test.wal"))
        assert wal.get_pending_count() == 0
        wal.log_operation("add", "x", {})
        assert wal.get_pending_count() == 1


# ══════════════════════════════════════════════════════════════
#  Test Compliance Layer
# ══════════════════════════════════════════════════════════════

class TestComplianceLayer:
    """Test GDPR, SOC2, HIPAA compliance features."""

    def test_audit_trail(self, tmp_dir):
        audit_path = os.path.join(tmp_dir, "audit.log")
        comp = ComplianceLayer(audit_path)

        comp.log_operation("add", "item_001", accessor="user_A")
        comp.log_operation("read", "item_001", accessor="user_B")

        trail = comp.get_audit_trail()
        assert len(trail) == 2
        assert trail[0].operation == "add"
        assert trail[1].accessor == "user_B"

    def test_audit_chain_integrity(self, tmp_dir):
        comp = ComplianceLayer(os.path.join(tmp_dir, "audit.log"))
        comp.log_operation("add", "x")
        comp.log_operation("read", "x")
        comp.log_operation("delete", "x")

        result = comp.verify_audit_chain()
        assert result["valid"]
        assert result["entries"] == 3

    def test_gdpr_crypto_erase(self, tmp_dir):
        comp = ComplianceLayer(os.path.join(tmp_dir, "audit.log"))
        items = {
            "a": KnowledgeItem(item_id="a", domain="user", title="PII", content="secret data", namespace="tenant_A"),
            "b": KnowledgeItem(item_id="b", domain="user", title="Other", content="other", namespace="tenant_B"),
        }
        receipt = comp.crypto_erase(items, {"namespace": "tenant_A"})
        assert receipt["erased_count"] == 1
        assert "a" not in items
        assert "b" in items
        assert receipt["proof_hash"]

    def test_namespace_isolation(self, tmp_dir):
        comp = ComplianceLayer(os.path.join(tmp_dir, "audit.log"))
        items = {
            "a": KnowledgeItem(item_id="a", domain="x", title="A", content="a", namespace="ns1"),
            "b": KnowledgeItem(item_id="b", domain="x", title="B", content="b", namespace="ns2"),
        }
        ns1_items = comp.get_namespace_items(items, "ns1")
        assert "a" in ns1_items
        assert "b" not in ns1_items


# ══════════════════════════════════════════════════════════════
#  Test Working Memory
# ══════════════════════════════════════════════════════════════

class TestWorkingMemory:
    """Test token-budget-aware L1 cache."""

    def test_basic_inject(self):
        wm = WorkingMemory(token_budget=1000)
        assert wm.inject("Hello world", priority=5.0, ttl=10)
        assert wm.tokens_used > 0
        assert len(wm.slots) == 1

    def test_budget_enforcement(self):
        wm = WorkingMemory(token_budget=20)  # ~80 chars
        wm.inject("Short text", priority=5.0)
        # This should evict the previous one
        wm.inject("Another short text", priority=6.0)
        assert wm.tokens_used <= wm.token_budget

    def test_priority_eviction(self):
        wm = WorkingMemory(token_budget=30)
        wm.inject("Low priority item here", priority=1.0)
        wm.inject("High priority item here", priority=9.0)
        # High priority should survive
        assert any(s.priority == 9.0 for s in wm.slots)

    def test_ttl_expiry(self):
        wm = WorkingMemory(token_budget=1000)
        wm.inject("Expiring soon", priority=5.0, ttl=2)
        assert len(wm.slots) == 1
        wm.tick()
        assert len(wm.slots) == 1  # ttl=1 now
        wm.tick()
        assert len(wm.slots) == 0  # expired

    def test_prompt_context_generation(self):
        wm = WorkingMemory(token_budget=1000)
        wm.inject("Security rule one", priority=9.0, source="policy")
        wm.inject("User preference", priority=3.0, source="user")
        context = wm.get_prompt_context()
        assert "Working Memory" in context
        assert "Security rule one" in context

    def test_oversized_content_rejected(self):
        wm = WorkingMemory(token_budget=5)  # ~20 chars
        assert not wm.inject("This content is way too long to fit in the budget", priority=10.0)

    def test_remove_by_source(self):
        wm = WorkingMemory(token_budget=1000)
        wm.inject("A", priority=5.0, source="src1")
        wm.inject("B", priority=5.0, source="src2")
        wm.inject("C", priority=5.0, source="src1")
        removed = wm.remove_by_source("src1")
        assert removed == 2
        assert len(wm.slots) == 1

    def test_usage_report(self):
        wm = WorkingMemory(token_budget=1000)
        wm.inject("Test", priority=7.0)
        usage = wm.get_usage()
        assert usage["slots"] == 1
        assert usage["tokens_budget"] == 1000
        assert 0 < usage["utilization"] < 1


# ══════════════════════════════════════════════════════════════
#  Test Temporal Reasoner
# ══════════════════════════════════════════════════════════════

class TestTemporalReasoner:
    """Test time-bounded fact management."""

    def test_add_current_fact(self):
        tr = TemporalReasoner()
        fact_id = tr.add_fact(TemporalFact(
            fact_id="f1", content="Deploy freeze active",
            valid_from=(datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
            valid_until=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        ))
        assert tr.is_valid(fact_id)
        assert len(tr.get_current_facts()) == 1

    def test_expired_fact(self):
        tr = TemporalReasoner()
        tr.add_fact(TemporalFact(
            fact_id="f1", content="Old fact",
            valid_from="2020-01-01",
            valid_until="2020-12-31",
        ))
        assert not tr.is_valid("f1")
        assert len(tr.get_expired_facts()) == 1

    def test_future_fact(self):
        tr = TemporalReasoner()
        tr.add_fact(TemporalFact(
            fact_id="f1", content="Future release",
            valid_from="2099-01-01",
            valid_until="2099-12-31",
        ))
        assert not tr.is_valid("f1")
        assert len(tr.get_upcoming_facts()) == 1

    def test_cleanup_expired(self):
        tr = TemporalReasoner()
        tr.add_fact(TemporalFact(fact_id="f1", content="Old", valid_until="2020-01-01"))
        tr.add_fact(TemporalFact(fact_id="f2", content="Current",
                                  valid_until=(datetime.now(timezone.utc) + timedelta(days=1)).isoformat()))
        removed = tr.cleanup_expired()
        assert removed == 1
        assert len(tr.facts) == 1


# ══════════════════════════════════════════════════════════════
#  Test Decision Explorer
# ══════════════════════════════════════════════════════════════

class TestDecisionExplorer:
    """Test fork/merge decision branching."""

    def test_fork_creates_branches(self):
        explorer = DecisionExplorer()
        branches = explorer.fork("Database", ["PostgreSQL", "MongoDB"])
        assert len(branches) == 2
        b1 = explorer.get_branch(branches[0])
        assert b1.option == "PostgreSQL"
        assert b1.status == "exploring"

    def test_merge_commits_and_discards_siblings(self):
        explorer = DecisionExplorer()
        branches = explorer.fork("Framework", ["React", "Vue", "Svelte"])
        explorer.merge(branches[1])  # Commit Vue

        assert explorer.get_branch(branches[0]).status == "discarded"
        assert explorer.get_branch(branches[1]).status == "committed"
        assert explorer.get_branch(branches[2]).status == "discarded"

    def test_add_notes(self):
        explorer = DecisionExplorer()
        branches = explorer.fork("Cache", ["Redis", "Memcached"])
        assert explorer.add_note(branches[0], "Better data structures")
        assert explorer.add_note(branches[1], "Simpler deployment")
        comparison = explorer.compare_branches(branches)
        assert comparison["branches"][0]["notes_count"] == 1

    def test_cannot_modify_committed(self):
        explorer = DecisionExplorer()
        branches = explorer.fork("DB", ["PG", "MySQL"])
        explorer.merge(branches[0])
        assert not explorer.add_note(branches[0], "Late note")


# ══════════════════════════════════════════════════════════════
#  Test Memory Orchestrator
# ══════════════════════════════════════════════════════════════

class TestMemoryOrchestrator:
    """Test unified memory coordination."""

    @pytest.fixture
    def orchestrator(self, tmp_dir):
        kb = KnowledgeBase(os.path.join(tmp_dir, "orch_kb"))
        ctx = ContextStateEngine()
        wm = WorkingMemory(token_budget=4096)
        return MemoryOrchestrator(kb, ctx, wm)

    def test_remember_and_recall(self, orchestrator):
        orchestrator.remember(
            "Use RS256 for JWT tokens",
            source="arch_review", domain="auth",
            tags=["jwt", "security"],
        )
        result = orchestrator.recall("JWT configuration")
        assert result["total_results"] > 0
        assert "RS256" in result["knowledge_items"][0]["content"]

    def test_auto_tier_classification(self, orchestrator):
        # "never" keyword → strategic
        item_id = orchestrator.remember("Never store passwords in plaintext")
        item = orchestrator.kb.items.get(item_id)
        assert item.tier == "strategic"

    def test_verify_before_output(self, orchestrator):
        orchestrator.remember(
            "Use RS256 for JWT tokens",
            source="security_team", domain="auth",
        )
        result = orchestrator.verify_before_output("Use RS256 for JWT tokens")
        assert result["verdict"] in ("verified", "uncertain", "insufficient_evidence")
        assert "confidence" in result

    def test_prefetch_loads_working_memory(self, orchestrator):
        orchestrator.remember("Python async patterns for high-throughput APIs",
                               source="docs", domain="backend")
        loaded = orchestrator.prefetch("async API patterns")
        assert loaded > 0
        assert orchestrator.wm.tokens_used > 0

    def test_budget_report(self, orchestrator):
        orchestrator.remember("Test item", source="test")
        report = orchestrator.get_memory_budget_report()
        assert "working_memory" in report
        assert "knowledge_base" in report
        assert "temporal_facts" in report

    def test_auto_manage_tiers(self, orchestrator):
        # Create an episodic item with high access and reinforcement
        item_id = orchestrator.remember("Important pattern", domain="code", tier="episodic")
        item = orchestrator.kb.items[item_id]
        item.access_count = 10
        item.reinforcement_count = 5
        item.confidence = 0.95

        result = orchestrator.auto_manage_tiers()
        # Should promote from episodic to semantic
        assert result["total_changes"] >= 1


# ══════════════════════════════════════════════════════════════
#  Test Knowledge Base Integration
# ══════════════════════════════════════════════════════════════

class TestKnowledgeBaseIntegration:
    """End-to-end tests for the full Knowledge Base V2."""

    def test_add_search_verify(self, kb):
        kb.add(KnowledgeItem(
            item_id="arch_001", domain="architecture",
            title="Microservices Pattern",
            content="Use event-driven microservices with CQRS pattern",
            tier="strategic", source="design_doc",
            tags=["microservices", "cqrs", "architecture"],
        ))
        results = kb.search("microservices architecture")
        assert len(results) > 0
        assert results[0].domain == "architecture"

    def test_persistence(self, tmp_dir):
        path = os.path.join(tmp_dir, "persist_kb")
        kb1 = KnowledgeBase(path)
        kb1.add(KnowledgeItem(
            item_id="p1", domain="test", title="Persist",
            content="This should persist across restarts",
        ))
        del kb1  # destroy instance

        kb2 = KnowledgeBase(path)
        assert "p1" in kb2.items
        assert kb2.items["p1"].content == "This should persist across restarts"

    def test_cross_references(self, kb):
        kb.add(KnowledgeItem(item_id="a", domain="x", title="A", content="Alpha"))
        kb.add(KnowledgeItem(item_id="b", domain="x", title="B", content="Beta"))
        assert kb.add_cross_reference("a", "b")
        related = kb.get_related("a")
        assert len(related) == 1
        assert related[0].item_id == "b"

    def test_stats(self, kb):
        kb.add(KnowledgeItem(item_id="s1", domain="d1", title="S1", content="C1", tier="semantic"))
        kb.add(KnowledgeItem(item_id="s2", domain="d2", title="S2", content="C2", tier="strategic"))
        stats = kb.get_stats()
        assert stats.total_items == 2
        assert "semantic" in stats.items_by_tier
        assert "strategic" in stats.items_by_tier

    def test_delete(self, kb):
        kb.add(KnowledgeItem(item_id="del1", domain="x", title="Delete Me", content="bye"))
        assert kb.delete("del1")
        assert "del1" not in kb.items

    def test_export_markdown(self, kb):
        kb.add(KnowledgeItem(
            item_id="md1", domain="test", title="Markdown Test",
            content="Export me", tier="semantic",
        ))
        md = kb.export_markdown()
        assert "Markdown Test" in md
        assert "Saturday Knowledge Base" in md


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
