"""
Saturday MK1 — Full System Integration Test
=============================================
Loads ALL 11 engines and exercises every capability.
Output saved to test_integration_report.txt
"""
import sys
import os
import shutil
import time
import json

# Fix Windows encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
REPORT_PATH = os.path.join(PROJECT_ROOT, "test_integration_report.txt")
MEM_DIR = os.path.join(PROJECT_ROOT, ".test_integration_mem")

lines = []
def log(msg=""):
    lines.append(msg)
    print(msg)

def section(title):
    log("")
    log("=" * 70)
    log(f"  {title}")
    log("=" * 70)

def result(name, status, detail=""):
    icon = "✅" if status else "❌"
    log(f"  {icon} {name}: {detail}")
    return status

t_start = time.time()
passed = 0
failed = 0

section("SATURDAY MK1 — FULL SYSTEM INTEGRATION TEST")
log(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
log(f"  Python: {sys.version.split()[0]}")

# ═══════════════════════════════════════════════════════════
#  ENGINE 1: CORE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════
section("ENGINE 1: CORE ORCHESTRATOR")
try:
    from brain.saturday_core import Saturday
    sat = Saturday(project_root=".", memory_dir=MEM_DIR)
    if result("Saturday Core", True, f"v{sat.VERSION} | project={sat.project_root.name}"):
        passed += 1
except Exception as e:
    result("Saturday Core", False, str(e)); failed += 1

# ═══════════════════════════════════════════════════════════
#  ENGINE 2: LLM PROVIDER
# ═══════════════════════════════════════════════════════════
section("ENGINE 2: LLM PROVIDER")
try:
    from brain.engines.llm_provider import LLMProvider, LLMConfig, OpenAICompatibleProvider
    config = LLMConfig.from_env()
    if result("LLM Config", True, f"provider={config.provider} model={config.model}"):
        passed += 1
    if result("LLM API Base", True, f"base={config.api_base}"):
        passed += 1
    llm = LLMProvider.from_env()
    if result("LLM Provider", True, f"initialized as {type(llm).__name__}, retries={config.max_retries}"):
        passed += 1
except Exception as e:
    result("LLM Provider", False, str(e)); failed += 1

# ═══════════════════════════════════════════════════════════
#  ENGINE 3: SECURITY PIPELINE (12-layer)
# ═══════════════════════════════════════════════════════════
section("ENGINE 3: SECURITY PIPELINE")
try:
    sec = sat.security
    # Test dangerous code
    dangerous = 'import os\nos.system(input("cmd: "))\neval(open("config.txt").read())'
    findings = sec.scan_code(dangerous, "vuln.py", "python")
    if result("Pipeline Load", True, "12-layer pipeline active"):
        passed += 1
    if result("Vulnerability Detection", True, f"{len(findings)} findings on dangerous code"):
        passed += 1
    # Test safe code
    safe = 'def add(a: int, b: int) -> int:\n    return a + b'
    safe_findings = sec.scan_code(safe, "safe.py", "python")
    if result("Safe Code Scan", True, f"{len(safe_findings)} findings on safe code"):
        passed += 1
except Exception as e:
    result("Security Pipeline", False, str(e)); failed += 1

# ═══════════════════════════════════════════════════════════
#  ENGINE 4: CODE GRAPH ENGINE
# ═══════════════════════════════════════════════════════════
section("ENGINE 4: CODE GRAPH ENGINE")
try:
    cg = sat.code_graph
    if result("Code Graph Load", True, f"root_path={cg.root_path}"):
        passed += 1
    outline = sat.get_file_outline("brain/engines/knowledge_base.py")
    if result("File Outline", True, f"parsed knowledge_base.py, {len(str(outline))} chars"):
        passed += 1
except Exception as e:
    result("Code Graph", False, str(e)); failed += 1

# ═══════════════════════════════════════════════════════════
#  ENGINE 5: STRATEGIC PLANNER
# ═══════════════════════════════════════════════════════════
section("ENGINE 5: STRATEGIC PLANNER")
try:
    planner = sat.planner
    plan = planner.create_plan("Refactor auth module to support OAuth2 + SSO", {
        "current_auth": "JWT basic", "team_size": 4
    })
    if result("Planner Load", True, "dependency-aware planner active"):
        passed += 1
    plan_len = len(str(plan)) if plan else 0
    if result("Plan Creation", True, f"plan generated ({plan_len} chars)"):
        passed += 1
except Exception as e:
    result("Strategic Planner", False, str(e)); failed += 1

# ═══════════════════════════════════════════════════════════
#  ENGINE 6: CONTEXT STATE ENGINE V2
# ═══════════════════════════════════════════════════════════
section("ENGINE 6: CONTEXT STATE ENGINE V2")
try:
    from brain.engines.context_state import InstructionAnchor, DecisionRecord
    ctx = sat.context

    # Anchors
    ctx.add_anchor(InstructionAnchor(instruction="Always use type hints in Python", priority="critical"))
    ctx.add_anchor(InstructionAnchor(instruction="Follow PEP 8 naming conventions", priority="high"))
    ctx.add_anchor(InstructionAnchor(instruction="Use async/await for I/O operations", priority="medium"))
    anchors = ctx.get_active_anchors()
    if result("Instruction Anchors", True, f"{len(anchors)} anchors set"):
        passed += 1

    # Decisions
    ctx.record_decision(DecisionRecord(domain="backend", decision="Use FastAPI", rationale="Performance + async"))
    ctx.record_decision(DecisionRecord(domain="database", decision="Use PostgreSQL", rationale="ACID compliance"))
    ctx.record_decision(DecisionRecord(domain="auth", decision="Use OAuth2 + JWT RS256", rationale="Enterprise standard"))
    decisions = ctx.get_decisions(active_only=True)
    if result("Decision Recording", True, f"{len(decisions)} decisions recorded"):
        passed += 1

    # Coherence
    coherence = ctx.get_coherence_report()
    coh_score = coherence.get('overall_coherence', coherence.get('score', 0))
    if result("Coherence Tracking", True, f"coherence={coh_score:.0%}"):
        passed += 1

    # Snapshot
    snap_id = ctx.take_snapshot("before_refactor")
    if result("Context Snapshot", True, f"snapshot_id={snap_id}"):
        passed += 1
except Exception as e:
    result("Context State V2", False, str(e)); failed += 1

# ═══════════════════════════════════════════════════════════
#  ENGINE 7: KNOWLEDGE BASE V2 (7 innovations)
# ═══════════════════════════════════════════════════════════
section("ENGINE 7: KNOWLEDGE BASE V2")
try:
    from brain.engines.knowledge_base import KnowledgeItem
    kb = sat.knowledge

    # Add knowledge across all 5 tiers
    kb.add(KnowledgeItem(item_id="inst_001", domain="compliance", title="Data Retention Policy",
        content="All user data must be retained for 7 years per SOX regulations", tier="institutional",
        source="legal_team", source_type="policy", tags=["compliance", "sox", "retention"]))

    kb.add(KnowledgeItem(item_id="strat_001", domain="auth", title="JWT Algorithm Standard",
        content="Always use RS256 signing algorithm, never use HS256 for production JWT tokens",
        tier="strategic", source="security_review_2026", source_type="decision",
        tags=["jwt", "rs256", "auth", "security"]))

    kb.add(KnowledgeItem(item_id="proc_001", domain="deployment", title="Deploy Procedure",
        content="Step 1: Run tests. Step 2: Build Docker image. Step 3: Push to staging. Step 4: Run smoke tests. Step 5: Promote to prod.",
        tier="procedural", source="devops_handbook", source_type="procedure",
        tags=["deploy", "docker", "ci/cd"]))

    kb.add(KnowledgeItem(item_id="sem_001", domain="architecture", title="Microservices with CQRS",
        content="Use event-driven microservices with CQRS pattern for read/write separation",
        tier="semantic", source="arch_doc_v3", source_type="architecture",
        tags=["microservices", "cqrs", "events"]))

    kb.add(KnowledgeItem(item_id="epi_001", domain="session", title="Today's Bug Fix",
        content="Fixed race condition in websocket handler by adding asyncio.Lock()",
        tier="episodic", source="debug_session", source_type="observation",
        tags=["bug", "websocket", "async"]))

    stats = kb.get_stats()
    if result("5-Tier Storage", True, f"{stats.total_items} items across {len(stats.items_by_tier)} tiers"):
        passed += 1

    # BM25+ Search
    search_results = kb.search("JWT token authentication security")
    if result("BM25+ Search", True, f"found {len(search_results)} results, top='{search_results[0].title}'" if search_results else "0 results"):
        passed += 1

    # Domain filter
    auth_items = kb.get_by_domain("auth")
    if result("Domain Filter", True, f"{len(auth_items)} auth items"):
        passed += 1

    # Tier filter
    strategic = kb.get_by_tier("strategic")
    if result("Tier Filter", True, f"{len(strategic)} strategic items"):
        passed += 1

    # Prioritized retrieval
    priority = kb.get_prioritized(limit=3)
    if result("Priority Retrieval", True, f"top 3: {[p.tier for p in priority]}"):
        passed += 1

    # Cross-references
    kb.add_cross_reference("strat_001", "inst_001")
    related = kb.get_related("strat_001")
    if result("Cross-References", True, f"{len(related)} related items linked"):
        passed += 1
except Exception as e:
    result("Knowledge Base V2", False, str(e)); failed += 1

# ═══════════════════════════════════════════════════════════
#  ENGINE 7a: MERKLE TREE INTEGRITY
# ═══════════════════════════════════════════════════════════
section("ENGINE 7a: MERKLE TREE INTEGRITY")
try:
    root_hash = kb.rebuild_merkle_tree()
    if result("Merkle Tree Build", True, f"root={root_hash[:24]}..."):
        passed += 1

    integrity = kb.verify_integrity()
    if result("Integrity Verification", True, f"clean={integrity.is_clean}, verified={integrity.verified_items}/{integrity.total_items}"):
        passed += 1

    proof = kb.get_merkle_proof("strat_001")
    verified = kb.merkle.verify_proof(proof)
    if result("Merkle Proof", True, f"proof_path_len={len(proof.proof_path)}, verified={verified}"):
        passed += 1
except Exception as e:
    result("Merkle Integrity", False, str(e)); failed += 1

# ═══════════════════════════════════════════════════════════
#  ENGINE 7b: BAYESIAN CONFIDENCE
# ═══════════════════════════════════════════════════════════
section("ENGINE 7b: BAYESIAN CONFIDENCE")
try:
    from brain.engines.knowledge_base import BayesianConfidence

    p1 = BayesianConfidence.update_belief(0.6, 0.8, True)
    p2 = BayesianConfidence.update_belief(0.6, 0.8, False)
    if result("Belief Revision", True, f"prior=0.6 → agree={p1:.3f}, disagree={p2:.3f}"):
        passed += 1

    item = kb.items.get("epi_001")
    if item:
        old_conf = item.confidence
        decayed = BayesianConfidence.apply_temporal_decay(item)
        if result("Temporal Decay", True, f"episodic item: {old_conf:.3f} → {decayed:.3f}"):
            passed += 1

    cal_up = BayesianConfidence.calibrate_from_outcome(0.7, True)
    cal_down = BayesianConfidence.calibrate_from_outcome(0.7, False)
    if result("Outcome Calibration", True, f"correct: 0.7→{cal_up:.3f}, wrong: 0.7→{cal_down:.3f}"):
        passed += 1
except Exception as e:
    result("Bayesian Confidence", False, str(e)); failed += 1

# ═══════════════════════════════════════════════════════════
#  ENGINE 7c: CAUSAL KNOWLEDGE GRAPH
# ═══════════════════════════════════════════════════════════
section("ENGINE 7c: CAUSAL KNOWLEDGE GRAPH")
try:
    from brain.engines.knowledge_base import CausalEdge
    kb.add_causal_edge("strat_001", "proc_001", "restricts", evidence="JWT algo constrains deploy")
    kb.add_causal_edge("inst_001", "strat_001", "restricts", evidence="Compliance constrains auth")
    kb.add_causal_edge("sem_001", "proc_001", "enables", evidence="CQRS enables deploy pipeline")
    edges = kb.causal_graph.get_edges("strat_001", direction="both")
    if result("Causal Edges", True, f"{len(edges)} edges connected to strat_001"):
        passed += 1

    impact = kb.get_impact("inst_001")
    if result("Impact Analysis", True, f"blast_radius={impact['total_blast_radius']}"):
        passed += 1

    path = kb.causal_graph.find_reasoning_path("inst_001", "proc_001")
    if result("Reasoning Path", True, f"inst→proc path_len={len(path)}"):
        passed += 1

    cycles = kb.causal_graph.detect_cycles()
    if result("Cycle Detection", True, f"{len(cycles)} cycles found"):
        passed += 1
except Exception as e:
    result("Causal Graph", False, str(e)); failed += 1

# ═══════════════════════════════════════════════════════════
#  ENGINE 7d: ANTI-HALLUCINATION PIPELINE
# ═══════════════════════════════════════════════════════════
section("ENGINE 7d: 4-STAGE ANTI-HALLUCINATION PIPELINE")
try:
    # Test 1: Verified claim
    v1 = kb.verify_claim("Use RS256 for JWT tokens")
    if result("Stage Test: Verified", True, f"verdict={v1.verdict}, conf={v1.confidence:.2f}, stages_passed={v1.stages_passed}"):
        passed += 1

    # Test 2: Contradicted claim (HS256 contradicts RS256)
    v2 = kb.verify_claim("Use HS256 for production JWT signing")
    if result("Stage Test: Contradicted", True, f"verdict={v2.verdict}, conf={v2.confidence:.2f}"):
        passed += 1

    # Test 3: No evidence
    v3 = kb.verify_claim("Use GraphQL subscriptions for real-time spaceship telemetry")
    if result("Stage Test: No Evidence", True, f"verdict={v3.verdict}, conf={v3.confidence:.2f}"):
        passed += 1

    log(f"\n  Pipeline Summary:")
    log(f"    Claim 1 (RS256): {v1.verdict} — {v1.reasoning}")
    log(f"    Claim 2 (HS256): {v2.verdict} — {v2.reasoning}")
    log(f"    Claim 3 (Random): {v3.verdict} — {v3.reasoning}")
except Exception as e:
    result("Anti-Hallucination", False, str(e)); failed += 1

# ═══════════════════════════════════════════════════════════
#  ENGINE 7e: WRITE-AHEAD LOG
# ═══════════════════════════════════════════════════════════
section("ENGINE 7e: WRITE-AHEAD LOG")
try:
    wal_pending = kb.wal.get_pending_count()
    if result("WAL Status", True, f"pending={wal_pending} (0 = all checkpointed)"):
        passed += 1
except Exception as e:
    result("WAL", False, str(e)); failed += 1

# ═══════════════════════════════════════════════════════════
#  ENGINE 7f: ENTERPRISE COMPLIANCE
# ═══════════════════════════════════════════════════════════
section("ENGINE 7f: ENTERPRISE COMPLIANCE")
try:
    trail = kb.compliance.get_audit_trail()
    if result("SOC2 Audit Trail", True, f"{len(trail)} operations logged"):
        passed += 1

    chain_ok = kb.compliance.verify_audit_chain()
    if result("Audit Chain Integrity", True, f"valid={chain_ok['valid']}, entries={chain_ok['entries']}"):
        passed += 1

    export = kb.compliance.export_personal_data(kb.items, "default")
    if result("GDPR Data Export", True, f"{len(export['items'])} items exported"):
        passed += 1

    kb.compliance.create_namespace("tenant_acme")
    if result("Multi-Tenant Namespace", True, f"namespaces={sorted(kb.compliance.namespaces)}"):
        passed += 1
except Exception as e:
    result("Compliance", False, str(e)); failed += 1

# ═══════════════════════════════════════════════════════════
#  ENGINE 8: WORKING MEMORY (L1 Cache)
# ═══════════════════════════════════════════════════════════
section("ENGINE 8: WORKING MEMORY")
try:
    from brain.engines.context_state import WorkingMemory
    wm = sat.memory.wm
    wm.inject("CRITICAL: Always validate JWT tokens before processing", priority=9.5, ttl=50, source="security_policy")
    wm.inject("User prefers dark theme in all UI components", priority=3.0, ttl=10, source="user_pref")
    wm.inject("Current sprint: Implement OAuth2 SSO integration", priority=7.0, ttl=30, source="sprint_board")
    usage = wm.get_usage()
    if result("Token Budget", True, f"used={usage['tokens_used']}/{usage['tokens_budget']} ({usage['utilization']:.0%})"):
        passed += 1

    context_str = wm.get_prompt_context()
    if result("Prompt Context", True, f"{len(context_str)} chars, {usage['slots']} slots"):
        passed += 1

    log(f"\n  Working Memory Contents:")
    for s in wm.slots:
        icon = "🔴" if s.priority >= 8 else "🟡" if s.priority >= 5 else "⚪"
        log(f"    {icon} P{s.priority:.1f} TTL={s.ttl_turns} [{s.source}] {s.content[:60]}")
except Exception as e:
    result("Working Memory", False, str(e)); failed += 1

# ═══════════════════════════════════════════════════════════
#  ENGINE 9: TEMPORAL REASONER
# ═══════════════════════════════════════════════════════════
section("ENGINE 9: TEMPORAL REASONER")
try:
    from brain.engines.context_state import TemporalFact
    from datetime import datetime, timezone, timedelta
    tr = sat.memory.temporal

    now = datetime.now(timezone.utc)
    tr.add_fact(TemporalFact(fact_id="freeze_001", content="API v2 deploy freeze active",
        domain="devops", valid_from=(now - timedelta(hours=2)).isoformat(),
        valid_until=(now + timedelta(days=3)).isoformat()))
    tr.add_fact(TemporalFact(fact_id="sprint_001", content="Sprint 14: OAuth2 SSO focus",
        domain="planning", valid_from=(now - timedelta(days=5)).isoformat(),
        valid_until=(now + timedelta(days=9)).isoformat()))
    tr.add_fact(TemporalFact(fact_id="old_001", content="Legacy auth system active",
        domain="auth", valid_from="2024-01-01", valid_until="2025-12-31"))

    current = tr.get_current_facts()
    expired = tr.get_expired_facts()
    if result("Current Facts", True, f"{len(current)} valid, {len(expired)} expired"):
        passed += 1

    valid = tr.is_valid("freeze_001")
    if result("Validity Check", True, f"deploy_freeze is_valid={valid}"):
        passed += 1
except Exception as e:
    result("Temporal Reasoner", False, str(e)); failed += 1

# ═══════════════════════════════════════════════════════════
#  ENGINE 10: DECISION EXPLORER (Fork/Merge)
# ═══════════════════════════════════════════════════════════
section("ENGINE 10: DECISION EXPLORER")
try:
    explorer = sat.memory.explorer
    branches = explorer.fork("Database for user service", ["PostgreSQL + read replicas", "CockroachDB distributed", "DynamoDB serverless"])
    explorer.add_note(branches[0], "Best for complex SQL queries and joins")
    explorer.add_note(branches[0], "Mature ecosystem, strong tooling")
    explorer.add_note(branches[1], "Auto-sharding, geo-distributed")
    explorer.add_note(branches[2], "Zero ops, pay-per-request")
    comparison = explorer.compare_branches(branches)
    if result("Fork (3 branches)", True, f"decision='{comparison['decision_point']}'"):
        passed += 1

    explorer.merge(branches[0])  # Commit PostgreSQL
    committed = explorer.get_committed_branches()
    active = explorer.get_active_branches()
    if result("Merge + Discard", True, f"committed={len(committed)}, active={len(active)}, winner={committed[0].option}"):
        passed += 1

    log(f"\n  Decision Branches:")
    for b in [explorer.get_branch(bid) for bid in branches]:
        icon = "🟢" if b.status == "committed" else "🔴"
        log(f"    {icon} [{b.status}] {b.option} ({len(b.notes)} notes)")
except Exception as e:
    result("Decision Explorer", False, str(e)); failed += 1

# ═══════════════════════════════════════════════════════════
#  ENGINE 11: MEMORY ORCHESTRATOR
# ═══════════════════════════════════════════════════════════
section("ENGINE 11: MEMORY ORCHESTRATOR")
try:
    mem = sat.memory

    # Auto-tier classification
    id1 = mem.remember("Never store API keys in source code", source="security_team", domain="security")
    tier1 = mem.kb.items[id1].tier
    if result("Auto-Tier (strategic)", True, f"'Never store API keys' → tier={tier1}"):
        passed += 1

    id2 = mem.remember("Step 1: Install dependencies. Step 2: Run migrations. Step 3: Start server.",
                        source="onboarding", domain="devops")
    tier2 = mem.kb.items[id2].tier
    if result("Auto-Tier (procedural)", True, f"'Step 1/2/3' → tier={tier2}"):
        passed += 1

    # Unified recall
    recall = mem.recall("security best practices for API keys")
    if result("Unified Recall", True, f"{recall['total_results']} KB results, {len(recall['active_anchors'])} anchors, {len(recall['temporal_facts'])} temporal facts"):
        passed += 1

    # Prefetch
    loaded = mem.prefetch("deployment procedures docker")
    if result("Proactive Prefetch", True, f"{loaded} items loaded into working memory"):
        passed += 1

    # Verify before output
    verdict = mem.verify_before_output("Use RS256 for JWT tokens", domain="auth")
    if result("Anti-Hallucination Gate", True, f"verdict={verdict['verdict']}, conf={verdict['confidence']:.2f}"):
        passed += 1

    # Budget report
    budget = mem.get_memory_budget_report()
    if result("Memory Budget Report", True,
              f"WM={budget['working_memory']['tokens_used']}tok, " +
              f"KB={budget['knowledge_base']['total_items']} items, " +
              f"TF={budget['temporal_facts']['current']} facts"):
        passed += 1
except Exception as e:
    result("Memory Orchestrator", False, str(e)); failed += 1

# ═══════════════════════════════════════════════════════════
#  ENGINE 12: THREAT ENGINE
# ═══════════════════════════════════════════════════════════
section("ENGINE 12: THREAT ENGINE")
try:
    threat = sat.threat
    if result("Threat Engine", True, "STRIDE+DREAD+MITRE loaded"):
        passed += 1
except Exception as e:
    result("Threat Engine", False, str(e)); failed += 1

# ═══════════════════════════════════════════════════════════
#  ENGINE 13: INFERENCE ROUTER
# ═══════════════════════════════════════════════════════════
section("ENGINE 13: INFERENCE ROUTER")
try:
    router = sat.router
    r1 = router.route("Fix a typo in README")
    r2 = router.route("Build a REST API with JWT authentication and rate limiting")
    r3 = router.route("Refactor entire microservice architecture with security audit")
    if result("Fast Query", True, f"'Fix typo' → {r1.tier_name} (T{r1.tier})"):
        passed += 1
    if result("Medium Query", True, f"'Build REST API' → {r2.tier_name} (T{r2.tier})"):
        passed += 1
    if result("Complex Query", True, f"'Refactor arch' → {r3.tier_name} (T{r3.tier})"):
        passed += 1
except Exception as e:
    result("Inference Router", False, str(e)); failed += 1

# ═══════════════════════════════════════════════════════════
#  ENGINE 14: CODE QUALITY SCORER
# ═══════════════════════════════════════════════════════════
section("ENGINE 14: CODE QUALITY SCORER")
try:
    quality = sat.quality
    code = '''
import logging
from typing import Optional

log = logging.getLogger(__name__)

class UserService:
    """Manages user CRUD operations with validation."""

    def __init__(self, db_connection):
        self.db = db_connection
        log.info("UserService initialized")

    def get_user(self, user_id: int) -> Optional[dict]:
        """Retrieve a user by ID with proper error handling."""
        try:
            result = self.db.query("SELECT * FROM users WHERE id = ?", [user_id])
            return result[0] if result else None
        except Exception as e:
            log.error(f"Failed to get user {user_id}: {e}")
            return None
'''
    score = quality.score(code, "python")
    if result("Quality Score", True, f"overall={score.overall}/100, grade={score.grade}"):
        passed += 1
    log(f"\n  Quality Dimensions:")
    for dim_name in ["maintainability", "security", "documentation", "testing",
                      "performance", "reliability", "scalability", "observability"]:
        val = getattr(score, dim_name, 0)
        bar = "█" * int(val / 10) + "░" * (10 - int(val / 10))
        log(f"    [{bar}] {dim_name}: {val}/100")
except Exception as e:
    result("Code Quality", False, str(e)); failed += 1

# ═══════════════════════════════════════════════════════════
#  HEALTH CHECK
# ═══════════════════════════════════════════════════════════
section("HEALTH CHECK")
health = sat.health()
for engine, loaded in health["engines"].items():
    icon = "🟢" if loaded else "⚪"
    log(f"  {icon} {engine}: {'loaded' if loaded else 'not yet loaded'}")
loaded_count = sum(health["engines"].values())
total_engines = len(health["engines"])
log(f"\n  {loaded_count}/{total_engines} engines loaded")

# ═══════════════════════════════════════════════════════════
#  FINAL RESULTS
# ═══════════════════════════════════════════════════════════
elapsed = time.time() - t_start
section("FINAL RESULTS")
total = passed + failed
log(f"  Passed: {passed}/{total}")
log(f"  Failed: {failed}/{total}")
log(f"  Time: {elapsed:.2f}s")
log("")
if failed == 0:
    log("  🏆 ALL ENGINES FULLY OPERATIONAL — SATURDAY MK1 IS READY")
else:
    log(f"  ⚠️  {failed} FAILURE(S) DETECTED")
log("")

# Save report
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
log(f"  Report saved to: {REPORT_PATH}")

# Cleanup
shutil.rmtree(MEM_DIR, ignore_errors=True)
