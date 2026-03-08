"""
Saturday MK1 — Enterprise AI Coding Engine (Core Orchestrator)
===============================================================
Main class integrating all engine modules + LLM into a unified
AI coding system. This is the single entrypoint for all Saturday
operations: code generation, security validation, threat analysis,
strategic planning, and conversational AI.

Usage:
    saturday = Saturday(project_root="./my_project")
    saturday.scan_project()

    # Generate code with auto-validation
    result = saturday.generate("Build a REST API for user management", language="python")

    # Chat with full context
    response = saturday.chat("How should I structure the auth module?")

    # Validate existing code
    validation = saturday.validate_code(code, "app.py", "python")

    # Plan before executing
    plan = saturday.plan("Refactor auth module to support SSO")
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("saturday-core")


@dataclass
class ValidationResult:
    """Result of a code validation check."""
    passed: bool
    findings: list = field(default_factory=list)
    security_score: float = 1.0
    quality_score: float = 1.0
    quality_grade: str = ""


@dataclass
class PlanResult:
    """Result of strategic planning."""
    plan: Any = None
    risks: list = field(default_factory=list)
    estimated_effort: str = ""


@dataclass
class GenerateResult:
    """Result of code generation."""
    code: str = ""
    language: str = ""
    validation: Optional[ValidationResult] = None
    tokens_used: int = 0
    latency_seconds: float = 0.0
    model: str = ""


class Saturday:
    """
    Saturday MK1 — Enterprise AI Coding Engine.

    Core orchestrator integrating:
    - LLMProvider: Multi-backend LLM calls (OpenAI/Anthropic/HuggingFace)
    - SecurityPipeline: 12-layer defense-in-depth with CVSS scoring
    - CodeGraphEngine: Multi-language AST, call graph, architecture detection
    - StrategicPlanner: Dependency-aware planning with blast radius v2
    - ContextStateEngine: Priority escalation, drift detection, snapshots
    - KnowledgeBase: 5-tier hierarchical memory with anti-hallucination guard
    - ThreatEngine: Red Team simulation (STRIDE+DREAD+MITRE ATT&CK)
    - InferenceRouter: 4-tier + MoE expert routing with cost optimization
    - CodeQualityScorer: 8-dimension enterprise quality grading
    - DataPipeline: 12-stage ML training data processing
    """

    VERSION = "1.0.0"

    def __init__(self, project_root: str = ".", memory_dir: Optional[str] = None):
        self.project_root = Path(project_root).resolve()
        self.memory_dir = Path(memory_dir or self.project_root / ".saturday_memory")
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        # Lazy-loaded engines
        self._llm = None
        self._security = None
        self._code_graph = None
        self._planner = None
        self._context = None
        self._knowledge = None
        self._threat = None
        self._router = None
        self._quality = None
        self._memory = None

        log.info(f"Saturday MK1 v{self.VERSION} initialized | project={self.project_root}")

    # ── Engine Properties (Lazy Init) ──

    @property
    def llm(self):
        """LLM provider — auto-configured from environment variables."""
        if self._llm is None:
            from .engines.llm_provider import LLMProvider
            self._llm = LLMProvider.from_env()
            log.info(f"LLM provider: {self._llm.config.provider} / {self._llm.config.model}")
        return self._llm

    @property
    def security(self):
        if self._security is None:
            from .engines.security_pipeline import SecurityPipeline
            self._security = SecurityPipeline()
        return self._security

    @property
    def code_graph(self):
        if self._code_graph is None:
            from .engines.code_graph import CodeGraphEngine
            graph_path = str(self.memory_dir / "project_graph.json")
            self._code_graph = CodeGraphEngine(str(self.project_root), graph_path)
        return self._code_graph

    @property
    def planner(self):
        if self._planner is None:
            from .engines.strategic_planner import StrategicPlanner
            self._planner = StrategicPlanner()
        return self._planner

    @property
    def context(self):
        if self._context is None:
            from .engines.context_state import ContextStateEngine
            self._context = ContextStateEngine()
        return self._context

    @property
    def knowledge(self):
        if self._knowledge is None:
            from .engines.knowledge_base import KnowledgeBase
            kb_path = str(self.memory_dir / "knowledge")
            self._knowledge = KnowledgeBase(kb_path)
        return self._knowledge

    @property
    def memory(self):
        """Memory Orchestrator — unified coordinator for all memory layers."""
        if self._memory is None:
            from .engines.memory_orchestrator import MemoryOrchestrator
            from .engines.context_state import WorkingMemory
            wm = WorkingMemory(token_budget=8192)
            self._memory = MemoryOrchestrator(self.knowledge, self.context, wm)
        return self._memory

    @property
    def threat(self):
        if self._threat is None:
            from .engines.threat_engine import ThreatEngine
            self._threat = ThreatEngine()
        return self._threat

    @property
    def router(self):
        if self._router is None:
            from .engines.inference_router import InferenceRouter
            self._router = InferenceRouter()
        return self._router

    @property
    def quality(self):
        if self._quality is None:
            from .engines.code_quality import CodeQualityScorer
            self._quality = CodeQualityScorer()
        return self._quality

    # ── LLM Configuration ──

    def set_llm_provider(self, provider):
        """Manually set the LLM provider (for testing or custom configs)."""
        self._llm = provider

    # ── Code Generation ──

    def generate(
        self,
        task: str,
        language: str = "python",
        context: str = "",
        validate: bool = True,
        max_tokens: int = 4096,
        max_fix_retries: int = 1,
    ) -> GenerateResult:
        """
        Generate production-ready code with automatic security validation
        and self-healing fix loop.

        Args:
            task: What the code should do
            language: Target programming language
            context: Additional context (project structure, conventions, etc.)
            validate: Whether to auto-validate generated code
            max_tokens: Maximum tokens for generation
            max_fix_retries: Number of fix attempts if validation fails (0 to disable)

        Returns:
            GenerateResult with code, validation results, and metadata
        """
        # Build context from project understanding
        full_context = self._build_context(context)

        # Route to determine optimal model parameters
        routing = self.router.route(task)
        log.info(
            f"Generating code: tier={routing.tier_name}, "
            f"expert={routing.expert_model}"
        )

        # Generate via LLM
        from .engines.llm_provider import LLMProvider
        response = self.llm.generate_code(
            task=task,
            language=language,
            context=full_context,
            max_tokens=max_tokens,
        )

        # Extract code from response
        code = LLMProvider.extract_code(response.content, language) or response.content
        total_tokens = response.tokens_used
        total_latency = response.latency_seconds

        result = GenerateResult(
            code=code,
            language=language,
            tokens_used=total_tokens,
            latency_seconds=total_latency,
            model=response.model,
        )

        # Auto-validate if enabled
        if validate and code:
            result.validation = self.validate_code(code, f"generated.{language}", language)

            # Self-healing fix loop: if validation fails, send findings back to LLM
            if not result.validation.passed and max_fix_retries > 0 and result.validation.findings:
                for retry in range(max_fix_retries):
                    log.info(
                        f"Fix attempt {retry + 1}/{max_fix_retries}: "
                        f"{len(result.validation.findings)} findings to fix"
                    )

                    # Build fix prompt with original code + findings
                    findings_text = "\n".join(
                        f"- {getattr(f, 'rule', 'issue')}: {getattr(f, 'message', str(f))}"
                        for f in result.validation.findings[:10]
                    )
                    fix_prompt = (
                        f"The following {language} code has security/quality issues that must be fixed.\n\n"
                        f"ORIGINAL CODE:\n```{language}\n{code}\n```\n\n"
                        f"FINDINGS TO FIX:\n{findings_text}\n\n"
                        f"Rewrite the COMPLETE code with ALL findings fixed. "
                        f"Output the corrected code in a single ```{language} code block."
                    )

                    fix_response = self.llm.generate(
                        prompt=fix_prompt,
                        system=(
                            "You are Saturday MK1 fixing security and quality issues in code. "
                            "Fix ALL identified issues. Use parameterized queries, bcrypt/argon2 "
                            "for passwords, secrets.token_hex() for tokens, and proper error handling."
                        ),
                        max_tokens=max_tokens,
                        temperature=0.1,
                    )

                    fixed_code = LLMProvider.extract_code(fix_response.content, language) or fix_response.content
                    total_tokens += fix_response.tokens_used
                    total_latency += fix_response.latency_seconds

                    # Re-validate fixed code
                    new_validation = self.validate_code(fixed_code, f"generated.{language}", language)

                    if new_validation.passed or len(new_validation.findings) < len(result.validation.findings):
                        code = fixed_code
                        result.code = code
                        result.validation = new_validation
                        result.tokens_used = total_tokens
                        result.latency_seconds = total_latency
                        log.info(f"Fix attempt {retry + 1} improved code: {len(new_validation.findings)} findings remaining")

                    if new_validation.passed:
                        break

            if not result.validation.passed:
                log.warning(
                    f"Generated code has {len(result.validation.findings)} security findings"
                )

        return result

    # ── Conversational AI ──

    def chat(
        self,
        message: str,
        history: Optional[list[dict]] = None,
        include_context: bool = True,
    ) -> dict:
        """
        Conversational interface with full engine context.

        Args:
            message: User's message
            history: Previous conversation messages [{"role": "user/assistant", "content": "..."}]
            include_context: Whether to inject project context

        Returns:
            Dict with response, tokens_used, latency
        """
        from .engines.llm_provider import LLMMessage, SATURDAY_SYSTEM_PROMPT

        # Build system prompt with context
        system = SATURDAY_SYSTEM_PROMPT
        if include_context:
            ctx = self._build_context("")
            if ctx:
                system += f"\n\nProject Context:\n{ctx}"

        # Build message list
        messages = []
        if history:
            for msg in history:
                messages.append(LLMMessage(
                    role=msg.get("role", "user"),
                    content=msg.get("content", ""),
                ))
        messages.append(LLMMessage(role="user", content=message))

        # Call LLM
        response = self.llm.chat(
            messages=messages,
            system=system,
        )

        return {
            "response": response.content,
            "tokens_used": response.tokens_used,
            "latency_seconds": response.latency_seconds,
            "model": response.model,
        }

    # ── Code Validation ──

    def validate_code(self, code: str, filename: str, language: str) -> ValidationResult:
        """
        Run security + quality validation on code.

        Args:
            code: Source code to validate
            filename: Name of the file
            language: Programming language

        Returns:
            ValidationResult with findings and pass/fail
        """
        # Security scan
        sec_findings = self.security.scan_code(code, filename, language)

        # Quality check
        quality_result = self.quality.score(code, language)

        # Determine pass/fail
        critical_findings = [
            f for f in sec_findings
            if getattr(f, 'severity', 'medium') == 'critical'
        ]
        passed = len(critical_findings) == 0 and quality_result.overall >= 40

        return ValidationResult(
            passed=passed,
            findings=sec_findings,
            security_score=max(0, 100 - len(sec_findings) * 15) / 100,
            quality_score=quality_result.overall / 100,
            quality_grade=quality_result.grade,
        )

    # ── Project Scanning ──

    def scan_project(self) -> dict:
        """
        Scan the project to build architectural understanding.

        Returns:
            Project summary with file counts, structure, and key patterns.
        """
        summary = self.code_graph.scan_directory()
        log.info(f"Project scanned: {self.project_root}")
        return summary

    # ── Strategic Planning ──

    def plan(self, task_description: str, context: Optional[dict] = None) -> PlanResult:
        """
        Create a strategic execution plan before coding.

        Args:
            task_description: What needs to be done
            context: Additional context (project state, constraints)

        Returns:
            PlanResult with execution plan and risk assessment
        """
        plan = self.planner.create_plan(task_description, context or {})
        return PlanResult(
            plan=plan,
            risks=[r for r in plan.risks] if hasattr(plan, 'risks') else [],
            estimated_effort=getattr(plan, 'estimated_effort', 'unknown'),
        )

    # ── Threat Analysis ──

    def analyze_threats(self, code: str, context: str = "") -> dict:
        """
        Perform threat modeling on code or architecture.

        Args:
            code: Code or architecture description to analyze
            context: Additional context

        Returns:
            Threat report with vectors, vulnerabilities, and recommendations
        """
        report = self.threat.analyze(code, context)
        return {
            "threat_vectors": len(report.vectors) if hasattr(report, 'vectors') else 0,
            "vulnerabilities": len(report.vulnerabilities) if hasattr(report, 'vulnerabilities') else 0,
            "risk_level": getattr(report, 'overall_risk', 'unknown'),
            "report": report,
        }

    # ── Query Routing ──

    def route_query(self, query: str) -> dict:
        """
        Route a query to the optimal processing tier.

        Args:
            query: User's coding request

        Returns:
            Routing decision with tier, expert, and resource estimates
        """
        decision = self.router.route(query)
        return {
            "tier": decision.tier_name,
            "expert": decision.expert_model,
            "complexity": decision.tier,
            "confidence": decision.confidence,
            "estimated_tokens": decision.estimated_tokens,
        }

    # ── Context & Knowledge ──

    def record_decision(self, domain: str, decision: str, rationale: str):
        """Record an architectural decision for cross-session persistence."""
        from .engines.context_state import DecisionRecord
        record = DecisionRecord(
            domain=domain,
            decision=decision,
            rationale=rationale,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self.context.record_decision(record)

    def add_anchor(self, instruction: str, priority: str = "high"):
        """Add an instruction anchor that persists across turns."""
        from .engines.context_state import InstructionAnchor
        anchor = InstructionAnchor(
            instruction=instruction,
            priority=priority,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self.context.add_anchor(anchor)

    def get_file_outline(self, filepath: str) -> dict:
        """Get a structured outline of a source file."""
        return self.code_graph.get_file_outline(filepath)

    def verify_response(self, claim: str, domain: str = None) -> dict:
        """Anti-hallucination gate: verify a claim before outputting it."""
        return self.memory.verify_before_output(claim, domain=domain)

    # ── Internal Helpers ──

    def _build_context(self, user_context: str) -> str:
        """Build rich context from project understanding + user-provided context."""
        parts = []

        if user_context:
            parts.append(user_context)

        # Add project structure if scanned
        try:
            graph_data = self.code_graph.graph
            if graph_data and hasattr(graph_data, 'nodes') and graph_data.nodes:
                file_count = len(graph_data.nodes)
                parts.append(f"Project: {self.project_root.name} ({file_count} files)")
        except Exception:
            pass

        # Add active instruction anchors
        try:
            anchors = self.context.get_active_anchors()
            if anchors:
                anchor_text = "\n".join(
                    f"- [{a.priority}] {a.instruction}" for a in anchors[:5]
                )
                parts.append(f"Active Instructions:\n{anchor_text}")
        except Exception:
            pass

        # Add relevant knowledge
        try:
            if parts:
                query = parts[0][:200]
                relevant = self.knowledge.search(query, top_k=3)
                if relevant:
                    kb_text = "\n".join(
                        f"- {item.content[:150]}" for item in relevant[:3]
                    )
                    parts.append(f"Relevant Knowledge:\n{kb_text}")
        except Exception:
            pass

        return "\n\n".join(parts) if parts else ""

    # ── Health Check ──

    def health(self) -> dict:
        """Return system health status."""
        status = {
            "version": self.VERSION,
            "project_root": str(self.project_root),
            "engines": {
                "security": self._security is not None,
                "code_graph": self._code_graph is not None,
                "planner": self._planner is not None,
                "context": self._context is not None,
                "knowledge": self._knowledge is not None,
                "memory": self._memory is not None,
                "threat": self._threat is not None,
                "router": self._router is not None,
                "quality": self._quality is not None,
                "llm": self._llm is not None,
            },
            "llm_stats": self._llm.get_stats() if self._llm else None,
            "memory_budget": self._memory.get_memory_budget_report() if self._memory else None,
        }
        return status
