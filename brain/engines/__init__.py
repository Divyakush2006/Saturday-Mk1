# Saturday Brain — Engine Package
# ================================
# This package contains Saturday's 11 core engine modules + LLM provider.
# Together they form the most advanced AI coding engine in the market.

from .knowledge_base import (
    KnowledgeBase, KnowledgeItem, VerificationResult,
    BM25PlusEngine, MerkleIntegrityLayer, BayesianConfidence,
    CausalKnowledgeGraph, CausalEdge, AntiHallucinationPipeline,
    WriteAheadLog, ComplianceLayer, HallucinationVerdict,
    ContradictionReport, MerkleProof, IntegrityReport, AuditEntry, MemoryStats,
)
from .context_state import (
    ContextStateEngine, InstructionAnchor, DecisionRecord, CoherenceMetrics,
    WorkingMemory, WorkingMemorySlot,
    TemporalReasoner, TemporalFact,
    DecisionExplorer, DecisionBranch,
    ContextEvent, ContextSnapshot, DriftAlert,
)
from .memory_orchestrator import MemoryOrchestrator
from .security_pipeline import SecurityPipeline, SecurityFinding
from .code_graph import CodeGraphEngine, ProjectNode, CallEdge, ArchitecturePattern
from .strategic_planner import StrategicPlanner, ExecutionPlan, TaskNode, BlastRadius
from .threat_engine import ThreatEngine, ThreatVector, Vulnerability, ThreatReport, AttackChain
from .inference_router import InferenceRouter, RoutingResult
from .code_quality import CodeQualityScorer, QualityScore, QualityIssue
from .data_pipeline import DataPipeline, DataSample, PipelineStats
from .llm_provider import (
    LLMProvider, LLMConfig, LLMMessage, LLMResponse,
    OpenAICompatibleProvider, AnthropicProvider, HuggingFaceLocalProvider,
    SATURDAY_SYSTEM_PROMPT, PromptEngine,
)

__all__ = [
    # Memory Architecture V2
    "KnowledgeBase", "KnowledgeItem", "VerificationResult",
    "BM25PlusEngine", "MerkleIntegrityLayer", "BayesianConfidence",
    "CausalKnowledgeGraph", "CausalEdge", "AntiHallucinationPipeline",
    "WriteAheadLog", "ComplianceLayer", "HallucinationVerdict",
    "ContradictionReport", "MerkleProof", "IntegrityReport", "AuditEntry", "MemoryStats",
    # Context State V2
    "ContextStateEngine", "InstructionAnchor", "DecisionRecord", "CoherenceMetrics",
    "WorkingMemory", "WorkingMemorySlot",
    "TemporalReasoner", "TemporalFact",
    "DecisionExplorer", "DecisionBranch",
    "ContextEvent", "ContextSnapshot", "DriftAlert",
    # Memory Orchestrator
    "MemoryOrchestrator",
    # Security & Threat
    "SecurityPipeline", "SecurityFinding",
    "ThreatEngine", "ThreatVector", "Vulnerability", "ThreatReport", "AttackChain",
    # Code Understanding
    "CodeGraphEngine", "ProjectNode", "CallEdge", "ArchitecturePattern",
    "CodeQualityScorer", "QualityScore", "QualityIssue",
    # Planning & Routing
    "StrategicPlanner", "ExecutionPlan", "TaskNode", "BlastRadius",
    "InferenceRouter", "RoutingResult",
    # Data
    "DataPipeline", "DataSample", "PipelineStats",
    # LLM Provider
    "LLMProvider", "LLMConfig", "LLMMessage", "LLMResponse",
    "OpenAICompatibleProvider", "AnthropicProvider", "HuggingFaceLocalProvider",
    "SATURDAY_SYSTEM_PROMPT", "PromptEngine",
]
