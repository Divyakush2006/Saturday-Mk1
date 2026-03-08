"""
Saturday MK1 — LLM Provider Abstraction Layer
===============================================
Universal interface for connecting Saturday's brain engines to any LLM backend.
Supports OpenAI-compatible APIs (Kimi K2.5, vLLM, GPT, Qwen, DeepSeek),
Anthropic (Claude), and local HuggingFace inference.

Usage:
    provider = LLMProvider.from_env()
    response = provider.generate("Write a Python REST API", system="You are Saturday MK1")

Environment Variables:
    SATURDAY_PROVIDER    — "openai" (default), "anthropic", "huggingface"
    SATURDAY_API_KEY     — API key for the provider
    SATURDAY_API_BASE    — Base URL (for OpenAI-compatible endpoints)
    SATURDAY_MODEL       — Model name/path
    SATURDAY_MAX_RETRIES — Max retry attempts (default: 3)
"""

import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

log = logging.getLogger("saturday-llm")

# Auto-load .env file if present
def _load_dotenv():
    """Load .env file from project root (works without python-dotenv)."""
    import pathlib
    env_paths = [
        pathlib.Path(".env"),
        pathlib.Path(__file__).parent.parent.parent / ".env",
    ]
    for env_path in env_paths:
        if env_path.exists():
            try:
                with open(env_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        key, _, value = line.partition("=")
                        key = key.strip()
                        value = value.strip().strip("'\"")
                        if key and value and key not in os.environ:
                            os.environ[key] = value
                log.debug(f"Loaded .env from {env_path}")
                return
            except Exception:
                pass

_load_dotenv()


# ═══════════════════════════════════════════
# DATA TYPES
# ═══════════════════════════════════════════

@dataclass
class LLMMessage:
    """A single message in a conversation."""
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class LLMResponse:
    """Response from an LLM call."""
    content: str
    model: str = ""
    tokens_used: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_seconds: float = 0.0
    finish_reason: str = ""
    raw: dict = field(default_factory=dict)


@dataclass
class LLMConfig:
    """Configuration for an LLM provider."""
    provider: str = "openai"
    api_key: str = ""
    api_base: str = ""
    model: str = ""
    max_retries: int = 3
    timeout: int = 120
    default_max_tokens: int = 4096
    default_temperature: float = 0.2

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Load configuration from environment variables."""
        return cls(
            provider=os.getenv("SATURDAY_PROVIDER", "openai"),
            api_key=os.getenv("SATURDAY_API_KEY", ""),
            api_base=os.getenv("SATURDAY_API_BASE", "https://api.openai.com/v1"),
            model=os.getenv("SATURDAY_MODEL", "gpt-4o"),
            max_retries=int(os.getenv("SATURDAY_MAX_RETRIES", "3")),
            timeout=int(os.getenv("SATURDAY_TIMEOUT", "120")),
            default_max_tokens=int(os.getenv("SATURDAY_MAX_TOKENS", "4096")),
            default_temperature=float(os.getenv("SATURDAY_TEMPERATURE", "0.2")),
        )


# ═══════════════════════════════════════════
# SATURDAY SYSTEM PROMPTS
# ═══════════════════════════════════════════

SATURDAY_SYSTEM_PROMPT = (
    "You are Saturday MK1, the world's most advanced enterprise AI coding engine.\n\n"
    "You are NOT a chatbot. You are a production-grade software engineering system "
    "designed to replace entire development teams at Fortune 500 companies.\n\n"
    "THINKING METHODOLOGY:\n"
    "Before writing ANY code, briefly think step-by-step:\n"
    "1. Identify the core problem and requirements\n"
    "2. Consider security implications and attack vectors\n"
    "3. Choose the right design pattern for the use case\n"
    "4. Plan error handling and edge cases\n"
    "5. Then write the implementation\n\n"
    "SECURITY RULES (mandatory — never skip):\n"
    "- Passwords: ALWAYS use bcrypt or argon2 for hashing. NEVER use MD5 or SHA1 for passwords\n"
    "- Tokens: ALWAYS use secrets.token_hex() or secrets.token_urlsafe(). NEVER use random.randint()\n"
    "- SQL: ALWAYS use parameterized queries with %s or ? placeholders. NEVER use f-strings or .format()\n"
    "- Auth: ALWAYS validate JWT with expiry, issuer, and audience claims\n"
    "- Input: ALWAYS validate and sanitize ALL user inputs before processing\n"
    "- Secrets: NEVER hardcode API keys, passwords, or credentials in code\n"
    "- Crypto: Use AES-256-GCM or ChaCha20. NEVER use DES, RC4, or ECB mode\n\n"
    "ENTERPRISE PATTERNS (use when applicable):\n"
    "- Circuit Breaker: CLOSED/OPEN/HALF_OPEN states with configurable thresholds\n"
    "- Retry with exponential backoff and jitter\n"
    "- Thread safety: use threading.Lock or asyncio.Lock for shared state\n"
    "- Dependency injection for testability\n"
    "- Structured logging with context (not print statements)\n\n"
    "CODE QUALITY STANDARDS:\n"
    "- Type hints on ALL function signatures\n"
    "- Docstrings on ALL classes and public methods\n"
    "- Specific exception types (never bare except)\n"
    "- async/await for I/O-bound operations\n"
    "- Use dataclasses or Pydantic for data models\n"
    "- Follow the project's existing conventions and patterns\n\n"
    "DEBUGGING METHODOLOGY:\n"
    "- First, identify the ROOT CAUSE (not just the symptom)\n"
    "- Explain WHY the bug occurs (e.g., floating-point precision, type coercion)\n"
    "- Provide the fix with logging added for observability\n"
    "- Mention edge cases that could cause similar issues\n\n"
    "RESPONSE FORMAT (follow strictly):\n"
    "1. Start with ## Approach — a brief 1-3 line summary of your strategy\n"
    "2. Use ## Implementation — with clean code in fenced blocks with language tags\n"
    "3. End with ## Key Decisions — bullet points explaining important choices\n\n"
    "FORMATTING RULES:\n"
    "- Use ## headers to organize response sections\n"
    "- Use `backticks` for inline code, **bold** for important terms\n"
    "- Use bullet points and numbered lists for structured information\n"
    "- Use markdown tables when comparing options or listing parameters\n"
    "- Use > blockquotes for important notes or warnings\n\n"
    "You write code that is so good, human reviewers learn from it."
)

SATURDAY_CODE_PROMPT = (
    "You are Saturday MK1 generating production-ready code.\n\n"
    "Requirements:\n"
    "- Language: {language}\n"
    "- Task: {task}\n"
    "{context_section}\n\n"
    "{domain_instructions}"
    "Rules:\n"
    "1. Include all necessary imports at the top\n"
    "2. Add type hints to ALL function signatures and return types\n"
    "3. Add docstrings to ALL classes and public methods\n"
    "4. Use try/except with specific exception types (never bare except)\n"
    "5. Use parameterized queries with %s or ? placeholders for ALL database operations\n"
    "6. Use bcrypt or argon2 for password hashing (never MD5/SHA1)\n"
    "7. Use secrets.token_hex() or secrets.token_urlsafe() for token generation\n"
    "8. Use threading.Lock or asyncio.Lock for thread-safe shared state\n"
    "9. Use structured logging (import logging, not print)\n"
    "10. Never hardcode secrets, credentials, or API keys\n\n"
    "Output your code inside a single ```{language} code block."
)


# ═══════════════════════════════════════════
# PROMPT ENGINE — Multi-Stage Intelligence
# ═══════════════════════════════════════════

class PromptEngine:
    """
    Enterprise-grade multi-stage prompt pipeline.

    Architecture:
        Task → Classify → Adapt (multi-domain fusion) → Generate → Quality Gate → Refine

    This engine ensures Saturday produces comprehensive, professional output
    regardless of the specific task. It classifies the domain, injects the
    right methodology, and post-validates completeness.
    """

    # ── Domain Configurations ──
    # Each domain has: keywords (with weights), protocol instructions,
    # and required_concepts for the quality gate.
    DOMAINS = {
        "security_review": {
            "keywords": {
                # High-signal keywords (weight 3)
                "vulnerability": 3, "vulnerabilities": 3, "injection": 3,
                "sql injection": 3, "xss": 3, "csrf": 3, "owasp": 3,
                "exploit": 3, "cwe": 3,
                # Medium-signal (weight 2)
                "security": 2, "auth": 2, "secure": 2, "hack": 2,
                "sanitize": 2, "unsafe": 2, "dangerous": 2, "malicious": 2,
                # Low-signal (weight 1) - also appear in other contexts
                "fix": 1, "review": 1, "md5": 2, "sha1": 2,
                "hardcoded": 2, "random.randint": 3, "f-string": 1,
                "breach": 2, "attack": 2,
            },
            "protocol": (
                "SATURDAY SECURITY REVIEW PROTOCOL\n"
                "══════════════════════════════════\n"
                "You are performing a professional security audit. Follow this methodology:\n\n"
                "PHASE 1 — VULNERABILITY IDENTIFICATION:\n"
                "For EVERY vulnerability found, document:\n"
                "  • Vulnerability name (e.g., SQL Injection)\n"
                "  • CWE ID (e.g., CWE-89)\n"
                "  • Severity: Critical / High / Medium / Low\n"
                "  • Attack vector: how an attacker would exploit this\n"
                "  • Real-world impact: data breach, RCE, privilege escalation, etc.\n\n"
                "PHASE 2 — MANDATORY REMEDIATION:\n"
                "Apply ALL of these fixes when the vulnerability is present:\n\n"
                "  SQL Injection (CWE-89):\n"
                "    ✗ NEVER: f-strings, .format(), string concatenation in queries\n"
                "    ✓ ALWAYS: parameterized queries with prepared statements\n"
                "    ✓ Use placeholder values (%s for MySQL/PostgreSQL, ? for SQLite)\n"
                "    Example: cursor.execute(\"SELECT * FROM users WHERE name = %s\", (username,))\n\n"
                "  Weak Password Hashing (CWE-328):\n"
                "    ✗ NEVER: MD5, SHA1, SHA256 for passwords (these are NOT password hashes)\n"
                "    ✓ PRIMARY: Use bcrypt — import bcrypt; bcrypt.hashpw(password, bcrypt.gensalt())\n"
                "    ✓ ALTERNATIVE: Use argon2 — the winner of the Password Hashing Competition\n"
                "    ✓ ALWAYS mention BOTH bcrypt and argon2 as industry-standard options\n\n"
                "  Insecure Randomness (CWE-338):\n"
                "    ✗ NEVER: random.randint(), random.random() for security tokens\n"
                "    ✓ ALWAYS: import secrets; secrets.token_hex(32) for session tokens\n"
                "    ✓ The secrets module is designed for cryptographic security\n\n"
                "  Hardcoded Credentials (CWE-798):\n"
                "    ✓ Move ALL secrets to environment variables: os.environ.get('KEY')\n\n"
                "PHASE 3 — COMPLETE FIXED CODE:\n"
                "  Provide the entire rewritten code with ALL vulnerabilities fixed.\n\n"
                "PHASE 4 — SUMMARY TABLE:\n"
                "  | # | Vulnerability | CWE | Severity | Fix Applied |\n\n"
            ),
            "required_concepts": [
                "parameterized", "prepared", "placeholder", "%s",
                "bcrypt", "argon2", "secrets", "token_hex",
                "SQL injection", "MD5",
            ],
        },
        "debugging": {
            "keywords": {
                "traceback": 3, "exception": 2, "stack trace": 3,
                "error": 2, "bug": 2, "debug": 2,
                "not working": 2, "broken": 2, "crash": 2, "failing": 2,
                "assertionerror": 3, "typeerror": 3, "valueerror": 3,
                "keyerror": 3, "attributeerror": 3, "indexerror": 3,
                "runtime": 1, "production": 1, "causing": 1,
                "why": 1, "what's wrong": 2, "how do i fix": 2,
            },
            "protocol": (
                "SATURDAY DEBUGGING PROTOCOL\n"
                "═══════════════════════════\n"
                "You are diagnosing a production issue. Follow this structured methodology:\n\n"
                "PHASE 1 — ROOT CAUSE ANALYSIS:\n"
                "  • Identify the EXACT root cause (not symptoms)\n"
                "  • Explain the underlying mechanism:\n"
                "    - Numeric: Decimal vs float precision loss, int() truncation behavior\n"
                "    - Type: coercion rules, implicit conversions\n"
                "    - Concurrency: race conditions, deadlocks\n"
                "    - Memory: leaks, dangling references\n"
                "  • Trace the data flow step by step showing exact values\n"
                "  • For monetary calculations: explain how precision affects cents values\n\n"
                "PHASE 2 — PRODUCTION-SAFE FIX:\n"
                "  • Use round() for all precision-sensitive calculations (money, percentages)\n"
                "  • Replace assert statements with proper exceptions:\n"
                "    ✗ NEVER: assert in production (disabled with -O flag)\n"
                "    ✓ ALWAYS: raise ValueError('descriptive message') with proper context\n"
                "  • Add structured logging (import logging) for observability:\n"
                "    ✓ Log input values, intermediate calculations, and final results\n"
                "    ✓ Use log.warning() for edge cases, log.error() for failures\n"
                "  • Handle all edge cases: zero, None, negative, empty string, overflow\n\n"
                "PHASE 3 — COMPLETE FIXED CODE:\n"
                "  Provide the entire rewritten function with all fixes applied.\n\n"
                "PHASE 4 — PREVENTION:\n"
                "  • How to prevent similar bugs (type annotations, unit tests, linting rules)\n\n"
            ),
            "required_concepts": [
                "root cause", "precision", "round",
                "ValueError", "logging", "int",
            ],
        },
        "algorithm": {
            "keywords": {
                "implement": 2, "algorithm": 3, "data structure": 3,
                "cache": 2, "lru": 3, "tree": 2, "graph": 2,
                "sort": 2, "search": 2, "queue": 2, "stack": 2,
                "heap": 2, "linked list": 3, "hash": 1, "trie": 3,
                "o(1)": 3, "o(n)": 2, "o(log": 2,
                "complexity": 2, "optimal": 2, "efficient": 1,
                "from scratch": 2, "ordereddict": 3, "doubly-linked": 3,
            },
            "protocol": (
                "SATURDAY ALGORITHM PROTOCOL\n"
                "═══════════════════════════\n"
                "You are implementing a production-grade data structure. Follow this approach:\n\n"
                "PHASE 1 — COMPLEXITY ANALYSIS:\n"
                "  • State time complexity for each operation (get, put, insert, delete)\n"
                "  • State space complexity\n"
                "  • Justify your data structure choice\n\n"
                "PHASE 2 — IMPLEMENTATION:\n"
                "  • Use collections.OrderedDict when it fits (caches, LRU, ordered maps)\n"
                "    - Use move_to_end() to mark recently accessed items\n"
                "    - Use popitem(last=False) to evict oldest items\n"
                "  • Accept capacity/size as a constructor parameter\n"
                "  • Use a class-based design with clean public API\n\n"
                "PHASE 3 — THREAD SAFETY:\n"
                "  • import threading\n"
                "  • Use threading.Lock() for all mutable shared state\n"
                "  • Wrap every read/write operation with the lock\n\n"
                "PHASE 4 — ROBUSTNESS:\n"
                "  • Handle edge cases: zero capacity, missing keys, duplicate inserts\n"
                "  • Add comprehensive type hints (Optional, Union, Any as needed)\n"
                "  • Raise ValueError for invalid inputs\n"
                "  • Add docstrings with usage examples\n\n"
            ),
            "required_concepts": [
                "OrderedDict", "class", "get", "put", "capacity",
                "threading", "Lock", "move_to_end", "popitem",
            ],
        },
        "enterprise_pattern": {
            "keywords": {
                "circuit breaker": 3, "retry": 2, "rate limit": 3,
                "middleware": 2, "microservice": 3, "distributed": 2,
                "health check": 2, "resilience": 3, "bulkhead": 3,
                "saga": 3, "cqrs": 3, "event sourcing": 3,
                "observer": 2, "factory": 2, "singleton": 2,
                "decorator pattern": 3, "strategy": 2,
                "repository": 2, "service layer": 2, "clean architecture": 3,
            },
            "protocol": (
                "SATURDAY ENTERPRISE PATTERN PROTOCOL\n"
                "════════════════════════════════════\n"
                "Implement to Fortune 500 production standards:\n\n"
                "1. STATE MANAGEMENT: Use enums for all states (CLOSED/OPEN/HALF_OPEN etc.)\n"
                "2. CONFIGURABILITY: All thresholds as parameters with sensible defaults\n"
                "3. ASYNC SUPPORT: Provide both sync and async variants\n"
                "4. DECORATOR: Use functools.wraps for proper decorator implementation\n"
                "5. OBSERVABILITY: Structured logging (import logging) at every state transition\n"
                "6. METRICS: Track success/failure counts, latency, and state changes\n"
                "7. THREAD SAFETY: Use threading.Lock for concurrent access\n\n"
            ),
            "required_concepts": [
                "CircuitBreaker", "CLOSED", "OPEN", "HALF_OPEN",
                "threshold", "timeout", "async", "decorator",
                "wraps", "logging",
            ],
        },
        "code_generation": {
            "keywords": {
                "write": 1, "create": 1, "build": 1, "generate": 1,
                "make": 1, "endpoint": 2, "api": 2, "service": 1,
                "module": 1, "class": 1, "function": 1,
                "component": 1, "page": 1, "app": 1, "server": 1,
            },
            "protocol": (
                "APPROACH: Before writing code, state your approach in 2-3 lines.\n"
                "Consider the architecture, key design decisions, and edge cases.\n\n"
            ),
            "required_concepts": [],  # No specific requirements for general code gen
        },
    }

    # ── Stage 1: Classification ──

    @classmethod
    def classify(cls, task: str) -> list[str]:
        """
        Classify a task into domains using weighted keyword scoring.

        Returns list of matching domains ordered by weighted score (highest first).
        """
        task_lower = task.lower()
        scores = {}

        for domain, config in cls.DOMAINS.items():
            total_weight = 0
            for kw, weight in config["keywords"].items():
                if kw in task_lower:
                    total_weight += weight
            if total_weight > 0:
                scores[domain] = total_weight

        if not scores:
            return ["code_generation"]

        return sorted(scores, key=scores.get, reverse=True)

    # ── Stage 2: Prompt Adaptation ──

    @classmethod
    def adapt_prompt(cls, task: str, language: str, context: str = "") -> str:
        """
        Build a professional prompt with multi-domain fusion.

        Injects the primary domain's protocol. If a secondary domain
        has security implications, adds a security overlay.
        """
        context_section = f"\nContext:\n{context}" if context else ""
        domains = cls.classify(task)

        # Primary domain protocol
        primary = domains[0]
        protocol = cls.DOMAINS[primary]["protocol"]

        # Multi-domain fusion: if secondary domain is security, add overlay
        if len(domains) > 1 and "security_review" in domains[1:]:
            protocol += (
                "SECURITY OVERLAY: This task has security implications.\n"
                "Apply all security rules from the system prompt.\n\n"
            )

        return SATURDAY_CODE_PROMPT.format(
            language=language,
            task=task,
            context_section=context_section,
            domain_instructions=protocol,
        )

    # ── Stage 4: Quality Gate ──

    @classmethod
    def check_quality(cls, response: str, task: str) -> dict:
        """
        Post-generation quality gate.

        Checks if the response covers all required professional concepts
        for the detected domain. Returns gap analysis.
        """
        domains = cls.classify(task)
        primary = domains[0]
        required = cls.DOMAINS[primary].get("required_concepts", [])

        if not required:
            return {"passed": True, "missing": [], "coverage": 1.0}

        response_lower = response.lower()
        present = []
        missing = []

        for concept in required:
            if concept.lower() in response_lower:
                present.append(concept)
            else:
                missing.append(concept)

        coverage = len(present) / len(required) if required else 1.0

        return {
            "passed": coverage >= 0.95,  # Allow 1 miss out of 10+
            "missing": missing,
            "present": present,
            "coverage": round(coverage, 3),
            "domain": primary,
        }

    # ── Stage 5: Refinement ──

    @classmethod
    def build_refinement_prompt(cls, original_response: str, quality_result: dict, task: str) -> str:
        """
        Build a targeted refinement prompt to fill gaps.

        Only called when the quality gate detects missing concepts.
        """
        missing = quality_result["missing"]
        domain = quality_result["domain"]

        refinement = (
            f"Your previous response was good but incomplete for {domain} standards.\n\n"
            f"ORIGINAL TASK: {task[:500]}\n\n"
            f"YOUR PREVIOUS RESPONSE (abbreviated):\n{original_response[:2000]}\n\n"
            f"MISSING PROFESSIONAL CONCEPTS that MUST be covered:\n"
        )

        for concept in missing:
            refinement += f"  • You must mention or use: {concept}\n"

        refinement += (
            "\nENHANCE your response to include ALL missing concepts.\n"
            "Keep your existing code but add the missing elements.\n"
            "Output the COMPLETE enhanced response."
        )

        return refinement


# Backward compatibility alias
PromptAdapter = PromptEngine


# ═══════════════════════════════════════════
# ABSTRACT BASE CLASS
# ═══════════════════════════════════════════

class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All Saturday engine integrations go through this interface,
    ensuring the system is model-agnostic and can switch providers
    without touching any engine code.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self.total_tokens_used = 0
        self.total_requests = 0
        self.total_errors = 0

    @abstractmethod
    def _call_api(
        self,
        messages: list[LLMMessage],
        max_tokens: int,
        temperature: float,
        stop: Optional[list[str]] = None,
    ) -> LLMResponse:
        """Make the actual API call. Implemented by each provider."""
        ...

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[list[str]] = None,
    ) -> LLMResponse:
        """
        Generate a single response from a prompt.

        Args:
            prompt: User prompt
            system: System prompt (defaults to Saturday system prompt)
            max_tokens: Max tokens to generate
            temperature: Sampling temperature (0=deterministic, 1=creative)
            stop: Stop sequences

        Returns:
            LLMResponse with generated content
        """
        messages = []
        if system:
            messages.append(LLMMessage(role="system", content=system))
        messages.append(LLMMessage(role="user", content=prompt))

        return self._call_with_retry(
            messages,
            max_tokens=max_tokens or self.config.default_max_tokens,
            temperature=temperature if temperature is not None else self.config.default_temperature,
            stop=stop,
        )

    def chat(
        self,
        messages: list[LLMMessage],
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """
        Multi-turn conversation.

        Args:
            messages: List of conversation messages
            system: System prompt (prepended if provided)
            max_tokens: Max tokens to generate
            temperature: Sampling temperature

        Returns:
            LLMResponse with assistant's reply
        """
        all_messages = []
        if system:
            all_messages.append(LLMMessage(role="system", content=system))
        all_messages.extend(messages)

        return self._call_with_retry(
            all_messages,
            max_tokens=max_tokens or self.config.default_max_tokens,
            temperature=temperature if temperature is not None else self.config.default_temperature,
        )

    def generate_code(
        self,
        task: str,
        language: str = "python",
        context: str = "",
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """
        Generate production-ready code using the multi-stage prompt pipeline.

        Pipeline stages:
            1. Classify → detect task domain (security, debug, algorithm, etc.)
            2. Adapt   → inject domain-specific professional protocol
            3. Generate → call LLM with adapted prompt
            4. Quality Gate → check response covers required concepts
            5. Refine  → if gaps found, send targeted follow-up to LLM

        Args:
            task: What the code should do
            language: Programming language
            context: Additional context (project structure, conventions)
            max_tokens: Max tokens

        Returns:
            LLMResponse with generated code
        """
        # Stage 1+2: Classify and adapt
        domains = PromptEngine.classify(task)
        prompt = PromptEngine.adapt_prompt(task=task, language=language, context=context)
        log.info(f"PromptEngine: domains={domains}")

        # Stage 3: Generate
        response = self.generate(
            prompt=prompt,
            system=SATURDAY_SYSTEM_PROMPT,
            max_tokens=max_tokens,
            temperature=0.0,  # Deterministic for production
        )

        # Stage 4: Quality gate
        quality = PromptEngine.check_quality(response.content, task)
        log.info(
            f"Quality gate: coverage={quality['coverage']}, "
            f"missing={quality['missing']}"
        )

        # Stage 5: Refine if quality gate found gaps
        if not quality["passed"] and quality["missing"]:
            log.info(f"Refinement needed: {len(quality['missing'])} missing concepts")
            refinement_prompt = PromptEngine.build_refinement_prompt(
                original_response=response.content,
                quality_result=quality,
                task=task,
            )

            refined = self.generate(
                prompt=refinement_prompt,
                system=SATURDAY_SYSTEM_PROMPT,
                max_tokens=max_tokens,
                temperature=0.0,
            )

            # Merge: use refined response (it contains the enhanced version)
            return LLMResponse(
                content=refined.content,
                tokens_used=response.tokens_used + refined.tokens_used,
                latency_seconds=response.latency_seconds + refined.latency_seconds,
                model=response.model,
            )

        return response

    def _call_with_retry(
        self,
        messages: list[LLMMessage],
        max_tokens: int,
        temperature: float,
        stop: Optional[list[str]] = None,
    ) -> LLMResponse:
        """Call the API with exponential backoff retry."""
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                start = time.time()
                response = self._call_api(messages, max_tokens, temperature, stop)
                response.latency_seconds = round(time.time() - start, 3)

                self.total_tokens_used += response.tokens_used
                self.total_requests += 1

                return response

            except Exception as e:
                last_error = e
                self.total_errors += 1
                wait = min(2 ** attempt * 1.0, 30.0)

                if attempt < self.config.max_retries - 1:
                    log.warning(
                        f"LLM call failed (attempt {attempt + 1}/{self.config.max_retries}): {e}. "
                        f"Retrying in {wait:.0f}s..."
                    )
                    time.sleep(wait)
                else:
                    log.error(f"LLM call failed after {self.config.max_retries} attempts: {e}")

        raise ConnectionError(
            f"Failed to get LLM response after {self.config.max_retries} attempts: {last_error}"
        )

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token estimate (1 token ~ 3 chars for code)."""
        return max(1, len(text) // 3)

    @staticmethod
    def extract_code(response_text: str, language: str = "") -> Optional[str]:
        """Extract code block from an LLM response."""
        if language:
            pattern = rf"```{re.escape(language)}\s*\n(.*?)```"
            match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()

        pattern = r"```\w*\s*\n(.*?)```"
        matches = re.findall(pattern, response_text, re.DOTALL)
        if matches:
            return max(matches, key=len).strip()

        lines = response_text.strip().split("\n")
        code_indicators = ["import ", "def ", "class ", "from ", "const ", "function ", "pub fn "]
        if lines and any(lines[0].startswith(ind) for ind in code_indicators):
            return response_text.strip()

        return None

    def get_stats(self) -> dict:
        """Return usage statistics."""
        return {
            "total_requests": self.total_requests,
            "total_tokens_used": self.total_tokens_used,
            "total_errors": self.total_errors,
            "model": self.config.model,
            "provider": self.config.provider,
        }

    @classmethod
    def from_env(cls) -> "LLMProvider":
        """Create the appropriate provider from environment variables."""
        config = LLMConfig.from_env()

        if config.provider == "anthropic":
            return AnthropicProvider(config)
        elif config.provider == "huggingface":
            return HuggingFaceLocalProvider(config)
        else:
            return OpenAICompatibleProvider(config)


# ═══════════════════════════════════════════
# OPENAI-COMPATIBLE PROVIDER
# ═══════════════════════════════════════════

class OpenAICompatibleProvider(LLMProvider):
    """
    Provider for any OpenAI-compatible API.

    Works with: OpenAI, Kimi K2.5, vLLM, Qwen, DeepSeek,
    Together AI, Fireworks AI, Azure OpenAI, and more.
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        if not config.api_base:
            config.api_base = "https://api.openai.com/v1"

    def _call_api(
        self,
        messages: list[LLMMessage],
        max_tokens: int,
        temperature: float,
        stop: Optional[list[str]] = None,
    ) -> LLMResponse:
        import urllib.request
        import urllib.error

        url = f"{self.config.api_base.rstrip('/')}/chat/completions"

        payload = {
            "model": self.config.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if stop:
            payload["stop"] = stop

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }

        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise ConnectionError(f"HTTP {e.code}: {body[:500]}") from e
        except urllib.error.URLError as e:
            raise ConnectionError(f"Connection failed: {e.reason}") from e

        choice = result.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage = result.get("usage", {})

        return LLMResponse(
            content=message.get("content", ""),
            model=result.get("model", self.config.model),
            tokens_used=usage.get("total_tokens", 0),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            finish_reason=choice.get("finish_reason", ""),
            raw=result,
        )


# ═══════════════════════════════════════════
# ANTHROPIC PROVIDER
# ═══════════════════════════════════════════

class AnthropicProvider(LLMProvider):
    """
    Provider for Anthropic's Claude API.
    Supports Claude Opus 4.6, Sonnet 4.6, Haiku, etc.
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        if not config.api_base:
            config.api_base = "https://api.anthropic.com"
        if not config.model:
            config.model = "claude-sonnet-4-20250514"

    def _call_api(
        self,
        messages: list[LLMMessage],
        max_tokens: int,
        temperature: float,
        stop: Optional[list[str]] = None,
    ) -> LLMResponse:
        import urllib.request
        import urllib.error

        url = f"{self.config.api_base.rstrip('/')}/v1/messages"

        system_text = ""
        user_messages = []
        for m in messages:
            if m.role == "system":
                system_text += m.content + "\n"
            else:
                user_messages.append({"role": m.role, "content": m.content})

        if not user_messages or user_messages[0]["role"] != "user":
            user_messages.insert(0, {"role": "user", "content": "Begin."})

        payload: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": user_messages,
        }
        if system_text.strip():
            payload["system"] = system_text.strip()
        if stop:
            payload["stop_sequences"] = stop

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
        }

        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise ConnectionError(f"HTTP {e.code}: {body[:500]}") from e
        except urllib.error.URLError as e:
            raise ConnectionError(f"Connection failed: {e.reason}") from e

        content_blocks = result.get("content", [])
        content = ""
        for block in content_blocks:
            if block.get("type") == "text":
                content += block.get("text", "")

        usage = result.get("usage", {})

        return LLMResponse(
            content=content,
            model=result.get("model", self.config.model),
            tokens_used=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            finish_reason=result.get("stop_reason", ""),
            raw=result,
        )


# ═══════════════════════════════════════════
# HUGGINGFACE LOCAL PROVIDER
# ═══════════════════════════════════════════

class HuggingFaceLocalProvider(LLMProvider):
    """
    Provider for local inference using HuggingFace Transformers.
    Use this with the fine-tuned Saturday MK1 model or any local model.
    Requires: torch, transformers (optional: bitsandbytes for quantization).
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._model = None
        self._tokenizer = None
        if not config.model:
            config.model = "./saturday_model/final"

    def _ensure_loaded(self):
        """Lazy-load the model on first use."""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "Local inference requires: pip install torch transformers\n"
                "For quantized models: pip install bitsandbytes"
            )

        log.info(f"Loading local model: {self.config.model}")
        start = time.time()

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model, trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        load_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "device_map": "auto",
        }

        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            log.info("Using 4-bit quantization")
        except ImportError:
            load_kwargs["torch_dtype"] = torch.bfloat16
            log.info("Using bfloat16 (no quantization)")

        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model, **load_kwargs,
        )

        elapsed = time.time() - start
        log.info(f"Model loaded in {elapsed:.1f}s")

    def _call_api(
        self,
        messages: list[LLMMessage],
        max_tokens: int,
        temperature: float,
        stop: Optional[list[str]] = None,
    ) -> LLMResponse:
        import torch

        self._ensure_loaded()

        # Format as ChatML
        im_start = "<|im_start|>"
        im_end = "<|im_end|>"
        chat_text = ""
        for m in messages:
            chat_text += f"{im_start}{m.role}\n{m.content}{im_end}\n"
        chat_text += f"{im_start}assistant\n"

        inputs = self._tokenizer(chat_text, return_tensors="pt").to(self._model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                do_sample=temperature > 0,
                top_p=0.95,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        generated = outputs[0][input_len:]
        content = self._tokenizer.decode(generated, skip_special_tokens=True)

        # Clean up ChatML end tokens from output
        content = content.split(im_end)[0].strip()

        return LLMResponse(
            content=content,
            model=self.config.model,
            tokens_used=input_len + len(generated),
            prompt_tokens=input_len,
            completion_tokens=len(generated),
            finish_reason="stop",
            raw={},
        )
