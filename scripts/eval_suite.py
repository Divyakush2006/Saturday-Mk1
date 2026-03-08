import os as _os; _os.environ.setdefault("PYTHONUTF8", "1")  # noqa: E702 — Windows UTF-8 fix
"""
Saturday MK1 Evaluation Suite — Benchmark Against Claude Opus 4.6
================================================================
Automated benchmark runner that measures Saturday MK1's fine-tuned model
against Claude Opus 4.6 baselines across multiple coding dimensions.

Usage:
    # Run all benchmarks (requires model + API keys)
    python scripts/eval_suite.py --model-path ./saturday_model/final --provider anthropic

    # Run specific benchmark
    python scripts/eval_suite.py --model-path ./saturday_model/final --benchmarks humaneval enterprise

    # Dry run (test evaluation framework without model)
    python scripts/eval_suite.py --dry-run

    # Compare two models
    python scripts/eval_suite.py --compare results/saturday_mk1_v1.json results/opus_baseline.json
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Fix Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("saturday-eval")


# ═══════════════════════════════════════════
# DATA TYPES
# ═══════════════════════════════════════════

@dataclass
class EvalTask:
    """A single evaluation task."""
    task_id: str
    benchmark: str
    prompt: str
    expected: Optional[str] = None
    test_code: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalResult:
    """Result of evaluating a single task."""
    task_id: str
    benchmark: str
    passed: bool
    score: float  # 0-1
    model_output: str
    latency_seconds: float
    error: Optional[str] = None
    details: dict = field(default_factory=dict)


@dataclass
class BenchmarkSummary:
    """Aggregate results for a benchmark."""
    name: str
    total_tasks: int
    passed: int
    failed: int
    pass_rate: float
    avg_score: float
    avg_latency: float
    results: list[EvalResult] = field(default_factory=list)


# ═══════════════════════════════════════════
# CLAUDE OPUS 4.6 BASELINES
# ═══════════════════════════════════════════

OPUS_BASELINES = {
    "humaneval": {
        "pass_rate": 0.92,
        "description": "HumanEval code generation (Python function completion)",
    },
    "enterprise": {
        "pass_rate": 0.85,
        "description": "Enterprise architecture patterns and design",
    },
    "debug": {
        "pass_rate": 0.88,
        "description": "Bug detection and fixing from stack traces",
    },
    "security": {
        "pass_rate": 0.90,
        "description": "Vulnerability detection and secure code generation",
    },
    "multifile": {
        "pass_rate": 0.78,
        "description": "Multi-file reasoning and refactoring",
    },
    "swe_bench_lite": {
        "pass_rate": 0.808,
        "description": "SWE-bench Verified (real GitHub issue resolution)",
    },
}


# ═══════════════════════════════════════════
# BENCHMARK BASE CLASS
# ═══════════════════════════════════════════

class Benchmark(ABC):
    """Base class for all benchmarks."""

    name: str = "base"
    description: str = ""

    @abstractmethod
    def load_tasks(self) -> list[EvalTask]:
        """Load evaluation tasks."""
        ...

    @abstractmethod
    def evaluate_response(self, task: EvalTask, response: str) -> EvalResult:
        """Evaluate a model's response to a task."""
        ...


# ═══════════════════════════════════════════
# HUMANEVAL BENCHMARK
# ═══════════════════════════════════════════

class HumanEvalBenchmark(Benchmark):
    """
    HumanEval-style code generation benchmark.
    Tests: Write a Python function given a docstring + signature.
    Evaluation: Execute function against test cases.
    """

    name = "humaneval"
    description = "Code generation accuracy — Python function completion"

    # Representative subset of HumanEval-style problems
    PROBLEMS = [
        {
            "task_id": "HE_001",
            "prompt": "def has_close_elements(numbers: list[float], threshold: float) -> bool:\n    \"\"\"Check if in given list of numbers, are any two numbers closer to each other than given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"",
            "test": "assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\nassert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\nassert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\nassert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\nassert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\nassert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\nassert has_close_elements([], 1.0) == False\nassert has_close_elements([1.0], 1.0) == False",
        },
        {
            "task_id": "HE_002",
            "prompt": "def separate_paren_groups(paren_string: str) -> list[str]:\n    \"\"\"Input to this function is a string containing multiple groups of nested parentheses.\n    Your goal is to separate those group into separate strings and return the list of those.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other.\n    Ignore any spaces in the input string.\n    >>> separate_paren_groups('( ) (( )) (( )( ))')\n    ['()', '(())', '(()())']\n    \"\"\"",
            "test": "assert separate_paren_groups('( ) (( )) (( )( ))') == ['()', '(())', '(()())']\nassert separate_paren_groups('() (()) ((())())') == ['()', '(())', '((())())']\nassert separate_paren_groups('(()(()))') == ['(()(()))']\nassert separate_paren_groups('( ) ( )') == ['()', '()']",
        },
        {
            "task_id": "HE_003",
            "prompt": "def truncate_number(number: float) -> float:\n    \"\"\"Given a positive floating point number, it can be decomposed into an integer part (largest integer smaller than given number) and decimals (leftover part always smaller than 1).\n    Return the decimal part of the number.\n    >>> truncate_number(3.5)\n    0.5\n    \"\"\"",
            "test": "assert truncate_number(3.5) == 0.5\nassert abs(truncate_number(1.33) - 0.33) < 1e-6\nassert abs(truncate_number(123.456) - 0.456) < 1e-6",
        },
        {
            "task_id": "HE_004",
            "prompt": "def below_zero(operations: list[int]) -> bool:\n    \"\"\"You're given a list of deposit and withdrawal operations on a bank account that starts with zero balance. Your task is to detect if at any point the balance falls below zero, and at that point the function should return True. Otherwise it should return False.\n    >>> below_zero([1, 2, 3])\n    False\n    >>> below_zero([1, 2, -4, 5])\n    True\n    \"\"\"",
            "test": "assert below_zero([1, 2, 3]) == False\nassert below_zero([1, 2, -4, 5]) == True\nassert below_zero([1, 2, -3, 1, 2, -3]) == False\nassert below_zero([1, 2, -4, 5, 6]) == True\nassert below_zero([]) == False",
        },
        {
            "task_id": "HE_005",
            "prompt": "def mean_absolute_deviation(numbers: list[float]) -> float:\n    \"\"\"For a given list of input numbers, calculate Mean Absolute Deviation around the mean of this dataset.\n    Mean Absolute Deviation is the average absolute difference between each element and a centerpoint (mean in this case):\n    MAD = average | x - x_mean |\n    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])\n    1.0\n    \"\"\"",
            "test": "assert abs(mean_absolute_deviation([1.0, 2.0, 3.0, 4.0]) - 1.0) < 1e-6\nassert abs(mean_absolute_deviation([1.0, 2.0, 3.0]) - 2/3) < 1e-6\nassert abs(mean_absolute_deviation([10.0]) - 0.0) < 1e-6",
        },
        {
            "task_id": "HE_006",
            "prompt": "def intersperse(numbers: list[int], delimeter: int) -> list[int]:\n    \"\"\"Insert a number 'delimeter' between every two consecutive elements of input list 'numbers'\n    >>> intersperse([], 4)\n    []\n    >>> intersperse([1, 2, 3], 4)\n    [1, 4, 2, 4, 3]\n    \"\"\"",
            "test": "assert intersperse([], 4) == []\nassert intersperse([1, 2, 3], 4) == [1, 4, 2, 4, 3]\nassert intersperse([1], 4) == [1]\nassert intersperse([5, 6, 3, 2], 8) == [5, 8, 6, 8, 3, 8, 2]",
        },
        {
            "task_id": "HE_007",
            "prompt": "def parse_nested_parens(paren_string: str) -> list[int]:\n    \"\"\"Input to this function is a string represented multiple groups of nested parentheses separated by spaces. For each of the group, output the deepest level of nesting of parentheses.\n    E.g. (()()) has maximum two levels of nesting while ((())) has three.\n    >>> parse_nested_parens('(()()) ((())) () ((())()())')\n    [2, 3, 1, 3]\n    \"\"\"",
            "test": "assert parse_nested_parens('(()()) ((())) () ((())()())') == [2, 3, 1, 3]\nassert parse_nested_parens('() (())') == [1, 2]\nassert parse_nested_parens('((()))') == [3]",
        },
        {
            "task_id": "HE_008",
            "prompt": "def filter_by_substring(strings: list[str], substring: str) -> list[str]:\n    \"\"\"Filter an input list of strings only for ones that contain given substring\n    >>> filter_by_substring([], 'a')\n    []\n    >>> filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'a')\n    ['abc', 'bacd', 'array']\n    \"\"\"",
            "test": "assert filter_by_substring([], 'a') == []\nassert filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'a') == ['abc', 'bacd', 'array']\nassert filter_by_substring(['xxx', 'asd', 'xoy'], 'o') == ['xoy']",
        },
        {
            "task_id": "HE_009",
            "prompt": "def sum_product(numbers: list[int]) -> tuple[int, int]:\n    \"\"\"For a given list of integers, return a tuple consisting of a sum and a product of all the integers in a list.\n    Empty sum should be equal to 0 and empty product should be equal to 1.\n    >>> sum_product([])\n    (0, 1)\n    >>> sum_product([1, 2, 3, 4])\n    (10, 24)\n    \"\"\"",
            "test": "assert sum_product([]) == (0, 1)\nassert sum_product([1, 2, 3, 4]) == (10, 24)\nassert sum_product([1, 1, 1]) == (3, 1)\nassert sum_product([100]) == (100, 100)",
        },
        {
            "task_id": "HE_010",
            "prompt": "def rolling_max(numbers: list[int]) -> list[int]:\n    \"\"\"From a given list of integers, generate a list of rolling maximum element found until given moment in the sequence.\n    >>> rolling_max([1, 2, 3, 2, 3, 4, 2])\n    [1, 2, 3, 3, 3, 4, 4]\n    \"\"\"",
            "test": "assert rolling_max([1, 2, 3, 2, 3, 4, 2]) == [1, 2, 3, 3, 3, 4, 4]\nassert rolling_max([]) == []\nassert rolling_max([5]) == [5]\nassert rolling_max([3, 2, 1]) == [3, 3, 3]",
        },
    ]

    def load_tasks(self) -> list[EvalTask]:
        tasks = []
        for p in self.PROBLEMS:
            tasks.append(
                EvalTask(
                    task_id=p["task_id"],
                    benchmark=self.name,
                    prompt=f"Complete this Python function. Only provide the function body (no imports or class needed unless they're part of the function). Do NOT include the function signature — I will prepend it.\n\n{p['prompt']}",
                    test_code=p["test"],
                )
            )
        return tasks

    def evaluate_response(self, task: EvalTask, response: str) -> EvalResult:
        """Evaluate by extracting code and running tests."""
        start = time.time()

        # Extract function body from response
        code = self._extract_code(task.prompt, response)

        if not code:
            return EvalResult(
                task_id=task.task_id,
                benchmark=self.name,
                passed=False,
                score=0.0,
                model_output=response,
                latency_seconds=time.time() - start,
                error="Could not extract code from response",
            )

        # Execute tests
        passed, error = self._run_tests(code, task.test_code)

        return EvalResult(
            task_id=task.task_id,
            benchmark=self.name,
            passed=passed,
            score=1.0 if passed else 0.0,
            model_output=response,
            latency_seconds=time.time() - start,
            error=error,
        )

    def _extract_code(self, prompt: str, response: str) -> Optional[str]:
        """Extract Python code from model response."""
        # Try to find code blocks
        code_blocks = re.findall(r"```(?:python)?\n(.*?)```", response, re.DOTALL)

        if code_blocks:
            # Find the block that looks like a function implementation
            for block in code_blocks:
                if "def " in block or "return" in block:
                    return block

            # Fall back to first code block
            return code_blocks[0]

        # Try to find inline function definition
        func_match = re.search(r"(def \w+.*?)(?:\n\n|\Z)", response, re.DOTALL)
        if func_match:
            return func_match.group(1)

        # If response looks like just the function body
        lines = response.strip().split("\n")
        code_lines = [l for l in lines if not l.strip().startswith("#") and l.strip()]
        if code_lines:
            # Prepend function signature from prompt
            sig_match = re.search(r"(def \w+\(.*?\).*?:)", prompt)
            if sig_match:
                sig = sig_match.group(1)
                body = "\n".join("    " + l if not l.startswith("    ") else l for l in code_lines)
                return f"{sig}\n{body}"

        return None

    def _run_tests(self, code: str, test_code: str) -> tuple[bool, Optional[str]]:
        """Execute code + test assertions in a subprocess."""
        full_code = f"{code}\n\n{test_code}\nprint('ALL_TESTS_PASSED')"

        try:
            result = subprocess.run(
                [sys.executable, "-c", full_code],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if "ALL_TESTS_PASSED" in result.stdout:
                return True, None
            else:
                error = result.stderr.strip() or result.stdout.strip()
                return False, error[:500]

        except subprocess.TimeoutExpired:
            return False, "Execution timed out (10s)"
        except Exception as e:
            return False, str(e)[:500]


# ═══════════════════════════════════════════
# ENTERPRISE BENCHMARK
# ═══════════════════════════════════════════

class EnterpriseBenchmark(Benchmark):
    """
    Enterprise coding patterns benchmark.
    Tests: Architecture, design patterns, production-readiness.
    Evaluation: Rubric-based (code presence, patterns, error handling).
    """

    name = "enterprise"
    description = "Enterprise architecture patterns and production code quality"

    TASKS = [
        {
            "task_id": "ENT_001",
            "prompt": "Implement a connection pool manager in Python with: max connections, health checks, connection reuse, timeout handling, and thread safety. Use no external libraries.",
            "rubric": ["class", "pool", "lock", "timeout", "health", "try", "except"],
        },
        {
            "task_id": "ENT_002",
            "prompt": "Implement the Repository pattern with Unit of Work for a Python ORM-agnostic data access layer. Support: CRUD operations, bulk operations, transaction management, query builder, and caching.",
            "rubric": ["Repository", "UnitOfWork", "commit", "rollback", "cache", "query", "ABC"],
        },
        {
            "task_id": "ENT_003",
            "prompt": "Build a circuit breaker implementation in Python with: closed/open/half-open states, configurable thresholds, automatic recovery, fallback support, and metrics collection.",
            "rubric": ["CircuitBreaker", "OPEN", "CLOSED", "HALF_OPEN", "threshold", "fallback", "timeout"],
        },
        {
            "task_id": "ENT_004",
            "prompt": "Implement an event-driven message bus for a Python microservices application. Support: publish/subscribe, message routing, dead letter queue, retry with backoff, and message serialization.",
            "rubric": ["publish", "subscribe", "handler", "retry", "dead_letter", "serialize", "async"],
        },
        {
            "task_id": "ENT_005",
            "prompt": "Build a feature flag system in Python with: boolean and percentage-based flags, user segmentation, A/B testing support, flag inheritance, and an in-memory evaluation engine with <5ms latency.",
            "rubric": ["FeatureFlag", "evaluate", "percentage", "segment", "override", "cache", "user"],
        },
    ]

    def load_tasks(self) -> list[EvalTask]:
        return [
            EvalTask(
                task_id=t["task_id"],
                benchmark=self.name,
                prompt=t["prompt"],
                metadata={"rubric": t["rubric"]},
            )
            for t in self.TASKS
        ]

    def evaluate_response(self, task: EvalTask, response: str) -> EvalResult:
        """Evaluate using rubric-based scoring."""
        rubric = task.metadata.get("rubric", [])
        code_blocks = re.findall(r"```[\w]*\n(.*?)```", response, re.DOTALL)
        all_code = "\n".join(code_blocks).lower()
        all_text = response.lower()

        # Check rubric items
        hits = sum(1 for keyword in rubric if keyword.lower() in all_code or keyword.lower() in all_text)
        rubric_score = hits / len(rubric) if rubric else 0

        # Code quality checks
        quality_checks = {
            "has_code": len(code_blocks) > 0,
            "has_docstrings": '"""' in response or "'''" in response,
            "has_type_hints": any(h in response for h in ["-> ", ": str", ": int", ": list", "Optional["]),
            "has_error_handling": any(e in response for e in ["try:", "except", "raise "]),
            "has_logging": "logging" in response or "logger" in response or "log." in response,
            "sufficient_length": len(response) > 500,
        }
        quality_score = sum(quality_checks.values()) / len(quality_checks)

        # Combined score
        score = rubric_score * 0.6 + quality_score * 0.4
        passed = score >= 0.6

        return EvalResult(
            task_id=task.task_id,
            benchmark=self.name,
            passed=passed,
            score=round(score, 3),
            model_output=response,
            latency_seconds=0,
            details={"rubric_score": rubric_score, "quality_score": quality_score, **quality_checks},
        )


# ═══════════════════════════════════════════
# SECURITY BENCHMARK
# ═══════════════════════════════════════════

class SecurityBenchmark(Benchmark):
    """
    Security vulnerability detection benchmark.
    Tests: Finding and fixing security issues in code.
    Evaluation: Checks if specific vulnerabilities are identified.
    """

    name = "security"
    description = "Vulnerability detection and secure code generation"

    TASKS = [
        {
            "task_id": "SEC_001",
            "prompt": "Find ALL security vulnerabilities in this Flask code and provide fixes:\n\n```python\n@app.route('/api/user/<user_id>')\ndef get_user(user_id):\n    query = f\"SELECT * FROM users WHERE id = {user_id}\"\n    user = db.execute(query).fetchone()\n    return jsonify(dict(user))\n\n@app.route('/api/upload', methods=['POST'])\ndef upload():\n    f = request.files['file']\n    f.save(os.path.join('/uploads', f.filename))\n    return 'OK'\n\n@app.route('/api/run', methods=['POST'])\ndef run_cmd():\n    cmd = request.json['command']\n    result = os.popen(cmd).read()\n    return result\n```",
            "vulns": ["sql_injection", "path_traversal", "command_injection", "no_auth", "no_input_validation"],
        },
        {
            "task_id": "SEC_002",
            "prompt": "Audit this authentication code for security issues:\n\n```python\nimport hashlib\n\ndef login(username, password):\n    stored_hash = db.get_password_hash(username)\n    input_hash = hashlib.md5(password.encode()).hexdigest()\n    if input_hash == stored_hash:\n        token = hashlib.sha1(f\"{username}:{time.time()}\".encode()).hexdigest()\n        return {'token': token}\n    return {'error': 'Invalid credentials'}\n\ndef verify_token(token):\n    return db.token_exists(token)\n```",
            "vulns": ["md5_hashing", "no_salt", "weak_token", "timing_attack", "no_rate_limit", "no_expiry"],
        },
        {
            "task_id": "SEC_003",
            "prompt": "Find and fix the security issues in this Django template and view:\n\n```python\n# views.py\ndef search(request):\n    query = request.GET.get('q', '')\n    results = Product.objects.raw(f\"SELECT * FROM products WHERE name LIKE '%{query}%'\")\n    return render(request, 'search.html', {'query': query, 'results': results})\n```\n\n```html\n<!-- search.html -->\n<h1>Results for: {{ query|safe }}</h1>\n{% for p in results %}\n<div>{{ p.name|safe }} - ${{ p.price }}</div>\n{% endfor %}\n```",
            "vulns": ["sql_injection", "xss", "safe_filter_misuse", "no_csrf", "raw_query"],
        },
        {
            "task_id": "SEC_004",
            "prompt": "Review this JWT implementation for security flaws:\n\n```python\nimport jwt\nimport base64\n\nSECRET = 'my-secret-key-123'\n\ndef create_token(user_id, role):\n    payload = {'user_id': user_id, 'role': role}\n    return jwt.encode(payload, SECRET, algorithm='HS256')\n\ndef verify_token(token):\n    try:\n        header = json.loads(base64.b64decode(token.split('.')[0] + '=='))\n        algo = header.get('alg', 'HS256')\n        return jwt.decode(token, SECRET, algorithms=[algo])\n    except:\n        return None\n\ndef is_admin(token):\n    data = verify_token(token)\n    return data and data.get('role') == 'admin'\n```",
            "vulns": ["weak_secret", "algorithm_confusion", "no_expiry", "no_issuer", "bare_except", "none_algorithm"],
        },
        {
            "task_id": "SEC_005",
            "prompt": "Identify all SSRF, IDOR, and access control vulnerabilities in this API:\n\n```python\n@app.route('/api/proxy')\ndef proxy():\n    url = request.args.get('url')\n    response = requests.get(url)\n    return response.text\n\n@app.route('/api/document/<doc_id>')\ndef get_document(doc_id):\n    doc = Document.query.get(doc_id)\n    return jsonify(doc.to_dict())\n\n@app.route('/api/document/<doc_id>/delete', methods=['POST'])\ndef delete_document(doc_id):\n    doc = Document.query.get(doc_id)\n    db.session.delete(doc)\n    db.session.commit()\n    return 'Deleted'\n```",
            "vulns": ["ssrf", "idor", "no_auth", "no_ownership_check", "no_url_validation", "no_csrf"],
        },
    ]

    def load_tasks(self) -> list[EvalTask]:
        return [
            EvalTask(
                task_id=t["task_id"],
                benchmark=self.name,
                prompt=t["prompt"],
                metadata={"vulns": t["vulns"]},
            )
            for t in self.TASKS
        ]

    def evaluate_response(self, task: EvalTask, response: str) -> EvalResult:
        """Check if vulnerabilities are identified in the response."""
        expected_vulns = task.metadata.get("vulns", [])
        response_lower = response.lower()

        # Map vulnerability keywords to common phrases the model might use
        vuln_phrases = {
            "sql_injection": ["sql injection", "sqli", "parameterized", "prepared statement", "sql inject"],
            "xss": ["xss", "cross-site scripting", "cross site scripting", "script injection", "html escaping"],
            "path_traversal": ["path traversal", "directory traversal", "../" , "path injection", "secure_filename"],
            "command_injection": ["command injection", "os.popen", "subprocess", "shell injection", "os command"],
            "ssrf": ["ssrf", "server-side request forgery", "server side request", "url validation", "internal network"],
            "idor": ["idor", "insecure direct object", "authorization check", "ownership", "access control"],
            "no_auth": ["authentication", "unauthorized", "login required", "auth check"],
            "no_input_validation": ["input validation", "sanitiz", "validat"],
            "md5_hashing": ["md5", "weak hash", "bcrypt", "argon2", "pbkdf2", "scrypt"],
            "no_salt": ["salt", "without salt", "random salt"],
            "weak_token": ["weak token", "predictable", "token generation", "cryptograph", "secrets.token"],
            "timing_attack": ["timing attack", "constant time", "hmac.compare_digest", "time-safe"],
            "no_rate_limit": ["rate limit", "brute force", "throttl"],
            "no_expiry": ["expir", "expiry", "exp", "token lifetime", "ttl"],
            "safe_filter_misuse": ["|safe", "autoescape", "mark_safe", "xss"],
            "no_csrf": ["csrf", "cross-site request forgery"],
            "raw_query": ["raw query", "raw sql", "orm", "parameterized"],
            "weak_secret": ["weak secret", "hardcoded", "secret key", "environment variable", "strong key"],
            "algorithm_confusion": ["algorithm", "alg", "none", "algorithm confusion", "accepted algorithm"],
            "none_algorithm": ["none algorithm", "alg: none", "algorithm none"],
            "bare_except": ["bare except", "catch all", "specific exception"],
            "no_issuer": ["issuer", "iss", "audience", "aud"],
            "no_ownership_check": ["ownership", "belongs to", "user check", "authorization"],
            "no_url_validation": ["url validation", "allowlist", "whitelist", "internal", "localhost"],
        }

        detected = 0
        for vuln in expected_vulns:
            phrases = vuln_phrases.get(vuln, [vuln.replace("_", " ")])
            if any(phrase in response_lower for phrase in phrases):
                detected += 1

        score = detected / len(expected_vulns) if expected_vulns else 0
        passed = score >= 0.6

        return EvalResult(
            task_id=task.task_id,
            benchmark=self.name,
            passed=passed,
            score=round(score, 3),
            model_output=response,
            latency_seconds=0,
            details={
                "vulns_expected": len(expected_vulns),
                "vulns_detected": detected,
                "detection_rate": round(score * 100, 1),
            },
        )


# ═══════════════════════════════════════════
# DEBUG BENCHMARK
# ═══════════════════════════════════════════

class DebugBenchmark(Benchmark):
    """
    Debugging and bug-fixing benchmark.
    Tests: Identify root cause from stack traces and buggy code.
    """

    name = "debug"
    description = "Bug detection and root cause analysis"

    TASKS = [
        {
            "task_id": "DBG_001",
            "prompt": "This code has a subtle off-by-one error. Find it and explain the fix:\n\n```python\ndef binary_search(arr, target):\n    left, right = 0, len(arr)\n    while left < right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid\n        else:\n            right = mid\n    return -1\n```",
            "keywords": ["infinite loop", "left = mid + 1", "off-by-one", "mid"],
        },
        {
            "task_id": "DBG_002",
            "prompt": "Why does this code sometimes produce incorrect results for concurrent requests?\n\n```python\nclass Counter:\n    _instance = None\n    count = 0\n    \n    @classmethod\n    def get_instance(cls):\n        if cls._instance is None:\n            cls._instance = cls()\n        return cls._instance\n    \n    def increment(self):\n        current = self.count\n        time.sleep(0.001)  # simulate work\n        self.count = current + 1\n        return self.count\n```",
            "keywords": ["race condition", "thread", "lock", "atomic", "not thread-safe", "concurrent"],
        },
        {
            "task_id": "DBG_003",
            "prompt": "This recursive function causes a stack overflow for large inputs. Fix it while maintaining the same functionality:\n\n```python\ndef flatten(nested_list):\n    result = []\n    for item in nested_list:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result\n```\n\nInput: A list nested 10,000 levels deep.",
            "keywords": ["iterative", "stack", "recursion limit", "deque", "while", "append"],
        },
    ]

    def load_tasks(self) -> list[EvalTask]:
        return [
            EvalTask(
                task_id=t["task_id"],
                benchmark=self.name,
                prompt=t["prompt"],
                metadata={"keywords": t["keywords"]},
            )
            for t in self.TASKS
        ]

    def evaluate_response(self, task: EvalTask, response: str) -> EvalResult:
        keywords = task.metadata.get("keywords", [])
        response_lower = response.lower()
        hits = sum(1 for kw in keywords if kw.lower() in response_lower)
        score = hits / len(keywords) if keywords else 0

        has_code_fix = bool(re.findall(r"```[\w]*\n.*?```", response, re.DOTALL))
        if has_code_fix:
            score = min(1.0, score + 0.2)

        passed = score >= 0.5

        return EvalResult(
            task_id=task.task_id,
            benchmark=self.name,
            passed=passed,
            score=round(score, 3),
            model_output=response,
            latency_seconds=0,
            details={"keywords_found": hits, "total_keywords": len(keywords), "has_code_fix": has_code_fix},
        )


# ═══════════════════════════════════════════
# EVALUATION RUNNER
# ═══════════════════════════════════════════

class EvalRunner:
    """
    Runs benchmarks against a model and compares to Opus 4.6 baselines.
    """

    BENCHMARKS = {
        "humaneval": HumanEvalBenchmark,
        "enterprise": EnterpriseBenchmark,
        "security": SecurityBenchmark,
        "debug": DebugBenchmark,
    }

    def __init__(
        self,
        model_provider=None,
        output_dir: str = "./eval_results",
    ):
        self.model_provider = model_provider
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        benchmarks: Optional[list[str]] = None,
        system_prompt: Optional[str] = None,
    ) -> dict:
        """
        Run selected benchmarks.

        Args:
            benchmarks: List of benchmark names (None = all)
            system_prompt: System prompt to use

        Returns:
            Full results dict
        """
        if benchmarks is None:
            benchmarks = list(self.BENCHMARKS.keys())

        system = system_prompt or (
            "You are Saturday MK1, an enterprise coding AI. "
            "Provide clear, production-ready solutions."
        )

        all_summaries: dict[str, BenchmarkSummary] = {}

        for bench_name in benchmarks:
            if bench_name not in self.BENCHMARKS:
                log.warning(f"Unknown benchmark: {bench_name}")
                continue

            bench = self.BENCHMARKS[bench_name]()
            log.info(f"\n{'='*50}")
            log.info(f"Running benchmark: {bench.name} — {bench.description}")
            log.info(f"{'='*50}")

            tasks = bench.load_tasks()
            results = []

            for i, task in enumerate(tasks):
                log.info(f"  [{i+1}/{len(tasks)}] {task.task_id}...")

                # Get model response
                if self.model_provider:
                    try:
                        response, latency = self.model_provider.generate(system, task.prompt)
                    except Exception as e:
                        log.error(f"    Error: {e}")
                        results.append(EvalResult(
                            task_id=task.task_id,
                            benchmark=bench.name,
                            passed=False, score=0.0,
                            model_output="", latency_seconds=0,
                            error=str(e),
                        ))
                        continue
                else:
                    response, latency = "No model provider configured", 0

                # Evaluate
                result = bench.evaluate_response(task, response)
                result.latency_seconds = latency
                results.append(result)

                status = "[PASS]" if result.passed else "[FAIL]"
                log.info(f"    {status} Score: {result.score:.2f}")

            # Summary
            passed = sum(1 for r in results if r.passed)
            scores = [r.score for r in results]
            latencies = [r.latency_seconds for r in results if r.latency_seconds > 0]

            summary = BenchmarkSummary(
                name=bench.name,
                total_tasks=len(results),
                passed=passed,
                failed=len(results) - passed,
                pass_rate=round(passed / len(results) * 100, 1) if results else 0,
                avg_score=round(sum(scores) / len(scores), 3) if scores else 0,
                avg_latency=round(sum(latencies) / len(latencies), 2) if latencies else 0,
                results=results,
            )
            all_summaries[bench.name] = summary

            log.info(f"\n  >> {bench.name}: {summary.pass_rate}% pass rate | avg score: {summary.avg_score}")

        # Generate report
        report = self._generate_report(all_summaries)

        # Save results
        self._save_results(all_summaries, report)

        return report

    def _generate_report(self, summaries: dict[str, BenchmarkSummary]) -> dict:
        """Generate comparison report against Opus 4.6 baselines."""
        report = {
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "model": self.model_provider.model if self.model_provider else "unknown",
            "benchmarks": {},
            "overall": {},
        }

        total_pass = 0
        total_tasks = 0
        beat_opus = 0
        tied_opus = 0
        lost_opus = 0

        log.info(f"\n{'='*60}")
        log.info("SATURDAY MK1 vs CLAUDE OPUS 4.6 — COMPARISON")
        log.info(f"{'='*60}")
        log.info(f"{'Benchmark':<15} {'Saturday MK1':>10} {'Opus 4.6':>10} {'Delta':>8} {'Result':>8}")
        log.info(f"{'-'*60}")

        for bench_name, summary in summaries.items():
            opus = OPUS_BASELINES.get(bench_name, {})
            opus_rate = opus.get("pass_rate", 0) * 100

            sat_rate = summary.pass_rate
            delta = sat_rate - opus_rate

            if delta > 0:
                result = "[+] WIN"
                beat_opus += 1
            elif delta == 0:
                result = "[=] TIE"
                tied_opus += 1
            else:
                result = "[-] LOSE"
                lost_opus += 1

            log.info(f"{bench_name:<15} {sat_rate:>9.1f}% {opus_rate:>9.1f}% {delta:>+7.1f}% {result:>8}")

            report["benchmarks"][bench_name] = {
                "saturday_pass_rate": sat_rate,
                "opus_pass_rate": opus_rate,
                "delta": round(delta, 1),
                "avg_score": summary.avg_score,
                "avg_latency": summary.avg_latency,
                "tasks": len(summary.results),
                "won": delta > 0,
            }

            total_pass += summary.passed
            total_tasks += summary.total_tasks

        overall_rate = round(total_pass / total_tasks * 100, 1) if total_tasks else 0
        report["overall"] = {
            "total_tasks": total_tasks,
            "total_passed": total_pass,
            "overall_pass_rate": overall_rate,
            "benchmarks_won": beat_opus,
            "benchmarks_tied": tied_opus,
            "benchmarks_lost": lost_opus,
            "verdict": "SATURDAY MK1 WINS" if beat_opus > lost_opus else "NEED MORE TRAINING",
        }

        log.info(f"{'-'*60}")
        log.info(f"{'OVERALL':<15} {overall_rate:>9.1f}%")
        log.info(f"{'='*60}")
        log.info(f"Won: {beat_opus} | Tied: {tied_opus} | Lost: {lost_opus}")
        log.info(f"Verdict: {report['overall']['verdict']}")
        log.info(f"{'='*60}")

        return report

    def _save_results(self, summaries: dict, report: dict):
        """Save detailed results and report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        detailed = {}
        for name, summary in summaries.items():
            detailed[name] = {
                **{k: v for k, v in asdict(summary).items() if k != "results"},
                "results": [asdict(r) for r in summary.results],
            }

        detail_file = self.output_dir / f"eval_detailed_{timestamp}.json"
        with open(detail_file, "w") as f:
            json.dump(detailed, f, indent=2, ensure_ascii=False)

        # Save summary report
        report_file = self.output_dir / f"eval_report_{timestamp}.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        # Save markdown report
        md_file = self.output_dir / f"eval_report_{timestamp}.md"
        self._save_markdown_report(summaries, report, md_file)

        log.info(f"\nResults saved:")
        log.info(f"  Detailed: {detail_file}")
        log.info(f"  Report:   {report_file}")
        log.info(f"  Markdown: {md_file}")

    def _save_markdown_report(self, summaries: dict, report: dict, path: Path):
        """Generate markdown comparison report."""
        md = "# Saturday MK1 Evaluation Report\n\n"
        md += f"**Date**: {report['evaluated_at'][:10]}\n"
        md += f"**Model**: {report['model']}\n\n"

        md += "## Results vs Claude Opus 4.6\n\n"
        md += "| Benchmark | Saturday MK1 | Opus 4.6 | Delta | Result |\n"
        md += "|---|---|---|---|---|\n"

        for name, data in report["benchmarks"].items():
            result = "🟢 Win" if data["won"] else ("🟡 Tie" if data["delta"] == 0 else "🔴 Lose")
            md += (
                f"| {name} | {data['saturday_pass_rate']:.1f}% | "
                f"{data['opus_pass_rate']:.1f}% | {data['delta']:+.1f}% | {result} |\n"
            )

        md += f"\n**Overall**: {report['overall']['overall_pass_rate']:.1f}% pass rate\n"
        md += f"**Verdict**: {report['overall']['verdict']}\n"

        with open(path, "w") as f:
            f.write(md)


# ═══════════════════════════════════════════
# COMPARE MODE
# ═══════════════════════════════════════════

def compare_results(file_a: str, file_b: str):
    """Compare two evaluation result files side by side."""
    with open(file_a) as f:
        a = json.load(f)
    with open(file_b) as f:
        b = json.load(f)

    log.info(f"\n{'='*60}")
    log.info(f"COMPARISON: {Path(file_a).stem} vs {Path(file_b).stem}")
    log.info(f"{'='*60}")

    a_benchmarks = a.get("benchmarks", {})
    b_benchmarks = b.get("benchmarks", {})

    all_names = sorted(set(a_benchmarks.keys()) | set(b_benchmarks.keys()))

    log.info(f"{'Benchmark':<15} {'Model A':>10} {'Model B':>10} {'Winner':>10}")
    log.info(f"{'-'*50}")

    for name in all_names:
        a_score = a_benchmarks.get(name, {}).get("saturday_pass_rate", 0)
        b_score = b_benchmarks.get(name, {}).get("saturday_pass_rate", 0)
        winner = "A" if a_score > b_score else ("B" if b_score > a_score else "TIE")
        log.info(f"{name:<15} {a_score:>9.1f}% {b_score:>9.1f}% {winner:>10}")


# ═══════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Saturday MK1 Evaluation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to fine-tuned model directory",
    )
    parser.add_argument(
        "--provider", choices=["anthropic", "openai", "local", "dry-run"],
        default="dry-run",
        help="Model provider (default: dry-run)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model name for API providers",
    )
    parser.add_argument(
        "--base-url", type=str, default=None,
        help="Base URL for local/OpenAI-compatible endpoints",
    )
    parser.add_argument(
        "--benchmarks", nargs="+", default=None,
        choices=["humaneval", "enterprise", "security", "debug"],
        help="Specific benchmarks to run (default: all)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./eval_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Dry run — test framework without model",
    )
    parser.add_argument(
        "--compare", nargs=2, default=None, metavar=("FILE_A", "FILE_B"),
        help="Compare two result files",
    )

    args = parser.parse_args()

    # Compare mode
    if args.compare:
        compare_results(args.compare[0], args.compare[1])
        return

    # Import provider factory
    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir))

    try:
        from distill_from_api import get_provider, DryRunProvider
    except ImportError:
        # Inline fallback
        class DryRunProvider:
            model = "dry-run"
            def generate(self, system, prompt):
                return "Mock response for evaluation testing.", 0.1

        def get_provider(name, **kwargs):
            return DryRunProvider()

    # Setup provider
    if args.dry_run:
        provider = get_provider("dry-run")
    elif args.model_path:
        provider = get_provider(
            "local",
            model=args.model_path,
            base_url=args.base_url or "http://localhost:8000/v1",
        )
    else:
        provider = get_provider(
            args.provider,
            model=args.model,
            base_url=args.base_url,
        )

    log.info("=" * 60)
    log.info("SATURDAY MK1 EVALUATION SUITE")
    log.info(f"Model: {provider.model}")
    log.info("=" * 60)

    # Run evaluation
    runner = EvalRunner(
        model_provider=provider,
        output_dir=args.output_dir,
    )

    report = runner.run(benchmarks=args.benchmarks)

    log.info(f"\n>> Evaluation complete!")


if __name__ == "__main__":
    main()
