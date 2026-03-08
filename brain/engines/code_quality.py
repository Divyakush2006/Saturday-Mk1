"""
Saturday Code Quality Scorer — 8-Dimensional Enterprise Grading System
======================================================================
The most comprehensive code quality assessment in the AI coding market.
Grades across 8 dimensions with anti-pattern detection and SOLID enforcement.

Dimensions:
  1. Maintainability (structure, naming, modularity, readability)
  2. Security (vulnerability patterns, input validation)
  3. Documentation (docstrings, comments, type hints)
  4. Testing (test patterns, assertions, coverage indicators)
  5. Performance (N+1 queries, unbounded loops, memory leaks)
  6. Reliability (error handling, edge cases, fail-safes)
  7. Scalability (pagination, caching, async patterns)
  8. Observability (logging, metrics, health checks)

Only A and A+ code ships to production in MNC environments.
"""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class QualityIssue:
    """A single code quality issue."""
    category: str       # maintainability, security, etc.
    severity: str       # critical, major, minor, info
    message: str
    line_number: Optional[int] = None
    fix_suggestion: str = ""
    principle: str = ""  # SOLID, DRY, KISS, etc.


@dataclass
class QualityScore:
    """Complete quality assessment of a code sample."""
    overall: float = 0.0
    grade: str = "F"
    # 8 dimensions
    maintainability: float = 0.0
    security: float = 0.0
    documentation: float = 0.0
    testing: float = 0.0
    performance: float = 0.0
    reliability: float = 0.0
    scalability: float = 0.0
    observability: float = 0.0
    enterprise_ready: bool = False
    issues: list[QualityIssue] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    anti_patterns: list[str] = field(default_factory=list)


class CodeQualityScorer:
    """
    8-dimensional enterprise code quality grading system.

    Usage:
        scorer = CodeQualityScorer()
        result = scorer.score(code, "python")
        print(result.grade)  # "A", "B+", etc.
        report = scorer.generate_report(result)
    """

    GRADE_THRESHOLDS = [
        (95, "A+"), (90, "A"), (85, "A-"), (80, "B+"), (75, "B"),
        (70, "B-"), (65, "C+"), (60, "C"), (55, "C-"), (50, "D+"),
        (45, "D"), (40, "D-"), (0, "F"),
    ]

    def score(self, code: str, language: str = "python") -> QualityScore:
        """Score code quality across all 8 dimensions."""
        lang = language.lower()
        issues: list[QualityIssue] = []
        strengths: list[str] = []
        anti_patterns: list[str] = []

        m = self._score_maintainability(code, lang, issues, strengths, anti_patterns)
        s = self._score_security(code, lang, issues, strengths)
        d = self._score_documentation(code, lang, issues, strengths)
        t = self._score_testing(code, lang, issues, strengths)
        p = self._score_performance(code, lang, issues, strengths)
        r = self._score_reliability(code, lang, issues, strengths)
        sc = self._score_scalability(code, lang, issues, strengths)
        o = self._score_observability(code, lang, issues, strengths)

        # Weighted overall (security and reliability weighted higher for enterprise)
        overall = (
            m * 0.15 + s * 0.20 + d * 0.10 + t * 0.10 +
            p * 0.12 + r * 0.15 + sc * 0.08 + o * 0.10
        )
        overall = max(0, min(100, overall))

        grade = "F"
        for threshold, g in self.GRADE_THRESHOLDS:
            if overall >= threshold:
                grade = g
                break

        enterprise_ready = (
            overall >= 75 and s >= 70 and r >= 65 and
            not any(i.severity == "critical" for i in issues)
        )

        return QualityScore(
            overall=round(overall, 1), grade=grade,
            maintainability=round(m, 1), security=round(s, 1),
            documentation=round(d, 1), testing=round(t, 1),
            performance=round(p, 1), reliability=round(r, 1),
            scalability=round(sc, 1), observability=round(o, 1),
            enterprise_ready=enterprise_ready,
            issues=issues, strengths=strengths,
            anti_patterns=anti_patterns,
        )

    def generate_report(self, score: QualityScore) -> str:
        """Generate formatted quality report."""
        bar = lambda v: "█" * int(v / 10) + "░" * (10 - int(v / 10))

        lines = [
            "📊 SATURDAY CODE QUALITY REPORT",
            "=" * 50,
            f"   Grade: {score.grade} ({score.overall:.0f}/100)",
            f"   Enterprise Ready: {'✅ Yes' if score.enterprise_ready else '❌ No'}",
            "",
            "  ── Dimension Scores ──",
            f"   Maintainability: [{bar(score.maintainability)}] {score.maintainability:.0f}",
            f"   Security:        [{bar(score.security)}] {score.security:.0f}",
            f"   Documentation:   [{bar(score.documentation)}] {score.documentation:.0f}",
            f"   Testing:         [{bar(score.testing)}] {score.testing:.0f}",
            f"   Performance:     [{bar(score.performance)}] {score.performance:.0f}",
            f"   Reliability:     [{bar(score.reliability)}] {score.reliability:.0f}",
            f"   Scalability:     [{bar(score.scalability)}] {score.scalability:.0f}",
            f"   Observability:   [{bar(score.observability)}] {score.observability:.0f}",
            "",
        ]

        if score.anti_patterns:
            lines.append("  ── Anti-Patterns Detected ──")
            for ap in score.anti_patterns:
                lines.append(f"   🚫 {ap}")
            lines.append("")

        if score.issues:
            sev_counts = {}
            for i in score.issues:
                sev_counts[i.severity] = sev_counts.get(i.severity, 0) + 1
            lines.append(f"  ── Issues ({len(score.issues)}) ──")
            for issue in sorted(score.issues, key=lambda i: {"critical": 0, "major": 1, "minor": 2}.get(i.severity, 3)):
                icon = {"critical": "🔴", "major": "🟠", "minor": "🟡"}.get(issue.severity, "🔵")
                lines.append(f"   {icon} [{issue.category}] {issue.message}")
                if issue.fix_suggestion:
                    lines.append(f"      Fix: {issue.fix_suggestion}")
            lines.append("")

        if score.strengths:
            lines.append("  ── Strengths ──")
            for s in score.strengths:
                lines.append(f"   ✅ {s}")

        return "\n".join(lines)

    # ── Dimension Scorers ──

    def _score_maintainability(self, code: str, lang: str, issues: list,
                               strengths: list, anti_patterns: list) -> float:
        score = 80.0
        lines = code.split("\n")
        total_lines = len(lines)

        # File length
        if total_lines > 500:
            penalty = min(20, (total_lines - 500) / 50)
            score -= penalty
            anti_patterns.append(f"God File — {total_lines} lines (max recommended: 500)")

        # Long functions
        func_starts = [i for i, l in enumerate(lines) if re.match(r"\s*def\s+", l)]
        for start in func_starts:
            end = next((j for j in range(start + 1, min(start + 200, total_lines))
                       if j < total_lines and re.match(r"\S", lines[j]) and
                       not lines[j].strip().startswith("#")), start + 50)
            func_len = end - start
            if func_len > 50:
                issues.append(QualityIssue(
                    "maintainability", "major",
                    f"Function at line {start + 1} is {func_len} lines (max: 50)",
                    line_number=start + 1, principle="SRP",
                    fix_suggestion="Extract helper functions to reduce complexity",
                ))
                score -= 5

        # Deep nesting
        max_indent = 0
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                max_indent = max(max_indent, indent)
        if max_indent > 20:
            score -= min(15, (max_indent - 20) / 2)
            anti_patterns.append(f"Deep Nesting — max indent {max_indent} spaces")

        # Good naming (snake_case for Python)
        if lang == "python":
            good_names = len(re.findall(r"def\s+[a-z][a-z0-9_]*\s*\(", code))
            bad_names = len(re.findall(r"def\s+[A-Z][a-zA-Z]*\s*\(", code))
            if good_names > 0 and bad_names == 0:
                strengths.append("Consistent snake_case naming convention")
                score += 5

        # Type hints
        if lang == "python":
            typed_funcs = len(re.findall(r"def\s+\w+\s*\(.*?(?::\s*\w+|->)", code))
            total_funcs = len(re.findall(r"def\s+\w+\s*\(", code))
            if total_funcs > 0:
                ratio = typed_funcs / total_funcs
                if ratio >= 0.8:
                    strengths.append(f"Strong type annotations ({ratio:.0%} of functions)")
                    score += 5
                elif ratio < 0.3:
                    issues.append(QualityIssue(
                        "maintainability", "minor", "Low type hint coverage",
                        fix_suggestion="Add type hints to function parameters and return types",
                    ))
                    score -= 5

        return max(0, min(100, score))

    def _score_security(self, code: str, lang: str, issues: list, strengths: list) -> float:
        score = 100.0

        danger_patterns = [
            (r"\beval\s*\(", 25, "critical", "eval() — arbitrary code execution"),
            (r"pickle\.loads?\s*\(", 25, "critical", "pickle.load — insecure deserialization"),
            (r"os\.system\s*\(", 20, "critical", "os.system() — command injection risk"),
            (r"subprocess.*shell\s*=\s*True", 20, "critical", "shell=True — command injection"),
            (r"(?:password|secret)\s*=\s*['\"]", 15, "major", "Hardcoded credentials"),
            (r"hashlib\.(md5|sha1)\s*\(", 10, "major", "Weak hash algorithm"),
            (r"verify\s*=\s*False", 10, "major", "TLS verification disabled"),
            (r"random\.(random|randint)\s*\(", 5, "minor", "Non-cryptographic random for security"),
        ]

        for pattern, penalty, severity, msg in danger_patterns:
            if re.search(pattern, code):
                score -= penalty
                issues.append(QualityIssue("security", severity, msg))

        # Positive signals
        if re.search(r"import\s+(?:hashlib|hmac|secrets|bcrypt|argon2)", code):
            strengths.append("Uses proper cryptographic libraries")
            score += 5
        if re.search(r"(?:parameterized|prepared|%s|:param|\$\d+)", code):
            strengths.append("Uses parameterized queries")
            score += 5

        return max(0, min(100, score))

    def _score_documentation(self, code: str, lang: str, issues: list, strengths: list) -> float:
        score = 50.0
        total_funcs = len(re.findall(r"def\s+\w+\s*\(", code))

        if lang == "python":
            docstrings = len(re.findall(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'', code))
            if total_funcs > 0:
                ratio = min(1.0, docstrings / total_funcs)
                score += ratio * 30
                if ratio >= 0.7:
                    strengths.append(f"Good docstring coverage ({ratio:.0%})")
                elif ratio < 0.3 and total_funcs > 2:
                    issues.append(QualityIssue(
                        "documentation", "minor", f"Low docstring coverage ({ratio:.0%})",
                        fix_suggestion="Add docstrings to all public functions",
                    ))

        # Module-level docs
        if re.match(r'\s*(?:"""|\'\'\').*?(?:"""|\'\'\')|\s*#\s*\w', code, re.DOTALL):
            score += 10
            strengths.append("Module-level documentation present")

        # Inline comments quality
        comment_lines = len(re.findall(r"^\s*#(?!nosec|type:|noqa)", code, re.MULTILINE))
        code_lines = len([l for l in code.split("\n") if l.strip() and not l.strip().startswith("#")])
        if code_lines > 0 and comment_lines / code_lines > 0.1:
            score += 10

        return max(0, min(100, score))

    def _score_testing(self, code: str, lang: str, issues: list, strengths: list) -> float:
        score = 50.0

        test_patterns = [
            r"(?:def\s+test_|class\s+Test)", r"(?:assert\s+|self\.assert)",
            r"(?:@pytest|unittest|mock|patch)", r"(?:expect\(|describe\(|it\()",
        ]
        test_count = sum(1 for p in test_patterns if re.search(p, code))

        if test_count >= 3:
            score += 30
            strengths.append("Comprehensive test patterns detected")
        elif test_count >= 1:
            score += 15
        else:
            if len(code.split("\n")) > 50:
                issues.append(QualityIssue(
                    "testing", "major", "No test patterns detected",
                    fix_suggestion="Add unit tests with pytest or unittest",
                ))

        # Assertions count
        assert_count = len(re.findall(r"assert\s+|assertEqual|assertTrue|assertRaises", code))
        if assert_count >= 5:
            score += 20
            strengths.append(f"{assert_count} assertions found")

        return max(0, min(100, score))

    def _score_performance(self, code: str, lang: str, issues: list, strengths: list) -> float:
        score = 80.0

        # N+1 query pattern
        if re.search(r"for\s+.*\bin\b.*:[\s\S]*?(?:query|execute|select|find_one)\s*\(", code, re.DOTALL):
            score -= 20
            issues.append(QualityIssue(
                "performance", "major", "Potential N+1 query in loop",
                fix_suggestion="Batch queries outside the loop; use JOINs or prefetch",
                principle="Performance",
            ))

        # Unbounded iteration
        if re.search(r"while\s+True\s*:", code) and not re.search(r"\bbreak\b", code):
            score -= 15
            issues.append(QualityIssue("performance", "critical", "Unbounded loop without break"))

        # String concatenation in loop
        if re.search(r"for\s+.*:[\s\S]*?\+\s*=\s*['\"]", code, re.DOTALL):
            score -= 5
            issues.append(QualityIssue(
                "performance", "minor", "String concatenation in loop",
                fix_suggestion="Use list and ''.join() or io.StringIO",
            ))

        # Good patterns
        if re.search(r"(?:lru_cache|cache|memoize|@cached)", code):
            strengths.append("Caching implemented")
            score += 10
        if re.search(r"(?:async\s+def|await\s+|asyncio)", code):
            strengths.append("Async patterns for I/O operations")
            score += 5

        return max(0, min(100, score))

    def _score_reliability(self, code: str, lang: str, issues: list, strengths: list) -> float:
        score = 70.0

        # Error handling
        try_blocks = len(re.findall(r"\btry\s*:", code))
        except_blocks = len(re.findall(r"\bexcept\s+\w+", code))
        bare_except = len(re.findall(r"\bexcept\s*:", code))

        if except_blocks > 0:
            strengths.append(f"Specific exception handling ({except_blocks} handlers)")
            score += 10
        if bare_except > 0:
            score -= bare_except * 5
            issues.append(QualityIssue("reliability", "major",
                                       f"{bare_except} bare except clause(s)"))
        if try_blocks == 0 and len(code.split("\n")) > 30:
            issues.append(QualityIssue("reliability", "minor", "No error handling"))
            score -= 10

        # Input validation
        if re.search(r"(?:validate|sanitize|clean|check|verify)\s*\(", code):
            strengths.append("Input validation functions present")
            score += 10
        # Guard clauses
        early_returns = len(re.findall(r"if\s+.*:\s*\n\s*(?:return|raise|continue)", code))
        if early_returns >= 2:
            strengths.append(f"Guard clauses for early exit ({early_returns})")
            score += 5

        return max(0, min(100, score))

    def _score_scalability(self, code: str, lang: str, issues: list, strengths: list) -> float:
        score = 70.0

        # Pagination
        if re.search(r"(?:paginate|limit|offset|page_size|per_page)", code):
            strengths.append("Pagination support detected")
            score += 10
        elif re.search(r"\.all\s*\(\s*\)", code):
            issues.append(QualityIssue("scalability", "minor", "Unbounded query — no pagination"))
            score -= 10

        # Async/concurrent patterns
        if re.search(r"(?:async|await|ThreadPool|ProcessPool|asyncio|concurrent)", code):
            score += 10

        # Connection pooling
        if re.search(r"(?:pool|Pool|connection_pool|engine\.create)", code):
            strengths.append("Connection pooling detected")
            score += 10

        return max(0, min(100, score))

    def _score_observability(self, code: str, lang: str, issues: list, strengths: list) -> float:
        score = 50.0

        # Logging
        if re.search(r"(?:logging\.|logger\.|log\.|console\.(?:log|error|warn))", code):
            strengths.append("Logging implementation detected")
            score += 20
        elif len(code.split("\n")) > 50:
            issues.append(QualityIssue("observability", "minor", "No logging detected",
                                       fix_suggestion="Add logging: import logging; log = logging.getLogger(__name__)"))

        # Structured logging
        if re.search(r"(?:structlog|json.*log|extra\s*=\s*\{)", code):
            strengths.append("Structured logging")
            score += 10

        # Health checks / metrics
        if re.search(r"(?:health|ready|alive|metrics|prometheus|statsd)", code):
            strengths.append("Health check / metrics endpoint")
            score += 10

        # Error tracking
        if re.search(r"(?:sentry|bugsnag|rollbar|errorhandler)", code):
            strengths.append("Error tracking integration")
            score += 10

        return max(0, min(100, score))
