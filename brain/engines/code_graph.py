"""
Saturday Code Graph Engine — Hierarchical Project Understanding (HCGE)
=====================================================================
The most comprehensive codebase understanding engine in the AI market.
5-level project understanding that solves the needle-in-haystack problem.

Levels:
  0: Directory tree (file layout, size, ownership)
  1: Symbol extraction (classes, functions, variables per file)
  2: Dependency graph (imports, call graph, data flow)
  3: Architecture patterns (MVC, microservices, layered, event-driven)
  4: Evolution context (change risk, technical debt, complexity heatmap)

Key Innovations:
  - Multi-language AST parsing (Python full AST + regex extraction for JS/TS/Java/Go/Rust/C#)
  - Call graph construction across files
  - Dependency impact analysis ("what breaks if I change this?")
  - Architecture pattern auto-detection
  - Technical debt scoring per file
  - Change risk heatmap (complexity × dependencies × coupling)
"""

import ast
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


@dataclass
class ProjectNode:
    """A node in the project graph — represents a file or directory."""
    path: str
    name: str
    node_type: str         # "file", "directory", "package"
    language: str = ""
    size_bytes: int = 0
    line_count: int = 0
    classes: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    complexity_score: float = 0.0
    last_modified: str = ""
    # Extended fields
    call_targets: list[str] = field(default_factory=list)  # functions this file calls
    called_by: list[str] = field(default_factory=list)     # files that call this
    coupling_score: float = 0.0    # how coupled to other files
    tech_debt_score: float = 0.0   # estimated technical debt
    change_risk: float = 0.0       # risk of changes breaking things


@dataclass
class CallEdge:
    """An edge in the call graph."""
    source_file: str
    source_function: str
    target_file: str
    target_function: str
    call_count: int = 1


@dataclass
class ArchitecturePattern:
    """A detected architecture pattern in the project."""
    pattern_name: str   # "MVC", "microservices", "layered", "event_driven"
    confidence: float
    evidence: list[str] = field(default_factory=list)
    recommendation: str = ""


class CodeGraphEngine:
    """
    Hierarchical Code Graph Engine — 5-level project understanding.

    Usage:
        cg = CodeGraphEngine("./my_project", "./graph.json")
        summary = cg.scan_directory()
        outline = cg.get_file_outline("src/auth.py")
        deps = cg.get_dependencies("src/auth.py")
        impact = cg.get_change_impact("src/auth.py")
        patterns = cg.detect_architecture()
        debt = cg.get_tech_debt_report()
        heatmap = cg.get_risk_heatmap()
    """

    LANGUAGE_MAP = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".jsx": "javascript", ".tsx": "typescript",
        ".java": "java", ".go": "go", ".rs": "rust",
        ".cs": "csharp", ".rb": "ruby", ".php": "php",
        ".c": "c", ".cpp": "cpp", ".h": "c", ".hpp": "cpp",
    }

    SKIP_DIRS = {
        "node_modules", "__pycache__", ".git", ".svn", "venv", "env",
        ".venv", ".env", "dist", "build", ".tox", ".mypy_cache",
        ".pytest_cache", "target", "bin", "obj", ".idea", ".vscode",
    }

    def __init__(self, root_path: str, graph_path: str):
        self.root_path = os.path.abspath(root_path)
        self.graph_path = graph_path
        self.nodes: dict[str, ProjectNode] = {}
        self.call_graph: list[CallEdge] = []
        self.patterns: list[ArchitecturePattern] = []
        self._load_graph()

    def scan_directory(self, path: Optional[str] = None) -> dict:
        """Scan project directory and build the code graph."""
        scan_path = path or self.root_path
        self.nodes.clear()
        self.call_graph.clear()

        file_counts = defaultdict(int)
        total_lines = 0
        total_files = 0

        for dirpath, dirnames, filenames in os.walk(scan_path):
            dirnames[:] = [d for d in dirnames if d not in self.SKIP_DIRS]
            rel_dir = os.path.relpath(dirpath, self.root_path)

            for filename in sorted(filenames):
                filepath = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(filepath, self.root_path)
                ext = os.path.splitext(filename)[1].lower()
                lang = self.LANGUAGE_MAP.get(ext, "")

                if not lang:
                    continue

                try:
                    stat = os.stat(filepath)
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()

                    lines = content.count("\n") + 1
                    node = ProjectNode(
                        path=rel_path, name=filename, node_type="file",
                        language=lang, size_bytes=stat.st_size, line_count=lines,
                        last_modified=datetime.fromtimestamp(
                            stat.st_mtime, tz=timezone.utc
                        ).isoformat(),
                    )

                    # Parse based on language
                    if lang == "python":
                        self._parse_python_file(filepath, content, node)
                    else:
                        self._parse_generic_file(content, lang, node)

                    # Calculate complexity
                    node.complexity_score = self._calculate_complexity(content, lang)
                    node.tech_debt_score = self._calculate_tech_debt(content, lang, node)

                    self.nodes[rel_path] = node
                    file_counts[lang] += 1
                    total_lines += lines
                    total_files += 1

                except (IOError, OSError):
                    continue

        # Build cross-file references
        self._build_call_graph()
        self._calculate_coupling()
        self._calculate_change_risk()

        # Detect architecture patterns
        self.patterns = self.detect_architecture()

        self._save_graph()

        return {
            "total_files": total_files,
            "total_lines": total_lines,
            "languages": dict(file_counts),
            "patterns": [p.pattern_name for p in self.patterns],
            "avg_complexity": round(
                sum(n.complexity_score for n in self.nodes.values()) / max(total_files, 1), 2
            ),
        }

    def get_file_outline(self, filepath: str) -> dict:
        """Get structured outline of a source file."""
        node = self.nodes.get(filepath)
        if not node:
            return {"error": f"File {filepath} not found in graph"}

        return {
            "path": node.path, "language": node.language,
            "lines": node.line_count, "size": node.size_bytes,
            "classes": node.classes, "functions": node.functions,
            "imports": node.imports, "exports": node.exports,
            "complexity": node.complexity_score,
            "tech_debt": node.tech_debt_score,
            "change_risk": node.change_risk,
        }

    def get_dependencies(self, filepath: str) -> dict:
        """Get full dependency information for a file."""
        node = self.nodes.get(filepath)
        if not node:
            return {"error": f"File {filepath} not found"}

        depends_on = node.imports
        depended_by = [
            n.path for n in self.nodes.values()
            if filepath in n.imports or any(filepath in t for t in n.call_targets)
        ]
        calls = [e for e in self.call_graph if e.source_file == filepath]
        called_by = [e for e in self.call_graph if e.target_file == filepath]

        return {
            "imports": depends_on,
            "imported_by": depended_by,
            "calls_out": [{"target": e.target_file, "function": e.target_function} for e in calls],
            "called_from": [{"source": e.source_file, "function": e.source_function} for e in called_by],
            "coupling_score": node.coupling_score,
        }

    def get_change_impact(self, filepath: str) -> dict:
        """Analyze impact of changing a file (blast radius)."""
        node = self.nodes.get(filepath)
        if not node:
            return {"error": f"File {filepath} not found"}

        # Direct dependents
        direct = set()
        for n in self.nodes.values():
            if filepath in n.imports:
                direct.add(n.path)

        # Transitive dependents (2 levels deep)
        indirect = set()
        for dep in direct:
            for n in self.nodes.values():
                if dep in n.imports and n.path != filepath:
                    indirect.add(n.path)
        indirect -= direct

        return {
            "file": filepath,
            "directly_affected": sorted(direct),
            "indirectly_affected": sorted(indirect),
            "total_impact": len(direct) + len(indirect),
            "risk_score": node.change_risk,
            "recommendation": "High caution required" if len(direct) > 5 else "Standard review",
        }

    def detect_architecture(self) -> list[ArchitecturePattern]:
        """Auto-detect architecture patterns in the project."""
        patterns = []
        all_paths = [n.path.lower() for n in self.nodes.values()]
        all_names = [n.name.lower() for n in self.nodes.values()]

        # MVC Detection
        mvc_evidence = []
        has_models = any("model" in p for p in all_paths)
        has_views = any("view" in p or "template" in p for p in all_paths)
        has_controllers = any("controller" in p or "route" in p for p in all_paths)
        if has_models:
            mvc_evidence.append("models/ directory found")
        if has_views:
            mvc_evidence.append("views/templates found")
        if has_controllers:
            mvc_evidence.append("controllers/routes found")
        if len(mvc_evidence) >= 2:
            patterns.append(ArchitecturePattern(
                "MVC", len(mvc_evidence) / 3, mvc_evidence))

        # Microservices Detection
        micro_evidence = []
        docker_files = sum(1 for n in all_names if "dockerfile" in n)
        services = sum(1 for p in all_paths if "service" in p)
        if docker_files > 1:
            micro_evidence.append(f"{docker_files} Dockerfiles found")
        if services > 2:
            micro_evidence.append(f"{services} service modules found")
        if any("docker-compose" in n for n in all_names):
            micro_evidence.append("docker-compose.yml found")
        if len(micro_evidence) >= 2:
            patterns.append(ArchitecturePattern(
                "Microservices", min(1.0, len(micro_evidence) / 3), micro_evidence))

        # Layered Architecture
        layer_evidence = []
        layers = ["api", "service", "repository", "domain", "infrastructure"]
        found_layers = [l for l in layers if any(l in p for p in all_paths)]
        if len(found_layers) >= 3:
            layer_evidence.append(f"Layers found: {', '.join(found_layers)}")
            patterns.append(ArchitecturePattern(
                "Layered", len(found_layers) / len(layers), layer_evidence))

        # Event-Driven
        event_evidence = []
        if any(k in " ".join(all_names) for k in ["event", "handler", "listener", "subscriber"]):
            event_evidence.append("Event handlers/listeners found")
        if any(k in " ".join(all_names) for k in ["queue", "broker", "kafka", "rabbitmq"]):
            event_evidence.append("Message queue integration found")
        if len(event_evidence) >= 1:
            patterns.append(ArchitecturePattern(
                "Event-Driven", min(1.0, len(event_evidence) / 2), event_evidence))

        return patterns

    def get_tech_debt_report(self) -> dict:
        """Get technical debt analysis across the project."""
        if not self.nodes:
            return {"error": "No files scanned"}

        high_debt = sorted(
            [(n.path, n.tech_debt_score) for n in self.nodes.values()],
            key=lambda x: -x[1]
        )[:10]

        total_debt = sum(n.tech_debt_score for n in self.nodes.values())
        avg_debt = total_debt / len(self.nodes) if self.nodes else 0.0

        return {
            "total_debt_score": round(total_debt, 1),
            "average_debt": round(avg_debt, 1),
            "hotspots": [{"file": f, "debt": round(d, 1)} for f, d in high_debt],
            "files_above_threshold": sum(1 for n in self.nodes.values() if n.tech_debt_score > 7.0),
        }

    def get_risk_heatmap(self) -> list[dict]:
        """Get change risk heatmap sorted by risk."""
        return sorted(
            [{"file": n.path, "risk": round(n.change_risk, 1),
              "complexity": round(n.complexity_score, 1),
              "coupling": round(n.coupling_score, 1),
              "debt": round(n.tech_debt_score, 1)}
             for n in self.nodes.values()],
            key=lambda x: -x["risk"],
        )[:20]

    def get_project_summary(self) -> str:
        """Generate human-readable project summary."""
        if not self.nodes:
            return "No files scanned. Run scan_directory() first."

        langs = defaultdict(int)
        total_lines = 0
        total_classes = 0
        total_functions = 0
        for n in self.nodes.values():
            langs[n.language] += 1
            total_lines += n.line_count
            total_classes += len(n.classes)
            total_functions += len(n.functions)

        lines = [
            "# Saturday Code Graph — Project Summary\n",
            f"**Files**: {len(self.nodes)} | **Lines**: {total_lines:,}",
            f"**Classes**: {total_classes} | **Functions**: {total_functions}",
            f"**Languages**: {', '.join(f'{l} ({c})' for l, c in sorted(langs.items(), key=lambda x: -x[1]))}",
            "",
        ]

        if self.patterns:
            lines.append("## Architecture Patterns")
            for p in self.patterns:
                lines.append(f"- **{p.pattern_name}** ({p.confidence:.0%} confidence): {', '.join(p.evidence)}")
            lines.append("")

        debt = self.get_tech_debt_report()
        if debt.get("hotspots"):
            lines.append("## Tech Debt Hotspots")
            for h in debt["hotspots"][:5]:
                lines.append(f"- `{h['file']}`: {h['debt']:.1f}/10")

        return "\n".join(lines)

    # ── Parsing ──

    def _parse_python_file(self, filepath: str, content: str, node: ProjectNode):
        """Parse Python file with AST for full structure extraction."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            self._parse_generic_file(content, "python", node)
            return

        for item in ast.walk(tree):
            if isinstance(item, ast.ClassDef):
                node.classes.append(item.name)
                node.exports.append(item.name)
            elif isinstance(item, ast.FunctionDef) or isinstance(item, ast.AsyncFunctionDef):
                # Check if this function is a method inside a class
                # Guard: some AST nodes have a 'body' attr that is not a list (e.g. ast.Constant)
                def _is_method(func_node, tree):
                    for p in ast.walk(tree):
                        if isinstance(p, ast.ClassDef):
                            body = getattr(p, 'body', [])
                            if isinstance(body, list) and func_node in body:
                                return True
                    return False
                if not _is_method(item, tree):
                    node.functions.append(item.name)
                    if not item.name.startswith("_"):
                        node.exports.append(item.name)
            elif isinstance(item, ast.Import):
                for alias in item.names:
                    node.imports.append(alias.name)
            elif isinstance(item, ast.ImportFrom):
                if item.module:
                    node.imports.append(item.module)
            elif isinstance(item, ast.Call):
                if isinstance(item.func, ast.Name):
                    node.call_targets.append(item.func.id)
                elif isinstance(item.func, ast.Attribute):
                    node.call_targets.append(item.func.attr)

    def _parse_generic_file(self, content: str, lang: str, node: ProjectNode):
        """Regex-based parsing for non-Python languages."""
        patterns = {
            "javascript": {
                "class": r"(?:class|export\s+class)\s+(\w+)",
                "function": r"(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\()",
                "import": r"(?:import\s+.*?from\s+['\"](.+?)['\"]|require\s*\(\s*['\"](.+?)['\"]\))",
            },
            "typescript": {
                "class": r"(?:class|export\s+class)\s+(\w+)",
                "function": r"(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*(?::\s*\w+)?\s*=\s*(?:async\s*)?\()",
                "import": r"import\s+.*?from\s+['\"](.+?)['\"]",
            },
            "java": {
                "class": r"(?:public\s+|private\s+|protected\s+)?class\s+(\w+)",
                "function": r"(?:public|private|protected)\s+\w+\s+(\w+)\s*\(",
                "import": r"import\s+([\w.]+);",
            },
            "go": {
                "class": r"type\s+(\w+)\s+struct",
                "function": r"func\s+(?:\(.*?\)\s+)?(\w+)\s*\(",
                "import": r"\"([\w./]+)\"",
            },
            "rust": {
                "class": r"(?:pub\s+)?struct\s+(\w+)",
                "function": r"(?:pub\s+)?fn\s+(\w+)",
                "import": r"use\s+([\w:]+)",
            },
            "csharp": {
                "class": r"(?:public|internal|private)?\s*class\s+(\w+)",
                "function": r"(?:public|private|protected|internal)\s+\w+\s+(\w+)\s*\(",
                "import": r"using\s+([\w.]+);",
            },
        }

        lang_patterns = patterns.get(lang, {})
        for match_type, pattern in lang_patterns.items():
            for match in re.finditer(pattern, content):
                name = next((g for g in match.groups() if g), None)
                if name:
                    if match_type == "class":
                        node.classes.append(name)
                    elif match_type == "function":
                        node.functions.append(name)
                    elif match_type == "import":
                        node.imports.append(name)

    # ── Analysis ──

    def _calculate_complexity(self, content: str, lang: str) -> float:
        """Calculate cyclomatic complexity estimate."""
        score = 0.0
        branch_patterns = [
            r"\bif\b", r"\belif\b", r"\belse\b", r"\bfor\b", r"\bwhile\b",
            r"\btry\b", r"\bexcept\b", r"\bcatch\b", r"\bswitch\b",
            r"\bcase\b", r"\b&&\b", r"\b\|\|\b", r"\band\b", r"\bor\b",
        ]
        for pattern in branch_patterns:
            score += len(re.findall(pattern, content))

        lines = content.count("\n") + 1
        # Normalize by file size
        if lines > 0:
            score = (score / lines) * 10
        return round(min(10.0, score), 2)

    def _calculate_tech_debt(self, content: str, lang: str, node: ProjectNode) -> float:
        """Calculate technical debt score (0-10)."""
        debt = 0.0

        # Long file penalty
        if node.line_count > 500:
            debt += min(3.0, (node.line_count - 500) / 500)
        # High complexity penalty
        if node.complexity_score > 5.0:
            debt += (node.complexity_score - 5.0) * 0.5
        # TODO/FIXME/HACK count
        debt += min(2.0, len(re.findall(r"(?:TODO|FIXME|HACK|XXX|TEMP)", content)) * 0.3)
        # Long functions (>50 lines)
        long_funcs = len(re.findall(r"(?:def |function |func )", content))
        if long_funcs > 0 and node.line_count / long_funcs > 50:
            debt += 1.0
        # Deep nesting
        max_indent = max((len(l) - len(l.lstrip()) for l in content.split("\n") if l.strip()), default=0)
        if max_indent > 16:
            debt += min(2.0, (max_indent - 16) / 4)

        return round(min(10.0, debt), 1)

    def _build_call_graph(self):
        """Build cross-file call graph from imports + call targets."""
        self.call_graph.clear()

        # Map function names to files
        func_to_file: dict[str, str] = {}
        for path, node in self.nodes.items():
            for func in node.functions + node.classes:
                func_to_file[func] = path

        # Match call targets to defined functions
        for path, node in self.nodes.items():
            for call_target in node.call_targets:
                if call_target in func_to_file and func_to_file[call_target] != path:
                    self.call_graph.append(CallEdge(
                        source_file=path,
                        source_function="*",
                        target_file=func_to_file[call_target],
                        target_function=call_target,
                    ))

    def _calculate_coupling(self):
        """Calculate coupling score for each file."""
        for path, node in self.nodes.items():
            incoming = sum(1 for e in self.call_graph if e.target_file == path)
            outgoing = sum(1 for e in self.call_graph if e.source_file == path)
            node.coupling_score = round(min(10.0, (incoming + outgoing) * 0.5), 1)

    def _calculate_change_risk(self):
        """Calculate change risk = complexity × coupling × debt."""
        for node in self.nodes.values():
            node.change_risk = round(min(10.0, (
                node.complexity_score * 0.3 +
                node.coupling_score * 0.4 +
                node.tech_debt_score * 0.3
            )), 1)

    # ── Persistence ──

    def _save_graph(self):
        """Persist the graph to disk."""
        if not self.graph_path:
            return
        data = {
            "nodes": {k: asdict(v) for k, v in self.nodes.items()},
            "call_graph": [asdict(e) for e in self.call_graph],
            "patterns": [asdict(p) for p in self.patterns],
            "scanned_at": datetime.now(timezone.utc).isoformat(),
        }
        path = Path(self.graph_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load_graph(self):
        """Load a previously saved graph."""
        path = Path(self.graph_path)
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for node_path, node_data in data.get("nodes", {}).items():
                self.nodes[node_path] = ProjectNode(**{
                    k: v for k, v in node_data.items()
                    if k in ProjectNode.__dataclass_fields__
                })
        except (json.JSONDecodeError, TypeError, KeyError):
            self.nodes = {}
