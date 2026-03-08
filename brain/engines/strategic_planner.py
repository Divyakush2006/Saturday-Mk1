"""
Saturday Strategic Planner — Military-Grade Execution Planning
==============================================================
Dependency-aware task ordering with parallel execution detection,
blast radius analysis, and multi-phase rollout planning.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class TaskNode:
    """A single task in the execution plan."""
    task_id: str
    title: str
    description: str
    status: str = "pending"       # pending, in_progress, completed, blocked, deferred
    priority: str = "medium"      # critical, high, medium, low
    effort: str = "medium"        # small (< 1h), medium (1-4h), large (4-8h), xlarge (> 8h)
    dependencies: list[str] = field(default_factory=list)  # task_ids
    dependents: list[str] = field(default_factory=list)
    affected_files: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    parallel_group: Optional[str] = None  # tasks in same group can run in parallel
    phase: int = 1
    assigned_to: str = ""


@dataclass
class BlastRadius:
    """Impact analysis for a proposed change."""
    directly_affected: list[str] = field(default_factory=list)
    indirectly_affected: list[str] = field(default_factory=list)
    test_suites_impacted: list[str] = field(default_factory=list)
    rollback_complexity: str = "low"    # low, medium, high
    risk_score: float = 0.0             # 0-10
    recommendation: str = ""
    requires_review: bool = False


@dataclass
class ExecutionPlan:
    """Complete execution plan with phases and dependencies."""
    plan_id: str
    title: str
    created_at: str
    phases: list[dict] = field(default_factory=list)  # [{phase, tasks, parallel_groups}]
    total_effort: str = ""
    critical_path: list[str] = field(default_factory=list)
    blast_radius: Optional[BlastRadius] = None
    risks: list[str] = field(default_factory=list)
    rollback_plan: str = ""


class StrategicPlanner:
    """
    Military-grade execution planner with dependency analysis.

    Usage:
        planner = StrategicPlanner()
        tasks = planner.decompose_objective("Add OAuth2 to the API", context)
        plan = planner.create_plan("OAuth2 Implementation", tasks)
        report = planner.generate_report(plan)
    """

    def __init__(self):
        self._task_counter = 0
        self.task_templates = self._build_templates()

    def decompose_objective(self, objective: str, context: str = "") -> list[TaskNode]:
        """Break down a high-level objective into ordered tasks."""
        tasks = []
        obj_lower = objective.lower()
        combined = f"{objective} {context}".lower()

        # Template matching
        for template in self.task_templates:
            if any(kw in combined for kw in template["keywords"]):
                for task_def in template["tasks"]:
                    self._task_counter += 1
                    task = TaskNode(
                        task_id=f"T-{self._task_counter:04d}",
                        title=task_def["title"],
                        description=task_def.get("description", ""),
                        priority=task_def.get("priority", "medium"),
                        effort=task_def.get("effort", "medium"),
                        affected_files=task_def.get("files", []),
                        risks=task_def.get("risks", []),
                        phase=task_def.get("phase", 1),
                    )
                    tasks.append(task)

        if not tasks:
            # Generic decomposition
            generic = [
                {"title": f"Research: {objective}", "priority": "high", "effort": "small", "phase": 1},
                {"title": f"Design: {objective}", "priority": "high", "effort": "medium", "phase": 1},
                {"title": f"Implement: {objective}", "priority": "high", "effort": "large", "phase": 2},
                {"title": f"Test: {objective}", "priority": "high", "effort": "medium", "phase": 3},
                {"title": f"Review: {objective}", "priority": "medium", "effort": "small", "phase": 3},
                {"title": f"Deploy: {objective}", "priority": "medium", "effort": "small", "phase": 4},
            ]
            for td in generic:
                self._task_counter += 1
                tasks.append(TaskNode(
                    task_id=f"T-{self._task_counter:04d}", **td,
                    description=f"Auto-generated task for: {objective}",
                ))

        # Build dependency chain
        self._resolve_dependencies(tasks)
        # Detect parallelizable tasks
        self._detect_parallelism(tasks)

        return tasks

    def create_plan(self, title: str, tasks_or_context=None) -> ExecutionPlan:
        """Create a phased execution plan.

        Accepts either pre-decomposed TaskNode objects or raw context
        (dict/str) which triggers automatic objective decomposition.

        Args:
            title: Plan title or high-level objective description.
            tasks_or_context: One of:
                - list[TaskNode]: Pre-decomposed tasks (used directly).
                - dict: Context metadata passed to decompose_objective().
                - str: Additional context string for decomposition.
                - None: Auto-decompose title with no extra context.

        Returns:
            ExecutionPlan with phases, critical path, blast radius,
            and rollback strategy.
        """
        # ── Auto-decompose when caller provides context instead of tasks ──
        if (
            isinstance(tasks_or_context, list)
            and tasks_or_context
            and isinstance(tasks_or_context[0], TaskNode)
        ):
            tasks = tasks_or_context
        else:
            context = ""
            if isinstance(tasks_or_context, dict):
                context = " ".join(f"{k}={v}" for k, v in tasks_or_context.items())
            elif isinstance(tasks_or_context, str):
                context = tasks_or_context
            tasks = self.decompose_objective(title, context)

        plan_id = f"plan_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        # Group tasks by phase
        phases_dict: dict[int, list[TaskNode]] = {}
        for task in tasks:
            phases_dict.setdefault(task.phase, []).append(task)

        phases = []
        for phase_num in sorted(phases_dict.keys()):
            phase_tasks = phases_dict[phase_num]
            parallel_groups: dict[str, list[str]] = {}
            for t in phase_tasks:
                if t.parallel_group:
                    parallel_groups.setdefault(t.parallel_group, []).append(t.task_id)
            phases.append({
                "phase": phase_num,
                "tasks": [{"id": t.task_id, "title": t.title, "priority": t.priority,
                          "effort": t.effort, "status": t.status} for t in phase_tasks],
                "parallel_groups": parallel_groups,
                "can_parallel": len(parallel_groups) > 0,
            })

        # Calculate critical path
        critical = self._find_critical_path(tasks)

        # Calculate blast radius
        all_files = []
        for t in tasks:
            all_files.extend(t.affected_files)
        blast = self.analyze_blast_radius(list(set(all_files)))

        # Effort estimation
        effort_map = {"small": 1, "medium": 3, "large": 6, "xlarge": 10}
        total_hours = sum(effort_map.get(t.effort, 3) for t in tasks)
        total_effort = f"{total_hours}h estimated ({total_hours / 8:.1f} days)"

        # Risks
        all_risks = []
        for t in tasks:
            all_risks.extend(t.risks)

        return ExecutionPlan(
            plan_id=plan_id, title=title,
            created_at=datetime.now(timezone.utc).isoformat(),
            phases=phases, total_effort=total_effort,
            critical_path=[t.task_id for t in critical],
            blast_radius=blast, risks=list(set(all_risks)),
            rollback_plan=self._generate_rollback_plan(tasks),
        )

    def analyze_blast_radius(self, affected_files: list[str]) -> BlastRadius:
        """Analyze the blast radius of changes to given files."""
        direct = affected_files[:]
        indirect = []

        # Heuristic: config files affect everything
        for f in affected_files:
            if any(cfg in f.lower() for cfg in ["config", "settings", ".env", "package.json"]):
                indirect.append(f"All modules (via config: {f})")

        # Core/shared files have wider blast radius
        for f in affected_files:
            if any(k in f.lower() for k in ["core", "base", "shared", "common", "utils"]):
                indirect.append(f"Dependent modules (via shared: {f})")

        test_impact = [f"Tests for {f}" for f in affected_files
                      if not f.lower().startswith("test")]

        total = len(direct) + len(indirect)
        risk = min(10.0, total * 1.5)
        rollback = "high" if total > 10 else "medium" if total > 5 else "low"

        return BlastRadius(
            directly_affected=direct,
            indirectly_affected=indirect,
            test_suites_impacted=test_impact,
            rollback_complexity=rollback,
            risk_score=round(risk, 1),
            recommendation="Staged rollout recommended" if risk > 5 else "Standard deployment OK",
            requires_review=risk > 5,
        )

    def generate_report(self, plan: ExecutionPlan) -> str:
        """Generate human-readable execution plan report."""
        lines = [
            "📋 SATURDAY STRATEGIC EXECUTION PLAN",
            "=" * 50,
            f"   Plan: {plan.title}",
            f"   ID: {plan.plan_id}",
            f"   Effort: {plan.total_effort}",
            f"   Phases: {len(plan.phases)}",
            "",
        ]

        for phase in plan.phases:
            lines.append(f"  ── Phase {phase['phase']} ──")
            parallel_note = " (parallelizable)" if phase.get("can_parallel") else ""
            for task in phase["tasks"]:
                icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵"}.get(task["priority"], "⚪")
                lines.append(f"    {icon} [{task['id']}] {task['title']} ({task['effort']})")
            lines.append("")

        if plan.critical_path:
            lines.append(f"  Critical Path: {' → '.join(plan.critical_path)}")
            lines.append("")

        if plan.blast_radius:
            br = plan.blast_radius
            lines.append(f"  ── Blast Radius (Risk: {br.risk_score:.1f}/10) ──")
            lines.append(f"    Direct: {len(br.directly_affected)} files")
            lines.append(f"    Indirect: {len(br.indirectly_affected)} modules")
            lines.append(f"    Rollback: {br.rollback_complexity}")
            lines.append(f"    {br.recommendation}")
            lines.append("")

        if plan.risks:
            lines.append("  ── Risks ──")
            for r in plan.risks:
                lines.append(f"    ⚠️ {r}")
            lines.append("")

        if plan.rollback_plan:
            lines.append(f"  ── Rollback Plan ──")
            lines.append(f"    {plan.rollback_plan}")

        return "\n".join(lines)

    # ── Internal ──

    def _resolve_dependencies(self, tasks: list[TaskNode]):
        """Build dependency chain between tasks."""
        for i, task in enumerate(tasks):
            for j, other in enumerate(tasks):
                if i >= j:
                    continue
                if task.phase < other.phase:
                    other.dependencies.append(task.task_id)
                    task.dependents.append(other.task_id)

    def _detect_parallelism(self, tasks: list[TaskNode]):
        """Find tasks that can run in parallel (same phase, no mutual deps)."""
        by_phase: dict[int, list[TaskNode]] = {}
        for task in tasks:
            by_phase.setdefault(task.phase, []).append(task)

        group_counter = 0
        for phase, phase_tasks in by_phase.items():
            if len(phase_tasks) > 1:
                # Check for mutual independence
                for i, a in enumerate(phase_tasks):
                    for b in phase_tasks[i + 1:]:
                        if (a.task_id not in b.dependencies and
                            b.task_id not in a.dependencies):
                            group_counter += 1
                            group_name = f"PG-{group_counter:03d}"
                            if not a.parallel_group:
                                a.parallel_group = group_name
                            b.parallel_group = a.parallel_group

    def _find_critical_path(self, tasks: list[TaskNode]) -> list[TaskNode]:
        """Find the longest dependency chain (critical path)."""
        effort_map = {"small": 1, "medium": 3, "large": 6, "xlarge": 10}
        task_map = {t.task_id: t for t in tasks}
        longest = []

        def dfs(task: TaskNode, path: list[TaskNode]):
            nonlocal longest
            path = path + [task]
            if not task.dependents:
                if sum(effort_map.get(t.effort, 3) for t in path) > \
                   sum(effort_map.get(t.effort, 3) for t in longest):
                    longest = path[:]
            for dep_id in task.dependents:
                dep = task_map.get(dep_id)
                if dep and dep not in path:
                    dfs(dep, path)

        roots = [t for t in tasks if not t.dependencies]
        for root in roots:
            dfs(root, [])

        return longest

    def _generate_rollback_plan(self, tasks: list[TaskNode]) -> str:
        """Generate a rollback plan based on task risk."""
        high_risk = [t for t in tasks if t.risks]
        if not high_risk:
            return "Standard git revert — low-risk changes"
        return (f"Revert in reverse phase order. "
                f"{len(high_risk)} task(s) have identified risks. "
                f"Test rollback in staging before production.")

    def _build_templates(self) -> list[dict]:
        """Pre-built task templates for common objectives."""
        return [
            {
                "keywords": ["oauth", "authentication", "login", "auth"],
                "tasks": [
                    {"title": "Design auth flow and token strategy", "priority": "high",
                     "effort": "medium", "phase": 1},
                    {"title": "Implement OAuth2 provider registration", "priority": "high",
                     "effort": "medium", "phase": 2},
                    {"title": "Build token validation middleware", "priority": "critical",
                     "effort": "medium", "phase": 2},
                    {"title": "Add session management", "priority": "high",
                     "effort": "medium", "phase": 2},
                    {"title": "Write auth integration tests", "priority": "high",
                     "effort": "medium", "phase": 3, "risks": ["Breaking existing sessions"]},
                    {"title": "Security review of auth implementation", "priority": "critical",
                     "effort": "small", "phase": 3},
                ],
            },
            {
                "keywords": ["database", "migration", "schema", "model"],
                "tasks": [
                    {"title": "Design schema changes", "priority": "high", "effort": "medium", "phase": 1},
                    {"title": "Write migration script", "priority": "critical", "effort": "medium", "phase": 2,
                     "risks": ["Data loss if migration fails"]},
                    {"title": "Update ORM models", "priority": "high", "effort": "medium", "phase": 2},
                    {"title": "Test migration rollback", "priority": "critical", "effort": "small", "phase": 3},
                    {"title": "Run migration in staging", "priority": "critical", "effort": "small", "phase": 4},
                ],
            },
            {
                "keywords": ["api", "endpoint", "rest", "route"],
                "tasks": [
                    {"title": "Design API contract (OpenAPI spec)", "priority": "high",
                     "effort": "medium", "phase": 1},
                    {"title": "Implement endpoint handlers", "priority": "high",
                     "effort": "large", "phase": 2},
                    {"title": "Add input validation & error handling", "priority": "critical",
                     "effort": "medium", "phase": 2},
                    {"title": "Write API tests", "priority": "high", "effort": "medium", "phase": 3},
                    {"title": "Update API documentation", "priority": "medium", "effort": "small", "phase": 3},
                ],
            },
            {
                "keywords": ["refactor", "cleanup", "restructure", "rewrite"],
                "tasks": [
                    {"title": "Identify refactoring targets and dependencies", "priority": "high",
                     "effort": "medium", "phase": 1},
                    {"title": "Write characterization tests for existing behavior", "priority": "critical",
                     "effort": "medium", "phase": 1,
                     "risks": ["Missing edge cases in current behavior"]},
                    {"title": "Perform incremental refactoring", "priority": "high",
                     "effort": "large", "phase": 2},
                    {"title": "Verify all existing tests pass", "priority": "critical",
                     "effort": "small", "phase": 3},
                    {"title": "Performance comparison (before/after)", "priority": "medium",
                     "effort": "small", "phase": 3},
                ],
            },
        ]
