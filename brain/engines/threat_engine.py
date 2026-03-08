"""
Saturday Threat Engine — Red Team Simulation & Attack Surface Analysis
======================================================================
Saturday thinks like the world's top 1% penetration tester. This engine
simulates attacks, builds exploit chains, and finds every entry point.

Triple-Framework Analysis:
  STRIDE: Threat classification (Spoofing, Tampering, Repudiation,
          Information Disclosure, Denial of Service, Elevation of Privilege)
  DREAD:  Risk scoring (Damage, Reproducibility, Exploitability,
          Affected Users, Discoverability) — 0-10 per factor
  MITRE ATT&CK: Real-world adversary technique mapping

Key Innovations:
  - Attack chain simulation: multi-step exploit paths
  - Penetration test narrative: step-by-step attack walkthrough
  - Kill chain mapping: Lockheed Martin Cyber Kill Chain stages
  - Zero-day pattern detection: code patterns matching historical exploits
  - Data breach impact assessment: regulatory and financial impact
  - Professional red team report generation

Why Saturday dominates:
  - Claude Opus 4.6: Zero threat modeling capability
  - Antigravity: No offensive security simulation
  - Saturday: Full red team simulation with triple-framework analysis,
    attack chain construction, and data breach impact assessment
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


# ══════════════════════════════════════════════════════════════
#  DATA CLASSES
# ══════════════════════════════════════════════════════════════

STRIDE_CATEGORIES = {
    "S": "Spoofing",
    "T": "Tampering",
    "R": "Repudiation",
    "I": "Information Disclosure",
    "D": "Denial of Service",
    "E": "Elevation of Privilege",
}

KILL_CHAIN_STAGES = [
    "reconnaissance", "weaponization", "delivery",
    "exploitation", "installation", "command_control", "actions_on_objective",
]

MITRE_TACTICS = {
    "TA0001": "Initial Access",
    "TA0002": "Execution",
    "TA0003": "Persistence",
    "TA0004": "Privilege Escalation",
    "TA0005": "Defense Evasion",
    "TA0006": "Credential Access",
    "TA0007": "Discovery",
    "TA0008": "Lateral Movement",
    "TA0009": "Collection",
    "TA0010": "Exfiltration",
    "TA0011": "Command and Control",
    "TA0040": "Impact",
}


@dataclass
class DREADScore:
    """DREAD risk scoring model — each factor scored 0-10."""
    damage: int = 0            # How bad if exploited?
    reproducibility: int = 0   # How easy to reproduce?
    exploitability: int = 0    # How easy to exploit?
    affected_users: int = 0    # How many users affected?
    discoverability: int = 0   # How easy to discover?

    @property
    def total(self) -> float:
        """Average DREAD score (0-10)."""
        return round((self.damage + self.reproducibility + self.exploitability +
                      self.affected_users + self.discoverability) / 5, 1)

    @property
    def risk_level(self) -> str:
        t = self.total
        if t >= 8:
            return "critical"
        elif t >= 6:
            return "high"
        elif t >= 4:
            return "medium"
        else:
            return "low"


@dataclass
class ThreatVector:
    """A potential attack path identified in the system."""
    vector_id: str
    name: str
    stride_category: str     # S, T, R, I, D, E
    description: str
    attack_complexity: str   # low, medium, high
    likelihood: str          # certain, likely, possible, unlikely
    impact: str              # critical, high, medium, low
    dread: DREADScore = field(default_factory=DREADScore)
    mitre_technique: str = ""      # e.g., "T1059" (Command Scripting)
    mitre_tactic: str = ""         # e.g., "TA0002" (Execution)
    kill_chain_stage: str = ""     # which kill chain stage
    prerequisites: list[str] = field(default_factory=list)
    mitigations: list[str] = field(default_factory=list)
    attack_narrative: str = ""     # Step-by-step attack walkthrough


@dataclass
class Vulnerability:
    """A specific vulnerability found in the code."""
    vuln_id: str
    cve_pattern: str
    description: str
    severity: str            # critical, high, medium, low
    location: str = ""
    affected_data: list[str] = field(default_factory=list)
    remediation: str = ""
    cvss_estimate: float = 0.0
    exploit_available: bool = False
    exploit_difficulty: str = "medium"


@dataclass
class AttackChain:
    """A multi-step exploit path from entry to objective."""
    chain_id: str
    name: str
    steps: list[dict] = field(default_factory=list)  # [{stage, action, technique, target}]
    entry_point: str = ""
    objective: str = ""        # data_exfil, privilege_escalation, dos, etc.
    success_probability: float = 0.0
    total_dread_score: float = 0.0
    mitigations: list[str] = field(default_factory=list)


@dataclass
class DataBreachImpact:
    """Assessment of potential data breach impact."""
    records_at_risk: str = "unknown"
    pii_types: list[str] = field(default_factory=list)
    regulatory_frameworks: list[str] = field(default_factory=list)
    estimated_fine_range: str = ""
    notification_required: bool = False
    time_to_detect: str = ""
    reputational_impact: str = ""


@dataclass
class ThreatReport:
    """Complete red team threat analysis report."""
    report_id: str
    analyzed_at: str
    vectors: list[ThreatVector] = field(default_factory=list)
    vulnerabilities: list[Vulnerability] = field(default_factory=list)
    attack_chains: list[AttackChain] = field(default_factory=list)
    breach_impact: Optional[DataBreachImpact] = None
    overall_risk: str = "low"
    risk_score: float = 0.0
    recommendations: list[str] = field(default_factory=list)
    compliance_gaps: list[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════
#  THREAT ENGINE — RED TEAM SIMULATION
# ══════════════════════════════════════════════════════════════

class ThreatEngine:
    """
    Red Team Simulation Engine — thinks like the top 1% hacker.

    Usage:
        te = ThreatEngine()
        report = te.analyze(code, context="REST API endpoint for user login")
        print(report.overall_risk)
        print(te.generate_report(report))
    """

    def __init__(self):
        self._vector_counter = 0
        self._vuln_counter = 0
        self._chain_counter = 0

    def analyze(self, code: str, context: str = "") -> ThreatReport:
        """
        Full red team analysis: identify threats, build attack chains,
        assess breach impact, and generate recommendations.
        """
        report_id = f"threat_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        vectors = self._identify_vectors(code, context)
        vulns = self._identify_vulnerabilities(code)
        chains = self._build_attack_chains(vectors, vulns)
        breach = self._assess_breach_impact(code, vulns)
        risk = self._calculate_risk(vectors, vulns, chains)
        recommendations = self._generate_recommendations(vectors, vulns, chains)
        compliance = self._check_compliance(code)

        overall = "critical" if risk >= 80 else "high" if risk >= 60 else \
                  "medium" if risk >= 40 else "low"

        return ThreatReport(
            report_id=report_id,
            analyzed_at=datetime.now(timezone.utc).isoformat(),
            vectors=vectors,
            vulnerabilities=vulns,
            attack_chains=chains,
            breach_impact=breach,
            overall_risk=overall,
            risk_score=risk,
            recommendations=recommendations,
            compliance_gaps=compliance,
        )

    def generate_report(self, report: ThreatReport) -> str:
        """Generate professional red team penetration test report."""
        lines = [
            "🗡️ SATURDAY RED TEAM THREAT REPORT",
            "=" * 55,
            f"   Report ID:    {report.report_id}",
            f"   Analyzed at:  {report.analyzed_at[:19]}",
            f"   Overall Risk: {report.overall_risk.upper()} ({report.risk_score:.0f}/100)",
            f"   Vectors:      {len(report.vectors)}",
            f"   Vulns:        {len(report.vulnerabilities)}",
            f"   Attack Chains: {len(report.attack_chains)}",
            "",
        ]

        # STRIDE Summary
        if report.vectors:
            lines.append("  ── STRIDE Threat Vectors ──")
            for v in sorted(report.vectors, key=lambda x: -x.dread.total):
                icon = {"S": "🎭", "T": "✏️", "R": "🚫", "I": "📤",
                        "D": "💥", "E": "👑"}.get(v.stride_category, "⚠️")
                cat = STRIDE_CATEGORIES.get(v.stride_category, v.stride_category)
                lines.append(f"    {icon} [{cat}] {v.name}")
                lines.append(f"       DREAD: {v.dread.total}/10 "
                           f"(D:{v.dread.damage} R:{v.dread.reproducibility} "
                           f"E:{v.dread.exploitability} A:{v.dread.affected_users} "
                           f"D:{v.dread.discoverability})")
                if v.mitre_technique:
                    tactic = MITRE_TACTICS.get(v.mitre_tactic, v.mitre_tactic)
                    lines.append(f"       ATT&CK: {v.mitre_technique} ({tactic})")
                if v.kill_chain_stage:
                    lines.append(f"       Kill Chain: {v.kill_chain_stage}")
                if v.attack_narrative:
                    lines.append(f"       Attack: {v.attack_narrative[:150]}")
                lines.append("")

        # Attack Chains
        if report.attack_chains:
            lines.append("  ── Attack Chains (Multi-Step Exploits) ──")
            for chain in report.attack_chains:
                lines.append(f"    🔗 {chain.name} (DREAD: {chain.total_dread_score:.1f}/10)")
                lines.append(f"       Entry: {chain.entry_point}")
                lines.append(f"       Objective: {chain.objective}")
                lines.append(f"       Success Rate: {chain.success_probability:.0%}")
                for j, step in enumerate(chain.steps, 1):
                    lines.append(f"       Step {j}: [{step.get('stage', '?')}] {step.get('action', '?')}")
                lines.append("")

        # Data Breach Impact
        if report.breach_impact:
            bi = report.breach_impact
            lines.append("  ── Data Breach Impact Assessment ──")
            lines.append(f"    Records at Risk:  {bi.records_at_risk}")
            if bi.pii_types:
                lines.append(f"    PII Types:        {', '.join(bi.pii_types)}")
            if bi.regulatory_frameworks:
                lines.append(f"    Regulations:      {', '.join(bi.regulatory_frameworks)}")
            if bi.estimated_fine_range:
                lines.append(f"    Estimated Fines:  {bi.estimated_fine_range}")
            lines.append(f"    Notif Required:   {'Yes' if bi.notification_required else 'No'}")
            lines.append("")

        # Recommendations
        if report.recommendations:
            lines.append("  ── Prioritized Recommendations ──")
            for i, rec in enumerate(report.recommendations, 1):
                lines.append(f"    {i}. {rec}")
            lines.append("")

        # Compliance Gaps
        if report.compliance_gaps:
            lines.append("  ── Compliance Gaps ──")
            for gap in report.compliance_gaps:
                lines.append(f"    ⚠️ {gap}")
            lines.append("")

        return "\n".join(lines)

    # ── STRIDE Vector Identification ──

    def _identify_vectors(self, code: str, context: str) -> list[ThreatVector]:
        vectors = []
        combined = f"{code}\n{context}".lower()

        stride_patterns = [
            # Spoofing
            {"stride": "S", "name": "Authentication Bypass",
             "patterns": [r"(?:login|auth|session).*(?:skip|bypass|disable|mock)",
                         r"(?:token|jwt).*(?:none|null|empty|debug)"],
             "dread": DREADScore(9, 8, 7, 10, 6),
             "mitre": "T1078", "tactic": "TA0001",
             "kill_chain": "exploitation",
             "narrative": "Attacker exploits weak/missing authentication to impersonate legitimate users",
             "mitigations": ["Implement MFA", "Use OAuth2/OIDC", "Session token rotation"]},
            {"stride": "S", "name": "Identity Spoofing via Missing Verification",
             "patterns": [r"(?:user_id|email)\s*=\s*(?:request|args|params)",
                         r"(?:from_user|sender)\s*=\s*(?:request|header)"],
             "dread": DREADScore(8, 9, 7, 9, 5),
             "mitre": "T1134", "tactic": "TA0004",
             "kill_chain": "exploitation",
             "narrative": "Attacker modifies user_id in request to access another user's data (IDOR)",
             "mitigations": ["Validate user ownership server-side", "Use session-bound user IDs"]},

            # Tampering
            {"stride": "T", "name": "Input Validation Bypass",
             "patterns": [r"(?:request|body).*(?:direct|raw|unvalidated)",
                         r"(?:update|modify|write).*(?:without|no).*(?:check|valid)"],
             "dread": DREADScore(8, 8, 6, 8, 5),
             "mitre": "T1565", "tactic": "TA0040",
             "kill_chain": "exploitation",
             "narrative": "Attacker sends crafted input to modify data integrity",
             "mitigations": ["Input validation at every boundary", "Immutable audit logs"]},
            {"stride": "T", "name": "Mass Assignment Attack",
             "patterns": [r"\*\*(?:request|kwargs|body)", r"\.update\s*\(\s*request"],
             "dread": DREADScore(8, 9, 8, 7, 6),
             "mitre": "T1565.001", "tactic": "TA0040",
             "kill_chain": "exploitation",
             "narrative": "Attacker adds extra fields (role=admin) to update requests",
             "mitigations": ["Explicit field whitelisting", "Parameter schemas"]},

            # Repudiation
            {"stride": "R", "name": "Insufficient Logging",
             "patterns": [r"except.*(?:pass|continue)\s*$", r"(?:no|missing|without).*(?:log|audit)"],
             "dread": DREADScore(6, 10, 9, 8, 3),
             "mitre": "T1070", "tactic": "TA0005",
             "kill_chain": "actions_on_objective",
             "narrative": "Attacker performs actions with no audit trail — denies involvement",
             "mitigations": ["Log all security events", "Tamper-proof audit trail", "SIEM integration"]},

            # Information Disclosure
            {"stride": "I", "name": "Sensitive Data Exposure",
             "patterns": [r"(?:password|secret|token|key|ssn|credit.?card).*(?:response|return|json|print)",
                         r"(?:stacktrace|traceback|debug).*(?:true|enable|show)"],
             "dread": DREADScore(9, 8, 5, 10, 7),
             "mitre": "T1005", "tactic": "TA0009",
             "kill_chain": "actions_on_objective",
             "narrative": "Attacker extracts PII, credentials, or internal data from API responses",
             "mitigations": ["Response field filtering", "Error message sanitization", "Data classification"]},
            {"stride": "I", "name": "Error-Based Information Leakage",
             "patterns": [r"(?:debug\s*=\s*true|verbose.*error|stack.*trace)",
                         r"(?:app\.debug|DEBUG\s*=\s*True)"],
             "dread": DREADScore(5, 10, 9, 8, 9),
             "mitre": "T1592", "tactic": "TA0001",
             "kill_chain": "reconnaissance",
             "narrative": "Attacker triggers errors to expose internal paths, versions, and config",
             "mitigations": ["Custom error pages", "Disable debug in production"]},

            # Denial of Service
            {"stride": "D", "name": "Resource Exhaustion",
             "patterns": [r"(?:while\s+True|for.*range\s*\(\s*\d{6,})",
                         r"\.(?:all|find)\s*\(\s*\).*(?:no|without).*(?:limit|page)"],
             "dread": DREADScore(7, 9, 8, 10, 8),
             "mitre": "T1499", "tactic": "TA0040",
             "kill_chain": "actions_on_objective",
             "narrative": "Attacker sends requests triggering unbounded operations to crash the service",
             "mitigations": ["Query pagination", "Rate limiting", "Request timeouts", "Circuit breakers"]},

            # Elevation of Privilege
            {"stride": "E", "name": "Privilege Escalation via Role Manipulation",
             "patterns": [r"(?:role|is_admin|permission|privilege).*(?:=\s*(?:request|args|params|body))",
                         r"(?:admin|superuser|root).*(?:true|1|yes).*(?:request|input)"],
             "dread": DREADScore(10, 8, 7, 10, 5),
             "mitre": "T1548", "tactic": "TA0004",
             "kill_chain": "exploitation",
             "narrative": "Attacker modifies role/permission fields to gain admin access",
             "mitigations": ["Server-side role assignment only", "RBAC with least privilege"]},
            {"stride": "E", "name": "Path Traversal to System Files",
             "patterns": [r"open\s*\(.*(?:request|user|args)", r"(?:\.\./|\.\.\\\\)"],
             "dread": DREADScore(9, 9, 7, 7, 7),
             "mitre": "T1083", "tactic": "TA0007",
             "kill_chain": "exploitation",
             "narrative": "Attacker uses ../../etc/passwd to read system files",
             "mitigations": ["Path validation with realpath", "Chroot/sandbox", "Input sanitization"]},
        ]

        for sdata in stride_patterns:
            for pattern in sdata["patterns"]:
                if re.search(pattern, combined):
                    self._vector_counter += 1
                    vectors.append(ThreatVector(
                        vector_id=f"TV-{self._vector_counter:04d}",
                        name=sdata["name"],
                        stride_category=sdata["stride"],
                        description=sdata["narrative"],
                        attack_complexity="medium",
                        likelihood="likely" if sdata["dread"].total >= 7 else "possible",
                        impact="critical" if sdata["dread"].damage >= 8 else "high",
                        dread=sdata["dread"],
                        mitre_technique=sdata.get("mitre", ""),
                        mitre_tactic=sdata.get("tactic", ""),
                        kill_chain_stage=sdata.get("kill_chain", ""),
                        mitigations=sdata.get("mitigations", []),
                        attack_narrative=sdata["narrative"],
                    ))
                    break  # One match per pattern group
        return vectors

    # ── Vulnerability Detection ──

    def _identify_vulnerabilities(self, code: str) -> list[Vulnerability]:
        vulns = []

        vuln_patterns = [
            (r"(?:execute|query)\s*\(.*(?:f['\"]|\+|\.format)", "SQL Injection",
             "Parameterize all database queries", "CWE-89", 9.8),
            (r"(?:eval|exec)\s*\(.*(?:request|input|args|user)", "Remote Code Execution",
             "Never evaluate user-controlled input", "CWE-94", 10.0),
            (r"pickle\.loads?\s*\(", "Insecure Deserialization",
             "Use JSON for untrusted data", "CWE-502", 9.8),
            (r"(?:os\.system|subprocess.*shell\s*=\s*True)", "Command Injection",
             "Use subprocess.run with shell=False", "CWE-78", 9.8),
            (r"(?:password|secret)\s*=\s*['\"][^'\"]+['\"]", "Hardcoded Credentials",
             "Use environment variables or secret manager", "CWE-798", 9.1),
            (r"(?:innerHTML|document\.write)\s*=?\s*\(?\s*.*(?:user|input|request)",
             "Cross-Site Scripting", "Sanitize and encode all output", "CWE-79", 6.1),
            (r"(?:CORS|cors).*\*", "Overly Permissive CORS",
             "Restrict CORS to trusted origins only", "CWE-942", 5.3),
            (r"(?:MD5|SHA1|DES|RC4)", "Weak Cryptography",
             "Use AES-256-GCM, SHA-256+, bcrypt for passwords", "CWE-327", 7.5),
            (r"verify\s*=\s*False", "TLS Verification Disabled",
             "Enable certificate verification", "CWE-295", 7.4),
            (r"random\.(random|randint)\s*\(.*(?:token|key|secret|password|session)",
             "Insecure Random", "Use secrets module for security tokens", "CWE-330", 5.3),
        ]

        for pattern, name, fix, cve, cvss in vuln_patterns:
            matches = list(re.finditer(pattern, code, re.IGNORECASE))
            for match in matches:
                self._vuln_counter += 1
                line_num = code[:match.start()].count("\n") + 1
                vulns.append(Vulnerability(
                    vuln_id=f"VULN-{self._vuln_counter:04d}",
                    cve_pattern=cve,
                    description=name,
                    severity="critical" if cvss >= 9 else "high" if cvss >= 7 else "medium",
                    location=f"Line {line_num}",
                    remediation=fix,
                    cvss_estimate=cvss,
                    exploit_available=cvss >= 9.0,
                    exploit_difficulty="low" if cvss >= 9 else "medium",
                ))
        return vulns

    # ── Attack Chain Construction ──

    def _build_attack_chains(self, vectors: list[ThreatVector],
                            vulns: list[Vulnerability]) -> list[AttackChain]:
        """Build multi-step exploit chains like a real red team would."""
        chains = []

        # Chain 1: Data Exfiltration
        info_vectors = [v for v in vectors if v.stride_category == "I"]
        sqli_vulns = [v for v in vulns if "SQL" in v.description]
        if info_vectors or sqli_vulns:
            self._chain_counter += 1
            chains.append(AttackChain(
                chain_id=f"CHAIN-{self._chain_counter:04d}",
                name="Data Exfiltration via SQL Injection",
                entry_point="User-facing input field or API parameter",
                objective="Extract sensitive data from database",
                steps=[
                    {"stage": "reconnaissance", "action": "Enumerate API endpoints and parameters",
                     "technique": "T1595", "target": "API surface"},
                    {"stage": "delivery", "action": "Inject SQL payload via input parameter",
                     "technique": "T1190", "target": "Database query"},
                    {"stage": "exploitation", "action": "Extract table schema with UNION SELECT",
                     "technique": "T1005", "target": "Database schema"},
                    {"stage": "actions_on_objective", "action": "Dump sensitive records via blind SQLi",
                     "technique": "T1530", "target": "User PII, credentials"},
                ],
                success_probability=0.8 if sqli_vulns else 0.3,
                total_dread_score=max((v.dread.total for v in info_vectors), default=5),
                mitigations=["Parameterized queries", "WAF rules", "Database monitoring"],
            ))

        # Chain 2: Privilege Escalation
        elev_vectors = [v for v in vectors if v.stride_category == "E"]
        spoof_vectors = [v for v in vectors if v.stride_category == "S"]
        if elev_vectors or spoof_vectors:
            self._chain_counter += 1
            chains.append(AttackChain(
                chain_id=f"CHAIN-{self._chain_counter:04d}",
                name="Privilege Escalation to Admin",
                entry_point="Standard user account or public endpoint",
                objective="Gain administrative access to the application",
                steps=[
                    {"stage": "reconnaissance", "action": "Map user roles and permission model",
                     "technique": "T1087", "target": "Authorization system"},
                    {"stage": "exploitation", "action": "Exploit IDOR or mass assignment to modify role",
                     "technique": "T1548", "target": "User role field"},
                    {"stage": "installation", "action": "Create persistent admin session",
                     "technique": "T1098", "target": "Session management"},
                    {"stage": "actions_on_objective", "action": "Access admin panels and sensitive data",
                     "technique": "T1078", "target": "Admin functionality"},
                ],
                success_probability=0.6 if elev_vectors else 0.2,
                total_dread_score=max((v.dread.total for v in elev_vectors), default=5),
                mitigations=["RBAC with server-side enforcement", "Audit logging", "Session monitoring"],
            ))

        # Chain 3: Remote Code Execution
        rce_vulns = [v for v in vulns if "Code Execution" in v.description or "Command" in v.description]
        if rce_vulns:
            self._chain_counter += 1
            chains.append(AttackChain(
                chain_id=f"CHAIN-{self._chain_counter:04d}",
                name="Remote Code Execution & Server Takeover",
                entry_point="Vulnerable eval()/exec()/os.system() call",
                objective="Full server compromise with arbitrary code execution",
                steps=[
                    {"stage": "reconnaissance", "action": "Identify code injection points",
                     "technique": "T1595", "target": "Input handling"},
                    {"stage": "weaponization", "action": "Craft payload for reverse shell",
                     "technique": "T1059", "target": "Server runtime"},
                    {"stage": "exploitation", "action": "Execute payload via injection point",
                     "technique": "T1203", "target": "Application process"},
                    {"stage": "installation", "action": "Establish persistence via cron/service",
                     "technique": "T1053", "target": "Operating system"},
                    {"stage": "command_control", "action": "Create reverse shell to attacker C2",
                     "technique": "T1571", "target": "Network"},
                    {"stage": "actions_on_objective", "action": "Exfiltrate data, pivot to other systems",
                     "technique": "T1041", "target": "Internal network"},
                ],
                success_probability=0.9,
                total_dread_score=9.5,
                mitigations=["Remove eval/exec", "Sandboxing", "Network segmentation", "EDR"],
            ))

        return chains

    # ── Data Breach Impact Assessment ──

    def _assess_breach_impact(self, code: str, vulns: list[Vulnerability]) -> DataBreachImpact:
        pii_patterns = {
            r"(?:email|e_mail)": "Email addresses",
            r"(?:password|passwd)": "Passwords",
            r"(?:ssn|social_security)": "Social Security Numbers",
            r"(?:credit.?card|card_number|pan)": "Credit card numbers",
            r"(?:phone|mobile|tel)": "Phone numbers",
            r"(?:address|street|zip)": "Physical addresses",
            r"(?:dob|birth|birthdate)": "Dates of birth",
            r"(?:health|medical|diagnosis)": "Medical records",
            r"(?:salary|income|bank)": "Financial data",
        }

        found_pii = []
        for pattern, pii_type in pii_patterns.items():
            if re.search(pattern, code, re.IGNORECASE):
                found_pii.append(pii_type)

        frameworks = []
        if any(p in found_pii for p in ["Credit card numbers"]):
            frameworks.append("PCI-DSS")
        if any(p in found_pii for p in ["Medical records"]):
            frameworks.append("HIPAA")
        if found_pii:
            frameworks.extend(["GDPR", "CCPA", "SOC2"])

        high_severity = len([v for v in vulns if v.severity in ("critical", "high")])

        return DataBreachImpact(
            records_at_risk="high" if high_severity >= 3 else "medium" if high_severity >= 1 else "low",
            pii_types=found_pii,
            regulatory_frameworks=list(set(frameworks)),
            estimated_fine_range="$1M-$10M+" if "PCI-DSS" in frameworks or "HIPAA" in frameworks
                                else "$100K-$1M" if frameworks else "N/A",
            notification_required=bool(found_pii and high_severity > 0),
            time_to_detect="Hours to days without monitoring" if high_severity > 0 else "N/A",
            reputational_impact="severe" if high_severity >= 3 else "moderate" if high_severity >= 1 else "minimal",
        )

    # ── Risk Calculation ──

    def _calculate_risk(self, vectors: list[ThreatVector],
                       vulns: list[Vulnerability],
                       chains: list[AttackChain]) -> float:
        """Calculate overall risk score 0-100 (100 = maximum risk)."""
        score = 0.0
        for v in vectors:
            score += v.dread.total * 2
        for v in vulns:
            score += v.cvss_estimate * 1.5
        for c in chains:
            score += c.success_probability * 15
        return min(100, score)

    # ── Recommendations ──

    def _generate_recommendations(self, vectors: list[ThreatVector],
                                  vulns: list[Vulnerability],
                                  chains: list[AttackChain]) -> list[str]:
        recs = []
        severity_weights = {"critical": 4, "high": 3, "medium": 2, "low": 1}

        # From vulnerabilities
        for v in sorted(vulns, key=lambda x: -x.cvss_estimate)[:5]:
            recs.append(f"[CVSS {v.cvss_estimate:.1f}] {v.description}: {v.remediation}")

        # From attack chains
        for chain in chains:
            if chain.success_probability >= 0.6:
                recs.append(f"[CHAIN] Block '{chain.name}': {'; '.join(chain.mitigations[:3])}")

        # From STRIDE vectors
        for v in sorted(vectors, key=lambda x: -x.dread.total)[:3]:
            if v.mitigations:
                recs.append(f"[{STRIDE_CATEGORIES.get(v.stride_category, '?')}] "
                          f"{v.name}: {v.mitigations[0]}")

        return recs[:10]  # Top 10 recommendations

    # ── Compliance Checking ──

    def _check_compliance(self, code: str) -> list[str]:
        gaps = []
        checks = [
            (r"(?:log|audit|logger)", False, "SOC2: No logging/audit trail detected"),
            (r"(?:encrypt|AES|RSA|bcrypt)", False, "PCI-DSS: No encryption detected for sensitive data"),
            (r"(?:rate_limit|throttle|limiter)", False, "OWASP: No rate limiting detected"),
            (r"(?:csrf|xsrf|CSRFProtect)", False, "OWASP: No CSRF protection detected"),
            (r"(?:helmet|security.headers|Content-Security-Policy)", False, "OWASP: No security headers detected"),
            (r"(?:input.*valid|sanitiz|escape|clean)", False, "ISO 27001: No input validation detected"),
        ]

        for pattern, should_fail, message in checks:
            found = bool(re.search(pattern, code, re.IGNORECASE))
            if not found:
                gaps.append(message)

        return gaps
