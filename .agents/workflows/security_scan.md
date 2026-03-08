---
description: Saturday security scan — validates code against OWASP Top 10, checks dependencies, detects secrets
---

# Security Scan Workflow

Run this workflow when you want Saturday to perform a comprehensive security review of your code.

## Steps

1. **Identify the target scope**
   - Determine which files or directories to scan
   - If no scope is specified, scan the entire project root

2. **Static Analysis (SAST)**
   - Review all source files for OWASP Top 10 vulnerabilities
   - Check for injection risks (SQL, NoSQL, OS command, XSS)
   - Verify authentication and authorization patterns
   - Flag insecure cryptographic usage

3. **Dependency Audit**
   // turbo
   - For Python projects: `pip audit --format json 2>&1 || echo "pip audit not installed — install with: pip install pip-audit"`
   // turbo
   - For Node.js projects: `npm audit --json 2>&1 || echo "No package-lock.json found"`
   // turbo
   - For Rust projects: `cargo audit 2>&1 || echo "cargo-audit not installed"`

4. **Secrets Detection**
   // turbo
   - Scan for hardcoded secrets: `findstr /S /I /M "password\|secret\|api_key\|token\|private_key" *.py *.js *.ts *.java *.go *.rs *.env 2>nul || echo "No secrets patterns found"`
   - Review any matches and classify as true positives or false positives

5. **Compliance Review**
   - Check if code handles PII → flag GDPR requirements
   - Check if code handles health data → flag HIPAA requirements
   - Check if code handles payment data → flag PCI-DSS requirements

6. **Generate Security Report**
   - Create a markdown report with findings
   - Categorize by severity: CRITICAL / HIGH / MEDIUM / LOW / INFO
   - Include remediation steps for each finding
   - Output format:

   ```
   # 🔒 Saturday Security Scan Report
   
   ## Summary
   - Files Scanned: [count]
   - Critical: [count] | High: [count] | Medium: [count] | Low: [count]
   
   ## Findings
   ### [SEVERITY] Finding Title
   - **File**: path/to/file.py:42
   - **Description**: What was found
   - **Risk**: What could happen
   - **Fix**: How to fix it
   ```
