---
description: Saturday incident response — diagnose production errors and generate fix PRs automatically
---

# Incident Response Workflow

Run this workflow when a production error needs rapid diagnosis and fixing.

## Steps

1. **Collect Error Context**
   - Ask the user for:
     - Error message / stack trace
     - Which service/component is affected
     - When it started happening
     - Any recent deployments or changes

2. **Error Classification**
   - Classify the error:
     - **Runtime Exception**: Null pointer, type error, out of bounds
     - **Infrastructure**: Connection timeout, OOM, disk full
     - **Logic Bug**: Incorrect output, wrong calculation
     - **Security Incident**: Unauthorized access, data breach indicators
     - **Performance**: Slow queries, memory leak, CPU spike

3. **Root Cause Analysis**
   - Trace the error from the stack trace to the source code
   - Identify the specific file, function, and line causing the issue
   - Check recent git history for related changes:
   // turbo
   - `git log --oneline -10 2>&1 || echo "Not a git repo"`
   - Determine if this is a regression, new bug, or edge case

4. **Impact Assessment**
   - How many users/systems are affected?
   - Is data integrity compromised?
   - Is there a security implication?
   - What is the business criticality?

5. **Generate Fix**
   - Write the minimal, targeted fix for the issue
   - Include error handling for the edge case that caused it
   - Add a regression test that would have caught this
   - Follow all Saturday security and quality standards

6. **Validate Fix**
   - Run existing tests to ensure no regressions
   - Run the new regression test
   - Security scan the fix

7. **Generate Incident Report**
   - Output format:

   ```
   # 🚨 Incident Response Report

   ## Summary
   - **Error**: [error message]
   - **Severity**: CRITICAL / HIGH / MEDIUM / LOW
   - **Impact**: [scope of impact]
   - **Root Cause**: [explanation]
   - **Time to Diagnosis**: [duration]

   ## Root Cause
   [Detailed explanation of what went wrong and why]

   ## Fix Applied
   [Description of the fix + code diff]

   ## Regression Test Added
   [Test code that would have caught this]

   ## Prevention Recommendations
   1. [How to prevent similar issues]
   2. [Monitoring/alerting improvements]
   3. [Code review checklist additions]
   ```
