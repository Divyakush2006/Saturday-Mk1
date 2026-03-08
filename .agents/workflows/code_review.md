---
description: Saturday code review — enterprise-grade review with security, quality, and architecture checks
---

# Code Review Workflow

Run this workflow for Saturday to perform a comprehensive code review on changed files.

## Steps

1. **Identify changed files**
   // turbo
   - Get list of modified files: `git diff --name-only HEAD~1 2>&1 || git status --porcelain 2>&1 || echo "Not a git repo — specify files manually"`

2. **Architecture Alignment Check**
   - Read `saturday_project_graph.md` to understand the project structure
   - For each changed file, verify:
     - Is it in the correct directory for its purpose?
     - Does it follow established naming conventions?
     - Does it respect the dependency graph (no circular deps)?
     - Does it follow the project's architecture pattern?

3. **Code Quality Review**
   - For each changed file, check:
     - Error handling: Are all failure paths handled with typed exceptions?
     - Type safety: Are type annotations/hints present?
     - Documentation: Do public functions have docstrings?
     - DRY: Is there duplicated logic that should be extracted?
     - SOLID: Do classes have single responsibility?
     - Performance: Any O(n²) algorithms or N+1 query risks?

4. **Security Review**
   - Run through the OWASP Top 10 checklist from `saturday_security.md`
   - Check for hardcoded secrets
   - Verify input validation on all external inputs
   - Check authentication/authorization on endpoints

5. **Test Coverage Assessment**
   - Are there tests for the changed code?
   - Do tests cover happy path, edge cases, and error cases?
   - Are test names descriptive?

6. **Generate Review Report**
   - Output format:

   ```
   # 📋 Saturday Code Review

   ## Overall Assessment: ✅ APPROVED / ⚠️ NEEDS CHANGES / ❌ BLOCKED

   ## Files Reviewed
   | File | Quality | Security | Architecture | Tests |
   |---|---|---|---|---|
   | file.py | ✅ | ✅ | ⚠️ | ❌ |

   ## Findings
   ### Must Fix (Blocking)
   - [description + recommendation]

   ### Should Fix (Non-blocking)
   - [description + recommendation]

   ### Nice to Have
   - [description + recommendation]

   ## Quality Score
   📊 Maintainability: X/10 | Readability: X/10 | Security: X/10
   ```
