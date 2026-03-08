---
description: Saturday project scan — analyzes project structure and populates the Hierarchical Code Graph Engine
---

# Project Scan Workflow

Run this workflow when Saturday needs to understand a new or updated project.

## Steps

1. **Scan Project Root**
   // turbo
   - List root structure: `dir /B /AD 2>nul && dir /B /A-D 2>nul || ls -la 2>/dev/null`
   - Identify entry points (main.py, index.js, App.java, main.go, etc.)
   - Identify config files (package.json, requirements.txt, Cargo.toml, pom.xml, etc.)

2. **Detect Technology Stack**
   - Read package.json / requirements.txt / Cargo.toml / pom.xml / go.mod
   - Record languages, frameworks, and key dependencies
   - Update `saturday_memory.md` → Technology Stack section

3. **Map Architecture Pattern**
   - Analyze directory structure to identify:
     - MVC (controllers/, models/, views/)
     - Clean/Hexagonal (domain/, ports/, adapters/)
     - Microservices (services/, api/, gateway/)
     - Monolith (src/, lib/, app/)
   - Record the pattern in `saturday_project_graph.md` → Level 1

4. **Build Dependency Graph**
   - For each key module, identify imports/dependencies
   - Map which modules depend on which
   - Flag any circular dependencies
   - Record in `saturday_project_graph.md` → Level 1

5. **Index Key Files**
   - For critical files (entry points, core logic, API routes):
     - Record file path, type, key functions/classes, line count
   - Record in `saturday_project_graph.md` → Level 2

6. **Build Semantic Index**
   - Map business concepts to code locations
   - Example: "user authentication" → `auth/login_service.py:authenticate()`
   - Record in `saturday_project_graph.md` → Level 3

7. **Discover Conventions**
   - Naming style (camelCase vs snake_case vs PascalCase)
   - File naming conventions
   - Error handling patterns
   - Import organization style
   - Comment/documentation style
   - Record in `saturday_memory.md` → Project Conventions section

8. **Summary**
   - Print a brief project overview
   - Flag any architectural concerns or code smells found
   - Mark `saturday_project_graph.md` as populated
