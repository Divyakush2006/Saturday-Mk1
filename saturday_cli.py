"""
Saturday MK1 — Command Line Interface
=======================================
CLI tool for interacting with Saturday's AI coding engine.

Usage:
    python saturday_cli.py generate "Build a REST API for user management" --lang python
    python saturday_cli.py validate path/to/file.py
    python saturday_cli.py scan ./project_dir
    python saturday_cli.py chat "How should I structure the auth module?"
    python saturday_cli.py server --port 8000
    python saturday_cli.py health
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure brain package is importable
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("saturday-cli")


def cmd_generate(args):
    """Generate code from a task description."""
    from brain.saturday_core import Saturday

    saturday = Saturday(project_root=args.project)

    print(f"\n{'='*60}")
    print(f"  SATURDAY MK1 — Code Generation")
    print(f"{'='*60}")
    print(f"  Task:     {args.task}")
    print(f"  Language: {args.lang}")
    print(f"{'='*60}\n")

    result = saturday.generate(
        task=args.task,
        language=args.lang,
        validate=not args.no_validate,
    )

    # Output code
    print(f"```{result.language}")
    print(result.code)
    print("```\n")

    # Validation results
    if result.validation:
        v = result.validation
        status = "PASSED" if v.passed else "FAILED"
        print(f"  Validation: {status}")
        print(f"  Security:   {v.security_score:.0%}")
        print(f"  Quality:    {v.quality_grade} ({v.quality_score:.0%})")
        if v.findings:
            print(f"  Findings:   {len(v.findings)}")

    print(f"\n  Model:    {result.model}")
    print(f"  Tokens:   {result.tokens_used}")
    print(f"  Latency:  {result.latency_seconds:.1f}s")
    print(f"{'='*60}\n")


def cmd_validate(args):
    """Validate a code file for security and quality issues."""
    from brain.saturday_core import Saturday

    filepath = Path(args.file)
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    code = filepath.read_text(encoding="utf-8")
    language = filepath.suffix.lstrip(".")
    lang_map = {"py": "python", "js": "javascript", "ts": "typescript", "rb": "ruby",
                "go": "go", "rs": "rust", "java": "java", "cs": "csharp", "cpp": "cpp"}
    language = lang_map.get(language, language)

    saturday = Saturday(project_root=args.project)
    result = saturday.validate_code(code, filepath.name, language)

    print(f"\n{'='*60}")
    print(f"  SATURDAY MK1 — Code Validation")
    print(f"{'='*60}")
    print(f"  File:     {filepath.name}")
    print(f"  Language: {language}")
    print(f"  Lines:    {len(code.splitlines())}")
    print(f"  Status:   {'PASSED' if result.passed else 'FAILED'}")
    print(f"  Security: {result.security_score:.0%}")
    print(f"  Quality:  {result.quality_grade} ({result.quality_score:.0%})")
    print(f"{'='*60}")

    if result.findings:
        print(f"\n  Findings ({len(result.findings)}):")
        for i, f in enumerate(result.findings[:20], 1):
            sev = getattr(f, 'severity', 'medium')
            msg = getattr(f, 'message', str(f))
            line = getattr(f, 'line', '?')
            print(f"    {i}. [{sev.upper()}] L{line}: {msg}")

    print()
    sys.exit(0 if result.passed else 1)


def cmd_scan(args):
    """Scan a project directory."""
    from brain.saturday_core import Saturday

    saturday = Saturday(project_root=args.project)
    result = saturday.scan_project()

    print(f"\n{'='*60}")
    print(f"  SATURDAY MK1 — Project Scan")
    print(f"{'='*60}")
    print(json.dumps(result, indent=2, default=str))
    print(f"{'='*60}\n")


def cmd_chat(args):
    """Chat with Saturday."""
    from brain.saturday_core import Saturday

    saturday = Saturday(project_root=args.project)
    result = saturday.chat(message=args.message)

    print(f"\n{result['response']}\n")
    print(f"  --- Model: {result['model']} | Tokens: {result['tokens_used']} | {result['latency_seconds']:.1f}s ---\n")


def cmd_server(args):
    """Start the API server."""
    from saturday_server import create_app
    import uvicorn

    app = create_app(project_root=args.project, api_key=args.api_key)

    print(f"\n{'='*60}")
    print(f"  SATURDAY MK1 — API Server")
    print(f"  http://{args.host}:{args.port}")
    print(f"  Docs: http://localhost:{args.port}/docs")
    print(f"{'='*60}\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def cmd_health(args):
    """Show system health."""
    from brain.saturday_core import Saturday

    saturday = Saturday(project_root=args.project)
    health = saturday.health()

    print(f"\n{'='*60}")
    print(f"  SATURDAY MK1 — System Health")
    print(f"{'='*60}")
    print(f"  Version:  {health['version']}")
    print(f"  Project:  {health['project_root']}")
    print(f"  Engines:")
    for name, loaded in health['engines'].items():
        status = "loaded" if loaded else "standby"
        print(f"    - {name}: {status}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        prog="saturday",
        description="Saturday MK1 — Enterprise AI Coding Engine CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Set SATURDAY_API_KEY and SATURDAY_MODEL environment variables to configure the LLM.",
    )
    parser.add_argument("--project", default=".", help="Project root directory")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # generate
    gen = subparsers.add_parser("generate", help="Generate code from a task description")
    gen.add_argument("task", help="What the code should do")
    gen.add_argument("--lang", default="python", help="Target language")
    gen.add_argument("--no-validate", action="store_true", help="Skip security validation")
    gen.set_defaults(func=cmd_generate)

    # validate
    val = subparsers.add_parser("validate", help="Validate a code file")
    val.add_argument("file", help="Path to the file to validate")
    val.set_defaults(func=cmd_validate)

    # scan
    sc = subparsers.add_parser("scan", help="Scan a project directory")
    sc.set_defaults(func=cmd_scan)

    # chat
    ch = subparsers.add_parser("chat", help="Chat with Saturday MK1")
    ch.add_argument("message", help="Your message")
    ch.set_defaults(func=cmd_chat)

    # server
    srv = subparsers.add_parser("server", help="Start the API server")
    srv.add_argument("--host", default="0.0.0.0", help="Bind host")
    srv.add_argument("--port", type=int, default=8000, help="Bind port")
    srv.add_argument("--api-key", default=None, help="API key for authentication")
    srv.set_defaults(func=cmd_server)

    # health
    hl = subparsers.add_parser("health", help="Show system health")
    hl.set_defaults(func=cmd_health)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
