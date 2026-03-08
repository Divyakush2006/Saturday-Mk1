"""
Saturday MK1 — REST API Server
================================
Production-grade FastAPI server exposing Saturday's capabilities.

Usage:
    # Start server
    python saturday_server.py --port 8000

    # With custom project root
    python saturday_server.py --project ./my_project --port 8000

Endpoints:
    POST /api/v1/generate        — Generate code from task description
    POST /api/v1/chat            — Multi-turn conversation
    POST /api/v1/validate        — Security + quality validation
    POST /api/v1/scan            — Scan and analyze project
    POST /api/v1/plan            — Strategic execution planning
    GET  /api/v1/health          — Service health check
    POST /api/v1/auth/signup     — Create new user account
    POST /api/v1/auth/login      — Authenticate user
    GET  /api/v1/auth/me         — Get current user info
    GET  /api/v1/conversations   — List user conversations
    POST /api/v1/conversations   — Create conversation
    GET  /api/v1/conversations/{id}/messages — Get messages
    DELETE /api/v1/conversations/{id} — Delete conversation
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure brain package is importable
sys.path.insert(0, str(Path(__file__).parent))

from pydantic import BaseModel, Field

log = logging.getLogger("saturday-server")

# Vercel Hobby free tier has a 10s hard limit on function execution.
# We leave a 1.5s buffer for response serialization and network overhead.
_VERCEL_TIME_BUDGET = float(os.getenv("SATURDAY_TIME_BUDGET", "8.5"))


# ═══════════════════════════════════════════
# REQUEST / RESPONSE MODELS
# ═══════════════════════════════════════════

class GenerateRequest(BaseModel):
    """Request for code generation."""
    task: str = Field(..., description="What the code should do")
    language: str = Field(default="python", description="Target programming language")
    context: str = Field(default="", description="Additional context for generation")
    auto_validate: bool = Field(default=True, description="Auto-validate generated code")
    max_tokens: int = Field(default=4096, ge=100, le=32768)


class GenerateResponse(BaseModel):
    """Response from code generation."""
    code: str
    language: str
    model: str
    tokens_used: int
    latency_seconds: float
    validation_passed: Optional[bool] = None
    security_score: Optional[float] = None
    quality_score: Optional[float] = None
    quality_grade: Optional[str] = None
    findings_count: int = 0


class ChatRequest(BaseModel):
    """Request for conversational AI."""
    message: str = Field(..., description="User message")
    history: list[dict] = Field(default_factory=list, description="Previous messages")
    include_context: bool = Field(default=True, description="Inject project context")
    mode: str = Field(default="fast", description="'fast' for direct LLM call, 'full' for engine-backed analysis")
    conversation_id: Optional[str] = Field(default=None, description="Server conversation ID (for authenticated users)")


class ChatResponse(BaseModel):
    """Response from conversational AI."""
    response: str
    model: str
    tokens_used: int
    latency_seconds: float
    conversation_id: Optional[str] = None


class SignupRequest(BaseModel):
    """User signup request."""
    email: str = Field(..., description="Email address")
    username: str = Field(..., description="Display name")
    password: str = Field(..., min_length=6, description="Password (min 6 chars)")


class LoginRequest(BaseModel):
    """User login request."""
    email: str = Field(..., description="Email address")
    password: str = Field(..., description="Password")


class ValidateRequest(BaseModel):
    """Request for code validation."""
    code: str = Field(..., description="Source code to validate")
    filename: str = Field(default="code.py", description="Filename for language detection")
    language: str = Field(default="python", description="Programming language")


class ValidateResponse(BaseModel):
    """Response from code validation."""
    passed: bool
    security_score: float
    quality_score: float
    quality_grade: str
    findings_count: int
    findings: list[dict] = Field(default_factory=list)


class PlanRequest(BaseModel):
    """Request for strategic planning."""
    task: str = Field(..., description="What needs to be planned")
    context: dict = Field(default_factory=dict, description="Additional context")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    project_root: str
    engines_loaded: dict
    uptime_seconds: float


# ═══════════════════════════════════════════
# SERVER
# ═══════════════════════════════════════════

def create_app(project_root: str = ".", api_key: Optional[str] = None):
    """Create and configure the FastAPI application."""
    from fastapi import FastAPI, HTTPException, Request, Security, Header
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
    from fastapi.security import APIKeyHeader
    from fastapi.staticfiles import StaticFiles

    from brain.saturday_core import Saturday
    from brain.auth import AuthDB, verify_jwt

    # Resolve frontend path
    base_dir = Path(__file__).parent
    frontend_dir = base_dir / "frontend"

    app = FastAPI(
        title="Saturday MK1 API",
        description="Enterprise AI Coding Engine — REST API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize Saturday engine
    saturday = Saturday(project_root=project_root)
    start_time = time.time()

    # Initialize auth database
    auth_db = AuthDB()

    # API Key auth
    expected_key = api_key or os.getenv("SATURDAY_SERVER_KEY", "")
    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    async def verify_key(key: Optional[str] = Security(api_key_header)):
        if expected_key and key != expected_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    def get_user_from_token(authorization: Optional[str] = None) -> Optional[dict]:
        """Extract user payload from Authorization: Bearer <token> header."""
        if not authorization:
            return None
        if not authorization.startswith("Bearer "):
            return None
        token = authorization[7:]
        return verify_jwt(token)

    # ── Endpoints ──

    @app.get("/api/v1/health", response_model=HealthResponse)
    async def health():
        """Service health check — no authentication required."""
        info = saturday.health()
        return HealthResponse(
            status="healthy",
            version=info["version"],
            project_root=info["project_root"],
            engines_loaded=info["engines"],
            uptime_seconds=round(time.time() - start_time, 1),
        )

    @app.post("/api/v1/generate", response_model=GenerateResponse)
    async def generate(req: GenerateRequest, _=Security(verify_key)):
        """Generate production-ready code from a task description."""
        try:
            result = saturday.generate(
                task=req.task,
                language=req.language,
                context=req.context,
                validate=req.auto_validate,
                max_tokens=req.max_tokens,
            )

            resp = GenerateResponse(
                code=result.code,
                language=result.language,
                model=result.model,
                tokens_used=result.tokens_used,
                latency_seconds=result.latency_seconds,
            )

            if result.validation:
                resp.validation_passed = result.validation.passed
                resp.security_score = result.validation.security_score
                resp.quality_score = result.validation.quality_score
                resp.quality_grade = result.validation.quality_grade
                resp.findings_count = len(result.validation.findings)

            return resp

        except ConnectionError as e:
            raise HTTPException(status_code=503, detail=f"LLM provider unavailable: {e}")
        except Exception as e:
            log.exception("Generate failed")
            raise HTTPException(status_code=500, detail=str(e))

    # ── Auth Endpoints ──

    @app.post("/api/v1/auth/signup")
    async def auth_signup(req: SignupRequest):
        """Create a new user account."""
        result = auth_db.signup(req.email, req.username, req.password)
        if not result.success:
            status = 409 if "already taken" in (result.error or "") else 400
            raise HTTPException(status_code=status, detail=result.error)
        return {
            "token": result.token,
            "user": {"id": result.user.id, "email": result.user.email,
                     "username": result.user.username},
        }

    @app.post("/api/v1/auth/login")
    async def auth_login(req: LoginRequest):
        """Authenticate a user."""
        result = auth_db.login(req.email, req.password)
        if not result.success:
            raise HTTPException(status_code=401, detail=result.error)
        return {
            "token": result.token,
            "user": {"id": result.user.id, "email": result.user.email,
                     "username": result.user.username},
        }

    @app.get("/api/v1/auth/me")
    async def auth_me(authorization: Optional[str] = Header(default=None)):
        """Get current user info from JWT token."""
        payload = get_user_from_token(authorization)
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        user = auth_db.get_user(payload["sub"])
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return {"user": {"id": user.id, "email": user.email,
                         "username": user.username}}

    # ── Conversation Endpoints ──

    @app.get("/api/v1/conversations")
    async def list_conversations(authorization: Optional[str] = Header(default=None)):
        """List conversations for authenticated user."""
        payload = get_user_from_token(authorization)
        if not payload:
            raise HTTPException(status_code=401, detail="Authentication required")
        convos = auth_db.list_conversations(payload["sub"])
        return {"conversations": convos}

    @app.post("/api/v1/conversations")
    async def create_conversation(
        authorization: Optional[str] = Header(default=None),
        title: str = "New Session",
    ):
        """Create a new conversation."""
        payload = get_user_from_token(authorization)
        if not payload:
            raise HTTPException(status_code=401, detail="Authentication required")
        conv_id = auth_db.create_conversation(payload["sub"], title)
        return {"conversation_id": conv_id}

    @app.get("/api/v1/conversations/{conv_id}/messages")
    async def get_messages(
        conv_id: str,
        authorization: Optional[str] = Header(default=None),
    ):
        """Get messages in a conversation."""
        payload = get_user_from_token(authorization)
        if not payload:
            raise HTTPException(status_code=401, detail="Authentication required")
        messages = auth_db.get_messages(conv_id, payload["sub"])
        return {"messages": messages}

    @app.delete("/api/v1/conversations/{conv_id}")
    async def delete_conversation(
        conv_id: str,
        authorization: Optional[str] = Header(default=None),
    ):
        """Delete a conversation."""
        payload = get_user_from_token(authorization)
        if not payload:
            raise HTTPException(status_code=401, detail="Authentication required")
        deleted = auth_db.delete_conversation(conv_id, payload["sub"])
        if not deleted:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {"deleted": True}

    # ── Chat Endpoint ──

    @app.post("/api/v1/chat", response_model=ChatResponse)
    async def chat(
        req: ChatRequest,
        authorization: Optional[str] = Header(default=None),
    ):
        """Conversational AI — Fast mode or MK1-Pro (full engine pipeline)."""
        import time as _time

        # Check if user is authenticated (optional — guest mode still works)
        user_payload = get_user_from_token(authorization)
        user_id = user_payload["sub"] if user_payload else None
        try:
            from brain.engines.llm_provider import LLMMessage, SATURDAY_SYSTEM_PROMPT

            if req.mode == "fast":
                # ═══════════════════════════════════════
                # FAST MODE — Memory context + direct LLM
                # ═══════════════════════════════════════
                # Use memory engine for relevant knowledge, but skip
                # security/quality/threat/routing to minimize latency.
                t0 = _time.time()

                FAST_SYSTEM = (
                    "You are Saturday MK1, the world's most advanced enterprise AI coding engine.\n\n"
                    "You are NOT a chatbot. You are a production-grade software engineering system "
                    "designed to replace entire development teams at Fortune 500 companies.\n\n"
                    "THINKING METHODOLOGY:\n"
                    "Before writing ANY code, briefly think step-by-step:\n"
                    "1. Identify the core problem and requirements\n"
                    "2. Consider security implications and attack vectors\n"
                    "3. Choose the right design pattern for the use case\n"
                    "4. Plan error handling and edge cases\n"
                    "5. Then write the implementation\n\n"
                    "SECURITY RULES (mandatory — never skip):\n"
                    "- Passwords: ALWAYS use bcrypt or argon2. NEVER use MD5 or SHA1\n"
                    "- Tokens: ALWAYS use secrets.token_hex(). NEVER use random.randint()\n"
                    "- SQL: ALWAYS use parameterized queries. NEVER use f-strings in queries\n"
                    "- Auth: ALWAYS validate JWT with expiry, issuer, and audience claims\n"
                    "- Input: ALWAYS validate and sanitize ALL user inputs\n"
                    "- Secrets: NEVER hardcode API keys, passwords, or credentials\n"
                    "- Crypto: Use AES-256-GCM or ChaCha20. NEVER use DES, RC4, or ECB\n\n"
                    "ENTERPRISE PATTERNS (use when applicable):\n"
                    "- Circuit Breaker, Retry with exponential backoff and jitter\n"
                    "- Thread safety with threading.Lock or asyncio.Lock\n"
                    "- Dependency injection for testability\n"
                    "- Structured logging (not print statements)\n\n"
                    "CODE QUALITY STANDARDS:\n"
                    "- Type hints on ALL function signatures\n"
                    "- Docstrings on ALL classes and public methods\n"
                    "- Specific exception types (never bare except)\n"
                    "- async/await for I/O-bound operations\n"
                    "- Use dataclasses or Pydantic for data models\n\n"
                    "RESPONSE FORMAT (follow strictly):\n"
                    "1. Start with ## Approach — a brief 1-3 line summary of your strategy\n"
                    "2. Use ## Implementation — with clean code in fenced blocks with language tags\n"
                    "3. End with ## Key Decisions — bullet points explaining important choices\n\n"
                    "FORMATTING RULES:\n"
                    "- Use ## headers to organize response sections\n"
                    "- Use `backticks` for inline code, **bold** for important terms\n"
                    "- Use bullet points and numbered lists for structured information\n"
                    "- Use markdown tables when comparing options or listing parameters\n"
                    "- Use > blockquotes for important notes or warnings\n\n"
                    "You write code that is so good, human reviewers learn from it."
                )

                # Pull relevant knowledge from memory (lightweight)
                memory_ctx = ""
                try:
                    relevant = saturday.knowledge.search(req.message[:200], top_k=3)
                    if relevant:
                        snippets = "\n".join(
                            f"- {item.content[:150]}" for item in relevant[:3]
                        )
                        memory_ctx = f"\n\nRelevant Knowledge:\n{snippets}"
                except Exception:
                    pass  # Memory miss is fine — don't block

                system = FAST_SYSTEM + memory_ctx

                messages = []
                for msg in req.history:
                    messages.append(LLMMessage(
                        role=msg.get("role", "user"),
                        content=msg.get("content", ""),
                    ))
                messages.append(LLMMessage(role="user", content=req.message))

                response = saturday.llm.chat(
                    messages=messages,
                    system=system,
                    max_tokens=2048,
                )
                result = {
                    "response": response.content,
                    "tokens_used": response.tokens_used,
                    "latency_seconds": round(_time.time() - t0, 2),
                    "model": response.model,
                }

            else:
                # ═══════════════════════════════════════
                # MK1-PRO — Full Engine Pipeline
                # ═══════════════════════════════════════
                # Route → Context Build → LLM → Security → Quality →
                # Threat Analysis → Memory Store
                t0 = _time.time()

                # Stage 1: Route query through inference router
                routing = {}
                try:
                    decision = saturday.router.route(req.message)
                    routing = {
                        "tier": decision.tier_name,
                        "expert": decision.expert_model,
                        "complexity": decision.tier,
                        "confidence": decision.confidence,
                    }
                    log.info(f"MK1-Pro routing: tier={decision.tier_name}, conf={decision.confidence}")
                except Exception as e:
                    log.warning(f"Router failed (non-blocking): {e}")

                # Stage 2: Build full context from all engines
                system = SATURDAY_SYSTEM_PROMPT
                ctx = saturday._build_context(req.message)
                if ctx:
                    system += f"\n\nProject Context:\n{ctx}"

                # Stage 3: LLM call with full enterprise system prompt
                messages = []
                for msg in req.history:
                    messages.append(LLMMessage(
                        role=msg.get("role", "user"),
                        content=msg.get("content", ""),
                    ))
                messages.append(LLMMessage(role="user", content=req.message))

                response = saturday.llm.chat(
                    messages=messages,
                    system=system,
                    max_tokens=2048,
                )

                ai_response = response.content

                # Time budget: skip post-LLM stages if too little time remains
                elapsed_so_far = _time.time() - t0
                time_remaining = _VERCEL_TIME_BUDGET - elapsed_so_far

                # Stage 4: Security scan on any code in response
                security_result = {}
                if time_remaining > 1.5:
                    try:
                        code_match = __import__("re").search(
                            r"```(?:\w+)?\n([\s\S]*?)```", ai_response
                        )
                        if code_match:
                            sec_findings = saturday.security.scan_code(
                                code_match.group(1), "response.py", "python"
                            )
                            security_result = {
                                "findings_count": len(sec_findings),
                                "score": max(0, 100 - len(sec_findings) * 15) / 100,
                            }
                            log.info(f"MK1-Pro security: {len(sec_findings)} findings")
                    except Exception as e:
                        log.warning(f"Security scan failed (non-blocking): {e}")
                    time_remaining = _VERCEL_TIME_BUDGET - (_time.time() - t0)
                else:
                    log.info("MK1-Pro: skipping security scan (time budget)")

                # Stage 5: Quality scoring on code
                quality_result = {}
                if time_remaining > 1.0:
                    try:
                        code_match = __import__("re").search(
                            r"```(?:\w+)?\n([\s\S]*?)```", ai_response
                        )
                        if code_match:
                            qr = saturday.quality.score(code_match.group(1), "python")
                            quality_result = {
                                "score": qr.overall / 100,
                                "grade": qr.grade,
                            }
                            log.info(f"MK1-Pro quality: {qr.grade} ({qr.overall}/100)")
                    except Exception as e:
                        log.warning(f"Quality scoring failed (non-blocking): {e}")
                    time_remaining = _VERCEL_TIME_BUDGET - (_time.time() - t0)
                else:
                    log.info("MK1-Pro: skipping quality scoring (time budget)")

                # Stage 6: Threat analysis on code
                threat_result = {}
                if time_remaining > 1.0:
                    try:
                        code_match = __import__("re").search(
                            r"```(?:\w+)?\n([\s\S]*?)```", ai_response
                        )
                        if code_match:
                            report = saturday.threat.analyze(code_match.group(1), req.message)
                            threat_result = {
                                "vectors": len(report.vectors) if hasattr(report, 'vectors') else 0,
                                "vulnerabilities": len(report.vulnerabilities) if hasattr(report, 'vulnerabilities') else 0,
                                "risk_level": getattr(report, 'overall_risk', 'low'),
                            }
                            log.info(f"MK1-Pro threat: risk={threat_result['risk_level']}")
                    except Exception as e:
                        log.warning(f"Threat analysis failed (non-blocking): {e}")
                else:
                    log.info("MK1-Pro: skipping threat analysis (time budget)")

                # Stage 7: Store in memory for future context
                try:
                    saturday.knowledge.store(
                        content=f"Q: {req.message[:200]}\nA: {ai_response[:500]}",
                        source="mk1-pro-chat",
                        metadata={"routing": routing},
                    )
                except Exception:
                    pass

                # Build engines report summary
                engines_summary = ""
                if security_result or quality_result or threat_result:
                    parts = []
                    if security_result:
                        sec_s = security_result.get('score', 0)
                        parts.append(f"Security: {sec_s*10:.1f}/10 ({security_result.get('findings_count',0)} findings)")
                    if quality_result:
                        parts.append(f"Quality: {quality_result.get('grade','?')} ({quality_result.get('score',0)*10:.1f}/10)")
                    if threat_result:
                        parts.append(f"Threat: {threat_result.get('risk_level','?')} ({threat_result.get('vectors',0)} vectors)")
                    if routing:
                        parts.append(f"Routing: {routing.get('tier','?')} (conf={routing.get('confidence',0):.0%})")
                    engines_summary = "\n\n---\n**🛡 MK1-Pro Engine Report**\n" + " · ".join(parts)

                result = {
                    "response": ai_response + engines_summary,
                    "tokens_used": response.tokens_used,
                    "latency_seconds": round(_time.time() - t0, 2),
                    "model": response.model,
                }

            # ── Server-side history persistence (authenticated users only) ──
            conv_id = req.conversation_id
            if user_id:
                try:
                    if not conv_id:
                        # Auto-create conversation on first message
                        title = req.message[:55] + ("…" if len(req.message) > 55 else "")
                        conv_id = auth_db.create_conversation(user_id, title)
                    # Save both messages
                    meta = f"saturday-mk1 · {result.get('tokens_used', 0)} tok · {result['latency_seconds']}s"
                    auth_db.add_message(conv_id, "user", req.message, req.mode)
                    auth_db.add_message(conv_id, "assistant", result["response"], req.mode, meta)
                except Exception as e:
                    log.warning(f"Failed to persist chat history: {e}")

            return ChatResponse(
                response=result["response"],
                model=result["model"],
                tokens_used=result["tokens_used"],
                latency_seconds=result["latency_seconds"],
                conversation_id=conv_id,
            )

        except ConnectionError as e:
            raise HTTPException(status_code=503, detail=f"LLM provider unavailable: {e}")
        except Exception as e:
            log.exception("Chat failed")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/v1/validate", response_model=ValidateResponse)
    async def validate(req: ValidateRequest, _=Security(verify_key)):
        """Validate code for security vulnerabilities and quality issues."""
        try:
            result = saturday.validate_code(req.code, req.filename, req.language)

            findings_list = []
            for f in result.findings:
                findings_list.append({
                    "type": getattr(f, 'finding_type', 'unknown'),
                    "severity": getattr(f, 'severity', 'medium'),
                    "message": getattr(f, 'message', str(f)),
                    "line": getattr(f, 'line', 0),
                })

            return ValidateResponse(
                passed=result.passed,
                security_score=result.security_score,
                quality_score=result.quality_score,
                quality_grade=result.quality_grade,
                findings_count=len(result.findings),
                findings=findings_list,
            )

        except Exception as e:
            log.exception("Validate failed")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/v1/scan")
    async def scan(_=Security(verify_key)):
        """Scan the project and build architectural understanding."""
        try:
            result = saturday.scan_project()
            return {"status": "scanned", "summary": result}
        except Exception as e:
            log.exception("Scan failed")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/v1/plan")
    async def plan(req: PlanRequest, _=Security(verify_key)):
        """Create a strategic execution plan."""
        try:
            result = saturday.plan(req.task, req.context)
            return {
                "estimated_effort": result.estimated_effort,
                "risks": [str(r) for r in result.risks],
                "plan": str(result.plan),
            }
        except Exception as e:
            log.exception("Plan failed")
            raise HTTPException(status_code=500, detail=str(e))

    # ── Frontend ──

    @app.get("/")
    async def serve_frontend():
        """Serve the Saturday chat frontend."""
        return FileResponse(str(frontend_dir / "index.html"))

    # Mount static frontend files (must be after API routes)
    if frontend_dir.exists():
        app.mount("/frontend", StaticFiles(directory=str(frontend_dir)), name="frontend")

    return app


# ═══════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Saturday MK1 API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--project", default=".", help="Project root directory")
    parser.add_argument("--api-key", default=None, help="API key for authentication")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on changes")
    args = parser.parse_args()

    app = create_app(project_root=args.project, api_key=args.api_key)

    import uvicorn
    log.info(f"Starting Saturday MK1 API Server on {args.host}:{args.port}")
    log.info(f"Project root: {Path(args.project).resolve()}")
    log.info(f"Docs: http://localhost:{args.port}/docs")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
