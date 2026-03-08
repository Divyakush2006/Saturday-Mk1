"""
Saturday MK1 — Authentication & User Management
=================================================
Handles user signup, login, JWT token management, and server-side
conversation persistence for authenticated users.

Uses:
    - bcrypt for password hashing (industry-standard, argon2 alternative noted)
    - PyJWT for JWT token creation/verification
    - SQLite for persistence (upgrade to PostgreSQL for production)

Security:
    - Passwords hashed with bcrypt (cost factor 12)
    - JWT tokens with 7-day expiry, issuer claim, and audience claim
    - No plaintext passwords stored anywhere
    - SQL injection prevented via parameterized queries throughout
"""

import hashlib
import hmac
import json
import logging
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger("saturday-auth")


# ═══════════════════════════════════════════
# DATA TYPES
# ═══════════════════════════════════════════

@dataclass
class User:
    """Authenticated user."""
    id: str
    email: str
    username: str
    created_at: str


@dataclass
class AuthResult:
    """Result of authentication operation."""
    success: bool
    user: Optional[User] = None
    token: Optional[str] = None
    error: Optional[str] = None


# ═══════════════════════════════════════════
# PASSWORD HASHING
# ═══════════════════════════════════════════

def _hash_password(password: str) -> str:
    """
    Hash a password using bcrypt.

    Falls back to PBKDF2-HMAC-SHA256 if bcrypt is not installed,
    which is still cryptographically secure.
    """
    try:
        import bcrypt
        return bcrypt.hashpw(
            password.encode("utf-8"),
            bcrypt.gensalt(rounds=12),
        ).decode("utf-8")
    except ImportError:
        # Fallback: PBKDF2-HMAC-SHA256 (stdlib, still secure)
        salt = os.urandom(32)
        key = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, iterations=600_000
        )
        return f"pbkdf2:{salt.hex()}:{key.hex()}"


def _verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its hash."""
    try:
        import bcrypt
        if password_hash.startswith("pbkdf2:"):
            raise ImportError("Use PBKDF2 path")
        return bcrypt.checkpw(
            password.encode("utf-8"),
            password_hash.encode("utf-8"),
        )
    except ImportError:
        if not password_hash.startswith("pbkdf2:"):
            return False
        _, salt_hex, key_hex = password_hash.split(":")
        salt = bytes.fromhex(salt_hex)
        expected_key = bytes.fromhex(key_hex)
        actual_key = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, iterations=600_000
        )
        return hmac.compare_digest(actual_key, expected_key)


# ═══════════════════════════════════════════
# JWT TOKENS
# ═══════════════════════════════════════════

# Secret key for JWT signing — from env var (Vercel) or file (local dev)
_JWT_SECRET_FILE = Path(__file__).parent.parent / ".saturday_jwt_secret"
_JWT_EXPIRY_DAYS = 7
_JWT_ISSUER = "saturday-mk1"
_JWT_AUDIENCE = "saturday-users"


def _get_jwt_secret() -> str:
    """Get or create the JWT signing secret."""
    # Prefer environment variable (Vercel / production)
    env_secret = os.getenv("SATURDAY_JWT_SECRET")
    if env_secret:
        return env_secret
    # Fall back to local file (development)
    if _JWT_SECRET_FILE.exists():
        return _JWT_SECRET_FILE.read_text(encoding="utf-8").strip()
    import secrets
    secret = secrets.token_hex(64)
    try:
        _JWT_SECRET_FILE.write_text(secret, encoding="utf-8")
        log.info("Generated new JWT secret")
    except OSError:
        log.warning("Could not persist JWT secret to file — using ephemeral secret")
    return secret


def create_jwt(user_id: str, email: str) -> str:
    """
    Create a JWT token for an authenticated user.

    Uses PyJWT if available, falls back to manual HMAC-SHA256 JWT.
    Token includes: sub (user_id), email, iss, aud, iat, exp.
    """
    now = int(time.time())
    payload = {
        "sub": user_id,
        "email": email,
        "iss": _JWT_ISSUER,
        "aud": _JWT_AUDIENCE,
        "iat": now,
        "exp": now + (_JWT_EXPIRY_DAYS * 86400),
    }

    try:
        import jwt
        return jwt.encode(payload, _get_jwt_secret(), algorithm="HS256")
    except ImportError:
        # Manual JWT (HMAC-SHA256)
        import base64
        header = base64.urlsafe_b64encode(
            json.dumps({"alg": "HS256", "typ": "JWT"}).encode()
        ).rstrip(b"=").decode()
        body = base64.urlsafe_b64encode(
            json.dumps(payload).encode()
        ).rstrip(b"=").decode()
        sig_input = f"{header}.{body}".encode()
        signature = hmac.new(
            _get_jwt_secret().encode(), sig_input, hashlib.sha256
        ).digest()
        sig = base64.urlsafe_b64encode(signature).rstrip(b"=").decode()
        return f"{header}.{body}.{sig}"


def verify_jwt(token: str) -> Optional[dict]:
    """
    Verify a JWT token and return the payload.

    Returns None if the token is invalid or expired.
    """
    try:
        import jwt
        return jwt.decode(
            token, _get_jwt_secret(),
            algorithms=["HS256"],
            issuer=_JWT_ISSUER,
            audience=_JWT_AUDIENCE,
        )
    except ImportError:
        # Manual JWT verification
        import base64
        try:
            parts = token.split(".")
            if len(parts) != 3:
                return None
            header_b, body_b, sig_b = parts
            # Verify signature
            sig_input = f"{header_b}.{body_b}".encode()
            expected_sig = hmac.new(
                _get_jwt_secret().encode(), sig_input, hashlib.sha256
            ).digest()
            # Pad base64
            sig_bytes = base64.urlsafe_b64decode(sig_b + "==")
            if not hmac.compare_digest(expected_sig, sig_bytes):
                return None
            # Decode payload
            payload = json.loads(base64.urlsafe_b64decode(body_b + "=="))
            # Check expiry
            if payload.get("exp", 0) < time.time():
                return None
            # Check issuer
            if payload.get("iss") != _JWT_ISSUER:
                return None
            return payload
        except Exception:
            return None
    except Exception:
        return None


# ═══════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════

def _is_postgres() -> bool:
    """Check if we should use PostgreSQL (via DATABASE_URL env var)."""
    url = os.getenv("DATABASE_URL", "")
    return url.startswith("postgres://") or url.startswith("postgresql://")


class AuthDB:
    """
    Dual-mode user and conversation storage.

    Automatically selects backend based on DATABASE_URL env var:
    - PostgreSQL: when DATABASE_URL starts with postgres:// (Vercel/Neon)
    - SQLite: fallback for local development

    All methods share identical signatures regardless of backend.
    """

    def __init__(self, db_path: Optional[str] = None):
        self._use_pg = _is_postgres()
        if self._use_pg:
            self._pg_url = os.getenv("DATABASE_URL", "")
            # Neon requires sslmode=require
            if "sslmode" not in self._pg_url:
                sep = "&" if "?" in self._pg_url else "?"
                self._pg_url += f"{sep}sslmode=require"
            log.info("AuthDB: using PostgreSQL backend")
        else:
            if db_path is None:
                db_path = str(Path(__file__).parent.parent / "saturday.db")
            self.db_path = db_path
            log.info(f"AuthDB: using SQLite backend ({db_path})")
        self._init_db()

    # ── Connection helpers ──

    def _get_conn(self):
        """Get a database connection (SQLite or PostgreSQL)."""
        if self._use_pg:
            import psycopg2
            import psycopg2.extras
            conn = psycopg2.connect(self._pg_url)
            conn.autocommit = False
            return conn
        else:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            return conn

    def _execute(self, conn, sql, params=None):
        """Execute SQL with dialect-appropriate parameter style."""
        if self._use_pg:
            import psycopg2.extras
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            # Convert ? placeholders to %s for PostgreSQL
            sql = sql.replace("?", "%s")
            cur.execute(sql, params or ())
            return cur
        else:
            return conn.execute(sql, params or ())

    def _fetchone(self, cur):
        """Fetch one row as a dict-like object."""
        row = cur.fetchone()
        if row is None:
            return None
        if self._use_pg:
            return row  # Already a RealDictRow
        return row  # sqlite3.Row acts like dict

    def _fetchall(self, cur):
        """Fetch all rows as list of dicts."""
        rows = cur.fetchall()
        if self._use_pg:
            return [dict(r) for r in rows]
        return [dict(r) for r in rows]

    def _init_db(self):
        """Create tables if they don't exist."""
        conn = self._get_conn()
        try:
            if self._use_pg:
                cur = conn.cursor()
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id TEXT PRIMARY KEY,
                        email TEXT UNIQUE NOT NULL,
                        username TEXT UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT NOW()
                    );

                    CREATE TABLE IF NOT EXISTS conversations (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                        title TEXT DEFAULT 'New Session',
                        created_at TIMESTAMP DEFAULT NOW(),
                        updated_at TIMESTAMP DEFAULT NOW()
                    );

                    CREATE TABLE IF NOT EXISTS messages (
                        id SERIAL PRIMARY KEY,
                        conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        mode TEXT,
                        meta TEXT,
                        created_at TIMESTAMP DEFAULT NOW()
                    );

                    CREATE INDEX IF NOT EXISTS idx_conversations_user
                        ON conversations(user_id);
                    CREATE INDEX IF NOT EXISTS idx_messages_conversation
                        ON messages(conversation_id);
                """)
                conn.commit()
                log.info("Auth database initialized: PostgreSQL (Neon)")
            else:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS users (
                        id TEXT PRIMARY KEY,
                        email TEXT UNIQUE NOT NULL,
                        username TEXT UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE TABLE IF NOT EXISTS conversations (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                        title TEXT DEFAULT 'New Session',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        mode TEXT,
                        meta TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE INDEX IF NOT EXISTS idx_conversations_user
                        ON conversations(user_id);
                    CREATE INDEX IF NOT EXISTS idx_messages_conversation
                        ON messages(conversation_id);
                """)
                conn.commit()
                log.info(f"Auth database initialized: {self.db_path}")
        finally:
            conn.close()

    # ── User Operations ──

    def signup(self, email: str, username: str, password: str) -> AuthResult:
        """Create a new user account."""
        email = email.strip().lower()
        username = username.strip()

        # Validation
        if not email or "@" not in email:
            return AuthResult(success=False, error="Invalid email address")
        if not username or len(username) < 2:
            return AuthResult(success=False, error="Username must be at least 2 characters")
        if not password or len(password) < 6:
            return AuthResult(success=False, error="Password must be at least 6 characters")

        conn = self._get_conn()
        try:
            # Check duplicates
            cur = self._execute(
                conn,
                "SELECT id FROM users WHERE email = ? OR username = ?",
                (email, username),
            )
            if self._fetchone(cur):
                return AuthResult(success=False, error="Email or username already taken")

            user_id = str(uuid.uuid4())
            password_hash = _hash_password(password)

            self._execute(
                conn,
                "INSERT INTO users (id, email, username, password_hash) VALUES (?, ?, ?, ?)",
                (user_id, email, username, password_hash),
            )
            conn.commit()

            user = User(id=user_id, email=email, username=username,
                        created_at=datetime.now(timezone.utc).isoformat())
            token = create_jwt(user_id, email)

            log.info(f"User signed up: {username} ({email})")
            return AuthResult(success=True, user=user, token=token)

        except (sqlite3.IntegrityError, Exception) as e:
            if "unique" in str(e).lower() or "duplicate" in str(e).lower() or "integrity" in str(e).lower():
                return AuthResult(success=False, error="Email or username already taken")
            raise
        finally:
            conn.close()

    def login(self, email: str, password: str) -> AuthResult:
        """Authenticate a user with email and password."""
        email = email.strip().lower()

        conn = self._get_conn()
        try:
            cur = self._execute(
                conn,
                "SELECT id, email, username, password_hash, created_at FROM users WHERE email = ?",
                (email,),
            )
            row = self._fetchone(cur)

            if not row:
                return AuthResult(success=False, error="Invalid email or password")

            if not _verify_password(password, row["password_hash"]):
                return AuthResult(success=False, error="Invalid email or password")

            created_at = row["created_at"]
            if hasattr(created_at, "isoformat"):
                created_at = created_at.isoformat()

            user = User(
                id=row["id"], email=row["email"],
                username=row["username"], created_at=str(created_at),
            )
            token = create_jwt(user.id, user.email)

            log.info(f"User logged in: {user.username}")
            return AuthResult(success=True, user=user, token=token)

        finally:
            conn.close()

    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        conn = self._get_conn()
        try:
            cur = self._execute(
                conn,
                "SELECT id, email, username, created_at FROM users WHERE id = ?",
                (user_id,),
            )
            row = self._fetchone(cur)
            if not row:
                return None
            created_at = row["created_at"]
            if hasattr(created_at, "isoformat"):
                created_at = created_at.isoformat()
            return User(
                id=row["id"], email=row["email"],
                username=row["username"], created_at=str(created_at),
            )
        finally:
            conn.close()

    # ── Conversation Operations ──

    def create_conversation(self, user_id: str, title: str = "New Session") -> str:
        """Create a new conversation for a user. Returns conversation ID."""
        conv_id = str(uuid.uuid4())
        conn = self._get_conn()
        try:
            self._execute(
                conn,
                "INSERT INTO conversations (id, user_id, title) VALUES (?, ?, ?)",
                (conv_id, user_id, title),
            )
            conn.commit()
            return conv_id
        finally:
            conn.close()

    def list_conversations(self, user_id: str) -> list[dict]:
        """List all conversations for a user, newest first."""
        conn = self._get_conn()
        try:
            cur = self._execute(
                conn,
                "SELECT id, title, created_at, updated_at FROM conversations "
                "WHERE user_id = ? ORDER BY updated_at DESC",
                (user_id,),
            )
            rows = self._fetchall(cur)
            # Normalize datetimes to strings
            for r in rows:
                for k in ("created_at", "updated_at"):
                    if k in r and hasattr(r[k], "isoformat"):
                        r[k] = r[k].isoformat()
            return rows
        finally:
            conn.close()

    def get_messages(self, conversation_id: str, user_id: str) -> list[dict]:
        """Get all messages in a conversation (with ownership check)."""
        conn = self._get_conn()
        try:
            # Verify ownership
            cur = self._execute(
                conn,
                "SELECT user_id FROM conversations WHERE id = ?",
                (conversation_id,),
            )
            owner = self._fetchone(cur)
            if not owner or owner["user_id"] != user_id:
                return []

            cur = self._execute(
                conn,
                "SELECT role, content, mode, meta, created_at FROM messages "
                "WHERE conversation_id = ? ORDER BY id ASC",
                (conversation_id,),
            )
            rows = self._fetchall(cur)
            for r in rows:
                if "created_at" in r and hasattr(r["created_at"], "isoformat"):
                    r["created_at"] = r["created_at"].isoformat()
            return rows
        finally:
            conn.close()

    def add_message(
        self, conversation_id: str, role: str, content: str,
        mode: str = "", meta: str = "",
    ):
        """Add a message to a conversation."""
        conn = self._get_conn()
        try:
            ts_fn = "NOW()" if self._use_pg else "CURRENT_TIMESTAMP"
            self._execute(
                conn,
                "INSERT INTO messages (conversation_id, role, content, mode, meta) "
                "VALUES (?, ?, ?, ?, ?)",
                (conversation_id, role, content, mode, meta),
            )
            self._execute(
                conn,
                f"UPDATE conversations SET updated_at = {ts_fn}, "
                "title = CASE WHEN title = 'New Session' THEN ? ELSE title END "
                "WHERE id = ?",
                (content[:55] + ("…" if len(content) > 55 else ""), conversation_id),
            )
            conn.commit()
        finally:
            conn.close()

    def delete_conversation(self, conversation_id: str, user_id: str) -> bool:
        """Delete a conversation (with ownership check)."""
        conn = self._get_conn()
        try:
            cur = self._execute(
                conn,
                "DELETE FROM conversations WHERE id = ? AND user_id = ?",
                (conversation_id, user_id),
            )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()
