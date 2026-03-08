"""
Saturday MK1 — Vercel Serverless Adapter
=========================================
Exposes the FastAPI application as a Vercel Python serverless function.

Vercel expects a module-level `app` (ASGI) or `handler` (WSGI) variable.
We import the FastAPI app from saturday_server and re-export it.

Environment variables (SATURDAY_API_KEY, SATURDAY_MODEL, DATABASE_URL, etc.)
are configured in the Vercel dashboard — no .env file needed in production.
"""

import os
import sys
from pathlib import Path

# Ensure the project root is on the Python path so `brain.*` imports work.
# In Vercel, the working directory is the project root, but we add it
# explicitly to guarantee resolution.
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load environment from .env only if it exists (local dev).
# On Vercel, env vars are injected directly.
env_file = Path(project_root) / ".env"
if env_file.exists():
    from dotenv import load_dotenv
    load_dotenv(env_file)

# Import the FastAPI app from the server module.
# create_app() initializes Saturday core + all routes.
from saturday_server import create_app

app = create_app(
    project_root=project_root,
    api_key=os.getenv("SATURDAY_SERVER_KEY") or None,
)
