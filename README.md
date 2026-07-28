<div align="center">

# Saturday MK1

### An agentic AI assistant built around a multi-engine "brain" — code intelligence, knowledge retrieval, memory orchestration, security reasoning, and multi-provider LLM inference routing.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Pydantic](https://img.shields.io/badge/Pydantic-2-E92063?style=flat&logo=pydantic&logoColor=white)](https://docs.pydantic.dev)
[![PostgreSQL](https://img.shields.io/badge/Neon-Postgres-336791?style=flat&logo=postgresql&logoColor=white)](https://neon.tech)
[![Vercel](https://img.shields.io/badge/Vercel-serverless-000000?style=flat&logo=vercel&logoColor=white)](https://vercel.com)
[![Portfolio](https://img.shields.io/badge/Portfolio-divyakush.com-C9A96E?style=flat&logo=googlechrome&logoColor=white)](https://www.divyakush.com)

</div>

---

## Overview

**Saturday MK1** is an agentic AI assistant designed as a set of specialised, composable engines rather than a single prompt-and-response loop. A central core routes each request through the engines it needs — building a graph of the codebase, retrieving relevant knowledge, consulting long-term memory, running security and quality checks, and dispatching inference to the best available LLM provider.

It ships as a **FastAPI service** (deployable to Vercel as a serverless function), a **CLI**, and a **frontend**, all backed by a **Neon Postgres** store for persistent memory and state.

## The brain — engine architecture

Each engine is a focused, independently reasoned subsystem under `brain/engines/`:

| Engine | Responsibility |
|--------|----------------|
| `code_graph` | Builds a structural graph of the codebase to give the agent grounded, navigable context |
| `knowledge_base` | Retrieval over a curated knowledge store for grounded answers |
| `memory_orchestrator` | Long- and short-term memory management across sessions |
| `context_state` | Tracks and assembles the working context for each request |
| `inference_router` | Routes each task to the most suitable model |
| `llm_provider` | Provider abstraction over multiple LLM backends |
| `strategic_planner` | Decomposes goals into ordered, executable steps |
| `data_pipeline` | Ingestion and transformation of inputs into engine-ready data |
| `code_quality` | Static quality signals over generated / analysed code |
| `security_pipeline` | Security-focused processing over requests and outputs |
| `threat_engine` | Threat detection and risk reasoning |

The core (`brain/saturday_core.py`) orchestrates these engines; `brain/auth.py` handles authentication.

## Architecture

```
                        ┌──────────────┐  ┌─────┐  ┌──────────┐
   Client / CLI  ─────▶ │  FastAPI     │  │ CLI │  │ Frontend │
                        │  (api/index) │  └─────┘  └──────────┘
                        └──────┬───────┘
                               ▼
                        ┌──────────────┐
                        │ saturday_core│  orchestrates engines
                        └──────┬───────┘
        ┌──────────────┬───────┼────────────┬──────────────┐
        ▼              ▼       ▼            ▼              ▼
   code_graph   knowledge_base memory   inference_router security /
   code_quality               orchestrator │             threat_engine
                                           ▼
                                     llm_provider (multi-backend)
                               │
                               ▼
                        ┌──────────────┐
                        │ Neon Postgres│  memory & state
                        └──────────────┘
```

## Tech stack

| Layer | Technology |
|-------|------------|
| API | FastAPI, Pydantic v2 |
| Runtime | Python 3.11, deployed serverless on Vercel |
| Persistence | Neon (PostgreSQL) via `psycopg2` |
| Config | YAML system definition (`saturday_system.yaml`), `.env` |
| Interfaces | REST API, CLI (`saturday_cli.py`), web frontend |

## Project structure

```
saturday-mk1/
├── api/                  Vercel serverless entrypoint (index.py)
├── brain/
│   ├── saturday_core.py  engine orchestrator
│   ├── auth.py           authentication
│   ├── engines/          code_graph, knowledge_base, memory_orchestrator,
│   │                     inference_router, llm_provider, strategic_planner,
│   │                     security_pipeline, threat_engine, ...
│   └── context_state/    working-context management
├── frontend/             web UI
├── scripts/              tooling
├── tests/                test suite
├── saturday_server.py    FastAPI server
├── saturday_cli.py       command-line interface
├── saturday_system.yaml  system configuration
├── requirements.txt
└── vercel.json           deployment config
```

## Getting started

```bash
git clone https://github.com/Divyakush2006/Saturday-Mk1.git
cd Saturday-Mk1

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env        # set DATABASE_URL (Neon) and LLM provider keys

# run the API server
python saturday_server.py

# or use the CLI
python saturday_cli.py
```

Deployment details are in [`DEPLOY.md`](./DEPLOY.md). The system design is documented across `saturday_blueprint.md`, `saturday_context.md`, `saturday_security.md`, and `saturday_project_graph.md`.

## Author

**Divyakush Punjabi** — [www.divyakush.com](https://www.divyakush.com) · [GitHub](https://github.com/Divyakush2006) · [LinkedIn](https://linkedin.com/in/divyakush-punjabi)
