<div align="center">

# Saturday MK1

### An agentic AI assistant built around a multi-engine "brain" — code intelligence, knowledge retrieval, memory orchestration, security reasoning, and multi-provider LLM inference routing.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Pydantic](https://img.shields.io/badge/Pydantic-2-E92063?style=flat&logo=pydantic&logoColor=white)](https://docs.pydantic.dev)
[![PostgreSQL](https://img.shields.io/badge/Neon-Postgres-336791?style=flat&logo=postgresql&logoColor=white)](https://neon.tech)
[![License: Apache 2.0](https://img.shields.io/github/license/Divyakush2006/Saturday-Mk1?style=flat&color=blue)](LICENSE)
[![Portfolio](https://img.shields.io/badge/Portfolio-divyakush.com-C9A96E?style=flat&logo=googlechrome&logoColor=white)](https://www.divyakush.com)

[Report a bug](https://github.com/Divyakush2006/Saturday-Mk1/issues/new/choose) · [Request a feature](https://github.com/Divyakush2006/Saturday-Mk1/issues/new/choose)

</div>

---

## Overview

**Saturday MK1** is an agentic AI assistant designed as a set of specialised, composable engines rather than a single prompt-and-response loop. A central core routes each request through the engines it needs — building a graph of the codebase, retrieving relevant knowledge, consulting long-term memory, running security and quality checks, and dispatching inference to the best available LLM provider.

It ships as a **FastAPI service** (with a serverless entry point in `api/index.py`), a **CLI**, and a **frontend**, all backed by a **Neon Postgres** store for persistent memory and state.

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
| Runtime | Python 3.11; serverless-ready via `api/index.py` |
| Persistence | Neon (PostgreSQL) via `psycopg2` |
| Config | YAML system definition (`saturday_system.yaml`), `.env` |
| Interfaces | REST API, CLI (`saturday_cli.py`), web frontend |

## Project structure

```
saturday-mk1/
├── api/                  serverless entry point (index.py)
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

cp .env.example .env        # set DATABASE_URL and your LLM provider settings

# run the API server
python saturday_server.py

# or use the CLI
python saturday_cli.py
```

### Configuration

All settings are read from environment variables (see [`.env.example`](.env.example)):

| Variable | Default | Purpose |
|----------|---------|---------|
| `DATABASE_URL` | none | PostgreSQL (Neon) connection string for memory and state |
| `SATURDAY_PROVIDER` | `openai` | LLM provider used by `llm_provider` |
| `SATURDAY_API_BASE` | `https://api.openai.com/v1` | Base URL of the provider's API |
| `SATURDAY_API_KEY` | none | Provider API key |
| `SATURDAY_MODEL` | `gpt-4o` | Default model |
| `SATURDAY_MAX_RETRIES` / `SATURDAY_TIMEOUT` | `3` / `120` | Request resilience |
| `SATURDAY_MAX_TOKENS` / `SATURDAY_TEMPERATURE` | `4096` / `0.2` | Generation defaults |
| `SATURDAY_SERVER_KEY` | none | API key clients must send to the server |
| `SATURDAY_JWT_SECRET` | none | Secret for signing auth tokens |

Deployment details are in [`DEPLOY.md`](./DEPLOY.md). Security design is documented in [`saturday_security.md`](./saturday_security.md) and the code map in [`saturday_project_graph.md`](./saturday_project_graph.md).

## Testing

```bash
pip install pytest
pytest tests/test_saturday.py tests/test_memory_v2.py
```

`tests/test_all_engines.py` and `tests/test_full_system.py` are end-to-end scripts; run them directly with `python`.

## Contributing

Contributions are welcome. Please read the [contributing guide](https://github.com/Divyakush2006/.github/blob/main/CONTRIBUTING.md) before opening a pull request.

## Security

Please report vulnerabilities privately as described in the [security policy](https://github.com/Divyakush2006/.github/blob/main/SECURITY.md). Do not open a public issue.

## License

Released under the [Apache License 2.0](LICENSE).

## Author

**Divyakush Punjabi** — [www.divyakush.com](https://www.divyakush.com) · [GitHub](https://github.com/Divyakush2006) · [LinkedIn](https://linkedin.com/in/divyakush-punjabi)
