# Saturday — Persistent Memory Store
# ==================================
# This file persists across coding sessions. Saturday reads and updates it
# to maintain long-term knowledge about projects, decisions, and conventions.

## Architecture Decision Records (ADR)

### ADR-001: Base Model Selection
- **Decision**: Use Kimi K2.5 as base model for Saturday MK1
- **Date**: 2026-03-07
- **Rationale**: Closest free model to Claude Opus 4.6 on SWE-bench (76.8% vs 80.8%), HumanEval (99%), LiveCodeBench (85%). 1T MoE with 32B active params, 256K context.
- **Alternatives Considered**: Qwen 3.5 397B (performance drops on complex tasks), DeepSeek V3.2 (lower SWE-bench), GPT-oss 120B (lowest benchmark scores)
- **Status**: Approved

### ADR-002: Branding Strategy
- **Decision**: External brand is "Saturday MK1" — never expose base model details
- **Date**: 2026-03-07
- **Rationale**: Maintain proprietary positioning for MNC market. Base model is an implementation detail.
- **Status**: Approved

### ADR-003: Security-First Architecture
- **Decision**: All generated code passes 6-layer security pipeline before delivery
- **Date**: 2026-03-07
- **Rationale**: Enterprise clients require zero-vulnerability guarantee. ~48% baseline vulnerability rate in competitor models necessitates mandatory validation.
- **Status**: Approved

## Project Conventions

_To be populated as Saturday learns project-specific conventions during coding sessions._

## Security Decisions

- All API keys must use environment variables, never hardcoded
- Default to parameterized queries for all database operations
- HTTPS-only for all external API calls
- JWT tokens must include expiry, issuer, and audience claims
- Password hashing must use bcrypt/argon2, never MD5/SHA1

## Known Patterns

_To be populated as Saturday identifies recurring patterns in codebases._

## Compliance Requirements

- **SOC2**: Audit logging for all data access operations
- **GDPR**: Right-to-delete handlers for all PII data stores
- **HIPAA**: Encryption at rest and in transit for health data
- **PCI-DSS**: Tokenization for payment card data, never store raw card numbers
