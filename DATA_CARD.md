# Aura Data Card

## Memory Systems

| Store | Type | Persistence | Encryption | Privacy |
|-------|------|-------------|------------|---------|
| Working Memory | In-process | Session only | N/A | Cleared on shutdown |
| Conversation History | SQLite | Durable | Available (vault) | User-deletable |
| Semantic Memory (RAG) | Vector DB | Durable | At-rest available | User-deletable |
| ColdStore (Long-term) | SQLite | Durable | Available (vault) | User-deletable |
| State Snapshots | JSON/SQLite | Durable | Available (vault) | User-exportable |
| Will Receipt Log | Append-only log | Durable | Integrity-hashed | Audit-readable |

## Data Retention

| Data Type | Default Retention | User Control |
|-----------|-------------------|--------------|
| Conversation history | Indefinite | Delete any/all |
| Semantic memories | Indefinite | Delete any/all |
| Long-term memories | Indefinite | Delete any/all |
| Will receipts | Indefinite (audit) | Export only |
| Logs | 30 days (rotation) | Export/delete |
| Metrics | 7 days | Export/delete |
| Backups | 3 most recent | Delete any/all |

## Data Flow

```
User Input → Sanitizer → Working Memory → Model Context
                                              ↓
                                         Model Output
                                              ↓
                                    Integrity Check → User
                                              ↓
                              Will Decision → Memory Write
                                              ↓
                                    State Snapshot → Backup
```

## Privacy Controls

| Control | Mechanism |
|---------|-----------|
| Cloud fallback opt-in | `AURA_CLOUD_FALLBACK_POLICY` |
| Prompt classification | Auto-classify before cloud send |
| Memory export | `make memory-export` |
| Memory delete | `make memory-purge` (all) / app memory panel `POST /memory/delete` (one) |
| Log purge | `make log-purge` |
| Full data export | `make data-export` (GDPR-style) |
| Full data delete | `make data-purge` |

## No External Data Collection

Aura in default configuration:
- Sends no data to external services
- Has no telemetry phone-home
- Has no analytics collection
- Has no crash reporting to external services
- All data stays on the user's machine

Cloud fallback, if enabled, sends only the classified-safe portions of prompts.
