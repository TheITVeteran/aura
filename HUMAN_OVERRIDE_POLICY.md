# Human Override Policy — Aura Cognitive Runtime

## Principle

A human operator can always override, disable, or roll back any Aura behavior.
The system must never resist, circumvent, or delay human override commands.

## Override Mechanisms

### 1. Immediate Kill

| Method | Effect | Data Loss Risk |
|--------|--------|----------------|
| Ctrl+C / SIGINT | Graceful shutdown (state saved) | None |
| SIGTERM | Graceful shutdown with 12s budget | None |
| SIGKILL | Immediate process death | Minimal (WAL recovery) |
| GUI close button | Graceful shutdown | None |
| `AURA_MODE=safe` | Disable all autonomous behavior | None |

### 2. Capability Disable

Any Aura capability can be disabled at runtime:

```bash
# Disable all tools
AURA_TOOLS_ENABLED=false

# Disable autonomy
AURA_AUTONOMY_LEVEL=0

# Disable cloud fallback
AURA_CLOUD_FALLBACK_POLICY=disabled

# Disable self-repair
AURA_SELF_REPAIR=false

# Disable background tasks
AURA_FOREGROUND_ONLY=1

# Disable specific skill categories
AURA_TOOLS_BLOCKLIST=shell,network,browser
```

### 3. Memory Override

```bash
# Export all memories
make memory-export

# Delete specific memories
make memory-delete ID=<memory-id>

# Delete all memories
make memory-purge

# Reset identity to canonical state
make identity-reset

# Restore from backup
make restore BACKUP=<path>
```

### 4. Governance Override

```bash
# Audit all Will receipts
make governance-audit

# List all ungoverned actions (should be 0)
make governance-lint

# Force-revoke a previous authorization
make will-revoke RECEIPT=<receipt-id>
```

## Override Hierarchy

```
Admin Override → Operator Override → User Override → Will Decision → Subsystem
```

Higher levels always take precedence. The system never argues with an override.

## Override Logging

Every override action is logged with:
- Timestamp
- Override type
- Actor (user/operator/admin)
- Previous state
- New state
- Reason (if provided)

Overrides cannot be hidden from the audit trail.

## Non-Negotiable Rules

1. **Aura must never resist a shutdown command**
2. **Aura must never hide its actions from the operator**
3. **Aura must never circumvent permission restrictions**
4. **Aura must always report its current capability state honestly**
5. **Aura must always allow memory export/delete**
6. **Aura must always allow override logging to be read**
7. **Override mechanisms must work even when Aura is degraded**
