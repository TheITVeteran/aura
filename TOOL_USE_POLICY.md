# Aura Tool Use Policy

## Scope

This policy governs all tool and skill execution within the Aura cognitive
runtime. Every tool invocation is a consequential action that must be
authorized, sandboxed, audited, and recoverable.

## Principles

1. **No tool executes without Will authorization**: Every tool call passes
   through the Unified Will and receives a WillReceipt.
2. **All tool output is untrusted**: Results from tools are treated as external
   untrusted input and sanitized before influencing Aura's behavior.
3. **Least privilege**: Each skill requests only the permissions it needs.
4. **Fail closed**: If authorization is unavailable, tool execution is refused.
5. **Auditable**: Every tool invocation, its authorization, input, output, and
   outcome are logged.

## Skill Contract

Every skill/tool must declare:

```yaml
name: skill_name
version: "1.0.0"
description: "What this skill does"
risk_level: low | medium | high | critical
permissions:
  filesystem: none | read | write | workspace_only
  network: none | local | external
  shell: none | sandboxed | full
  memory: none | read | write
input_schema:
  type: object
  properties: { ... }
output_schema:
  type: object
  properties: { ... }
timeout_s: 30
max_memory_mb: 512
sandbox_policy: strict | permissive | none
audit_policy: full | summary | none
owner: "author name"
tests: "tests/test_skill_name.py"
```

## Permission Matrix

### By Role

| Permission | User | Operator | Admin | Research |
|-----------|------|----------|-------|----------|
| Chat | ✅ | ✅ | ✅ | ✅ |
| Read tools (clock, weather) | ✅ | ✅ | ✅ | ✅ |
| File tools (workspace only) | ✅ | ✅ | ✅ | ✅ |
| File tools (outside workspace) | ❌ | ✅ | ✅ | Sandbox |
| Shell (sandboxed) | ❌ | ✅ | ✅ | Sandbox |
| Shell (unrestricted) | ❌ | ❌ | ✅ | ❌ |
| Browser | Limited | ✅ | ✅ | Sandbox |
| Network (external) | ❌ | ✅ | ✅ | Sandbox |
| Memory read | Own | All | All | Sandbox |
| Memory write | Own | All | All | Sandbox |
| Memory delete | ❌ | ❌ | ✅ | ❌ |
| Self-repair | ❌ | Approve | ✅ | Sandbox |
| Plugin install | ❌ | ❌ | ✅ | ❌ |
| Model change | ❌ | ✅ | ✅ | ✅ |
| Feature flags | ❌ | Limited | ✅ | Limited |
| Cloud fallback | ❌ | ✅ | ✅ | ❌ |

### By Risk Level

| Risk Level | Authorization | Sandbox | Audit | Example |
|-----------|---------------|---------|-------|---------|
| Low | Auto-approve | Optional | Summary | Clock, calculator |
| Medium | Will decision | Recommended | Full | File read, web search |
| High | Will + operator confirm | Required | Full | Shell exec, file write |
| Critical | Will + admin confirm | Required + isolated | Full | Self-modification, plugin install |

## Operator Controls

Operators can configure tool access via environment variables or config file:

```bash
# Disable all tool execution
AURA_TOOLS_ENABLED=false

# Allow specific tools only
AURA_TOOLS_ALLOWLIST=clock,calculator,file_read

# Block specific tools
AURA_TOOLS_BLOCKLIST=shell,network

# Set workspace boundary
AURA_WORKSPACE_ROOT=/path/to/workspace

# Require confirmation for high-risk tools
AURA_TOOL_CONFIRM_HIGH_RISK=true
```

## Production Mode Rules

In production mode (`AURA_MODE=production`):
- Unsigned or unmanifested skills do not load
- Self-modification tools are disabled
- Shell execution requires operator-level permissions
- Network tools require explicit configuration
- All tool output is sanitized before processing
- Tool execution timeout is strictly enforced
