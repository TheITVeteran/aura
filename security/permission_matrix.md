# Aura Permission Matrix

## Overview

This document defines the Role-Based Access Control (RBAC) model for Aura.
Every operator/user interaction with Aura is governed by these permissions.

## Roles

| Role | Description | Default |
|------|-------------|---------|
| **User** | End user interacting with Aura via chat | ✅ Default |
| **Operator** | System administrator managing Aura | Requires config |
| **Admin** | Full administrative access | Requires config |
| **Research** | Research mode with sandboxed capabilities | Requires `AURA_MODE=research` |

## Permission Matrix

### Conversation & Memory

| Permission | User | Operator | Admin | Research |
|-----------|:----:|:--------:|:-----:|:--------:|
| Chat with Aura | ✅ | ✅ | ✅ | ✅ |
| View own conversation history | ✅ | ✅ | ✅ | ✅ |
| View all conversation history | ❌ | ✅ | ✅ | ❌ |
| Delete own conversations | ✅ | ✅ | ✅ | ✅ |
| Delete all conversations | ❌ | ❌ | ✅ | ❌ |
| Read own memories | ✅ | ✅ | ✅ | Sandbox |
| Read all memories | ❌ | ✅ | ✅ | Sandbox |
| Write memories | ✅ (own) | ✅ | ✅ | Sandbox |
| Delete memories | ❌ | ❌ | ✅ | ❌ |
| Export memories | ✅ (own) | ✅ | ✅ | ✅ |

### Tool & Skill Execution

| Permission | User | Operator | Admin | Research |
|-----------|:----:|:--------:|:-----:|:--------:|
| Read-only tools (clock, calc) | ✅ | ✅ | ✅ | ✅ |
| File read (workspace) | ✅ | ✅ | ✅ | ✅ |
| File write (workspace) | Limited | ✅ | ✅ | Sandbox |
| File access (outside workspace) | ❌ | ✅ | ✅ | ❌ |
| Shell (sandboxed) | ❌ | ✅ | ✅ | Sandbox |
| Shell (unrestricted) | ❌ | ❌ | ✅ | ❌ |
| Browser (read) | ✅ | ✅ | ✅ | Sandbox |
| Browser (interact) | ❌ | ✅ | ✅ | Sandbox |
| Network (external) | ❌ | ✅ | ✅ | ❌ |

### System Administration

| Permission | User | Operator | Admin | Research |
|-----------|:----:|:--------:|:-----:|:--------:|
| View health status | ✅ | ✅ | ✅ | ✅ |
| View detailed diagnostics | ❌ | ✅ | ✅ | ✅ |
| Change runtime mode | ❌ | ✅ | ✅ | ❌ |
| Change model | ❌ | ✅ | ✅ | ✅ |
| Enable/disable cloud fallback | ❌ | ✅ | ✅ | ❌ |
| Manage feature flags | ❌ | Limited | ✅ | Limited |
| Install plugins/skills | ❌ | ❌ | ✅ | ❌ |
| Remove plugins/skills | ❌ | ❌ | ✅ | ❌ |
| Trigger self-repair | ❌ | Approve | ✅ | Sandbox |
| Trigger self-modification | ❌ | ❌ | ✅ | ❌ |
| Backup/restore | ❌ | ✅ | ✅ | ❌ |
| View audit logs | ❌ | ✅ | ✅ | ✅ |
| Shutdown Aura | ❌ | ✅ | ✅ | ✅ |

## Operator Configuration

Operators configure permissions via environment or config file:

```bash
# Set operator role
AURA_ROLE=operator

# Configure workspace boundary
AURA_WORKSPACE_ROOT=/Users/me/projects

# Tool allowlist (overrides defaults)
AURA_TOOLS_ALLOWLIST=clock,calculator,file_read,file_write,browser_read

# Tool blocklist (overrides allowlist)
AURA_TOOLS_BLOCKLIST=shell_unrestricted,network_external

# Require confirmation for risky actions
AURA_CONFIRM_HIGH_RISK=true

# Cloud fallback
AURA_CLOUD_FALLBACK_POLICY=disabled
```

## Capability Statements

An operator/user should be able to express:

```text
Aura may read this folder.
Aura may not write outside this workspace.
Aura may use browser but not shell.
Aura may use shell but not network.
Aura may remember project facts but not secrets.
Aura may propose patches but not apply them.
Aura may use cloud fallback only for non-private prompts.
```

Each of these maps to a specific permission configuration.
