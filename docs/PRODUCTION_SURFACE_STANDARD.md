# Production Surface Standard

*Reviewed against the tree: 2026-08-01. See [documentation status map](DOC_STATUS.md) for how to read this file.*

This document defines the strict, static design contracts for production files in the Aura runtime. Bypassing these gates is prohibited and will cause the static validation linter to fail closed.

## 1. Gateway Isolation Contracts

All production execution paths must execute through canonical, audited owner gateways rather than raw OS or network commands:

### A. Memory Writes
* **Direct Path**: `Path.write_text` or raw filesystem file descriptors are blocked.
* **Canonical Path**: All writes must route through `core/runtime/atomic_writer.py` or a registered Memory Gateway, generating durable, transaction-safe updates.

### B. Subprocess Invocations
* **Direct Path**: `subprocess.run`, `subprocess.Popen`, or `os.system` are blocked in general production paths.
* **Canonical Path**: All subprocesses must route through a sandboxed Subprocess Gateway or the Authority Gateway, logging arguments, working directories, timeouts, and execution receipts.

### C. State Mutations
* **Direct Path**: Direct mutation of cognitive structures without state transitions is blocked.
* **Canonical Path**: All updates to internal belief/affect layers must route through `core/state/aura_state.py` or the `StateGateway`.

### D. Model Runtime & LLM Calls
* **Direct Path**: Direct creation of HTTPX/urllib clients to call LLM APIs is blocked.
* **Canonical Path**: All model routing must go through the canonical model/LLM router, enforcing rate limits, token budgets, and receipt signatures.

## 2. Coding Restrictions

- **No Swallowed Exceptions**: Banish empty `except:` or broad `except Exception:` blocks that do not log or re-raise the exception or emit a structured degradation receipt.
- **No Import-Time Async Primitives**: Async locks, queues, and loops must not be initialized globally at import time to prevent binding to the wrong/stale event loop.
- **No Async sys.exit**: Do not invoke `sys.exit` from inside async functions, as it disrupts running event loops and bypasses standard cleanup frameworks.
