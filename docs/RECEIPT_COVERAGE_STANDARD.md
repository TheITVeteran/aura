# Receipt Coverage Standard

*Reviewed against the tree: 2026-08-01. See [documentation status map](DOC_STATUS.md) for how to read this file.*

This document defines the requirements for structural audit and governance receipts for all consequential runtime actions executed by the Aura cognitive agent runtime.

## 1. Consequential Actions Require Receipts

A receipt is a structured, cryptographically consistent transaction log signed by the Unified Will. The following consequential actions must always produce a receipt:
1. **Model/LLM Calls**: Tracking prompts, parameters, and generated tokens.
2. **Tool Invocations**: Specifying the tool name, arguments, and return status.
3. **Memory Writes**: Tracking written content hashes and namespaces.
4. **State Mutations**: Logging affect steering updates and homeostatic shifts.
5. **Subprocess Calls**: Capturing arguments, exit codes, and stdout/stderr hashes.
6. **Self-Modification / Self-Repair**: Logging patch proposals, compile status, and rollback results.

## 2. Receipt Integrity Rules

- **Pre-Action Authorization**: A receipt must prove that the Authority Gateway approved the action *before* the action occurred. A post-action log without pre-authorization does not count as a valid receipt.
- **Fail Closed**: If receipt creation or logging fails (e.g., due to filesystem write blocks), the runtime must immediately raise a critical failure and stop execution.
- **Cryptographic Signature**: Each receipt must carry a unique transaction ID, timestamp, and signature verifying that the Unified Will authorized the decision.
