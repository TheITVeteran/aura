from __future__ import annotations

"""Shared live desktop task contract.

The chat planner, response-generation prompt, and desktop_task executor all
consume this list. Keeping it in one lightweight runtime module prevents the
live UI from advertising a narrower or stale action surface than the executor
can actually govern and verify.
"""

from typing import Any

DESKTOP_TASK_ALLOWED_ACTIONS: tuple[str, ...] = (
    "click",
    "type",
    "hotkey",
    "scroll",
    "inspect_screen",
    "read_screen_text",
    "read_menu_clock",
    "open_app",
    "open_url",
    "run_command",
    "set_clipboard",
    "get_clipboard",
    "wait",
    "run_applescript",
    "write_text_file",
    "render_text_pdf",
    "move_file",
    "create_folder",
    "fetch_topic_image",
    "system_control",
)

DESKTOP_TASK_RETRY_SAFE_ACTIONS: frozenset[str] = frozenset(
    {
        "create_folder",
        "get_clipboard",
        "inspect_screen",
        "open_app",
        "read_menu_clock",
        "read_screen_text",
        "wait",
    }
)


def desktop_task_action_schema() -> str:
    return "|".join(DESKTOP_TASK_ALLOWED_ACTIONS)


def desktop_task_action_sentence() -> str:
    if not DESKTOP_TASK_ALLOWED_ACTIONS:
        return ""
    if len(DESKTOP_TASK_ALLOWED_ACTIONS) == 1:
        return DESKTOP_TASK_ALLOWED_ACTIONS[0]
    return (
        ", ".join(DESKTOP_TASK_ALLOWED_ACTIONS[:-1])
        + f", and {DESKTOP_TASK_ALLOWED_ACTIONS[-1]}"
    )


def desktop_task_planning_schema() -> dict[str, Any]:
    """Return the canonical schema sent to every live cognition entrypoint."""
    return {
        "document_body": "optional prose to type, write, or export",
        "steps": [
            {
                "action": desktop_task_action_schema(),
                "target": (
                    "string or JSON payload; use {{document_body}} for composed prose "
                    "and {{steps.1.result.path}} or {{last.result.path}} for a prior "
                    "verified step result"
                ),
                "reason": "why this step is needed",
                "expect": "observable effect evidence required after the step",
                "critical": (
                    "true by default; false only when the objective can still succeed "
                    "after this step fails"
                ),
            }
        ],
    }
