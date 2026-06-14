from __future__ import annotations

"""Shared live desktop task contract.

The chat planner, response-generation prompt, and desktop_task executor all
consume this list. Keeping it in one lightweight runtime module prevents the
live UI from advertising a narrower or stale action surface than the executor
can actually govern and verify.
"""

DESKTOP_TASK_ALLOWED_ACTIONS: tuple[str, ...] = (
    "click",
    "type",
    "hotkey",
    "scroll",
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
