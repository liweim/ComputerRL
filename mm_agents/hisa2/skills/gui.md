---
name: gui
domain: all
priority: high
when_to_use: Always load for GUI interaction tasks
---

# Skill: GUI Interaction

- Use `gui_action` for clicks, double-clicks, right-clicks, drag, move, scroll, typing, and hotkeys.
- Use flat top-level fields like `action`, `x`, `y`, `text`, `key`, `keys`, `amount`, not a nested `input` object.
- Supported actions: `click`, `double_click`, `right_click`, `move`, `drag`, `type`, `press`, `hotkey`, `scroll`.
- For mouse-position actions, `description` must clearly identify the target because the executor grounds coordinates from it when coordinates are omitted.
- Keep `thought`, `description`, and `action` consistent.
- Prefer omitting coordinates and relying on `description` unless you are highly confident in the exact screen position.
