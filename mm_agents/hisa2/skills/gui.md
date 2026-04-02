---
name: gui
domain: all
priority: high
when_to_use: Always load for GUI interaction tasks
---

# Skill: GUI Interaction

- Use the concrete GUI tool directly: `click`, `double_click`, `right_click`, `move`, `drag`, `type`, `press`, `hotkey`, or `scroll`.
- Put tool parameters inside `input`.
- For mouse-position actions, `input` should include a precise target description because the executor grounds coordinates from it when coordinates are omitted.
- Keep `thought`, `tool`, and the target described in `input` consistent.
- Prefer omitting coordinates and relying on a precise description unless you are highly confident in the exact screen position.
