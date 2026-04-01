---
name: gui
domain: all
priority: high
when_to_use: Always load for GUI interaction tasks
---

# Skill: GUI Interaction

- Use `gui_action` for clicks, double-clicks, right-clicks, drag, move, scroll, typing, and hotkeys.
- For mouse-position actions, `description` must clearly identify the target because the executor grounds coordinates from it.
- Keep `thought`, `description`, and `input` consistent.
- Exact APIs:
  - `pyautogui.click(x, y)`
  - `pyautogui.doubleClick(x, y)`
  - `pyautogui.rightClick(x, y)`
  - `pyautogui.moveTo(x, y)`
  - `pyautogui.moveTo(x1, y1); pyautogui.dragTo(x2, y2, duration=0.5, button='left')`
  - `pyautogui.write('text')`
  - `pyautogui.press('enter')`
  - `pyautogui.hotkey('ctrl', 'c')`
  - `pyautogui.moveTo(x, y); pyautogui.scroll(amount)` with amount in `[-10, 10]`
