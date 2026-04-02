---
name: libreoffice_impress
domain: libreoffice_impress
priority: high
when_to_use: Load for LibreOffice Impress presentation tasks
---

# Skill: LibreOffice Impress

- Close the presentation in the GUI before using scripts like `python-pptx` to modify slides.
- When specifying colors, use exact RGB or Hex values instead of visual approximation.
- Use scripts for bulk text changes or slide creation, but visually verify the final layout.
- Impress does not have built-in video export; video conversion requires external tools.
- For tasks that modify multiple slides or multiple objects, keep an explicit checklist in mind and verify each requested slide/object before finishing.
- If a numeric position or size field becomes corrupted after failed edits, stop repeating the same typing method. Clear it differently, switch to another control/dialog, or use a script-based edit path.
- If repeated drags or field edits fail on the same object, mark the subgoal `replan` or `stuck` and switch strategy instead of retrying the same interaction loop.
