---
name: libreoffice_calc
domain: libreoffice_calc
priority: high
when_to_use: Load for LibreOffice Calc spreadsheet tasks
---

# Skill: LibreOffice Calc

- Always use GUI operations for pivot tables.
- If no new sheet name is required, use `Sheet2` as the new sheet name.
- LibreOffice Calc does not support Excel-style sparklines; use regular charts or conditional formatting instead.
- For spreadsheet fill/copy tasks, verify that the destination range actually changed instead of assuming a copy or paste action worked.
- If copy-paste, fill handle, or fill-down keeps failing on the same range, mark the subgoal `replan` or `stuck` and switch to a different method, including bash/Python when appropriate.
- For tasks affecting many rows or cells, keep track of the full target range and do not stop after updating only the first visible cell.
