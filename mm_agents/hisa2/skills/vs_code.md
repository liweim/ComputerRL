---
name: vs_code
domain: vs_code
priority: high
when_to_use: Load for VS Code tasks
---

# Skill: VS Code

- Edits to `settings.json` must remain valid JSON and be saved to disk.
- Core environment changes like display language or theme often require window reload or restart.
- VS Code supports one workspace per window; multiple `.code-workspace` files require multiple windows.
- Many advanced behaviors require extensions; check whether the requested feature is extension-dependent before attempting it through settings.
- Display language changes require the target language pack to already be installed.
- VS Code has no built-in editor visualization for complex data structures like numpy arrays without debugger support or extensions.
