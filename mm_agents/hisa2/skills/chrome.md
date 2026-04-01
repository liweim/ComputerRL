---
name: chrome
domain: chrome
priority: high
when_to_use: Load for Chrome benchmark tasks
---

# Skill: Chrome

- System-level browser changes like Flags or Languages usually require a relaunch to take effect.
- After triggering page loads, downloads, or form submissions, do a `wait` before judging the page state.
- When filtering or searching, visually verify the final state, such as URL parameters or result counts.
- Chrome settings do not control search-engine-specific behavior like results per page.
- Changing Chrome interface language requires the language pack to already exist on the system.
- On Linux, Chrome dark mode may follow OS-level appearance settings.
