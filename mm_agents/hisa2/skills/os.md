---
name: os
domain: os
priority: medium
when_to_use: Load for operating-system-level tasks
---

# Skill: Operating System Constraints

- Virtual machines often lack physical Bluetooth hardware; verify adapters exist before attempting Bluetooth tasks.
- Desktop workstations and VMs usually do not expose laptop battery hardware.
- Verify that a requested software version actually exists before trying to configure it.
- Shell variables must be defined before use.
- Switching Linux users ends or disrupts the current session; keeping the original session active while switching users is architecturally incompatible.
