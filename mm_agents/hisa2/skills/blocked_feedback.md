---
name: blocked_feedback
domain: all
priority: medium
when_to_use: Load when recent steps failed or the agent is stuck
---

# Skill: Blocked Feedback

- If an action clearly failed, switch strategy: different target, different tool, or different command.
- Do not repeat the same failed action sequence.
- After async operations, use `wait` before judging the result.
- Use `infeasible` only when the task is objectively impossible after verification.
