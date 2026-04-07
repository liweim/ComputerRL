#!/usr/bin/env python3
import ast
import base64
import builtins
import contextlib
import copy
import json
import os
import logging
import subprocess as py_subprocess
import traceback
import re
import hashlib
from typing import Optional, Dict, List, Tuple, Callable, Any
from mm_agents.hisa.llm import AbstractLLM
from utils import serialize_json, get_change_roi
from json_repair import repair_json
from utils import postprocess_action
from PIL import Image
import io
import time

GUI_ACTION_TOOLS = {
    "click",
    "double_click",
    "right_click",
    "move",
    "drag",
    "write",
    "type",
    "hotkey",
    "scroll",
}
GROUNDED_GUI_TOOLS = {"click", "double_click", "right_click", "move", "drag", "scroll"}
NON_GUI_TOOLS = {"bash_execution", "wait", "termination", "infeasible"}
VALID_TOOLS = GUI_ACTION_TOOLS | NON_GUI_TOOLS
ACTION_SCRIPT_FUNCTIONS = {
    "import_subprocess",
    "click",
    "double_click",
    "right_click",
    "move",
    "drag",
    "write",
    "hotkey",
    "scroll",
    "termination",
    "infeasible",
}


class _TerminalActionSignal(Exception):
    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class _TerminalActionSignal(Exception):
    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def _validate_scroll_amount(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("scroll amount must be an integer")
    if value < -5 or value > 5:
        raise ValueError("scroll amount must be within [-5, 5]")
    return value


def _infer_scroll_amount(description: str, default_magnitude: int = 5) -> int:
    text = f" {str(description or '').lower()} "
    if " up " in text:
        return default_magnitude
    if " down " in text:
        return -default_magnitude
    raise ValueError("scroll() requires an amount or a description containing 'up' or 'down'")


# ==================== PROMPTS ====================
GLOBAL_PLANNER_PROMPT = """You are an expert GUI and bash task agent.

Core rules:
- Do only what the task asks.
- Use as few steps as possible, but verify before termination.
- Review task, summaries, recent steps, and past patterns before deciding.
- Never change the user's requested target, object, destination, or requirement to something nearby or easier.
- Always review what the latest steps actually achieved before choosing the next action.
- Use the latest screenshot as the source of truth for the current UI state.
- If the current screenshot does not show the target yet, do not click it directly; reveal it first with scroll or another prerequisite action.
- Avoid repeating the same action when the latest screenshot and step evaluation show no meaningful new progress.
- If the previous step did not produce the expected UI change, correct the mistake instead of mechanically repeating the same action.
- Read visible text directly from screenshots when useful and record key evidence in `thought`.
- Step evaluation reports immediate UI response, not task completion. Use it to detect wrong clicks, wrong pages, or missing progress.
- Do not treat a click itself as success for the subgoal; rely on the resulting UI state shown in the screenshot.
- If the latest result is inconsistent with the intended action, analyze the likely mistake and choose a corrective action instead of repeating the same plan.
- Before termination, verify the exact requested outcome with concrete evidence from the current screenshot or a read-only command.
- Do not terminate based only on a click succeeding, a popup/toast appearing, or a page/dialog opening.
- If verification is inconclusive, continue working instead of terminating.
- Do not use `termination(...)` if your own verification message says a requirement is still missing, mismatched, or only partially satisfied.
- If the task cannot actually be completed because of a product limitation, missing capability, unsupported format, or objectively impossible constraint, use `infeasible(str)` instead of `termination(str)`.
- Explaining why a task is impossible is not completion. Do not use `termination(...)` just because you explained the limitation.

Response format:
```json
{
    "thought": "string",
    "subgoal": "string",
    "code": "string"
}
```

Field guide:
- `thought`: brief reasoning about the next action, including key evidence from the screenshot or recent history when useful.
- `subgoal`: a phase-level objective, not a single action. Use `continue` if the next action is still pursuing the current subgoal. Otherwise provide a new phase-level objective.
- Good: `Open Chrome settings`, `Configure the default search engine`, `Verify the final result`.
- Bad: `Click Settings`, `Type the filename`, `Press Enter`, `Scroll down`.
- `code`: valid Python code to execute next. You may use normal Python statements.

Code guide:
- Allowed GUI/helper functions: `click(str)`, `double_click(str)`, `right_click(str)`, `move(str)`, `drag(str, str, str)`, `write(str)`, `hotkey(str, ...)`, `scroll(str, int)`, `termination(str)`, `infeasible(str)`
- For shell commands, write normal Python code with `import subprocess` and `subprocess.run(...)`. When you pass a shell command string, use `shell=True`. Example: `import subprocess` then `subprocess.run('ls -la ~/Downloads', shell=True)`.
- For grounded GUI functions, use a detailed, self-contained instruction string that uniquely identifies the target on the current screenshot.
- Every string argument in `code` must be a valid Python string literal. Put the entire natural-language instruction inside quotes. Good: `click("Click the Search engine option in the left sidebar of Chrome Settings")`. Bad: `click("Search engine" option in the left sidebar of Chrome Settings)`
- `drag(instruction, source, destination)`: `instruction` should describe the full drag action, and `source` / `destination` should name the start and end targets clearly. Example: `drag('Drag from cell A12 to cell A22 in column A', 'cell A12', 'cell A22')`
- `scroll(description, amount)`: make the scroll direction clear in the description. `amount` must be an integer in `[-5, 5]`.
- `write(str)`: keep the entire text or formula inside one Python string.
- `hotkey(...)`: wrap keys as function calls, for example `hotkey('enter')` or `hotkey('ctrl', 'l')`.
- `subprocess.run(...)`: use it directly for shell execution.
- `termination(str)`: completion summary after the task is fully verified.
- `infeasible(str)`: objective reason the task cannot be completed. Use `infeasible(...)` when the screenshot, app error message, or command output shows the requested end state cannot be achieved in this environment.

- Response examples:
```json
{
    "thought": "The address bar is visible and focused interaction is straightforward, so I can enter the URL and submit it in one short sequence.",
    "subgoal": "Open the target website",
    "code": "click('Click the browser address bar at the top of the window')\nwrite('https://example.com')\nhotkey('enter')"
}
```

```json
{
    "thought": "The needed information is easier to verify from the shell than from the GUI, so I should run one read-only command.",
    "subgoal": "Verify the generated file",
    "code": "import subprocess\nsubprocess.run('ls -la ~/Downloads', shell=True)"
}
```"""

FIX_RESPONSE_PROMPT = """Error: Failed to parse your response.
Error message: {error_message}
{extra_guidance}

Your response was:
{response}

Please provide a valid JSON response in the exact format:
```json
{{
    "thought": "string",
    "subgoal": "string",
    "code": "string"
}}
```

Important:
- Always include `subgoal`
- Use `subgoal: "continue"` when staying on the same subgoal
- Otherwise `subgoal` must be a stage goal, not a single click or keystroke
- `code` must be valid Python code
- Every natural-language target or text value must stay inside quotes
- `scroll(...)` must include a description string; if you omit the second argument, include `up` or `down` in the description so the system can infer the direction"""

STEP_ABSTRACTION_PROMPT = """Compare the latest observations and summarize what happened in 1-2 concise sentences.

Action: {action_description}
Blocked hint: {blocked_hint}

Rules:
- Describe what changed, or say no visible change.
- Mention clear errors if shown.
- Do not judge subgoal status or long-horizon task completion.
- Keep the summary concise and concrete.
"""

FINAL_VERIFICATION_PROMPT = """Decide whether the GUI task is fully completed.

You will receive:
- the original task
- the current subgoal
- the latest execution logs
- the current screenshot

Return JSON:
{
  "result": "pass|fail"
}

Rules:
- Return only `pass` or `fail`.
- Use `pass` only if the task requirements appear fully satisfied in the current screenshot and latest execution logs.
- If there is uncertainty, return `fail`.
- The screenshot is the primary evidence. If the screenshot does not directly show the requested final state, return `fail` even if prior logs claimed success.
- Do not treat an explanation of impossibility, a shell message, or a transient status/toast as completion for a task that asked for an actual GUI, file, or configuration result.
- For multi-target tasks (`all`, `both`, `each`, `respectively`), fail unless every requested target is explicitly covered.
- For relative-date tasks, fail unless the exact resolved absolute date is explicitly covered.
"""

CONTEXT_REFINEMENT_PROMPT = """Analyze task execution progress and provide guidance.

You will receive: a task instruction, execution history range and execution history

Instructions:
- Summarize the full execution history into one unified summary covering the entire range
- List what was done in order (successes and failures)
- **IMPORTANT**: Preserve coordinates in click actions (e.g., "click(500,300)") - these can be reused later
- Identify if we're stuck in loops, making progress, or blocked
- Provide actionable suggestions for the next step if there are issues

Return a concise summary string in this format:
Steps X~Y: [ordered list of what was done, keeping coordinates]. Suggestion: [actionable advice, or 'Continue' if progressing well]

Examples:
- Steps 1~5: Opened file, tried to edit (failed 3 times with permission error), attempted sudo (failed). Suggestion: Try a different approach - copy file to temp location first.
- Steps 1~5: Clicked Submit button at click(850,620), typed text, clicked Save at click(920,580). Suggestion: Continue - forms being filled correctly.
- Steps 1~10: Previously installed package and ran script (steps 1~5). Then verified output, tested functionality (steps 6~10). Suggestion: Continue - good progress.
- Steps 1~15: Clicked the same button 5 times with no response, tried alternative buttons (failed). Suggestion: This approach isn't working - try an alternative method or termination as infeasible.
"""

PATTERN_INDUCTION_PROMPT = """Analyze this task execution and extract ONLY the most important, reusable lessons.

Task: {task_instruction}

Execution history:
{step_abstracts}

Extract ONLY verified lessons (maximum 3) that would help with similar tasks.

IMPORTANT Guidelines:
- **Data leakage prevention**: You do NOT know final success/failure - focus on execution process only
- **Only VERIFIED lessons**: If stuck on the same step for multiple attempts, record as failed approach (e.g., "DON'T use X for Y")
- **Clear evidence required**: Only include what clearly worked or clearly failed after attempts
- **No speculation**: Omit uncertain/unverified observations - if unsure, don't include
- **Focus on**: Failed methods (tried multiple times), successful strategies, critical pitfalls
- **Avoid**: Vague suggestions, unverified hypotheses, trivial details
- **Generalize lessons**: Do NOT include specific values (text content, file names, field values, etc.) - describe patterns and methods instead
- Each lesson must be specific and actionable
- Return an empty list if no significant verified lessons

Format as a JSON list of objects with type and lesson (maximum 3 items):
[
  {{"type": "domain", "lesson": "For this task family, method X worked: ..."}},
  {{"type": "env", "lesson": "In this environment, UI Y needed wait or special handling"}},
  {{"type": "failure", "lesson": "DON'T use method Y: tried 3 times, doesn't work"}}
]

Type values (ONLY these three):
- "domain": A reusable strategy for similar GUI tasks
- "env": An environment-specific quirk or UI behavior that clearly mattered
- "failure": A method/strategy that clearly failed after multiple attempts"""

PATTERN_SYNTHESIS_PROMPT = """Given the current task and past lessons from the same domain, provide a concise, refined summary of actionable advice.

Current task: {current_task}

Past lessons:
{pattern_summary}

IMPORTANT: Items marked as "REQUIREMENTS (MUST FOLLOW)" are mandatory rules that MUST be followed.

Your task:
1. **Filter** - Select ONLY the most relevant lessons for this specific task
2. **Synthesize** - Combine similar lessons into unified advice
3. **Refine** - Express advice concisely and actionably (5 bullet points maximum)
4. **Prioritize** - Focus on: mandatory requirements first, then environment quirks, then critical pitfalls, then helpful strategies
5. **Conflict Resolution** - If domain/env/failure lessons conflict with required lessons, prioritize and follow the required lessons.

Return empty string if no relevant lessons exist."""

MEMORY_SELECTION_PROMPT = """You are selecting local memory files that will clearly help with the current GUI task.

You will receive:
- the current task
- the current task signature and tags
- a list of candidate memory entries with zero-based list indices, type, and short description

Return a JSON object:
{
  "selected": [0, 2]
}

Rules:
- Select at most 5 files.
- Be selective. Only choose memories that are clearly useful for this specific task.
- Prefer environment quirks, workflow tactics, and failure warnings over generic notes.
- Prefer memories whose task tags align with the current task tags.
- Do not include require entries; those are injected separately."""

# ==================== PATTERN MANAGER ====================

class PatternManager:
    """Manage file-based task memories by software/domain."""

    def __init__(
        self,
        llm: Optional[AbstractLLM] = None,
        prompt_dump_callback: Optional[Callable[..., None]] = None,
        qdrant_path: str = "./qdrant_storage",
        embedding_service_url: str = "http://localhost:8888",
        similarity_threshold: float = 0.7,
        use_qdrant_server: bool = True,  # Default to server mode for multi-process
        qdrant_server_url: str = "http://localhost:6333"
    ):
        self.llm = llm
        self.prompt_dump_callback = prompt_dump_callback
        self.logger = logging.getLogger("desktopenv.pattern")
        base_memory_root = qdrant_path or os.path.join(os.path.dirname(__file__), "memories")
        self.memory_root = os.path.abspath(base_memory_root)
        os.makedirs(self.memory_root, exist_ok=True)
        self.logger.info(
            f"File memory initialized. memory_root={self.memory_root}"
        )

    def _dump_prompt(self, stage: str, payload: Any, **metadata) -> None:
        if self.prompt_dump_callback:
            try:
                self.prompt_dump_callback(stage=stage, payload=payload, **metadata)
            except Exception as e:
                self.logger.warning(f"Failed to dump prompt for stage {stage}: {e}")

    def _normalize_domain(self, domain: str) -> str:
        domain = (domain or "general").strip().lower()
        domain = re.sub(r"[^a-z0-9_]+", "_", domain)
        domain = re.sub(r"_+", "_", domain).strip("_")
        return domain or "general"

    def _normalize_memory_prefix(self, domain: str, task_id: str) -> str:
        normalized_domain = self._normalize_domain(domain)
        normalized_task_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(task_id or "unknown").strip())
        normalized_task_id = re.sub(r"_+", "_", normalized_task_id).strip("_") or "unknown"
        return f"{normalized_domain}_{normalized_task_id}"

    def _get_memory_dir(self, domain: str) -> str:
        return os.path.join(self.memory_root, self._normalize_domain(domain))

    def _ensure_memory_dir(self, domain: str) -> str:
        memory_dir = self._get_memory_dir(domain)
        os.makedirs(memory_dir, exist_ok=True)
        return memory_dir

    def _load_seed_requirements(self, domain: str) -> List[Dict]:
        return []

    def _parse_memory_file(self, file_path: str) -> Optional[Dict]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw = f.read().strip()
        except Exception as e:
            self.logger.warning(f"Failed to read memory file {file_path}: {e}")
            return None

        if not raw:
            return None

        metadata = {}
        body = raw
        if raw.startswith("---\n"):
            parts = raw.split("\n---\n", 1)
            if len(parts) == 2:
                header, body = parts
                for line in header.splitlines()[1:]:
                    if ":" not in line:
                        continue
                    key, value = line.split(":", 1)
                    metadata[key.strip()] = value.strip()

        lesson = body.strip()
        if not lesson:
            return None

        return {
            "filename": os.path.basename(file_path),
            "path": file_path,
            "type": metadata.get("type", "domain"),
            "description": metadata.get("description", lesson),
            "confidence": metadata.get("confidence", ""),
            "task_signature": metadata.get("task_signature", ""),
            "task_tags": [tag.strip() for tag in metadata.get("task_tags", "").split(",") if tag.strip()],
            "source_task_id": metadata.get("source_task_id", ""),
            "lesson": lesson,
            "source": "memory",
            "mtime": os.path.getmtime(file_path),
        }

    def _load_memory_entries(self, domain: str) -> List[Dict]:
        memory_dir = self._ensure_memory_dir(domain)
        entries = []
        for file_name in sorted(os.listdir(memory_dir)):
            if not file_name.endswith(".md") or file_name == "MEMORY.md":
                continue
            entry = self._parse_memory_file(os.path.join(memory_dir, file_name))
            if entry:
                entries.append(entry)
        entries.sort(key=lambda item: item.get("mtime", 0), reverse=True)
        return entries

    def _write_memory_index(self, domain: str, entries: List[Dict]) -> None:
        memory_dir = self._ensure_memory_dir(domain)
        index_path = os.path.join(memory_dir, "MEMORY.md")
        lines = [f"# Memory Index: {self._normalize_domain(domain)}", ""]
        for entry in entries:
            lines.append(
                f"- [{entry['filename']}]({entry['filename']}) [{entry.get('type', 'domain')}] - {entry.get('description', '')}"
            )
        with open(index_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines).strip() + "\n")

    def _score_memory_entry(
        self,
        entry: Dict,
        task_signature: str = "",
        task_tags: Optional[List[str]] = None,
    ) -> float:
        score = 0.0
        entry_tags = set(entry.get("task_tags", []))
        current_tags = set(task_tags or [])
        overlap = len(entry_tags & current_tags)
        score += overlap * 10.0

        if task_signature and entry.get("task_signature") == task_signature:
            score += 12.0

        confidence = (entry.get("confidence") or "").lower()
        if confidence == "high":
            score += 3.0
        elif confidence == "medium":
            score += 1.5

        entry_type = (entry.get("type") or "").lower()
        if entry_type == "env":
            score += 2.0
        elif entry_type == "failure":
            score += 1.0

        mtime = float(entry.get("mtime") or 0.0)
        if mtime > 0:
            age_days = max(0.0, (time.time() - mtime) / 86400.0)
            score -= min(age_days * 0.05, 3.0)

        return score

    def _prefilter_memory_entries(
        self,
        entries: List[Dict],
        task_signature: str = "",
        task_tags: Optional[List[str]] = None,
        limit: int = 12,
    ) -> List[Dict]:
        if not entries:
            return []

        enriched = []
        for entry in entries:
            scored = dict(entry)
            scored["_score"] = self._score_memory_entry(
                scored,
                task_signature=task_signature,
                task_tags=task_tags,
            )
            enriched.append(scored)

        enriched.sort(
            key=lambda item: (
                item.get("_score", 0.0),
                item.get("mtime", 0.0),
            ),
            reverse=True,
        )

        strong_matches = [item for item in enriched if item.get("_score", 0.0) > 0][:limit]
        if len(strong_matches) >= min(5, limit):
            return strong_matches
        return enriched[:limit]

    def save_pattern(
        self,
        domain: str,
        lessons: List[Dict],
        task_instruction: str = "",
        task_id: str = "",
        task_signature: str = "",
        task_tags: Optional[List[str]] = None,
    ):
        """Save learned lessons as local markdown files under the software/domain directory."""
        try:
            normalized_domain = self._normalize_domain(domain)
            memory_dir = self._ensure_memory_dir(normalized_domain)
            existing_entries = self._load_memory_entries(normalized_domain)
            existing_keys = {
                (entry.get("type", "domain"), re.sub(r"\s+", " ", entry.get("lesson", "").strip().lower()))
                for entry in existing_entries
            }
            existing_by_type = {}
            for entry in existing_entries:
                existing_by_type.setdefault(entry.get("type", "domain"), []).append(entry)

            added_count = 0
            removed_count = 0
            for lesson_obj in lessons:
                lesson_text = (lesson_obj.get("lesson") or "").strip()
                lesson_type = (lesson_obj.get("type") or "domain").strip().lower()
                if not lesson_text or lesson_type == "require":
                    continue

                dedup_key = (lesson_type, re.sub(r"\s+", " ", lesson_text.lower()))
                if dedup_key in existing_keys:
                    continue

                file_stub = hashlib.sha256(f"{lesson_type}:{lesson_text}".encode("utf-8")).hexdigest()[:10]
                memory_prefix = self._normalize_memory_prefix(normalized_domain, task_id or "unknown")
                file_name = f"{memory_prefix}_{lesson_type}_{int(time.time())}_{file_stub}.md"
                file_path = os.path.join(memory_dir, file_name)
                description = lesson_text
                typed_signature = task_signature or self._normalize_domain(domain)
                tag_list = ",".join(task_tags or [])
                confidence = "high" if lesson_type in ["env", "failure"] else "medium"
                content = (
                    "---\n"
                    f"type: {lesson_type}\n"
                    f"software: {normalized_domain}\n"
                    f"description: {description}\n"
                    f"confidence: {confidence}\n"
                    f"task_signature: {typed_signature}\n"
                    f"task_tags: {tag_list}\n"
                    f"source_task_id: {task_id or 'unknown'}\n"
                    f"created_at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    "---\n\n"
                    f"{lesson_text}\n"
                )
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                existing_keys.add(dedup_key)
                added_count += 1
                existing_by_type.setdefault(lesson_type, []).append({
                    "filename": file_name,
                    "path": file_path,
                    "type": lesson_type,
                    "description": description,
                    "lesson": lesson_text,
                    "confidence": confidence,
                    "task_signature": typed_signature,
                    "task_tags": task_tags or [],
                    "source_task_id": task_id or "unknown",
                    "mtime": os.path.getmtime(file_path),
                })

            # Keep each software/type bucket bounded so memories do not grow without limit.
            for lesson_type, type_entries in existing_by_type.items():
                type_entries.sort(
                    key=lambda item: (
                        self._score_memory_entry(
                            item,
                            task_signature=task_signature,
                            task_tags=task_tags,
                        ),
                        item.get("mtime", 0.0),
                    ),
                    reverse=True,
                )
                for stale_entry in type_entries[20:]:
                    stale_path = stale_entry.get("path")
                    if stale_path and os.path.exists(stale_path):
                        try:
                            os.remove(stale_path)
                            removed_count += 1
                        except Exception as e:
                            self.logger.warning(f"Failed to prune stale memory file {stale_path}: {e}")

            updated_entries = self._load_memory_entries(normalized_domain)
            self._write_memory_index(normalized_domain, updated_entries)
            self.logger.info(
                f"File memory update for domain {normalized_domain}: added {added_count} new lesson(s), pruned {removed_count} old lesson(s)"
            )
        except Exception as e:
            self.logger.error(f"Failed to save file-based memory: {e}")
            raise

    def pattern_induction(self, task_instruction: str, action_logs: List[Dict]) -> List[str]:
        """Use LLM to extract key lessons from task execution.

        Returns:
            List of lesson strings
        """
        if not self.llm:
            return []

        step_abstracts = []
        for log in action_logs:
            compact = log.get("compact") or {}
            step = log.get("step", "?")
            tool_type = log.get("type", "")
            result = "success" if log.get("execution_success", False) else "failure"
            detail = (log.get("detail") or "").strip()
            intent = (compact.get("intent") or "").strip()
            verified = (compact.get("verified") or "").strip()
            next_hint = (compact.get("next_hint") or "").strip()

            parts = [f"Step {step}: tool={tool_type}; result={result}"]
            if detail:
                parts.append(f"detail={detail}")
            if intent:
                parts.append(f"intent={intent}")
            if verified:
                parts.append(f"verified={verified}")
            if next_hint:
                parts.append(f"next_hint={next_hint}")
            step_abstracts.append(" | ".join(parts))

        prompt = PATTERN_INDUCTION_PROMPT.format(
            task_instruction=task_instruction,
            step_abstracts='\n'.join(step_abstracts)
        )

        try:
            messages = [
                {"role": "system", "content": "You are an expert at analyzing task execution patterns and extracting the most critical, reusable lessons. Be highly selective - only extract truly valuable insights. CRITICAL: focus only on the execution process."},
                {"role": "user", "content": prompt}
            ]
            self._dump_prompt(
                stage="pattern_induction",
                payload={"messages": messages},
            )

            response = self.llm(messages, enable_thinking=True)
            self._dump_prompt(
                stage="pattern_induction_response",
                payload=response,
            )

            if not isinstance(response, str) or not response.strip():
                self.logger.warning("Pattern induction returned empty response")
                return []

            # Try to parse as JSON
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            else:
                json_str = response.strip()

            lessons = json.loads(repair_json(json_str))
            if isinstance(lessons, list):
                # Validate that each item is a dict with 'type' and 'lesson'
                validated_lessons = []
                for item in lessons[:3]:  # Max 3 lessons
                    if isinstance(item, dict) and "type" in item and "lesson" in item:
                        if item["type"] in ["domain", "env", "failure", "require", "success"]:
                            if item["type"] == "success":
                                item = {"type": "domain", "lesson": item["lesson"]}
                            validated_lessons.append(item)
                        else:
                            self.logger.warning(f"Invalid lesson type '{item['type']}', skipping")
                    else:
                        self.logger.warning(f"Invalid lesson format: {item}, skipping")
                if validated_lessons:
                    self.logger.info(
                        f"Pattern induction extracted {len(validated_lessons)} lesson(s)"
                    )
                else:
                    self.logger.info("Pattern induction extracted no valid lessons")
                return validated_lessons
            else:
                self.logger.warning(f"Expected list, got {type(lessons)}")
                return []

        except Exception as e:
            self.logger.error(f"Failed to extract lessons: {e}")
            return []

    def get_relevant_pattern(
        self,
        domain: str,
        current_task: str,
        task_signature: str = "",
        task_tags: Optional[List[str]] = None,
    ) -> str:
        """Retrieve relevant memories from learned markdown memories."""
        try:
            normalized_domain = self._normalize_domain(domain)
            require_patterns = self._load_seed_requirements(normalized_domain)
            learned_entries = self._load_memory_entries(normalized_domain)
            learned_entries = self._prefilter_memory_entries(
                learned_entries,
                task_signature=task_signature,
                task_tags=task_tags,
                limit=20,
            )

            selected_entries = []
            if learned_entries and self.llm:
                manifest_lines = [
                    (
                        f"- {idx} [{entry.get('type', 'domain')}, confidence={entry.get('confidence', '')}, "
                        f"signature={entry.get('task_signature', '')}, tags={','.join(entry.get('task_tags', []))}]: "
                        f"{entry.get('description', '')}"
                    )
                    for idx, entry in enumerate(learned_entries)
                ]
                messages = [
                    {"role": "system", "content": MEMORY_SELECTION_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Task: {current_task}\n\n"
                            f"Task signature: {task_signature}\n"
                            f"Task tags: {', '.join(task_tags or [])}\n\n"
                            "Candidate memory entries:\n" + "\n".join(manifest_lines)
                        )
                    }
                ]
                try:
                    self._dump_prompt(
                        stage="memory_selection",
                        payload={"messages": messages},
                    )
                    response = self.llm(messages, enable_thinking=False)
                    self._dump_prompt(
                        stage="memory_selection_response",
                        payload=response,
                    )
                    json_str = response.strip()
                    if "```json" in response:
                        json_start = response.find("```json") + 7
                        json_end = response.find("```", json_start)
                        json_str = response[json_start:json_end].strip()
                    elif "```" in response:
                        json_start = response.find("```") + 3
                        json_end = response.find("```", json_start)
                        json_str = response[json_start:json_end].strip()
                    parsed = json.loads(repair_json(json_str))
                    selected_indices = parsed.get("selected", []) if isinstance(parsed, dict) else []
                    normalized_indices = []
                    for item in selected_indices:
                        try:
                            idx = int(item)
                        except (TypeError, ValueError):
                            continue
                        if 0 <= idx < len(learned_entries) and idx not in normalized_indices:
                            normalized_indices.append(idx)
                    selected_entries = [learned_entries[idx] for idx in normalized_indices[:5]]
                except Exception as e:
                    self.logger.warning(f"Local memory selection failed, falling back to recency: {e}")
                    selected_entries = learned_entries[:5]
            elif learned_entries:
                selected_entries = learned_entries[:5]

            pattern_summary = []
            if require_patterns:
                pattern_summary.append("\n--- REQUIREMENTS (MUST FOLLOW) ---")
                for pattern in require_patterns:
                    pattern_summary.append(pattern["lesson"])

            grouped = {
                "env": [],
                "domain": [],
                "failure": [],
            }
            for entry in selected_entries:
                grouped.setdefault(entry.get("type", "domain"), []).append(entry)

            if grouped.get("env"):
                pattern_summary.append("\n--- ENVIRONMENT QUIRKS ---")
                for pattern in grouped["env"]:
                    pattern_summary.append(pattern["lesson"])

            if grouped.get("domain"):
                pattern_summary.append("\n--- DOMAIN STRATEGIES ---")
                for pattern in grouped["domain"]:
                    pattern_summary.append(pattern["lesson"])

            if grouped.get("failure"):
                pattern_summary.append("\n--- FAILURE Patterns ---")
                for pattern in grouped["failure"]:
                    pattern_summary.append(pattern["lesson"])

            if not pattern_summary:
                return ""

            prompt = PATTERN_SYNTHESIS_PROMPT.format(
                current_task=current_task,
                pattern_summary="\n".join(pattern_summary)
            )
            if not self.llm:
                return "\n".join(pattern_summary)

            try:
                messages = [
                    {"role": "system", "content": "You are an expert at analyzing past lessons and providing actionable advice for new tasks."},
                    {"role": "user", "content": prompt}
                ]
                self._dump_prompt(
                    stage="pattern_synthesis",
                    payload={"messages": messages},
                )
                response = self.llm(messages, enable_thinking=False)
                self._dump_prompt(
                    stage="pattern_synthesis_response",
                    payload=response,
                )
                self.logger.info(
                    f"Retrieved file memories for domain {normalized_domain}: "
                    f"require={len(require_patterns)}, learned_selected={len(selected_entries)}"
                )
                return response.strip()
            except Exception as e:
                self.logger.error(f"Failed to summarize file memories: {e}")
                return "\n".join(pattern_summary)
        except Exception as e:
            self.logger.error(f"Failed to get relevant file memories: {e}")
            raise

# ==================== AGENT FRAMEWORK ====================

class HiSA:
    """Cognitive Memory Model Agent."""

    def __init__(
        self,
        env,
        global_planner_model: str = "gpt-5",
        visual_grounder_model: str = "gta1-7b",
        visual_grounder_scale: float = 1.0,
        state_manager_model: str = "gpt-5-mini",
        client_password: str = "password",
        screen_width: int = 1920,
        screen_height: int = 1080,
        sleep_after_execution: float = 0.5,
        max_steps: int = 15,
        save_dir: str = "",
        record: bool = False,
        max_parse_retries: int = 3,
        wo_pattern: bool = False,  # If True, disable pattern induction (default: False means pattern induction is enabled)
        pattern_dir: str = "",
        use_qdrant_server: bool = False,  # Use server mode by default for multi-process
        qdrant_server_url: str = "http://localhost:6333",
        wo_roi: bool = False,  # If True, disable ROI cropping (default: False means ROI cropping is enabled)
        roi_margin: int = 50,  # Margin around ROI when cropping
        refine_period: int = 10,
        bash_timeout: int = 60,  # Timeout for bash script execution in seconds
        bash_working_dir: str = "~",  # Working directory for bash execution
        wo_step: bool = False,  # If True, skip step abstraction and use full conversation history
        wo_refinement: bool = False,  # If True, disable context refinement and use sliding window
        sliding_window_size: int = 5,  # Sliding window size (number of conversation turns to keep)
    ):
        self.env = env
        self.global_planner_model = global_planner_model
        self.visual_grounder_model = visual_grounder_model
        self.visual_grounder_scale = visual_grounder_scale
        self.state_manager_model = state_manager_model
        self.client_password = client_password
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.sleep_after_execution = sleep_after_execution
        self.max_steps = max_steps
        self.save_dir = save_dir
        self.record = record
        self.max_parse_retries = max_parse_retries
        self.wo_pattern = wo_pattern  # If True, disable pattern induction (default: False means pattern induction is enabled)
        self.wo_roi = wo_roi  # If True, disable ROI cropping (default: False means ROI cropping is enabled)
        self.roi_margin = roi_margin
        self.refine_period = refine_period
        self.bash_timeout = bash_timeout  # Timeout for bash script execution
        self.bash_working_dir = bash_working_dir
        self.wo_step = wo_step  # Skip step abstraction if True
        self.wo_refinement = wo_refinement  # Disable context refinement if True
        self.sliding_window_size = sliding_window_size  # Sliding window size for conversation history

        self.logger = logging.getLogger("desktopenv")
        self.skills_dir = os.path.join(os.path.dirname(__file__), "skills")

        # Initialize LLM clients
        self.global_planner_llm = AbstractLLM(global_planner_model, logger=self.logger)
        self.visual_grounder_llm = AbstractLLM(visual_grounder_model, logger=self.logger)
        self.state_manager_llm = AbstractLLM(state_manager_model, logger=self.logger)

        # Initialize pattern manager
        if not pattern_dir:
            pattern_dir = os.path.join(os.path.dirname(__file__), "memories")

        if not self.wo_pattern:
            self.pattern_manager = PatternManager(
                llm=self.global_planner_llm,
                prompt_dump_callback=self._dump_prompt_entry,
                qdrant_path=pattern_dir,
                similarity_threshold=0.7,
                use_qdrant_server=use_qdrant_server,
                qdrant_server_url=qdrant_server_url
            )
            self.logger.info(f"Pattern manager initialized")

        # Execution state
        self.operation_count = 0
        self.operations_dir = ""
        self.action_logs = []
        self.last_error_feedback = None  # Backward-compatible alias; planner uses blocked feedback semantics
        self.last_blocked_feedback = None
        self.last_full_summary = None  # Last complete history summary
        self.last_summary_log_index = 0  # Number of action logs already folded into last_full_summary
        self.last_refinement_log_count = 0
        self.last_refinement_step = 0
        self.step_token_usage = {}  # Store token usage for current step
        self.current_thought = ""  # Store current step's thought for step_abstract
        self.current_proposed_subgoal = ""
        self.last_tool_output = None  # Store last tool execution result for wo_step mode
        self.prompt_dump_path = ""
        self.prompt_dump_counter = 0
        self.current_subgoal = ""
        self.consecutive_stuck_subgoals = 0
        self.awaiting_final_verification = False
        self.final_verification_observed = False
        self.last_execution_status = "continue"
        self.last_blocked_feedback_event = None
        self.last_blocked_feedback = None
        self.last_error_feedback = None
        self.current_step_id = 0
        self.post_action_wait_timeout = 10.0
        self.explicit_wait_timeout = 20.0
        self.wait_poll_interval = 1.0
        self.screenshot_wait_timeout = 6.0
        self.evaluation_wait_timeout = 25.0
        self.evaluation_settle_seconds = 10.0

    def _sanitize_prompt_payload(self, value: Any):
        if isinstance(value, dict):
            sanitized = {}
            for key, item in value.items():
                if key == "image_url" and isinstance(item, str) and item.startswith("data:image/"):
                    sanitized[key] = f"<omitted data image url, chars={len(item)}>"
                else:
                    sanitized[key] = self._sanitize_prompt_payload(item)
            return sanitized
        if isinstance(value, list):
            return [self._sanitize_prompt_payload(item) for item in value]
        if isinstance(value, bytes):
            return f"<omitted bytes, len={len(value)}>"
        return value

    def _init_prompt_dump_file(self) -> None:
        self.prompt_dump_counter = 0
        base_dir = self.save_dir or "."
        self.prompt_dump_path = os.path.join(base_dir, "model_trace.txt")
        os.makedirs(base_dir, exist_ok=True)
        header = [
            "# Model Log",
            f"task_id: {getattr(self, 'current_task_id', '')}",
            f"domain: {getattr(self, 'current_domain', '')}",
            f"signature: {getattr(self, 'current_task_signature', '')}",
            f"created_at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]
        with open(self.prompt_dump_path, "w", encoding="utf-8") as f:
            f.write("\n".join(header))

    def _dump_prompt_entry(self, stage: str, payload: Any, step: Optional[int] = None, attempt: Optional[int] = None, **metadata) -> None:
        if not self.prompt_dump_path:
            return
        self.prompt_dump_counter += 1
        block_meta = {
            "index": self.prompt_dump_counter,
            "stage": stage,
            "step": step if step is not None else self.operation_count + 1,
        }
        if attempt is not None:
            block_meta["attempt"] = attempt
        for key, value in metadata.items():
            if value is not None:
                block_meta[key] = value

        sanitized_payload = self._sanitize_prompt_payload(payload)
        with open(self.prompt_dump_path, "a", encoding="utf-8") as f:
            f.write(f"\n## Prompt {self.prompt_dump_counter:04d}\n")
            for key, value in block_meta.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            if isinstance(sanitized_payload, str):
                f.write(sanitized_payload.rstrip() + "\n")
            else:
                f.write(json.dumps(serialize_json(sanitized_payload), indent=2, ensure_ascii=False))
                f.write("\n")

    def _load_skill_text(self, name: str) -> str:
        path = os.path.join(self.skills_dir, name)
        if not os.path.exists(path):
            self.logger.warning(f"Skill file missing: {path}")
            return ""
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read().strip()
        except Exception as e:
            self.logger.warning(f"Failed to load skill file {path}: {e}")
            return ""

        metadata = {}
        body = raw
        if raw.startswith("---\n"):
            parts = raw.split("\n---\n", 1)
            if len(parts) == 2:
                header, body = parts
                for line in header.splitlines()[1:]:
                    if ":" not in line:
                        continue
                    key, value = line.split(":", 1)
                    metadata[key.strip().lower()] = value.strip()

        skill_domain = metadata.get("domain", "").lower()
        current_domain = (getattr(self, "current_domain", "") or "").lower()
        if skill_domain and skill_domain not in ["all", current_domain]:
            return ""
        return body.strip()

    def _skill_file_exists(self, name: str) -> bool:
        if not name:
            return False
        return os.path.exists(os.path.join(self.skills_dir, name))

    def _get_domain_skill_name(self) -> str:
        domain = getattr(self, "current_domain", "") or getattr(self, "task_domain", "") or ""
        domain = re.sub(r"[^a-z0-9_]+", "_", domain.strip().lower())
        domain = re.sub(r"_+", "_", domain).strip("_")
        return f"{domain}.md" if domain else ""

    def _task_prefers_gui_skill(self) -> bool:
        task_text = getattr(self, "task_instruction", "").lower()
        domain = (getattr(self, "current_domain", "") or "").lower()

        strong_gui_hints = [
            "browser", "chrome", "tab", "menu", "button", "dropdown", "dialog",
            "window", "settings", "preferences", "address bar", "toolbar",
            "click", "double-click", "right-click", "drag", "scroll", "hover",
            "open the app", "navigate to", "select from the menu", "toggle",
            "pivot table", "pivot chart",
        ]
        gui_domains = {
            "chrome", "thunderbird", "vlc", "gimp",
        }
        return any(token in task_text for token in strong_gui_hints) or domain in gui_domains

    def _task_requires_bash_skill(self) -> bool:
        task_text = getattr(self, "task_instruction", "").lower()
        if "pivot table" in task_text or "pivot chart" in task_text:
            return False
        bash_hints = [
            "file", "code", "python", "bash", "terminal", "script",
            "excel", "calc", "spreadsheet", "csv", "json", "yaml",
            "docx", "xlsx", "tsv", "modify", "edit", "replace", "update",
            "writer", "document", "paragraph", "cell", "column", "row",
        ]
        return any(token in task_text for token in bash_hints)

    def _extract_task_tags(self, task_instruction: str, domain: str = "") -> List[str]:
        text = f"{domain} {task_instruction}".lower()
        tag_rules = {
            "browser": ["chrome", "browser", "tab", "page", "website", "search"],
            "email": ["email", "mail", "smtp", "inbox", "thunderbird"],
            "settings": ["setting", "preferences", "language", "theme", "config"],
            "edit": ["edit", "modify", "replace", "update", "change"],
            "verify": ["verify", "confirm", "check", "ensure", "validate"],
            "search_filter": ["search", "filter", "sort", "find", "lookup"],
            "sheet": ["sheet", "spreadsheet", "cell", "column", "row", "calc", "excel"],
            "slide": ["slide", "presentation", "impress", "ppt"],
            "document": ["document", "writer", "docx", "paragraph", "heading"],
            "image": ["image", "photo", "gimp", "resize", "crop", "color"],
            "code": ["code", "script", "python", "notebook", "jupyter", "vs code", "vscode"],
            "file_io": ["file", "save", "export", "import", "download", "upload"],
        }
        tags = []
        for tag, keywords in tag_rules.items():
            if any(keyword in text for keyword in keywords):
                tags.append(tag)
        if domain:
            tags.append(re.sub(r"[^a-z0-9_]+", "_", domain.strip().lower()))
        deduped = []
        seen = set()
        for tag in tags:
            if tag and tag not in seen:
                seen.add(tag)
                deduped.append(tag)
        return deduped

    def _build_task_signature(self, task_instruction: str, domain: str = "") -> str:
        tags = self._extract_task_tags(task_instruction, domain)
        return "|".join(tags) if tags else (domain or "general")

    def _should_use_blocked_feedback_skill(self) -> bool:
        if self._get_active_blocked_feedback():
            return True
        recent_logs = self.action_logs[-2:]
        return any(not log.get("execution_success", True) for log in recent_logs)

    def _get_active_blocked_feedback(self) -> str:
        return str(self.last_blocked_feedback or self.last_error_feedback or "").strip()

    def _build_planner_system_prompt(self) -> str:
        sections = [
            GLOBAL_PLANNER_PROMPT,
        ]
        domain_skill_name = self._get_domain_skill_name()
        if domain_skill_name and self._skill_file_exists(domain_skill_name):
            sections.append(self._load_skill_text(domain_skill_name))
        if self._should_use_blocked_feedback_skill() and self._skill_file_exists("blocked_feedback.md"):
            sections.append(self._load_skill_text("blocked_feedback.md"))
        return "\n\n".join(section.strip() for section in sections if section)

    def _normalize_subgoal(self, value: str) -> str:
        text = re.sub(r"\s+", " ", str(value or "").strip())
        if not text:
            raise ValueError("Subgoal cannot be empty")
        return text

    def _normalize_execution_status(self, value: str) -> str:
        status = re.sub(r"\s+", " ", str(value or "").strip().lower())
        if status not in {"continue", "done", "blocked", "finish"}:
            return "continue"
        return status

    def _resolve_planner_subgoal(self, value: str, code: str = "") -> str:
        text = re.sub(r"\s+", " ", str(value or "").strip())
        if not text:
            raise ValueError("Subgoal cannot be empty")
        if text.lower() == "continue":
            if "termination(" in str(code or ""):
                return self.current_subgoal or "Verify task completion"
            if not self.current_subgoal:
                raise ValueError("subgoal='continue' is invalid before an initial subgoal is established")
            return self.current_subgoal
        return self._normalize_subgoal(text)

    def _normalize_blocking_reason(self, value: str) -> str:
        reason = re.sub(r"\s+", " ", str(value or "").strip().lower())
        if reason not in {"behavior_loop", "progress_stall"}:
            return ""
        return reason

    def _parse_abstraction_payload(self, raw_text: str) -> str:
        summary = re.sub(r"\s+", " ", str(raw_text or "").strip())
        return summary or "Step abstraction failed due to error."

    def _build_subgoal_context_lines(self) -> List[str]:
        lines = []
        if self.current_subgoal:
            lines.append(f"Current subgoal: {self.current_subgoal}")
        if self.awaiting_final_verification and not self.final_verification_observed:
            lines.append("Final verification is still required before termination. Use one more action to verify the exact final state.")
        return lines

    def _append_context_section(self, sections: List[str], title: str, content: Optional[str]) -> None:
        text = str(content or "").strip()
        if text:
            sections.append(f"{title}:\n{text}")

    def _get_current_date_context(self) -> str:
        return time.strftime("%Y-%m-%d (%A)")

    def _build_planner_context_message(
        self,
        *,
        include_task_header: bool,
        history_items: Optional[List[str]] = None,
        prompt_text: str = "",
        observation_text: str = "",
        blocked_feedback_text: str = "",
    ) -> Optional[Dict]:
        sections: List[str] = []

        if include_task_header:
            self._append_context_section(sections, "Task", self.task_instruction)
            self._append_context_section(sections, "Current date", self._get_current_date_context())
            self._append_context_section(sections, "Relevant past patterns", self.past_pattern_text)
            if not self.wo_refinement and self.last_full_summary:
                self._append_context_section(sections, "Summary of previous steps", self.last_full_summary)

        subgoal_lines = self._build_subgoal_context_lines()
        if subgoal_lines:
            sections.append("\n\n".join(subgoal_lines))

        if history_items:
            history_text = "\n".join(str(item).strip() for item in history_items if str(item).strip())
            self._append_context_section(sections, "Execution history", history_text)

        self._append_context_section(sections, "Observation from previous action", observation_text)
        self._append_context_section(sections, "Blocked feedback", blocked_feedback_text)

        prompt_text = str(prompt_text or "").strip()
        if prompt_text:
            sections.append(prompt_text)

        if not sections:
            return None
        return {"role": "user", "content": "\n\n".join(sections)}

    def _build_action_description(self, decision: Dict) -> str:
        action = str(decision.get("tool", "")).strip().lower()
        thought = re.sub(r"\s+", " ", str(decision.get("thought", "") or "").strip())
        tool_input = decision.get("input")
        input_desc = ""
        if isinstance(tool_input, str) and action in GROUNDED_GUI_TOOLS:
            input_desc = re.sub(r"\s+", " ", tool_input.strip())

        if input_desc:
            return input_desc

        base_desc = f"{thought} [{action}]" if thought and action in GROUNDED_GUI_TOOLS else thought
        if base_desc:
            if action == "scroll":
                return f"Scroll target: {base_desc}"
            if action in {"click", "double_click", "right_click"}:
                return f"Click target: {base_desc}"
            if action == "move":
                return f"Hover target: {base_desc}"
            if action == "drag":
                return f"Drag target: {base_desc}"
            return base_desc
        return ""

    def _is_gui_tool(self, tool: str) -> bool:
        return str(tool or "").strip().lower() in GUI_ACTION_TOOLS

    def _build_compact_log_entry(
        self,
        step: int,
        tool_type: str,
        success: bool,
        detail: str,
        verification: str = "",
        next_hint: str = "",
    ) -> Dict:
        detail = re.sub(r"\s+", " ", str(detail or "").strip())
        verification = re.sub(r"\s+", " ", str(verification or "").strip())
        next_hint = re.sub(r"\s+", " ", str(next_hint or "").strip())
        return {
            "intent": self.current_thought if self.current_thought else "",
            "verified": verification,
            "next_hint": next_hint,
        }

    def _render_compact_log(self, log: Dict) -> str:
        compact = log.get("compact") or {}
        parts = [
            f"Step {log.get('step', '?')}",
            f"tool={log.get('type', '')}",
            f"result={'success' if log.get('execution_success', False) else 'failure'}",
        ]
        if log.get("subgoal"):
            parts.append(f"subgoal={log.get('subgoal')}")
        if log.get("execution_status"):
            parts.append(f"execution_status={log.get('execution_status')}")
        if log.get("blocking_reason"):
            parts.append(f"blocking_reason={log.get('blocking_reason')}")
        detail = log.get("detail") or compact.get("detail") or ""
        if detail:
            parts.append(f"detail={detail}")
        if compact.get("intent"):
            parts.append(f"intent={compact['intent']}")
        if compact.get("verified"):
            parts.append(f"verified={compact['verified']}")
        if compact.get("next_hint"):
            parts.append(f"next_hint={compact['next_hint']}")
        return " | ".join(parts)

    def _maybe_refine_context(self, reason: str = "") -> None:
        total_logs = len(self.action_logs)
        if self.wo_refinement or total_logs <= 0:
            return

        if reason not in {"done", "blocked"}:
            return

        current_end_step = int(self.action_logs[-1]["step"])
        last_refinement_step = int(getattr(self, "last_refinement_step", 0) or 0)
        if current_end_step - last_refinement_step <= 3:
            return

        if self.last_full_summary:
            logs_to_summarize = self.action_logs[self.last_summary_log_index:]
            start_step = self.action_logs[0]["step"]
            end_step = self.action_logs[-1]["step"]
            summary = self._context_refinement(
                logs_to_summarize,
                start_step,
                end_step,
                previous_summary=self.last_full_summary,
            )
        else:
            logs_to_summarize = self.action_logs
            start_step = logs_to_summarize[0]["step"]
            end_step = logs_to_summarize[-1]["step"]
            summary = self._context_refinement(logs_to_summarize, start_step, end_step)

        self.last_full_summary = summary
        self.last_summary_log_index = total_logs
        self.last_refinement_log_count = total_logs
        self.last_refinement_step = end_step
        self.logger.info(f"[refinement:{reason or 'periodic'}] {summary}")

        if self.wo_step:
            self.conversation_messages = []
            self.last_tool_output = None

    def _derive_execution_status(self, decision: Dict) -> Tuple[str, str]:
        proposed_subgoal = self._normalize_subgoal(decision.get("subgoal", ""))
        latest_log = self.action_logs[-1] if self.action_logs else {}
        verified = str((latest_log.get("compact") or {}).get("verified", "") or "").lower()
        if "termination(" in str(decision.get("code", "") or ""):
            return "finish", ""
        if latest_log.get("execution_success") is False:
            return "blocked", "progress_stall"
        if "no visible change" in verified or "timeout/no visible change" in verified:
            same_subgoal_logs = []
            for log in reversed(self.action_logs):
                if log.get("subgoal") and log.get("subgoal") != proposed_subgoal:
                    break
                same_subgoal_logs.append(log)
            no_change_count = 0
            for log in same_subgoal_logs:
                log_verified = str((log.get("compact") or {}).get("verified", "") or "").lower()
                if "no visible change" in log_verified or "timeout/no visible change" in log_verified:
                    no_change_count += 1
                else:
                    break
            if no_change_count >= 2:
                return "blocked", "progress_stall"
        if self.current_subgoal and proposed_subgoal != self.current_subgoal:
            return "done", ""
        return "continue", ""

    def _record_subgoal_transition(self, decision: Dict) -> str:
        proposed_subgoal = self._normalize_subgoal(decision.get("subgoal", ""))
        previous_subgoal = self.current_subgoal
        status = self._normalize_execution_status(decision.get("execution_status", "continue"))
        blocking_reason = self._normalize_blocking_reason(decision.get("blocking_reason", ""))

        if self.action_logs:
            self.action_logs[-1]["subgoal"] = proposed_subgoal
            self.action_logs[-1]["execution_status"] = status
            if blocking_reason:
                self.action_logs[-1]["blocking_reason"] = blocking_reason
            else:
                self.action_logs[-1].pop("blocking_reason", None)

        self.last_execution_status = status

        if status == "blocked":
            self.consecutive_stuck_subgoals += 1
        else:
            self.consecutive_stuck_subgoals = 0

        if not previous_subgoal:
            self.current_subgoal = proposed_subgoal
        elif proposed_subgoal != previous_subgoal:
            self.current_subgoal = proposed_subgoal

        return status

    def _build_termination_guard_feedback(self) -> str:
        if not self.awaiting_final_verification:
            self.awaiting_final_verification = True
            self.final_verification_observed = False
            target = self.current_subgoal or "the requested final state"
            return (
                f"Termination blocked. You must verify the exact final state for subgoal '{target}' before terminating.\n"
                "Do one more verification-focused action, then terminate only if the result clearly confirms task completion."
            )
        return (
            "Termination blocked. Final verification has not been observed yet.\n"
            "Use one more action to inspect the final UI or output and collect concrete evidence, then terminate only if verified."
        )

    def _build_blocked_feedback(self, event_type: str, detail: str = "") -> str:
        detail = re.sub(r"\s+", " ", str(detail or "").strip())
        if event_type == "loop_detected":
            return (
                "Blocked feedback: repeated the same action/result loop. Do not retry the same target.\n"
                "Switch target, switch tool, or mark the current subgoal as blocked."
            )
        if event_type == "no_visible_change":
            return (
                "Blocked feedback: the last action produced no visible change before timeout.\n"
                "Re-locate the target, try a different interaction, or switch tools."
                + (f"\nContext: {detail}" if detail else "")
            )
        if event_type == "tool_execution_failed":
            return (
                "Blocked feedback: the last tool execution failed.\n"
                "Fix the concrete failure instead of repeating the same action."
                + (f"\nError: {detail}" if detail else "")
            )
        if event_type == "termination_verification_failed":
            return (
                "Blocked feedback: final verification did not confirm task completion.\n"
                "Do one more targeted verification or finish the missing requirement."
                + (f"\nMissing: {detail}" if detail else "")
            )
        if event_type == "termination_self_contradiction":
            return (
                "Blocked feedback: your termination message itself says the task is not fully complete.\n"
                "Do not terminate until the missing requirement is actually satisfied."
                + (f"\nMissing: {detail}" if detail else "")
            )
        return detail or "Blocked feedback: reassess the current subgoal and choose a different strategy."

    def _termination_message_has_unmet_requirement(self, message: str) -> bool:
        text = re.sub(r"\s+", " ", str(message or "").strip()).lower()
        if not text:
            return False
        unmet_markers = [
            "not complete",
            "not fully complete",
            "not completed",
            "not achieved",
            "not applied",
            "not visible",
            "not found",
            "not installed",
            "not configured",
            "not set",
            "failed",
            "still needs",
            "needs additional",
            "requires additional",
            "does not match",
            "did not match",
            "differs from",
            "instead of",
            "however",
            "but ",
        ]
        positive_markers = [
            "task completed",
            "fully verified",
            "successfully completed",
            "requested final state is visible",
        ]
        return any(marker in text for marker in unmet_markers) and not any(marker in text for marker in positive_markers)

    def _run_final_verification(self) -> bool:
        screenshot = self._wait_until_screenshot_available(timeout_seconds=2.0)
        screenshot_b64 = base64.b64encode(screenshot).decode("utf-8") if screenshot else ""
        recent_logs = self.action_logs[-4:]
        history_lines = [self._render_compact_log(log) for log in recent_logs if log.get("compact") or log.get("detail")]
        messages = [
            {"role": "system", "content": FINAL_VERIFICATION_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Task:\n{self.task_instruction}\n\n"
                    f"Current subgoal:\n{self.current_subgoal or 'None'}\n\n"
                    f"Latest execution logs:\n" + ("\n".join(history_lines) if history_lines else "None")
                ),
            },
        ]
        if screenshot_b64:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Current screenshot:"},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{screenshot_b64}"},
                ],
            })

        self._dump_prompt_entry(
            stage="final_verification",
            payload={"messages": messages},
        )
        response = self.state_manager_llm(messages, enable_thinking=False)
        self._dump_prompt_entry(
            stage="final_verification_response",
            payload=response,
        )
        parsed = json.loads(repair_json(response))
        result = str(parsed.get("result", "") or "").strip().lower()
        if result not in {"pass", "fail"}:
            raise ValueError("Final verification response must contain result=pass|fail")
        return result == "pass"

    def _screenshots_meaningfully_different(self, before_screenshot: bytes, after_screenshot: bytes) -> bool:
        if not before_screenshot or not after_screenshot:
            return False
        if before_screenshot == after_screenshot:
            return False
        try:
            before_img = Image.open(io.BytesIO(before_screenshot))
            after_img = Image.open(io.BytesIO(after_screenshot))
            if before_img.size != after_img.size:
                return True
            cropped_before, cropped_after = get_change_roi(
                before_img,
                after_img,
                margin=self.roi_margin,
            )
            return cropped_before is not None and cropped_after is not None
        except Exception:
            return self._hash_bytes(before_screenshot) != self._hash_bytes(after_screenshot)

    def _wait_for_environment_change(
        self,
        before_screenshot: bytes,
        timeout_seconds: Optional[float] = None,
        initial_after_screenshot: Optional[bytes] = None,
    ) -> Tuple[bytes, bool, float]:
        timeout = self.explicit_wait_timeout if timeout_seconds is None else max(1.0, float(timeout_seconds))
        poll_interval = max(0.2, float(self.wait_poll_interval))
        start_time = time.time()
        latest_screenshot = initial_after_screenshot or before_screenshot
        changed_detected = False
        stable_since = None

        if initial_after_screenshot and self._screenshots_meaningfully_different(before_screenshot, initial_after_screenshot):
            changed_detected = True
            stable_since = start_time

        while True:
            now = time.time()
            elapsed = now - start_time
            if elapsed >= timeout:
                break

            if changed_detected and stable_since is not None and (now - stable_since) >= poll_interval:
                return latest_screenshot, True, elapsed

            time.sleep(min(poll_interval, max(0.0, timeout - elapsed)))
            current_screenshot = self.env.controller.get_screenshot()
            if current_screenshot is None:
                continue
            previous_screenshot = latest_screenshot
            latest_screenshot = current_screenshot

            if self._screenshots_meaningfully_different(before_screenshot, current_screenshot):
                changed_detected = True
                if self._screenshots_meaningfully_different(previous_screenshot, current_screenshot):
                    stable_since = time.time()
                elif stable_since is None:
                    stable_since = time.time()

        return latest_screenshot, changed_detected, time.time() - start_time

    def _wait_until_screenshot_available(
        self,
        timeout_seconds: Optional[float] = None,
    ) -> Optional[bytes]:
        timeout = self.screenshot_wait_timeout if timeout_seconds is None else max(0.5, float(timeout_seconds))
        start_time = time.time()
        while True:
            screenshot = self.env.controller.get_screenshot()
            if screenshot is not None:
                return screenshot
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                return None
            time.sleep(min(self.wait_poll_interval, max(0.0, timeout - elapsed)))

    def _evaluate_with_polling(self) -> float:
        timeout = self.evaluation_wait_timeout
        start_time = time.time()
        attempt = 0
        last_error = None
        while True:
            attempt += 1
            try:
                return self.env.evaluate()
            except Exception as eval_error:
                last_error = eval_error
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    raise last_error
                remaining = timeout - elapsed
                wait_time = min(self.wait_poll_interval * max(1, min(attempt, 4)), remaining)
                self.logger.warning(
                    f"Evaluation attempt {attempt} failed: {eval_error}. Retrying in {wait_time:.1f} seconds..."
                )
                time.sleep(wait_time)

    def _get_usage_snapshot(self) -> Dict:
        """Get current token usage snapshot from all LLMs."""
        global_planner_cost, global_planner_prompt, global_planner_completion, global_planner_images = self.global_planner_llm.get_usage()
        visual_grounder_cost, visual_grounder_prompt, visual_grounder_completion, visual_grounder_images = self.visual_grounder_llm.get_usage()
        state_manager_cost, state_manager_prompt, state_manager_completion, state_manager_images = self.state_manager_llm.get_usage()

        return {
            "global_planner": {
                "cost": global_planner_cost,
                "prompt_tokens": global_planner_prompt,
                "completion_tokens": global_planner_completion,
                "image_count": global_planner_images
            },
            "visual_grounder": {
                "cost": visual_grounder_cost,
                "prompt_tokens": visual_grounder_prompt,
                "completion_tokens": visual_grounder_completion,
                "image_count": visual_grounder_images
            },
            "state_manager": {
                "cost": state_manager_cost,
                "prompt_tokens": state_manager_prompt,
                "completion_tokens": state_manager_completion,
                "image_count": state_manager_images
            }
        }

    def _calculate_usage_delta(self, before: Dict, after: Dict) -> Dict:
        """Calculate the difference in token usage between two snapshots."""
        delta = {}
        for model in ["global_planner", "visual_grounder", "state_manager"]:
            delta[model] = {
                "cost": after[model]["cost"] - before[model]["cost"],
                "prompt_tokens": after[model]["prompt_tokens"] - before[model]["prompt_tokens"],
                "completion_tokens": after[model]["completion_tokens"] - before[model]["completion_tokens"],
                "image_count": after[model]["image_count"] - before[model]["image_count"]
            }
        return delta

    def _hash_text(self, text: str) -> str:
        """Create a stable fingerprint for text."""
        if text is None:
            text = ""
        return hashlib.sha256(str(text).encode("utf-8", errors="replace")).hexdigest()

    def _hash_bytes(self, content: bytes) -> str:
        """Create a stable fingerprint for bytes."""
        return hashlib.sha256(content or b"").hexdigest()

    def _get_decision_action_fingerprint(self, decision: Dict) -> str:
        """Compute action fingerprint from current decision before execution."""
        if decision.get("code"):
            return self._hash_text(str(decision.get("code", "")).strip())
        tool = decision.get("tool", "")
        tool_input = decision.get("input", "")

        if tool == "bash_execution":
            command = decision.get("command", tool_input)
            return self._hash_text(self._normalize_bash_command(command))
        if self._is_gui_tool(tool):
            gui_input = {}
            if isinstance(tool_input, dict):
                gui_input.update(tool_input)
            gui_input["tool"] = tool
            return self._hash_text(gui_input)
        return ""

    def _normalize_gui_description(self, description: str) -> str:
        """Normalize GUI target description for repeat detection."""
        if not description:
            return ""
        return re.sub(r"\s+", " ", str(description).strip()).lower()

    def _normalize_action_arg(self, value: Any) -> str:
        if isinstance(value, dict):
            items = []
            for key in sorted(value.keys()):
                items.append(f"{key}={self._normalize_action_arg(value[key])}")
            return "{" + ",".join(items) + "}"
        if isinstance(value, (list, tuple)):
            return "[" + ",".join(self._normalize_action_arg(item) for item in value) + "]"
        return re.sub(r"\s+", " ", str(value).strip()).lower()

    def _coerce_action_script_to_python(self, code: str) -> str:
        text = str(code or "").strip()
        if not text:
            return text

        helper_names = [name for name in ACTION_SCRIPT_FUNCTIONS if name != "import_subprocess"]
        helper_pattern = r"\b(?:%s)\s*\(" % "|".join(re.escape(name) for name in helper_names)
        if re.search(helper_pattern, text):
            return text
        if "import subprocess" in text or "subprocess.run" in text:
            return text

        first_line = next((line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")), "")
        if not first_line:
            return text

        if re.match(r"^(import|from|if|for|while|try|with|def|class|return|print|assert|del|pass|break|continue|raise)\b", first_line):
            return text
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*(\(|=|\.|\[|:)", first_line):
            return text

        return f"import subprocess\nsubprocess.run({text!r}, shell=True)"

    def _get_script_action_units(self, decision: Dict) -> List[Dict[str, str]]:
        code = self._coerce_action_script_to_python(str(decision.get("code", "") or "").strip())
        if not code:
            return []
        try:
            actions = self._parse_action_script(code)
        except Exception:
            return []

        units: List[Dict[str, str]] = []
        for action in actions:
            name = str(action.get("name", "") or "").strip()
            args = action.get("args", [])
            normalized_args = [self._normalize_action_arg(arg) for arg in args]
            fingerprint = f"{name}(" + "|".join(normalized_args) + ")"
            units.append({
                "name": name,
                "fingerprint": fingerprint,
            })
        return units

    def _get_recent_decision_logs(self) -> List[Dict]:
        decision_logs: List[Dict] = []
        seen_steps = set()
        for log in reversed(self.action_logs):
            step = log.get("step")
            if step in seen_steps:
                continue
            seen_steps.add(step)
            decision_logs.append(log)
        decision_logs.reverse()
        return decision_logs

    def _get_script_fingerprint_from_decision(self, decision: Dict) -> str:
        units = self._get_script_action_units(decision)
        if not units:
            return ""
        return " -> ".join(unit["fingerprint"] for unit in units)

    def _detect_repeated_gui_description(self, decision: Dict) -> Optional[str]:
        """Fail fast if the same GUI tool description is planned 3 consecutive times."""
        if decision.get("code"):
            units = self._get_script_action_units(decision)
            gui_units = [unit for unit in units if unit["name"] in GUI_ACTION_TOOLS]
            if not gui_units:
                return None

            normalized_description = " || ".join(unit["fingerprint"] for unit in gui_units)
            recent_logs = self._get_recent_decision_logs()
            repeat_count = 1
            for log in reversed(recent_logs):
                if (log.get("decision_gui_fingerprint") or "") != normalized_description:
                    break
                repeat_count += 1

            if repeat_count >= 3:
                return (
                    "Detected repeated GUI action-script loop: "
                    f"same GUI action sequence repeated {repeat_count} consecutive decisions."
                )
            return None
        if not self._is_gui_tool(decision.get("tool", "")):
            return None

        normalized_description = self._normalize_gui_description(decision.get("description", ""))
        if not normalized_description:
            return None

        repeat_count = 1  # Count current candidate decision.
        for log in reversed(self.action_logs):
            if log.get("type") not in GUI_ACTION_TOOLS:
                break
            if self._normalize_gui_description(log.get("description", "")) != normalized_description:
                break
            repeat_count += 1

        if repeat_count >= 3:
            return (
                "Detected repeated GUI-tool description loop: "
                f"'{decision.get('description', '')}' repeated {repeat_count} consecutive times."
            )
        return None

    def _detect_execution_loop(self, decision: Dict) -> Optional[str]:
        """Detect strict loops with identical action/result fingerprints."""
        if decision.get("code"):
            candidate_action_fp = self._get_script_fingerprint_from_decision(decision)
            if not candidate_action_fp or not self.action_logs:
                return None

            recent_logs = self._get_recent_decision_logs()
            if not recent_logs:
                return None

            last_log = recent_logs[-1]
            last_action_fp = last_log.get("decision_action_fingerprint", "")
            last_result_fp = last_log.get("decision_result_fingerprint", "")
            if not last_action_fp or not last_result_fp:
                return None
            if candidate_action_fp != last_action_fp:
                return None

            repeat_count = 0
            for log in reversed(recent_logs):
                if log.get("decision_action_fingerprint") != last_action_fp:
                    break
                if log.get("decision_result_fingerprint") != last_result_fp:
                    break
                repeat_count += 1

            if repeat_count >= 3:
                return (
                    "Detected repeated action-script loop: same action sequence and decision result "
                    f"repeated {repeat_count} consecutive decisions."
                )
            return None
        tool = decision.get("tool", "")
        if tool not in GUI_ACTION_TOOLS.union({"bash_execution"}) or not self.action_logs:
            return None

        threshold = 5 if tool in GUI_ACTION_TOOLS else 3
        candidate_action_fp = self._get_decision_action_fingerprint(decision)
        if not candidate_action_fp:
            return None

        last_log = self.action_logs[-1]
        if last_log.get("type") != tool:
            return None

        last_action_fp = last_log.get("loop_action_fingerprint", "")
        last_result_fp = last_log.get("loop_result_fingerprint", "")
        if not last_action_fp or not last_result_fp:
            return None
        if candidate_action_fp != last_action_fp:
            return None

        repeat_count = 0
        for log in reversed(self.action_logs):
            if log.get("type") != tool:
                break
            if log.get("loop_action_fingerprint") != last_action_fp:
                break
            if log.get("loop_result_fingerprint") != last_result_fp:
                break
            repeat_count += 1

        if repeat_count >= threshold:
            return (
                f"Detected strict execution loop: same {tool} action fingerprint and result fingerprint "
                f"repeated {repeat_count} consecutive times (threshold={threshold})."
            )
        return None

    def _context_refinement(self, logs: List[Dict], start_step: int, end_step: int, previous_summary: str = "") -> str:
        """Summarize a segment of action logs with context refinement."""
        
        if not logs and not previous_summary:
            return f"Steps {start_step}~{end_step}: No actions. Suggestion: Continue"

        # Build detailed history of new logs
        if self.wo_step:
            history_lines = []
            
            if logs:
                for i, log in enumerate(logs):
                    step_num = log.get("step", 0)
                    msg_idx = i * 2 
                    
                    if msg_idx + 1 < len(self.conversation_messages):
                        user_msg = self.conversation_messages[msg_idx]
                        assistant_msg = self.conversation_messages[msg_idx + 1]
                        
                        # Extract text content
                        user_text = ""
                        if isinstance(user_msg.get('content'), list):
                            for content_item in user_msg['content']:
                                if content_item.get('type') in ['text', 'input_text']:
                                    user_text = content_item.get('text', '')
                                    break
                        else:
                            user_text = user_msg.get('content', '')
                        
                        assistant_text = ""
                        if isinstance(assistant_msg.get('content'), list):
                            for content_item in assistant_msg['content']:
                                if content_item.get('type') in ['text', 'input_text']:
                                    assistant_text = content_item.get('text', '')
                                    break
                        else:
                            assistant_text = assistant_msg.get('content', '')
                        
                        history_lines.append(f"Step {step_num}:\n  User: {user_text}\n  Assistant: {assistant_text}")
        else:
            # Original step_abstract approach
            history_lines = []
            for log in logs:
                if log.get("compact"):
                    history_lines.append(self._render_compact_log(log))
                elif "step_abstract" in log:
                    history_lines.append(log["step_abstract"])

        if not previous_summary and not history_lines:
            return f"Steps {start_step}~{end_step}: No detailed records. Suggestion: Continue"

        try:
            execution_history_parts = []
            if previous_summary:
                execution_history_parts.append(
                    f"Previous summary covering earlier steps:\n{previous_summary}"
                )
            if history_lines:
                execution_history_parts.append(
                    "Newly added detailed steps:\n" + "\n".join(history_lines)
                )
            execution_history = "\n\n".join(execution_history_parts)

            messages = [
                {"role": "system", "content": CONTEXT_REFINEMENT_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Task instruction: {self.task_instruction}\n"
                        f"Execution history range: Steps {start_step}~{end_step}\n"
                        f"Execution history:\n{execution_history}"
                    )
                },
            ]
            self._dump_prompt_entry(
                stage="context_refinement",
                payload={"messages": messages},
                step=end_step,
            )
            summary_with_context_refinement = self.state_manager_llm(
                messages,
                enable_thinking=False,
            )
            self._dump_prompt_entry(
                stage="context_refinement_response",
                payload=summary_with_context_refinement,
                step=end_step,
            )
            return summary_with_context_refinement.strip()
        except Exception as e:
            self.logger.error(f"Failed to summarize history segment with context refinement: {e}")
            raise

    def execute_task(
        self,
        task_config: dict,
        additional_context: Optional[str] = None,
    ) -> float:
        """Execute task using tool-calling loop."""
        
        # Record start time for execution time tracking
        self.start_time = time.time()

        # Reset state
        self.global_planner_llm.reset_stats()
        self.visual_grounder_llm.reset_stats()
        self.state_manager_llm.reset_stats()
        self.env.reset(task_config=task_config)
        self.operation_count = 0
        self.action_logs = []
        self.last_full_summary = None
        self.last_summary_log_index = 0
        self.last_refinement_log_count = 0
        self.last_refinement_step = 0
        self.conversation_messages = []  # Store full conversation history when wo_step=True
        self.last_tool_output = None  # Store last tool execution result for wo_step mode
        self.current_subgoal = ""
        self.current_proposed_subgoal = ""
        self.consecutive_stuck_subgoals = 0
        self.awaiting_final_verification = False
        self.final_verification_observed = False
        self.last_execution_status = "continue"
        self.last_blocked_feedback_event = None
        self.last_blocked_feedback = None
        self.last_error_feedback = None

        if self.record:
            self.env.controller.start_recording()

        # Setup directories
        self.operations_dir = os.path.join(self.save_dir, "operations")
        os.makedirs(self.operations_dir, exist_ok=True)

        self.logger.info(f"Global Planner: {self.global_planner_model}")
        self.logger.info(f"Visual Grounder: {self.visual_grounder_model}")
        self.logger.info(f"State Manager: {self.state_manager_model}")
        self.logger.info(f"Max steps: {self.max_steps}")
        self.logger.info(f"wo_step: {self.wo_step}")
        
        # Initial message
        task_instruction = task_config["instruction"]
        if additional_context:
            task_instruction += f"\n\n{additional_context}"

        # Save task instruction as instance variable for later use
        self.task_instruction = task_instruction
        self.current_domain = task_config.get("domain", "general")
        self.current_task_id = str(
            task_config.get("id")
            or task_config.get("task_id")
            or os.path.basename(self.save_dir)
            or "unknown"
        )
        self.current_task_tags = self._extract_task_tags(task_instruction, self.current_domain)
        self.current_task_signature = self._build_task_signature(task_instruction, self.current_domain)
        self._init_prompt_dump_file()

        # Load relevant pattern
        domain = self.current_domain
        past_pattern_text = ""
        if not self.wo_pattern:
            past_pattern_text = self.pattern_manager.get_relevant_pattern(
                domain,
                task_instruction,
                task_signature=self.current_task_signature,
                task_tags=self.current_task_tags,
            )
            if past_pattern_text:
                self.logger.info(f"Found relevant past pattern for domain: {domain}\n{past_pattern_text}")
            else:
                self.logger.info(f"No relevant past pattern found for domain: {domain}")

        # Save past pattern as instance variable for later use
        self.past_pattern_text = past_pattern_text

        # Main execution loop
        is_infeasible = False
        infeasible_reason = ""
        try:
            while self.operation_count < self.max_steps:
                self.logger.info(f"Step {self.operation_count + 1}/{self.max_steps}")

                # Capture token usage before this step
                usage_before_step = self._get_usage_snapshot()

                # Get global planner decision
                decision = self._get_global_planner_decision()

                if decision is None:
                    self.logger.error("Failed to get valid decision")
                    # Send "FAIL" action to environment for task failure
                    try:
                        self.env.step("FAIL", 0)
                    except Exception as e:
                        self.logger.warning(f"Failed to send FAIL action: {e}")
                    break

                # Capture token usage after global planner decision
                usage_after_global_planner = self._get_usage_snapshot()
                global_planner_usage = self._calculate_usage_delta(usage_before_step, usage_after_global_planner)

                repeated_description_error = self._detect_repeated_gui_description(decision)
                if repeated_description_error:
                    self.logger.warning(repeated_description_error)
                    is_infeasible = True
                    infeasible_reason = repeated_description_error
                    try:
                        self.env.step("FAIL", 0)
                    except Exception as e:
                        self.logger.warning(f"Failed to send FAIL action: {e}")
                    break

                loop_error = self._detect_execution_loop(decision)
                if loop_error:
                    self.logger.warning(loop_error)
                    step = self.operation_count + 1
                    self.action_logs.append({
                        "step": step,
                        "type": "loop_block",
                        "execution_success": False,
                        "screenshot": "",
                        "subgoal": decision.get("subgoal", self.current_subgoal or ""),
                        "execution_status": "blocked",
                        "blocking_reason": "behavior_loop",
                        "detail": loop_error,
                        "compact": self._build_compact_log_entry(
                            step=step,
                            tool_type="loop_block",
                            success=False,
                            detail=loop_error,
                            verification="Repeated the same GUI action pattern without meaningful progress.",
                            next_hint="Switch target, tool, or overall approach instead of repeating the same action script."
                        ),
                        "step_time": 0.0,
                        "token_usage": self.step_token_usage,
                    })
                    self.last_blocked_feedback = self._build_blocked_feedback("loop_detected", loop_error)
                    self.last_error_feedback = self.last_blocked_feedback
                    self.last_execution_status = "blocked"
                    self.logger.info(
                        "[execution_status] step=%s status=blocked blocking_reason=behavior_loop subgoal=%s",
                        step,
                        decision.get("subgoal", self.current_subgoal or ""),
                    )
                    self._maybe_refine_context("blocked")
                    if self.wo_step:
                        self.last_tool_output = f"Execution blocked: {loop_error}"
                    # Count this as a consumed step to avoid infinite planner-loop cycles.
                    self.operation_count += 1
                    continue

                # Execute tool and capture execution result text
                self.last_blocked_feedback_event = None
                self.current_step_id = self.operation_count + 1
                execution_result_text, terminal_status, terminal_message = self._execute_tool(decision, global_planner_usage)
                self.operation_count += 1
                
                # Store execution result for wo_step mode to maintain dialogue structure
                if self.wo_step and execution_result_text:
                    self.last_tool_output = execution_result_text

                if terminal_status == "termination":
                    if self._termination_message_has_unmet_requirement(terminal_message):
                        self.awaiting_final_verification = True
                        self.final_verification_observed = False
                        self.last_blocked_feedback = self._build_blocked_feedback(
                            "termination_self_contradiction",
                            terminal_message,
                        )
                        self.last_error_feedback = self.last_blocked_feedback
                        if self.wo_step:
                            suffix = "\n\n" if execution_result_text else ""
                            self.last_tool_output = (execution_result_text or "") + suffix + "Termination blocked: the completion message still describes an unmet requirement."
                        self.logger.info("Termination blocked: completion message indicates unmet requirements")
                        continue

                    verification = self._run_final_verification()
                    if not verification:
                        self.awaiting_final_verification = True
                        self.final_verification_observed = False
                        self.last_blocked_feedback = self._build_blocked_feedback("termination_verification_failed", "")
                        self.last_error_feedback = self.last_blocked_feedback
                        if self.wo_step:
                            suffix = "\n\n" if execution_result_text else ""
                            self.last_tool_output = (execution_result_text or "") + suffix + "Final verification failed."
                        self.logger.info("Termination blocked: final verification failed")
                        continue
                    self.logger.info("Final verification passed")

                    step = self.current_step_id or (self.operation_count + 1)
                    screenshot_file = f"step_{step}.png"
                    try:
                        screenshot = self.env.controller.get_screenshot()
                        with open(os.path.join(self.operations_dir, screenshot_file), "wb") as f:
                            f.write(screenshot)
                    except Exception as e:
                        self.logger.warning(f"Failed to capture termination screenshot: {e}")
                        screenshot_file = ""

                    self.action_logs.append({
                        "step": step,
                        "type": "termination",
                        "execution_success": True,
                        "screenshot": screenshot_file,
                        "subgoal": self.current_subgoal or "Verify task completion",
                        "execution_status": "finish",
                        "detail": terminal_message or "Task completed.",
                        "compact": self._build_compact_log_entry(
                            step=step,
                            tool_type="termination",
                            success=True,
                            detail=terminal_message or "Task completed.",
                            verification="Task marked complete after explicit verification."
                        ),
                        "step_time": 0.0,
                        "token_usage": {
                            "global_planner": self._zero_usage_entry(),
                            "visual_grounder": self._zero_usage_entry(),
                            "state_manager": self._zero_usage_entry(),
                            "total": self._zero_usage_entry(),
                        }
                    })
                    is_infeasible = False
                    self.logger.info("Task COMPLETED")
                    break

                if terminal_status == "infeasible":
                    is_infeasible = True
                    infeasible_reason = terminal_message or "Task is objectively impossible to complete"
                    self.logger.info(f"Task INFEASIBLE: {infeasible_reason}")
                    try:
                        self.env.step("FAIL", 0)
                    except Exception as e:
                        self.logger.warning(f"Failed to send FAIL action: {e}")
                    break

                # Continue with next iteration
                # (screenshot will be fetched in next _get_global_planner_decision call)

            # Check if reached max_steps without completion
            if self.operation_count >= self.max_steps and not is_infeasible:
                is_infeasible = True
                infeasible_reason = f"Reached maximum steps ({self.max_steps}) without completing the task. Task may be infeasible or requires a different approach."
                self.logger.info(f"Reached max_steps ({self.max_steps}), marking as INFEASIBLE")
                # Send "FAIL" action to environment so action_history ends with "FAIL"
                # This is required for OSWorld's infeasible task evaluation
                try:
                    self.env.step("FAIL", 0)
                except Exception as e:
                    self.logger.warning(f"Failed to send FAIL action: {e}")

            # Evaluation
            score = self._evaluate_and_save(task_config, additional_context or "", is_infeasible, infeasible_reason)

        except Exception as e:
            self.logger.error(f"Execution error: {e}")
            self.logger.error(traceback.format_exc())
            # Send "FAIL" action to environment for unexpected task failure
            try:
                self.env.step("FAIL", 0)
            except Exception as fail_error:
                self.logger.warning(f"Failed to send FAIL action: {fail_error}")
            score = self._save_error_log(task_config, additional_context or "", e)
        
        if self.record:
            self.env.controller.end_recording(os.path.join(self.save_dir, "recording.mp4"))
        
        return score

    def _get_global_planner_decision(self) -> Optional[Dict]:
        """Get decision from global planner with retry on parsing errors."""

        for attempt in range(self.max_parse_retries):
            response = ""
            json_str = ""
            try:
                # Get current screenshot
                screenshot = self._wait_until_screenshot_available()
                if screenshot is None:
                    raise RuntimeError("Failed to capture screenshot for planning after retries.")
                screenshot_b64 = base64.b64encode(screenshot).decode("utf-8")

                planner_system_prompt = self._build_planner_system_prompt()
                active_blocked_feedback = self._get_active_blocked_feedback()

                # ========== Build Messages ==========
                if self.wo_step:
                    messages = [
                        {"role": "system", "content": planner_system_prompt},
                    ]

                    conversation_to_append = self.conversation_messages
                    max_messages = self.sliding_window_size * 2
                    if self.wo_refinement and len(self.conversation_messages) > max_messages:
                        conversation_to_append = copy.deepcopy(self.conversation_messages[-max_messages:])
                        first_content = conversation_to_append[0].get("content")
                        if (
                            isinstance(first_content, list)
                            and first_content
                            and isinstance(first_content[0], dict)
                            and "text" in first_content[0]
                        ):
                            first_content[0]["text"] = f'Task: {self.task_instruction}\n\n{first_content[0]["text"]}'

                    messages.extend(conversation_to_append)

                    observation_text = self.last_tool_output or ""
                    if self.last_tool_output:
                        self.last_tool_output = None

                    prompt_text = (
                        "Based on the execution history and current screenshot, what is the next action?"
                        if len(conversation_to_append) == 0
                        else "Based on the conversation history and current screenshot, what is the next action?"
                    )
                    if active_blocked_feedback:
                        prompt_text += "\nPlease use the blocked feedback above to correct the next step."

                    context_message = self._build_planner_context_message(
                        include_task_header=(len(conversation_to_append) == 0),
                        prompt_text=prompt_text,
                        observation_text=observation_text,
                        blocked_feedback_text=active_blocked_feedback,
                    )
                    if context_message:
                        messages.append(context_message)

                    current_user_message = {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Current screenshot:"},
                            {"type": "input_image", "image_url": f"data:image/png;base64,{screenshot_b64}"}
                        ]
                    }
                    messages.append(current_user_message)
                else:
                    logs_to_use = self.action_logs
                    if self.wo_refinement and len(self.action_logs) > self.sliding_window_size:
                        logs_to_use = self.action_logs[-self.sliding_window_size:]

                    condensed_history = []
                    if not self.wo_refinement and self.last_full_summary:
                        for log in self.action_logs[self.last_summary_log_index:]:
                            if log.get("compact"):
                                condensed_history.append(self._render_compact_log(log))
                            elif log.get("detail"):
                                condensed_history.append(self._render_compact_log(log))
                    else:
                        for log in logs_to_use:
                            if log.get("compact"):
                                condensed_history.append(self._render_compact_log(log))
                            elif log.get("detail"):
                                condensed_history.append(self._render_compact_log(log))

                    messages = [{"role": "system", "content": planner_system_prompt}]
                    prompt_text = "Based on the execution history and current screenshot, decide the next action. Prefer the shortest reliable path and avoid repeating failed actions."
                    if active_blocked_feedback:
                        prompt_text += "\nPlease use the blocked feedback above to correct the next step."
                    context_message = self._build_planner_context_message(
                        include_task_header=True,
                        history_items=condensed_history,
                        prompt_text=prompt_text,
                        blocked_feedback_text=active_blocked_feedback,
                    )
                    if context_message:
                        messages.append(context_message)

                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Current screenshot:"},
                            {"type": "input_image", "image_url": f"data:image/png;base64,{screenshot_b64}"}
                        ]
                    })

                if attempt > 0:
                    self.logger.warning(f"Retry attempt {attempt}/{self.max_parse_retries}")

                # Call global planner
                self._dump_prompt_entry(
                    stage="global_planner",
                    payload={"messages": messages},
                    attempt=attempt + 1,
                )
                response = self.global_planner_llm(
                    messages,
                    enable_thinking=True,
                )
                self._dump_prompt_entry(
                    stage="global_planner_response",
                    payload=response,
                    attempt=attempt + 1,
                )
                
                # Extract JSON
                json_str = response
                if "```json" in response:
                    json_start = response.find("```json") + 7
                    json_end = response.find("```", json_start)
                    json_str = response[json_start:json_end].strip()
                elif "```" in response:
                    json_start = response.find("```") + 3
                    json_end = response.find("```", json_start)
                    json_str = response[json_start:json_end].strip()

                # Parse JSON
                decision = json.loads(repair_json(json_str))

                if "subgoal" not in decision:
                    raise ValueError("Missing 'subgoal' field in decision")

                if "code" not in decision:
                    raise ValueError("Missing 'code' field in decision")
                if not isinstance(decision.get("code"), str) or not decision["code"].strip():
                    raise ValueError("Decision 'code' must be a non-empty string")
                decision["code"] = self._coerce_action_script_to_python(decision["code"])
                decision["subgoal"] = self._resolve_planner_subgoal(decision["subgoal"], decision["code"])
                self._parse_action_script(decision["code"])

                try:
                    self.logger.info(f"[decision]: {json.dumps(decision, indent=4)}")
                except Exception as e:
                    self.logger.info(f"[decision]: {decision}")

                # Clear error feedback on success
                self.last_blocked_feedback = None
                self.last_error_feedback = None
                
                # Store conversation for wo_step mode ONLY after successful parsing
                if self.wo_step:
                    self.conversation_messages.append(current_user_message)
                    self.conversation_messages.append({
                        "role": "assistant",
                        "content": response
                    })

                return decision
                
            except Exception as e:
                self.logger.error(f"Decision parsing error (attempt {attempt + 1}/{self.max_parse_retries}): {e}")
                self.logger.error(
                    "Raw model response (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_parse_retries,
                    response or "<empty response>",
                )
                if json_str and json_str != response:
                    self.logger.error(
                        "Extracted JSON candidate (attempt %d/%d): %s",
                        attempt + 1,
                        self.max_parse_retries,
                        json_str,
                    )
                
                # If not last attempt, set error feedback for retry
                if attempt < self.max_parse_retries - 1:
                    extra_guidance = self._build_action_script_parse_guidance(
                        response=response,
                        error_message=str(e),
                    )
                    error_feedback = FIX_RESPONSE_PROMPT.format(
                        operation_count=self.operation_count,
                        max_steps=self.max_steps,
                        error_message=str(e),
                        extra_guidance=extra_guidance,
                        response=response
                    )
                    self._dump_prompt_entry(
                        stage="fix_response",
                        payload=error_feedback,
                        attempt=attempt + 1,
                    )

                    # Store error feedback for next iteration
                    self.last_error_feedback = error_feedback

                    # Continue to next retry
                    continue
                else:
                    self.logger.error("All retry attempts exhausted, cannot get valid decision")
                    with open(os.path.join(self.save_dir, "err_reason.txt"), "w") as f:
                        f.write("All retry attempts exhausted, cannot get valid decision")
                    raise ValueError("All retry attempts exhausted, cannot get valid decision")

    def _execute_tool(self, decision: Dict, planner_usage: Dict) -> Tuple[str, Optional[str], str]:
        """Execute planner-produced action script and return output, terminal status, and terminal message."""
        self.current_thought = decision.get("thought", "")
        self.current_proposed_subgoal = self._normalize_subgoal(decision["subgoal"])
        decision["code"] = self._coerce_action_script_to_python(decision["code"])
        actions = self._parse_action_script(decision["code"])
        self.current_decision_action_fingerprint = self._get_script_fingerprint_from_decision(decision)
        gui_units = [unit["fingerprint"] for unit in self._get_script_action_units(decision) if unit["name"] in GUI_ACTION_TOOLS]
        self.current_decision_gui_fingerprint = " || ".join(gui_units)
        outputs: List[str] = []
        terminal_status = None
        terminal_message = ""

        action_index = 0

        def _finalize_runtime_action(execution_result_text: str, include_planner: bool, usage_before_tool: Dict) -> str:
            nonlocal action_index
            outputs.append(execution_result_text)
            usage_after_tool = self._get_usage_snapshot()
            tool_usage = self._calculate_usage_delta(usage_before_tool, usage_after_tool)
            step_token_usage = self._build_step_token_usage(planner_usage, tool_usage, include_planner=include_planner)
            self._patch_latest_step_token_usage(self.current_step_id or (self.operation_count + 1), step_token_usage)
            self.step_token_usage = step_token_usage

            derived_status, blocking_reason = self._derive_execution_status(decision)
            decision["execution_status"] = derived_status
            if blocking_reason:
                decision["blocking_reason"] = blocking_reason
            else:
                decision.pop("blocking_reason", None)
            if blocking_reason:
                self.logger.info(
                    "[execution_status] step=%s status=%s blocking_reason=%s subgoal=%s",
                    self.current_step_id or (self.operation_count + 1),
                    derived_status,
                    blocking_reason,
                    decision.get("subgoal", ""),
                )
            else:
                self.logger.info(
                    "[execution_status] step=%s status=%s subgoal=%s",
                    self.current_step_id or (self.operation_count + 1),
                    derived_status,
                    decision.get("subgoal", ""),
                )
            status = self._record_subgoal_transition(decision)
            if self.awaiting_final_verification:
                self.final_verification_observed = True
            if status == "done":
                self._maybe_refine_context("done")
            elif status == "blocked":
                self._maybe_refine_context("blocked")
            if self.last_blocked_feedback_event:
                event_type = self.last_blocked_feedback_event.get("type", "")
                event_detail = self.last_blocked_feedback_event.get("detail", "")
                self.last_blocked_feedback = self._build_blocked_feedback(event_type, event_detail)
                self.last_error_feedback = self.last_blocked_feedback
            action_index += 1
            return execution_result_text

        def _run_runtime_action(fn: Callable[[], str]) -> str:
            include_planner = action_index == 0
            usage_before_tool = self._get_usage_snapshot()
            self.step_token_usage = self._build_step_token_usage(
                planner_usage=planner_usage,
                tool_usage={"visual_grounder": self._zero_usage_entry(), "state_manager": self._zero_usage_entry()},
                include_planner=include_planner,
            )
            return _finalize_runtime_action(fn(), include_planner, usage_before_tool)

        class _RuntimeSubprocessModule:
            def __init__(self, outer):
                self._outer = outer

            def run(self, *args, **kwargs):
                result_holder: Dict[str, Any] = {}

                def _call():
                    execution_text, completed_process = self._outer._subprocess_run_execution(
                        list(args),
                        dict(kwargs),
                        return_completed_process=True,
                    )
                    result_holder["value"] = completed_process
                    return execution_text

                _run_runtime_action(_call)
                return result_holder.get("value")

        def _runtime_click(arg: str):
            return _run_runtime_action(lambda: self._execute_gui_tool("click", arg, arg))

        def _runtime_double_click(arg: str):
            return _run_runtime_action(lambda: self._execute_gui_tool("double_click", arg, arg))

        def _runtime_right_click(arg: str):
            return _run_runtime_action(lambda: self._execute_gui_tool("right_click", arg, arg))

        def _runtime_move(arg: str):
            return _run_runtime_action(lambda: self._execute_gui_tool("move", arg, arg))

        def _runtime_drag(*args):
            if len(args) == 3:
                drag_instruction, drag_source, drag_destination = args
                drag_input = f"{drag_source} -> {drag_destination}"
                drag_description = drag_instruction
            elif len(args) == 2:
                drag_source, drag_destination = args
                drag_input = f"{drag_source} -> {drag_destination}"
                drag_description = f"Drag from {drag_source} to {drag_destination}"
            else:
                raise ValueError("drag() expects two or three string arguments")
            return _run_runtime_action(lambda: self._execute_gui_tool("drag", drag_input, drag_description))

        def _runtime_write(arg: str):
            return _run_runtime_action(lambda: self._execute_gui_tool("write", arg, ""))

        def _runtime_hotkey(*args):
            if not args or not all(isinstance(arg, str) for arg in args):
                raise ValueError("hotkey() expects one or more string arguments")
            hotkey_input = args[0] if len(args) == 1 else "+".join(args)
            return _run_runtime_action(lambda: self._execute_gui_tool("hotkey", hotkey_input, ""))

        def _runtime_scroll(description: str, amount: Optional[int] = None):
            resolved_amount = _validate_scroll_amount(amount) if amount is not None else _infer_scroll_amount(description)
            return _run_runtime_action(
                lambda: self._execute_gui_tool(
                    "scroll",
                    {"instruction": description, "amount": resolved_amount},
                    description,
                )
            )

        def _runtime_termination(message: str):
            raise _TerminalActionSignal("termination", message)

        def _runtime_infeasible(message: str):
            raise _TerminalActionSignal("infeasible", message)

        def _runtime_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "subprocess" and level == 0:
                return _RuntimeSubprocessModule(self)
            return builtins.__import__(name, globals, locals, fromlist, level)

        runtime_builtins = dict(builtins.__dict__)
        runtime_builtins["__import__"] = _runtime_import

        safe_globals = {
            "__builtins__": runtime_builtins,
            "click": _runtime_click,
            "double_click": _runtime_double_click,
            "right_click": _runtime_right_click,
            "move": _runtime_move,
            "drag": _runtime_drag,
            "write": _runtime_write,
            "hotkey": _runtime_hotkey,
            "scroll": _runtime_scroll,
            "termination": _runtime_termination,
            "infeasible": _runtime_infeasible,
            "subprocess": _RuntimeSubprocessModule(self),
        }
        runtime_locals: Dict[str, Any] = {}

        captured_stdout = io.StringIO()
        captured_stderr = io.StringIO()

        try:
            with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
                exec(compile(decision["code"], "<action_script>", "exec"), safe_globals, runtime_locals)
        except _TerminalActionSignal as signal:
            terminal_status = signal.status
            terminal_message = signal.message
        except Exception as exec_error:
            self.logger.error(f"Python execution error: {exec_error}")
            step = self.current_step_id or (self.operation_count + 1)
            screenshot = self.env.controller.get_screenshot()
            screenshot_file = f"step_{step}.png"
            with open(os.path.join(self.operations_dir, screenshot_file), "wb") as f:
                f.write(screenshot)

            error_text = f"{type(exec_error).__name__}: {exec_error}"
            self.action_logs.append({
                "step": step,
                "type": "python_execution",
                "execution_success": False,
                "screenshot": screenshot_file,
                "detail": decision.get("code", ""),
                "execution_status": "blocked",
                "blocking_reason": "progress_stall",
                "compact": self._build_compact_log_entry(
                    step=step,
                    tool_type="python_execution",
                    success=False,
                    detail=decision.get("code", ""),
                    verification=f"Error: {error_text}",
                    next_hint="Fix the Python code or helper call instead of repeating the same failing step."
                ),
                "token_usage": self.step_token_usage,
                "loop_action_fingerprint": self.current_decision_action_fingerprint,
                "loop_result_fingerprint": self._hash_text(f"python_error={error_text}"),
                "decision_action_fingerprint": self.current_decision_action_fingerprint,
                "decision_gui_fingerprint": self.current_decision_gui_fingerprint,
                "decision_result_fingerprint": self._hash_text(
                    f"status=blocked|python_error={error_text}|code={decision.get('code', '')}"
                ),
            })
            self.last_blocked_feedback_event = {
                "type": "tool_execution_failed",
                "detail": error_text,
            }
            outputs.append(f"[python error]\n{error_text}")

        stdout_text = captured_stdout.getvalue().strip()
        stderr_text = captured_stderr.getvalue().strip()
        if stdout_text:
            outputs.append(f"[python stdout]\n{stdout_text}")
        if stderr_text:
            outputs.append(f"[python stderr]\n{stderr_text}")

        return "\n\n".join(part for part in outputs if part), terminal_status, terminal_message

    def _normalize_bash_command(self, code: str) -> str:
        """Normalize bash command to enforce non-interactive sudo usage."""
        if not isinstance(code, str) or not code.strip():
            raise ValueError("Bash command must be a non-empty string")

        # Collapse common forms to plain "sudo ...":
        # "echo '' | sudo -S cmd", "echo 'password' | sudo -S cmd", "sudo -S cmd"
        normalized = re.sub(
            r"(?:echo\s+(?:'[^']*'|\"[^\"]*\"|\S+)\s*\|\s*)?sudo\s+-S\s+",
            "sudo ",
            code,
        )

        # Enforce sudo prefix with configured client password.
        quoted_password = "'" + self.client_password.replace("'", "'\"'\"'") + "'"
        sudo_prefix = f"echo {quoted_password} | sudo -S"
        normalized = re.sub(r"\bsudo\b", sudo_prefix, normalized)

        return normalized.strip()

    def _normalize_pyautogui_code(self, code: str) -> str:
        """Normalize planner-produced pyautogui code before parsing/execution."""
        if not isinstance(code, str) or not code.strip():
            return code

        # Some planner outputs emit named coordinates like click(x=123, y=456).
        # Downstream executors expect plain positional coordinates, so strip only
        # the redundant x=/y= markers and preserve all other kwargs.
        return re.sub(r"(?<=\(|,)\s*([xy])\s*=\s*", "", code)

    def _build_action_script_parse_guidance(self, response: str, error_message: str) -> str:
        error_text = str(error_message)
        if (
            "Failed to parse action script" not in error_text
        ):
            return ""

        code_snippet = ""
        try:
            json_str = str(response or "").strip()
            if "```json" in json_str:
                json_start = json_str.find("```json") + 7
                json_end = json_str.find("```", json_start)
                json_str = json_str[json_start:json_end].strip()
            elif "```" in json_str:
                json_start = json_str.find("```") + 3
                json_end = json_str.find("```", json_start)
                json_str = json_str[json_start:json_end].strip()
            parsed = json.loads(repair_json(json_str))
            if isinstance(parsed, dict):
                code_snippet = str(parsed.get("code", "") or "").strip()
        except Exception:
            code_snippet = ""

        guidance_lines = [
            "Action-script fix requirements:",
            "- `code` must be valid Python syntax.",
            "- Put the entire natural-language target inside a single quoted string argument.",
            "- Do not leave text outside quotes.",
            "- Balance every `(` with `)`.",
            "- Use valid Python code. Helper calls such as `click(...)`, `drag(...)`, `write(...)`, `hotkey(...)`, and `scroll(...)` are available.",
            "- For shell execution, write normal Python such as `import subprocess` and `subprocess.run(...)`.",
            "- Put the full `write(...)` text or spreadsheet formula inside one valid Python string literal.",
            "- For formulas with quotes inside them, keep the outer Python quotes different. Example: `write('=TEXT(C2;\"0000000\")')`.",
        ]
        if "unterminated string literal" in error_text or "'(' was never closed" in error_text:
            guidance_lines.append(
                "- If `write(...)` contains a spreadsheet formula, keep the whole formula inside one quoted Python string, for example `write('=TEXT(C2;\"0000000\")')`."
            )
        if code_snippet:
            guidance_lines.append(f"- Current invalid `code`: {code_snippet}")
        return "\n".join(guidance_lines)

    def _repair_action_script_line(self, line: str) -> str:
        stripped = str(line or "").strip()
        if not stripped:
            return ""

        if stripped in {"wait", "wait()"}:
            return ""

        if stripped in {"enter", "backspace", "tab", "esc", "escape"}:
            return f"hotkey({stripped!r})"

        call_match = re.match(r"^([A-Za-z_]\w*)\((.*)\)\s*$", stripped)
        if not call_match:
            return stripped

        name, raw_args = call_match.groups()
        raw_args = raw_args.strip()

        def split_simple_args(text: str) -> List[str]:
            parts: List[str] = []
            current: List[str] = []
            quote_char = ""
            depth = 0
            for ch in str(text or ""):
                if quote_char:
                    current.append(ch)
                    if ch == quote_char:
                        quote_char = ""
                    continue
                if ch in {"'", '"'}:
                    quote_char = ch
                    current.append(ch)
                    continue
                if ch in "([{":
                    depth += 1
                    current.append(ch)
                    continue
                if ch in ")]}":
                    depth = max(0, depth - 1)
                    current.append(ch)
                    continue
                if ch == "," and depth == 0:
                    part = "".join(current).strip()
                    if part:
                        parts.append(part)
                    current = []
                    continue
                current.append(ch)
            tail = "".join(current).strip()
            if tail:
                parts.append(tail)
            return parts

        def as_single_string_arg(text: str) -> str:
            cleaned = re.sub(r"\s+", " ", str(text or "").strip())
            if cleaned:
                try:
                    parsed = ast.literal_eval(cleaned)
                    if isinstance(parsed, str):
                        return repr(parsed)
                except Exception:
                    pass
            cleaned = cleaned.strip().strip(",")
            if cleaned.startswith(("'", '"')):
                cleaned = cleaned[1:]
            if cleaned.endswith(("'", '"')):
                cleaned = cleaned[:-1]
            cleaned = cleaned.replace('"', "").replace("'", "").strip()
            return repr(cleaned)

        if name in {"click", "double_click", "right_click", "move", "write", "termination", "infeasible"}:
            return f"{name}({as_single_string_arg(raw_args)})"

        if name == "hotkey":
            parts = split_simple_args(raw_args)
            if not parts:
                return stripped
            if len(parts) == 1:
                single = parts[0].strip()
                if not single.startswith(("'", '"')) and "+" in single:
                    parts = [part.strip() for part in single.split("+") if part.strip()]
            repaired_parts = [as_single_string_arg(part) for part in parts if str(part).strip()]
            return f"hotkey({', '.join(repaired_parts)})" if repaired_parts else stripped

        if name == "scroll":
            arg_parts = split_simple_args(raw_args)
            if len(arg_parts) <= 1:
                description = arg_parts[0] if arg_parts else raw_args
                lowered = description.lower()
                amount = -5 if "down" in lowered else 5 if "up" in lowered else -5
                return f"scroll({as_single_string_arg(description)}, {amount})"
            amount_text = str(arg_parts[1]).strip()
            try:
                amount_value = int(ast.literal_eval(amount_text))
            except Exception:
                try:
                    amount_value = int(amount_text)
                except Exception:
                    lowered = str(arg_parts[0]).lower()
                    amount_value = -5 if "down" in lowered else 5 if "up" in lowered else -5
            return f"scroll({as_single_string_arg(arg_parts[0])}, {amount_value})"

        if name == "drag":
            arg_parts = split_simple_args(raw_args)
            if len(arg_parts) >= 3:
                repaired_parts = [as_single_string_arg(part) for part in arg_parts[:3]]
                return f"drag({', '.join(repaired_parts)})"
            if len(arg_parts) == 2:
                repaired_parts = [as_single_string_arg(part) for part in arg_parts]
                return f"drag({', '.join(repaired_parts)})"
            instruction = arg_parts[0] if arg_parts else raw_args
            instruction_text = str(instruction).strip().strip("'\"")
            match = re.search(r"\bfrom\s+(.+?)\s+\bto\s+(.+)$", instruction_text, re.IGNORECASE)
            if match:
                source = match.group(1).strip(" .")
                destination = match.group(2).strip(" .")
                return f"drag({as_single_string_arg(instruction_text)}, {as_single_string_arg(source)}, {as_single_string_arg(destination)})"
            return f"drag({as_single_string_arg(instruction_text)}, {as_single_string_arg('drag source')}, {as_single_string_arg('drag destination')})"

        return stripped

    def _repair_action_script(self, code: str) -> str:
        if not isinstance(code, str):
            return code
        repaired_lines = []
        for raw_line in code.splitlines():
            repaired_line = self._repair_action_script_line(raw_line)
            if repaired_line:
                repaired_lines.append(repaired_line)
        repaired = "\n".join(repaired_lines).strip()
        return repaired or code

    def _parse_call_args(self, call: ast.Call) -> Tuple[List[Any], Dict[str, Any]]:
        try:
            args = [ast.literal_eval(arg) for arg in call.args]
            kwargs = {
                str(kw.arg): ast.literal_eval(kw.value)
                for kw in call.keywords
                if kw.arg is not None
            }
        except Exception as e:
            raise ValueError(f"Failed to parse call arguments: {e}") from e
        return args, kwargs

    def _validate_action_signature(self, name: str, args: List[Any], kwargs: Dict[str, Any]) -> None:
        if name == "import_subprocess":
            return
        if kwargs and name in ACTION_SCRIPT_FUNCTIONS:
            raise ValueError(f"Keyword arguments are not supported for {name}()")
        if name in {"click", "double_click", "right_click", "move", "write", "termination", "infeasible"}:
            if len(args) != 1 or not isinstance(args[0], str):
                raise ValueError(f"{name}() expects one string argument")
            return
        if name == "drag":
            if len(args) not in {2, 3} or not all(isinstance(arg, str) for arg in args):
                raise ValueError("drag() expects two or three string arguments")
            return
        if name == "hotkey":
            if not args or not all(isinstance(arg, str) for arg in args):
                raise ValueError("hotkey() expects one or more string arguments")
            return
        if name == "scroll":
            if len(args) != 2 or not isinstance(args[0], str):
                raise ValueError("scroll() expects description string and integer amount")
            _validate_scroll_amount(args[1])
            return
        if name == "subprocess.run":
            if not args:
                raise ValueError("subprocess.run() expects at least one positional argument")
            command = args[0]
            if not isinstance(command, (str, list, tuple)):
                raise ValueError("subprocess.run() command must be a string or list")
            if isinstance(command, (list, tuple)) and not all(isinstance(part, str) for part in command):
                raise ValueError("subprocess.run() list command must contain only strings")
            unsupported_kwargs = set(kwargs) - {"shell", "capture_output", "text", "check", "timeout", "cwd"}
            if unsupported_kwargs:
                raise ValueError(f"Unsupported subprocess call: unsupported kwargs {sorted(unsupported_kwargs)}")
            return

    def _parse_action_script(self, code: str) -> List[Dict[str, Any]]:
        if not isinstance(code, str) or not code.strip():
            raise ValueError("Action script must be a non-empty string")

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise ValueError(f"Failed to parse action script: {e}") from e

        actions: List[Dict[str, Any]] = []
        call_nodes = sorted(
            (node for node in ast.walk(tree) if isinstance(node, ast.Call)),
            key=lambda node: (getattr(node, "lineno", 0), getattr(node, "col_offset", 0)),
        )
        for call in call_nodes:
            name = ""
            if isinstance(call.func, ast.Name):
                candidate = str(call.func.id or "").strip()
                if candidate in ACTION_SCRIPT_FUNCTIONS:
                    name = candidate
            elif (
                isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "subprocess"
                and call.func.attr == "run"
            ):
                name = "subprocess.run"

            if not name:
                continue

            try:
                args, kwargs = self._parse_call_args(call)
            except Exception:
                args, kwargs = [], {}
            source = ast.get_source_segment(code, call) or ast.unparse(call).strip()
            actions.append({"name": name, "args": args, "kwargs": kwargs, "source": source})

        return actions

    def _zero_usage_entry(self) -> Dict[str, float]:
        return {"cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "image_count": 0}

    def _build_step_token_usage(self, planner_usage: Dict, tool_usage: Dict, include_planner: bool) -> Dict:
        global_planner_usage = planner_usage["global_planner"] if include_planner else self._zero_usage_entry()
        state_manager_usage = planner_usage["state_manager"] if include_planner else self._zero_usage_entry()
        return {
            "global_planner": global_planner_usage,
            "visual_grounder": tool_usage["visual_grounder"],
            "state_manager": {
                "cost": state_manager_usage["cost"] + tool_usage["state_manager"]["cost"],
                "prompt_tokens": state_manager_usage["prompt_tokens"] + tool_usage["state_manager"]["prompt_tokens"],
                "completion_tokens": state_manager_usage["completion_tokens"] + tool_usage["state_manager"]["completion_tokens"],
                "image_count": state_manager_usage["image_count"] + tool_usage["state_manager"]["image_count"],
            },
            "total": {
                "cost": global_planner_usage["cost"] + tool_usage["visual_grounder"]["cost"] + state_manager_usage["cost"] + tool_usage["state_manager"]["cost"],
                "prompt_tokens": global_planner_usage["prompt_tokens"] + tool_usage["visual_grounder"]["prompt_tokens"] + state_manager_usage["prompt_tokens"] + tool_usage["state_manager"]["prompt_tokens"],
                "completion_tokens": global_planner_usage["completion_tokens"] + tool_usage["visual_grounder"]["completion_tokens"] + state_manager_usage["completion_tokens"] + tool_usage["state_manager"]["completion_tokens"],
                "image_count": global_planner_usage["image_count"] + tool_usage["visual_grounder"]["image_count"] + state_manager_usage["image_count"] + tool_usage["state_manager"]["image_count"],
            },
        }

    def _patch_latest_step_token_usage(self, step: int, token_usage: Dict) -> None:
        if self.action_logs and self.action_logs[-1].get("step") == step:
            self.action_logs[-1]["token_usage"] = token_usage

    def _normalize_gui_tool_input(self, tool: str, code):
        """Normalize GUI-tool input to the dict format expected by desktop_env.step."""
        if tool == "click" and isinstance(code, str):
            return {"tool": tool, "action_type": "pyautogui", "command": "pyautogui.click(X_COORD, Y_COORD)"}
        if tool == "double_click" and isinstance(code, str):
            return {"tool": tool, "action_type": "pyautogui", "command": "pyautogui.doubleClick(X_COORD, Y_COORD)"}
        if tool == "right_click" and isinstance(code, str):
            return {"tool": tool, "action_type": "pyautogui", "command": "pyautogui.rightClick(X_COORD, Y_COORD)"}
        if tool == "move" and isinstance(code, str):
            return {"tool": tool, "action_type": "pyautogui", "command": "pyautogui.moveTo(X_COORD, Y_COORD)"}
        if tool == "drag" and isinstance(code, str):
            return {
                "tool": tool,
                "action_type": "pyautogui",
                "command": "pyautogui.moveTo(START_X_COORD, START_Y_COORD); pyautogui.dragTo(END_X_COORD, END_Y_COORD, duration=0.5, button='left')",
            }
        if tool == "scroll" and isinstance(code, dict):
            return {
                "tool": tool,
                "action_type": "pyautogui",
                "command": f"pyautogui.moveTo(X_COORD, Y_COORD); pyautogui.scroll({_validate_scroll_amount(code.get('amount'))})",
            }
        if tool in {"write", "type"} and isinstance(code, str):
            return {"tool": tool, "action_type": "pyautogui", "command": f"pyautogui.write({code!r})"}
        if tool == "hotkey" and isinstance(code, str) and code.strip():
            keys = [part.strip() for part in re.split(r"\s*\+\s*", code.strip()) if part.strip()]
            if keys:
                rendered = ", ".join(repr(str(k)) for k in keys)
                return {"tool": tool, "action_type": "pyautogui", "command": f"pyautogui.hotkey({rendered})"}

        normalized_code = self._normalize_pyautogui_code(code)
        if isinstance(normalized_code, str) and normalized_code.strip():
            return {
                "tool": tool,
                "action_type": "pyautogui",
                "command": normalized_code,
            }
        return normalized_code

    def _parse_pyautogui_code(self, code: str) -> List[Dict]:
        code = self._normalize_pyautogui_code(code)
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise ValueError(f"Failed to parse GUI tool code: {e}") from e

        statements = []
        for stmt in tree.body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                call = stmt.value
                if (
                    isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "pyautogui"
                ):
                    statements.append(
                        {
                            "type": "call",
                            "method": call.func.attr,
                            "args": [ast.literal_eval(arg) for arg in call.args],
                            "kwargs": [(kw.arg, ast.literal_eval(kw.value)) for kw in call.keywords],
                        }
                    )
                    continue
            statements.append({"type": "raw", "code": ast.unparse(stmt)})
        return statements

    def _format_py_value(self, value) -> str:
        return repr(value)

    def _build_pyautogui_call(self, method: str, args: List, kwargs: List[Tuple[str, object]]) -> str:
        params = [self._format_py_value(arg) for arg in args]
        params.extend(f"{key}={self._format_py_value(value)}" for key, value in kwargs)
        return f"pyautogui.{method}({', '.join(params)})"

    def _serialize_pyautogui_code(self, statements: List[Dict]) -> str:
        rendered = []
        for stmt in statements:
            if stmt["type"] == "call":
                rendered.append(self._build_pyautogui_call(stmt["method"], stmt["args"], stmt["kwargs"]))
            else:
                rendered.append(stmt["code"])
        return "; ".join(part for part in rendered if part).strip()

    def _find_first_call(self, statements: List[Dict], method: str) -> Optional[Dict]:
        for stmt in statements:
            if stmt.get("type") == "call" and stmt.get("method") == method:
                return stmt
        return None

    def _set_call_point(self, stmt: Dict, x: int, y: int) -> None:
        kwargs = dict(stmt["kwargs"])
        if "x" in kwargs or "y" in kwargs:
            kwargs["x"] = x
            kwargs["y"] = y
            stmt["kwargs"] = [(key, kwargs[key]) for key, _ in stmt["kwargs"] if key in kwargs] + [
                (key, value) for key, value in kwargs.items() if key not in {k for k, _ in stmt["kwargs"]}
            ]
            return

        args = list(stmt["args"])
        if len(args) >= 2:
            args[0], args[1] = x, y
        else:
            args = [x, y] + args
        stmt["args"] = args

    def _insert_move_to_before(self, statements: List[Dict], target_stmt: Dict, x: int, y: int) -> None:
        move_stmt = {"type": "call", "method": "moveTo", "args": [x, y], "kwargs": []}
        for idx, stmt in enumerate(statements):
            if stmt is target_stmt:
                statements.insert(idx, move_stmt)
                return
        statements.insert(0, move_stmt)

    def _extract_grounded_point(self, grounded_cmd: str, action_name: str = "moveTo") -> Tuple[int, int]:
        match = re.search(rf"pyautogui\.{re.escape(action_name)}\((\d+), (\d+)\)", grounded_cmd)
        if not match:
            raise ValueError(f"Failed to extract grounded coordinates from: {grounded_cmd}")
        return int(match.group(1)), int(match.group(2))

    def _ground_gui_code(self, code: str, description: str, screenshot: bytes) -> str:
        """Auto-ground mouse-position GUI tool code from action type and description."""
        if not isinstance(code, str) or not code.strip():
            return code

        grounded_code = code
        has_placeholders = any(
            token in grounded_code
            for token in [
                "X_COORD", "Y_COORD",
                "START_X_COORD", "START_Y_COORD", "END_X_COORD", "END_Y_COORD",
            ]
        )
        if has_placeholders:
            if not description:
                raise ValueError("Grounding requires action intent, but thought was empty.")
            return self._call_visual_grounder(description, screenshot, grounded_code)
        statements = self._parse_pyautogui_code(grounded_code)

        if "pyautogui.dragTo(" in grounded_code:
            if not description:
                return grounded_code

            start_desc = f"Locate the drag starting point for: {description}"
            end_desc = f"Locate the drag ending point for: {description}"
            start_cmd = self._call_visual_grounder(start_desc, screenshot, "pyautogui.moveTo(X_COORD, Y_COORD)")
            end_cmd = self._call_visual_grounder(end_desc, screenshot, "pyautogui.moveTo(X_COORD, Y_COORD)")
            start_x, start_y = self._extract_grounded_point(start_cmd)
            end_x, end_y = self._extract_grounded_point(end_cmd)
            drag_stmt = self._find_first_call(statements, "dragTo")
            if drag_stmt is None:
                raise ValueError("Failed to find dragTo action in GUI tool code")
            move_stmt = self._find_first_call(statements, "moveTo")
            if move_stmt is not None:
                self._set_call_point(move_stmt, start_x, start_y)
            else:
                self._insert_move_to_before(statements, drag_stmt, start_x, start_y)
            self._set_call_point(drag_stmt, end_x, end_y)
            return self._serialize_pyautogui_code(statements)

        single_point_actions = ["click", "doubleClick", "rightClick", "moveTo"]
        matched_single_action = next(
            (name for name in single_point_actions if f"pyautogui.{name}(" in grounded_code),
            None
        )
        if matched_single_action:
            if not description:
                raise ValueError(f"Grounding requires action intent for {matched_single_action}, but thought was empty.")
            grounded_point = self._call_visual_grounder(description, screenshot, "pyautogui.moveTo(X_COORD, Y_COORD)")
            point_x, point_y = self._extract_grounded_point(grounded_point)
            action_stmt = self._find_first_call(statements, matched_single_action)
            if action_stmt is None:
                raise ValueError(f"Failed to find {matched_single_action} action in GUI tool code")
            self._set_call_point(action_stmt, point_x, point_y)
            return self._serialize_pyautogui_code(statements)

        if "pyautogui.scroll(" in grounded_code:
            if not description:
                raise ValueError("Grounding requires action intent for scroll, but thought was empty.")
            grounded_point = self._call_visual_grounder(description, screenshot, "pyautogui.moveTo(X_COORD, Y_COORD)")
            x, y = self._extract_grounded_point(grounded_point)
            move_stmt = self._find_first_call(statements, "moveTo")
            scroll_stmt = self._find_first_call(statements, "scroll")
            if scroll_stmt is None:
                raise ValueError("Failed to find scroll action in GUI tool code")
            scroll_kwargs = dict(scroll_stmt["kwargs"])
            if "x" in scroll_kwargs or "y" in scroll_kwargs:
                self._set_call_point(scroll_stmt, x, y)
            elif move_stmt is not None:
                self._set_call_point(move_stmt, x, y)
            else:
                self._insert_move_to_before(statements, scroll_stmt, x, y)
            return self._serialize_pyautogui_code(statements)

        return grounded_code
    
    def _call_visual_grounder(self, description: str, screenshot: bytes, code: str):
        """Call visual grounder to get coordinates or code using call_cua.
        
        Returns:
            - If GTA1: dict with {"x": x, "y": y} for coordinate replacement
            - If other models: string with complete pyautogui code
        """
        # Convert screenshot bytes to PIL Image
        img = Image.open(io.BytesIO(screenshot))

        def call_grounder(target_desc: str):
            scale = self.visual_grounder_scale if self.visual_grounder_llm.model_name.startswith("gta1") else 1.0
            self._dump_prompt_entry(
                stage="visual_grounder",
                payload={
                    "target_description": target_desc,
                    "code_template": code,
                    "environment": "linux",
                    "screen_width": self.screen_width,
                    "screen_height": self.screen_height,
                    "scale": scale,
                    "screenshot": "<omitted image bytes>",
                },
            )
            py_cmd, reasoning = self.visual_grounder_llm.call_cua(
                target_desc,
                img,
                environment="linux",
                screen_width=self.screen_width,
                screen_height=self.screen_height,
                scale=scale
            )
            self._dump_prompt_entry(
                stage="visual_grounder_response",
                payload={
                    "py_cmd": py_cmd,
                    "reasoning": reasoning,
                },
            )
            if not py_cmd:
                raise ValueError(f"Visual Grounder failed to provide result. Reasoning: {reasoning}")
            return py_cmd

        # Single-point placeholder mode
        py_cmd = call_grounder(description)
        if "gta1" in self.visual_grounder_model.lower():
            if isinstance(py_cmd, tuple) and len(py_cmd) == 2:
                x, y = py_cmd
                return code.replace("X_COORD", str(x)).replace("Y_COORD", str(y))
            raise ValueError(f"[GTA1] Expected (x, y) tuple, got: {py_cmd}")
        return py_cmd

    def _execute_gui_tool(self, tool: str, code: Any, description: str = "") -> str:
        """Execute a concrete GUI tool via pyautogui code with optional grounding."""
        code = self._normalize_gui_tool_input(tool, code)
        requested_action_fingerprint = self._hash_text(code)
        if description:
            self.logger.info(f"[{tool}] {description}")
        else:
            self.logger.info(f"[{tool}] {code}")

        # Record step start time
        step_start_time = time.time()

        step = self.current_step_id or (self.operation_count + 1)

        try:
            # Get before screenshot
            before_screenshot = self.env.controller.get_screenshot()
            screenshot_file = f"step_{step}.png"
            grounded_code = self._ground_gui_code(
                code.get("command", "") if isinstance(code, dict) else code,
                description,
                before_screenshot,
            )
            if isinstance(code, dict):
                code["command"] = grounded_code
            else:
                code = grounded_code

            # Execute code
            if isinstance(code, dict):
                final_code = dict(code)
                final_code["command"] = postprocess_action(final_code.get("command", ""))
            else:
                final_code = postprocess_action(code)
            obs, *_ = self.env.step(final_code, self.sleep_after_execution)

            observed_screenshot = obs.get('screenshot') if isinstance(obs, dict) else None
            after_screenshot, env_changed, wait_elapsed = self._wait_for_environment_change(
                before_screenshot,
                timeout_seconds=self.post_action_wait_timeout,
                initial_after_screenshot=observed_screenshot,
            )
            with open(os.path.join(self.operations_dir, screenshot_file), "wb") as f:
                f.write(after_screenshot)

            # Create description for step abstraction
            eval_desc = description if description else code
            
            blocked_hint = (
                f"no_visible_change after {wait_elapsed:.1f}s"
                if not env_changed else ""
            )
            step_abstraction_result = self._step_abstraction(
                before_screenshot, after_screenshot, eval_desc,
                blocked_hint=blocked_hint,
                wo_roi=self.wo_roi, roi_margin=self.roi_margin,
                execution_context={
                    "type": "gui",
                    "code": final_code,
                    "status": "success" if env_changed else "timeout",
                },
            )
            step_abstraction_summary = "Result: " + step_abstraction_result
            self.logger.info(f"[step_abstraction] Step {step}: {step_abstraction_summary}")

            # Generate step_abstract
            thought_prefix = self.current_thought if self.current_thought else ""
            if description:
                step_abstract = (
                    f"Step {step}:\n"
                    f"GUI tool: {tool}.\n"
                    f"Description: {description}.\n"
                    f"Code: {final_code}."
                )
            else:
                step_abstract = (
                    f"Step {step}:\n"
                    f"GUI tool: {tool}.\n"
                    f"Code: {final_code}."
                )
            if thought_prefix:
                step_abstract += f"\nReasoning: {thought_prefix}"
            if step_abstraction_summary:
                step_abstract += f"\n{step_abstraction_summary}"
            step_abstract += (
                f"\nWait observation: {'changed' if env_changed else 'timeout/no visible change'} after {wait_elapsed:.1f}s."
            )

            # Calculate step execution time
            step_time = time.time() - step_start_time
            # GUI loop detection should be robust to dynamic pixels (clock/cursor/animations),
            # so do not fingerprint raw screenshots.
            result_fingerprint = self._hash_text("success=True")

            self.action_logs.append({
                "step": step,
                "type": tool,
                "description": description,
                "execution_success": True,
                "screenshot": screenshot_file,
                "detail": description or str(final_code),
                "compact": self._build_compact_log_entry(
                    step=step,
                    tool_type=tool,
                    success=True,
                    detail=description or final_code,
                    verification=(
                        (step_abstraction_summary.replace("Result: ", "") + f" Wait: {'changed' if env_changed else 'timeout/no visible change'} after {wait_elapsed:.1f}s.")
                        if step_abstraction_summary else f"wait={'changed' if env_changed else 'timeout'} after {wait_elapsed:.1f}s"
                    ),
                    next_hint="Verify the exact requested outcome before terminating."
                ),
                "step_time": round(step_time, 2),
                "token_usage": self.step_token_usage,
                "loop_action_fingerprint": requested_action_fingerprint,
                "loop_result_fingerprint": result_fingerprint,
                "decision_action_fingerprint": self.current_decision_action_fingerprint,
                "decision_gui_fingerprint": self.current_decision_gui_fingerprint,
                "decision_result_fingerprint": self._hash_text(
                    f"env_changed={env_changed}|detail={description or final_code}"
                ),
            })
            if not env_changed:
                self.last_blocked_feedback_event = {
                    "type": "no_visible_change",
                    "detail": description or str(final_code),
                }

            # Return execution result text for wo_step mode
            if description:
                return f"GUI Tool ({tool}): {description}\nCode: {final_code}\nStatus: Success\nWait: {'changed' if env_changed else 'timeout/no visible change'} after {wait_elapsed:.1f}s\n{step_abstraction_summary}"
            else:
                return f"GUI Tool ({tool}) Code: {final_code}\nStatus: Success\nWait: {'changed' if env_changed else 'timeout/no visible change'} after {wait_elapsed:.1f}s\n{step_abstraction_summary}"

        except Exception as e:
            self.logger.error(f"GUI tool execution error ({tool}): {e}")

            # Generate step_abstract for error
            thought_prefix = self.current_thought if self.current_thought else ""
            if description:
                step_abstract = (
                    f"Step {step}:\n"
                    f"GUI tool failed: {tool}.\n"
                    f"Description: {description}.\n"
                    f"Code: {code}."
                )
            else:
                step_abstract = (
                    f"Step {step}:\n"
                    f"GUI tool failed: {tool}.\n"
                    f"Code: {code}."
                )
            if thought_prefix:
                step_abstract += f"\nReasoning: {thought_prefix}"
            step_abstract += f"\nError: {str(e)}"

            # Calculate step execution time
            step_time = time.time() - step_start_time
            result_fingerprint = self._hash_text(f"success=False|error={str(e)}")

            self.action_logs.append({
                "step": step,
                "type": tool,
                "description": description,
                "execution_success": False,
                "screenshot": screenshot_file,
                "detail": description or str(code),
                "execution_status": "blocked",
                "blocking_reason": "progress_stall",
                "compact": self._build_compact_log_entry(
                    step=step,
                    tool_type=tool,
                    success=False,
                    detail=description or code,
                    verification=f"Error: {str(e)}",
                    next_hint="Switch target or tool instead of repeating the same GUI interaction."
                ),
                "step_time": round(step_time, 2),
                "token_usage": self.step_token_usage,
                "loop_action_fingerprint": requested_action_fingerprint,
                "loop_result_fingerprint": result_fingerprint,
                "decision_action_fingerprint": self.current_decision_action_fingerprint,
                "decision_gui_fingerprint": self.current_decision_gui_fingerprint,
                "decision_result_fingerprint": self._hash_text(
                    f"status=blocked|error={str(e)}|detail={description or code}"
                ),
            })
            self.last_blocked_feedback_event = {
                "type": "tool_execution_failed",
                "detail": str(e),
            }

            # Return execution result text for wo_step mode
            if description:
                return f"GUI Tool ({tool}): {description}\nCode: {code}\nStatus: Failed\nError: {str(e)}"
            else:
                return f"GUI Tool ({tool}) Code: {code}\nStatus: Failed\nError: {str(e)}"

    def _step_abstraction(self, before_screenshot: Optional[bytes], after_screenshot: Optional[bytes],
            action_description: str, blocked_hint: str = "", wo_roi: bool = False,
            roi_margin: int = 50, execution_context: Optional[Dict[str, Any]] = None) -> str:
        """Abstract step from execution observations.

        Args:
            before_screenshot: Screenshot before action
            after_screenshot: Screenshot after action
            action_description: Description of the action performed
            wo_roi: If True, disable ROI cropping (default: False means ROI cropping is enabled)
            roi_margin: Margin to add around ROI when cropping (default: 50)
            execution_context: Optional execution payload for non-GUI or mixed-output steps

        Returns:
            Concise summary string.
        """
        try:
            prompt = STEP_ABSTRACTION_PROMPT.format(
                action_description=action_description,
                blocked_hint=blocked_hint if blocked_hint else "None",
            )
            observation_lines: List[str] = []
            if execution_context:
                context_type = str(execution_context.get("type", "execution") or "execution")
                observation_lines.append(f"Execution type: {context_type}")
                code_text = str(execution_context.get("code", "") or "").strip()
                if code_text:
                    observation_lines.append(f"Code or command:\n{code_text}")
                status_text = str(execution_context.get("status", "") or "").strip()
                if status_text:
                    observation_lines.append(f"Status: {status_text}")
                exitcode = execution_context.get("exitcode", None)
                if exitcode not in (None, ""):
                    observation_lines.append(f"Exit code: {exitcode}")
                stdout_text = str(execution_context.get("stdout", "") or "").strip()
                if stdout_text:
                    observation_lines.append(f"stdout:\n{stdout_text}")
                stderr_text = str(execution_context.get("stderr", "") or "").strip()
                if stderr_text:
                    observation_lines.append(f"stderr:\n{stderr_text}")
                logs_text = str(execution_context.get("logs", "") or "").strip()
                if logs_text:
                    observation_lines.append(f"Output:\n{logs_text}")

            include_screenshots = (
                before_screenshot is not None
                and after_screenshot is not None
                and self._screenshots_meaningfully_different(before_screenshot, after_screenshot)
            )

            if include_screenshots:
                before_img = Image.open(io.BytesIO(before_screenshot))
                after_img = Image.open(io.BytesIO(after_screenshot))

                if before_img.size != after_img.size:
                    self.logger.error(f"[ANOMALY] Screenshot size mismatch detected!")

                    import datetime
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    error_dir = os.path.join(self.operations_dir, "size_mismatch_errors")
                    os.makedirs(error_dir, exist_ok=True)
                    before_img.save(os.path.join(error_dir, f"{timestamp}_before.png"))
                    after_img.save(os.path.join(error_dir, f"{timestamp}_after.png"))
                    self.logger.error(f"  Saved error screenshots to: {error_dir}")

                if not wo_roi:
                    try:
                        cropped_before, cropped_after = get_change_roi(
                            before_img, after_img,
                            margin=roi_margin,
                        )

                        if cropped_before is not None and cropped_after is not None:
                            before_img = cropped_before
                            after_img = cropped_after
                        else:
                            raise ValueError("Step abstraction found no visual change ROI")
                    except Exception as roi_error:
                        self.logger.warning(f"ROI detection failed, using full screenshots: {roi_error}")

                before_buffer = io.BytesIO()
                after_buffer = io.BytesIO()
                before_img.save(before_buffer, format="PNG")
                after_img.save(after_buffer, format="PNG")

                before_b64 = base64.b64encode(before_buffer.getvalue()).decode("utf-8")
                after_b64 = base64.b64encode(after_buffer.getvalue()).decode("utf-8")

                content: List[Dict[str, str]] = []
                if observation_lines:
                    content.append({"type": "input_text", "text": "Execution observations:\n" + "\n\n".join(observation_lines)})
                content.extend([
                    {"type": "input_text", "text": "Before screenshot:"},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{before_b64}"},
                    {"type": "input_text", "text": "After screenshot:"},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{after_b64}"},
                    {"type": "input_text", "text": prompt},
                ])
                messages = [{"role": "user", "content": content}]
            else:
                if not observation_lines:
                    observation_lines.append("No meaningful screenshot change was observed.")
                messages = [
                    {
                        "role": "user",
                        "content": "Execution observations:\n" + "\n\n".join(observation_lines) + "\n\n" + prompt,
                    }
                ]

            self._dump_prompt_entry(stage="step_abstraction", payload={"messages": messages})
            step_abstraction = self.state_manager_llm(messages, enable_thinking=False)
            self._dump_prompt_entry(stage="step_abstraction_response", payload=step_abstraction)
            return self._parse_abstraction_payload(step_abstraction.strip())

        except Exception as e:
            self.logger.error(f"Failed to abstract step: {e}")
            raise

    def _bash_execution(self, code: str) -> str:
        """Execute bash commands or Python scripts (not pyautogui)."""
        code = self._normalize_bash_command(code)
        action_fingerprint = self._hash_text(code)
        self.logger.info(f"[bash_execution] {code}")

        # Record step start time
        step_start_time = time.time()

        step = self.current_step_id or (self.operation_count + 1)

        try:
            # Provider workaround:
            # run_bash_script is unstable on some providers, so execute bash via run_python_script.
            escaped_code = json.dumps(code)
            escaped_working_dir = json.dumps(self.bash_working_dir)
            py_wrapper = f"""
import subprocess
import sys
import os

cmd = {escaped_code}
working_dir = os.path.expanduser({escaped_working_dir})
if not os.path.isdir(working_dir):
    working_dir = os.path.expanduser("~")
env = os.environ.copy()
env.setdefault("HOME", os.path.expanduser("~"))
env["SHELL"] = "/bin/bash"
env.setdefault("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
try:
    result = subprocess.run(
        ["/bin/bash", "-lc", cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout={int(self.bash_timeout)},
        cwd=working_dir,
        env=env,
    )
    sys.stdout.write(result.stdout or "")
    sys.exit(result.returncode)
except subprocess.TimeoutExpired as e:
    sys.stdout.write((e.stdout or "") + "\\n[TimeoutExpired]")
    sys.exit(124)
"""
            output_dict = self.env.controller.run_python_script(py_wrapper)
            output_dict = output_dict or {}
            status = output_dict.get("status", "error")
            exitcode = output_dict.get("return_code", 1)
            logs = output_dict.get("output", "")
            if not logs and output_dict.get("message"):
                logs = output_dict.get("message", "")
            if status != "success" and output_dict.get("error"):
                logs = (logs + "\n" + output_dict.get("error", "")).strip()

            before_screenshot = self.env.controller.get_screenshot()
            after_screenshot, env_changed, wait_elapsed = self._wait_for_environment_change(
                before_screenshot,
                timeout_seconds=self.post_action_wait_timeout,
            )
            screenshot_file = f"step_{step}.png"

            with open(os.path.join(self.operations_dir, screenshot_file), "wb") as f:
                f.write(after_screenshot)

            blocked_hint = ""
            if exitcode != 0 or status != "success":
                blocked_hint = logs or f"exitcode={exitcode}"
            step_abstraction_result = self._step_abstraction(
                before_screenshot=None,
                after_screenshot=None,
                action_description=f"Bash command: {code}",
                blocked_hint=blocked_hint,
                execution_context={
                    "type": "bash_execution",
                    "code": code,
                    "logs": logs,
                    "status": status,
                    "exitcode": exitcode,
                },
            )
            step_abstraction_summary = "Result: " + step_abstraction_result
            self.logger.info(f"[step_abstraction] Step {step}: {step_abstraction_summary}")

            # Generate step_abstract summary
            thought_prefix = self.current_thought if self.current_thought else ""
            if step_abstraction_summary:
                step_abstract = (
                    f"Step {step}:\n"
                    f"Bash command.\n"
                    f"Code: {code}."
                )
            else:
                step_abstract = (
                    f"Step {step}:\n"
                    f"Bash command.\n"
                    f"Code: {code}."
                )
            if thought_prefix:
                step_abstract += f"\nReasoning: {thought_prefix}"
            if step_abstraction_summary:
                step_abstract += f"\n{step_abstraction_summary}"
            step_abstract += (
                f"\nWait observation: {'changed' if env_changed else 'timeout/no visible change'} after {wait_elapsed:.1f}s."
            )
            result_fingerprint = self._hash_text(
                f"status={status}|exitcode={exitcode}|output={logs}|wait={'changed' if env_changed else 'timeout'}"
            )

            # Calculate step execution time
            step_time = time.time() - step_start_time

            self.action_logs.append({
                "step": step,
                "type": "bash_execution",
                "execution_success": exitcode == 0 and status == "success",
                "screenshot": screenshot_file,
                "detail": code,
                "compact": self._build_compact_log_entry(
                    step=step,
                    tool_type="bash_execution",
                    success=(exitcode == 0 and status == "success"),
                    detail=code,
                    verification=(
                        (step_abstraction_summary.replace("Result: ", "") + f" Wait: {'changed' if env_changed else 'timeout/no visible change'} after {wait_elapsed:.1f}s.")
                        if step_abstraction_summary else f"exitcode={exitcode}; wait={'changed' if env_changed else 'timeout'} after {wait_elapsed:.1f}s"
                    ),
                    next_hint="Use the command output to decide whether GUI verification is still needed."
                ),
                "step_time": round(step_time, 2),
                "token_usage": self.step_token_usage,
                "loop_action_fingerprint": action_fingerprint,
                "loop_result_fingerprint": result_fingerprint,
                "decision_action_fingerprint": self.current_decision_action_fingerprint,
                "decision_gui_fingerprint": self.current_decision_gui_fingerprint,
                "decision_result_fingerprint": self._hash_text(
                    f"exitcode={exitcode}|output={logs}"
                ),
            })
            if exitcode != 0 or status != "success":
                self.last_blocked_feedback_event = {
                    "type": "tool_execution_failed",
                    "detail": logs or f"exitcode={exitcode}",
                }

            # Return execution result text for wo_step mode
            status_str = "Success" if (exitcode == 0 and status == "success") else "Failed"
            return f"Bash Command: {code}\nStatus: {status_str}\nWait: {'changed' if env_changed else 'timeout/no visible change'} after {wait_elapsed:.1f}s\nOutput:\n{logs}"
            
        except Exception as e:
            self.logger.error(f"Bash execution error: {e}")

            screenshot = self.env.controller.get_screenshot()
            screenshot_file = f"step_{step}.png"

            with open(os.path.join(self.operations_dir, screenshot_file), "wb") as f:
                f.write(screenshot)

            # Generate step_abstract summary for error
            thought_prefix = self.current_thought if self.current_thought else ""
            step_abstract = (
                f"Step {step}:\n"
                f"Bash execution failed.\n"
                f"Code: {code}."
            )
            if thought_prefix:
                step_abstract += f"\nReasoning: {thought_prefix}"
            step_abstract += f"\nError: {str(e)}"
            result_fingerprint = self._hash_text(f"error={str(e)}")

            self.action_logs.append({
                "step": step,
                "type": "bash_execution",
                "execution_success": False,
                "screenshot": screenshot_file,
                "detail": code,
                "execution_status": "blocked",
                "blocking_reason": "progress_stall",
                "compact": self._build_compact_log_entry(
                    step=step,
                    tool_type="bash_execution",
                    success=False,
                    detail=code,
                    verification=f"Error: {str(e)}",
                    next_hint="Try a different command or switch back to GUI if the shell path is brittle."
                ),
                "token_usage": self.step_token_usage,
                "loop_action_fingerprint": action_fingerprint,
                "loop_result_fingerprint": result_fingerprint,
                "decision_action_fingerprint": self.current_decision_action_fingerprint,
                "decision_gui_fingerprint": self.current_decision_gui_fingerprint,
                "decision_result_fingerprint": self._hash_text(
                    f"status=blocked|bash_error={str(e)}|code={code}"
                ),
            })
            self.last_blocked_feedback_event = {
                "type": "tool_execution_failed",
                "detail": str(e),
            }

            # Return execution result text for wo_step mode
            return f"Bash Command: {code}\nStatus: Failed\nError: {str(e)}"

    def _subprocess_run_execution(self, args: List[Any], kwargs: Dict[str, Any], return_completed_process: bool = False):
        command = args[0]
        shell = bool(kwargs.get("shell", isinstance(command, str)))
        timeout = kwargs.get("timeout", self.bash_timeout)
        cwd = kwargs.get("cwd", self.bash_working_dir)
        capture_output = bool(kwargs.get("capture_output", True))
        text_mode = bool(kwargs.get("text", True))
        check = bool(kwargs.get("check", False))
        command_repr = command if isinstance(command, str) else list(command)
        command_text = f"subprocess.run({command_repr!r})"
        action_fingerprint = self._hash_text(
            f"{command_repr}|shell={shell}|timeout={timeout}|cwd={cwd}|capture_output={capture_output}|text={text_mode}|check={check}"
        )
        self.logger.info(f"[subprocess.run] {command_text}")

        step_start_time = time.time()
        step = self.current_step_id or (self.operation_count + 1)

        try:
            py_wrapper = f"""
import subprocess
import sys
import os

command = {json.dumps(command)}
working_dir = os.path.expanduser({json.dumps(cwd)})
if not os.path.isdir(working_dir):
    working_dir = os.path.expanduser("~")
env = os.environ.copy()
env.setdefault("HOME", os.path.expanduser("~"))
env["SHELL"] = "/bin/bash"
env.setdefault("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
result = subprocess.run(
    command,
    shell={repr(shell)},
    stdout=subprocess.PIPE if {repr(capture_output)} else None,
    stderr=subprocess.STDOUT if {repr(capture_output)} else None,
    text={repr(text_mode)},
    timeout={repr(timeout)},
    cwd=working_dir,
    env=env,
    check=False,
)
output = result.stdout if result.stdout is not None else ""
sys.stdout.write(output)
sys.exit(result.returncode)
"""
            output_dict = self.env.controller.run_python_script(py_wrapper) or {}
            status = output_dict.get("status", "error")
            exitcode = output_dict.get("return_code", 1)
            logs = output_dict.get("output", "")
            if not logs and output_dict.get("message"):
                logs = output_dict.get("message", "")
            if status != "success" and output_dict.get("error"):
                logs = (logs + "\n" + output_dict.get("error", "")).strip()

            if check and exitcode != 0:
                status = "error"

            before_screenshot = self.env.controller.get_screenshot()
            after_screenshot, env_changed, wait_elapsed = self._wait_for_environment_change(
                before_screenshot,
                timeout_seconds=self.post_action_wait_timeout,
            )
            screenshot_file = f"step_{step}.png"
            with open(os.path.join(self.operations_dir, screenshot_file), "wb") as f:
                f.write(after_screenshot)

            blocked_hint = ""
            if exitcode != 0 or status != "success":
                blocked_hint = logs or f"exitcode={exitcode}"
            step_abstraction_result = self._step_abstraction(
                before_screenshot=None,
                after_screenshot=None,
                action_description=command_text,
                blocked_hint=blocked_hint,
                execution_context={
                    "type": "subprocess.run",
                    "code": command_text,
                    "logs": logs,
                    "status": status,
                    "exitcode": exitcode,
                },
            )
            step_abstraction_summary = "Result: " + step_abstraction_result
            self.logger.info(f"[step_abstraction] Step {step}: {step_abstraction_summary}")

            step_time = time.time() - step_start_time
            result_fingerprint = self._hash_text(
                f"status={status}|exitcode={exitcode}|output={logs}|wait={'changed' if env_changed else 'timeout'}"
            )
            success = exitcode == 0 and status == "success"
            self.action_logs.append({
                "step": step,
                "type": "subprocess_execution",
                "execution_success": success,
                "screenshot": screenshot_file,
                "detail": command_text,
                "compact": self._build_compact_log_entry(
                    step=step,
                    tool_type="subprocess_execution",
                    success=success,
                    detail=command_text,
                    verification=(
                        (step_abstraction_summary.replace("Result: ", "") + f" Wait: {'changed' if env_changed else 'timeout/no visible change'} after {wait_elapsed:.1f}s.")
                        if step_abstraction_summary else f"exitcode={exitcode}; wait={'changed' if env_changed else 'timeout'} after {wait_elapsed:.1f}s"
                    ),
                    next_hint="Use the command output to decide whether GUI verification is still needed."
                ),
                "step_time": round(step_time, 2),
                "token_usage": self.step_token_usage,
                "loop_action_fingerprint": action_fingerprint,
                "loop_result_fingerprint": result_fingerprint,
                "decision_action_fingerprint": self.current_decision_action_fingerprint,
                "decision_gui_fingerprint": self.current_decision_gui_fingerprint,
                "decision_result_fingerprint": self._hash_text(
                    f"exitcode={exitcode}|output={logs}"
                ),
            })
            if not success:
                self.last_blocked_feedback_event = {
                    "type": "tool_execution_failed",
                    "detail": logs or f"exitcode={exitcode}",
                }

            status_str = "Success" if success else "Failed"
            result_text = f"{command_text}\nStatus: {status_str}\nWait: {'changed' if env_changed else 'timeout/no visible change'} after {wait_elapsed:.1f}s\nOutput:\n{logs}"
            completed_process = py_subprocess.CompletedProcess(
                args=command_repr,
                returncode=exitcode,
                stdout=logs if capture_output else None,
                stderr=None,
            )
            if return_completed_process:
                return result_text, completed_process
            return result_text
        except Exception as e:
            self.logger.error(f"subprocess.run execution error: {e}")
            screenshot = self.env.controller.get_screenshot()
            screenshot_file = f"step_{step}.png"
            with open(os.path.join(self.operations_dir, screenshot_file), "wb") as f:
                f.write(screenshot)

            self.action_logs.append({
                "step": step,
                "type": "subprocess_execution",
                "execution_success": False,
                "screenshot": screenshot_file,
                "detail": command_text,
                "execution_status": "blocked",
                "blocking_reason": "progress_stall",
                "compact": self._build_compact_log_entry(
                    step=step,
                    tool_type="subprocess_execution",
                    success=False,
                    detail=command_text,
                    verification=f"Error: {str(e)}",
                    next_hint="Fix the subprocess.run call instead of repeating the same failing command."
                ),
                "token_usage": self.step_token_usage,
                "loop_action_fingerprint": action_fingerprint,
                "loop_result_fingerprint": self._hash_text(f"error={str(e)}"),
                "decision_action_fingerprint": self.current_decision_action_fingerprint,
                "decision_gui_fingerprint": self.current_decision_gui_fingerprint,
                "decision_result_fingerprint": self._hash_text(
                    f"status=blocked|subprocess_error={str(e)}|code={command_text}"
                ),
            })
            self.last_blocked_feedback_event = {
                "type": "tool_execution_failed",
                "detail": str(e),
            }
            result_text = f"{command_text}\nStatus: Failed\nError: {str(e)}"
            if return_completed_process:
                raise
            return result_text

    def _wait(self) -> str:
        """Wait until the environment changes or timeout is reached."""
        wait_timeout = self.explicit_wait_timeout
        self.logger.info(f"[wait] Polling for environment change with timeout={wait_timeout:.1f}s...")

        # Record step start time
        step_start_time = time.time()

        step = self.current_step_id or (self.operation_count + 1)

        try:
            # Get before screenshot
            before_screenshot = self.env.controller.get_screenshot()
            screenshot_file = f"step_{step}.png"

            after_screenshot, env_changed, wait_elapsed = self._wait_for_environment_change(
                before_screenshot,
                timeout_seconds=wait_timeout,
            )
            with open(os.path.join(self.operations_dir, screenshot_file), "wb") as f:
                f.write(after_screenshot)

            blocked_hint = (
                f"wait timed out after {wait_elapsed:.1f}s"
                if not env_changed else ""
            )
            step_abstraction_result = self._step_abstraction(
                before_screenshot, after_screenshot,
                f"Waited until the environment changed or timed out after {wait_timeout:.1f} seconds",
                blocked_hint=blocked_hint,
                wo_roi=self.wo_roi, roi_margin=self.roi_margin,
                execution_context={
                    "type": "wait",
                    "status": "changed" if env_changed else "timeout",
                },
            )
            step_abstraction_summary = "Result: " + step_abstraction_result
            self.logger.info(f"[step_abstraction] Step {step}: {step_abstraction_summary}")

            # Generate step_abstract
            thought_prefix = self.current_thought if self.current_thought else ""
            step_abstract = (
                f"Step {step}:\n"
                f"Wait.\n"
                f"Mode: event-driven wait with timeout {wait_timeout:.1f} seconds."
            )
            if thought_prefix:
                step_abstract += f"\nReasoning: {thought_prefix}"
            if step_abstraction_summary:
                step_abstract += f"\n{step_abstraction_summary}"
            step_abstract += f"\nOutcome: {'environment changed' if env_changed else 'timeout/no visible change'} after {wait_elapsed:.1f}s."

            # Calculate step execution time
            step_time = time.time() - step_start_time

            self.action_logs.append({
                "step": step,
                "type": "wait",
                "execution_success": True,
                "screenshot": screenshot_file,
                "detail": f"event-driven wait ({'changed' if env_changed else 'timeout'})",
                "compact": self._build_compact_log_entry(
                    step=step,
                    tool_type="wait",
                    success=True,
                    detail=f"event-driven wait ({'changed' if env_changed else 'timeout'})",
                    verification=(
                        (step_abstraction_summary.replace("Result: ", "") + f" Outcome: {'changed' if env_changed else 'timeout/no visible change'} after {wait_elapsed:.1f}s.")
                        if step_abstraction_summary else f"{'changed' if env_changed else 'timeout'} after {wait_elapsed:.1f}s"
                    ),
                    next_hint="Check whether the requested UI state is now visible."
                ),
                "step_time": round(step_time, 2),
                "token_usage": self.step_token_usage
            })
            if not env_changed:
                self.last_blocked_feedback_event = {
                    "type": "no_visible_change",
                    "detail": f"event-driven wait timed out after {wait_elapsed:.1f}s",
                }

            # Return execution result text for wo_step mode
            return f"Wait: event-driven\nStatus: Success\nOutcome: {'changed' if env_changed else 'timeout/no visible change'} after {wait_elapsed:.1f}s\n{step_abstraction_summary}"

        except Exception as e:
            self.logger.error(f"Wait execution error: {e}")

            # Generate step_abstract for error
            thought_prefix = self.current_thought if self.current_thought else ""
            step_abstract = (
                f"Step {step}:\n"
                f"Wait failed.\n"
                f"Mode: event-driven wait with timeout {wait_timeout:.1f} seconds."
            )
            if thought_prefix:
                step_abstract += f"\nReasoning: {thought_prefix}"
            step_abstract += f"\nError: {str(e)}"

            # Calculate step execution time
            step_time = time.time() - step_start_time

            self.action_logs.append({
                "step": step,
                "type": "wait",
                "execution_success": False,
                "screenshot": screenshot_file if 'screenshot_file' in locals() else "",
                "detail": "event-driven wait failed",
                "execution_status": "blocked",
                "blocking_reason": "progress_stall",
                "compact": self._build_compact_log_entry(
                    step=step,
                    tool_type="wait",
                    success=False,
                    detail="event-driven wait failed",
                    verification=f"Error: {str(e)}",
                    next_hint="Re-check the app state directly instead of relying on the failed wait."
                ),
                "step_time": round(step_time, 2),
                "token_usage": self.step_token_usage
            })
            self.last_blocked_feedback_event = {
                "type": "tool_execution_failed",
                "detail": str(e),
            }

            # Return execution result text for wo_step mode
            return f"Wait: event-driven\nStatus: Failed\nError: {str(e)}"


    def _evaluate_and_save(self, task_config: dict, additional_context: str,
                          is_infeasible: bool = False, termination_reason: str = "") -> float:
        """Evaluate task and save results."""
        self.logger.info(f"\n{'='*80}")
        self.logger.info("Task Evaluation")
        self.logger.info("="*80)

        # Extract and save pattern BEFORE evaluating score
        # This prevents data leakage - lessons should be based on execution process only
        domain = task_config.get("domain", "general")
        task_instruction = task_config["instruction"]
        if additional_context:
            task_instruction += f"\n{additional_context}"

        if not self.wo_pattern:
            self.logger.info("Inducing pattern...")
            key_lessons = self.pattern_manager.pattern_induction(
                task_instruction=task_instruction,
                action_logs=self.action_logs
            )

            if key_lessons:
                self.pattern_manager.save_pattern(
                    domain,
                    key_lessons,
                    task_instruction=task_instruction,
                    task_id=getattr(self, "current_task_id", ""),
                    task_signature=getattr(self, "current_task_signature", ""),
                    task_tags=getattr(self, "current_task_tags", []),
                )
                # Format lessons as numbered list for logging
                formatted_lessons = "\n".join(f"  {i+1}. [{lesson['type']}] {lesson['lesson']}" for i, lesson in enumerate(key_lessons))
                memory_dir = self.pattern_manager._get_memory_dir(domain)
                self.logger.info(
                    f"Saved {len(key_lessons)} lesson(s) to {memory_dir}:\n{formatted_lessons}"
                )
            else:
                self.logger.info("No significant lessons to save")

        # Now evaluate score
        try:
            # self.logger.info("Closing temporary windows...")
            # self.env.step("pyautogui.press('esc')", 0.5)

            if self.evaluation_settle_seconds > 0:
                self.logger.info(
                    f"Waiting {self.evaluation_settle_seconds:.1f} seconds before evaluation to let the environment settle..."
                )
                time.sleep(self.evaluation_settle_seconds)

            # Poll evaluation until it succeeds or the timeout expires.
            self.logger.info("Polling evaluation until the VM is ready...")
            score = self._evaluate_with_polling()
        except Exception as e:
            self.logger.error(
                f"Evaluation failed within {self.evaluation_wait_timeout:.1f} seconds: {e}"
            )
            score = 0.0

        gui_steps = len({
            log.get("step")
            for log in self.action_logs
            if log.get("type") in GUI_ACTION_TOOLS and log.get("step") is not None
        })
        bash_steps = len([log for log in self.action_logs if log["type"] == "bash_execution"])
        wait_steps = len([log for log in self.action_logs if log["type"] == "wait"])
        termination_steps = len([log for log in self.action_logs if log["type"] == "termination"])
        infeasible_steps = len([log for log in self.action_logs if log["type"] == "infeasible"])

        global_planner_cost, global_planner_prompt, global_planner_completion, global_planner_images = self.global_planner_llm.get_usage()
        visual_grounder_cost, visual_grounder_prompt, visual_grounder_completion, visual_grounder_images = self.visual_grounder_llm.get_usage()
        state_manager_cost, state_manager_prompt, state_manager_completion, state_manager_images = self.state_manager_llm.get_usage()

        total_cost = global_planner_cost + visual_grounder_cost + state_manager_cost
        total_images = global_planner_images + visual_grounder_images + state_manager_images

        # Calculate execution time
        execution_time = time.time() - self.start_time

        # Determine success and failure reason (score is 0 or 1)
        failure_reason = ""
        if is_infeasible:
            # Use termination_reason if provided (contains detailed infeasible explanation)
            failure_reason = termination_reason if termination_reason else "Task marked as infeasible"
        elif termination_reason:
            failure_reason = termination_reason

        execution_log = {
            "statistics": {
                "score": score,
                "total_steps": self.operation_count,
                "cua_steps": gui_steps,
                "coding_steps": bash_steps,
                "wait_steps": wait_steps,
                "termination_steps": termination_steps,
                "infeasible_steps": infeasible_steps,
                "image_count": total_images,
                "total_cost": total_cost,
                "prompt_tokens": global_planner_prompt + visual_grounder_prompt + state_manager_prompt,
                "completion_tokens": global_planner_completion + visual_grounder_completion + state_manager_completion,
                "execution_time": execution_time,
                "model_usage": {
                    "global_planner": {
                        "model_name": self.global_planner_model,
                        "cost": global_planner_cost,
                        "prompt_tokens": global_planner_prompt,
                        "completion_tokens": global_planner_completion,
                        "image_count": global_planner_images
                    },
                    "visual_grounder": {
                        "model_name": self.visual_grounder_model,
                        "cost": visual_grounder_cost,
                        "prompt_tokens": visual_grounder_prompt,
                        "completion_tokens": visual_grounder_completion,
                        "image_count": visual_grounder_images
                    },
                    "state_manager": {
                        "model_name": self.state_manager_model,
                        "cost": state_manager_cost,
                        "prompt_tokens": state_manager_prompt,
                        "completion_tokens": state_manager_completion,
                        "image_count": state_manager_images
                    }
                }
            },
            "task_config": task_config,
            "additional_context": additional_context,
            "action_logs": self.action_logs,
            "success": score == 1.0,
            "failure_reason": failure_reason
        }

        with open(os.path.join(self.save_dir, "execution_log.json"), "w") as f:
            json.dump(serialize_json(execution_log), f, indent=2)

        with open(os.path.join(self.save_dir, "result.txt"), "w") as f:
            f.write(str(score))

        self.logger.info("="*80)

        return score

    def _save_error_log(self, task_config: dict, additional_context: str, error: Exception) -> float:
        """Save error log and return 0 score."""
        # Save result.txt with 0 score
        with open(os.path.join(self.save_dir, "result.txt"), "w") as f:
            f.write("0.0")
        
        # Save err_reason.txt with error details
        with open(os.path.join(self.save_dir, "err_reason.txt"), "w") as f:
            f.write(f"Fatal error: {str(error)}\n\n{traceback.format_exc()}")
        
        # Skip saving execution_log when error occurs (err_reason.txt already saved)
        
        return 0.0
    
    def cleanup(self):
        """Clean up resources."""
        if self.env:
            self.logger.info("Closing environment...")
            self.env.close()
            self.env = None

        # Close pattern manager to release Qdrant lock
        if hasattr(self, 'pattern_manager') and self.pattern_manager:
            self.logger.info("Closing pattern manager...")
            try:
                if hasattr(self.pattern_manager, 'qdrant') and self.pattern_manager.qdrant:
                    if hasattr(self.pattern_manager.qdrant, 'client'):
                        self.pattern_manager.qdrant.client.close()
            except Exception as e:
                self.logger.warning(f"Error closing Qdrant client: {e}")
            self.pattern_manager = None
