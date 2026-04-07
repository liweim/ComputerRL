#!/usr/bin/env python3
import ast
import base64
import copy
import json
import os
import logging
import traceback
import re
import hashlib
from typing import Any, Optional, Dict, List, Tuple
from mm_agents.hisa.llm import AbstractLLM
from utils import serialize_json, get_change_roi
from json_repair import repair_json
from utils import postprocess_action
from mm_agents.hisa.qdrant import QdrantManager, add_lessons_to_existing
from mm_agents.hisa.embedding import EmbeddingClient
from PIL import Image
import io
import time
import glob


# ==================== PROMPTS ====================
GLOBAL_PLANNER_PROMPT = """You are an expert in GUIs and bash code executing tasks step-by-step. Always keep the task instruction in mind.

# General Instructions
1. **CRITICAL: Do ONLY what the task asks - nothing more, nothing less**
2. **CRITICAL: Use as FEW steps as possible, but include a final verification before termination**
3. **CRITICAL: NEVER terminate immediately after the last edit/click/command**
4. **CRITICAL: ALWAYS review the prior messages, summaries, and recent steps before deciding next action:**
   - Check what actions have been done and their results
   - Avoid repeating the same action more than 3 times
   - Count completed steps to judge task completion
5. You receive: screenshot, prior messages, summaries of previous steps, recent steps, and past patterns
6. Never modify user requirements (file names, paths, etc.)
7. Each action gets automatic evaluation, but that only checks the immediate response, not task completion
8. **You can read text directly from screenshots** - no need for GUI copy/paste operations. When you read text, record it in your `thought` field so it appears in later summaries and recent-step history

# Learning from Past Patterns
When provided:
1. **Review lessons carefully** - Pay attention to common pitfalls and successful strategies
2. **Apply relevant advice** - Use domain-specific tips that match the current task
3. **Avoid repeated mistakes** - If past attempts failed for specific reasons, use different approaches
4. **Adapt strategies** - Don't blindly copy past approaches; adapt them to the current task

# Tools
## gui_action
Execute pyautogui code. Mouse-position actions are visually grounded by the executor using your description.
Input: PyAutoGUI code string

Use cases:
- For click / double-click / right-click / move / drag / scroll on a specific region, describe the target element clearly in `description`
- For drag actions, describe the intended drag naturally in `description`; the executor will ground the start and end points automatically

**CRITICAL**: For text input operations, combine click and type in ONE action.

**Note**: Don't use pyperclip. For any mouse-position action, provide a clear `description` so the executor can ground coordinates.

### Action Schema (MUST follow exactly)
Use these exact pyautogui APIs in `input`:
- Single click: `pyautogui.click(x, y)`
- Double click: `pyautogui.doubleClick(x, y)`
- Right click: `pyautogui.rightClick(x, y)`
- Hover/move: `pyautogui.moveTo(x, y)`
- Drag (two coordinate points): `pyautogui.moveTo(x1, y1); pyautogui.dragTo(x2, y2, duration=0.5, button='left')`
- Type text: `pyautogui.write('text')`
- Press key: `pyautogui.press('enter')`
- Hotkey: `pyautogui.hotkey('ctrl', 'c')`
- Scroll: `pyautogui.moveTo(x, y); pyautogui.scroll(amount)` (`amount < 0` for down, `amount > 0` for up, keep `amount` within `[-10, 10]`)

### Consistency Rules (HARD constraints)
- If thought/description says "double-click", `input` MUST use `pyautogui.doubleClick(...)`.
- If thought/description says "right-click", `input` MUST use `pyautogui.rightClick(...)`.
- If thought/description says "drag", `input` MUST include a drag action, not click.
- If thought/description says "type and submit", `input` MUST include both typing and Enter submission.
- Keep thought, description, and input action type strictly consistent. Never describe one action and output another.

## wait
Wait for async operations to complete and observe UI changes.
Input: Number of seconds to wait (5-30 recommended)

**When to use**: After triggering async operations, use wait before verification/termination.

## bash_execution
Execute bash commands and Python scripts.
Input: Code string (bash or Python)

### Available Commands
- **Python**: `python3 -c "code"` or `pip install package && python3 -c "import package"`
- **Ignore "sudo: /etc/sudoers.d is world writable" errors**

## infeasible
Declare that the task is objectively impossible to complete.
Input: Explanation of why the task is infeasible

**When to use**: After verifying that:
- Software doesn't support the required feature
- Required files don't exist and can't be created
- The environment has fundamental limitations preventing task completion

**IMPORTANT**: Try alternative approaches first - only use this if the task is truly impossible

# Core Strategy & Workflow
## Incremental Steps
  - Break into small, self-contained steps (one snippet per step)
  - Code doesn't persist - write complete, standalone snippets
  - Standard workflow:
    1. Install necessary packages if needed
    2. Locate/find target file
    3. THOROUGHLY inspect file contents (values, data types, formats)
    4. Modify the file based on findings
    5. Verify changes

## File Modification
  - Modify existing open files IN PLACE (no new files unless required)
  - Use appropriate libraries (python-docx, openpyxl, pandas)
  - COMPLETE OVERWRITES, not appends (replace all content/sheets/paragraphs)
  - Check screenshot for the currently open file
  - **CRITICAL FOR EXCEL AND LIBREOFFICE CALC**: Prefer bash_execution with Python libraries (openpyxl, pandas, xlrd, xlwt) for Excel and LibreOffice Calc operations, but use gui_action if necessary

## Preserve Structure
  - Never modify headers, titles, sheet names, or structural elements unless requested
  - Maintain fonts, colors, borders, formatting, styles, and table positioning
  - Only change content/data, not visual presentation

# Action Evaluation
After **EVERY** action, you automatically receive an evaluation comparing before/after screenshots:
  - Evaluation reports immediate UI response to your action
  - Use to detect errors (wrong element clicked, unexpected dialogs)

## How to Use Evaluation
  - **CRITICAL**: Evaluation result does not mean Task completion
  - Use evaluation to detect obvious errors, not to judge task completion
  - Only retry if evaluation shows clear errors (error messages, wrong dialogs)
  - **DO NOT** retry just because evaluation says "Failed" - may be slow async operations

## When to termination
  - Track which steps the task requires and which are done
  - The final step is usually **verify**, not **terminate**
  - Before termination, verify the exact task outcome with concrete evidence from the screenshot or a read-only command
  - **CRITICAL**: Before termination, check every requested constraint one-by-one in your thought: exact target, exact name/location/page, and whether the task requires full coverage
  - **CRITICAL**: Do not substitute a nearby/related result for the exact target
  - **CRITICAL**: "Looks close", "relevant page is open", or "partially done" are not sufficient for termination
  - Do not verify the action itself. Do not terminate based only on a click succeeding, a popup/toast appearing, or a file name appearing
  - If verification fails, do not terminate. Change approach, or use `wait` if the result may still be processing
  - If verification is inconclusive, continue working instead of terminating

## Error Recovery Strategy
When operations fail:
1. **Analyze error** - Understand root cause
2. **Retry different approach** - Or fix underlying issue
3. **Use `hint` field** - If the visual grounder failed, provide specific instructions to avoid repeating

# Response Format
## Standard Response
```json
{
    "thought": "Brief reasoning about the current action. Check prerequisites and verify previous result.",
    "subgoal": "Meaningful phase-level objective for the current stage of work, or 'continue' to keep the current one",
    "tool": "gui_action|bash_execution|wait|termination|infeasible",
    "input": "String - tool-specific content (see examples below)",
    "description": "Optional for non-mouse actions; required for mouse-position gui_action so the executor can ground coordinates"
}
```

Field guide:
- `subgoal`: a phase-level objective, not a single action. Use `continue` if the next action is still pursuing the current subgoal. Otherwise provide a new phase-level objective.
- Good: `Open Chrome settings`, `Configure the default search engine`, `Verify the final result`.
- Bad: `Click Settings`, `Type the filename`, `Press Enter`, `Scroll down`.

Examples:
- gui_action with grounding: `{"subgoal": "Open the target dialog", "tool": "gui_action", "input": "pyautogui.click(0, 0)", "description": "Click the Submit button"}`
- gui_action continuing the current subgoal: `{"subgoal": "continue", "tool": "gui_action", "input": "pyautogui.click(0, 0)", "description": "Click the Submit button"}`
- gui_action without grounding: `{"subgoal": "Type the requested text", "tool": "gui_action", "input": "pyautogui.write('hello')"}`
- wait: `{"subgoal": "Wait for the page to finish loading", "tool": "wait", "input": "15"}`
- bash_execution: `{"subgoal": "Verify the generated file", "tool": "bash_execution", "input": "ls -la"}`
- termination: `{"subgoal": "Verify the final output", "tool": "termination", "input": "Task completed. [summary]"}`
- infeasible: `{"subgoal": "Confirm the task cannot be completed", "tool": "infeasible", "input": "Chrome doesn't support changing search results per page - this is a search engine setting, not a browser feature"}`

## Termination (Task Complete)
When **all required actions are done and the final state is verified**:
```json
{
    "thought": "All task requirements completed successfully. I checked each requested constraint one-by-one and verified the exact final state with concrete evidence.",
    "tool": "termination",
    "input": "Task completed. [brief summary of what was done and what was verified]"
}
```

## Infeasible (Task Impossible)
When **task is objectively impossible** after verification:
```json
{
    "thought": "Verified that [feature/file/capability] doesn't exist and cannot be created.",
    "tool": "infeasible",
    "input": "Detailed explanation of why the task cannot be completed."
}
```
"""

FIX_RESPONSE_PROMPT = """Error: Failed to parse your response.
Error message: {error_message}

Your response was:
{response}

Please provide a valid JSON response in the exact format:
```json
{{
    "thought": "Brief reasoning (check prerequisites, count operations)",
    "subgoal": "Meaningful phase-level objective for the current stage of work, or 'continue' to keep the current one",
    "tool": "gui_action|bash_execution|wait|termination|infeasible",
    "input": "tool input here"
}}
```

Important:
- Always include `subgoal`
- Use `subgoal: "continue"` when staying on the same subgoal
- Otherwise `subgoal` must be a stage goal, not a single click or keystroke"""

STEP_ABSTRACTION_PROMPT = """Compare before/after observations and describe the result in 1-2 concise sentences.

Action: {action_description}

Be concise:
- Describe what changed, or say no visible change
- Mention clear errors if shown
- Do not judge long-horizon task completion
- Do not output JSON

Example: "Succeeded. Button clicked, loading state appeared."
Example: "Succeeded. Dialog opened with language options."
Example: "No visible change."
Example: "Failed. Error dialog appeared saying the file already exists."
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

BASH_OUTPUT_ABSTRACTION_PROMPT = """Summarize a bash execution result in 1-3 concise sentences for future planning.

Focus on:
- whether the command succeeded or failed
- the most important outcome or error
- any concrete next-step signal that matters

Rules:
- Be concise
- Do not repeat the full output
- Prefer key files/results/errors over incidental logs
- If output is long, compress it to the essential result only

Example: "Succeeded. Listed the target directory and confirmed report.csv exists."
Example: "Failed. Python raised ModuleNotFoundError for openpyxl."
Example: "Succeeded. Script updated the spreadsheet and printed 12 matching rows."
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
  {{"type": "success", "lesson": "Method X worked: ..."}},
  {{"type": "failure", "lesson": "DON'T use method Y: tried 3 times, doesn't work"}}
]

Type values (ONLY these two):
- "success": A method/strategy that clearly worked during execution
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
4. **Prioritize** - Focus on: mandatory requirements first, then critical pitfalls, then helpful strategies
5. **Conflict Resolution** - If success/fail lessons conflict with required lessons, prioritize and follow the required lessons.

Return empty string if no relevant lessons exist."""

# ==================== PATTERN MANAGER ====================

class PatternManager:
    """Manage task execution pattern by domain."""

    def __init__(
        self,
        llm: Optional[AbstractLLM] = None,
        qdrant_path: str = "./qdrant_storage",
        embedding_service_url: str = "http://localhost:8888",
        similarity_threshold: float = 0.7,
        use_qdrant_server: bool = True,  # Default to server mode for multi-process
        qdrant_server_url: str = "http://localhost:6333"
    ):
        self.llm = llm
        self.similarity_threshold = similarity_threshold
        self.logger = logging.getLogger("desktopenv.pattern")
        if not os.path.exists(qdrant_path):
            for json_file in glob.glob("mm_agents/hisa/patterns/*.json"):
                collection_name = os.path.basename(json_file).split(".")[0]
                add_lessons_to_existing(
                    json_file=json_file,
                    collection_name=collection_name,
                    use_server=False,
                    path=qdrant_path
                )
        self.qdrant = QdrantManager(
            path=qdrant_path,
            use_server=use_qdrant_server,
            server_url=qdrant_server_url
        )
        self.embedding_client = EmbeddingClient(service_url=embedding_service_url)
        mode = "server" if use_qdrant_server else "local"
        self.logger.info(f"Vector database ({mode} mode) and embedding service initialized")

    def _ensure_collection(self, collection_name: str):
        """Ensure Qdrant collection exists for a domain."""
        try:
            collections = self.qdrant.list_collections()
            if collection_name not in collections:
                self.qdrant.create_collection(
                    collection_name=collection_name,
                    vector_size=1024,
                    distance="Cosine"
                )
                self.logger.info(f"Created Qdrant collection: {collection_name}")
        except Exception as e:
            self.logger.error(f"Failed to ensure collection {collection_name}: {e}")

    def save_pattern(self, domain: str, lessons: List[Dict]):
        """Save lessons using Qdrant vector database with deduplication.

        Args:
            domain: The domain to save lessons to
            lessons: List of lesson dicts, each with 'type' and 'lesson' fields
                    type must be: 'success' or 'failure' (determined by LLM from execution)

        Note:
            - Each lesson is vectorized and stored in Qdrant
            - Similar lessons (cosine similarity > threshold) are detected and removed
            - New lessons replace similar old ones
            - Different domains use different Qdrant collections
        """
        try:
            self._ensure_collection(domain)

            # Get current max ID from Qdrant
            try:
                count = self.qdrant.count_points(domain)
                all_points = self.qdrant.scroll_all(domain, limit=1000, with_vectors=False)
                max_id = max([p["id"] for p in all_points], default=0) if all_points else 0
                next_id = max_id + 1
            except:
                next_id = 1

            added_count = 0
            replaced_count = 0

            for lesson_obj in lessons:
                lesson_text = lesson_obj.get("lesson", "")
                lesson_type = lesson_obj.get("type", "failure")

                if not lesson_text:
                    continue

                # Generate embedding for the lesson
                try:
                    lesson_vector = self.embedding_client(lesson_text)
                except Exception as e:
                    self.logger.error(f"Failed to generate embedding: {e}")
                    continue

                # Search for similar lessons
                try:
                    similar_results = self.qdrant.search(
                        collection_name=domain,
                        query_vector=lesson_vector,
                        limit=5,
                        score_threshold=self.similarity_threshold
                    )
                except Exception as e:
                    self.logger.warning(f"Search failed: {e}, assuming no similar lessons")
                    similar_results = []

                # Filter out lessons with type="require" from deletion candidates
                # IMPORTANT: Never delete or modify lessons with type="require"
                deletable_similar = []
                for r in similar_results:
                    similar_type = r.get("payload", {}).get("type", "")
                    if similar_type != "require":
                        deletable_similar.append(r)
                    else:
                        self.logger.info(f"Skipping deletion of require-type lesson (id={r['id']}) - these are protected")

                # Delete similar old lessons (excluding require type)
                if deletable_similar:
                    deletable_ids = [r["id"] for r in deletable_similar]
                    self.logger.info(
                        f"Found {len(deletable_similar)} similar lesson(s) with similarity > {self.similarity_threshold}, "
                        f"replacing them with new lesson"
                    )
                    try:
                        self.qdrant.delete_by_ids(domain, deletable_ids)
                        replaced_count += len(deletable_ids)
                    except Exception as e:
                        self.logger.error(f"Failed to delete similar lessons: {e}")

                # Add new lesson
                try:
                    self.qdrant.insert_points(
                        collection_name=domain,
                        points=[{
                            "id": next_id,
                            "vector": lesson_vector,
                            "payload": {
                                "lesson": lesson_text,
                                "type": lesson_type,
                                "domain": domain
                            }
                        }]
                    )
                    added_count += 1
                    next_id += 1
                except Exception as e:
                    self.logger.error(f"Failed to insert lesson: {e}")

            self.logger.info(
                f"Vector DB update for domain {domain}: "
                f"added {added_count} new lesson(s), replaced {replaced_count} similar lesson(s)"
            )

        except Exception as e:
            self.logger.error(f"Failed to save pattern with vector DB: {e}")
            raise

    def pattern_induction(self, task_instruction: str, action_logs: List[Dict]) -> List[str]:
        """Use LLM to extract key lessons from task execution.

        Returns:
            List of lesson strings
        """
        if not self.llm:
            return []

        # Use step_trace when available because it preserves the full narrative.
        step_abstracts = []
        for log in action_logs:
            if "step_trace" in log:
                step_abstracts.append(log["step_trace"])
            elif "step_abstract" in log:
                step_abstracts.append(self._render_step_abstract(log))

        prompt = PATTERN_INDUCTION_PROMPT.format(
            task_instruction=task_instruction,
            step_abstracts='\n'.join(step_abstracts)
        )

        try:
            messages = [
                {"role": "system", "content": "You are an expert at analyzing task execution patterns and extracting the most critical, reusable lessons. Be highly selective - only extract truly valuable insights. CRITICAL: focus only on the execution process."},
                {"role": "user", "content": prompt}
            ]

            response = self.llm(messages, enable_thinking=True)

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
                        # Validate type is success or failure
                        if item["type"] in ["success", "failure"]:
                            validated_lessons.append(item)
                        else:
                            self.logger.warning(f"Invalid lesson type '{item['type']}', skipping")
                    else:
                        self.logger.warning(f"Invalid lesson format: {item}, skipping")
                if validated_lessons:
                    formatted_lessons = "\n".join(
                        f"  {i+1}. [{lesson['type']}] {lesson['lesson']}"
                        for i, lesson in enumerate(validated_lessons)
                    )
                    self.logger.info(
                        f"Pattern induction extracted {len(validated_lessons)} lesson(s):\n{formatted_lessons}"
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

    def get_relevant_pattern(self, domain: str, current_task: str) -> str:
        """Retrieve relevant patterns using vector similarity search.

        Returns:
            Actionable advice string based on relevant patterns
        """
        try:
            self._ensure_collection(domain)

            # Check if collection has any points
            try:
                count = self.qdrant.count_points(domain)
                if count == 0:
                    self.logger.info(f"No pattern found in collection {domain}")
                    return ""
            except Exception as e:
                self.logger.warning(f"Failed to check collection count: {e}")
                return ""

            # First, retrieve ALL lessons with type="require" (mandatory requirements)
            require_patterns = []
            try:
                all_require_results = self.qdrant.search_by_filter(
                    collection_name=domain,
                    filter_conditions={"type": "require"},
                    limit=100  # Get all require type lessons
                )
                for result in all_require_results:
                    payload = result["payload"]
                    lesson_text = payload.get("lesson", "")
                    entry = {"id": result["id"], "lesson": lesson_text, "score": result["score"]}
                    require_patterns.append(entry)
                
            except Exception as e:
                self.logger.warning(f"Failed to retrieve require type lessons: {e}")

            # Vectorize current task
            try:
                task_vector = self.embedding_client(current_task)
            except Exception as e:
                self.logger.error(f"Failed to generate task embedding: {e}")
                raise

            # Search for similar lessons (top 5, threshold 0.75 for high quality matching)
            try:
                search_results = self.qdrant.search(
                    collection_name=domain,
                    query_vector=task_vector,
                    limit=5,
                    score_threshold=0.5
                )
            except Exception as e:
                self.logger.error(f"Vector search failed: {e}")
                raise

            # Group by type (excluding require since we already have them all)
            success_patterns = []
            failure_patterns = []

            for result in search_results:
                payload = result["payload"]
                lesson_type = payload.get("type", "failure")
                lesson_text = payload.get("lesson", "")
                score = result["score"]

                entry = {"id": result["id"], "lesson": lesson_text, "score": score}

                # Skip require type here as we already retrieved all of them above
                if lesson_type == "require":
                    continue
                elif lesson_type == "success":
                    success_patterns.append(entry)
                else:
                    failure_patterns.append(entry)

            # Build summary
            pattern_summary = []
            if require_patterns:
                pattern_summary.append("\n--- REQUIREMENTS (MUST FOLLOW) ---")
                for pattern in require_patterns:
                    pattern_summary.append(f"[{pattern['id']}] {pattern['lesson']} (similarity: {pattern['score']:.2f})")

            if success_patterns:
                pattern_summary.append("\n--- SUCCESS Patterns ---")
                for pattern in success_patterns:
                    pattern_summary.append(f"[{pattern['id']}] {pattern['lesson']} (similarity: {pattern['score']:.2f})")

            if failure_patterns:
                pattern_summary.append("\n--- FAILURE Patterns ---")
                for pattern in failure_patterns:
                    pattern_summary.append(f"[{pattern['id']}] {pattern['lesson']} (similarity: {pattern['score']:.2f})")

            if not pattern_summary:
                return ""

            prompt = PATTERN_SYNTHESIS_PROMPT.format(
                current_task=current_task,
                pattern_summary='\n'.join(pattern_summary)
            )

            try:
                messages = [
                    {"role": "system", "content": "You are an expert at analyzing past lessons and providing actionable advice for new tasks."},
                    {"role": "user", "content": prompt}
                ]

                response = self.llm(messages)
                self.logger.info(f"Retrieved {len(pattern_summary)} relevant lesson(s) using vector search")
                return response.strip()

            except Exception as e:
                self.logger.error(f"Failed to summarize patterns: {e}")
                return '\n'.join(pattern_summary)

        except Exception as e:
            self.logger.error(f"Failed to get relevant patterns with vector DB: {e}")
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
        pattern_dir: str = "../qdrant/qdrant_storage",
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

        # Initialize LLM clients
        self.global_planner_llm = AbstractLLM(global_planner_model, logger=self.logger)
        self.visual_grounder_llm = AbstractLLM(visual_grounder_model, logger=self.logger)
        self.state_manager_llm = AbstractLLM(state_manager_model, logger=self.logger)

        # Initialize pattern manager
        if not self.wo_pattern:
            self.pattern_manager = PatternManager(
                llm=self.global_planner_llm,
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
        self.last_error_feedback = None  # Store error feedback for retry
        self.last_full_summary = None  # Last complete history summary
        self.last_summary_log_index = 0  # Number of action logs already folded into last_full_summary
        self.last_refinement_log_count = 0
        self.step_token_usage = {}  # Store token usage for current step
        self.current_thought = ""  # Store current step's thought for step_abstract
        self.current_proposed_subgoal = ""
        self.current_subgoal = ""
        self.last_execution_status = "continue"
        self.last_blocking_reason = ""
        self.awaiting_final_verification = False
        self.final_verification_observed = False
        self.last_tool_output = None  # Store last tool execution result for wo_step mode
        self.prompt_dump_path = ""
        self.prompt_dump_counter = 0
        self.last_dumped_system_prompt_hash = ""

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
        self.last_dumped_system_prompt_hash = ""
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

        payload_to_dump = payload
        if (
            stage == "global_planner"
            and isinstance(payload, dict)
            and isinstance(payload.get("messages"), list)
            and payload["messages"]
        ):
            messages = copy.deepcopy(payload["messages"])
            first_message = messages[0] if isinstance(messages[0], dict) else None
            system_content = first_message.get("content") if isinstance(first_message, dict) else None
            if first_message and first_message.get("role") == "system" and isinstance(system_content, str):
                prompt_hash = self._hash_text(system_content)
                if self.last_dumped_system_prompt_hash == prompt_hash:
                    first_message["content"] = "<same as previous global_planner system prompt>"
                else:
                    self.last_dumped_system_prompt_hash = prompt_hash
            payload_to_dump = {"messages": messages}

        sanitized_payload = self._sanitize_prompt_payload(payload_to_dump)
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

    def _resolve_planner_subgoal(self, value: str, tool: str) -> str:
        text = re.sub(r"\s+", " ", str(value or "").strip())
        if not text:
            raise ValueError("Subgoal cannot be empty")
        if text.lower() == "continue":
            if tool == "termination":
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

    def _parse_abstraction_summary(self, raw_text: str) -> str:
        summary = re.sub(r"\s+", " ", str(raw_text or "").strip())
        return summary or "Step abstraction failed due to error."

    def _build_subgoal_context_lines(self) -> List[str]:
        lines = []
        if self.current_subgoal:
            lines.append(f"Current subgoal: {self.current_subgoal}")
        if self.awaiting_final_verification and not self.final_verification_observed:
            lines.append(
                "Final verification is still required before termination. "
                "Use one more action to verify the exact final state."
            )
        return lines

    def _build_termination_guard_feedback(self, target_subgoal: str = "") -> str:
        target = target_subgoal or self.current_subgoal or "the requested final state"
        if not self.awaiting_final_verification:
            self.awaiting_final_verification = True
            self.final_verification_observed = False
            return (
                f"Termination blocked. You must verify the exact final state for subgoal '{target}' before terminating.\n"
                "Do one more verification-focused action, then terminate only if the result clearly confirms task completion."
            )
        return (
            "Termination blocked. Final verification has not been observed yet.\n"
            "Use one more action to inspect the final UI or output and collect concrete evidence, then terminate only if verified."
        )

    def _run_final_verification(self, target_subgoal: str = "") -> bool:
        screenshot = None
        for _ in range(3):
            screenshot = self.env.controller.get_screenshot()
            if screenshot is not None:
                break
            time.sleep(0.5)
        screenshot_b64 = base64.b64encode(screenshot).decode("utf-8") if screenshot else ""
        recent_logs = self.action_logs[-4:]
        history_lines = [self._render_step_abstract(log) for log in recent_logs if log.get("step_abstract")]
        messages = [
            {"role": "system", "content": FINAL_VERIFICATION_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Task:\n{self.task_instruction}\n\n"
                    f"Current subgoal:\n{target_subgoal or self.current_subgoal or 'None'}\n\n"
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

        self._dump_prompt_entry(stage="final_verification", payload={"messages": messages})
        response = self.state_manager_llm(messages, enable_thinking=False)
        self._dump_prompt_entry(stage="final_verification_response", payload=response)
        parsed = json.loads(repair_json(response))
        result = str(parsed.get("result", "") or "").strip().lower()
        if result not in {"pass", "fail"}:
            raise ValueError("Final verification response must contain result=pass|fail")
        return result == "pass"

    def _derive_execution_status(self, decision: Dict) -> Tuple[str, str]:
        proposed_subgoal = self._normalize_subgoal(decision.get("subgoal", ""))
        previous_subgoal = self.current_subgoal
        latest_log = self.action_logs[-1] if self.action_logs else {}
        verified = str((latest_log.get("step_abstract") or {}).get("verified", "") or "").lower()

        if decision.get("tool") == "termination":
            return "finish", ""

        if latest_log.get("execution_success") is False:
            return "blocked", "progress_stall"

        if "no visible change" in verified:
            same_subgoal_logs = []
            for log in reversed(self.action_logs):
                if log.get("subgoal") and log.get("subgoal") != proposed_subgoal:
                    break
                same_subgoal_logs.append(log)
            no_change_count = 0
            for log in same_subgoal_logs:
                log_verified = str((log.get("step_abstract") or {}).get("verified", "") or "").lower()
                if "no visible change" in log_verified:
                    no_change_count += 1
                else:
                    break
            if no_change_count >= 2:
                return "blocked", "progress_stall"

        if previous_subgoal and proposed_subgoal != previous_subgoal:
            return "done", ""

        return "continue", ""

    def _record_subgoal_transition(self, decision: Dict) -> str:
        proposed_subgoal = self._normalize_subgoal(decision.get("subgoal", ""))
        previous_subgoal = self.current_subgoal
        status = self._normalize_execution_status(decision.get("execution_status", "continue"))
        blocking_reason = self._normalize_blocking_reason(decision.get("blocking_reason", ""))
        decision["execution_status"] = status
        if blocking_reason:
            decision["blocking_reason"] = blocking_reason
        elif "blocking_reason" in decision:
            decision.pop("blocking_reason", None)

        if self.action_logs:
            self.action_logs[-1]["subgoal"] = proposed_subgoal
            self.action_logs[-1]["execution_status"] = status
            if blocking_reason:
                self.action_logs[-1]["blocking_reason"] = blocking_reason
            else:
                self.action_logs[-1].pop("blocking_reason", None)

        self.last_execution_status = status
        self.last_blocking_reason = blocking_reason
        if not previous_subgoal or status in {"done", "blocked", "continue"}:
            self.current_subgoal = proposed_subgoal
        return status

    def _maybe_refine_context(self, reason: str = "") -> None:
        total_logs = len(self.action_logs)
        if self.wo_refinement or total_logs <= 0:
            return

        logs_since_last_refinement = total_logs - int(getattr(self, "last_refinement_log_count", 0) or 0)
        if logs_since_last_refinement <= 3:
            return

        if reason not in {"done", "blocked"}:
            return

        if self.last_full_summary:
            logs_to_summarize = self.action_logs[self.last_summary_log_index:]
            start_step = self.action_logs[0]["step"]
            end_step = self.action_logs[-1]["step"]
            summary = self._context_refinement(
                logs_to_summarize,
                start_step,
                end_step,
                previous_summary=self.last_full_summary
            )
        else:
            logs_to_summarize = self.action_logs
            start_step = logs_to_summarize[0]["step"]
            end_step = logs_to_summarize[-1]["step"]
            summary = self._context_refinement(logs_to_summarize, start_step, end_step)

        self.last_full_summary = summary
        self.last_summary_log_index = total_logs
        self.last_refinement_log_count = total_logs
        self.logger.info(f"[refinement:{reason or 'periodic'}] {summary}")

        if self.wo_step:
            self.conversation_messages = []
            self.last_tool_output = None

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

    def _build_step_abstract(
        self,
        step: int,
        tool_type: str,
        success: bool,
        detail: str,
        verification: str = "",
    ) -> Dict:
        detail = re.sub(r"\s+", " ", str(detail or "").strip())
        verification = re.sub(r"\s+", " ", str(verification or "").strip())
        return {
            "detail": detail,
            "verified": verification,
        }

    def _render_step_abstract(self, log: Dict) -> str:
        step_abstract = log.get("step_abstract") or {}
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
        detail = log.get("detail") or step_abstract.get("detail") or ""
        if detail:
            parts.append(f"detail={detail}")
        if step_abstract.get("verified"):
            parts.append(f"verified={step_abstract['verified']}")
        return " | ".join(parts)

    def _get_decision_action_fingerprint(self, decision: Dict) -> str:
        """Compute action fingerprint from current decision before execution."""
        tool = decision.get("tool", "")
        tool_input = decision.get("input", "")

        if tool == "bash_execution":
            return self._hash_text(self._normalize_bash_command(tool_input))
        if tool == "gui_action":
            return self._hash_text(tool_input)
        return ""

    def _normalize_gui_description(self, description: str) -> str:
        """Normalize gui_action description for repeat detection."""
        if not description:
            return ""
        return re.sub(r"\s+", " ", str(description).strip()).lower()

    def _detect_repeated_gui_description(self, decision: Dict) -> Optional[str]:
        """Fail fast if the same gui_action description is planned 3 consecutive times."""
        if decision.get("tool") != "gui_action":
            return None

        normalized_description = self._normalize_gui_description(decision.get("description", ""))
        if not normalized_description:
            return None

        repeat_count = 1  # Count current candidate decision.
        for log in reversed(self.action_logs):
            if log.get("type") != "gui_action":
                break
            if self._normalize_gui_description(log.get("description", "")) != normalized_description:
                break
            repeat_count += 1

        if repeat_count >= 3:
            return (
                "Detected repeated gui_action description loop: "
                f"'{decision.get('description', '')}' repeated {repeat_count} consecutive times."
            )
        return None

    def _detect_execution_loop(self, decision: Dict) -> Optional[str]:
        """Detect strict loops with identical action/result fingerprints."""
        tool = decision.get("tool", "")
        if tool not in ["gui_action", "bash_execution"] or not self.action_logs:
            return None

        threshold = 5 if tool == "gui_action" else 3
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
                if log.get("step_abstract"):
                    history_lines.append(self._render_step_abstract(log))
                elif "step_trace" in log:
                    history_lines.append(log["step_trace"])

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
            summary_with_context_refinement = self.state_manager_llm(
                messages,
                enable_thinking=False,
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
        self.conversation_messages = []  # Store full conversation history when wo_step=True
        self.last_tool_output = None  # Store last tool execution result for wo_step mode
        self.current_subgoal = ""
        self.current_proposed_subgoal = ""
        self.last_execution_status = "continue"
        self.last_blocking_reason = ""
        self.awaiting_final_verification = False
        self.final_verification_observed = False

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

        # Load relevant pattern
        domain = task_config.get("domain", "general")
        self.current_task_id = task_config.get("id", task_config.get("task_id", ""))
        self.current_domain = domain
        self.current_task_signature = self._build_task_signature(task_instruction, domain)
        self._init_prompt_dump_file()
        past_pattern_text = ""
        if not self.wo_pattern:
            past_pattern_text = self.pattern_manager.get_relevant_pattern(
                domain, task_instruction
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

                # Check termination or infeasible
                if decision["tool"] == "termination":
                    target_subgoal = self._normalize_subgoal(decision.get("subgoal", "")) or self.current_subgoal
                    verification = self._run_final_verification(target_subgoal)
                    if not verification:
                        self.last_error_feedback = self._build_termination_guard_feedback(target_subgoal)
                        if self.wo_step:
                            self.last_tool_output = "Final verification failed."
                        self.logger.info("Termination blocked: final verification failed")
                        self.operation_count += 1
                        continue

                    step = self.operation_count + 1
                    global_planner_usage = self._calculate_usage_delta(usage_before_step, usage_after_global_planner)
                    screenshot_file = f"step_{step}.png"
                    try:
                        screenshot = self.env.controller.get_screenshot()
                        with open(os.path.join(self.operations_dir, screenshot_file), "wb") as f:
                            f.write(screenshot)
                    except Exception as e:
                        self.logger.warning(f"Failed to capture termination screenshot: {e}")
                        screenshot_file = ""

                    thought_prefix = f"Thought: {decision.get('thought', '')} | " if decision.get('thought') else ""
                    step_abstract = (
                        f"Step {step}: termination | {thought_prefix}"
                        f"Summary: {decision.get('input', 'Task completed.')} | Result: Terminated"
                    )
                    self.action_logs.append({
                        "step": step,
                        "type": "termination",
                        "execution_success": True,
                        "screenshot": screenshot_file,
                        "subgoal": target_subgoal,
                        "execution_status": "finish",
                        "detail": decision.get("input", "Task completed."),
                        "step_abstract": self._build_step_abstract(
                            step=step,
                            tool_type="termination",
                            success=True,
                            detail=decision.get("input", "Task completed."),
                            verification="Task marked complete after explicit verification."
                        ),
                        "step_trace": step_abstract,
                        "step_time": 0.0,
                        "token_usage": {
                            "global_planner": global_planner_usage["global_planner"],
                            "visual_grounder": {"cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "image_count": 0},
                            "state_manager": global_planner_usage["state_manager"],
                            "total": {
                                "cost": global_planner_usage["global_planner"]["cost"] + global_planner_usage["state_manager"]["cost"],
                                "prompt_tokens": global_planner_usage["global_planner"]["prompt_tokens"] + global_planner_usage["state_manager"]["prompt_tokens"],
                                "completion_tokens": global_planner_usage["global_planner"]["completion_tokens"] + global_planner_usage["state_manager"]["completion_tokens"],
                                "image_count": global_planner_usage["global_planner"]["image_count"] + global_planner_usage["state_manager"]["image_count"]
                            }
                        }
                    })
                    self.awaiting_final_verification = False
                    self.final_verification_observed = True
                    self.operation_count += 1
                    is_infeasible = False
                    self.logger.info("Task COMPLETED")
                    break
                elif decision["tool"] == "infeasible":
                    is_infeasible = True
                    infeasible_reason = decision.get('input', 'Task is objectively impossible to complete')
                    self.current_subgoal = decision.get("subgoal", self.current_subgoal)
                    self.logger.info(f"Task INFEASIBLE: {infeasible_reason}")
                    # Send "FAIL" action to environment so action_history ends with "FAIL"
                    # This is required for OSWorld's infeasible task evaluation
                    try:
                        self.env.step("FAIL", 0)
                    except Exception as e:
                        self.logger.warning(f"Failed to send FAIL action: {e}")
                    break

                # Pre-calculate global planner token usage and set step_token_usage before tool execution
                # This ensures _gui_action/_bash_execution can use it when creating action_log
                global_planner_usage = self._calculate_usage_delta(usage_before_step, usage_after_global_planner)

                # Initialize step_token_usage with global planner data (visual_grounder/state_manager will be updated after execution)
                self.step_token_usage = {
                    "global_planner": global_planner_usage["global_planner"],
                    "visual_grounder": {"cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "image_count": 0},
                    "state_manager": {"cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "image_count": 0},
                    "total": global_planner_usage["global_planner"].copy()
                }

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
                    self.last_execution_status = "blocked"
                    self.last_blocking_reason = "behavior_loop"
                    self.last_error_feedback = (
                        f"{loop_error}\n"
                        "Do not repeat the same action. Switch strategy immediately "
                        "(different target, different tool, or different command)."
                    )
                    if self.wo_step:
                        self.last_tool_output = f"Execution blocked: {loop_error}"
                    # Count this as a consumed step to avoid infinite planner-loop cycles.
                    self.operation_count += 1
                    continue

                # Execute tool and capture execution result text
                execution_result_text = self._execute_tool(decision)
                derived_status, blocking_reason = self._derive_execution_status(decision)
                decision["execution_status"] = derived_status
                if blocking_reason:
                    decision["blocking_reason"] = blocking_reason
                else:
                    decision.pop("blocking_reason", None)
                if blocking_reason:
                    self.logger.info(
                        "[execution_status] step=%s status=%s blocking_reason=%s subgoal=%s",
                        self.operation_count + 1,
                        derived_status,
                        blocking_reason,
                        decision.get("subgoal", ""),
                    )
                else:
                    self.logger.info(
                        "[execution_status] step=%s status=%s subgoal=%s",
                        self.operation_count + 1,
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
                if status == "blocked":
                    self.last_error_feedback = (
                        f"The last step was blocked while working on subgoal '{self.current_subgoal}'.\n"
                        "Switch strategy, tool, or target instead of repeating the same failed action."
                    )
                
                # Store execution result for wo_step mode to maintain dialogue structure
                if self.wo_step and execution_result_text:
                    self.last_tool_output = execution_result_text

                # Capture token usage after tool execution
                usage_after_tool = self._get_usage_snapshot()

                # Calculate state_manager/visual_grounder token usage and update step_token_usage
                tool_usage = self._calculate_usage_delta(usage_after_global_planner, usage_after_tool)
                total_step_usage = self._calculate_usage_delta(usage_before_step, usage_after_tool)

                # Update token usage: combine state_manager usage from history summarization and step abstraction
                self.step_token_usage = {
                    "global_planner": global_planner_usage["global_planner"],
                    "visual_grounder": tool_usage["visual_grounder"],
                    "state_manager": {
                        "cost": global_planner_usage["state_manager"]["cost"] + tool_usage["state_manager"]["cost"],
                        "prompt_tokens": global_planner_usage["state_manager"]["prompt_tokens"] + tool_usage["state_manager"]["prompt_tokens"],
                        "completion_tokens": global_planner_usage["state_manager"]["completion_tokens"] + tool_usage["state_manager"]["completion_tokens"],
                        "image_count": global_planner_usage["state_manager"]["image_count"] + tool_usage["state_manager"]["image_count"]
                    },
                    "total": {
                        "cost": total_step_usage["global_planner"]["cost"] + total_step_usage["visual_grounder"]["cost"] + total_step_usage["state_manager"]["cost"],
                        "prompt_tokens": total_step_usage["global_planner"]["prompt_tokens"] + total_step_usage["visual_grounder"]["prompt_tokens"] + total_step_usage["state_manager"]["prompt_tokens"],
                        "completion_tokens": total_step_usage["global_planner"]["completion_tokens"] + total_step_usage["visual_grounder"]["completion_tokens"] + total_step_usage["state_manager"]["completion_tokens"],
                        "image_count": total_step_usage["global_planner"]["image_count"] + total_step_usage["visual_grounder"]["image_count"] + total_step_usage["state_manager"]["image_count"]
                    }
                }

                # Update the action_log entry that was already added with complete token usage
                if self.action_logs and self.action_logs[-1]["step"] == self.operation_count + 1:
                    self.action_logs[-1]["token_usage"] = self.step_token_usage

                self.operation_count += 1

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
                screenshot = None
                for screenshot_attempt in range(3):
                    screenshot = self.env.controller.get_screenshot()
                    if screenshot is not None:
                        break
                    self.logger.warning(
                        f"Screenshot unavailable for planning (retry {screenshot_attempt + 1}/3), waiting 2s..."
                    )
                    time.sleep(2)
                if screenshot is None:
                    raise RuntimeError("Failed to capture screenshot for planning after retries.")
                screenshot_b64 = base64.b64encode(screenshot).decode("utf-8")

                self._maybe_refine_context()

                # ========== Build Messages ==========
                if self.wo_step:
                    # Use full conversation history approach
                    messages = [
                        {"role": "system", "content": GLOBAL_PLANNER_PROMPT},
                    ]

                    # ========== Sliding Window Logic (for wo_step mode) ==========
                    # If context refinement is disabled, apply sliding window
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
                    
                    # Add task / summary context when starting a fresh conversation window
                    if len(conversation_to_append) == 0:
                        messages.append({"role": "user", "content": f"Task: {self.task_instruction}"})
                        if self.past_pattern_text:
                            messages.append({
                                "role": "user",
                                "content": f"Relevant past patterns:\n{self.past_pattern_text}"
                            })
                        if not self.wo_refinement and self.last_full_summary:
                            messages.append({
                                "role": "user",
                                "content": f"Summary of previous steps:\n{self.last_full_summary}"
                            })
                    for line in self._build_subgoal_context_lines():
                        messages.append({"role": "user", "content": line})

                    messages.extend(conversation_to_append)
                    
                    # Add observation from previous action to maintain dialogue structure
                    if self.last_tool_output:
                        messages.append({
                            "role": "user",
                            "content": f"Observation from previous action:\n{self.last_tool_output}"
                        })
                        self.last_tool_output = None  # Clear after use to prevent duplicate appending

                    # Add error feedback or standard prompt
                    if self.last_error_feedback:
                        messages.append({
                            "role": "user",
                            "content": (
                                f"<error_feedback>\n{self.last_error_feedback}\n</error_feedback>\n\n"
                                "Please fix the error and try again."
                            )
                        })
                    else:
                        prompt_text = (
                            "Based on the execution history and current screenshot, what's the next action?"
                            if len(conversation_to_append) == 0
                            else "Based on the conversation history and current screenshot, what's the next action?"
                        )
                        messages.append({"role": "user", "content": prompt_text})

                    # Add current user message with screenshot
                    current_user_message = {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Current screenshot:"},
                            {"type": "input_image", "image_url": f"data:image/png;base64,{screenshot_b64}"}
                        ]
                    }
                    messages.append(current_user_message)
                else:
                    # Original approach with step_abstract
                    
                    # ========== Sliding Window Logic (for step mode) ==========
                    # When wo_refinement=True, apply sliding window to action_logs
                    logs_to_use = self.action_logs
                    if self.wo_refinement and len(self.action_logs) > self.sliding_window_size:
                        logs_to_use = self.action_logs[-self.sliding_window_size:]
                    
                    # Build condensed_history: summary + recent step summaries
                    condensed_history = []
                    if not self.wo_refinement and self.last_full_summary:
                        # Context refinement enabled: use summary + recent logs
                        for log in self.action_logs[self.last_summary_log_index:]:
                            if log.get("step_abstract"):
                                condensed_history.append(self._render_step_abstract(log))
                            elif "step_trace" in log:
                                condensed_history.append(log["step_trace"])
                    else:
                        # wo_refinement=True or no summary yet: use logs (with sliding window applied)
                        for log in logs_to_use:
                            if log.get("step_abstract"):
                                condensed_history.append(self._render_step_abstract(log))
                            elif "step_trace" in log:
                                condensed_history.append(log["step_trace"])
                    
                    # Build messages array
                    messages = [
                        {"role": "system", "content": GLOBAL_PLANNER_PROMPT},
                        {"role": "user", "content": f"Task: {self.task_instruction}"}
                    ]

                    if self.past_pattern_text:
                        messages.append({
                            "role": "user",
                            "content": f"Relevant past patterns:\n{self.past_pattern_text}"
                        })

                    for line in self._build_subgoal_context_lines():
                        messages.append({"role": "user", "content": line})

                    if not self.wo_refinement and self.last_full_summary:
                        messages.append({
                            "role": "user",
                            "content": f"Summary of previous steps:\n{self.last_full_summary}"
                        })
                        for log in self.action_logs[self.last_summary_log_index:]:
                            if log.get("step_abstract"):
                                messages.append({
                                    "role": "user",
                                    "content": self._render_step_abstract(log)
                                })
                            elif "step_trace" in log:
                                messages.append({
                                    "role": "user",
                                    "content": log["step_trace"]
                                })
                    else:
                        for history_item in condensed_history:
                            messages.append({
                                "role": "user",
                                "content": history_item
                            })

                    if self.last_error_feedback:
                        messages.append({
                            "role": "user",
                            "content": (
                                f"<error_feedback>\n{self.last_error_feedback}\n</error_feedback>\n\n"
                                "Please fix the error and try again."
                            )
                        })
                    else:
                        messages.append({
                            "role": "user",
                            "content": """Based on the execution history and current screenshot, decide the next action. Avoid repeating failed actions. You should strictly follow the JSON format below:
```json
{
    "thought": "Brief reasoning about the current action. Check prerequisites and verify previous result.",
    "subgoal": "Meaningful phase-level objective for the current stage of work, or 'continue' to keep the current one",
    "tool": "gui_action|bash_execution|wait|termination|infeasible",
    "input": "String - tool-specific content (see examples below)",
    "description": "Optional for non-mouse actions; required for mouse-position gui_action so the executor can ground coordinates"
}
```"""
                        })

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

                # Validate decision structure
                if "subgoal" not in decision:
                    raise ValueError("Missing 'subgoal' field in decision")
                if "tool" not in decision:
                    raise ValueError("Missing 'tool' field in decision")
                if decision["tool"] not in ["gui_action", "bash_execution", "wait", "termination", "infeasible"]:
                    raise ValueError(f"Invalid tool: {decision['tool']}")
                decision["subgoal"] = self._resolve_planner_subgoal(
                    decision["subgoal"],
                    decision["tool"],
                )
                decision.pop("execution_status", None)

                try:
                    self.logger.info(f"[decision]: {json.dumps(decision, indent=4)}")
                except Exception as e:
                    self.logger.info(f"[decision]: {decision}")

                # Clear error feedback on success
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
                    error_feedback = FIX_RESPONSE_PROMPT.format(
                        operation_count=self.operation_count,
                        max_steps=self.max_steps,
                        error_message=str(e),
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
                    # Last attempt failed, return None
                    self.logger.error("All retry attempts exhausted, cannot get valid decision")
                    with open(os.path.join(self.save_dir, "err_reason.txt"), "w") as f:
                        f.write("All retry attempts exhausted, cannot get valid decision")
                    return None
        
        return None

    def _execute_tool(self, decision: Dict) -> str:
        """Execute tool based on decision and return execution result text."""
        tool = decision.get("tool", "")
        tool_input = decision.get("input", "")
        description = decision.get("description", "")

        # Store thought for step_abstract
        self.current_thought = decision.get("thought", "")
        self.current_proposed_subgoal = self._normalize_subgoal(decision.get("subgoal", ""))

        if tool == "gui_action":
            # Input is pyautogui code string, description is optional for placeholder
            return self._gui_action(tool_input, description)

        elif tool == "bash_execution":
            return self._bash_execution(tool_input)

        elif tool == "wait":
            return self._wait(tool_input)

        return ""

    def _normalize_bash_command(self, code: str) -> str:
        """Normalize bash command to enforce non-interactive sudo usage."""
        if not isinstance(code, str) or not code.strip():
            return code

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
        """Normalize planner-produced gui_action code before parsing/execution."""
        if not isinstance(code, str) or not code.strip():
            return code

        # Some planner outputs emit named coordinates like click(x=123, y=456).
        # Downstream executors expect plain positional coordinates, so strip only
        # the redundant x=/y= markers and preserve all other kwargs.
        return re.sub(r"(?<=\(|,)\s*([xy])\s*=\s*", "", code)

    def _parse_pyautogui_code(self, code: str) -> List[Dict]:
        code = self._normalize_pyautogui_code(code)
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise ValueError(f"Failed to parse gui_action code: {e}") from e

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
        """Auto-ground mouse-position gui_action code from action type and description."""
        if not isinstance(code, str) or not code.strip():
            return code

        grounded_code = code
        statements = self._parse_pyautogui_code(grounded_code)
        has_placeholders = any(
            token in grounded_code
            for token in [
                "X_COORD", "Y_COORD",
                "START_X_COORD", "START_Y_COORD", "END_X_COORD", "END_Y_COORD",
            ]
        )
        if has_placeholders:
            if not description:
                raise ValueError("Description required when using placeholders")
            return self._call_visual_grounder(description, screenshot, grounded_code)

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
                raise ValueError("Failed to find dragTo action in gui_action code")
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
                raise ValueError(f"Description is required for {matched_single_action} actions")
            grounded_point = self._call_visual_grounder(description, screenshot, "pyautogui.moveTo(X_COORD, Y_COORD)")
            point_x, point_y = self._extract_grounded_point(grounded_point)
            action_stmt = self._find_first_call(statements, matched_single_action)
            if action_stmt is None:
                raise ValueError(f"Failed to find {matched_single_action} action in gui_action code")
            self._set_call_point(action_stmt, point_x, point_y)
            return self._serialize_pyautogui_code(statements)

        if "pyautogui.scroll(" in grounded_code:
            if not description:
                raise ValueError("Description is required for scroll actions")
            grounded_point = self._call_visual_grounder(description, screenshot, "pyautogui.moveTo(X_COORD, Y_COORD)")
            x, y = self._extract_grounded_point(grounded_point)
            move_stmt = self._find_first_call(statements, "moveTo")
            scroll_stmt = self._find_first_call(statements, "scroll")
            if scroll_stmt is None:
                raise ValueError("Failed to find scroll action in gui_action code")
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
            py_cmd, reasoning = self.visual_grounder_llm.call_cua(
                target_desc,
                img,
                environment="linux",
                screen_width=self.screen_width,
                screen_height=self.screen_height,
                scale=scale
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

    def _gui_action(self, code: str, description: str = "") -> str:
        """Execute gui_action tool - pyautogui code with optional placeholder replacement."""
        code = self._normalize_pyautogui_code(code)
        requested_action_fingerprint = self._hash_text(code)
        if description:
            self.logger.info(f"[gui_action] {description}")
        else:
            self.logger.info(f"[gui_action] {code}")

        # Record step start time
        step_start_time = time.time()

        step = self.operation_count + 1

        try:
            # Get before screenshot
            before_screenshot = self.env.controller.get_screenshot()
            screenshot_file = f"step_{step}.png"
            code = self._ground_gui_code(code, description, before_screenshot)

            # Execute code
            final_code = postprocess_action(code)
            obs, *_ = self.env.step(final_code, self.sleep_after_execution)

            # Wait 10 seconds for action to take effect
            time.sleep(10)

            # Get after screenshot and evaluate
            after_screenshot = obs['screenshot']
            with open(os.path.join(self.operations_dir, screenshot_file), "wb") as f:
                f.write(after_screenshot)

            # Create description for step abstraction
            eval_desc = description if description else code
            
            if self.wo_step:
                step_abstraction_summary = ""
            else:
                step_abstraction_summary = self._step_abstraction(
                    before_screenshot, after_screenshot, eval_desc,
                    wo_roi=self.wo_roi, roi_margin=self.roi_margin
                )
                self.logger.info(f"[step_abstraction] Step {step}: Result: {step_abstraction_summary}")

            # Generate step_abstract
            thought_prefix = self.current_thought if self.current_thought else ""
            if description:
                step_abstract = (
                    f"Step {step}:\n"
                    f"GUI action.\n"
                    f"Description: {description}.\n"
                    f"Code: {final_code}."
                )
            else:
                step_abstract = (
                    f"Step {step}:\n"
                    f"GUI action.\n"
                    f"Code: {final_code}."
                )
            if thought_prefix:
                step_abstract += f"\nReasoning: {thought_prefix}"
            if step_abstraction_summary:
                step_abstract += f"\nResult: {step_abstraction_summary}"

            # Calculate step execution time
            step_time = time.time() - step_start_time
            # GUI loop detection should be robust to dynamic pixels (clock/cursor/animations),
            # so do not fingerprint raw screenshots.
            result_fingerprint = self._hash_text("success=True")

            self.action_logs.append({
                "step": step,
                "type": "gui_action",
                "description": description,
                "execution_success": True,
                "screenshot": screenshot_file,
                "detail": description or str(final_code),
                "step_abstract": self._build_step_abstract(
                    step=step,
                    tool_type="gui_action",
                    success=True,
                    detail=description or final_code,
                    verification=step_abstraction_summary
                ),
                "step_trace": step_abstract,
                "step_time": round(step_time, 2),
                "token_usage": self.step_token_usage,
                "loop_action_fingerprint": requested_action_fingerprint,
                "loop_result_fingerprint": result_fingerprint
            })

            # Return execution result text for wo_step mode
            if description:
                return f"GUI Action: {description}\nCode: {final_code}\nStatus: Success\nResult: {step_abstraction_summary}"
            else:
                return f"GUI Action Code: {final_code}\nStatus: Success\nResult: {step_abstraction_summary}"

        except Exception as e:
            self.logger.error(f"GUI action execution error: {e}")

            # Generate step_abstract for error
            thought_prefix = self.current_thought if self.current_thought else ""
            if description:
                step_abstract = (
                    f"Step {step}:\n"
                    f"GUI action failed.\n"
                    f"Description: {description}.\n"
                    f"Code: {code}."
                )
            else:
                step_abstract = (
                    f"Step {step}:\n"
                    f"GUI action failed.\n"
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
                "type": "gui_action",
                "description": description,
                "execution_success": False,
                "screenshot": screenshot_file,
                "execution_status": "blocked",
                "blocking_reason": "progress_stall",
                "detail": description or str(code),
                "step_abstract": self._build_step_abstract(
                    step=step,
                    tool_type="gui_action",
                    success=False,
                    detail=description or code,
                    verification=f"Error: {str(e)}"
                ),
                "step_trace": step_abstract,
                "step_time": round(step_time, 2),
                "token_usage": self.step_token_usage,
                "loop_action_fingerprint": requested_action_fingerprint,
                "loop_result_fingerprint": result_fingerprint
            })

            # Return execution result text for wo_step mode
            if description:
                return f"GUI Action: {description}\nCode: {code}\nStatus: Failed\nError: {str(e)}"
            else:
                return f"GUI Action Code: {code}\nStatus: Failed\nError: {str(e)}"

    def _step_abstraction(self, before_screenshot: Optional[bytes], after_screenshot: Optional[bytes],
            action_description: str, wo_roi: bool = False,
            roi_margin: int = 50, bash_context: Optional[Dict] = None) -> str:
        """Abstract step by comparing before/after screenshots or bash output.

        Args:
            before_screenshot: Screenshot before action
            after_screenshot: Screenshot after action
            action_description: Description of the action performed
            wo_roi: If True, disable ROI cropping (default: False means ROI cropping is enabled)
            roi_margin: Margin to add around ROI when cropping (default: 50)

        Returns:
            Summary string.
        """
        try:
            prompt = STEP_ABSTRACTION_PROMPT.format(
                action_description=action_description,
            )

            if bash_context is not None:
                messages = [
                    {
                        "role": "user",
                        "content": (
                            f"Command or action:\n{action_description}\n\n"
                            f"Execution status: {bash_context.get('status', 'unknown')}\n"
                            f"Exit code: {bash_context.get('exitcode', 'unknown')}\n"
                            f"Output:\n{bash_context.get('logs', '')}\n\n"
                            f"{prompt}"
                        ),
                    }
                ]
            else:
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
                            return "No change detected."
                    except Exception as roi_error:
                        self.logger.warning(f"ROI detection failed, using full screenshots: {roi_error}")

                before_buffer = io.BytesIO()
                after_buffer = io.BytesIO()
                before_img.save(before_buffer, format="PNG")
                after_img.save(after_buffer, format="PNG")

                before_b64 = base64.b64encode(before_buffer.getvalue()).decode("utf-8")
                after_b64 = base64.b64encode(after_buffer.getvalue()).decode("utf-8")

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Before screenshot:"},
                            {"type": "input_image", "image_url": f"data:image/png;base64,{before_b64}"},
                            {"type": "input_text", "text": "After screenshot:"},
                            {"type": "input_image", "image_url": f"data:image/png;base64,{after_b64}"},
                            {"type": "input_text", "text": prompt}
                        ]
                    }
                ]

            self._dump_prompt_entry(stage="step_abstraction", payload={"messages": messages})
            step_abstraction_raw = self.state_manager_llm(
                messages,
                enable_thinking=False,
            )
            self._dump_prompt_entry(stage="step_abstraction_response", payload=step_abstraction_raw)
            return self._parse_abstraction_summary(step_abstraction_raw)

        except Exception as e:
            self.logger.error(f"Failed to abstract step: {e}")
            return "Step abstraction failed due to error."

    def _bash_output_abstraction_result(self, code: str, logs: str, status: str, exitcode: int) -> str:
        """Abstract bash execution result from command output instead of screenshots."""
        try:
            summary = self._step_abstraction(
                None,
                None,
                code,
                bash_context={
                    "status": status,
                    "exitcode": exitcode,
                    "logs": logs,
                    "blocked_hint": "command_failed" if exitcode != 0 or status != "success" else "None",
                },
            )
            if summary:
                return summary
        except Exception as e:
            self.logger.error(f"Failed to abstract bash output: {e}")
        try:
            messages = [
                {"role": "system", "content": BASH_OUTPUT_ABSTRACTION_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Command:\n{code}\n\n"
                        f"Status: {status}\n"
                        f"Exit code: {exitcode}\n\n"
                        f"Output:\n{logs}"
                    ),
                },
            ]
            step_abstraction = self.state_manager_llm(
                messages,
                enable_thinking=False,
            )
            summary = re.sub(r"\s+", " ", step_abstraction.strip())
            return summary or "No output."
        except Exception:
            status_str = "Succeeded" if exitcode == 0 and status == "success" else "Failed"
            fallback_output = (logs or "").strip().replace("\n", " ")
            if len(fallback_output) > 200:
                fallback_output = fallback_output[:200] + "..."
            return f"{status_str}. {fallback_output}" if fallback_output else f"{status_str}. No output."

    def _bash_execution(self, code: str) -> str:
        """Execute bash commands or Python scripts (not pyautogui)."""
        code = self._normalize_bash_command(code)
        action_fingerprint = self._hash_text(code)
        self.logger.info(f"[bash_execution] {code}")

        # Record step start time
        step_start_time = time.time()

        step = self.operation_count + 1

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

            # Wait 10 seconds for action to take effect
            time.sleep(10)

            # Get after screenshot
            after_screenshot = self.env.controller.get_screenshot()
            screenshot_file = f"step_{step}.png"

            with open(os.path.join(self.operations_dir, screenshot_file), "wb") as f:
                f.write(after_screenshot)

            # Step abstraction for bash execution
            # Skip step abstraction if wo_step is True
            if self.wo_step:
                step_abstraction_summary = ""
            else:
                step_abstraction_summary = self._bash_output_abstraction_result(
                    code=code,
                    logs=logs,
                    status=status,
                    exitcode=exitcode,
                )
                self.logger.info(f"[step_abstraction] Step {step}: Result: {step_abstraction_summary}")

            # Generate step_abstract summary
            thought_prefix = self.current_thought if self.current_thought else ""
            step_abstract = (
                f"Step {step}:\n"
                f"Bash command.\n"
                f"Code: {code}."
            )
            if thought_prefix:
                step_abstract += f"\nReasoning: {thought_prefix}"
            if step_abstraction_summary:
                step_abstract += f"\nResult: {step_abstraction_summary}"
            result_fingerprint = self._hash_text(
                f"status={status}|exitcode={exitcode}|output={logs}"
            )

            # Calculate step execution time
            step_time = time.time() - step_start_time

            self.action_logs.append({
                "step": step,
                "type": "bash_execution",
                "execution_success": exitcode == 0 and status == "success",
                "screenshot": screenshot_file,
                "detail": code,
                "step_abstract": self._build_step_abstract(
                    step=step,
                    tool_type="bash_execution",
                    success=exitcode == 0 and status == "success",
                    detail=code,
                    verification=step_abstraction_summary
                ),
                "step_trace": step_abstract,
                "step_time": round(step_time, 2),
                "token_usage": self.step_token_usage,
                "loop_action_fingerprint": action_fingerprint,
                "loop_result_fingerprint": result_fingerprint
            })

            # Return execution result text for wo_step mode
            status_str = "Success" if (exitcode == 0 and status == "success") else "Failed"
            return f"Bash Command: {code}\nStatus: {status_str}\nOutput:\n{logs}"
            
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
                "execution_status": "blocked",
                "blocking_reason": "progress_stall",
                "detail": code,
                "step_abstract": self._build_step_abstract(
                    step=step,
                    tool_type="bash_execution",
                    success=False,
                    detail=code,
                    verification=f"Error: {str(e)}"
                ),
                "step_trace": step_abstract,
                "token_usage": self.step_token_usage,
                "loop_action_fingerprint": action_fingerprint,
                "loop_result_fingerprint": result_fingerprint
            })

            # Return execution result text for wo_step mode
            return f"Bash Command: {code}\nStatus: Failed\nError: {str(e)}"

    def _wait(self, seconds_str: str) -> str:
        """Wait for specified seconds and observe UI changes."""
        try:
            wait_seconds = float(seconds_str)
            # Limit wait time to reasonable range
            wait_seconds = max(5, min(wait_seconds, 60))
        except:
            self.logger.warning(f"Invalid wait time '{seconds_str}', using default 15 seconds")
            wait_seconds = 15

        self.logger.info(f"[wait] Waiting for {wait_seconds} seconds...")

        # Record step start time
        step_start_time = time.time()

        step = self.operation_count + 1

        try:
            # Get before screenshot
            before_screenshot = self.env.controller.get_screenshot()
            screenshot_file = f"step_{step}.png"

            # Wait
            time.sleep(wait_seconds)

            # Get after screenshot
            after_screenshot = self.env.controller.get_screenshot()
            with open(os.path.join(self.operations_dir, screenshot_file), "wb") as f:
                f.write(after_screenshot)

            # Step abstraction for wait
            # Skip step abstraction if wo_step is True
            if self.wo_step:
                step_abstraction_summary = ""
            else:
                step_abstraction_summary = self._step_abstraction(
                    before_screenshot, after_screenshot,
                    f"Waited {wait_seconds} seconds to observe UI changes",
                    wo_roi=self.wo_roi, roi_margin=self.roi_margin
                )
                self.logger.info(f"[step_abstraction] Step {step}: Result: {step_abstraction_summary}")

            # Generate step_abstract
            thought_prefix = self.current_thought if self.current_thought else ""
            step_abstract = (
                f"Step {step}:\n"
                f"Wait.\n"
                f"Duration: {wait_seconds} seconds."
            )
            if thought_prefix:
                step_abstract += f"\nReasoning: {thought_prefix}"
            if step_abstraction_summary:
                step_abstract += f"\nResult: {step_abstraction_summary}"

            # Calculate step execution time
            step_time = time.time() - step_start_time

            self.action_logs.append({
                "step": step,
                "type": "wait",
                "execution_success": True,
                "screenshot": screenshot_file,
                "detail": f"wait {wait_seconds}s",
                "step_abstract": self._build_step_abstract(
                    step=step,
                    tool_type="wait",
                    success=True,
                    detail=f"wait {wait_seconds}s",
                    verification=step_abstraction_summary
                ),
                "step_trace": step_abstract,
                "step_time": round(step_time, 2),
                "token_usage": self.step_token_usage
            })

            # Return execution result text for wo_step mode
            return f"Wait: {wait_seconds}s\nStatus: Success\nResult: {step_abstraction_summary}"

        except Exception as e:
            self.logger.error(f"Wait execution error: {e}")

            # Generate step_abstract for error
            thought_prefix = self.current_thought if self.current_thought else ""
            step_abstract = (
                f"Step {step}:\n"
                f"Wait failed.\n"
                f"Duration: {wait_seconds} seconds."
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
                "execution_status": "blocked",
                "blocking_reason": "progress_stall",
                "detail": f"wait {wait_seconds}s",
                "step_abstract": self._build_step_abstract(
                    step=step,
                    tool_type="wait",
                    success=False,
                    detail=f"wait {wait_seconds}s",
                    verification=f"Error: {str(e)}"
                ),
                "step_trace": step_abstract,
                "step_time": round(step_time, 2),
                "token_usage": self.step_token_usage
            })

            # Return execution result text for wo_step mode
            return f"Wait: {wait_seconds}s\nStatus: Failed\nError: {str(e)}"


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
                self.pattern_manager.save_pattern(domain, key_lessons)
                # Format lessons as numbered list for logging
                formatted_lessons = "\n".join(f"  {i+1}. [{lesson['type']}] {lesson['lesson']}" for i, lesson in enumerate(key_lessons))
                self.logger.info(f"Saved {len(key_lessons)} lesson(s):\n{formatted_lessons}")
            else:
                self.logger.info("No significant lessons to save")

        # Now evaluate score
        try:
            # self.logger.info("Closing temporary windows...")
            # self.env.step("pyautogui.press('esc')", 0.5)

            # Wait for VM HTTP service to stabilize after task execution
            self.logger.info("Waiting for VM to stabilize before evaluation...")
            time.sleep(10)
            
            # Retry evaluation with exponential backoff to handle transient VM service issues
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    score = self.env.evaluate()
                    break
                except Exception as eval_error:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 5  # 5s, 10s, 15s
                        self.logger.warning(f"Evaluation attempt {attempt + 1} failed: {eval_error}. Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        raise
        except Exception as e:
            self.logger.error(f"Evaluation failed after {max_retries} attempts: {e}")
            score = 0.0

        gui_steps = len([log for log in self.action_logs if log["type"] == "gui_action"])
        bash_steps = len([log for log in self.action_logs if log["type"] == "bash_execution"])
        wait_steps = len([log for log in self.action_logs if log["type"] == "wait"])

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
