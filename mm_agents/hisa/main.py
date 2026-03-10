#!/usr/bin/env python3
import base64
import json
import os
import logging
import traceback
import re
import hashlib
from typing import Optional, Dict, List
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
2. **CRITICAL: Use as LEAST steps as possible to complete the task**
3. **CRITICAL: When all required steps are done, termination IMMEDIATELY**
4. **CRITICAL: ALWAYS review <execution_history> before deciding next action:**
   - Check what actions have been done and their results
   - Avoid repeating the same action more than 3 times
   - Count completed steps to judge task completion
5. You receive: screenshot, execution history, past patterns
6. Never modify user requirements (file names, paths, etc.)
7. Each action gets automatic evaluation - you don't need separate verification steps
8. **You can read text directly from screenshots** - no need for GUI copy/paste operations. When you read text, record it in your `thought` field so it appears in execution history

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
- For drag actions, `description` MUST include both targets using this format:
  `START: <start target description>; END: <end target description>`

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

**When to use**: After triggering async operations (Submit/Apply/Run buttons, page loads, etc.), use wait to confirm completion before termination.

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
**Judge task completion by counting required steps, NOT by evaluation results:**
  - Track which steps the task requires and which are done
  - When all required steps are executed, termination IMMEDIATELY
  - **Exception**: If unsure whether the async operation finished, use the wait tool first, then termination
  - Ignore "Failed" evaluations if all required steps are done
  - **DO NOT** add verification steps unless the task explicitly asks

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
    "tool": "gui_action|bash_execution|wait|termination|infeasible",
    "input": "String - tool-specific content (see examples below)",
    "description": "Optional for non-mouse actions; required for mouse-position gui_action so the executor can ground coordinates"
}
```

Examples:
- gui_action with grounding: `{"tool": "gui_action", "input": "pyautogui.click(0, 0)", "description": "Click the Submit button"}`
- gui_action without grounding: `{"tool": "gui_action", "input": "pyautogui.write('hello')"}`
- wait: `{"tool": "wait", "input": "15"}`
- bash_execution: `{"tool": "bash_execution", "input": "ls -la"}`
- termination: `{"tool": "termination", "input": "Task completed. [summary]"}`
- infeasible: `{"tool": "infeasible", "input": "Chrome doesn't support changing search results per page - this is a search engine setting, not a browser feature"}`

## Termination (Task Complete)
When **all required actions are done and succeeded**:
```json
{
    "thought": "All task requirements completed successfully.",
    "tool": "termination",
    "input": "Task completed. [brief summary of what was done]"
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
    "tool": "gui_action|bash_execution|wait|termination|infeasible",
    "input": "tool input here"
}}
```"""

STEP_ABSTRACTION_PROMPT = """Compare before/after screenshots and describe the UI response in 1-2 sentences:

Action: {action_description}

Be concise:
- Loading/waiting states = action triggered successfully
- Check cursor position for confirmation
- Only report what changed

Example: "Succeeded. Button clicked, loading state appeared."
Example: "Succeeded. Cursor at target, no immediate change."
Example: "Failed. Error dialog: [text]."
"""

CONTEXT_REFINEMENT_PROMPT = """Analyze task execution progress and provide guidance.

Task instruction: {task_instruction}

Execution history (Steps {start_step}~{end_step}):
{history_text}

Instructions:
- If history contains <previous_summary>, combine it with <new_steps> to create a comprehensive summary
- If no <previous_summary>, directly summarize the provided steps
- List what was done in order (successes and failures)
- **IMPORTANT**: Preserve coordinates in click actions (e.g., "click(500,300)") - these can be reused later
- Identify if we're stuck in loops, making progress, or blocked
- Provide actionable suggestions for the next step if there are issues

Return a concise summary string in this format:
"Steps {start_step}~{end_step}: [ordered list of what was done, keeping coordinates]. Suggestion: [actionable advice, or 'Continue' if progressing well]"

Examples:
- "Steps 1~5: Opened file, tried to edit (failed 3 times with permission error), attempted sudo (failed). Suggestion: Try a different approach - copy file to temp location first."
- "Steps 1~5: Clicked Submit button at click(850,620), typed text, clicked Save at click(920,580). Suggestion: Continue - forms being filled correctly."
- "Steps 1~10: Previously installed package and ran script (steps 1~5). Then verified output, tested functionality (steps 6~10). Suggestion: Continue - good progress."
- "Steps 1~15: Clicked the same button 5 times with no response, tried alternative buttons (failed). Suggestion: This approach isn't working - try an alternative method or termination as infeasible."
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
        qdrant_path: str = "D:/projects/qdrant/qdrant_storage",
        embedding_service_url: str = "http://localhost:8000",
        similarity_threshold: float = 0.7,
        use_qdrant_server: bool = False,  # Default to server mode for multi-process
        qdrant_server_url: str = "http://localhost:6333"
    ):
        self.llm = llm
        self.similarity_threshold = similarity_threshold
        self.logger = logging.getLogger("desktopenv.pattern")
        if not os.path.exists(qdrant_path):
            for json_file in glob.glob("../GUIAgent/patterns/*.json"):
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

        # Use step_abstract directly (already contains step, action, result)
        step_abstracts = []
        for log in action_logs:
            if "step_abstract" in log:
                step_abstracts.append(log["step_abstract"])

        prompt = PATTERN_INDUCTION_PROMPT.format(
            task_instruction=task_instruction,
            step_abstracts='\n'.join(step_abstracts)
        )

        try:
            messages = [
                {"role": "system", "content": "You are an expert at analyzing task execution patterns and extracting the most critical, reusable lessons. Be highly selective - only extract truly valuable insights. CRITICAL: focus only on the execution process."},
                {"role": "user", "content": prompt}
            ]

            response = self.llm(messages)

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
        pattern_dir: str = "D:/projects/qdrant/qdrant_storage",
        use_qdrant_server: bool = False,  # Use server mode by default for multi-process
        qdrant_server_url: str = "http://localhost:6333",
        wo_roi: bool = False,  # If True, disable ROI cropping (default: False means ROI cropping is enabled)
        roi_margin: int = 50,  # Margin around ROI when cropping
        refine_period: int = 5,
        bash_timeout: int = 60,  # Timeout for bash script execution in seconds
        bash_working_dir: str = "~",  # Working directory for bash execution
        wo_step: bool = False,  # If True, skip step abstraction and use full conversation history
        wo_refinement: bool = False,  # If True, disable context refinement and use sliding window
        sliding_window_size: int = 5  # Sliding window size (number of conversation turns to keep)
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
        self.last_summary_step = 0  # Step number of last summary
        self.step_token_usage = {}  # Store token usage for current step
        self.current_thought = ""  # Store current step's thought for step_abstract
        self.last_tool_output = None  # Store last tool execution result for wo_step mode

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
        tool = decision.get("tool", "")
        tool_input = decision.get("input", "")

        if tool == "bash_execution":
            return self._hash_text(self._normalize_bash_command(tool_input))
        if tool == "gui_action":
            return self._hash_text(tool_input)
        return ""

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

    def _summarize_history_segment(self, logs: List[Dict], start_step: int, end_step: int, previous_summary: str = "") -> str:
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
                if "step_abstract" in log:
                    history_lines.append(log["step_abstract"])

        # Build complete history text
        if previous_summary:
            # Include previous summary + new logs
            if history_lines:
                history_text = f"<previous_summary>\n{previous_summary}\n</previous_summary>\n\n<new_steps>\n" + "\n".join(history_lines) + "\n</new_steps>"
            else:
                # Only previous summary, no new steps
                history_text = f"<previous_summary>\n{previous_summary}\n</previous_summary>"
        else:
            # First time, only new logs
            history_text = "\n".join(history_lines) if history_lines else ""

        if not history_text.strip():
            return f"Steps {start_step}~{end_step}: No detailed records. Suggestion: Continue"

        # Use LLM to summarize with context refinement
        prompt = CONTEXT_REFINEMENT_PROMPT.format(
            task_instruction=self.task_instruction,
            start_step=start_step,
            end_step=end_step,
            history_text=history_text
        )

        try:
            messages = [
                {"role": "user", "content": prompt}
            ]
            summary_with_context_refinement = self.state_manager_llm(messages)
            return summary_with_context_refinement.strip()
        except Exception as e:
            self.logger.error(f"Failed to summarize history segment with context refinement: {e}")
            # Fallback: combine previous summary with brief new summary
            if previous_summary:
                brief_new = f"Steps {start_step}~{end_step}: {len(logs)} actions" if logs else "no new actions"
                return f"{previous_summary} + {brief_new}. Suggestion: Continue"
            else:
                brief_summary = f"{len(logs)} actions executed"
                return f"Steps {start_step}~{end_step}: {brief_summary}. Suggestion: Continue"

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
        self.last_summary_step = 0
        self.conversation_messages = []  # Store full conversation history when wo_step=True
        self.last_tool_output = None  # Store last tool execution result for wo_step mode

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
                        "step_abstract": step_abstract,
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
                    self.operation_count += 1
                    is_infeasible = False
                    self.logger.info("Task COMPLETED")
                    break
                elif decision["tool"] == "infeasible":
                    is_infeasible = True
                    infeasible_reason = decision.get('input', 'Task is objectively impossible to complete')
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

                loop_error = self._detect_execution_loop(decision)
                if loop_error:
                    self.logger.warning(loop_error)
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

                # Context Refinement
                total_logs = len(self.action_logs)
                
                # Only trigger context refinement if not disabled (wo_refinement=False)
                if not self.wo_refinement and total_logs > 0 and total_logs % self.refine_period == 0:
                    # Trigger context refinement
                    if self.last_full_summary:
                        # Not first time: use previous summary + new logs since last summary
                        logs_to_summarize = self.action_logs[self.last_summary_step:]
                        start_step = self.action_logs[0]["step"]
                        end_step = self.action_logs[-1]["step"]
                        summary = self._summarize_history_segment(
                            logs_to_summarize, start_step, end_step,
                            previous_summary=self.last_full_summary
                        )
                    else:
                        # First time: summarize all logs without previous summary
                        logs_to_summarize = self.action_logs
                        start_step = logs_to_summarize[0]["step"]
                        end_step = logs_to_summarize[-1]["step"]
                        summary = self._summarize_history_segment(logs_to_summarize, start_step, end_step)

                    self.last_full_summary = summary
                    self.last_summary_step = total_logs
                    
                    # Clear conversation messages and last tool output after context refinement
                    if self.wo_step:
                        self.conversation_messages = []
                        self.last_tool_output = None  # Clear observation as it's now in summary

                # ========== Build Messages ==========
                if self.wo_step:
                    # Use full conversation history approach
                    messages = [
                        {"role": "system", "content": GLOBAL_PLANNER_PROMPT},
                    ]
                    
                    # Build current query text
                    current_query_parts = []
                    
                    # Add observation from previous action to maintain dialogue structure
                    if self.last_tool_output:
                        current_query_parts.append(f"Observation from previous action:\n{self.last_tool_output}\n")
                        self.last_tool_output = None  # Clear after use to prevent duplicate appending
                    
                    # Task instruction + pattern + summary (only after context refinement)
                    if len(self.conversation_messages) == 0:
                        current_query_parts.append(f"Task: {self.task_instruction}\n")
                        if self.past_pattern_text:
                            current_query_parts.append(f"\n<past_pattern>\n{self.past_pattern_text}\n</past_pattern>\n")
                        
                        # Add refined context if available (only when context refinement is enabled)
                        if not self.wo_refinement and self.last_full_summary:
                            current_query_parts.append(f"\n<execution_history_summary>\n{self.last_full_summary}\n</execution_history_summary>\n")
                        
                    # Add error feedback or standard prompt
                    if self.last_error_feedback:
                        current_query_parts.append(f"\n<error_feedback>\n{self.last_error_feedback}\n</error_feedback>\n\nPlease fix the error and try again. Current screenshot:")
                    else:
                        if len(self.conversation_messages) == 0:
                            current_query_parts.append("\nBased on the execution history and current screenshot, what's the next action?")
                        else:
                            current_query_parts.append("\nBased on the conversation history and current screenshot, what's the next action?")
                    
                    # ========== Sliding Window Logic (for wo_step mode) ==========
                    # If context refinement is disabled, apply sliding window
                    conversation_to_append = self.conversation_messages
                    max_messages = self.sliding_window_size * 2
                    if self.wo_refinement and len(self.conversation_messages) > max_messages:
                        conversation_to_append = self.conversation_messages[-max_messages:]
                        conversation_to_append[0]["content"][0]["text"] = f'Task: {self.task_instruction}\n\n{conversation_to_append[0]["content"][0]["text"]}'
                    
                    messages.extend(conversation_to_append)
                    
                    # Add current user message with screenshot
                    current_user_message = {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "".join(current_query_parts)},
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
                        condensed_history = [self.last_full_summary]
                        for log in self.action_logs[self.last_summary_step:]:
                            if "step_abstract" in log:
                                condensed_history.append(log["step_abstract"])
                    else:
                        # wo_refinement=True or no summary yet: use logs (with sliding window applied)
                        for log in logs_to_use:
                            if "step_abstract" in log:
                                condensed_history.append(log["step_abstract"])
                    
                    user_message_parts = []
                    user_message_parts.append(f"Task: {self.task_instruction}\n")

                    if self.past_pattern_text:
                        user_message_parts.append(f"\n<past_pattern>\n{self.past_pattern_text}\n</past_pattern>\n")

                    # Add condensed history
                    if condensed_history:
                        user_message_parts.append(f"\n<execution_history>\n" + "\n".join(condensed_history) + "\n</execution_history>\n")

                    # Add error feedback if this is a retry
                    if self.last_error_feedback:
                        user_message_parts.append(f"\n<error_feedback>\n{self.last_error_feedback}\n</error_feedback>\n")
                        user_message_parts.append("\nPlease fix the error and try again.")
                    else:
                        user_message_parts.append("""
Based on the execution_history and current screenshot, decide the next action. Avoid repeating failed actions. You should strictly follow the JSON format below: 
```json
{
    "thought": "Brief reasoning about the current action. Check prerequisites and verify previous result.",
    "tool": "gui_action|bash_execution|wait|termination|infeasible",
    "input": "String - tool-specific content (see examples below)",
    "description": "Optional for non-mouse actions; required for mouse-position gui_action so the executor can ground coordinates"
}
```""")

                    user_message_text = "".join(user_message_parts)

                    # Build messages array
                    messages = [
                        {"role": "system", "content": GLOBAL_PLANNER_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": user_message_text},
                                {"type": "input_image", "image_url": f"data:image/png;base64,{screenshot_b64}"}
                            ]
                        }
                    ]

                if attempt > 0:
                    self.logger.warning(f"Retry attempt {attempt}/{self.max_parse_retries}")

                # Call global planner
                response = self.global_planner_llm(messages)
                
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
                if "tool" not in decision:
                    raise ValueError("Missing 'tool' field in decision")
                if decision["tool"] not in ["gui_action", "bash_execution", "wait", "termination", "infeasible"]:
                    raise ValueError(f"Invalid tool: {decision['tool']}")

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
                
                # If not last attempt, set error feedback for retry
                if attempt < self.max_parse_retries - 1:
                    error_feedback = FIX_RESPONSE_PROMPT.format(
                        operation_count=self.operation_count,
                        max_steps=self.max_steps,
                        error_message=str(e),
                        response=response
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

    def _replace_first_point_action(self, code: str, action_name: str, x: int, y: int) -> str:
        pattern = rf"(pyautogui\.{re.escape(action_name)}\()\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?"
        replacement = rf"\g<1>{x}, {y}"
        new_code, count = re.subn(pattern, replacement, code, count=1)
        if count == 0:
            raise ValueError(f"Failed to inject grounded coordinates for {action_name}")
        return new_code

    def _replace_first_drag_target(self, code: str, x: int, y: int) -> str:
        pattern = r"(pyautogui\.dragTo\()\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?"
        replacement = rf"\g<1>{x}, {y}"
        new_code, count = re.subn(pattern, replacement, code, count=1)
        if count == 0:
            raise ValueError("Failed to inject grounded drag target coordinates")
        return new_code

    def _ground_gui_code(self, code: str, description: str, screenshot: bytes) -> str:
        """Auto-ground mouse-position gui_action code from action type and description."""
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
                raise ValueError("Description required when using placeholders")
            return self._call_visual_grounder(description, screenshot, grounded_code)

        if "pyautogui.dragTo(" in grounded_code:
            if not description:
                raise ValueError("Description is required for drag actions")
            start_desc, end_desc = self._parse_drag_descriptions(description)
            start_cmd = self._call_visual_grounder(start_desc, screenshot, "pyautogui.moveTo(X_COORD, Y_COORD)")
            end_cmd = self._call_visual_grounder(end_desc, screenshot, "pyautogui.moveTo(X_COORD, Y_COORD)")
            start_match = re.search(r"pyautogui\.moveTo\((\d+), (\d+)\)", start_cmd)
            end_match = re.search(r"pyautogui\.moveTo\((\d+), (\d+)\)", end_cmd)
            if not start_match or not end_match:
                raise ValueError("Failed to ground drag start/end coordinates")
            grounded_code = self._replace_first_point_action(
                grounded_code, "moveTo", int(start_match.group(1)), int(start_match.group(2))
            )
            grounded_code = self._replace_first_drag_target(
                grounded_code, int(end_match.group(1)), int(end_match.group(2))
            )
            return grounded_code

        single_point_actions = ["click", "doubleClick", "rightClick", "moveTo"]
        matched_single_action = next(
            (name for name in single_point_actions if f"pyautogui.{name}(" in grounded_code),
            None
        )
        if matched_single_action:
            if not description:
                raise ValueError(f"Description is required for {matched_single_action} actions")
            grounded_point = self._call_visual_grounder(description, screenshot, "pyautogui.moveTo(X_COORD, Y_COORD)")
            point_match = re.search(r"pyautogui\.moveTo\((\d+), (\d+)\)", grounded_point)
            if not point_match:
                raise ValueError(f"Failed to ground coordinates for {matched_single_action}")
            grounded_code = self._replace_first_point_action(
                grounded_code, matched_single_action, int(point_match.group(1)), int(point_match.group(2))
            )
            return grounded_code

        if "pyautogui.scroll(" in grounded_code:
            if not description:
                raise ValueError("Description is required for scroll actions")
            grounded_point = self._call_visual_grounder(description, screenshot, "pyautogui.moveTo(X_COORD, Y_COORD)")
            point_match = re.search(r"pyautogui\.moveTo\((\d+), (\d+)\)", grounded_point)
            if not point_match:
                raise ValueError("Failed to ground coordinates for scroll action")
            x, y = int(point_match.group(1)), int(point_match.group(2))
            if "pyautogui.moveTo(" in grounded_code:
                grounded_code = self._replace_first_point_action(grounded_code, "moveTo", x, y)
            elif re.search(r"pyautogui\.scroll\([^)]*x\s*=\s*-?\d+(?:\.\d+)?\s*,\s*y\s*=\s*-?\d+(?:\.\d+)?", grounded_code):
                grounded_code = re.sub(
                    r"(pyautogui\.scroll\([^)]*x\s*=\s*)-?\d+(?:\.\d+)?(\s*,\s*y\s*=\s*)-?\d+(?:\.\d+)?",
                    rf"\1{x}\2{y}",
                    grounded_code,
                    count=1,
                )
            else:
                grounded_code = f"pyautogui.moveTo({x}, {y}); " + grounded_code
            return grounded_code

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

        has_drag_placeholders = all(
            token in code for token in ["START_X_COORD", "START_Y_COORD", "END_X_COORD", "END_Y_COORD"]
        )

        if has_drag_placeholders:
            start_desc, end_desc = self._parse_drag_descriptions(description)
            start_cmd = call_grounder(start_desc)
            end_cmd = call_grounder(end_desc)

            if "gta1" in self.visual_grounder_model.lower():
                if not (isinstance(start_cmd, tuple) and len(start_cmd) == 2):
                    raise ValueError(f"[GTA1] Expected START (x, y) tuple, got: {start_cmd}")
                if not (isinstance(end_cmd, tuple) and len(end_cmd) == 2):
                    raise ValueError(f"[GTA1] Expected END (x, y) tuple, got: {end_cmd}")
                sx, sy = start_cmd
                ex, ey = end_cmd
                return (
                    code.replace("START_X_COORD", str(sx))
                        .replace("START_Y_COORD", str(sy))
                        .replace("END_X_COORD", str(ex))
                        .replace("END_Y_COORD", str(ey))
                )
            raise ValueError("Drag placeholders are currently supported only for gta1 visual grounder.")

        # Single-point placeholder mode
        py_cmd = call_grounder(description)
        if "gta1" in self.visual_grounder_model.lower():
            if isinstance(py_cmd, tuple) and len(py_cmd) == 2:
                x, y = py_cmd
                return code.replace("X_COORD", str(x)).replace("Y_COORD", str(y))
            raise ValueError(f"[GTA1] Expected (x, y) tuple, got: {py_cmd}")
        return py_cmd

    def _parse_drag_descriptions(self, description: str):
        """Parse drag START/END descriptions from one description field."""
        if not description:
            raise ValueError(
                "Description is required for drag placeholders. "
                "Use: START: <start target>; END: <end target>"
            )
        normalized = description.replace("\n", " ").strip()
        upper = normalized.upper()
        start_idx = upper.find("START:")
        end_idx = upper.find("END:")
        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            raise ValueError(
                "Invalid drag description format. "
                "Use: START: <start target>; END: <end target>"
            )
        start_desc = normalized[start_idx + len("START:"):end_idx].strip(" ;")
        end_desc = normalized[end_idx + len("END:"):].strip(" ;")
        if not start_desc or not end_desc:
            raise ValueError(
                "Both START and END descriptions must be non-empty. "
                "Use: START: <start target>; END: <end target>"
            )
        return start_desc, end_desc

    def _gui_action(self, code: str, description: str = "") -> str:
        """Execute gui_action tool - pyautogui code with optional placeholder replacement."""
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
            
            # Skip step abstraction if wo_step is True
            if self.wo_step:
                step_abstraction = ""
            else:
                step_abstraction = "Result: " + self._step_abstraction_result(
                    before_screenshot, after_screenshot, eval_desc,
                    wo_roi=self.wo_roi, roi_margin=self.roi_margin
                )

            # Generate step_abstract
            thought_prefix = f"Thought: {self.current_thought} | " if self.current_thought else ""
            if description:
                step_abstract = f"Step {step}: gui_action | {thought_prefix}Description: {description} | Code: {final_code} | {step_abstraction}"
            else:
                step_abstract = f"Step {step}: gui_action | {thought_prefix}Code: {final_code} | {step_abstraction}"

            # Calculate step execution time
            step_time = time.time() - step_start_time
            # GUI loop detection should be robust to dynamic pixels (clock/cursor/animations),
            # so do not fingerprint raw screenshots.
            result_fingerprint = self._hash_text("success=True")

            self.action_logs.append({
                "step": step,
                "type": "gui_action",
                "execution_success": True,
                "screenshot": screenshot_file,
                "step_abstract": step_abstract,
                "step_time": round(step_time, 2),
                "token_usage": self.step_token_usage,
                "loop_action_fingerprint": requested_action_fingerprint,
                "loop_result_fingerprint": result_fingerprint
            })

            # Return execution result text for wo_step mode
            if description:
                return f"GUI Action: {description}\nCode: {final_code}\nStatus: Success\n{step_abstraction}"
            else:
                return f"GUI Action Code: {final_code}\nStatus: Success\n{step_abstraction}"

        except Exception as e:
            self.logger.error(f"GUI action execution error: {e}")

            # Generate step_abstract for error
            thought_prefix = f"Thought: {self.current_thought} | " if self.current_thought else ""
            if description:
                step_abstract = f"Step {step}: gui_action | {thought_prefix}Description: {description} | Code: {code} | Result: Error - {str(e)}"
            else:
                step_abstract = f"Step {step}: gui_action | {thought_prefix}Code: {code} | Result: Error - {str(e)}"

            # Calculate step execution time
            step_time = time.time() - step_start_time
            result_fingerprint = self._hash_text(f"success=False|error={str(e)}")

            self.action_logs.append({
                "step": step,
                "type": "gui_action",
                "execution_success": False,
                "screenshot": screenshot_file,
                "step_abstract": step_abstract,
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

    def _step_abstraction_result(self, before_screenshot: bytes, after_screenshot: bytes,
            action_description: str, wo_roi: bool = False,
            roi_margin: int = 50) -> str:
        """Abstract step by comparing before/after screenshots.

        Args:
            before_screenshot: Screenshot before action
            after_screenshot: Screenshot after action
            action_description: Description of the action performed
            wo_roi: If True, disable ROI cropping (default: False means ROI cropping is enabled)
            roi_margin: Margin to add around ROI when cropping (default: 50)

        Returns:
            Step abstract text (e.g., "Succeeded. Menu opened." or "Failed. No UI change.")
        """
        try:
            # Convert screenshots to PIL Images for ROI detection
            before_img = Image.open(io.BytesIO(before_screenshot))
            after_img = Image.open(io.BytesIO(after_screenshot))
            
            # Check for size mismatch and log detailed info for debugging
            if before_img.size != after_img.size:
                self.logger.error(f"[ANOMALY] Screenshot size mismatch detected!")
                
                # Save problematic screenshots for later analysis
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                error_dir = os.path.join(self.operations_dir, "size_mismatch_errors")
                os.makedirs(error_dir, exist_ok=True)
                before_img.save(os.path.join(error_dir, f"{timestamp}_before.png"))
                after_img.save(os.path.join(error_dir, f"{timestamp}_after.png"))
                self.logger.error(f"  Saved error screenshots to: {error_dir}")
                
            # Optionally crop to change ROI (enabled by default, disabled when wo_roi=True)
            if not wo_roi:
                try:
                    cropped_before, cropped_after = get_change_roi(
                        before_img, after_img,
                        margin=roi_margin,
                    )

                    # If ROI detected, use cropped images
                    if cropped_before is not None and cropped_after is not None:
                        before_img = cropped_before
                        after_img = cropped_after
                    else:
                        # No change detected - directly return without calling LLM
                        return "No change detected."
                except Exception as roi_error:
                    self.logger.warning(f"ROI detection failed, using full screenshots: {roi_error}")

            # Convert (possibly cropped) images to base64
            before_buffer = io.BytesIO()
            after_buffer = io.BytesIO()
            before_img.save(before_buffer, format="PNG")
            after_img.save(after_buffer, format="PNG")

            before_b64 = base64.b64encode(before_buffer.getvalue()).decode("utf-8")
            after_b64 = base64.b64encode(after_buffer.getvalue()).decode("utf-8")

            prompt = STEP_ABSTRACTION_PROMPT.format(
                action_description=action_description
            )

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

            step_abstraction = self.state_manager_llm(messages)
            return step_abstraction.strip()

        except Exception as e:
            self.logger.error(f"Failed to abstract step: {e}")
            return "Step abstraction failed due to error."

    def _bash_execution(self, code: str) -> str:
        """Execute bash commands or Python scripts (not pyautogui)."""
        code = self._normalize_bash_command(code)
        action_fingerprint = self._hash_text(code)
        self.logger.info(f"[bash_execution] {code}")

        # Record step start time
        step_start_time = time.time()

        step = self.operation_count + 1

        try:
            # Get before screenshot
            before_screenshot = self.env.controller.get_screenshot()

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
                step_abstraction = ""
            else:
                bash_description = f"Bash command: {code}\nOutput: {logs}..."  # Truncate long output
                step_abstraction = "Result: " + self._step_abstraction_result(
                    before_screenshot, after_screenshot, bash_description,
                    wo_roi=self.wo_roi, roi_margin=self.roi_margin
                )

            # Generate step_abstract summary
            thought_prefix = f"Thought: {self.current_thought} | " if self.current_thought else ""
            step_abstract = f"Step {step}: Bash execution | {thought_prefix}Code: {code} | Output: {logs} | {step_abstraction}"
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
                "step_abstract": step_abstract,
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
            thought_prefix = f"Thought: {self.current_thought} | " if self.current_thought else ""
            step_abstract = f"Step {step}: Bash execution | {thought_prefix}Code: {code} | Result: Error - {str(e)}"
            result_fingerprint = self._hash_text(f"error={str(e)}")

            self.action_logs.append({
                "step": step,
                "type": "bash_execution",
                "execution_success": False,
                "screenshot": screenshot_file,
                "step_abstract": step_abstract,
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
                step_abstraction = ""
            else:
                step_abstraction = "Result: " + self._step_abstraction_result(
                    before_screenshot, after_screenshot,
                    f"Waited {wait_seconds} seconds to observe UI changes",
                    wo_roi=self.wo_roi, roi_margin=self.roi_margin
                )

            # Generate step_abstract
            thought_prefix = f"Thought: {self.current_thought} | " if self.current_thought else ""
            step_abstract = f"Step {step}: wait | {thought_prefix}Duration: {wait_seconds}s | {step_abstraction}"

            # Calculate step execution time
            step_time = time.time() - step_start_time

            self.action_logs.append({
                "step": step,
                "type": "wait",
                "execution_success": True,
                "screenshot": screenshot_file,
                "step_abstract": step_abstract,
                "step_time": round(step_time, 2),
                "token_usage": self.step_token_usage
            })

            # Return execution result text for wo_step mode
            return f"Wait: {wait_seconds}s\nStatus: Success\n{step_abstraction}"

        except Exception as e:
            self.logger.error(f"Wait execution error: {e}")

            # Generate step_abstract for error
            thought_prefix = f"Thought: {self.current_thought} | " if self.current_thought else ""
            step_abstract = f"Step {step}: wait | {thought_prefix}Duration: {wait_seconds}s | Result: Error - {str(e)}"

            # Calculate step execution time
            step_time = time.time() - step_start_time

            self.action_logs.append({
                "step": step,
                "type": "wait",
                "execution_success": False,
                "screenshot": screenshot_file if 'screenshot_file' in locals() else "",
                "step_abstract": step_abstract,
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
