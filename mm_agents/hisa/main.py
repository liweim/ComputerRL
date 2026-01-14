"""
HiSA (Hierarchical Self-Adaptive) Agent implementation.
Adapted from GUIAgent/agents/hisa.py with AutoGLM-compatible pseudo-code format.
"""

import base64
import json
import logging
import os
import re
import time
import traceback
from io import BytesIO
from typing import Dict, List, Optional, Tuple

from PIL import Image

from .prompts import (
    GLOBAL_PLANNER_PROMPT,
    FIX_RESPONSE_PROMPT,
    STEP_ABSTRACTION_PROMPT,
    CONTEXT_REFINEMENT_PROMPT,
    PATTERN_INDUCTION_PROMPT,
    PATTERN_SYNTHESIS_PROMPT,
)

        # Import from autoglm_v for code parsing and grounding
from ..autoglm_v.prompt.grounding_agent import GroundingAgent as Agent
from ..autoglm_v.prompt.accessibility_tree_handle import linearize_accessibility_tree, trim_accessibility_tree
from ..autoglm_v.tools.package.google_chrome import BrowserTools  # Needed for eval() in execute()

logger = logging.getLogger("desktopenv.hisa")


def resize_image(image: bytes, w: int, h: int) -> bytes:
    """Resize image to specified dimensions."""
    img = Image.open(BytesIO(image))
    img = img.resize((w, h))
    buf = BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


def parse_code_from_string(input_string: str) -> List[str]:
    """Parse code from response string (AutoGLM format)."""
    input_string = input_string.strip()
    
    if input_string in ["WAIT", "DONE", "FAIL"]:
        return [input_string]

    # Match ```python code``` or ```code```
    pattern = r"```(?:\w+\s+)?(.*?)```"
    matches = re.findall(pattern, input_string, re.DOTALL)

    codes = []
    for match in matches:
        match = match.strip()
        commands = ["WAIT", "DONE", "FAIL"]

        if match in commands:
            codes.append(match)
        elif match.split("\n")[-1] in commands:
            if len(match.split("\n")) > 1:
                codes.append("\n".join(match.split("\n")[:-1]))
            codes.append(match.split("\n")[-1])
        else:
            codes.append(match)

    return codes


class HiSAAgent:
    """
    HiSA Agent with AutoGLM-compatible pseudo-code format.
    
    Combines hisa.py's advanced features (pattern learning, step abstraction, 
    context refinement) with AutoGLM's pseudo-code action format.
    """

    def __init__(
        self,
        action_space: str = "autoglm_computer_use",
        screen_size: Tuple[int, int] = (1920, 1080),
        image_size: Tuple[int, int] = (1280, 720),
        max_trajectory_length: int = 3,
        a11y_tree_max_items: int = 300,
        with_atree: bool = False,
        client_password: str = "password",
        gen_func=None,
        state_manager_func=None,
        max_parse_retries: int = 3,
        max_steps: int = 50,
        # HiSA advanced features
        wo_pattern: bool = True,  # Disable pattern learning by default
        wo_step: bool = False,  # Enable step abstraction by default
        wo_refinement: bool = False,  # Enable context refinement by default
        refine_period: int = 5,
        sliding_window_size: int = 5,
        bash_timeout: int = 60,
    ):
        self.action_space = action_space
        self.screen_size = screen_size
        self.image_size = image_size
        self.max_trajectory_length = max_trajectory_length
        self.a11y_tree_max_items = a11y_tree_max_items
        self.with_atree = with_atree
        self.client_password = client_password
        self.gen_func = gen_func
        self.state_manager_func = state_manager_func

        # Tool list for application-specific tools
        self.tool_list = {
            "libreoffice_calc": "CalcTools",
            "libreoffice_impress": "ImpressTools",
            "libreoffice_writer": "WriterTools",
            "code": "CodeTools",
            "vlc": "VLCTools",
            "google_chrome": "BrowserTools",
        }

        # Pattern induction settings
        self.wo_pattern = wo_pattern

        self.max_parse_retries = max_parse_retries
        self.max_steps = max_steps

        # HiSA advanced features
        self.wo_step = wo_step
        self.wo_refinement = wo_refinement
        self.refine_period = refine_period
        self.sliding_window_size = sliding_window_size
        self.bash_timeout = bash_timeout

        # Set Agent's coordinate mode
        Agent.relative_coordinate = True

        # Execution state
        self.contents = []
        self.action_logs = []
        self.last_error_feedback = None
        self.last_parse_error = None
        self.last_full_summary = None
        self.last_summary_step = 0
        self.current_thought = ""
        self.past_pattern_text = ""
        self.task_instruction = ""

    def tool_commands(self, code: str, tool_name: str):
        """Generate tool commands for application-specific tools."""
        command = f"from {tool_name} import *; "
        command += code

        tool_class = self.tool_list[tool_name]
        command += f"; {tool_class}.print_result()"

        return [
            command,
        ]

    @property
    def turn_number(self) -> int:
        return len(self.contents)

    def reset(self, _logger=None):
        """Reset agent state for a new task."""
        global logger
        logger = _logger if _logger is not None else logging.getLogger("desktopenv.hisa")

        self.contents = []
        self.action_logs = []
        self.last_error_feedback = None
        self.last_parse_error = None
        self.last_full_summary = None
        self.last_summary_step = 0
        self.current_thought = ""
        self.past_pattern_text = ""
        self.task_instruction = ""

    def _build_execution_history(self) -> str:
        """Build execution history text from action logs."""
        if not self.action_logs:
            return ""

        # Apply sliding window if context refinement is disabled
        logs_to_use = self.action_logs
        if self.wo_refinement and len(self.action_logs) > self.sliding_window_size:
            logs_to_use = self.action_logs[-self.sliding_window_size:]

        # Build condensed history
        history_lines = []
        if not self.wo_refinement and self.last_full_summary:
            # Use summary + recent logs
            history_lines.append(self.last_full_summary)
            for log in self.action_logs[self.last_summary_step:]:
                if "step_abstract" in log:
                    history_lines.append(log["step_abstract"])
        else:
            # Use logs directly (with sliding window if applicable)
            for log in logs_to_use:
                if "step_abstract" in log:
                    history_lines.append(log["step_abstract"])

        return "\n".join(history_lines)

    def _summarize_history_segment(
        self, 
        logs: List[Dict], 
        start_step: int, 
        end_step: int, 
        previous_summary: str = ""
    ) -> str:
        """Summarize a segment of action logs with context refinement."""
        if not logs and not previous_summary:
            return f"Steps {start_step}~{end_step}: No actions. Suggestion: Continue"

        # Build history text
        history_lines = []
        for log in logs:
            if "step_abstract" in log:
                history_lines.append(log["step_abstract"])

        if previous_summary:
            if history_lines:
                history_text = f"<previous_summary>\n{previous_summary}\n</previous_summary>\n\n<new_steps>\n" + "\n".join(history_lines) + "\n</new_steps>"
            else:
                history_text = f"<previous_summary>\n{previous_summary}\n</previous_summary>"
        else:
            history_text = "\n".join(history_lines) if history_lines else ""

        if not history_text.strip():
            return f"Steps {start_step}~{end_step}: No detailed records. Suggestion: Continue"

        # Use state_manager to summarize if available
        if self.state_manager_func:
            prompt = CONTEXT_REFINEMENT_PROMPT.format(
                task_instruction=self.task_instruction,
                start_step=start_step,
                end_step=end_step,
                history_text=history_text
            )
            try:
                messages = [{"role": "user", "content": prompt}]
                summary = self.state_manager_func(messages)
                # Extract content from <answer> tag if present
                # First try to match <answer>...</answer>
                answer_match = re.search(r'<answer>(.*?)</answer>', summary, re.DOTALL)
                if answer_match:
                    summary = answer_match.group(1).strip()
                else:
                    # If no closing tag, try to get content after first <answer> until next tag or end
                    answer_match = re.search(r'<answer>([^<]+)', summary, re.DOTALL)
                    if answer_match:
                        summary = answer_match.group(1).strip()
                # Remove code block markers if present
                summary = re.sub(r'```\w*\s*', '', summary).strip()
                summary = re.sub(r'```', '', summary).strip()
                return summary
            except Exception as e:
                logger.error(f"Failed to summarize history: {e}")

        # Fallback: simple concatenation
        if previous_summary:
            return f"{previous_summary} + Steps {start_step}~{end_step}: {len(logs)} actions. Suggestion: Continue"
        return f"Steps {start_step}~{end_step}: {len(logs)} actions executed. Suggestion: Continue"

    def _do_context_refinement(self):
        """Perform context refinement if needed."""
        total_logs = len(self.action_logs)
        
        if self.wo_refinement or total_logs == 0 or total_logs % self.refine_period != 0:
            return

        logger.info(f"[Context Refinement] Triggered at step {total_logs}")
        if self.last_full_summary:
            # Not first time: use previous summary + new logs
            logs_to_summarize = self.action_logs[self.last_summary_step:]
            start_step = self.action_logs[0]["step"]
            end_step = self.action_logs[-1]["step"]
            summary = self._summarize_history_segment(
                logs_to_summarize, start_step, end_step,
                previous_summary=self.last_full_summary
            )
        else:
            # First time: summarize all logs
            logs_to_summarize = self.action_logs
            start_step = logs_to_summarize[0]["step"]
            end_step = logs_to_summarize[-1]["step"]
            summary = self._summarize_history_segment(logs_to_summarize, start_step, end_step)

        self.last_full_summary = summary
        self.last_summary_step = total_logs
        logger.info(f"[Context Refinement] Completed: {summary}")

    def _step_abstraction(
        self, 
        before_screenshot: bytes, 
        after_screenshot: bytes, 
        action_description: str
    ) -> str:
        """Abstract step by comparing before/after screenshots."""
        if self.wo_step or not self.state_manager_func:
            return ""

        try:
            # Resize images for comparison
            before_img = Image.open(BytesIO(before_screenshot))
            after_img = Image.open(BytesIO(after_screenshot))
            
            # Convert to base64
            before_buffer = BytesIO()
            after_buffer = BytesIO()
            before_img.save(before_buffer, format="PNG")
            after_img.save(after_buffer, format="PNG")
            before_b64 = base64.b64encode(before_buffer.getvalue()).decode("utf-8")
            after_b64 = base64.b64encode(after_buffer.getvalue()).decode("utf-8")

            prompt = STEP_ABSTRACTION_PROMPT.format(action_description=action_description)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Before screenshot:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{before_b64}"}},
                        {"type": "text", "text": "After screenshot:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{after_b64}"}},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            response = self.state_manager_func(messages)
            result = response
            # Extract content from <answer> tag if present
            # First try to match <answer>...</answer>
            answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
            if answer_match:
                result = answer_match.group(1).strip()
            else:
                # If no closing tag, try to get content after first <answer> until next tag or end
                answer_match = re.search(r'<answer>([^<]+)', response, re.DOTALL)
                if answer_match:
                    result = answer_match.group(1).strip()
            # Remove code block markers if present
            result = re.sub(r'```\w*\s*', '', result).strip()
            result = re.sub(r'```', '', result).strip()
            logger.info(f"[Step Abstraction]: {response}")
            return result

        except Exception as e:
            logger.error(f"[Step Abstraction] Failed: {e}")
            return "Step abstraction failed."

    def pattern_induction(self, task_instruction: str, action_logs: List[Dict]) -> List[Dict]:
        """Extract key lessons from task execution (simplified version).

        Returns:
            List of lesson dicts with 'type' and 'lesson' fields
        """
        if self.wo_pattern or not self.state_manager_func:
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

            response = self.state_manager_func(messages)

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

            import json
            lessons = json.loads(json_str)
            if isinstance(lessons, list):
                # Validate that each item is a dict with 'type' and 'lesson'
                validated_lessons = []
                for item in lessons[:3]:  # Max 3 lessons
                    if isinstance(item, dict) and "type" in item and "lesson" in item:
                        # Validate type is success or failure
                        if item["type"] in ["success", "failure"]:
                            validated_lessons.append(item)
                return validated_lessons
            else:
                logger.warning(f"Expected list, got {type(lessons)}")
                return []

        except Exception as e:
            logger.error(f"Failed to extract lessons: {e}")
            return []

    def prepare(self, instruction: str, obs: Dict, last_result: str = "") -> List[Dict]:
        """Prepare messages for the model."""
        self.task_instruction = instruction

        # Update last result in contents
        if "exe_result" in obs and not last_result:
            last_result = obs["exe_result"]
            if self.contents:
                self.contents[-1]["exe_result"] = last_result

        # Do context refinement if needed
        self._do_context_refinement()

        # Determine current tool/app
        cur_app = obs.get("cur_app", "").strip().replace("-", "_").lower() if obs.get("cur_app") else None
        tool_name = cur_app if cur_app in self.tool_list else None

        # Build system message with dynamic prompt construction
        from .prompt.procedural_memory import Prompt as HiSAPrompt
        setup_prompt, func_def_prompt, note_prompt = HiSAPrompt.construct_procedural_memory(
            Agent, app_name=tool_name, client_password=self.client_password, with_image=True, with_atree=self.with_atree, relative_coordinate=True, glm41v_format=True
        )

        system_message = setup_prompt + "\n\n" + func_def_prompt + "\n\n" + note_prompt
        system_message += f"\n\n**IMPORTANT** You are asked to complete the following task: {instruction}"

        messages = [{"role": "system", "content": system_message}]

        # Build execution history
        execution_history = self._build_execution_history()

        # Build observation info
        app_str = "None"
        if obs.get("apps"):
            app_str = "Window ID    App Name    Title\n"
            for window_id, app in obs["apps"].items():
                app_str += f"{window_id}    {app['app_name']}    {app['title']}\n"

        last_result = last_result.strip() if last_result else "None"

        # Process A11y tree if enabled
        tree = ""
        if self.with_atree and obs.get("accessibility_tree"):
            tree = linearize_accessibility_tree(obs["accessibility_tree"], "Ubuntu")
            tree = trim_accessibility_tree(tree, self.a11y_tree_max_items)

        app_info = obs.get("app_info", "").strip() if obs.get("app_info") else "None"

        # Build user message
        user_text_parts = []

        # Add past patterns if available
        if self.past_pattern_text:
            user_text_parts.append(f"<past_pattern>\n{self.past_pattern_text}\n</past_pattern>\n")

        # Add execution history
        if execution_history:
            user_text_parts.append(f"<execution_history>\n{execution_history}\n</execution_history>\n")

        # Add error feedback if retry
        if self.last_error_feedback:
            user_text_parts.append(f"<error_feedback>\n{self.last_error_feedback}\n</error_feedback>\n")

        # Add current observation
        user_text_parts.append(f"* Apps: {app_str.strip()}")
        user_text_parts.append(f"* Current App: {obs.get('cur_window_id', 'None')}")
        if tree:
            user_text_parts.append(f"* A11y Tree: {tree.strip()}")
        user_text_parts.append(f"* App Info: {app_info}")
        user_text_parts.append(f"* Previous Action Result: {last_result}")

        user_text = "\n\n".join(user_text_parts)

        # Build content with image
        content = [{"type": "text", "text": user_text}]
        if obs.get('screenshot'):
            screenshot = resize_image(obs['screenshot'], self.image_size[0], self.image_size[1])
            content = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64.b64encode(screenshot).decode('utf-8')}",
                        "detail": "high",
                    },
                }
            ] + content

        messages.append({"role": "user", "content": content})

        return messages

    def execute(self, response: str, obs: Dict) -> List:
        """Parse and execute response, return list of actions."""
        self.last_parse_error = None

        try:
            # Extract thinking for logging
            think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
            if think_match:
                self.current_thought = think_match.group(1).strip()

            # Parse code from response
            actions = parse_code_from_string(response)
            action = actions[0]
            
            action = re.sub(r'^python\s*(\\+n|\n)+', '', action, flags=re.IGNORECASE)
            action = re.sub(r'^(\\+n|\n)+', '', action)
            action = re.sub(r'(\\+n|\n)+$', '', action)

            # Fix text parameter with escaped quotes (e.g. text=\'I\'m happy\')
            match = re.search(r"text=\\'(.*)\\'(?=[,)])", action)
            if match:
                content = match.group(1).replace("\\'", "'")
                action = action[:match.start()] + f"text={repr(content)}" + action[match.end():]
            
            # Fix other simple escaped quotes (e.g. button_type=\'left\')
            action = re.sub(r"=\\'([^'\\]*)\\'", r"='\1'", action)
            if 'button=' in action:
                action = action.replace('button=', 'button_type=')

            logger.info(f"Parsed pseudo action: {action}")

            # Handle special bash action
            if action.startswith("Agent.bash("):
                # Extract bash command and return as special action
                match = re.search(r"Agent\.bash\(command=['\"](.+?)['\"]\)", action)
                if match:
                    return [{"type": "bash", "command": match.group(1)}]
                raise ValueError(f"Failed to parse bash command: {action}")

            # Convert pseudo-code to pyautogui command
            if "Agent." in action:
                actions = [eval(action)]
            elif "BrowserTools." in action:
                actions = [eval(action)]
            else:
                cur_app = obs.get("cur_app", "").strip().replace("-", "_").lower() if obs.get("cur_app") else None
                if cur_app and cur_app in self.tool_list:
                    actions = self.tool_commands(action, cur_app)
                else:
                    # Try direct eval
                    actions = [eval(action)]

            logger.info(f"Grounded action: {actions[0] if actions else 'None'}")
            return actions

        except Exception as e:
            self.last_parse_error = str(e)
            logger.error(f"Failed to parse action: {e}")
            return []

    def format_history(self, max_turns: int = 30) -> List[Dict]:
        """Format conversation history for context."""
        history = []
        for ix in range(self.turn_number):
            if ix == 0:
                env_input = "**Environment State (Omitted)**"
            else:
                env_input = f"**Environment State (Omitted)**\nPrevious Action Result: {self.contents[ix - 1].get('exe_result', 'None')}"

            response = self.contents[ix].get("response", "")

            history.append({"role": "user", "content": [{"type": "text", "text": env_input}]})
            history.append({"role": "assistant", "content": [{"type": "text", "text": response}]})

        return history[-max_turns * 2:]

    def predict(self, instruction: str, obs: Dict) -> Tuple[str, List]:
        """Predict the next action based on observation."""
        messages = self.prepare(instruction, obs)

        assert self.gen_func is not None, "gen_func is not set"

        response = None
        actions = []

        # Retry loop for parsing errors
        for attempt in range(self.max_parse_retries):
            # Add error feedback if retry
            if attempt > 0 and self.last_error_feedback:
                logger.warning(f"Retry attempt {attempt}/{self.max_parse_retries}")
                retry_messages = messages.copy()
                retry_messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": self.last_error_feedback}]
                })
            else:
                retry_messages = messages

            # Call model with network retry
            for _ in range(3):
                try:
                    response = self.gen_func(retry_messages)
                    break
                except Exception as e:
                    logger.error(f"Failed to call gen_func: {e}")
            else:
                raise RuntimeError("Failed to call gen_func after retries")

            logger.info(f"Response: {response}")

            # Try to parse response
            actions = self.execute(response, obs)

            if actions:
                self.last_error_feedback = None
                break
            else:
                parse_error_msg = self.last_parse_error or "Failed to parse action"
                logger.error(f"Parse error (attempt {attempt + 1}/{self.max_parse_retries}): {parse_error_msg}")

                if attempt < self.max_parse_retries - 1:
                    self.last_error_feedback = FIX_RESPONSE_PROMPT.format(
                        error_message=parse_error_msg,
                        response=response
                    )
                else:
                    logger.error("All retry attempts exhausted. Marking as FAIL.")
                    self.last_error_feedback = None
                    actions = ["FAIL"]

        # Update contents
        self.contents.append({
            "instruction": instruction,
            "index": len(self.contents),
            "response": response,
            "action": "Parse error" if not actions else (actions[0] if isinstance(actions[0], str) else str(actions[0])),
            "exe_result": "Invalid action" if not actions else "",
            "thought": self.current_thought,
            **{k: v for k, v in obs.items() if k not in ['screenshot', 'accessibility_tree']},
        })

        return response, actions

    def add_action_log(
        self,
        step: int,
        action_type: str,
        action: str,
        execution_success: bool,
        screenshot_file: str = "",
        exe_result: str = "",
        step_time: float = 0.0,
        before_screenshot: bytes = None,
        after_screenshot: bytes = None,
    ):
        """Add action log entry with optional step abstraction."""
        step_abstract = ""
        
        if not self.wo_step and before_screenshot and after_screenshot:
            abstraction_result = self._step_abstraction(
                before_screenshot, after_screenshot, action
            )
            step_abstract = f"Step {step}: {action_type} | Action: {action} | Result: {abstraction_result}"
        else:
            step_abstract = f"Step {step}: {action_type} | Action: {action} | Result: {exe_result}"

        self.action_logs.append({
            "step": step,
            "type": action_type,
            "execution_success": execution_success,
            "screenshot": screenshot_file,
            "step_abstract": step_abstract,
            "step_time": round(step_time, 2),
        })

    def get_action_logs(self) -> List[Dict]:
        """Get all action logs."""
        return self.action_logs
