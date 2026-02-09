import ast
import logging
import re
from base64 import b64encode
from PIL import Image
from io import BytesIO
from typing import Dict, List, Optional, Tuple
import time

from .prompt.accessibility_tree_handle import linearize_accessibility_tree, trim_accessibility_tree
from .prompt.grounding_agent import GroundingAgent as Agent
from .tools.package.google_chrome import BrowserTools
from .prompt.procedural_memory import Prompt

logger = logging.getLogger("desktopenv.agent")

pure_text_settings = ["a11y_tree"]

# Prompt template for fixing parsing errors
FIX_RESPONSE_PROMPT = """Your previous response could not be parsed correctly. Please fix the format issue and try again.

Error message: {error_message}

Your previous response:
{response}

Please provide a corrected response with proper format."""

def resize_image(image, w, h):
    img = Image.open(BytesIO(image))
    # resize to max_pixel_num max_pixels
    img = img.resize((w, h))
    buf = BytesIO()
    img.save(buf, format='PNG')
    img_bytes = buf.getvalue()
    return img_bytes

def parse_code_from_string(input_string):
    # input_string = "\n".join([line.strip() for line in input_string.split(';') if line.strip()])
    if input_string.strip() in ["WAIT", "DONE", "FAIL", "RESET", "RECOVERY_DONE", "ROLLBACK"]:
        return [input_string.strip()]

    # This regular expression will match both ```code``` and ```python code```
    # and capture the `code` part. It uses a non-greedy match for the content inside.
    pattern = r"```(?:\w+\s+)?(.*?)```"
    # Find all non-overlapping matches in the string
    matches = re.findall(pattern, input_string, re.DOTALL)

    # The regex above captures the content inside the triple backticks.
    # The `re.DOTALL` flag allows the dot `.` to match newline characters as well,
    # so the code inside backticks can span multiple lines.

    # matches now contains all the captured code snippets

    codes = []

    for match in matches:
        match = match.strip()
        commands = ["WAIT", "DONE", "FAIL", "RESET", "RECOVERY_DONE", "ROLLBACK"]  # fixme: updates this part when we have more commands

        if match in commands:
            codes.append(match.strip())
        elif match.split("\n")[-1] in commands:
            if len(match.split("\n")) > 1:
                codes.append("\n".join(match.split("\n")[:-1]))
            codes.append(match.split("\n")[-1])
        else:
            codes.append(match)

    return codes


class AutoGLMAgent:
    def __init__(
        self,
        action_space="autoglm_computer_use",
        observation_type="a11y_tree",
        max_trajectory_length=3,
        a11y_tree_max_items=300,
        with_image: bool = True,
        screen_size = (1280, 720),
        image_size=(1280, 720),
        with_atree: bool = False,
        glm41v_format: bool = True,
        relative_coordinate: bool = True,
        client_password="password",
        gen_func=None,
        tool_in_sys_msg: bool = True,
        max_parse_retries: int = 3,
        enable_recovery: bool = True,
        max_failure_memory: int = 8,
        visual_grounder_model=None,
    ):
        self.action_space = action_space
        self.observation_type = observation_type
        assert action_space in ["autoglm_computer_use"], "Invalid action space"
        # assert observation_type in ["a11y_tree"], "Invalid observation type"
        self.max_trajectory_length = max_trajectory_length
        self.a11y_tree_max_items = a11y_tree_max_items
        self.with_image = with_image
        self.screen_size = screen_size
        self.image_size = image_size
        self.with_atree = with_atree
        self.glm41v_format = glm41v_format
        self.visual_grounder_model = visual_grounder_model
        self.relative_coordinate = relative_coordinate
        if self.visual_grounder_model is not None:
            model_name = getattr(self.visual_grounder_model, "model_name", "")
            if isinstance(model_name, str) and model_name.startswith("gta1"):
                self.relative_coordinate = False
        self.client_password = client_password
        self.gen_func = gen_func
        self.tool_in_sys_msg = tool_in_sys_msg
        self.max_parse_retries = max_parse_retries
        self.last_error_feedback = None
        self.last_parse_error = None
        self.enable_recovery = enable_recovery
        self.recovery_mode = False
        self.recovery_context = None
        self.failure_memory = []
        self.max_failure_memory = max_failure_memory
        self.failure_sequences = []

        self.tool_list = {
            "libreoffice_calc": "CalcTools",
            "libreoffice_impress": "ImpressTools",
            "libreoffice_writer": "WriterTools",
            "code": "CodeTools",
            "vlc": "VLCTools",
            "google_chrome": "BrowserTools",
        }
        
        Agent.relative_coordinate = self.relative_coordinate
        
        self.contents = []

    @property
    def turn_number(self):
        return len(self.contents)

    def set_recovery_mode(self, enabled: bool, context: Optional[Dict] = None):
        if not self.enable_recovery:
            self.recovery_mode = False
            self.recovery_context = None
            return
        self.recovery_mode = bool(enabled)
        self.recovery_context = context if enabled else None

    def record_failure(self, failure: Dict):
        if not self.enable_recovery or not failure:
            return
        item = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "action": str(failure.get("action", ""))[:300],
            "exe_result": str(failure.get("exe_result", ""))[:600],
            "app": failure.get("app") or "unknown",
            "reason": failure.get("reason") or "execution_error",
            "note": str(failure.get("note", ""))[:300],
        }
        self.failure_memory.append(item)
        if len(self.failure_memory) > self.max_failure_memory:
            self.failure_memory = self.failure_memory[-self.max_failure_memory :]

    def record_failure_sequence(self, sequence):
        if not self.enable_recovery or not sequence:
            return
        if isinstance(sequence, dict):
            seq = [str(item) for item in sequence.get("sequence", [])]
            if not seq:
                return
            item = {
                "sequence": seq,
                "meta": sequence.get("meta", {}),
            }
        else:
            seq = [str(item) for item in sequence]
            item = {"sequence": seq, "meta": {}}
        self.failure_sequences.append(item)
        if len(self.failure_sequences) > self.max_failure_memory:
            self.failure_sequences = self.failure_sequences[-self.max_failure_memory :]

    def _format_failure_memory(self) -> str:
        if not self.failure_memory:
            return ""
        lines = ["Failure Memory (do NOT repeat these mistakes):"]
        for idx, item in enumerate(self.failure_memory[-self.max_failure_memory :], start=1):
            action = item.get("action", "")
            exe_result = item.get("exe_result", "")
            reason = item.get("reason", "")
            app = item.get("app", "")
            note = item.get("note", "")
            line = f"{idx}. app={app}; reason={reason}; action={action}"
            if exe_result:
                line += f"; result={exe_result}"
            if note:
                line += f"; note={note}"
            lines.append(line)
        return "\n".join(lines)

    def _format_failure_sequences(self) -> str:
        if not self.failure_sequences:
            return ""
        lines = ["Failure Sequences (avoid repeating these action chains):"]
        for idx, item in enumerate(self.failure_sequences[-self.max_failure_memory :], start=1):
            seq = item.get("sequence", []) if isinstance(item, dict) else item
            meta = item.get("meta", {}) if isinstance(item, dict) else {}
            chain = " -> ".join(seq)
            meta_str = ""
            if meta:
                parts = []
                if "branch_id" in meta:
                    parts.append(f"branch={meta.get('branch_id')}")
                if "parent_branch_id" in meta:
                    parts.append(f"parent={meta.get('parent_branch_id')}")
                if "start_index" in meta and "end_index" in meta:
                    parts.append(f"range={meta.get('start_index')}..{meta.get('end_index')}")
                if "reason" in meta:
                    parts.append(f"reason={meta.get('reason')}")
                if parts:
                    meta_str = " [" + ", ".join(parts) + "]"
            lines.append(f"{idx}. {chain}{meta_str}")
        return "\n".join(lines)

    def _format_recovery_instructions(self) -> str:
        if not self.enable_recovery:
            return ""
        parts = []
        if self.recovery_mode:
            parts.append(
                "RECOVERY MODE: Roll back recent changes to return to a stable, re-plannable state. "
                "Use undo/close dialog/page back/scroll restore/clear input, etc. "
                "Keep actions minimal and safe. "
                "After performing ONE rollback step, output `ROLLBACK` to decrement the replay index. "
                "When stable, output `RECOVERY_DONE`. "
                "If rollback is impossible and you are blocked, output `RESET`."
            )
            if self.recovery_context:
                parts.append(
                    "Last failure context: "
                    f"action={self.recovery_context.get('action')}; "
                    f"result={self.recovery_context.get('exe_result')}; "
                    f"app={self.recovery_context.get('app')}"
                )
                if self.recovery_context.get("note"):
                    parts.append(f"Note: {self.recovery_context.get('note')}")
                failed_seq = self.recovery_context.get("failed_sequence")
                if failed_seq:
                    parts.append("Failed sequence to avoid: " + " -> ".join([str(s) for s in failed_seq]))
        memory = self._format_failure_memory()
        if memory:
            parts.append(memory)
        seq_memory = self._format_failure_sequences()
        if seq_memory:
            parts.append(seq_memory)
        return "\n\n".join(parts)

    def _parse_agent_call(self, code: str):
        tree = ast.parse(code)
        if len(tree.body) != 1:
            raise ValueError("Expected a single Agent call")
        node = tree.body[0]
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            raise ValueError("Expected a call expression")
        call = node.value
        if not isinstance(call.func, ast.Attribute):
            raise ValueError("Expected attribute call")
        if not isinstance(call.func.value, ast.Name) or call.func.value.id != "Agent":
            raise ValueError("Expected Agent.<method> call")
        method = call.func.attr
        args = [ast.literal_eval(arg) for arg in call.args]
        kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
        return method, args, kwargs

    def _ground_coordinates(self, description: str, screenshot: bytes) -> Tuple[int, int]:
        if self.visual_grounder_model is None:
            raise ValueError("Visual grounder model is not configured")
        img = Image.open(BytesIO(screenshot))
        model_name = getattr(self.visual_grounder_model, "model_name", "")
        scale = 1.5 if isinstance(model_name, str) and model_name.startswith("gta1") else 1.0
        py_cmd, reasoning = self.visual_grounder_model.call_cua(
            description,
            img,
            environment="linux",
            screen_width=self.screen_size[0],
            screen_height=self.screen_size[1],
            scale=scale,
        )
        if not py_cmd:
            raise ValueError(f"Visual Grounder failed to provide result. Reasoning: {reasoning}")
        if isinstance(py_cmd, tuple) and len(py_cmd) == 2:
            return py_cmd
        raise ValueError(f"Visual Grounder did not return coordinates: {py_cmd}")

    def _resolve_agent_call(self, code: str, desc: str, screenshot: bytes):
        method, args, kwargs = self._parse_agent_call(code)

        if method == "click":
            coordinate = self._ground_coordinates(desc, screenshot)
            num_clicks = kwargs.get("num_clicks", args[1] if len(args) > 1 else 1)
            button_type = kwargs.get("button_type", args[2] if len(args) > 2 else "left")
            return Agent.click(coordinate, num_clicks=num_clicks, button_type=button_type)
        if method == "type":
            coordinate = self._ground_coordinates(desc, screenshot)
            text = kwargs.get("text", args[1] if len(args) > 1 else "")
            overwrite = kwargs.get("overwrite", False)
            enter = kwargs.get("enter", False)
            return Agent.type(coordinate=coordinate, text=text, overwrite=overwrite, enter=enter)
        if method == "scroll":
            coordinate = self._ground_coordinates(desc, screenshot)
            direction = kwargs.get("direction", args[1] if len(args) > 1 else "down")
            return Agent.scroll(coordinate, direction)
        if method == "drag_and_drop":
            desc_text = desc.strip()
            start_desc = ""
            end_desc = ""
            if "->" in desc_text:
                parts = desc_text.split("->", 1)
                start_desc, end_desc = parts[0].strip(), parts[1].strip()
            elif " to " in desc_text.lower():
                parts = re.split(r"\s+to\s+", desc_text, maxsplit=1, flags=re.IGNORECASE)
                if len(parts) == 2:
                    start_desc, end_desc = parts[0].strip(), parts[1].strip()
            elif ";" in desc_text:
                parts = desc_text.split(";", 1)
                start_desc, end_desc = parts[0].strip(), parts[1].strip()

            if not start_desc:
                start_desc = f"{desc_text} (drag start)"
            if not end_desc:
                end_desc = f"{desc_text} (drop target)"

            drag_from = self._ground_coordinates(start_desc, screenshot)
            drop_on = self._ground_coordinates(end_desc, screenshot)
            return Agent.drag_and_drop(drag_from, drop_on)
        if method == "exit":
            success = kwargs.get("success", args[0] if len(args) > 0 else True)
            return Agent.exit(success=success)

        raise ValueError(f"Grounding not supported for Agent.{method}")

    def prepare(self, instruction: str, obs: Dict, history: List, last_result: str = "") -> List:
        """
        Predict the next action(s) based on the current observation.
        """
        if "exe_result" in obs and not last_result:
            last_result = obs["exe_result"]
            if self.contents:
                self.contents[-1]["exe_result"] = last_result

        cur_app = obs["cur_app"]
        logger.info(f"current app is {cur_app}")

        if cur_app:
            tool_name = cur_app.strip().lower().replace("-", "_")
            tool_name = tool_name if tool_name in self.tool_list.keys() else None
        else:
            tool_name = None

        setup_prompt, func_def_prompt, note_prompt = Prompt.construct_procedural_memory(
            Agent, app_name=tool_name, client_password=self.client_password, with_image=self.with_image, with_atree=self.with_atree, relative_coordinate=self.relative_coordinate, glm41v_format=self.glm41v_format
        )
        if self.tool_in_sys_msg:
            system_message = setup_prompt + "\n\n" + func_def_prompt + "\n\n" + note_prompt
        else:
            system_message = setup_prompt + "\n\n" + note_prompt
        system_message += "\n\n**IMPORTANT** You are asked to complete the following task: {}".format(instruction)
        recovery_msg = self._format_recovery_instructions()
        if recovery_msg:
            system_message += "\n\n" + recovery_msg

        messages = [
            {
                "role": "system",
                "content": system_message,
            }
        ]
        messages.extend(history)

        if obs["apps"]:
            app_str = "Window ID    App Name    Title\n"
            for window_id, app in obs["apps"].items():
                app_str += f"{window_id}    {app['app_name']}    {app['title']}\n"
        else:
            app_str = "None"

        last_result = last_result.strip() if last_result else "None"
        last_result = last_result[:2000] + "..." if len(last_result) > 2000 else last_result

        tree = linearize_accessibility_tree(obs["accessibility_tree"], "Ubuntu")
        tree = trim_accessibility_tree(tree, 300)

        app_info = obs["app_info"].strip() if obs["app_info"] else "None"
        app_info = app_info[:5000] + "..." if len(app_info) > 5000 else app_info

        prompt = "* Apps: {}\n\n* Current App: {}{}\n\n* App Info: {}\n\n* Previous Action Result: {}".format(
            app_str.strip(),
            obs["cur_window_id"].strip() if obs["cur_window_id"] in app_str else "None",
            '\n\n* A11y Tree: {}'.format(tree.strip()) if self.with_atree else "",
            app_info,
            last_result if last_result else "None",
        ) + (
            "\n\n" + func_def_prompt if not self.tool_in_sys_msg else ""
        )

        content = [{"type": "text", "text": prompt}]
        if self.with_image and obs.get('screenshot'):
            screenshot = resize_image(obs['screenshot'], self.image_size[0], self.image_size[1])
            content = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64encode(screenshot).decode('utf-8')}",
                        "detail": "high",
                    },
                }
            ] + content

        messages.append({"role": "user", "content": content})

        return messages

    def execute(self, response, obs):
        self.last_parse_error = None  # Reset error before each attempt
        try:
            actions = parse_code_from_string(response)
            action = actions[0]
            
            pattern = r'^python\s*(\\+n|\n)+'
            action = re.sub(pattern, '', action, flags=re.IGNORECASE)
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
            
            thought = re.sub(pattern, '', response, flags=re.DOTALL).strip()
            
            logger.info(f"The pesudo action is {action}")

            if action in ["WAIT", "DONE", "FAIL", "RESET", "RECOVERY_DONE", "ROLLBACK"]:
                actions = [action]
            elif "Agent." in action:
                if self.visual_grounder_model is not None and obs.get("screenshot"):
                    try:
                        grounded_action = self._resolve_agent_call(action, thought, obs["screenshot"])
                        actions = [grounded_action]
                    except Exception as e:
                        logger.error(f"Visual grounding failed, falling back to raw Agent action: {e}")
                        actions = [eval(action)]
                else:
                    actions = [
                        eval(action),
                    ]
            elif "BrowserTools." in action:  # TODO: special check for BrowserTools
                actions = [
                    eval(action),
                ]
            else:
                actions = Agent.tool_commands(action, obs["cur_app"].strip().replace("-", "_").lower())
                logger.info(f"The grounded action is {actions[0]}")
        except Exception as e:
            self.last_parse_error = str(e)  # Store error message for retry feedback
            logger.error(f"Failed to parse action from response: {e}")
            actions = []

        return actions

    def format_history(self, max_turns=30):
        history = []
        for ix in range(self.turn_number):
            if ix == 0:
                env_input = "**Environment State (Omitted)**"
            else:
                env_input = (
                    f"**Environment State (Omitted)**\nPrevious Action Result: {self.contents[ix - 1]['exe_result']}"
                )

            env_input = env_input[:2000] + "..." if len(env_input) > 2000 else env_input
            response = (
                self.contents[ix]["response"][:1500] + "..."
                if len(self.contents[ix]["response"]) > 1500
                else self.contents[ix]["response"]
            )
            history.append({"role": "user", "content": [{"type": "text", "text": env_input}]})
            history.append({"role": "assistant", "content": [{"type": "text", "text": response}]})

        return history[-max_turns * 2:]

    def predict(self, instruction: str, obs: Dict) -> List:
        history = self.format_history()
        messages = self.prepare(instruction, obs, history)

        assert self.gen_func is not None, "gen_func is not set"
        
        response = None
        actions = []
        parse_error_msg = None
        
        # Retry loop for parsing errors
        for attempt in range(self.max_parse_retries):
            # Add error feedback if this is a retry
            if attempt > 0 and self.last_error_feedback:
                logger.warning(f"Retry attempt {attempt}/{self.max_parse_retries} due to parsing error")
                retry_messages = messages.copy()
                retry_messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": self.last_error_feedback}]
                })
            else:
                retry_messages = messages
            
            # Call gen_func with network retry
            for _ in range(3):
                try:
                    response = self.gen_func(retry_messages)
                    break
                except Exception as e:
                    logger.error("Failed to call gen_func, Error: " + str(e))
            else:
                raise RuntimeError("Failed to call gen_func after retries")

            logger.info("RESPONSE: %s", response)

            # Try to execute/parse the response
            actions = self.execute(response, obs)
            
            if actions:
                # Successfully parsed, clear error feedback
                self.last_error_feedback = None
                break
            else:
                # Parsing failed, use stored error or generic message
                parse_error_msg = self.last_parse_error or "Failed to parse action from response"
                logger.error(f"Action parsing error (attempt {attempt + 1}/{self.max_parse_retries}): {parse_error_msg}")
                
                # If not last attempt, set error feedback for retry
                if attempt < self.max_parse_retries - 1:
                    self.last_error_feedback = FIX_RESPONSE_PROMPT.format(
                        error_message=parse_error_msg,
                        response=response
                    )
                else:
                    # Last attempt failed, mark task as FAIL
                    logger.error("All retry attempts exhausted, cannot parse valid action. Marking task as FAIL.")
                    self.last_error_feedback = None
                    actions = ["FAIL"]

        # update the contents
        self.contents.append(
            {
                "instruction": instruction,
                "index": len(self.contents),
                "response": response,
                "action": "Parse error" if not actions else actions[0],
                "exe_result": "Invalid action" if not actions else "",
                **obs,
            }
        )
        return response, actions

    def reset(self, _logger=None):
        global logger
        logger = _logger if _logger is not None else logging.getLogger("desktopenv.aguvis_agent")

        self.contents = []
