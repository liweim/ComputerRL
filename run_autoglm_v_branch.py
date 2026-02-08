#!/usr/bin/env python3
"""Screenshot-only GUI Agent with online branch search and rollback/restart.

Based on run_autoglm_v.py + mm_agents/autoglm_v action format, but uses only screenshots
for planning, grounding, and success checking.
"""

import argparse
import base64
import json
import os
import re
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple
import requests
from PIL import Image
from io import BytesIO
from json_repair import repair_json
from utils import get_change_roi, count_images_in_messages
from run_autoglm_v import DesktopEnv
from utils import setup_logger
from mm_agents.autoglm_v import AutoGLMAgent
import logging

logger = logging.getLogger("desktopenv.branch_agent")

# =========================
# Helpers: LLM + JSON parse
# =========================

FIX_RESPONSE_PROMPT = """Your previous response could not be parsed correctly.
Please follow the format strictly:
1) A line: Expected change: <text>
2) A ```python``` code block for the main action
3) Optional additional ```python``` blocks for rollback actions
Do NOT include any other text.

Error message: {error_message}

Your previous response:
{response}
"""


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run screenshot-only branch-search GUI agent")

    # environment config (similar to run_autoglm_v)
    parser.add_argument("--path_to_vm", type=str)
    parser.add_argument("--provider_name", type=str, default="docker")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--action_space", type=str, default="autoglm_computer_use")
    parser.add_argument("--screen_width", type=int, default=1920)
    parser.add_argument("--screen_height", type=int, default=1080)
    parser.add_argument("--sleep_after_execution", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--client_password", type=str, default="")

    # LLM config
    parser.add_argument("--planner_model", type=str, default="autoglm-os")
    parser.add_argument("--planner_temperature", type=float, default=0.2)
    parser.add_argument("--planner_top_p", type=float, default=0.1)
    parser.add_argument("--planner_max_tokens", type=int, default=512)
    parser.add_argument("--planner_repetition_penalty", type=float, default=1.0)
    parser.add_argument("--grounder_model", type=str, default="")
    parser.add_argument("--checker_model", type=str, default="")
    parser.add_argument("--image_width", type=int, default=1280)
    parser.add_argument("--image_height", type=int, default=720)

    # task config
    parser.add_argument("--domain", type=str, default="all")
    parser.add_argument("--test_all_meta_path", type=str, default="evaluation_examples/test_one.json")
    parser.add_argument("--test_config_base_dir", type=str, default="evaluation_examples/examples")
    parser.add_argument("--result_dir", type=str, default="./results/branch_agent")
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--rerun", action="store_true", help="Rerun all tasks (ignore existing results)")
    parser.add_argument("--rerun_fail", action="store_true", help="Rerun only failed tasks")

    # branch search config
    parser.add_argument("--max_branches", type=int, default=3)
    parser.add_argument("--max_grounding_retries", type=int, default=3)
    parser.add_argument("--restart_limit", type=int, default=3)
    parser.add_argument("--disable_vlm_success_check", action="store_true")
    parser.add_argument("--max_parse_retries", type=int, default=3)

    # ablations
    parser.add_argument("--wo_restart", action="store_true")
    parser.add_argument("--wo_pruning", action="store_true")
    parser.add_argument("--wo_multi_branch", action="store_true")

    return parser.parse_args()

def _extract_json(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text.strip()


def _safe_json_loads(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    text = _extract_json(text)
    try:
        return json.loads(text)
    except Exception:
        logger.exception("Failed to parse JSON, attempting repair.")
        try:
            repaired = repair_json(text)
            return json.loads(repaired)
        except Exception:
            logger.exception("JSON repair failed; returning empty object.")
            return {}


def _extract_python_blocks(text: str) -> List[str]:
    if not text:
        return []
    pattern = r"```python\\s*(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    return [m.strip() for m in matches if m.strip()]


def _extract_expected_change(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"Expected change\\s*:\\s*(.+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def _resize_image(image_bytes: bytes, w: int, h: int) -> bytes:
    img = Image.open(BytesIO(image_bytes))
    img = img.resize((w, h))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class LLMCaller:
    def __init__(
        self,
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        repetition_penalty: float,
    ):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.repetition_penalty = repetition_penalty
        self.base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:30000/v1")
        self.api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.last_usage: Dict[str, Any] = {}
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.total_image_count = 0

    def __call__(self, messages: List[Dict[str, Any]]) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "stream": False,
        }
        url = f"{self.base_url}/chat/completions"
        response = requests.post(url, json=data, headers=headers, timeout=60.0)
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        image_count = count_images_in_messages(messages)
        self.last_usage = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "image_count": image_count,
        }
        self.total_prompt_tokens += self.last_usage["prompt_tokens"]
        self.total_completion_tokens += self.last_usage["completion_tokens"]
        self.total_tokens += self.last_usage["total_tokens"]
        self.total_image_count += image_count
        return content

    def reset_usage(self) -> None:
        self.last_usage = {}
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.total_image_count = 0


# =========================
# Action building
# =========================


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _resolve_coordinate(coord: List[float], screen_size: Tuple[int, int]) -> Tuple[int, int]:
    if coord is None or len(coord) != 2:
        return None, None
    x, y = coord
    if 0 <= x <= 1 and 0 <= y <= 1:
        x_abs = int(round(x * screen_size[0]))
        y_abs = int(round(y * screen_size[1]))
    elif 0 <= x <= 1000 and 0 <= y <= 1000:
        x_abs = int(round(x * screen_size[0] / 1000))
        y_abs = int(round(y * screen_size[1] / 1000))
    else:
        x_abs = int(round(x))
        y_abs = int(round(y))
    x_abs = int(_clamp(x_abs, 0, screen_size[0] - 1))
    y_abs = int(_clamp(y_abs, 0, screen_size[1] - 1))
    return x_abs, y_abs


def build_action_from_spec(
    action_spec: Any,
    screen_size: Tuple[int, int],
) -> Tuple[Any, Dict[str, Any]]:
    """Return (action, meta) where action is runnable by DesktopEnv.step."""
    meta = {}
    if action_spec is None:
        return "WAIT", meta
    if isinstance(action_spec, str):
        return action_spec.strip(), meta
    if isinstance(action_spec, dict) and "action_type" in action_spec:
        meta["action_type"] = str(action_spec.get("action_type", "")).lower()
        return action_spec, meta

    if not isinstance(action_spec, dict):
        return "WAIT", meta

    action_type = (action_spec.get("type") or action_spec.get("action_type") or "").lower().strip()
    meta["action_type"] = action_type or "unknown"
    if action_type in ["wait", "pause"]:
        meta["wait_seconds"] = float(action_spec.get("seconds", 1.0))
        return "WAIT", meta
    if action_type in ["done", "finish"]:
        return "DONE", meta
    if action_type in ["fail", "abort"]:
        return "FAIL", meta
    if action_type == "open_app":
        app_name = action_spec.get("app_name", "")
        launch_app_command = action_spec.get("launch_app_command", "")
        if not launch_app_command:
            # Fallback to autoglm_v launch name; DesktopEnv setup handles shell
            launch_app_command = action_spec.get("app_name", "")
        return {
            "action_type": "OPEN_APP",
            "parameters": {"launch_app_command": launch_app_command, "app_name": app_name},
        }, meta
    if action_type == "open_chrome_tab":
        urls = action_spec.get("urls_to_open") or action_spec.get("urls") or []
        return {
            "action_type": "OPEN_CHROME_TAB",
            "parameters": {"urls_to_open": urls},
        }, meta

    if action_type in ["click", "type", "scroll", "hotkey", "press", "drag"]:
        coord = action_spec.get("coordinate")
        x_abs, y_abs = _resolve_coordinate(coord, screen_size) if coord else (None, None)
        if action_type == "click":
            num_clicks = int(action_spec.get("num_clicks", 1))
            button = action_spec.get("button", action_spec.get("button_type", "left"))
            if x_abs is None:
                return "WAIT", meta
            cmd = (
                f"pyautogui.click({x_abs}, {y_abs}, clicks={num_clicks}, button={repr(button)}); "
                "print('Click Success')"
            )
            meta["coordinate"] = [x_abs, y_abs]
            return cmd, meta
        if action_type == "type":
            text = action_spec.get("text", "")
            overwrite = bool(action_spec.get("overwrite", False))
            enter = bool(action_spec.get("enter", False))
            cmd = ""
            if x_abs is not None:
                cmd += f"pyautogui.click({x_abs}, {y_abs}); "
                meta["coordinate"] = [x_abs, y_abs]
            if overwrite:
                cmd += "pyautogui.hotkey('ctrl', 'a'); pyautogui.press('backspace'); "
            cmd += f"pyautogui.write({repr(text)}); "
            if enter:
                cmd += "pyautogui.press('enter'); "
            cmd += "print('Type Success')"
            return cmd, meta
        if action_type == "scroll":
            direction = action_spec.get("direction", "down")
            amount = int(action_spec.get("amount", 100))
            amount = abs(amount) if direction == "up" else -abs(amount)
            if x_abs is None:
                x_abs, y_abs = screen_size[0] // 2, screen_size[1] // 2
            cmd = (
                f"import pyautogui; pyautogui.moveTo({x_abs}, {y_abs}); "
                f"pyautogui.scroll({amount}); print('Scroll Success')"
            )
            meta["coordinate"] = [x_abs, y_abs]
            return cmd, meta
        if action_type in ["hotkey", "press"]:
            keys = action_spec.get("keys") or action_spec.get("key") or []
            if isinstance(keys, str):
                keys = [keys]
            keys = [str(k).lower() for k in keys if k]
            if len(keys) == 1:
                cmd = f"pyautogui.press({repr(keys[0])}); print('Press Success')"
            else:
                cmd = f"pyautogui.hotkey({', '.join([repr(k) for k in keys])}); print('Hotkey Success')"
            return cmd, meta
        if action_type == "drag":
            drag_from = action_spec.get("drag_from_coordinate")
            drop_on = action_spec.get("drop_on_coordinate")
            x1, y1 = _resolve_coordinate(drag_from, screen_size) if drag_from else (None, None)
            x2, y2 = _resolve_coordinate(drop_on, screen_size) if drop_on else (None, None)
            if x1 is None or x2 is None:
                return "WAIT", meta
            cmd = (
                f"pyautogui.moveTo({x1}, {y1}); "
                f"pyautogui.dragTo({x2}, {y2}, duration=1.0); pyautogui.mouseUp(); "
                "print('Drag and Drop Success')"
            )
            meta["coordinate"] = [x2, y2]
            return cmd, meta

    return "WAIT", meta


# =========================
# Planner / Grounder prompts
# =========================


def build_planner_messages(
    screenshot: bytes,
    image_size: Tuple[int, int],
    task_instruction: str,
    history_summary: str,
    failed_branches: List[int],
    failure_memory: List[str],
    last_failure_info: str,
    max_branches: int,
    default_rollback: List[Dict[str, Any]],
    nudge: str = "",
) -> List[Dict[str, Any]]:
    if not hasattr(build_planner_messages, "_agent"):
        build_planner_messages._agent = AutoGLMAgent(
            action_space="autoglm_computer_use",
            observation_type="screenshot",
            screen_size=(1920, 1080),
            image_size=image_size,
            with_image=True,
            with_atree=False,
            gen_func=None,
        )
    agent = build_planner_messages._agent
    obs = {
        "screenshot": screenshot,
        "accessibility_tree": {},
        "instruction": task_instruction,
        "apps": {},
        "cur_window_id": "",
        "cur_app": "",
        "app_info": "",
    }
    messages = agent.prepare(task_instruction, obs, history=[], last_result="")

    extra_text = (
        "Output format requirements:\n"
        "- Line 1: Expected change: <text>\n"
        "- Then output the main action in a ```python``` block.\n"
        "- Then output rollback actions as additional ```python``` blocks (one per step, optional).\n"
        "- Do NOT include any other text outside the line and python blocks.\n"
        "Example:\n"
        "Expected change: open settings menu\n"
        "```python\npyautogui.click(100, 100)\n```\n"
        "```python\npyautogui.press('esc')\n```\n\n"
        f"Recent execution summary:\n{history_summary or 'None'}\n\n"
        f"Failed branch indices in current anchor (avoid repeating): {failed_branches}\n\n"
        f"Failure memory (soft constraints):\n{chr(10).join(failure_memory) if failure_memory else 'None'}\n\n"
        f"Last failure info:\n{last_failure_info or 'None'}\n\n"
        f"Default rollback ladder (system will fill if missing):\n{json.dumps(default_rollback, ensure_ascii=False)}\n\n"
        f"{nudge}\n"
        "Return the Expected change line and python blocks only."
    )
    messages.append({"role": "user", "content": [{"type": "text", "text": extra_text}]})
    return messages


def build_grounder_messages(
    screenshot: bytes,
    image_size: Tuple[int, int],
    target_desc: str,
    task_instruction: str,
) -> List[Dict[str, Any]]:
    system_text = (
        "You are a screenshot-only visual grounder.\n"
        "Return JSON ONLY: {\"coordinate\":[x,y],\"confidence\":0-1}.\n"
        "Coordinates are relative [0,1000] in the screenshot."
    )
    user_text = f"Task: {task_instruction}\nTarget: {target_desc}\nReturn JSON only."
    screenshot_resized = _resize_image(screenshot, image_size[0], image_size[1])
    content = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64.b64encode(screenshot_resized).decode('utf-8')}",
                "detail": "high",
            },
        },
        {"type": "text", "text": user_text},
    ]
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": content},
    ]


def build_success_check_messages(
    before_img: bytes,
    after_img: bytes,
    image_size: Tuple[int, int],
    expected_change: str,
    check_instruction: str,
) -> List[Dict[str, Any]]:
    system_text = (
        "You are a strict GUI change evaluator.\n"
        "Compare before/after screenshots and decide if the expected change happened.\n"
        "Return JSON ONLY: {\"success\":true|false,\"reason\":\"...\",\"irrecoverable\":false}."
    )
    user_text = (
        f"Expected change: {expected_change}\n"
        f"Additional check: {check_instruction or 'None'}"
    )
    before_resized = _resize_image(before_img, image_size[0], image_size[1])
    after_resized = _resize_image(after_img, image_size[0], image_size[1])
    content = [
        {"type": "text", "text": "Before screenshot:"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(before_resized).decode('utf-8')}", "detail": "high"}},
        {"type": "text", "text": "After screenshot:"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(after_resized).decode('utf-8')}", "detail": "high"}},
        {"type": "text", "text": user_text},
    ]
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": content},
    ]


def build_failure_summary_messages(
    screenshot: bytes,
    image_size: Tuple[int, int],
    task_instruction: str,
    recent_steps: List[Dict[str, Any]],
    failed_branches: List[int],
    rollback_info: str,
) -> List[Dict[str, Any]]:
    system_text = (
        "You are to summarize failure patterns. Output JSON ONLY.\n"
        "Return: {\"summary\":[\"constraint 1\", \"constraint 2\", ...]}.\n"
        "Do NOT propose actions; only constraints and avoidances."
    )
    user_text = (
        f"Task: {task_instruction}\n\n"
        f"Recent steps:\n{json.dumps(recent_steps, ensure_ascii=False)}\n\n"
        f"Failed branch indices: {failed_branches}\n\n"
        f"Rollback info: {rollback_info}\n\n"
        "Summarize repeated failures and what to avoid."
    )
    screenshot_resized = _resize_image(screenshot, image_size[0], image_size[1])
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(screenshot_resized).decode('utf-8')}", "detail": "high"}},
        {"type": "text", "text": user_text},
    ]
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": content},
    ]


# =========================
# Branch Agent
# =========================


class ScreenshotBranchAgent:
    def __init__(
        self,
        planner_llm: LLMCaller,
        grounder_llm: Optional[LLMCaller],
        checker_llm: Optional[LLMCaller],
        screen_size: Tuple[int, int],
        image_size: Tuple[int, int],
        max_branches: int = 3,
        max_grounding_retries: int = 3,
        restart_limit: int = 3,
        failure_memory_size: int = 50,
        use_vlm_success_check: bool = True,
        wo_restart: bool = False,
        wo_pruning: bool = False,
        wo_multi_branch: bool = False,
        max_parse_retries: int = 3,
    ):
        self.planner_llm = planner_llm
        self.grounder_llm = grounder_llm
        self.checker_llm = checker_llm or planner_llm
        self.screen_size = screen_size
        self.image_size = image_size
        self.max_branches = max_branches
        self.max_grounding_retries = max_grounding_retries
        self.restart_limit = restart_limit
        self.failure_memory_size = failure_memory_size
        self.use_vlm_success_check = use_vlm_success_check
        self.wo_restart = wo_restart
        self.wo_pruning = wo_pruning
        self.wo_multi_branch = wo_multi_branch
        self.max_parse_retries = max_parse_retries

        self.anchor_stack: List[Dict[str, Any]] = []
        self.exec_log: List[Dict[str, Any]] = []
        self.failure_memory: List[Dict[str, Any]] = []
        self.restart_counter = 0
        self.global_step_count = 0
        self.anchor_id_counter = 0
        self.grounding_retry_count = 0
        self.pending_rollback: List[Dict[str, Any]] = []
        self.pending_backtrack = False
        self.current_branch: Optional[Dict[str, Any]] = None
        self.current_anchor_id: Optional[int] = None
        self.current_branch_id: Optional[int] = None
        self.last_success_anchor_id: Optional[int] = None
        self.last_success_branch_id: Optional[int] = None
        self.last_failure_info: str = ""
        self.rollback_count = 0
        self.anchor_backtrack_count = 0
        self.last_rollback_info = ""

        self.default_rollback_ladder = [
            {"action": {"type": "hotkey", "keys": ["esc"]}, "stop_condition": "modal closed or focus cleared"},
            {"action": {"type": "click", "coordinate": [10, 10]}, "stop_condition": "focus moved to empty area"},
            {"action": {"type": "hotkey", "keys": ["alt", "left"]}, "stop_condition": "navigated back"},
            {"action": {"type": "hotkey", "keys": ["ctrl", "r"]}, "stop_condition": "page refreshed"},
        ]

    def last_token_usage(self) -> Dict[str, Any]:
        llms = [self.planner_llm]
        if self.grounder_llm:
            llms.append(self.grounder_llm)
        if self.checker_llm and self.checker_llm is not self.planner_llm:
            llms.append(self.checker_llm)
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "image_count": 0}
        for llm in llms:
            if not llm or not llm.last_usage:
                continue
            usage["prompt_tokens"] += llm.last_usage.get("prompt_tokens", 0)
            usage["completion_tokens"] += llm.last_usage.get("completion_tokens", 0)
            usage["total_tokens"] += llm.last_usage.get("total_tokens", 0)
            usage["image_count"] += llm.last_usage.get("image_count", 0)
        return usage

    def reset(self) -> None:
        self.anchor_stack = []
        self.exec_log = []
        self.failure_memory = []
        self.restart_counter = 0
        self.global_step_count = 0
        self.anchor_id_counter = 0
        self.grounding_retry_count = 0
        self.pending_rollback = []
        self.pending_backtrack = False
        self.current_branch = None
        self.current_anchor_id = None
        self.current_branch_id = None
        self.last_success_anchor_id = None
        self.last_success_branch_id = None
        self.last_failure_info = ""
        self.rollback_count = 0
        self.anchor_backtrack_count = 0
        self.last_rollback_info = ""

    def reset_for_restart(self) -> None:
        """Reset transient search state but keep failure memory and counters."""
        self.anchor_stack = []
        self.grounding_retry_count = 0
        self.pending_rollback = []
        self.pending_backtrack = False
        self.current_branch = None
        self.current_anchor_id = None
        self.current_branch_id = None
        self.last_success_anchor_id = None
        self.last_success_branch_id = None
        self.last_failure_info = ""
        self.last_rollback_info = ""

    def _history_summary(self, max_items: int = 5) -> str:
        if not self.exec_log:
            return ""
        items = self.exec_log[-max_items:]
        lines = []
        for it in items:
            lines.append(
                f"step={it['step_id']} action={it['action_type']} success={it['success']} "
                f"expected={it.get('expected_change','')}"
            )
        return "\n".join(lines)

    def _failure_memory_text(self) -> List[str]:
        items = []
        for mem in self.failure_memory[-5:]:
            if mem.get("type") == "restart_summary":
                items.extend(mem.get("summary", []))
        return items

    def _current_anchor(self) -> Optional[Dict[str, Any]]:
        return self.anchor_stack[-1] if self.anchor_stack else None

    def _get_anchor_by_id(self, anchor_id: Optional[int]) -> Optional[Dict[str, Any]]:
        if anchor_id is None:
            return None
        for anchor in reversed(self.anchor_stack):
            if anchor["anchor_id"] == anchor_id:
                return anchor
        return None

    def _select_next_branch(self) -> Optional[Dict[str, Any]]:
        anchor = self._current_anchor()
        if anchor:
            for idx, step in enumerate(anchor["branch_set"]["steps"]):
                if idx in anchor["branch_set"]["tried"] or idx in anchor["branch_set"]["failed"]:
                    continue
                anchor["branch_set"]["tried"].add(idx)
                self.current_anchor_id = anchor["anchor_id"]
                self.current_branch_id = idx
                return step
            return None
        return None

    def _push_anchor(self, steps: List[Dict[str, Any]]) -> None:
        self.anchor_id_counter += 1
        parent_id = self.anchor_stack[-1]["anchor_id"] if self.anchor_stack else None
        anchor = {
            "anchor_id": self.anchor_id_counter,
            "branch_set": {"steps": steps, "tried": set(), "failed": set()},
            "parent_anchor_id": parent_id,
            "committed": False,
        }
        self.anchor_stack.append(anchor)

    def _pop_anchor(self) -> Optional[Dict[str, Any]]:
        if self.anchor_stack:
            return self.anchor_stack.pop()
        return None

    def _plan_branches(self, screenshot: bytes, task_instruction: str) -> List[Dict[str, Any]]:
        failed_branches = []
        anchor = self._current_anchor()
        if anchor:
            failed_branches = sorted(list(anchor["branch_set"]["failed"]))
        messages = build_planner_messages(
            screenshot=screenshot,
            image_size=self.image_size,
            task_instruction=task_instruction,
            history_summary=self._history_summary(),
            failed_branches=failed_branches,
            failure_memory=self._failure_memory_text(),
            last_failure_info=self.last_failure_info,
            max_branches=1,
            default_rollback=self.default_rollback_ladder,
            nudge="",
        )
        step = self._plan_single_step(messages)
        if step:
            return [self._coerce_branch_step(step)]
        raise RuntimeError(f"Planner output invalid after {self.max_parse_retries} retries")

    def _plan_single_step(self, messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        last_error = "Invalid or empty response"
        for attempt in range(self.max_parse_retries):
            response = self.planner_llm(messages)
            blocks = _extract_python_blocks(response)
            expected_change = _extract_expected_change(response)
            if blocks:
                return {
                    "expected_change": expected_change,
                    "action_python": blocks[0],
                    "rollback_python": blocks[1:] if len(blocks) > 1 else [],
                }
            logger.error(
                "Planner output invalid (attempt %d/%d): %s",
                attempt + 1,
                self.max_parse_retries,
                response,
            )
            feedback = FIX_RESPONSE_PROMPT.format(error_message=last_error, response=response)
            messages = [messages[0], messages[1], {"role": "user", "content": feedback}]
        return None

    def _coerce_branch_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        action = step.get("action_python", "")
        expected_change = step.get("expected_change") or "UI should change as expected."
        meta = {}
        return {
            "action": action,
            "expected_change": expected_change,
            "rollback_ladder": [
                {"action": rb, "stop_condition": f"Rollback step {i + 1} applied"}
                for i, rb in enumerate(step.get("rollback_python", []))
            ]
            or self.default_rollback_ladder,
            "meta": meta,
        }

    def _ground_coordinate(self, screenshot: bytes, target_desc: str, task_instruction: str) -> Optional[List[float]]:
        if not self.grounder_llm:
            return None
        messages = build_grounder_messages(screenshot, self.image_size, target_desc, task_instruction)
        response = self.grounder_llm(messages)
        data = _safe_json_loads(response)
        coord = data.get("coordinate")
        if coord and isinstance(coord, list) and len(coord) == 2:
            return coord
        return None

    def _success_check(
        self,
        before_img: bytes,
        after_img: bytes,
        expected_change: str,
        check_instruction: str = "",
    ) -> Tuple[bool, str, bool, Dict[str, Any]]:
        try:
            messages = build_success_check_messages(
                before_img=before_img,
                after_img=after_img,
                image_size=self.image_size,
                expected_change=expected_change,
                check_instruction=check_instruction,
            )
            response = self.checker_llm(messages)
            result = _safe_json_loads(response)
            success = bool(result.get("success", False))
            reason = result.get("reason", "")
            irrecoverable = bool(result.get("irrecoverable", False))
            return success, reason or "Checked by VLM.", irrecoverable, {"diff_ratio": None}
        except Exception as e:
            logger.exception("Success check failed.")
            return False, f"Success check error: {e}", False, {"diff_ratio": None}

    def _classify_grounding_failure(
        self,
        diff_ratio: float,
        action_meta: Dict[str, Any],
    ) -> bool:
        if diff_ratio is None:
            return False
        if diff_ratio < 0.005:
            return True
        coord = action_meta.get("coordinate")
        if coord is None:
            return False
        return diff_ratio < 0.01

    def _check_stop_condition(
        self,
        before_img: bytes,
        after_img: bytes,
        stop_condition: str,
    ) -> bool:
        if not stop_condition:
            return True
        if not self.use_vlm_success_check:
            # Fallback: any visible change is acceptable
            try:
                before_pil = Image.open(BytesIO(before_img))
                after_pil = Image.open(BytesIO(after_img))
                roi_before, roi_after = get_change_roi(before_pil, after_pil, margin=30)
                return roi_before is not None and roi_after is not None
            except Exception:
                logger.exception("Stop condition check failed (ROI fallback).")
                return False
        messages = build_success_check_messages(
            before_img=before_img,
            after_img=after_img,
            image_size=self.image_size,
            expected_change=stop_condition,
            check_instruction="Check whether stop condition is satisfied.",
        )
        response = self.checker_llm(messages)
        data = _safe_json_loads(response)
        return bool(data.get("success", False))

    def get_next_action(self, screenshot: bytes, task_instruction: str) -> Tuple[Any, Dict[str, Any]]:
        if self.pending_rollback:
            next_rb = self.pending_rollback[0]
            action, action_meta = build_action_from_spec(next_rb.get("action"), self.screen_size)
            meta = {
                "is_rollback": True,
                "stop_condition": next_rb.get("stop_condition", ""),
                "action_meta": action_meta,
                "expected_change": "rollback",
                "success_check": {"threshold": 0.01, "roi": "auto"},
            }
            return action, meta

        if self.pending_backtrack:
            try:
                steps = self._plan_branches(screenshot, task_instruction)
                self.pending_backtrack = False
                return self._prepare_branch_action(steps[0], screenshot, task_instruction)
            except Exception:
                self.pending_backtrack = False
                return None, {"need_restart": True}

        # Normal planning
        steps = self._plan_branches(screenshot, task_instruction)
        if not steps:
            return "WAIT", {"action_meta": {"action_type": "wait"}}

        self.current_anchor_id = None
        self.current_branch_id = 0
        return self._prepare_branch_action(steps[0], screenshot, task_instruction)

    def _prepare_branch_action(
        self,
        branch_step: Dict[str, Any],
        screenshot: bytes,
        task_instruction: str,
    ) -> Tuple[Any, Dict[str, Any]]:
        action_spec = branch_step.get("action")
        # Grounder fill-in if needed
        if isinstance(action_spec, dict):
            coord = action_spec.get("coordinate")
            target_desc = action_spec.get("target") or branch_step.get("expected_change") or ""
            if coord is None and self.grounder_llm:
                coord = self._ground_coordinate(screenshot, target_desc, task_instruction)
                if coord is not None:
                    action_spec["coordinate"] = coord
        action, action_meta = build_action_from_spec(action_spec, self.screen_size)
        self.current_branch = branch_step
        return action, {
            "is_rollback": False,
            "action_meta": action_meta,
            "expected_change": branch_step.get("expected_change", ""),
            "rollback_ladder": branch_step.get("rollback_ladder") or self.default_rollback_ladder,
        }

    def handle_action_result(
        self,
        before_img: bytes,
        after_img: bytes,
        action: Any,
        meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        self.global_step_count += 1
        is_rollback = bool(meta.get("is_rollback", False))
        action_meta = meta.get("action_meta", {})
        expected_change = meta.get("expected_change", "")
        success_check = meta.get("success_check", {})
        result: Dict[str, Any] = {"success": False, "irrecoverable": False}

        if isinstance(action, str) and action in ["WAIT", "DONE", "FAIL"]:
            result["success"] = True
            result["reason"] = f"{action} executed"
            return result

        # Rollback action handling
        if is_rollback:
            stop_condition = meta.get("stop_condition", "")
            stop_ok = self._check_stop_condition(before_img, after_img, stop_condition)
            if stop_ok:
                self.pending_rollback = []
            else:
                self.pending_rollback = self.pending_rollback[1:]
            if not self.pending_rollback:
                self.pending_backtrack = True
            result["success"] = stop_ok
            result["failure_type"] = "rollback" if not stop_ok else "rollback_success"
            return result

        # Normal action handling
        success, reason, irrecoverable, diff_info = self._success_check(
            before_img, after_img, expected_change
        )
        result["success"] = success
        result["reason"] = reason
        result["irrecoverable"] = irrecoverable
        result["diff_info"] = diff_info

        if success:
            # Mark current anchor committed
            anchor = self._current_anchor()
            if anchor and anchor["anchor_id"] == self.current_anchor_id:
                anchor["committed"] = True
                self.last_success_anchor_id = self.current_anchor_id
                self.last_success_branch_id = self.current_branch_id
            self.grounding_retry_count = 0
            return result

        # Failure handling
        diff_ratio = diff_info.get("diff_ratio", None)
        is_grounding_fail = self._classify_grounding_failure(diff_ratio, action_meta)
        if is_grounding_fail and self.grounding_retry_count < self.max_grounding_retries:
            self.grounding_retry_count += 1
            ratio_str = "N/A" if diff_ratio is None else f"{diff_ratio:.4f}"
            self.last_failure_info = (
                f"Grounding/no-op suspected. diff_ratio={ratio_str}, "
                f"coord={action_meta.get('coordinate')}, expected={expected_change}"
            )
            result["failure_type"] = "grounding_retry"
            return result

        self.grounding_retry_count = 0
        ratio_str = "N/A" if diff_ratio is None else f"{diff_ratio:.4f}"
        self.last_failure_info = f"Failure: {reason} diff_ratio={ratio_str}"

        if irrecoverable:
            self.anchor_stack = []
            self.pending_backtrack = True
            result["failure_type"] = "irrecoverable"
            return result

        # Mark branch failed in anchor (current or last success anchor)
        anchor = self._current_anchor()
        target_anchor = anchor
        target_branch_id = self.current_branch_id
        if target_anchor is None and self.last_success_anchor_id is not None:
            target_anchor = self._get_anchor_by_id(self.last_success_anchor_id)
            target_branch_id = self.last_success_branch_id
        if target_anchor and not self.wo_pruning and target_branch_id is not None:
            target_anchor["branch_set"]["failed"].add(target_branch_id)

        # Start rollback ladder
        ladder = meta.get("rollback_ladder") or self.default_rollback_ladder
        if ladder:
            self.pending_rollback = ladder
            self.rollback_count += 1
            self.last_rollback_info = f"rollback_ladder_len={len(ladder)}"
        else:
            self.pending_backtrack = True

        result["failure_type"] = "branch_failed"
        return result

    def should_restart(self) -> bool:
        return False

    def record_exec_log(
        self,
        action: Any,
        meta: Dict[str, Any],
        result: Dict[str, Any],
        screenshot_file: str,
        duration: float,
        token_usage: Dict[str, Any],
    ) -> None:
        action_type = meta.get("action_meta", {}).get("action_type", "unknown")
        entry = {
            "step_id": self.global_step_count,
            "anchor_id": self.current_anchor_id,
            "branch_id": self.current_branch_id,
            "action": action if isinstance(action, (str, dict, list)) else str(action),
            "action_type": action_type,
            "expected_change": meta.get("expected_change", ""),
            "success": bool(result.get("success", False)),
            "failure_type": result.get("failure_type", ""),
            "reason": result.get("reason", ""),
            "screenshot": screenshot_file,
            "time": round(duration, 2),
            "token": token_usage,
            "is_rollback": bool(meta.get("is_rollback", False)),
        }
        self.exec_log.append(entry)

    def record_failure_memory(self, entry: Dict[str, Any]) -> None:
        self.failure_memory.append(entry)
        if len(self.failure_memory) > self.failure_memory_size:
            self.failure_memory = self.failure_memory[-self.failure_memory_size:]


# =========================
# Runner
# =========================


def run_single_example_branch(
    agent: ScreenshotBranchAgent,
    env: DesktopEnv,
    example: Dict[str, Any],
    instruction: str,
    args: argparse.Namespace,
    example_result_dir: str,
    scores: List[float],
) -> None:
    start_time = time.time()
    agent.reset()
    agent.planner_llm.reset_usage()
    if agent.grounder_llm:
        agent.grounder_llm.reset_usage()
    if agent.checker_llm:
        agent.checker_llm.reset_usage()

    env.reset(task_config=example)
    time.sleep(60)
    obs = env._get_obs()

    operations_dir = os.path.join(example_result_dir, "operations")
    os.makedirs(operations_dir, exist_ok=True)

    done = False
    step_idx = 0
    restart_events = []

    env.controller.start_recording()
    while not done and step_idx < args.max_steps:
        screenshot = obs.get("screenshot", b"")
        action, meta = agent.get_next_action(screenshot, instruction)

        if action is None and meta.get("need_restart"):
            if agent.should_restart():
                restart_events.append(
                    {
                        "reason": "anchor_exhausted",
                        "step": step_idx + 1,
                        "recent_steps": agent.exec_log[-10:],
                        "rollback_info": agent.last_rollback_info,
                    }
                )
                _perform_restart(agent, env, example, instruction, screenshot, restart_events[-1])
                obs = env._get_obs()
                if agent.restart_counter > agent.restart_limit:
                    break
                continue
            else:
                break

        pause_override = meta.get("action_meta", {}).get("wait_seconds")
        pause = pause_override if pause_override is not None else args.sleep_after_execution

        before_img = screenshot
        step_start = time.time()
        obs, reward, done, info = env.step(action, pause)
        after_img = obs.get("screenshot", b"")
        step_time = time.time() - step_start

        screenshot_file = f"step_{step_idx + 1}_gui_action.png"
        with open(os.path.join(operations_dir, screenshot_file), "wb") as f:
            f.write(after_img)

        token_usage = agent.last_token_usage()
        result = agent.handle_action_result(before_img, after_img, action, meta)
        agent.record_exec_log(action, meta, result, screenshot_file, step_time, token_usage)

        # failure memory update
        if not result.get("success", False) and result.get("failure_type") != "grounding_retry":
            agent.record_failure_memory(
                {
                    "type": "failure",
                    "reason": result.get("reason", ""),
                    "expected_change": meta.get("expected_change", ""),
                    "action_type": meta.get("action_meta", {}).get("action_type", ""),
                }
            )

        # grounding retry triggers a replan in next step
        if result.get("failure_type") == "grounding_retry":
            step_idx += 1
            continue

        # check restart conditions
        if agent.should_restart():
            restart_events.append(
                {
                    "reason": "anchor_exhausted",
                    "step": step_idx + 1,
                    "recent_steps": agent.exec_log[-10:],
                    "rollback_info": agent.last_rollback_info,
                }
            )
            _perform_restart(agent, env, example, instruction, after_img, restart_events[-1])
            obs = env._get_obs()
            if agent.restart_counter > agent.restart_limit:
                break
            step_idx += 1
            continue

        step_idx += 1

    if not done:
        try:
            env.step("FAIL", 0)
        except Exception:
            logger.exception("Failed to send FAIL action.")

    result = env.evaluate()
    scores.append(result)

    execution_time = time.time() - start_time
    _save_execution_log(
        example_result_dir,
        result,
        step_idx,
        execution_time,
        agent,
        args,
        example,
        restart_events,
    )
    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")
    env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))


def _perform_restart(
    agent: ScreenshotBranchAgent,
    env: DesktopEnv,
    example: Dict[str, Any],
    instruction: str,
    screenshot: bytes,
    restart_event: Dict[str, Any],
) -> None:
    if agent.wo_restart:
        return

    # Failure summary before restart
    recent_steps = agent.exec_log[-20:]
    failed_branches = []
    anchor = agent._current_anchor()
    if anchor:
        failed_branches = sorted(list(anchor["branch_set"]["failed"]))
    messages = build_failure_summary_messages(
        screenshot=screenshot,
        image_size=agent.image_size,
        task_instruction=instruction,
        recent_steps=recent_steps,
        failed_branches=failed_branches,
        rollback_info=restart_event.get("rollback_info", ""),
    )
    response = agent.planner_llm(messages)
    summary = []
    try:
        data = _safe_json_loads(response)
        summary = data.get("summary", [])
    except Exception:
        logger.exception("Failed to parse failure summary JSON.")
        summary = []
    if summary:
        agent.record_failure_memory({"type": "restart_summary", "summary": summary})

    agent.global_step_count += 1
    agent.exec_log.append(
        {
            "step_id": agent.global_step_count,
            "anchor_id": None,
            "branch_id": None,
            "action": "RESTART",
            "action_type": "restart",
            "expected_change": "restart environment",
            "success": True,
            "failure_type": "",
            "reason": restart_event.get("reason", ""),
            "screenshot": "",
            "time": 0.0,
            "token": {},
            "is_rollback": False,
        }
    )

    agent.restart_counter += 1
    restart_event["restart_counter"] = agent.restart_counter
    agent.reset_for_restart()
    env.reset(task_config=example)
    time.sleep(60)


def _save_execution_log(
    example_result_dir: str,
    result: float,
    step_idx: int,
    execution_time: float,
    agent: ScreenshotBranchAgent,
    args: argparse.Namespace,
    example: Dict[str, Any],
    restart_events: List[Dict[str, Any]],
) -> None:
    llms = [agent.planner_llm]
    if agent.grounder_llm:
        llms.append(agent.grounder_llm)
    if agent.checker_llm and agent.checker_llm is not agent.planner_llm:
        llms.append(agent.checker_llm)

    total_prompt_tokens = sum(llm.total_prompt_tokens for llm in llms)
    total_completion_tokens = sum(llm.total_completion_tokens for llm in llms)
    total_tokens = sum(llm.total_tokens for llm in llms)
    total_image_count = sum(llm.total_image_count for llm in llms)

    rollback_rate = agent.rollback_count / max(step_idx, 1)
    restart_rate = agent.restart_counter / max(step_idx, 1)

    model_usage = {
        "planner": {
            "model_name": args.planner_model,
            "prompt_tokens": agent.planner_llm.total_prompt_tokens,
            "completion_tokens": agent.planner_llm.total_completion_tokens,
            "image_count": agent.planner_llm.total_image_count,
        }
    }
    if agent.grounder_llm:
        model_usage["grounder"] = {
            "model_name": args.grounder_model,
            "prompt_tokens": agent.grounder_llm.total_prompt_tokens,
            "completion_tokens": agent.grounder_llm.total_completion_tokens,
            "image_count": agent.grounder_llm.total_image_count,
        }
    if agent.checker_llm and agent.checker_llm is not agent.planner_llm:
        model_usage["checker"] = {
            "model_name": args.checker_model,
            "prompt_tokens": agent.checker_llm.total_prompt_tokens,
            "completion_tokens": agent.checker_llm.total_completion_tokens,
            "image_count": agent.checker_llm.total_image_count,
        }

    execution_log = {
        "statistics": {
            "score": result,
            "total_steps": step_idx,
            "execution_time": execution_time,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "image_count": total_image_count,
            "rollback_count": agent.rollback_count,
            "rollback_rate": rollback_rate,
            "anchor_backtrack_count": agent.anchor_backtrack_count,
            "restart_count": agent.restart_counter,
            "restart_rate": restart_rate,
            "model_usage": model_usage,
        },
        "task_config": example,
        "additional_context": "",
        "exec_log": agent.exec_log,
        "failure_memory": agent.failure_memory,
        "restart_events": restart_events,
        "ablations": {
            "wo_restart": args.wo_restart,
            "wo_pruning": args.wo_pruning,
            "wo_multi_branch": args.wo_multi_branch,
        },
    }

    with open(os.path.join(example_result_dir, "execution_log.json"), "w", encoding="utf-8") as f:
        json.dump(execution_log, f, indent=2, ensure_ascii=False)


def get_unfinished(target_dir, total_file_json, rerun=False, rerun_fail=False):
    if not os.path.exists(target_dir):
        return total_file_json
    tasks_to_run = {}
    for domain in total_file_json:
        tasks_to_run[domain] = []
        for example_id in total_file_json[domain]:
            example_dir = os.path.join(target_dir, domain, example_id)
            result_path = os.path.join(example_dir, "result.txt")
            err_reason_path = os.path.join(example_dir, "err_reason.txt")
            should_skip = False
            if not rerun and os.path.exists(result_path) and not os.path.exists(err_reason_path):
                try:
                    with open(result_path, "r") as f:
                        result = float(f.read().strip())
                    if result > 0.0 or not rerun_fail:
                        should_skip = True
                except (ValueError, IOError):
                    should_skip = False
            if not should_skip:
                tasks_to_run[domain].append(example_id)
    tasks_to_run = {k: v for k, v in tasks_to_run.items() if v}
    return tasks_to_run


def main() -> None:
    global logger
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = config()

    result_name = os.path.basename(args.result_dir)
    logger = setup_logger(result_name, args.log_level)
    import run_autoglm_v
    run_autoglm_v.logger = logger

    with open(args.test_all_meta_path, "r", encoding="utf-8") as f:
        test_all_meta = json.load(f)
    test_all_meta = get_unfinished(args.result_dir, test_all_meta, args.rerun, args.rerun_fail)

    planner_llm = LLMCaller(
        model=args.planner_model,
        temperature=args.planner_temperature,
        top_p=args.planner_top_p,
        max_tokens=args.planner_max_tokens,
        repetition_penalty=args.planner_repetition_penalty,
    )
    grounder_llm = None
    if args.grounder_model and args.grounder_model != args.planner_model:
        grounder_llm = LLMCaller(
            model=args.grounder_model,
            temperature=args.planner_temperature,
            top_p=args.planner_top_p,
            max_tokens=args.planner_max_tokens,
            repetition_penalty=args.planner_repetition_penalty,
        )
    checker_llm = None
    if args.checker_model:
        checker_llm = LLMCaller(
            model=args.checker_model,
            temperature=args.planner_temperature,
            top_p=args.planner_top_p,
            max_tokens=args.planner_max_tokens,
            repetition_penalty=args.planner_repetition_penalty,
        )

    env = DesktopEnv(
        provider_name=args.provider_name,
        region=os.environ.get("AWS_REGION", "us-east-1"),
        client_password=args.client_password,
        path_to_vm=args.path_to_vm,
        action_space=args.action_space,
        screen_size=(args.screen_width, args.screen_height),
        headless=args.headless,
        os_type="Ubuntu",
        require_a11y_tree=False,
    )

    agent = ScreenshotBranchAgent(
        planner_llm=planner_llm,
        grounder_llm=grounder_llm,
        checker_llm=checker_llm,
        screen_size=(args.screen_width, args.screen_height),
        image_size=(args.image_width, args.image_height),
        max_branches=args.max_branches,
        max_grounding_retries=args.max_grounding_retries,
        restart_limit=args.restart_limit,
        use_vlm_success_check=not args.disable_vlm_success_check,
        wo_restart=args.wo_restart,
        wo_pruning=args.wo_pruning,
        wo_multi_branch=args.wo_multi_branch,
        max_parse_retries=args.max_parse_retries,
    )

    scores = []
    for domain in test_all_meta:
        for example_id in test_all_meta[domain]:
            config_file = os.path.join(args.test_config_base_dir, f"{domain}/{example_id}.json")
            with open(config_file, "r", encoding="utf-8") as f:
                example = json.load(f)
            instruction = example["instruction"]
            logger.info(f"[Domain]: {domain} [Example ID]: {example_id}")
            logger.info(f"[Instruction]: {instruction}")

            example_result_dir = os.path.join(args.result_dir, domain, example_id)
            if args.rerun or args.rerun_fail:
                if os.path.exists(example_result_dir):
                    logger.info(f"Removing old results for {domain}/{example_id}")
                    shutil.rmtree(example_result_dir)
            os.makedirs(example_result_dir, exist_ok=True)

            try:
                run_single_example_branch(
                    agent=agent,
                    env=env,
                    example=example,
                    instruction=instruction,
                    args=args,
                    example_result_dir=example_result_dir,
                    scores=scores,
                )
            except Exception as e:
                logger.error(f"Exception in {domain}/{example_id}: {e}", exc_info=True)
                with open(os.path.join(example_result_dir, "result.txt"), "w") as f:
                    f.write("0.0\n")
                scores.append(0.0)

    env.close()
    if scores:
        logger.info(f"Average score: {sum(scores) / len(scores)}")
    else:
        logger.info("No tasks completed")


if __name__ == "__main__":
    main()
