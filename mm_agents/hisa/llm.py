import requests
import os
import time
import threading
from functools import wraps
from PIL import Image, ImageDraw
import json
import base64
import io
from dataclasses import dataclass
from typing import Any, Tuple, Optional, List, Dict
# from configs.config import OPENAI_API_KEY, ROAD2ALL_API_KEY, OPENROUTER_API_KEY
from openai import OpenAI
import logging
import sys
import math
import re
import ast
from io import BytesIO
from utils import smart_resize

# Fix numpy import issue in CUDA environment
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"

GTA1_SYSTEM_PROMPT = '''You are an expert UI element locator. Given a GUI image and a user's element description, provide the coordinates of the specified element as a single (x,y) point. The image resolution is height {resized_height} and width {resized_width}. For elements with area, return the center point.

Output the coordinate pair exactly:
(x,y)'''

UITARS_SYSTEM_PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
```
Thought: ...
Action: ...
```

## Action Space
click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
hotkey(key='')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.

## Note
- Use English in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part. Your response MUST contain "Action:"

## User Instruction
{instruction}
"""

@dataclass
class UsageStats:
    """Unified usage statistics data structure"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    image_count: int = 0
    
    def add(self, other: 'UsageStats'):
        """Accumulate statistics from another UsageStats object"""
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.image_count += other.image_count


@dataclass
class ModelConfig:
    """Complete configuration for a model including client and pricing"""
    client_class: str  # Name of the API client class
    real_model_name: str  # Actual model name used by the API
    prompt_price: float  # per 1M tokens
    completion_price: float  # per 1M tokens
    
    def calculate_cost(self, stats: UsageStats) -> float:
        """Calculate total cost based on usage statistics"""
        token_cost = (
            self.prompt_price * stats.prompt_tokens +
            self.completion_price * stats.completion_tokens
        ) / 1_000_000
        return token_cost


# Unified model configuration table
MODEL_CONFIGS = {
    "gpt-4o": ModelConfig("OpenAIAPI", "gpt-4o", 2.5, 10),
    "o3": ModelConfig("OpenAIAPI", "o3", 2, 8),
    "o3-2025-04-16": ModelConfig("OpenAIAPI", "o3-2025-04-16", 2, 8),
    "o4-mini": ModelConfig("OpenAIAPI", "o4-mini", 1.1, 4.4),
    "o4-mini-2025-04-16": ModelConfig("OpenAIAPI", "o4-mini-2025-04-16", 1.1, 4.4),
    "gpt-5.2": ModelConfig("OpenAIAPI", "gpt-5.2", 1.75, 14),
    # "gpt-5": ModelConfig("OpenAIAPI", "gpt-5", 1.25, 10),
    "gpt-5": ModelConfig("Road2allAPI", "gpt-5", 1.25, 10),
    "gpt-5-2025-08-07": ModelConfig("OpenAIAPI", "gpt-5-2025-08-07", 1.25, 10),
    # "gpt-5-mini": ModelConfig("OpenAIAPI", "gpt-5-mini", 0.25, 2),
    "gpt-5-mini": ModelConfig("Road2allAPI", "gpt-5-mini", 0.25, 2),
    "gpt-5-mini-2025-08-07": ModelConfig("OpenAIAPI", "gpt-5-mini-2025-08-07", 0.25, 2),
    "computer-use-preview": ModelConfig("OpenAIAPI", "computer-use-preview", 3, 12),
    "claude-3.5": ModelConfig("Road2allAPI", "claude-3-5-sonnet-20240620", 3, 15),
    "claude-3-7-sonnet-20250219": ModelConfig("Road2allAPI", "claude-3-7-sonnet-20250219", 3, 15),
    "claude-4": ModelConfig("Road2allAPI", "claude-sonnet-4-20250514", 3, 15),
    "claude-4.5": ModelConfig("OpenRouterAPI", "anthropic/claude-sonnet-4.5", 3, 15),
    "deepseek-v3": ModelConfig("Road2allAPI", "deepseek-v3", 0.24, 0.84),
    "deepseek-r1": ModelConfig("Road2allAPI", "deepseek-reasoner", 0.4, 1.75),
    "deepseek-v3.1": ModelConfig("OpenRouterAPI", "deepseek/deepseek-chat-v3.1:free", 0.2, 0.8),
    "qwen2.5-vl-72b": ModelConfig("OpenRouterAPI", "qwen/qwen2.5-vl-72b-instruct", 0.07, 0.28),
    "qwen3": ModelConfig("OpenRouterAPI", "qwen/qwen3-235b-a22b:free", 0, 0),
    "gemini-2.5-pro": ModelConfig("Road2allAPI", "gemini-2.5-pro-preview-05-06", 1.25, 10),
    "gemini-2.5-flash": ModelConfig("OpenRouterAPI", "google/gemini-2.5-flash", 0.3, 2.5),
    "uitars-1.5-7b": ModelConfig("LocalLLM", "uitars-1.5-7b", 0, 0),
    "gta1-7b": ModelConfig("LocalLLM", "gta1-7b", 0, 0),
    "qwen3.5-9b": ModelConfig("LocalLLM", "qwen3.5-9b", 0, 0),
}


def resize_image(img, max_size=1024):
    """Resize image to meet API constraints"""
    width, height = img.size
    max_edge = max(width, height)

    if max_edge > max_size:
        ratio = max_size / max_edge
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        img = img.resize((new_width, new_height))
    return img


def encode_image(image) -> str:
    """Encode PIL image or bytes to base64 string"""
    # Handle bytes input
    if isinstance(image, bytes):
        image = Image.open(io.BytesIO(image))

    # Convert to RGB if needed
    image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_base64}"


def count_images_in_messages(messages: list) -> int:
    """Count the number of images in message list"""
    count = 0
    for message in messages:
        if isinstance(message.get("content"), list):
            for content in message["content"]:
                if content.get("type") in ["image_url", "image", "input_image"]:
                    count += 1
    return count


def with_timeout(timeout_seconds):
    """Timeout decorator compatible with Windows"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e

            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout_seconds)

            if thread.is_alive():
                raise TimeoutError(
                    f"Function call timed out after {timeout_seconds} seconds"
                )

            if exception[0]:
                raise exception[0]

            return result[0]

        return wrapper
    return decorator


class MessageFormatter:
    """Base class for message format adaptation"""
    
    @staticmethod
    def normalize_messages(messages: List[Dict]) -> List[Dict]:
        """
        Normalize messages to a standard format (OpenAI style)
        This is the format we use internally across all APIs:
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "..."},
                {"type": "input_image", "image_url": "data:image/png;base64,..."}
            ]
        }
        """
        return messages
    
    @staticmethod
    def format_for_api(messages: List[Dict]) -> Any:
        """
        Convert normalized messages to API-specific format
        Must be implemented by subclasses
        """
        raise NotImplementedError

class OpenRouterMessageFormatter(MessageFormatter):
    """OpenRouter API message formatter - converts OpenAI format to standard format"""
    
    @staticmethod
    def format_for_api(messages: List[Dict]) -> List[Dict]:
        """
        Convert from OpenAI format to OpenRouter format:
        OpenAI format (input):
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "..."},
                {"type": "input_image", "image_url": "data:image/png;base64,..."}
            ]
        }
        
        OpenRouter format (output):
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "..."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
            ]
        }
        """
        formatted = []
        for msg in messages:
            formatted_msg = {"role": msg["role"]}
            
            if isinstance(msg["content"], list):
                # Convert content items to OpenRouter format
                content_list = []
                for item in msg["content"]:
                    if item.get("type") == "input_text":
                        # Convert "input_text" to "text"
                        content_list.append({
                            "type": "text",
                            "text": item["text"]
                        })
                    elif item.get("type") == "input_image":
                        # Convert "input_image" to "image_url" with nested structure
                        content_list.append({
                            "type": "image_url",
                            "image_url": {"url": item["image_url"]}
                        })
                    elif item.get("type") == "text":
                        # Already in correct format
                        content_list.append(item)
                    elif item.get("type") == "image_url":
                        # Already in correct format
                        content_list.append(item)
                    else:
                        # Keep other types as-is (fallback)
                        content_list.append(item)
                
                formatted_msg["content"] = content_list
            else:
                # Simple text message - keep as string
                formatted_msg["content"] = msg["content"]
            
            formatted.append(formatted_msg)
        
        return formatted


class Road2allMessageFormatter(MessageFormatter):
    """Road2all API message formatter - converts OpenAI format to standard format"""
    
    @staticmethod
    def format_for_api(messages: List[Dict]) -> List[Dict]:
        """
        Convert from OpenAI format to Road2all format:
        OpenAI format (input):
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "..."},
                {"type": "input_image", "image_url": "data:image/png;base64,..."}
            ]
        }
        
        Road2all format (output):
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "..."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
            ]
        }
        """
        formatted = []
        for msg in messages:
            formatted_msg = {"role": msg["role"]}
            
            if isinstance(msg["content"], list):
                # Convert content items to Road2all format
                content_list = []
                for item in msg["content"]:
                    if item.get("type") == "input_text":
                        # Convert "input_text" to "text"
                        content_list.append({
                            "type": "text",
                            "text": item["text"]
                        })
                    elif item.get("type") == "input_image":
                        # Convert "input_image" to "image_url" with nested structure
                        content_list.append({
                            "type": "image_url",
                            "image_url": {"url": item["image_url"]}
                        })
                    elif item.get("type") == "text":
                        # Already in correct format
                        content_list.append(item)
                    elif item.get("type") == "image_url":
                        # Already in correct format
                        content_list.append(item)
                    else:
                        # Keep other types as-is (fallback)
                        content_list.append(item)
                
                formatted_msg["content"] = content_list
            else:
                # Simple text message - keep as string
                formatted_msg["content"] = msg["content"]
            
            formatted.append(formatted_msg)
        
        return formatted


class BaseLLMClient:
    """Base class for LLM clients, defining unified interface"""
    
    def __init__(self, model_name: str, temperature: float = 0):
        self.model_name = model_name
        self.temperature = temperature
        self.usage_stats = UsageStats()
        self.logger = None
        # Set default formatter, will be overridden by subclasses
        self.message_formatter = MessageFormatter()
    
    def format_messages(self, messages: List[Dict]) -> Any:
        """
        Format messages for this API
        Uses the message_formatter to convert to API-specific format
        """
        return self.message_formatter.format_for_api(messages)
    
    def __call__(self, messages: list, enable_thinking: bool = False) -> str:
        """Send messages and return response"""
        raise NotImplementedError
    
    def call_cua(
        self,
        instruction: str,
        image: Image.Image,
        environment: str = "linux",
        screen_width: int = 1920,
        screen_height: int = 1080,
    ) -> Tuple[str, str]:
        """
        Computer Use API call (if supported)
        Workflow: build messages -> __call__ (get raw response) -> parse_cua_response (convert to pyautogui)

        Args:
            instruction: Text instruction describing the task
            image: PIL Image object
            environment: OS environment (linux/windows/mac)
            screen_width: Screen width
            screen_height: Screen height

        Returns:
            (py_cmd, reasoning): pyautogui code and reasoning text
        """
        # Build messages from instruction and image
        messages = self._build_cua_messages(instruction, image, screen_width, screen_height)

        # Use retry logic by default for general LLM models
        return self._call_cua_with_retry(messages, screen_width=screen_width, screen_height=screen_height)

    def _build_cua_messages(
        self,
        instruction: str,
        image: Image.Image,
        screen_width: int,  # Not used in base but subclasses may need it
        screen_height: int  # Not used in base but subclasses may need it
    ) -> list:
        """
        Build messages for CUA call
        Default implementation: simple user message with text and image
        Subclasses can override for model-specific message formatting

        Args:
            instruction: Text instruction
            image: PIL Image
            screen_width: Screen width (may be used by subclasses)
            screen_height: Screen height (may be used by subclasses)

        Returns:
            List of messages in standard format
        """
        # Encode image to base64
        image_b64 = encode_image(image)

        # Build standard message format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": instruction},
                    {"type": "input_image", "image_url": image_b64}
                ],
            }
        ]

        return messages

    def _call_cua_with_retry(self, messages: list, max_retries: int = 3, **kwargs) -> Tuple[str, str]:
        """
        Call CUA with retry if python code block not found
        Subclasses can override call_cua to skip retry logic if needed
        """
        raw_response = None
        for attempt in range(max_retries):
            raw_response = self(messages)

            if "```python" in raw_response:
                py_cmd, reasoning = self.parse_cua_response(raw_response, messages, **kwargs)
                return py_cmd, reasoning
            else:
                if self.logger:
                    self.logger.info(f"Invalid response format, retrying ({attempt + 1}/{max_retries}): {raw_response}")
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "You must return a python code within ```python``` code block to execute the task."},
                    ],
                })

        # Last attempt failed, return raw response
        py_cmd, reasoning = self.parse_cua_response(raw_response, messages, **kwargs)
        return py_cmd, reasoning

    def parse_cua_response(
        self,
        raw_response: str,
        messages: list,  # Not used in base but subclasses may need it
        **kwargs  # Not used in base but subclasses may need it
    ) -> Tuple[str, str]:
        """
        Parse raw model response to pyautogui code (default implementation)
        Subclasses can override this for custom parsing logic

        Default behavior: Extract python code block from response
        Returns: (py_cmd, reasoning)

        Note: messages and kwargs not used in base class but kept for
        consistent interface with subclasses
        """
        # Default: extract python code block
        if "```python" in raw_response:
            py_cmd = raw_response.split("```python")[1].split("```")[0].strip()
            reasoning = raw_response
            return py_cmd, reasoning
        else:
            # No python code block found, return raw response as-is
            return raw_response, ""

    def get_usage_stats(self) -> UsageStats:
        """Get usage statistics"""
        return self.usage_stats
    
    def reset_stats(self):
        """Reset statistics"""
        self.usage_stats = UsageStats()


class OpenAIAPI(BaseLLMClient):
    """OpenAI API client"""

    def __init__(self, model_name: str, temperature: float = 0):
        super().__init__(model_name, temperature)
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        # Store if this is computer-use-preview model
        self.model_name = model_name
    
    def __call__(self, messages: list, enable_thinking: bool = False) -> str:
        response = self.client.responses.create(
            model=self.model_name,
            input=messages,
        )

        # Update statistics
        self.usage_stats.prompt_tokens += response.usage.input_tokens
        self.usage_stats.completion_tokens += response.usage.output_tokens
        self.usage_stats.image_count += count_images_in_messages(messages)

        return next(block.text for item in response.output if item.type=="message" for block in item.content if block.type=="output_text")

    def cua_to_pyautogui(self, action) -> str:
        """
        Convert OpenAI CUA action to pyautogui command.

        Args:
            action: CUA action object or dict

        Returns:
            String containing pyautogui command
        """
        def fld(key: str, default: Any = None) -> Any:
            return (
                action.get(key, default)
                if isinstance(action, dict)
                else getattr(action, key, default)
            )

        act_type = fld("type")
        if not isinstance(act_type, str):
            act_type = str(act_type).split(".")[-1]
        act_type = act_type.lower()

        if act_type in ["click", "double_click"]:
            button = fld("button", "left")
            if button in [1, "left"]:
                button = "left"
            elif button in [2, "middle"]:
                button = "middle"
            elif button in [3, "right"]:
                button = "right"

            if act_type == "click":
                return f"pyautogui.click({fld('x')}, {fld('y')}, button='{button}')"
            if act_type == "double_click":
                return f"pyautogui.doubleClick({fld('x')}, {fld('y')}, button='{button}')"

        if act_type == "scroll":
            if fld("scroll_y", 0) != 0:
                return f"pyautogui.scroll({-fld('scroll_y', 0) / 100}, x={fld('x', 0)}, y={fld('y', 0)})"
            return ""

        if act_type == "drag":
            path = fld("path", [{"x": 0, "y": 0}, {"x": 0, "y": 0}])
            cmd = f"pyautogui.moveTo({path[0]['x']}, {path[0]['y']}, _pause=False); "
            cmd += f"pyautogui.dragTo({path[1]['x']}, {path[1]['y']}, duration=0.5, button='left')"
            return cmd

        if act_type == "move":
            return f"pyautogui.moveTo({fld('x')}, {fld('y')})"

        if act_type == "keypress":
            keys = fld("keys", []) or [fld("key")]
            if len(keys) == 1:
                return f"pyautogui.press('{keys[0].lower()}')"
            else:
                return "pyautogui.hotkey('{}')".format("', '".join(keys)).lower()

        if act_type == "type":
            text = str(fld("text", ""))
            return "pyautogui.typewrite({:})".format(repr(text))

        return "WAIT"

    def parse_cua_response(self, response) -> Tuple[str, str, str, str]:
        """
        Parse OpenAI Computer Use API response.

        Args:
            response: Response object from OpenAI CUA API

        Returns:
            Tuple of (unify_output, py_cmd, reasoning, message_text)
        """
        reasoning = ""
        py_cmd = ""
        message_text = ""

        for output_item in response.output:
            output_type = (
                output_item.get("type", "")
                if isinstance(output_item, dict)
                else getattr(output_item, "type", "")
            )

            if "computer_call" in str(output_type):
                action_call = (
                    output_item
                    if isinstance(output_item, dict)
                    else output_item.model_dump()
                )
                py_cmd = self.cua_to_pyautogui(action_call["action"])

            elif (
                "reasoning" in str(output_type)
                and hasattr(output_item, "summary")
                and len(output_item.summary) > 0
            ):
                reasoning = output_item.summary[0].text

            elif "message" in str(output_type) or "output_text" in str(output_type):
                if hasattr(output_item, "content") and len(output_item.content) > 0:
                    message_text = (
                        output_item.content[0].text
                        if hasattr(output_item.content[0], "text")
                        else str(output_item.content[0])
                    )
                else:
                    message_text = str(output_item)

        unify_output = ""
        if py_cmd:
            unify_output = py_cmd
        elif message_text:
            unify_output = message_text
        elif reasoning:
            unify_output = reasoning
        return unify_output, py_cmd, reasoning, message_text

    def _build_cua_messages(
        self,
        instruction: str,
        image: Image.Image,
        screen_width: int,
        screen_height: int
    ) -> list:
        """
        Build messages for CUA call
        For computer-use-preview: uses custom prompt for pyautogui code generation
        For other models: uses base class default
        """
        if self.model_name == 'computer-use-preview':
            # Computer-use-preview: custom prompt for pyautogui generation
            system_prompt = f"""You are an agent that performs desktop computer tasks as instructed.
You will receive a screenshot and predict the action based on the image.

Use `pyautogui` to perform actions. DO NOT use `pyautogui.locateCenterOnScreen` or `pyautogui.screenshot()`.
You must specify coordinates yourself based on the screenshot - be careful to ensure accuracy.

Return code in a code block:
```python
# your code here
```

Analyze the screenshot carefully and return the code."""

            # Encode image
            image_b64 = encode_image(image)

            # Build messages with system prompt
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": instruction},
                        {"type": "input_image", "image_url": image_b64}
                    ],
                }
            ]

            return messages
        else:
            # Other OpenAI models: use base class default
            return super()._build_cua_messages(instruction, image, screen_width, screen_height)

    def call_cua(
        self,
        instruction: str,
        image: Image.Image,
        environment: str = "linux",
        screen_width: int = 1920,
        screen_height: int = 1080,
    ) -> Tuple[str, str]:
        """
        Call Computer Use API
        For computer-use-preview: uses OpenAI responses API with tool calling
        For other models: uses base class workflow
        """
        if self.model_name == 'computer-use-preview':
            # Build messages
            messages = self._build_cua_messages(instruction, image, screen_width, screen_height)

            # Decode the image to get actual dimensions
            image_b64 = encode_image(image)
            screenshot = Image.open(io.BytesIO(base64.b64decode(image_b64.replace('data:image/png;base64,',''))))
            actual_width, actual_height = screenshot.size

            # Call the responses API with computer_use_preview tool
            response = self.client.responses.create(
                model=self.model_name,
                tools=[
                    {
                        "type": "computer_use_preview",
                        "display_width": actual_width,
                        "display_height": actual_height,
                        "environment": environment,
                    }
                ],
                input=messages,
                reasoning={"summary": "concise"},
                tool_choice="required",
                truncation="auto",
            )

            # Update token usage statistics
            if hasattr(response, "usage"):
                self.usage_stats.prompt_tokens += response.usage.input_tokens
                self.usage_stats.completion_tokens += response.usage.output_tokens
                self.usage_stats.image_count += count_images_in_messages(messages)

            # Parse response
            unify_output, py_cmd, reasoning, message_text = self.parse_cua_response(response)

            # Return in expected format (py_cmd, reasoning)
            return py_cmd, reasoning
        else:
            # Other models: use base class workflow
            return super().call_cua(instruction, image, environment, screen_width, screen_height)


class Road2allAPI(BaseLLMClient):
    """Road2all API client"""
    
    def __init__(self, model_name: str, temperature: float = 0):
        super().__init__(model_name, temperature)
        self.url = "https://api2.road2all.com/v1/chat/completions"
        self.message_formatter = Road2allMessageFormatter()
    
    def __call__(self, messages: list, enable_thinking: bool = False) -> str:
        headers = {
            "Authorization": f"Bearer {ROAD2ALL_API_KEY}",
            "Content-Type": "application/json",
        }
        
        # Format messages for Road2all API
        formatted_messages = self.format_messages(messages)
        
        data = {
            "model": self.model_name,
            "temperature": self.temperature,
            "messages": formatted_messages,
        }
        
        response = requests.post(self.url, headers=headers, json=data)
        
        if response.status_code == 200:
            result_json = response.json()
            result = result_json["choices"][0]["message"]["content"]
            usage = result_json["usage"]
            
            # Update statistics
            self.usage_stats.prompt_tokens += usage["prompt_tokens"]
            self.usage_stats.completion_tokens += usage["completion_tokens"]
            self.usage_stats.image_count += count_images_in_messages(messages)
            
            return result
        else:
            raise Exception(response.json()["error"]["message"])


class OpenRouterAPI(BaseLLMClient):
    """OpenRouter API client"""
    
    def __init__(self, model_name: str, temperature: float = 0):
        super().__init__(model_name, temperature)
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        self.message_formatter = OpenRouterMessageFormatter()
    
    def __call__(self, messages: list, enable_thinking: bool = False) -> str:
        # Format messages for OpenRouter API
        formatted_messages = self.format_messages(messages)
        
        data = json.dumps({
            "model": self.model_name,
            "temperature": self.temperature,
            "messages": formatted_messages,
        })
        
        response = requests.post(self.url, headers=self.headers, data=data)
        
        if response.status_code == 200:
            result_json = response.json()
            if "choices" not in result_json:
                self.logger.error(f"Invalid response format: {result_json}")
                raise Exception("Invalid response format")
            
            result = result_json["choices"][0]["message"]["content"]
            usage = result_json["usage"]
            
            # Update statistics
            self.usage_stats.prompt_tokens += usage["prompt_tokens"]
            self.usage_stats.completion_tokens += usage["completion_tokens"]
            self.usage_stats.image_count += count_images_in_messages(messages)
            
            return result
        else:
            raise Exception(response.json()["error"]["message"])


class LocalLLM(BaseLLMClient):
    """Local LLM client for models like uitars-1.5-7b"""

    def __init__(self, model_name: str, temperature: float = 0):
        super().__init__(model_name, temperature)
        normalized_model_name = model_name.lower()
        if normalized_model_name == "gta1-7b":
            base_url = 'http://localhost:1234/v1'
        elif normalized_model_name == "uitars-1.5-7b":
            base_url = 'http://localhost:1235/v1'
        elif normalized_model_name == "qwen3.5-9b":
            base_url = 'http://localhost:30000/v1'
        else:
            raise Exception("model not support")
        api_key = os.getenv('OPENAI_API_KEY', os.getenv('UITARS_API_KEY', 'empty'))
        self.client = OpenAI(base_url=base_url, api_key=api_key)

        # For uitars-1.5-7b, use OpenAI-compatible local server
        if "uitars" in model_name.lower():
            # Get base_url and api_key from environment variables
            self.model_type = "uitars"

            # UITars specific settings
            self.max_pixels = 16384 * 28 * 28
            self.min_pixels = 100 * 28 * 28
            self.action_parse_res_factor = 1000
            self.image_factor = 28
            self.max_ratio = 200
        elif "gta1" in model_name.lower():
            # GTA1 also uses OpenAI-compatible API (same as UITars)
            self.model_type = "gta1"

            # GTA1 specific settings (same as UITars for qwen2.5 image encoder)
            self.max_pixels = 16384 * 28 * 28
            self.min_pixels = 100 * 28 * 28
            self.image_factor = 28  # patch_size * merge_size = 14 * 2
            self.max_ratio = 200
        elif "qwen3.5" in model_name.lower():
            self.model_type = "qwen3.5"
        else:
            raise ValueError(f"Local model {model_name} not supported yet")

    @staticmethod
    def _strip_thinking_content(text: str, enable_thinking: bool = True) -> str:
        """Drop the reasoning block only when thinking mode is enabled."""
        if not text:
            return text
        if not enable_thinking:
            return text.strip()
        think_end = text.rfind("</think>")
        if think_end == -1:
            return text.strip()
        return text[think_end + len("</think>"):].strip()

    def __call__(self, messages: list, enable_thinking: bool = False) -> str:
        """
        Call local model with messages
        For uitars and gta1, this returns the raw model response
        """
        if self.model_type == "uitars":
            # Convert messages to uitars format if needed
            formatted_messages = self._format_messages_for_uitars(messages)

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=formatted_messages,
                temperature=self.temperature,
                frequency_penalty=1,
            )

            # Update statistics from response
            if hasattr(response, 'usage') and response.usage:
                self.usage_stats.prompt_tokens += getattr(response.usage, 'prompt_tokens', 0)
                self.usage_stats.completion_tokens += getattr(response.usage, 'completion_tokens', 0)
            self.usage_stats.image_count += count_images_in_messages(messages)

            return response.choices[0].message.content.strip()
        elif self.model_type == "gta1":
            # GTA1 uses similar format as UITars
            formatted_messages = self._format_messages_for_uitars(messages)

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=formatted_messages,
                temperature=self.temperature,
            )

            # Update statistics from response
            if hasattr(response, 'usage') and response.usage:
                self.usage_stats.prompt_tokens += getattr(response.usage, 'prompt_tokens', 0)
                self.usage_stats.completion_tokens += getattr(response.usage, 'completion_tokens', 0)
            self.usage_stats.image_count += count_images_in_messages(messages)

            return response.choices[0].message.content.strip()
        elif self.model_type == "qwen3.5":
            formatted_messages = self._format_messages_for_uitars(messages)

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=formatted_messages,
                temperature=self.temperature,
                top_p=0.95,
                presence_penalty=1.5,
                extra_body={
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": enable_thinking},
                },
            )

            # Update statistics from response
            if hasattr(response, 'usage') and response.usage:
                self.usage_stats.prompt_tokens += getattr(response.usage, 'prompt_tokens', 0)
                self.usage_stats.completion_tokens += getattr(response.usage, 'completion_tokens', 0)
            self.usage_stats.image_count += count_images_in_messages(messages)

            return self._strip_thinking_content(
                response.choices[0].message.content,
                enable_thinking=enable_thinking,
            )
        else:
            raise NotImplementedError(f"Model type {self.model_type} not implemented")

    def call_cua(
        self,
        instruction: str,
        image: Image.Image,
        environment: str = "linux",
        screen_width: int = 1920,
        screen_height: int = 1080,
        scale: float = 1.0,
    ) -> Tuple[str, str]:
        """
        Call CUA for local models (uitars, gta1) - skips python code block validation
        
        Args:
            instruction: Text instruction
            image: PIL Image object
            environment: OS environment (linux/windows/mac)
            screen_width: Screen width
            screen_height: Screen height
            scale: Scale factor for image resize (only used for gta1, default=1.0)
        """
        # Build messages (pass scale for gta1)
        messages = self._build_cua_messages(instruction, image, screen_width, screen_height, scale=scale)

        # Call model directly (no retry logic for code block check)
        raw_response = self(messages)

        # Parse response using model-specific parser
        py_cmd, reasoning = self.parse_cua_response(
            raw_response, messages, screen_width, screen_height
        )

        return py_cmd, reasoning

    def _build_cua_messages(
        self,
        instruction: str,
        image: Image.Image,
        screen_width: int,
        screen_height: int,
        scale: float = 1.0
    ) -> list:
        """
        Build messages for CUA call (override for model-specific formatting)
        For GTA1: adds system prompt with image dimensions
        For UITars: uses base class default
        
        Args:
            scale: Scale factor for image resize (only used for gta1)
        """
        if self.model_type == "gta1":
            # GTA1 uses qwen2.5 image encoder, needs smart_resize preprocessing
            original_width, original_height = image.size
            
            # Apply scale if scale != 1
            if scale != 1.0:
                scaled_width = int(original_width * scale)
                scaled_height = int(original_height * scale)
                image = image.resize((scaled_width, scaled_height))
                current_width, current_height = scaled_width, scaled_height
            else:
                current_width, current_height = original_width, original_height

            # GTA1 uses same settings as UITars for qwen2.5
            # Note: GTA1 uses patch_size * merge_size = 14 * 2 = 28 as factor
            resized_height, resized_width = smart_resize(
                current_height,
                current_width,
                factor=self.image_factor,  # 28
                min_pixels=self.min_pixels,  # 100 * 28 * 28
                max_pixels=self.max_pixels,   # 16384 * 28 * 28
                max_ratio=self.max_ratio
            )

            # Resize image
            resized_image = image.resize((resized_width, resized_height))

            # Store scale factors for later coordinate scaling
            # Total scale = scale * (smart_resize scale)
            self._gta1_scale_x = original_width / resized_width
            self._gta1_scale_y = original_height / resized_height

            # System prompt uses resized dimensions (what model actually sees)
            system_prompt = GTA1_SYSTEM_PROMPT.format(resized_height=resized_height, resized_width=resized_width)

            # Encode resized image
            image_b64 = encode_image(resized_image)

            # Build messages with system prompt
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": instruction},
                        {"type": "input_image", "image_url": image_b64}
                    ],
                }
            ]

            return messages
        elif self.model_type == "uitars":
            prompt = UITARS_SYSTEM_PROMPT.format(instruction=instruction)
            messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt.format(instruction=instruction)},
                            {"type": "input_image", "image_url": encode_image(image)}
                        ],
                    }
                ]
            return messages
        else:
            return super()._build_cua_messages(instruction, image, screen_width, screen_height)

    def parse_cua_response(
        self,
        raw_response: str,
        messages: list,
        screen_width: int,
        screen_height: int
    ) -> Tuple[str, str]:
        """
        Parse raw model response to pyautogui code (post-processing for CUA models)

        Args:
            raw_response: Raw text output from model
            messages: Original messages (needed for image dimensions)
            screen_width: Screen width
            screen_height: Screen height

        Returns:
            (py_cmd, reasoning): pyautogui code and reasoning text
        """
        if self.model_type == "uitars":
            return self._parse_uitars_response(raw_response, messages, screen_width, screen_height)
        elif self.model_type == "gta1":
            return self._parse_gta1_response(raw_response, messages, screen_width, screen_height)
        else:
            raise NotImplementedError(f"CUA parsing for model type {self.model_type} not implemented")

    def _format_messages_for_uitars(self, messages: list) -> list:
        """Format messages for UITars model"""
        formatted = []
        for msg in messages:
            formatted_msg = {"role": msg["role"]}

            if isinstance(msg["content"], list):
                # Convert to UITars format
                content_list = []
                for item in msg["content"]:
                    if item.get("type") == "input_text":
                        content_list.append({
                            "type": "text",
                            "text": item["text"]
                        })
                    elif item.get("type") == "input_image":
                        # UITars expects image_url format
                        content_list.append({
                            "type": "image_url",
                            "image_url": {"url": item["image_url"]}
                        })
                    elif item.get("type") == "text":
                        content_list.append(item)
                    elif item.get("type") == "image_url":
                        content_list.append(item)
                    else:
                        content_list.append(item)

                formatted_msg["content"] = content_list
            else:
                formatted_msg["content"] = msg["content"]

            formatted.append(formatted_msg)

        return formatted
    
    def escape_single_quotes(self, text):
        pattern = r"(?<!\\)'"
        return re.sub(pattern, r"\\'", text)
    
    def parse_action(self, action_str):
        try:
            node = ast.parse(action_str, mode='eval')

            if not isinstance(node, ast.Expression):
                raise ValueError("Not an expression")

            call = node.body

            if not isinstance(call, ast.Call):
                raise ValueError("Not a function call")

            if isinstance(call.func, ast.Name):
                func_name = call.func.id
            elif isinstance(call.func, ast.Attribute):
                func_name = call.func.attr
            else:
                func_name = None

            kwargs = {}
            for kw in call.keywords:
                key = kw.arg
                if isinstance(kw.value, ast.Constant):
                    value = kw.value.value
                elif isinstance(kw.value, ast.Str):
                    value = kw.value.s
                else:
                    value = None
                kwargs[key] = value

            return {
                'function': func_name,
                'args': kwargs
            }

        except Exception as e:
            print(f"Failed to parse action '{action_str}': {e}")
            return None
 

    def parse_action_to_structure_output(self, text, factor, origin_resized_height, origin_resized_width, model_type, max_pixels=16384*28*28, min_pixels=100*28*28):
        text = text.strip()
        if model_type == "qwen25vl":
            smart_resize_height, smart_resize_width = smart_resize(origin_resized_height, origin_resized_width, factor=self.image_factor, min_pixels=min_pixels, max_pixels=max_pixels, max_ratio=self.max_ratio)

        if text.startswith("Thought:"):
            thought_pattern = r"Thought: (.+?)(?=\s*Action:|$)"
        elif text.startswith("Reflection:"):
            thought_pattern = r"Reflection: (.+?)Action_Summary: (.+?)(?=\s*Action:|$)"
        elif text.startswith("Action_Summary:"):
            thought_pattern = r"Action_Summary: (.+?)(?=\s*Action:|$)"
        else:
            thought_pattern = r"Thought: (.+?)(?=\s*Action:|$)"
        reflection, thought = None, None
        thought_match = re.search(thought_pattern, text, re.DOTALL)
        if thought_match:
            if len(thought_match.groups()) == 1:
                thought = thought_match.group(1).strip()
            elif len(thought_match.groups()) == 2:
                thought = thought_match.group(2).strip()
                reflection = thought_match.group(1).strip()
        assert "Action:" in text
        action_str = text.split("Action:")[-1]

        tmp_all_action = action_str.split("\n\n")
        all_action = []
        for action_str in tmp_all_action:
            if "type(content" in action_str:
                def escape_quotes(match):
                    content = match.group(1)
                    return content

                pattern = r"type\(content='(.*?)'\)"
                content = re.sub(pattern, escape_quotes, action_str)

                action_str = self.escape_single_quotes(content)
                action_str = "type(content='" + action_str + "')"
            all_action.append(action_str)

        parsed_actions = [self.parse_action(action.replace("\n","\\n").lstrip()) for action in all_action]
        actions = []
        for action_instance, raw_str in zip(parsed_actions, all_action):
            if action_instance == None:
                print(f"Action can't parse: {raw_str}")
                raise ValueError(f"Action can't parse: {raw_str}") 
            action_type = action_instance["function"]
            params = action_instance["args"]

            action_inputs = {}
            for param_name, param in params.items():
                if param == "": continue
                param = param.lstrip()
                action_inputs[param_name.strip()] = param
                
                if "start_box" in param_name or "end_box" in param_name:
                    ori_box = param
                    # Remove parentheses and split the string by commas
                    numbers = ori_box.replace("(", "").replace(")", "").split(",")

                    # Convert to float and scale by 1000
                    # Qwen2.5vl output absolute coordinates, qwen2vl output relative coordinates
                    if model_type == "qwen25vl":
                        float_numbers = []
                        for num_idx, num in enumerate(numbers):
                            num = float(num)
                            if (num_idx + 1) % 2 == 0:
                                float_numbers.append(float(num/smart_resize_height))
                            else:
                                float_numbers.append(float(num/smart_resize_width))
                    else:
                        float_numbers = [float(num) / factor for num in numbers]

                    if len(float_numbers) == 2:
                        float_numbers = [float_numbers[0], float_numbers[1], float_numbers[0], float_numbers[1]]
                    action_inputs[param_name.strip()] = str(float_numbers)

            # import pdb; pdb.set_trace()
            actions.append({
                "reflection": reflection,
                "thought": thought,
                "action_type": action_type,
                "action_inputs": action_inputs,
                "text": text
            })
        return actions
    
    
    def linear_resize(self, height: int, width: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
        if width * height > max_pixels:
            resize_factor = math.sqrt(max_pixels / (width * height))
            width, height = int(width * resize_factor), int(height * resize_factor)
        if width * height < min_pixels:
            resize_factor = math.sqrt(min_pixels / (width * height))
            width, height = math.ceil(width * resize_factor), math.ceil(height * resize_factor)

        return height, width 


    def parsing_response_to_pyautogui_code(self, responses, image_height: int, image_width:int, input_swap:bool=False) -> str:
        """Parse model output to pyautogui code.

        Args:
            responses: Model output dict with action_type and action_inputs
            image_height: Screenshot height
            image_width: Screenshot width
            input_swap: Whether to use clipboard for typing

        Returns:
            Generated pyautogui code string
        """

        pyautogui_code = f"import pyautogui\nimport time\n"
        if isinstance(responses, dict):
            responses = [responses]
        for response_id, response in enumerate(responses):
            if "observation" in response:
                observation = response["observation"]
            else:
                observation = ""

            if "thought" in response:
                thought = response["thought"]
            else:
                thought = ""
            
            if response_id == 0:
                pyautogui_code += f"'''\nObservation:\n{observation}\n\nThought:\n{thought}\n'''\n"
            else:
                pyautogui_code += f"\ntime.sleep(1)\n"

            action_dict = response
            action_type = action_dict.get("action_type")
            action_inputs = action_dict.get("action_inputs", {})
            
            if action_type == "hotkey":
                # Parsing hotkey action
                if "key" in action_inputs:
                    hotkey = action_inputs.get("key", "")
                else:
                    hotkey = action_inputs.get("hotkey", "")

                if hotkey == "arrowleft":
                    hotkey = "left"

                elif hotkey == "arrowright":
                    hotkey = "right"
                
                elif hotkey == "arrowup":
                    hotkey = "up"
                
                elif hotkey == "arrowdown":
                    hotkey = "down"

                if hotkey:
                    # Handle other hotkeys
                    keys = hotkey.split()  # Split the keys by space
                    convert_keys = []
                    for key in keys:
                        if key == "space":
                            key = ' '
                        convert_keys.append(key)
                    pyautogui_code += f"\npyautogui.hotkey({', '.join([repr(k) for k in convert_keys])})"
            
            elif action_type == "press":
                # Parsing press action
                if "key" in action_inputs:
                    key_to_press = action_inputs.get("key", "")
                else:
                    key_to_press = action_inputs.get("press", "")

                if hotkey == "arrowleft":
                    hotkey = "left"

                elif hotkey == "arrowright":
                    hotkey = "right"
                
                elif hotkey == "arrowup":
                    hotkey = "up"
                
                elif hotkey == "arrowdown":
                    hotkey = "down"
                
                elif hotkey == "space":
                    hotkey = " "
                    
                if key_to_press:
                    # Simulate pressing a single key
                    pyautogui_code += f"\npyautogui.press({repr(key_to_press)})"
                
            elif action_type == "keyup":
                key_to_up = action_inputs.get("key", "")
                pyautogui_code += f"\npyautogui.keyUp({repr(key_to_up)})"
            
            elif action_type == "keydown":
                key_to_down = action_inputs.get("key", "")
                pyautogui_code += f"\npyautogui.keyDown({repr(key_to_down)})"

            elif action_type == "type":
                # Parsing typing action using clipboard
                content = action_inputs.get("content", "")
                content = self.escape_single_quotes(content)
                stripped_content = content
                if content.endswith("\n") or content.endswith("\\n"):
                    stripped_content = stripped_content.rstrip("\\n").rstrip("\n")
                if content:
                    if input_swap:
                        pyautogui_code += f"\nimport pyperclip"
                        pyautogui_code += f"\npyperclip.copy('{stripped_content}')"
                        pyautogui_code += f"\npyautogui.hotkey('ctrl', 'v')"
                        pyautogui_code += f"\ntime.sleep(0.5)\n"
                        if content.endswith("\n") or content.endswith("\\n"):
                            pyautogui_code += f"\npyautogui.press('enter')"
                    else:
                        pyautogui_code += f"\npyautogui.write('{stripped_content}', interval=0.1)"
                        pyautogui_code += f"\ntime.sleep(0.5)\n"
                        if content.endswith("\n") or content.endswith("\\n"):
                            pyautogui_code += f"\npyautogui.press('enter')"

            
            elif action_type in ["drag", "select"]:
                # Parsing drag or select action based on start and end_boxes
                start_box = action_inputs.get("start_box")
                end_box = action_inputs.get("end_box")
                if start_box and end_box:
                    x1, y1, x2, y2 = eval(start_box)  # Assuming box is in [x1, y1, x2, y2]
                    sx = round(float((x1 + x2) / 2) * image_width, 3)
                    sy = round(float((y1 + y2) / 2) * image_height, 3)
                    x1, y1, x2, y2 = eval(end_box)  # Assuming box is in [x1, y1, x2, y2]
                    ex = round(float((x1 + x2) / 2) * image_width, 3)
                    ey = round(float((y1 + y2) / 2) * image_height, 3)
                    pyautogui_code += (
                        f"\npyautogui.moveTo({sx}, {sy})\n"
                        f"\npyautogui.dragTo({ex}, {ey}, duration=1.0)\n"
                    )

            elif action_type == "scroll":
                # Parsing scroll action
                start_box = action_inputs.get("start_box")
                if start_box:
                    x1, y1, x2, y2 = eval(start_box)  # Assuming box is in [x1, y1, x2, y2]
                    x = round(float((x1 + x2) / 2) * image_width, 3)
                    y = round(float((y1 + y2) / 2) * image_height, 3)
                else:
                    x = None
                    y = None
                direction = action_inputs.get("direction", "")
                
                if x == None:
                    if "up" in direction.lower():
                        pyautogui_code += f"\npyautogui.scroll(5)"
                    elif "down" in direction.lower():
                        pyautogui_code += f"\npyautogui.scroll(-5)"
                else:
                    if "up" in direction.lower():
                        pyautogui_code += f"\npyautogui.scroll(5, x={x}, y={y})"
                    elif "down" in direction.lower():
                        pyautogui_code += f"\npyautogui.scroll(-5, x={x}, y={y})"

            elif action_type in ["click", "left_single", "left_double", "right_single", "hover"]:
                # Parsing mouse click actions
                start_box = action_inputs.get("start_box")
                start_box = str(start_box)
                if start_box:
                    start_box = eval(start_box)
                    if len(start_box) == 4:
                        x1, y1, x2, y2 = start_box  # Assuming box is in [x1, y1, x2, y2]
                    elif len(start_box) == 2:
                        x1, y1 = start_box
                        x2 = x1
                        y2 = y1
                    x = round(float((x1 + x2) / 2) * image_width, 3)
                    y = round(float((y1 + y2) / 2) * image_height, 3)
                    if action_type == "left_single" or action_type == "click":
                        pyautogui_code += f"\npyautogui.click({x}, {y}, button='left')"
                    elif action_type == "left_double":
                        pyautogui_code += f"\npyautogui.doubleClick({x}, {y}, button='left')"
                    elif action_type == "right_single":
                        pyautogui_code += f"\npyautogui.click({x}, {y}, button='right')"
                    elif action_type == "hover":
                        pyautogui_code += f"\npyautogui.moveTo({x}, {y})"
            
            elif action_type in ["finished"]:
                pyautogui_code = f"DONE"
            
            else:
                pyautogui_code += f"\n# Unrecognized action type: {action_type}"

        return pyautogui_code

    def _parse_uitars_response(
        self,
        raw_response: str,
        messages: list,
        screen_width: int,
        screen_height: int
    ) -> Tuple[str, str]:
        """
        Parse UITars raw response to pyautogui code
        Returns: (py_cmd, reasoning)
        """
        # Extract image from messages to get dimensions
        obs_image_height, obs_image_width = screen_height, screen_width
        resized_height, resized_width = screen_height, screen_width

        for msg in messages:
            if isinstance(msg.get("content"), list):
                for item in msg["content"]:
                    if item.get("type") == "input_image":
                        # Decode base64 image to get dimensions
                        try:
                            img_url = item["image_url"]
                            if isinstance(img_url, dict):
                                img_url = img_url.get("url", "")

                            if img_url.startswith("data:image"):
                                # Extract base64 data
                                img_data = img_url.split(",")[1]
                                img_bytes = base64.b64decode(img_data)
                                img = Image.open(BytesIO(img_bytes))
                                obs_image_width, obs_image_height = img.size

                                # Apply UITars resizing
                                resized_height, resized_width = self.linear_resize(
                                    obs_image_height, obs_image_width,
                                    min_pixels=self.min_pixels,
                                    max_pixels=self.max_pixels
                                )
                                break
                        except Exception as e:
                            self.logger.warning(f"Failed to decode image: {e}")

        try:
            # Parse action to get structured output
            parsed_responses = self.parse_action_to_structure_output(
                raw_response,
                self.action_parse_res_factor,
                resized_height,
                resized_width,
                model_type="qwen25vl",  # UITars uses qwen25vl format
                max_pixels=self.max_pixels,
                min_pixels=self.min_pixels
            )

            # Convert to pyautogui code
            py_cmd = ""
            for parsed_response in parsed_responses:
                if "action_type" in parsed_response:
                    if parsed_response["action_type"] in ["finished", "wait", "error_env", "call_user"]:
                        # Special actions, no pyautogui code
                        continue

                    pyautogui_code = self.parsing_response_to_pyautogui_code(
                        parsed_response,
                        obs_image_height,
                        obs_image_width,
                        input_swap=False
                    )

                    if pyautogui_code and pyautogui_code != "DONE":
                        py_cmd += pyautogui_code + "\n"

            # Clean up py_cmd
            py_cmd = py_cmd.strip()

            # Extract reasoning (thought) if available
            reasoning = ""
            if parsed_responses and "thought" in parsed_responses[0]:
                reasoning = parsed_responses[0]["thought"] or ""

            return py_cmd, reasoning

        except Exception as e:
            self.logger.error(f"UITars response parsing failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return "", f"Error: {str(e)}"

    def _extract_coordinates(self, raw_string: str) -> Tuple[int, int]:
        """Extract coordinates from GTA1 model output"""
        try:
            matches = re.findall(r"\((-?\d*\.?\d+),\s*(-?\d*\.?\d+)\)", raw_string)
            # matches[0] is a tuple of two strings, convert to int
            return tuple(map(int, map(float, matches[0])))
        except:
            return 0, 0

    def _parse_gta1_response(
        self,
        raw_response: str,
        messages: list,
        screen_width: int,
        screen_height: int
    ) -> Tuple[str, str]:
        """
        Parse GTA1 raw response to pyautogui code
        GTA1 outputs coordinates in resized image space, need to scale back to original

        Returns: (py_cmd, reasoning)
        """
        try:
            # Extract coordinates from model output (in resized image space)
            pred_x, pred_y = self._extract_coordinates(raw_response)

            # Use stored scale factors from _build_cua_messages
            scale_x = getattr(self, '_gta1_scale_x', 1.0)
            scale_y = getattr(self, '_gta1_scale_y', 1.0)

            # Scale coordinates back to original image space
            scaled_x = int(pred_x * scale_x)
            scaled_y = int(pred_y * scale_y)
            py_cmd = (scaled_x, scaled_y)

            return py_cmd, ''

        except Exception as e:
            self.logger.error(f"GTA1 response parsing failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return "", f"Error: {str(e)}"


class AbstractLLM:
    """
    LLM abstraction layer providing unified interface and retry mechanism
    Automatically handles format adaptation for different API platforms
    """
    
    def __init__(self, model_name: str, temperature: float = 0.1, logger: logging.Logger = None):
        """
        Initialize LLM instance
        
        Args:
            model_name: Model name
            temperature: Sampling temperature
            logger: Logger instance
        """
        if logger is None:
            logger = logging.getLogger("default")
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)

        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Model {model_name} not supported")
        
        temperature = max(0.1, temperature)
        self.model_name = model_name
        self.timeout_seconds = 600
        
        # Get model configuration
        self.model_config = MODEL_CONFIGS[model_name]
        
        # Create client instance - each client has its own message formatter
        client_class = globals()[self.model_config.client_class]
        self.client = client_class(self.model_config.real_model_name, temperature)
        self.logger = logger
        self.client.logger = logger
    
    def __call__(self, messages: list, max_retries: int = 1, enable_thinking: bool = False) -> Optional[str]:
        """
        Call LLM with timeout and retry mechanism
        Messages are automatically formatted for the specific API platform
        
        Args:
            messages: List of messages in normalized format (OpenAI style)
            max_retries: Maximum number of retries
        
        Returns:
            LLM response or None (on failure)
        """
        for attempt in range(max_retries):
            try:
                @with_timeout(self.timeout_seconds)
                def call_llm():
                    # The client will automatically format messages using its formatter
                    return self.client(messages, enable_thinking=enable_thinking)
                
                response = call_llm()
                return response
            
            except TimeoutError:
                self.logger.error(f"Attempt {attempt + 1}/{max_retries}: LLM call timed out after {self.timeout_seconds} seconds")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5
                    self.logger.error(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
            
            except Exception as e:
                self.logger.error(f"Attempt {attempt + 1}/{max_retries}: LLM call failed with error: {e}")
                
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5
                    self.logger.error(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
        
        self.logger.info(f"All {max_retries} attempts failed")
        return None
    
    def call_cua(
        self,
        instruction: str,
        image: Image.Image,
        environment: str = "linux",
        screen_width: int = 1920,
        screen_height: int = 1080,
        scale: float = 1.0,
    ) -> Tuple[str, str]:
        """
        Call Computer Use API

        Args:
            instruction: Text instruction describing the task
            image: PIL Image object
            environment: OS environment (linux/windows/mac)
            screen_width: Screen width
            screen_height: Screen height
            scale: Scale factor for image resize (only used for gta1, default=1.0)

        Returns:
            (py_cmd, reasoning): pyautogui code and reasoning text
        """
        # Check if client supports scale parameter (LocalLLM only)
        if hasattr(self.client, 'call_cua'):
            import inspect
            sig = inspect.signature(self.client.call_cua)
            if 'scale' in sig.parameters:
                return self.client.call_cua(instruction, image, environment, screen_width, screen_height, scale=scale)
        return self.client.call_cua(instruction, image, environment, screen_width, screen_height)
    
    def get_usage(self) -> Tuple[float, int, int, int]:
        """
        Get cost and usage statistics
        
        Returns:
            (total_cost, prompt_tokens, completion_tokens, image_count)
        """
        stats = self.client.get_usage_stats()
        cost = self.model_config.calculate_cost(stats)
        return cost, stats.prompt_tokens, stats.completion_tokens, stats.image_count
    
    def reset_stats(self):
        """Reset usage statistics"""
        self.client.reset_stats()


def calculate_image_tokens():
    model_name = 'gpt-5'
    client = AbstractLLM(model_name)
    path = "data/servicenow_low.png"
    image = Image.open(path)
    # screen_width, screen_height = 1280, 720
    screen_width, screen_height = 1920, 1080
    image = image.resize((screen_width, screen_height))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "what is in the image?"},
                {"type": "input_image", "image_url": encode_image(image)}
            ]
        }
    ]
    response = client(messages)
    cost, prompt_tokens, completion_tokens, image_count = client.get_usage()
    print(f"Cost: {cost}")
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Completion tokens: {completion_tokens}")
    print(f"Image count: {image_count}")


if __name__ == "__main__":
    # calculate_image_tokens()

    client = AbstractLLM('qwen3.5-9b')
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "hi"},
            ]
        }
    ]
    # image = Image.open("data/screenshot.png")
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "input_text", "text": "where is the coordinate of the chrome browser?"},
    #             {"type": "input_image", "image_url": encode_image(image)},
    #         ]
    #     }
    # ]
    response = client(messages)
    print(response)

    # draw = ImageDraw.Draw(image)
    # draw.ellipse((0, 35, 40, 106), fill="red")
    # image.save('tmp/tmp.jpg')
    exit()


    run = 1
    x, y = 137, 548
    model_name = "gta1-7b" #"gta1-7b" #"uitars-1.5-7b"
    
    # instruction = "Click the funnel (filter) icon in the top menu bar — the middle icon directly under the page title 'servicenow' above the Incidents list."
    instruction = "CClick on the 'Discussion' link under the COMMUNITY section"
    path = "/data1/lwm/projects/ComputerRL/results/hisa_qwen3.5-9b_wo_step_refinement_pattern/chrome/a96b564e-dbe9-42c3-9ccf-b4498073438a/operations/step_35_gui_action.png"
    image = Image.open(path)
    screen_width, screen_height = 1280, 720
    image = image.resize((screen_width, screen_height))

    if run:
        client = AbstractLLM(model_name)
        py_cmd, reasoning = client.call_cua(instruction, image, environment="linux", screen_width=screen_width, screen_height=screen_height, scale=1)
        print("PyAutoGUI Code:")
        print(py_cmd)
        print("\nReasoning:")
        print(reasoning)
        print(client.get_usage())
    else:
        draw = ImageDraw.Draw(image)
        draw.ellipse((x, y, x+10, y+10), fill="red")
        image.save('tmp/tmp.jpg')
