"""Script to run end-to-end evaluation on the benchmark.
Utils and basic architecture credit to https://github.com/web-arena-x/webarena/blob/main/run.py.
"""

import argparse
import datetime
import json
import os
import math
import ast
import time
import requests
from tqdm import tqdm
import shutil
import textwrap
from desktop_env.desktop_env import MAX_RETRIES, DesktopEnv as DesktopEnvBase
from mm_agents.autoglm_v_recovery import AutoGLMAgent
from mm_agents.hisa.llm import AbstractLLM
from typing import Optional, Dict, Any
from utils import summary, setup_logger
import datetime
import json
import logging
import os
import time
from wrapt_timeout_decorator import *

logger = logging.getLogger("desktopenv.experiment")


# Almost deprecated since it's not multi-env, use run_multienv_*.py instead
logger = None  # Will be initialized in main

def _ensure_vm_resolution(env, width: int, height: int, logger: logging.Logger) -> None:
    script = textwrap.dedent(f"""
        import os
        import subprocess

        os.environ["DISPLAY"] = ":0"
        output = subprocess.check_output(
            "xrandr --query | awk '/ connected/{{print $1; exit}}'",
            shell=True,
            text=True
        ).strip()
        if not output:
            raise RuntimeError("No connected display output found")

        mode = "{width}x{height}"
        modes = subprocess.check_output("xrandr | awk '{{print $1}}'", shell=True, text=True).split()
        if mode in modes:
            subprocess.check_call(["xrandr", "--output", output, "--mode", mode])
        else:
            if subprocess.call("command -v cvt >/dev/null 2>&1", shell=True) != 0:
                raise RuntimeError("cvt not found; install x11-xserver-utils in the VM")
            cvt_out = subprocess.check_output(
                "cvt {width} {height}",
                shell=True,
                text=True
            ).splitlines()
            if len(cvt_out) < 2:
                raise RuntimeError("cvt output is invalid")
            parts = cvt_out[1].split()
            if len(parts) < 3 or parts[0] != "Modeline":
                raise RuntimeError("Unexpected cvt output: " + cvt_out[1])
            name = parts[1].strip('"')
            params = parts[2:]
            subprocess.call(["xrandr", "--newmode", name, *params])
            subprocess.call(["xrandr", "--addmode", output, name])
            subprocess.check_call(["xrandr", "--output", output, "--mode", name])
    """).strip()

    try:
        result = env.controller.run_python_script(script)
    except Exception as exc:
        raise SystemExit(f"Failed to set VM resolution: {exc}")

    if result and result.get("status") == "error":
        raise SystemExit(f"Failed to set VM resolution: {result.get('error')}")

    size = env.controller.get_vm_screen_size() or {}
    if size.get("width") != width or size.get("height") != height:
        raise SystemExit(
            f"VM resolution mismatch: got {size.get('width')}x{size.get('height')}, expected {width}x{height}"
        )

def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end evaluation on the benchmark")

    # environment config
    parser.add_argument("--path_to_vm", type=str)
    parser.add_argument(
        "--provider_name",
        type=str,
        default="docker",
        help="Virtualization provider (vmware, docker, aws, azure, gcp, virtualbox)",
    )
    parser.add_argument("--headless", action="store_true", default=True, help="Run in headless machine")
    parser.add_argument("--action_space", type=str, default="autoglm_computer_use", help="Action type")
    parser.add_argument(
        "--observation_type",
        choices=["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"],
        default="a11y_tree",
        help="Observation type",
    )# NOTE: Only supports "a11y_tree" and actually uses screenshot (with_image=True, with_atree=False)
    parser.add_argument("--screen_width", type=int, default=1280) #1920
    parser.add_argument("--screen_height", type=int, default=720) #1080
    parser.add_argument("--sleep_after_execution", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=50)

    # agent config
    parser.add_argument("--max_trajectory_length", type=int, default=3)
    parser.add_argument("--test_config_base_dir", type=str, default="evaluation_examples/examples")

    # lm config
    parser.add_argument("--model", type=str, default="autoglm-os")
    parser.add_argument("--temperature", type=float, default=0.2) # original: 0.2
    parser.add_argument("--top_p", type=float, default=0.1)  # original: 0.1
    parser.add_argument("--max_tokens", type=int, default=256) # original: 2048
    parser.add_argument("--repetition_penalty", type=float, default=1)  # original: 1
    parser.add_argument("--stop_token", type=str, default=None)
    parser.add_argument("--visual_grounder_model", type=str, default="autoglm-os",
                        help="Model for visual grounding (e.g., autoglm-os, gta1-7b)")
    parser.add_argument("--image_width", type=int, default=1280)
    parser.add_argument("--image_height", type=int, default=720)
    parser.add_argument("--recovery", action="store_true", default=True, help="Enable recovery/rollback mode")
    parser.add_argument("--recovery_max_resets", type=int, default=1, help="Max environment resets allowed per task")
    parser.add_argument("--recovery_max_steps", type=int, default=5, help="Max recovery steps before auto-reset")
    parser.add_argument("--recovery_max_failure_memory", type=int, default=8, help="Max failure items kept for replanning")
    parser.add_argument("--recovery_debug_force", action="store_true", help="Force recovery mode from the first step (debug)")

    # example config
    parser.add_argument("--domain", type=str, default="all")
    parser.add_argument("--test_all_meta_path", type=str, default="evaluation_examples/test_nogdrive.json")

    # aws config
    parser.add_argument(
        "--region", type=str, default="us-east-1", help="AWS region for the VM"
    )
    parser.add_argument(
        "--client_password", type=str, default="", help="Client password"
    )

    # logging related
    parser.add_argument("--result_dir", type=str, default="./results")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    
    # rerun related
    parser.add_argument("--rerun", action="store_true", help="Rerun all tasks (ignore existing results)")
    parser.add_argument("--rerun_fail", action="store_true", help="Rerun only failed tasks (score == 0)")
    
    args = parser.parse_args()

    return args


class DesktopEnv(DesktopEnvBase):
    def step(self, action, pause=2):
        self._step_no += 1
        self.action_history.append(action)
        
        # Mark environment as used when step is called
        self.is_environment_used = True

        reward = 0  # todo: Define reward calculation for each example
        done = False  # todo: Define episode termination condition for each example
        info = {}
        logger.info(f"[Step] {self._step_no} in trajectory {self._traj_no} with action: {action}")

        # handle the special actions
        if action in ['WAIT', 'FAIL', 'DONE']:
            if action == 'WAIT':
                time.sleep(pause)
                exe_result = 'Wait ' + str(pause) + ' seconds'
            elif action == 'FAIL':
                done = True
                info = {"fail": True}
                exe_result = 'Finish: fail'
            elif action == 'DONE':
                done = True
                info = {"done": True}
                exe_result = 'Finish: success'
        elif type(action) == dict:
            if action['action_type'] == 'OPEN_APP':
                self.setup_controller._launch_setup(action['parameters']['launch_app_command'], shell=True)
                exe_result = 'Open ' + action['parameters']['app_name']
            elif action['action_type'] == 'OPEN_CHROME_TAB':
                self.setup_controller._chrome_open_tabs_setup(action['parameters']['urls_to_open'])
                exe_result = 'Open ' + str(action['parameters']['urls_to_open']) + ' in Chrome successfully'
        else:
            # the set of all possible python commands insides `pyautogui`
            result = self.controller.execute_python_command(action)
            if result is None:
                exe_result = 'Error: Failed to execute command on server'
                logger.error(f"execute_python_command returned None for action: {action}")
            elif result.get('error'):
                exe_result = result['error'].strip()
            else:
                exe_result = result.get('output', '').strip()

        time.sleep(pause)
        observation = self._get_obs()
        observation['exe_result'] = exe_result
        
        return observation, reward, done, info

    def reset(self, task_config: Optional[Dict[str, Any]] = None, seed=None, options=None) -> Dict[str, Any]:
        # Reset to certain task in OSWorld
        logger.info("Resetting environment...")
        logger.info("Switching task...")
        logger.info("Setting counters...")
        self._traj_no += 1
        self._step_no = 0
        self.action_history.clear()

        for attempt in range(MAX_RETRIES):
            # Only revert to snapshot if environment has been used (step/setup)
            # This optimization is especially important for cloud providers like AWS
            # where unnecessary snapshot operations are costly and time-consuming
            
            if task_config is not None:
                # Only consider task proxy requirement if proxy is enabled at system level
                task_use_proxy = task_config.get("proxy", False) and self.enable_proxy
                if not self.enable_proxy and task_config.get("proxy", False):
                    logger.info("Task requires proxy but proxy is disabled at system level, ignoring proxy requirement.")
                
                if task_use_proxy != self.current_use_proxy:
                    # keep because get_info_from_website depend on this
                    self.current_use_proxy = task_use_proxy
            
            if self.is_environment_used:
                logger.info("Environment has been used, reverting to snapshot {}...".format(self.snapshot_name))
                self._revert_to_snapshot()
                logger.info("Starting emulator...")
                self._start_emulator()
                logger.info("Emulator started.")
                # Reset the usage flag after reverting
                self.is_environment_used = False
            else:
                logger.info("Environment is clean, skipping snapshot revert (provider: {}).".format(self.provider_name))

            if task_config is not None:
                if task_config.get("proxy", False) and self.enable_proxy:
                    # If using proxy and proxy is enabled, set up the proxy configuration
                    self.setup_controller._proxy_setup(self.client_password)
                self._set_task_info(task_config)
                self.setup_controller.reset_cache_dir(self.cache_dir)
                logger.info("Setting up environment...")
                success = self.setup_controller.setup(self.config, task_config.get("proxy", False) and self.enable_proxy)
                if success:
                    # Mark environment as used when setup is successfully executed
                    if self.config:  # Only mark as used if there were actual setup operations
                        self.is_environment_used = True
                    break
                else:
                    logger.error(
                        "Environment setup failed, retrying (%d/%d)...",
                        attempt + 1,
                        MAX_RETRIES,
                    )
                    time.sleep(5)
            else:
                break
            
        logger.info("Environment setup complete.")

        # Upload tools from autoglm_v_recovery package
        import mm_agents.autoglm_v_recovery
        tool_dir = os.path.join(os.path.dirname(mm_agents.autoglm_v_recovery.__file__), 'tools', 'package')
        for file in os.listdir(tool_dir):
            if os.path.isdir(os.path.join(tool_dir, file)):
                continue
            self.setup_controller._upload_file_setup([{
                "local_path": os.path.join(tool_dir, file),
                "path": os.path.join('~', file)
            }])

        # start soffice service for office tools
        self.setup_controller._launch_setup('soffice --accept="socket,host=localhost,port=2002;urp;" --norestore --nologo --nodefault', shell=True)
        time.sleep(5)

        observation = self._get_obs()
        return observation

    def get_current_apps(self):
        apps_code = r"""import subprocess;
command = "wmctrl -xl";
apps = subprocess.run(command, shell=True, capture_output=True, text=True).stdout.strip().split('\n');
print(apps);"""
        window_code = r"""import subprocess;
command = "wmctrl -a :ACTIVE: -v 2>&1 | grep 'Using window' | awk '{print $3}'";
window_id = subprocess.run(command, shell=True, capture_output=True, text=True).stdout.strip();
print(window_id);"""

        apps = self.controller.execute_python_command(apps_code)['output'].strip()
        apps = ast.literal_eval(apps)
        app_list = {}
        
        for app in apps:
            parts = app.split(maxsplit=4)
            if len(parts) < 4:
                continue
            if parts[1] != '0':
                continue
            window_id = parts[0]
            app_name = '.'.join(parts[2].split('.')[-(math.ceil(parts[2].count('.') / 2)):])
            title = parts[3]
            app_list[window_id] = {
                'app_name': app_name,
                'title': title
            }
        
        cur_id = self.controller.execute_python_command(window_code)['output'].strip()

        return app_list, cur_id

    def maximize_window(self):
        window_state = r"""import subprocess;
command = "xprop -id $(xprop -root _NET_ACTIVE_WINDOW | awk -F' ' '{print $5}') _NET_WM_STATE"
output = subprocess.run(command, shell=True, capture_output=True, text=True).stdout.strip();
print(output);"""
        for _ in range(5):
            try:
                self.setup_controller._launch_setup('wmctrl -r :ACTIVE: -b add,maximized_vert,maximized_horz', shell=True)
                time.sleep(2)
                output = self.controller.execute_python_command(window_state)['output'].strip()
                if '_NET_WM_STATE_FOCUSED' not in output or '_NET_WM_STATE_SKIP_TASKBAR' in output or '_NET_WM_STATE_MODAL' in output or '_NET_WM_STATE_MAXIMIZED' in output: # 没有窗口 or popups or 模态窗口 or 窗口已经最大化
                    return
            except Exception as e:
                logger.error(f"Failed to maximize window: {e}")
                time.sleep(1)

    def _get_obs(self):
        tool_list = {
            "libreoffice_calc": "CalcTools",
            "libreoffice_impress": "ImpressTools",
            "libreoffice_writer": "WriterTools",
            "code": "CodeTools",
            "vlc": "VLCTools",
            "google_chrome": "BrowserTools"
        }
        
        self.maximize_window()
        
        for i in range(3):
            try:
                app_list, cur_id = self.get_current_apps()
            except Exception as e:
                if i == 2:
                    raise e
                logger.error(f"Failed to get current apps: {e}")
                time.sleep(1)
        
        if cur_id in app_list:
            cur_app = app_list[cur_id]['app_name']

            tool_name = cur_app.strip().lower().replace('-', '_')
            if tool_name in tool_list:
                class_name = tool_list[tool_name]
                command = f"from {tool_name} import *; "
                command += f"{class_name}.env_info(); "
                command += f"{class_name}.print_result();"
                app_info = self.controller.execute_python_command(command)['output'].strip()
            else:
                app_info = None
        else:
            cur_app = None
            app_info = None
        
        tree = self.controller.get_accessibility_tree()
        screenshot = self.controller.get_screenshot()
        if screenshot is None:
            logger.error("Failed to get screenshot.")
            screenshot = b''

        return {
            "screenshot": screenshot,
            "accessibility_tree": tree,
            "instruction": self.instruction,
            "apps": app_list,
            "cur_window_id": cur_id,
            "cur_app": cur_app,
            "app_info": app_info,
        }


def test(args: argparse.Namespace, test_all_meta: dict) -> None:
    scores = []
    max_steps = args.max_steps

    # log args
    logger.info("Args: %s", args)
    # set wandb project
    cfg_args = {
        "path_to_vm": args.path_to_vm,
        "provider_name": args.provider_name,
        "headless": args.headless,
        "action_space": args.action_space,
        "observation_type": args.observation_type,
        "screen_width": args.screen_width,
        "screen_height": args.screen_height,
        "sleep_after_execution": args.sleep_after_execution,
        "max_steps": args.max_steps,
        "max_trajectory_length": args.max_trajectory_length,
        "model": args.model,
        "visual_grounder_model": args.visual_grounder_model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "stop_token": args.stop_token,
        "repetition_penalty": args.repetition_penalty,
        "result_dir": args.result_dir,
    }

    def call_llm(messages):
        logger.info("Calling LLM...")
        
        # Prepare the request data
        data = {
            "model": args.model,
            "messages": messages,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "skip_special_tokens": False,
            "stream": False,
            "include_stop_str_in_output": True,
            "stop": ["<|user|>", "<|observation|>", "</answer>"]
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'EMPTY')}"
        }
        
        # Get API base URL from environment or use default
        base_url = os.environ.get('OPENAI_BASE_URL', 'http://localhost:30000/v1')
        url = f"{base_url}/chat/completions"
        
        response = requests.post(
            url,
            json=data,
            headers=headers,
            timeout=60.0
        )
        response.raise_for_status()
        
        result = response.json()
        logger.info("LLM called successfully.")
        
        # Return both content and usage information
        content = result['choices'][0]['message']['content']
        
        usage = result.get('usage', {})
        
        return {
            'content': content,
            'usage': {
                'prompt_tokens': usage.get('prompt_tokens', 0),
                'completion_tokens': usage.get('completion_tokens', 0),
                'total_tokens': usage.get('total_tokens', 0)
            }
        }

    # Create a wrapper to track token usage and image count
    class TokenTracker:
        def __init__(self):
            self.total_prompt_tokens = 0
            self.total_completion_tokens = 0
            self.total_tokens = 0
            self.total_image_count = 0
            self.last_usage = {}
        
        def __call__(self, messages):
            result = call_llm(messages)
            
            # Count images in the messages
            image_count = 0
            for msg in messages:
                if isinstance(msg.get('content'), list):
                    for item in msg['content']:
                        if item.get('type') in ['image_url', 'input_image']:
                            image_count += 1
            
            # Store usage info with image count
            self.last_usage = {
                **result['usage'],
                'image_count': image_count
            }
            self.total_prompt_tokens += result['usage']['prompt_tokens']
            self.total_completion_tokens += result['usage']['completion_tokens']
            self.total_tokens += result['usage']['total_tokens']
            self.total_image_count += image_count
            
            # Return only content for compatibility with AutoGLMAgent
            return result['content']
        
        def get_last_usage(self):
            return self.last_usage
        
        def reset(self):
            self.total_prompt_tokens = 0
            self.total_completion_tokens = 0
            self.total_tokens = 0
            self.total_image_count = 0
            self.last_usage = {}
    
    token_tracker = TokenTracker()

    if args.visual_grounder_model == args.model:
        visual_grounder_model = None
    else:
        visual_grounder_model = AbstractLLM(args.visual_grounder_model)
    
    env = DesktopEnv(
        provider_name=args.provider_name,
        region=args.region,
        client_password=args.client_password,
        path_to_vm=args.path_to_vm,
        action_space=args.action_space,
        screen_size=(args.screen_width, args.screen_height),
        headless=args.headless,
        os_type="Ubuntu",
        require_a11y_tree=args.observation_type in ["a11y_tree", "screenshot_a11y_tree", "som"],
    )
    _ensure_vm_resolution(env, args.screen_width, args.screen_height, logger)
    agent = AutoGLMAgent(
        action_space=args.action_space,
        observation_type=args.observation_type,
        screen_size=(args.screen_width, args.screen_height),
        image_size=(args.image_width, args.image_height),
        max_trajectory_length=args.max_trajectory_length,
        client_password=args.client_password,
        gen_func=token_tracker,
        enable_recovery=args.recovery,
        max_failure_memory=args.recovery_max_failure_memory,
        visual_grounder_model=visual_grounder_model,
    )
    
    # Attach token_tracker to agent for access in run_single_example_autoglm
    agent.token_tracker = token_tracker

    for domain in tqdm(test_all_meta, desc="Domain"):
        for example_id in tqdm(test_all_meta[domain], desc="Example", leave=False):
            config_file = os.path.join(args.test_config_base_dir, f"{domain}/{example_id}.json")
            with open(config_file, "r", encoding="utf-8") as f:
                example = json.load(f)

            logger.info(f"[Domain]: {domain}")
            logger.info(f"[Example ID]: {example_id}")

            instruction = example["instruction"]

            logger.info(f"[Instruction]: {instruction}")
            # wandb each example config settings
            cfg_args["instruction"] = instruction
            cfg_args["start_time"] = datetime.datetime.now().strftime("%Y:%m:%d-%H:%M:%S")

            example_result_dir = os.path.join(
                args.result_dir,
                domain,
                example_id,
            )
            # Clean up old results if this is a rerun task
            if args.rerun or args.rerun_fail:
                if os.path.exists(example_result_dir):
                    logger.info(f"Removing old results for {domain}/{example_id}")
                    shutil.rmtree(example_result_dir)
            os.makedirs(example_result_dir, exist_ok=True)
            # example start running
            try:
                run_single_example_autoglm(
                    agent,
                    env,
                    example,
                    max_steps,
                    instruction,
                    args,
                    example_result_dir,
                    scores,
                )
            except Exception as e:
                logger.error(f"Exception in {domain}/{example_id}: {e}")
                # Only attempt to end recording if controller exists (not Docker provider)
                if hasattr(env, "controller") and env.controller is not None:
                    env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))
                with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                    f.write(json.dumps({"Error": f"Exception in {domain}/{example_id}: {e}"}))
                    f.write("\n")
                # Write result.txt with score 0 to mark task as completed (failed)
                with open(os.path.join(example_result_dir, "result.txt"), "w") as f:
                    f.write("0.0\n")
                scores.append(0.0)

    env.close()
    if len(scores) > 0:
        logger.info(f"Average score: {sum(scores) / len(scores)}")
    else:
        logger.info("No tasks completed")


def get_unfinished(target_dir, total_file_json, rerun=False, rerun_fail=False, logger=None):
    """Get unfinished tasks (aligned with run_hisa.filter_tasks logic)."""

    if not os.path.exists(target_dir):
        return total_file_json

    tasks_to_run = {}
    for domain in total_file_json:
        tasks_to_run[domain] = []
        for example_id in total_file_json[domain]:
            example_dir = os.path.join(target_dir, domain, example_id)
            execution_log_path = os.path.join(example_dir, "execution_log.json")
            result_path = os.path.join(example_dir, "result.txt")
            err_reason_path = os.path.join(example_dir, "err_reason.txt")

            if not os.path.exists(execution_log_path) and os.path.exists(result_path):
                os.remove(result_path)

            should_skip = False
            if not rerun and os.path.exists(result_path) and not os.path.exists(err_reason_path):
                try:
                    with open(result_path, "r") as f:
                        result = float(f.read().strip())
                    if result > 0.0 or not rerun_fail:
                        should_skip = True
                except (ValueError, IOError) as e:
                    if logger is not None:
                        logger.warning(f"Failed to read result for {domain}/{example_id}: {e}")
                    else:
                        print(f"[Warning] Failed to read result for {domain}/{example_id}: {e}")

            if not should_skip:
                tasks_to_run[domain].append(example_id)

    tasks_to_run = {k: v for k, v in tasks_to_run.items() if v}
    return tasks_to_run


def run_single_example(agent, env, example, max_steps, instruction, args, example_result_dir, scores):
    runtime_logger = setup_logger(os.path.basename(args.result_dir), getattr(args, "log_level", "INFO"))
    try:
        agent.reset(runtime_logger)
    except Exception as e:
        agent.reset()

    env.reset(task_config=example)
    
    time.sleep(60) # Wait for the environment to be ready
    obs = env._get_obs() # Get the initial observation
    done = False
    step_idx = 0
    env.controller.start_recording()
    while not done and step_idx < max_steps:
        response, actions = agent.predict(
            instruction,
            obs
        )
        for action in actions:
            original_action = action
            # Capture the timestamp before executing the action
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
            logger.info("Step %d: %s", step_idx + 1, action)
            obs, reward, done, info = env.step(action, args.sleep_after_execution)

            logger.info("Reward: %.2f", reward)
            logger.info("Done: %s", done)
            # Save screenshot and trajectory information
            with open(os.path.join(example_result_dir, f"step_{step_idx + 1}_{action_timestamp}.png"),
                      "wb") as _f:
                _f.write(obs['screenshot'])
            if done:
                logger.info("The episode is done.")
                break
        step_idx += 1
    result = env.evaluate()
    logger.info("Result: %.2f", result)
    scores.append(result)
    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")
    env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))

def run_single_example_human(env, example, example_result_dir, scores):
    runtime_logger = setup_logger(os.path.basename(example_result_dir), "INFO")
    env.reset(task_config=example)
    time.sleep(60) # Wait for the environment to be ready
    obs = env._get_obs() # Get the initial observation
    
    # Save initial screenshot
    with open(os.path.join(example_result_dir, "initial_state.png"), "wb") as _f:
        _f.write(obs['screenshot'])
    
    # Save trajectory information
    # Evaluate the result
    result = env.evaluate()
    logger.info("Result: %.2f", result)
    scores.append(result)
    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")

def run_single_example_autoglm(agent, env, example, max_steps, instruction, args, example_result_dir, scores):
    runtime_logger = setup_logger(os.path.basename(args.result_dir), getattr(args, "log_level", "INFO"))
    try:
        agent.reset(runtime_logger)
    except Exception as e:
        agent.reset()

    # Record start time for execution time tracking
    start_time = time.time()
    
    # Reset token tracker if it exists
    if hasattr(agent, 'token_tracker'):
        agent.token_tracker.reset()
    
    env.reset(task_config=example)
    
    time.sleep(60) # Wait for the environment to be ready
    obs = env._get_obs() # Get the initial observation
    done = False
    step_idx = 0
    action_logs = []  # Store action logs for execution_log
    recovery_active = False
    recovery_context = None
    reset_count = 0
    max_resets = getattr(args, "recovery_max_resets", 0)
    recovery_steps = 0
    max_recovery_steps = getattr(args, "recovery_max_steps", 0)
    action_history_full = []
    replay_index = 0
    failure_end_index = None
    failure_sequences = []
    branch_id = 0
    branch_parents = {0: None}
    if getattr(args, "recovery_debug_force", False):
        recovery_active = True
        recovery_context = {"reason": "debug_force"}
        logger.info("RECOVERY_DEBUG_FORCE: enter recovery mode at start")
    
    # Create operations directory like hisa.py
    operations_dir = os.path.join(example_result_dir, "operations")
    os.makedirs(operations_dir, exist_ok=True)
    
    env.controller.start_recording()
    def _normalize_exe_result(obs):
        exe_result = obs.get("exe_result", "") if isinstance(obs, dict) else ""
        if isinstance(exe_result, bytes):
            exe_result = exe_result.decode("utf-8", errors="replace")
        return str(exe_result)

    def _is_tool_action(action):
        if not isinstance(action, str):
            return False
        if "Tools." in action:
            return True
        if action.strip().startswith("from ") and "Tools." in action:
            return True
        return False

    def _is_execution_error(exe_result, info, action):
        if _is_tool_action(action):
            return False
        if info and info.get("fail", False):
            return True
        if not exe_result:
            return False
        lower = exe_result.lower()
        error_markers = [
            "error",
            "exception",
            "traceback",
            "failed to execute",
            "no such file",
            "not found",
            "permission denied",
            "invalid",
        ]
        return any(marker in lower for marker in error_markers)

    def _is_escape_syntax_error(exe_result):
        if not exe_result:
            return False
        return "SyntaxError: unexpected character after line continuation character" in exe_result

    def _apply_auto_fix(action):
        if not isinstance(action, str):
            return action
        fixed = action
        # unwrap common double-escaped sequences
        for _ in range(2):
            fixed = fixed.replace("\\\\'", "\\'")
            fixed = fixed.replace("\\\\\"", "\\\"")
            fixed = fixed.replace("\\\\n", "\\n")
            fixed = fixed.replace("\\\\t", "\\t")
            fixed = fixed.replace("\\\\r", "\\r")
        # unescape quotes
        fixed = fixed.replace("\\'", "'")
        fixed = fixed.replace("\\\"", "\"")
        # normalize regex raw string escapes
        fixed = fixed.replace("r'\\\\s", "r'\\s")
        fixed = fixed.replace('r\"\\\\s', 'r\"\\s')
        fixed = fixed.replace("r'\\\\.", "r'\\.")
        fixed = fixed.replace('r\"\\\\.', 'r\"\\.')
        fixed = fixed.replace("r'\\\\d", "r'\\d")
        fixed = fixed.replace('r\"\\\\d', 'r\"\\d')
        fixed = fixed.replace("r'\\\\w", "r'\\w")
        fixed = fixed.replace('r\"\\\\w', 'r\"\\w')
        return fixed

    def _format_branch_tree():
        lines = ["Branch Tree:"]
        children = {}
        for child, parent in branch_parents.items():
            children.setdefault(parent, []).append(child)
        def _walk(node, indent):
            for child in sorted(children.get(node, [])):
                lines.append(f"{'  ' * indent}- branch {child} (parent {node})")
                _walk(child, indent + 1)
        _walk(None, 0)
        return "\n".join(lines)

    def _record_failure_sequence(seq, reason, start_index, end_index):
        if not seq:
            return
        meta = {
            "branch_id": branch_id,
            "parent_branch_id": branch_parents.get(branch_id),
            "start_index": start_index,
            "end_index": end_index,
            "reason": reason,
        }
        item = {"sequence": [str(a) for a in seq], "meta": meta}
        failure_sequences.append(item)
        if hasattr(agent, "record_failure_sequence"):
            agent.record_failure_sequence(item)

    def _replay_prefix():
        nonlocal obs
        if replay_index <= 0:
            return
        logger.info("REPLAY: start %d steps (branch %d)", replay_index, branch_id)
        for idx, action in enumerate(action_history_full[:replay_index], start=1):
            if isinstance(action, str) and action in ["WAIT", "DONE", "FAIL", "RESET", "ROLLBACK"]:
                continue
            logger.info("REPLAY: step %d/%d action=%s", idx, replay_index, str(action))
            obs, _, _, _ = env.step(action, args.sleep_after_execution)
            screenshot_file = f"replay_{idx}.png"
            with open(os.path.join(operations_dir, screenshot_file), "wb") as _f:
                _f.write(obs["screenshot"])
            action_logs.append({
                "step": step_idx + 1,
                "type": "replay",
                "execution_success": True,
                "screenshot": screenshot_file,
                "action": str(action),
                "response": "",
                "exe_result": f"Replay {idx}/{replay_index}",
                "step_time": 0.0,
                "token_usage": {},
                "recovery_mode": True,
                "branch_id": branch_id,
                "branch_parent_id": branch_parents.get(branch_id),
            })
    while not done and step_idx < max_steps:
        if getattr(args, "recovery", False) and hasattr(agent, "set_recovery_mode"):
            agent.set_recovery_mode(recovery_active, recovery_context)
            if recovery_active and max_recovery_steps and recovery_steps >= max_recovery_steps:
                recovery_steps = 0
                if reset_count < max_resets:
                    reset_count += 1
                    logger.info("AUTO_RESET: recovery_steps exceeded, count %d/%d (branch %d)", reset_count, max_resets, branch_id)
                    env.reset(task_config=example)
                    time.sleep(60)
                    obs = env._get_obs()
                    recovery_active = False
                    recovery_context = None
                    failure_end_index = None
                    if action_history_full and replay_index < len(action_history_full):
                        _record_failure_sequence(
                            action_history_full[replay_index:],
                            "branch_abandoned",
                            replay_index,
                            len(action_history_full) - 1,
                        )
                        action_history_full = action_history_full[:replay_index]
                    _replay_prefix()
                    action_logs.append({
                        "step": step_idx + 1,
                        "type": "reset",
                        "execution_success": True,
                        "screenshot": "",
                        "action": "AUTO_RESET",
                        "response": "",
                        "exe_result": f"Auto reset after recovery steps ({reset_count}/{max_resets})",
                        "step_time": 0.0,
                        "token_usage": {},
                        "recovery_mode": True,
                        "branch_id": branch_id,
                        "branch_parent_id": branch_parents.get(branch_id),
                    })
                    continue
                else:
                    env.action_history.append("FAIL")
                    done = True
                    info = {"fail": True}
                    logger.error("AUTO_RESET blocked: max resets exceeded")
                    logger.error("Auto reset requested but max resets exceeded")
                    break

        response, actions = agent.predict(
            instruction,
            obs
        )
        
        # Get token usage for this step if available
        step_usage = {}
        if hasattr(agent, 'token_tracker'):
            last_usage = agent.token_tracker.get_last_usage()
            step_usage = {
                'prompt_tokens': last_usage.get('prompt_tokens', 0),
                'completion_tokens': last_usage.get('completion_tokens', 0),
                'total_tokens': last_usage.get('total_tokens', 0),
                'image_count': last_usage.get('image_count', 0)
            }
        
        for action in actions:
            original_action = action
            # Capture the timestamp before executing the action
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
            
            # Record step start time
            step_start_time = time.time()

            if getattr(args, "recovery", False) and isinstance(action, str):
                if recovery_active and action == "ROLLBACK":
                    if reset_count < max_resets:
                        action = "RESET"
                    else:
                        action = "FAIL"
                if action == "ROLLBACK":
                    replay_index = max(0, replay_index - 1)
                    recovery_steps += 1
                    logger.info("ROLLBACK: replay_index -> %d (branch %d)", replay_index, branch_id)
                    recovery_active = True
                    recovery_context = {
                        "reason": "parse_error",
                        "action": str(original_action),
                        "exe_result": "No action code in response",
                    }
                    obs = env._get_obs()
                    if isinstance(obs, dict):
                        obs["exe_result"] = "Response contains no action code. Triggering ROLLBACK without retry."
                    screenshot_file = f"step_{step_idx + 1}_rollback.png"
                    with open(os.path.join(operations_dir, screenshot_file), "wb") as _f:
                        _f.write(obs["screenshot"])
                    action_logs.append({
                        "step": step_idx + 1,
                        "type": "rollback",
                        "execution_success": True,
                        "screenshot": screenshot_file,
                        "action": "ROLLBACK",
                        "response": str(response) if response else "",
                        "exe_result": f"Replay index -> {replay_index}",
                        "step_time": round(time.time() - step_start_time, 2),
                        "token_usage": step_usage,
                        "recovery_mode": True,
                        "branch_id": branch_id,
                        "branch_parent_id": branch_parents.get(branch_id),
                    })
                    continue
                if action == "RESET":
                    if reset_count < max_resets:
                        reset_count += 1
                        logger.info("RESET: count %d/%d (branch %d)", reset_count, max_resets, branch_id)
                        env.reset(task_config=example)
                        time.sleep(60)
                        obs = env._get_obs()
                        recovery_active = False
                        recovery_context = None
                        recovery_steps = 0
                        failure_end_index = None
                        if action_history_full and replay_index < len(action_history_full):
                            _record_failure_sequence(
                                action_history_full[replay_index:],
                                "branch_abandoned",
                                replay_index,
                                len(action_history_full) - 1,
                            )
                            action_history_full = action_history_full[:replay_index]
                        _replay_prefix()
                        screenshot_file = f"step_{step_idx + 1}_reset.png"
                        with open(os.path.join(operations_dir, screenshot_file), "wb") as _f:
                            _f.write(obs["screenshot"])
                        action_logs.append({
                            "step": step_idx + 1,
                            "type": "reset",
                            "execution_success": True,
                            "screenshot": screenshot_file,
                            "action": "RESET",
                            "response": str(response) if response else "",
                            "exe_result": f"Environment reset ({reset_count}/{max_resets})",
                            "step_time": round(time.time() - step_start_time, 2),
                            "token_usage": step_usage,
                            "recovery_mode": True,
                            "branch_id": branch_id,
                            "branch_parent_id": branch_parents.get(branch_id),
                        })
                        continue
                    else:
                        env.action_history.append("FAIL")
                        done = True
                        info = {"fail": True}
                        logger.error("RESET blocked: max resets exceeded")
                        logger.error("RESET requested but max resets exceeded")
                        break
            
            obs, reward, done, info = env.step(action, args.sleep_after_execution)

            logger.info("Reward: %.2f", reward)
            logger.info("Done: %s", done)
            
            # Calculate step execution time
            step_time = time.time() - step_start_time
            
            # Determine action type and create screenshot filename (hisa format)
            if isinstance(action, dict):
                # Dict actions like OPEN_APP, OPEN_CHROME_TAB
                action_type = "gui_action"
                screenshot_file = f"step_{step_idx + 1}_gui_action.png"
            elif isinstance(action, str):
                if action in ["WAIT", "DONE", "FAIL"]:
                    action_type = action.lower()
                    if action == "WAIT":
                        screenshot_file = f"step_{step_idx + 1}_wait.png"
                    else:
                        screenshot_file = f"step_{step_idx + 1}_{action.lower()}.png"
                elif "pyautogui" in action or "Agent." in action or "BrowserTools." in action:
                    action_type = "gui_action"
                    screenshot_file = f"step_{step_idx + 1}_gui_action.png"
                else:
                    action_type = "bash_execution"
                    screenshot_file = f"step_{step_idx + 1}_bash.png"
            else:
                action_type = "gui_action"
                screenshot_file = f"step_{step_idx + 1}_gui_action.png"
            
            # Save screenshot in operations directory only
            with open(os.path.join(operations_dir, screenshot_file), "wb") as _f:
                _f.write(obs['screenshot'])
            
            # Add to action_logs (ensure all values are JSON serializable)
            exe_result = _normalize_exe_result(obs)
            auto_fix_attempted = False
            auto_fix_applied = False
            if isinstance(action, str) and _is_escape_syntax_error(exe_result):
                auto_fix_attempted = True
                fixed_action = _apply_auto_fix(action)
                if fixed_action != action:
                    logger.info("AUTO_FIX: retry after unescaping quotes")
                    obs2, reward2, done2, info2 = env.step(fixed_action, args.sleep_after_execution)
                    exe_result2 = _normalize_exe_result(obs2)
                    if not _is_escape_syntax_error(exe_result2):
                        auto_fix_applied = True
                        action = fixed_action
                        obs, reward, done, info = obs2, reward2, done2, info2
                        exe_result = exe_result2
                    else:
                        obs, reward, done, info = obs2, reward2, done2, info2
                        exe_result = exe_result2
            if auto_fix_attempted:
                with open(os.path.join(operations_dir, screenshot_file), "wb") as _f:
                    _f.write(obs["screenshot"])
            response_str = str(response) if response else ""
            action_logs.append({
                "step": step_idx + 1,
                "type": action_type,
                "execution_success": reward >= 0 and not info.get("fail", False),
                "screenshot": screenshot_file,
                "action": str(action),
                "action_original": str(original_action),
                "response": response_str,  # Truncate long responses
                "exe_result": str(exe_result) if exe_result else "",
                "step_time": round(step_time, 2),
                "token_usage": step_usage,
                "recovery_mode": bool(recovery_active),
                "branch_id": branch_id,
                "branch_parent_id": branch_parents.get(branch_id),
                "auto_fix_attempted": auto_fix_attempted,
                "auto_fix_applied": auto_fix_applied,
            })

            if not recovery_active and action_type not in ["done", "fail"]:
                if replay_index < len(action_history_full):
                    _record_failure_sequence(
                        action_history_full[replay_index:],
                        "branch_abandoned",
                        replay_index,
                        len(action_history_full) - 1,
                    )
                    action_history_full = action_history_full[:replay_index]
                    new_branch_id = max(branch_parents.keys()) + 1
                    branch_parents[new_branch_id] = branch_id
                    branch_id = new_branch_id
                action_history_full.append(action if isinstance(action, (str, dict)) else str(action))
                replay_index = len(action_history_full)

            if getattr(args, "recovery", False) and _is_execution_error(exe_result, info, action):
                recovery_active = True
                recovery_steps = 0
                if action_history_full:
                    failure_end_index = len(action_history_full) - 1
                    replay_index = max(0, failure_end_index)
                else:
                    replay_index = 0
                recovery_context = {
                    "reason": "execution_error",
                    "action": str(action),
                    "exe_result": exe_result,
                    "app": obs.get("cur_app"),
                }
                if failure_end_index is not None:
                    failed_seq = action_history_full[replay_index:failure_end_index + 1]
                    _record_failure_sequence(
                        failed_seq,
                        "failure",
                        replay_index,
                        failure_end_index,
                    )
                if hasattr(agent, "record_failure"):
                    agent.record_failure(recovery_context)
                break
            if recovery_active:
                recovery_steps += 1
                
            if done:
                logger.info("The episode is done.")
                break
        
        # Invalid Action
        if not actions:
            obs = env._get_obs() # update observation
            # Record this as an invalid action step
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
            screenshot_file = f"step_{step_idx + 1}_invalid.png"
            
            # Save screenshot in operations directory only
            with open(os.path.join(operations_dir, screenshot_file), "wb") as _f:
                _f.write(obs['screenshot'])
            
            response_str = str(response) if response else ""
            action_logs.append({
                "step": step_idx + 1,
                "type": "invalid_action",
                "execution_success": False,
                "screenshot": screenshot_file,
                "action": "Parse error or invalid action",
                "response": response_str,
                "exe_result": "Invalid action - no actions returned",
                "step_time": 0.0,
                "token_usage": step_usage,
                "recovery_mode": bool(recovery_active),
                "branch_id": branch_id,
                "branch_parent_id": branch_parents.get(branch_id),
            })
            
        step_idx += 1
    
    if not done: # not completed the task yet
        env.action_history.append('FAIL')
        logger.error("TASK FAIL: not completed within max steps")
    
    result = env.evaluate()
    logger.info("Result: %.2f", result)
    scores.append(result)
    
    # Calculate execution time
    execution_time = time.time() - start_time
    
    # Count steps by type (invalid_action also counts as gui_action attempt)
    gui_steps = len([log for log in action_logs if log["type"] in ["gui_action", "invalid_action"]])
    bash_steps = len([log for log in action_logs if log["type"] == "bash_execution"])
    wait_steps = len([log for log in action_logs if log["type"] == "wait"])
    
    # Calculate total token usage and image count
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    total_image_count = 0
    if hasattr(agent, 'token_tracker'):
        total_prompt_tokens = agent.token_tracker.total_prompt_tokens
        total_completion_tokens = agent.token_tracker.total_completion_tokens
        total_tokens = agent.token_tracker.total_tokens
        total_image_count = agent.token_tracker.total_image_count
    
    # Calculate cost (assuming GPT-4 pricing, adjust as needed)
    # You can adjust these rates based on your actual model pricing
    prompt_token_cost = 0
    completion_token_cost = 0
    total_cost = (total_prompt_tokens * prompt_token_cost / 1000) + (total_completion_tokens * completion_token_cost / 1000)
    
    # Determine success and failure reason
    failure_reason = ""
    if result == 0.0:
        if not done and step_idx >= max_steps:
            failure_reason = f"Reached maximum steps ({max_steps}) without completing the task"
        elif env.action_history and env.action_history[-1] == 'FAIL':
            failure_reason = "Task marked as failed"
    if failure_reason:
        logger.error("FAILURE_REASON: %s", failure_reason)
    
    # Create execution_log similar to hisa.py
    execution_log = {
        "statistics": {
            "score": result,
            "total_steps": step_idx,
            "cua_steps": gui_steps,
            "coding_steps": bash_steps,
            "wait_steps": wait_steps,
            "image_count": total_image_count,
            "total_cost": total_cost,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "execution_time": execution_time,
            "model_usage": {
                "model": {
                    "model_name": args.model if hasattr(args, 'model') else "autoglm-os",
                    "cost": total_cost,
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "image_count": total_image_count
                }
            }
        },
        "task_config": example,
        "additional_context": "",
        "action_logs": action_logs,
        "recovery": {
            "enabled": getattr(args, "recovery", False),
            "reset_count": reset_count,
            "max_resets": max_resets,
            "failure_memory": getattr(agent, "failure_memory", []),
            "failure_sequences": failure_sequences,
            "branch_parents": branch_parents,
            "branch_summary_text": _format_branch_tree(),
        },
    }
    
    # Save execution_log.json
    with open(os.path.join(example_result_dir, "execution_log.json"), "w", encoding="utf-8") as f:
        json.dump(execution_log, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")
    env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))


if __name__ == "__main__":
    ####### The complete version of the list of examples #######
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = config()
    
    # Initialize logger after args are parsed
    result_name = os.path.basename(args.result_dir)
    logger = setup_logger(result_name, args.log_level)
    if args.client_password == "":
        if args.provider_name == "aws":
            args.client_password = "osworld-public-evaluation"
        else:
            args.client_password = "password"
    else:
        args.client_password = args.client_password

    # save args to json in result_dir/action_space/observation_type/model/args.json
    path_to_args = os.path.join(
        args.result_dir,
        "args.json",
    )
    os.makedirs(os.path.dirname(path_to_args), exist_ok=True)
    with open(path_to_args, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4)

    with open(args.test_all_meta_path, "r", encoding="utf-8") as f:
        test_all_meta = json.load(f)

    if args.domain != "all":
        test_all_meta = {args.domain: test_all_meta[args.domain]}

    test_file_list = get_unfinished(
        args.result_dir,
        test_all_meta,
        rerun=args.rerun,
        rerun_fail=args.rerun_fail,
        logger=logger,
    )
    summary(args.result_dir, test_all_meta)
    left_info = ""
    for domain in test_file_list:
        left_info += f"{domain}: {len(test_file_list[domain])}\n"
    logger.info(f"Left tasks:\n{left_info}")

    test(args, test_file_list)
    
    # Call summary() from utils.py after all tasks are completed
    logger.info("Generating summary...")
    summary(args.result_dir, test_all_meta)
