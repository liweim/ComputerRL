"""Script to run HiSA agent evaluation on the benchmark.
Adapted from run_autoglm_v.py with HiSA agent integration.
"""

import argparse
import datetime
import json
import logging
import os
import sys
import math
import ast
import time
import requests
from tqdm import tqdm
import shutil
import docker

from desktop_env.desktop_env import MAX_RETRIES, DesktopEnv as DesktopEnvBase
from mm_agents.hisa import HiSAAgent
from typing import Optional, Dict, Any
from utils import summary, setup_logger


def cleanup_osworld_containers(remove_running=False):
    """Clean up osworld docker containers before starting."""
    try:
        client = docker.from_env()
        containers = client.containers.list(all=True, filters={"ancestor": "happysixd/osworld-docker"})
        if containers:
            removed_count = 0
            skipped_count = 0
            for container in containers:
                try:
                    if container.status == "running":
                        if remove_running:
                            container.stop(timeout=5)
                            container.remove(force=True)
                            print(f"  Stopped and removed running container: {container.name}")
                            removed_count += 1
                        else:
                            skipped_count += 1
                    else:
                        container.remove(force=True)
                        print(f"  Removed exited container: {container.name}")
                        removed_count += 1
                except Exception as e:
                    print(f"  Failed to remove container {container.name}: {e}")
            print(f"Cleanup completed. Removed: {removed_count}, Skipped (running): {skipped_count}")
        else:
            print("No existing osworld containers found.")
    except Exception as e:
        print(f"Warning: Failed to cleanup containers: {e}")


logger = None


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HiSA agent evaluation on the benchmark")

    # Environment config
    parser.add_argument("--path_to_vm", type=str)
    parser.add_argument("--provider_name", type=str, default="docker")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--action_space", type=str, default="autoglm_computer_use")
    parser.add_argument("--with_atree", action="store_true", default=False, help="Include accessibility tree in observation")
    parser.add_argument("--screen_width", type=int, default=1920)
    parser.add_argument("--screen_height", type=int, default=1080)
    parser.add_argument("--sleep_after_execution", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=100)

    # Agent config
    parser.add_argument("--max_trajectory_length", type=int, default=3)
    parser.add_argument("--test_config_base_dir", type=str, default="evaluation_examples/examples")

    # LM config
    parser.add_argument("--model", type=str, default="autoglm-os")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.1)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--repetition_penalty", type=float, default=1)
    parser.add_argument("--stop_token", type=str, default=None)
    parser.add_argument("--image_width", type=int, default=1280)
    parser.add_argument("--image_height", type=int, default=720)

    # State manager LM config (uses same model as main, but different parameters)
    parser.add_argument("--sm_temperature", type=float, default=0.2, help="Temperature for state manager")
    parser.add_argument("--sm_top_p", type=float, default=0.9, help="Top-p for state manager")
    parser.add_argument("--sm_max_tokens", type=int, default=500, help="Max tokens for state manager")
    parser.add_argument("--sm_repetition_penalty", type=float, default=1, help="Repetition penalty for state manager")

    # HiSA specific config
    parser.add_argument("--wo_pattern", action="store_true", default=True, help="Disable pattern learning")
    parser.add_argument("--wo_step", action="store_true", default=False, help="Disable step abstraction")
    parser.add_argument("--wo_refinement", action="store_true", default=False, help="Disable context refinement")
    parser.add_argument("--refine_period", type=int, default=5, help="Steps between context refinements")
    parser.add_argument("--sliding_window_size", type=int, default=5, help="Sliding window size for history")

    # Example config
    parser.add_argument("--domain", type=str, default="all")
    parser.add_argument("--test_all_meta_path", type=str, default="evaluation_examples/test_nogdrive.json")

    # AWS config
    parser.add_argument("--region", type=str, default="us-east-1")
    parser.add_argument("--client_password", type=str, default="")

    # Logging
    parser.add_argument("--result_dir", type=str, default="./results")
    parser.add_argument("--log_level", type=str, default="INFO")

    # Rerun
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--rerun_fail", action="store_true")

    # Docker related
    parser.add_argument("--cleanup_docker", action="store_true", default=False, help="Cleanup docker containers before starting")

    args = parser.parse_args()
    return args


class DesktopEnv(DesktopEnvBase):
    """Extended DesktopEnv with HiSA-compatible step method."""

    def step(self, action, pause=2):
        self._step_no += 1
        self.action_history.append(action)
        self.is_environment_used = True

        reward = 0
        done = False
        info = {}
        logger.info(f"[Step] {self._step_no} in trajectory {self._traj_no} with action: {action}")

        # Handle special actions
        if action in ['WAIT', 'FAIL', 'DONE']:
            if action == 'WAIT':
                time.sleep(pause)
                exe_result = f'Wait {pause} seconds'
            elif action == 'FAIL':
                done = True
                info = {"fail": True}
                exe_result = 'Finish: fail'
            elif action == 'DONE':
                done = True
                info = {"done": True}
                exe_result = 'Finish: success'
        elif isinstance(action, dict):
            if action.get('action_type') == 'OPEN_APP':
                self.setup_controller._launch_setup(action['parameters']['launch_app_command'], shell=True)
                exe_result = 'Open ' + action['parameters']['app_name']
            elif action.get('action_type') == 'OPEN_CHROME_TAB':
                self.setup_controller._chrome_open_tabs_setup(action['parameters']['urls_to_open'])
                exe_result = 'Open ' + str(action['parameters']['urls_to_open']) + ' in Chrome successfully'
            elif action.get('type') == 'bash':
                # Handle bash commands from HiSA agent
                result = self.controller.run_bash_script(action['command'], timeout=60)
                exe_result = result.get('output', '') if result.get('status') == 'success' else result.get('error', 'Error')
            else:
                exe_result = f'Unknown action type: {action}'
        else:
            # Execute pyautogui command
            result = self.controller.execute_python_command(action)
            try:
                if result['error']:
                    exe_result = result['error'].strip()
                else:
                    exe_result = result['output'].strip()
            except Exception as e:
                exe_result = f'Error Action: {action}'
                logger.error(f"Error executing action: {e}")

        time.sleep(pause)
        observation = self._get_obs()
        observation['exe_result'] = exe_result

        return observation, reward, done, info

    def reset(self, task_config: Optional[Dict[str, Any]] = None, seed=None, options=None) -> Dict[str, Any]:
        """Reset environment for a new task."""
        logger.info("Resetting environment...")
        self._traj_no += 1
        self._step_no = 0
        self.action_history.clear()

        for attempt in range(MAX_RETRIES):
            if task_config is not None:
                task_use_proxy = task_config.get("proxy", False) and self.enable_proxy
                if not self.enable_proxy and task_config.get("proxy", False):
                    logger.info("Task requires proxy but proxy is disabled, ignoring.")
                if task_use_proxy != self.current_use_proxy:
                    self.current_use_proxy = task_use_proxy

            if self.is_environment_used:
                logger.info(f"Reverting to snapshot {self.snapshot_name}...")
                self._revert_to_snapshot()
                self._start_emulator()
                self.is_environment_used = False
            else:
                logger.info(f"Environment is clean, skipping snapshot revert (provider: {self.provider_name}).")

            if task_config is not None:
                if task_config.get("proxy", False) and self.enable_proxy:
                    self.setup_controller._proxy_setup(self.client_password)
                self._set_task_info(task_config)
                self.setup_controller.reset_cache_dir(self.cache_dir)
                success = self.setup_controller.setup(self.config, task_config.get("proxy", False) and self.enable_proxy)
                if success:
                    if self.config:
                        self.is_environment_used = True
                    break
                else:
                    logger.error(f"Setup failed, retrying ({attempt + 1}/{MAX_RETRIES})...")
                    time.sleep(5)
            else:
                break

        logger.info("Environment setup complete.")

        # Upload tools
        import mm_agents.autoglm_v
        tool_dir = os.path.join(os.path.dirname(mm_agents.autoglm_v.__file__), 'tools', 'package')
        for file in os.listdir(tool_dir):
            if os.path.isdir(os.path.join(tool_dir, file)):
                continue
            self.setup_controller._upload_file_setup([{
                "local_path": os.path.join(tool_dir, file),
                "path": os.path.join('~', file)
            }])

        # Start soffice service
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
            app_list[window_id] = {'app_name': app_name, 'title': title}

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
                if '_NET_WM_STATE_FOCUSED' not in output or '_NET_WM_STATE_SKIP_TASKBAR' in output or '_NET_WM_STATE_MODAL' in output or '_NET_WM_STATE_MAXIMIZED' in output:
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
                break
            except Exception as e:
                if i == 2:
                    raise e
                logger.error(f"Failed to get current apps: {e}")
                time.sleep(1)

        cur_app = None
        app_info = None
        if cur_id in app_list:
            cur_app = app_list[cur_id]['app_name']
            tool_name = cur_app.strip().lower().replace('-', '_')
            if tool_name in tool_list:
                class_name = tool_list[tool_name]
                command = f"from {tool_name} import *; "
                command += f"{class_name}.env_info(); "
                command += f"{class_name}.print_result();"
                app_info = self.controller.execute_python_command(command)['output'].strip()

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


def run_single_example_hisa(agent, env, example, max_steps, instruction, args, example_result_dir, scores):
    """Run a single example with HiSA agent."""
    runtime_logger = logging.getLogger(f"desktopenv.example.{example['id']}")
    runtime_logger.setLevel(logging.DEBUG)
    runtime_logger.addHandler(logging.FileHandler(os.path.join(example_result_dir, "runtime.log")))

    try:
        agent.reset(runtime_logger)
    except Exception:
        agent.reset()

    start_time = time.time()

    # Reset token tracker if available
    if hasattr(agent, 'token_tracker'):
        agent.token_tracker.reset()

    env.reset(task_config=example)
    time.sleep(60)  # Wait for environment

    obs = env._get_obs()
    done = False
    step_idx = 0
    action_logs = []

    operations_dir = os.path.join(example_result_dir, "operations")
    os.makedirs(operations_dir, exist_ok=True)

    env.controller.start_recording()

    while not done and step_idx < max_steps:
        # Get before screenshot for step abstraction
        before_screenshot = obs.get('screenshot', b'')

        response, actions = agent.predict(instruction, obs)

        # Get token usage if available
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
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
            step_start_time = time.time()

            obs, reward, done, info = env.step(action, args.sleep_after_execution)

            logger.info(f"Reward: {reward:.2f}")
            logger.info(f"Done: {done}")

            step_time = time.time() - step_start_time
            after_screenshot = obs.get('screenshot', b'')

            # Determine action type
            if isinstance(action, dict):
                action_type = action.get('type', 'gui_action')
                screenshot_file = f"step_{step_idx + 1}_{action_type}.png"
            elif isinstance(action, str):
                if action in ["WAIT", "DONE", "FAIL"]:
                    action_type = action.lower()
                else:
                    action_type = "gui_action"
                screenshot_file = f"step_{step_idx + 1}_{action_type}.png"
            else:
                action_type = "gui_action"
                screenshot_file = f"step_{step_idx + 1}_gui_action.png"

            # Save screenshot
            with open(os.path.join(operations_dir, screenshot_file), "wb") as f:
                f.write(after_screenshot)

            # Add action log with step abstraction
            exe_result = obs.get("exe_result", "") if "exe_result" in obs else ""
            agent.add_action_log(
                step=step_idx + 1,
                action_type=action_type,
                action=str(action),
                execution_success=reward >= 0 and not info.get("fail", False),
                screenshot_file=screenshot_file,
                exe_result=str(exe_result),
                step_time=step_time,
                before_screenshot=before_screenshot,
                after_screenshot=after_screenshot,
            )

            # Save trajectory
            with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                f.write(json.dumps({
                    "step_num": step_idx + 1,
                    "action_timestamp": action_timestamp,
                    "action": str(action),
                    "response": response,
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "screenshot_file": f"operations/{screenshot_file}",
                    "thought": agent.current_thought,
                }))
                f.write("\n")

            if done:
                logger.info("Episode done.")
                break

        if not actions:
            obs = env._get_obs()
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
            screenshot_file = f"step_{step_idx + 1}_invalid.png"

            with open(os.path.join(operations_dir, screenshot_file), "wb") as f:
                f.write(obs.get('screenshot', b''))

            agent.add_action_log(
                step=step_idx + 1,
                action_type="invalid_action",
                action="Parse error",
                execution_success=False,
                screenshot_file=screenshot_file,
                exe_result="Invalid action",
                step_time=0.0,
            )

        step_idx += 1

    if not done:
        env.action_history.append('FAIL')

    result = env.evaluate()
    logger.info(f"Result: {result:.2f}")
    scores.append(result)

    execution_time = time.time() - start_time

    # Count steps by type
    action_logs = agent.get_action_logs()
    gui_steps = len([log for log in action_logs if log["type"] in ["gui_action", "invalid_action"]])
    bash_steps = len([log for log in action_logs if log["type"] == "bash"])
    wait_steps = len([log for log in action_logs if log["type"] == "wait"])

    # Token usage
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_image_count = 0
    if hasattr(agent, 'token_tracker'):
        total_prompt_tokens = agent.token_tracker.total_prompt_tokens
        total_completion_tokens = agent.token_tracker.total_completion_tokens
        total_image_count = agent.token_tracker.total_image_count

    # Pattern induction
    lessons = []
    if not args.wo_pattern:
        lessons = agent.pattern_induction(instruction, action_logs)
        if lessons:
            logger.info(f"Extracted {len(lessons)} lesson(s) from task execution")
            for i, lesson in enumerate(lessons):
                logger.info(f"  Lesson {i+1}: [{lesson['type']}] {lesson['lesson']}")

    # Build execution log
    execution_log = {
        "statistics": {
            "score": result,
            "total_steps": step_idx,
            "cua_steps": gui_steps,
            "coding_steps": bash_steps,
            "wait_steps": wait_steps,
            "image_count": total_image_count,
            "total_cost": 0,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "execution_time": execution_time,
            "model_usage": {
                "model": {
                    "model_name": args.model,
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "image_count": total_image_count
                }
            }
        },
        "task_config": example,
        "additional_context": "",
        "action_logs": action_logs,
        "lessons": lessons,
    }

    with open(os.path.join(example_result_dir, "execution_log.json"), "w", encoding="utf-8") as f:
        json.dump(execution_log, f, indent=2, ensure_ascii=False)

    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")

    env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))


def test(args: argparse.Namespace, test_all_meta: dict) -> None:
    scores = []
    max_steps = args.max_steps

    logger.info(f"Args: {args}")

    def call_llm(messages):
        """Call LLM with messages."""
        logger.info("Calling LLM...")

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

        base_url = os.environ.get('OPENAI_BASE_URL', 'http://localhost:30000/v1')
        url = f"{base_url}/chat/completions"

        response = requests.post(url, json=data, headers=headers, timeout=60.0)
        response.raise_for_status()

        result = response.json()
        logger.info("LLM called successfully.")

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

    class TokenTracker:
        def __init__(self):
            self.total_prompt_tokens = 0
            self.total_completion_tokens = 0
            self.total_tokens = 0
            self.total_image_count = 0
            self.last_usage = {}

        def __call__(self, messages):
            result = call_llm(messages)

            image_count = 0
            for msg in messages:
                if isinstance(msg.get('content'), list):
                    for item in msg['content']:
                        if item.get('type') in ['image_url', 'input_image']:
                            image_count += 1

            self.last_usage = {**result['usage'], 'image_count': image_count}
            self.total_prompt_tokens += result['usage']['prompt_tokens']
            self.total_completion_tokens += result['usage']['completion_tokens']
            self.total_tokens += result['usage']['total_tokens']
            self.total_image_count += image_count

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

    # Create state manager function (uses same model with different parameters)
    def state_manager_call(messages):
        """Call LLM for state manager with different parameters."""
        data = {
            "model": args.model,
            "messages": messages,
            "max_tokens": args.sm_max_tokens,
            "temperature": args.sm_temperature,
            "top_p": args.sm_top_p,
            "repetition_penalty": args.sm_repetition_penalty,
            "skip_special_tokens": False,
            "stream": False,
            "include_stop_str_in_output": True,
            "stop": ["<|user|>", "<|observation|>", "</answer>"]
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'EMPTY')}"
        }
        base_url = os.environ.get('OPENAI_BASE_URL', 'http://localhost:30000/v1')
        response = requests.post(f"{base_url}/chat/completions", json=data, headers=headers, timeout=60.0)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    
    state_manager_func = state_manager_call

    env = DesktopEnv(
        provider_name=args.provider_name,
        region=args.region,
        client_password=args.client_password,
        path_to_vm=args.path_to_vm,
        action_space=args.action_space,
        screen_size=(args.screen_width, args.screen_height),
        headless=args.headless,
        os_type="Ubuntu",
        require_a11y_tree=args.with_atree,  # Get a11y tree if needed
    )

    agent = HiSAAgent(
        action_space=args.action_space,
        screen_size=(args.screen_width, args.screen_height),
        image_size=(args.image_width, args.image_height),
        max_trajectory_length=args.max_trajectory_length,
        with_atree=args.with_atree,
        client_password=args.client_password,
        gen_func=token_tracker,
        state_manager_func=state_manager_func,
        max_steps=args.max_steps,
        wo_pattern=args.wo_pattern,
        wo_step=args.wo_step,
        wo_refinement=args.wo_refinement,
        refine_period=args.refine_period,
        sliding_window_size=args.sliding_window_size,
    )

    agent.token_tracker = token_tracker

    for domain in tqdm(test_all_meta, desc="Domain"):
        for example_id in tqdm(test_all_meta[domain], desc="Example", leave=False):
            config_file = os.path.join(args.test_config_base_dir, f"{domain}/{example_id}.json")
            with open(config_file, "r", encoding="utf-8") as f:
                example = json.load(f)

            logger.info(f"[Domain]: {domain}")
            logger.info(f"[Example ID]: {example_id}")
            logger.info(f"[Instruction]: {example['instruction']}")

            example_result_dir = os.path.join(args.result_dir, domain, example_id)
            # Clean up old results if this is a rerun task
            if args.rerun or args.rerun_fail:
                if os.path.exists(example_result_dir):
                    logger.info(f"Removing old results for {domain}/{example_id}")
                    shutil.rmtree(example_result_dir)
            os.makedirs(example_result_dir, exist_ok=True)

            try:
                run_single_example_hisa(
                    agent, env, example, max_steps,
                    example['instruction'], args, example_result_dir, scores
                )
            except Exception as e:
                logger.error(f"Exception in {domain}/{example_id}: {e}")
                if hasattr(env, "controller") and env.controller is not None:
                    env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))
                with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                    f.write(json.dumps({"Error": f"Exception: {e}"}))
                    f.write("\n")
                with open(os.path.join(example_result_dir, "result.txt"), "w") as f:
                    f.write("0.0\n")
                scores.append(0.0)

    env.close()
    if scores:
        logger.info(f"Average score: {sum(scores) / len(scores)}")
    else:
        logger.info("No tasks completed")


def get_unfinished(target_dir, total_file_json, rerun=False, rerun_fail=False):
    """Get unfinished tasks."""
    if not os.path.exists(target_dir):
        return total_file_json

    if rerun:
        for domain in os.listdir(target_dir):
            domain_path = os.path.join(target_dir, domain)
            if os.path.isdir(domain_path):
                for example_id in os.listdir(domain_path):
                    if example_id == "onboard":
                        continue
                    example_path = os.path.join(domain_path, example_id)
                    if os.path.isdir(example_path):
                        shutil.rmtree(example_path)
        return total_file_json

    # If rerun_fail is True, only rerun failed tasks specified in total_file_json
    if rerun_fail:
        tasks_to_rerun = {}
        for domain in total_file_json:
            tasks_to_rerun[domain] = []
            if domain not in os.listdir(target_dir):
                # Domain doesn't exist, run all tasks in it
                tasks_to_rerun[domain] = total_file_json[domain]
                continue
            domain_path = os.path.join(target_dir, domain)
            if not os.path.isdir(domain_path):
                # Domain directory doesn't exist, run all tasks in it
                tasks_to_rerun[domain] = total_file_json[domain]
                continue
            for example_id in total_file_json[domain]:
                example_path = os.path.join(domain_path, example_id)
                if not os.path.isdir(example_path):
                    # Task directory doesn't exist, need to run
                    tasks_to_rerun[domain].append(example_id)
                elif "result.txt" not in os.listdir(example_path) or 'execution_log.json' not in os.listdir(example_path):
                    # Incomplete task, need to rerun
                    tasks_to_rerun[domain].append(example_id)
                else:
                    try:
                        result_file = os.path.join(example_path, "result.txt")
                        with open(result_file, "r") as f:
                            score = float(f.read().strip())
                        if score == 0.0:
                            # Failed task, need to rerun
                            tasks_to_rerun[domain].append(example_id)
                        # If score > 0, skip (don't add to tasks_to_rerun)
                    except Exception as e:
                        raise RuntimeError(f"Error reading {result_file}: {e}")

        # Remove empty domains
        tasks_to_rerun = {k: v for k, v in tasks_to_rerun.items() if v}

        return tasks_to_rerun

    # Normal case: check all existing directories and find unfinished tasks
    finished = {}
    for domain in os.listdir(target_dir):
        finished[domain] = []
        domain_path = os.path.join(target_dir, domain)
        if os.path.isdir(domain_path):
            for example_id in os.listdir(domain_path):
                if example_id == "onboard":
                    continue
                example_path = os.path.join(domain_path, example_id)
                if os.path.isdir(example_path):
                    if "result.txt" not in os.listdir(example_path) or 'execution_log.json' not in os.listdir(example_path):
                        print(f"[Cleanup] Removing incomplete: {example_path}")
                        shutil.rmtree(example_path)
                    else:
                        finished[domain].append(example_id)

    if not finished:
        return total_file_json

    for domain, examples in finished.items():
        if domain in total_file_json:
            total_file_json[domain] = [x for x in total_file_json[domain] if x not in examples]

    return total_file_json

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = config()

    # Clean up existing osworld containers before starting (docker only, if requested)
    if args.provider_name == "docker" and args.cleanup_docker:
        cleanup_osworld_containers()

    result_name = os.path.basename(args.result_dir)
    logger = setup_logger(result_name, args.log_level)

    if args.client_password == "":
        args.client_password = "osworld-public-evaluation" if args.provider_name == "aws" else "password"

    path_to_args = os.path.join(args.result_dir, "args.json")
    os.makedirs(os.path.dirname(path_to_args), exist_ok=True)
    with open(path_to_args, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4)

    with open(args.test_all_meta_path, "r", encoding="utf-8") as f:
        test_all_meta = json.load(f)

    if args.domain != "all":
        test_all_meta = {args.domain: test_all_meta[args.domain]}

    test_file_list = get_unfinished(args.result_dir, test_all_meta, rerun=args.rerun, rerun_fail=args.rerun_fail)

    left_info = ""
    for domain in test_file_list:
        left_info += f"{domain}: {len(test_file_list[domain])}\n"
    logger.info(f"Left tasks:\n{left_info}")

    test(args, test_file_list)

    logger.info("Generating summary...")
    summary(args.result_dir, test_all_meta)
