"""Script to run end-to-end evaluation on the benchmark.
Utils and basic architecture credit to https://github.com/web-arena-x/webarena/blob/main/run.py.
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
import backoff
import httpx
import requests
from requests.exceptions import SSLError
from tqdm import tqdm
import shutil

import lib_run_single
import docker
from desktop_env.desktop_env import MAX_RETRIES, DesktopEnv as DesktopEnvBase
from mm_agents.autoglm_v import AutoGLMAgent
from typing import Optional, Dict, Any
from utils import summary, setup_logger


def cleanup_osworld_containers(remove_running=False):
    """Clean up osworld docker containers before starting.
    
    Args:
        remove_running: If True, also stop and remove running containers.
                       If False (default), only remove exited containers.
    """
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
                        # Remove exited/stopped containers
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

# Almost deprecated since it's not multi-env, use run_multienv_*.py instead
logger = None  # Will be initialized in main

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
    )
    parser.add_argument("--screen_width", type=int, default=1920)
    parser.add_argument("--screen_height", type=int, default=1080)
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
    parser.add_argument("--image_width", type=int, default=1280)
    parser.add_argument("--image_height", type=int, default=720)

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
            try:
                if result['error']:
                    exe_result = result['error'].strip()
                else:
                    exe_result = result['output'].strip()
            except Exception as e:
                exe_result = 'Error Action: ' + action
                logger.error(f"Error executing action: {e}")

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

        # Upload tools from autoglm_v package
        import mm_agents.autoglm_v
        tool_dir = os.path.join(os.path.dirname(mm_agents.autoglm_v.__file__), 'tools', 'package')
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
    agent = AutoGLMAgent(
        action_space=args.action_space,
        observation_type=args.observation_type,
        screen_size=(args.screen_width, args.screen_height),
        image_size=(args.image_width, args.image_height),
        max_trajectory_length=args.max_trajectory_length,
        client_password=args.client_password,
        gen_func=token_tracker,
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
            os.makedirs(example_result_dir, exist_ok=True)
            # example start running
            try:
                lib_run_single.run_single_example_autoglm(
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
                    f.write(json.dumps({"Error": f"Time limit exceeded in {domain}/{example_id}"}))
                    f.write("\n")

    env.close()
    if len(scores) > 0:
        logger.info(f"Average score: {sum(scores) / len(scores)}")
    else:
        logger.info("No tasks completed")


def get_unfinished(target_dir, total_file_json, rerun=False, rerun_fail=False):
    """Get unfinished tasks."""
    
    if not os.path.exists(target_dir):
        return total_file_json

    # If rerun is True, return all tasks (ignore existing results)
    if rerun:
        # Clear all existing results
        for domain in os.listdir(target_dir):
            domain_path = os.path.join(target_dir, domain)
            if os.path.isdir(domain_path):
                for example_id in os.listdir(domain_path):
                    if example_id == "onboard":
                        continue
                    example_path = os.path.join(domain_path, example_id)
                    if os.path.isdir(example_path):
                        # Remove all files in the example directory
                        shutil.rmtree(example_path)
                        os.makedirs(example_path, exist_ok=True)
        return total_file_json

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
                    result_file = os.path.join(example_path, "result.txt")
                    if "result.txt" not in os.listdir(example_path):
                        # Task incomplete (no result.txt), clear and re-run
                        print(f"[Cleanup] Removing incomplete task directory: {example_path}")
                        shutil.rmtree(example_path)
                        os.makedirs(example_path, exist_ok=True)
                    else:
                        # Check if we should rerun failed tasks
                        if rerun_fail:
                            try:
                                with open(result_file, "r") as f:
                                    score = float(f.read().strip())
                                if score == 0.0:
                                    # Failed task, clear and re-run
                                    print(f"[Cleanup] Removing failed task (score=0) directory: {example_path}")
                                    shutil.rmtree(example_path)
                                    os.makedirs(example_path, exist_ok=True)
                                    continue
                            except Exception as e:
                                # Error reading result, report and exit instead of silently deleting
                                raise RuntimeError(f"Error reading result file {result_file}: {e}. Please check the file manually.")
                        finished[domain].append(example_id)

    if not finished:
        return total_file_json

    for domain, examples in finished.items():
        if domain in total_file_json:
            total_file_json[domain] = [x for x in total_file_json[domain] if x not in examples]

    return total_file_json


def get_result(target_dir):
    """Get results."""
    if not os.path.exists(target_dir):
        print("New experiment, no result yet.")
        return None

    all_result = []

    for domain in os.listdir(target_dir):
        domain_path = os.path.join(target_dir, domain)
        if os.path.isdir(domain_path):
            for example_id in os.listdir(domain_path):
                example_path = os.path.join(domain_path, example_id)
                if os.path.isdir(example_path):
                    if "result.txt" in os.listdir(example_path):
                        result_path = os.path.join(example_path, "result.txt")
                        try:
                            with open(result_path, "r") as rf:
                                res = rf.read().strip()
                                if res.lower() == "true":
                                    score = 1.0
                                else:
                                    score = float(res)
                        except Exception:
                            score = 0.0
                        all_result.append(score)

    if not all_result:
        print("New experiment, no result yet.")
        return None
    else:
        print("Current Success Rate:", sum(all_result) / len(all_result) * 100, "%")
        return all_result


if __name__ == "__main__":
    ####### The complete version of the list of examples #######
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = config()
    
    # Clean up existing osworld containers before starting (docker only)
    if args.provider_name == "docker":
        cleanup_osworld_containers()
    
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
    )
    left_info = ""
    for domain in test_file_list:
        left_info += f"{domain}: {len(test_file_list[domain])}\n"
    logger.info(f"Left tasks:\n{left_info}")

    get_result(args.result_dir)
    test(args, test_file_list)
    
    # Call summary() from utils.py after all tasks are completed
    logger.info("Generating summary...")
    summary(args.result_dir, test_all_meta)
