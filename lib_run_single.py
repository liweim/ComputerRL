import datetime
import json
import logging
import os
import time
from wrapt_timeout_decorator import *

logger = logging.getLogger("desktopenv.experiment")


def run_single_example(agent, env, example, max_steps, instruction, args, example_result_dir, scores):
    runtime_logger = setup_logger(example, example_result_dir)
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
            with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                f.write(json.dumps({
                    "step_num": step_idx + 1,
                    "action_timestamp": action_timestamp,
                    "action": action,
                    "response": response,
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "screenshot_file": f"step_{step_idx + 1}_{action_timestamp}.png"
                }))
                f.write("\n")
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


def setup_logger(example, example_result_dir):
    runtime_logger = logging.getLogger(f"desktopenv.example.{example['id']}")
    runtime_logger.setLevel(logging.DEBUG)
    runtime_logger.addHandler(logging.FileHandler(os.path.join(example_result_dir, "runtime.log")))
    return runtime_logger

def run_single_example_human(env, example, max_steps, instruction, args, example_result_dir, scores):
    runtime_logger = setup_logger(example, example_result_dir)
    env.reset(task_config=example)
    time.sleep(60) # Wait for the environment to be ready
    obs = env._get_obs() # Get the initial observation
    
    # Save initial screenshot
    with open(os.path.join(example_result_dir, "initial_state.png"), "wb") as _f:
        _f.write(obs['screenshot'])
    
    # Save trajectory information
    with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
        f.write(json.dumps({
            "instruction": instruction,
            "initial_state": "initial_state.png"
        }))
        f.write("\n")
    
    # Evaluate the result
    result = env.evaluate()
    logger.info("Result: %.2f", result)
    scores.append(result)
    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")

def run_single_example_autoglm(agent, env, example, max_steps, instruction, args, example_result_dir, scores):
    runtime_logger = setup_logger(example, example_result_dir)
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
    
    # Create operations directory like hisa.py
    operations_dir = os.path.join(example_result_dir, "operations")
    os.makedirs(operations_dir, exist_ok=True)
    
    env.controller.start_recording()
    while not done and step_idx < max_steps:
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
            # Capture the timestamp before executing the action
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
            
            # Record step start time
            step_start_time = time.time()
            
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
            
            with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                f.write(json.dumps({
                    "step_num": step_idx + 1,
                    "action_timestamp": action_timestamp,
                    "action": action if isinstance(action, (str, dict, list)) else str(action),
                    "response": response,
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "screenshot_file": f"operations/{screenshot_file}"
                }))
                f.write("\n")
            
            # Add to action_logs (ensure all values are JSON serializable)
            exe_result = obs.get("exe_result", "") if "exe_result" in obs else ""
            if isinstance(exe_result, bytes):
                exe_result = exe_result.decode('utf-8', errors='replace')
            response_str = str(response) if response else ""
            action_logs.append({
                "step": step_idx + 1,
                "type": action_type,
                "execution_success": reward >= 0 and not info.get("fail", False),
                "screenshot": screenshot_file,
                "action": str(action),
                "response": response_str,  # Truncate long responses
                "exe_result": str(exe_result) if exe_result else "",
                "step_time": round(step_time, 2),
                "token_usage": step_usage
            })
                
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
                "token_usage": step_usage
            })
            
        step_idx += 1
    
    if not done: # not completed the task yet
        env.action_history.append('FAIL')
    
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
    }
    
    # Save execution_log.json
    with open(os.path.join(example_result_dir, "execution_log.json"), "w", encoding="utf-8") as f:
        json.dump(execution_log, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")
    env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))
