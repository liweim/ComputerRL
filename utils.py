#!/usr/bin/env python3
"""
Common utilities for handling context construction including RAG and verbose instruction.
"""

import os
from typing import Tuple, Union
import json
import numpy as np
import datetime
import logging
import sys
from PIL import Image
import cv2
import re
import math


def get_change_roi(
    image1: Union[Image.Image, np.ndarray, str],
    image2: Union[Image.Image, np.ndarray, str],
    margin: int = 50,
) -> Union[Tuple[int, int, int, int], Tuple[Tuple[int, int, int, int], Image.Image]]:
    # Load image
    def load_image(img):
        if isinstance(img, str):
            # File path
            return cv2.imread(img)
        elif isinstance(img, Image.Image):
            # PIL Image
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        elif isinstance(img, np.ndarray):
            # numpy array
            return img
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")

    img1 = load_image(image1)
    img2 = load_image(image2)

    # Ensure both images have the same dimensions
    if img1.shape != img2.shape:
        raise ValueError(
            f"Images must have the same dimensions. "
            f"Got {img1.shape} and {img2.shape}"
        )

    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2

    # Calculate pixel differences
    diff = cv2.absdiff(gray1, gray2)

    # Apply threshold to get binary difference map
    _, binary = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY)

    # Find all changed pixels
    coords = cv2.findNonZero(binary)

    if coords is None:
        return None, None

    # Get bounding box of changed region
    x, y, w, h = cv2.boundingRect(coords)

    # Add margin
    height, width = img1.shape[:2]
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(width, x + w + margin)
    y2 = min(height, y + h + margin)

    # Crop original image (return ROI from second image)
    if isinstance(image2, Image.Image):
        cropped1 = image1.crop((x1, y1, x2, y2))
        cropped2 = image2.crop((x1, y1, x2, y2))
    else:
        # Crop from numpy array and convert to PIL Image
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) if len(img1.shape) == 3 else img1
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) if len(img2.shape) == 3 else img2
        cropped1 = Image.fromarray(img1_rgb[y1:y2, x1:x2])
        cropped2 = Image.fromarray(img2_rgb[y1:y2, x1:x2])

    # print(f"cropped size from {Image.fromarray(img1).size} to {cropped1.size}")
    # timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    # cropped1.save(f"results/tmp/cropped1_{timestamp}.png")
    # cropped2.save(f"results/tmp/cropped2_{timestamp}.png")
    return cropped1, cropped2


def count_images_in_messages(messages: list) -> int:
    """Count the number of images in message list"""
    count = 0
    for message in messages:
        if isinstance(message.get("content"), list):
            for content in message["content"]:
                if content.get("type") in ["image_url", "image", "input_image"]:
                    count += 1
    return count


def smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
    max_ratio=200,
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > max_ratio:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {max_ratio}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def serialize_json(obj):
    """Convert objects to JSON serializable format"""
    if hasattr(obj, "__dict__"):
        # For objects with __dict__, convert to dict but exclude non-serializable items
        result = {}
        for key, value in obj.__dict__.items():
            try:
                json.dumps(value)  # Test if value is serializable
                result[key] = value
            except (TypeError, ValueError):
                result[key] = str(value)  # Convert to string if not serializable
        return result
    elif isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            try:
                json.dumps(value)  # Test if value is serializable
                result[key] = value
            except (TypeError, ValueError):
                result[key] = str(value)  # Convert to string if not serializable
        return result
    elif isinstance(obj, (list, tuple)):
        return [serialize_json(item) for item in obj]
    else:
        try:
            json.dumps(obj)  # Test if obj is serializable
            return obj
        except (TypeError, ValueError):
            return str(obj)  # Convert to string if not serializable


def save_args_to_settings(args):
    """Save args to settings.txt in the result subdirectory"""
    os.makedirs(args.result_dir, exist_ok=True)
    settings_file = os.path.join(args.result_dir, "settings.txt")

    with open(settings_file, "w", encoding="utf-8") as f:
        args_dict = vars(args)
        for key, value in sorted(args_dict.items()):
            f.write(f"{key} = {value}\n")

def setup_logger(result_name, log_level):
    datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

    logger = logging.getLogger()
    # Remove any existing handlers (e.g., from logging.basicConfig) to avoid mixed formats.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    log_level = getattr(logging, log_level.upper())
    logger.setLevel(log_level)

    datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

    log_folder = f"logs/{result_name}"
    os.makedirs(log_folder, exist_ok=True)
    error_handler = logging.FileHandler(
        os.path.join(log_folder, "{:}-error-{:}.log".format(result_name, datetime_str)),
        encoding="utf-8",
    )
    debug_handler = logging.FileHandler(
        os.path.join(log_folder, "{:}-debug-{:}.log".format(result_name, datetime_str)),
        encoding="utf-8",
    )
    stdout_handler = logging.StreamHandler(sys.stdout)

    error_handler.setLevel(logging.ERROR)
    debug_handler.setLevel(logging.DEBUG)
    stdout_handler.setLevel(log_level)

    formatter = logging.Formatter(
        fmt="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] \x1b[0m%(message)s"
    )
    error_handler.setFormatter(formatter)
    debug_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)

    stdout_handler.addFilter(logging.Filter("desktopenv"))

    logger.addHandler(error_handler)
    logger.addHandler(debug_handler)
    logger.addHandler(stdout_handler)

    logger = logging.getLogger("desktopenv")
    return logger


def postprocess_action(action):
    new_action = ""
    if "pyautogui.scroll" in action:
        match = re.findall(r"pyautogui\.scroll\((.*?)\)", action)
        if len(match) > 0:
            scroll_amount = match[0].split(",")[0].strip()
            if float(scroll_amount) > 5:
                new_action = action.replace(scroll_amount, "5")
            elif float(scroll_amount) < -5:
                new_action = action.replace(scroll_amount, "-5")
    if "pyautogui.sleep" in action:
        match = re.findall(r"pyautogui\.sleep\((.*?)\)", action)
        for sleep_amount in match:
            if float(sleep_amount) < 0.5:
                new_action = action.replace(sleep_amount, "0.5")
    if "time.sleep" in action:
        match = re.findall(r"time\.sleep\((.*?)\)", action)
        for sleep_amount in match:
            if float(sleep_amount) < 0.5:
                new_action = action.replace(sleep_amount, "0.5")

    if new_action:
        return new_action
    else:
        return action


def summary(result_dir, test_all_meta):
    """
    Generate summary from test results.
    
    Args:
        result_dir: Path to results directory
        test_all_meta: Can be:
            - str: Path to JSON file
            - dict: {domain: [example_id, ...]}
            - list: [(domain, example_id), ...]
    """
    # Handle different input types
    if type(test_all_meta) == str:
        with open(test_all_meta, "r", encoding="utf-8") as f:
            test_all_meta = json.load(f)
    elif type(test_all_meta) == list:
        # Convert list of tuples to dict
        meta_dict = {}
        for domain, example_id in test_all_meta:
            if domain not in meta_dict:
                meta_dict[domain] = []
            meta_dict[domain].append(example_id)
        test_all_meta = meta_dict

    all_scores = []
    all_scores_50 = []
    all_costs = []
    all_prompt_tokens = []
    all_completion_tokens = []
    all_image_counts = []
    all_execution_times = []
    stats = {}
    global_model_usage = {}  # Track usage across all models
    count_remain = 0  # Tasks without result.txt
    count_errors = 0  # Tasks with err_reason.txt

    for domain in test_all_meta:
        stats[domain] = {
            "score": [],
            "cost": [],
            "gui_steps": [],
            "code_steps": [],
            "total_steps": [],
            "execution_time": [],
            "prompt_tokens": [],
            "completion_tokens": [],
            "image_counts": [],
        }

        for ex_id in test_all_meta[domain]:
            score_file = os.path.join(result_dir, f"{domain}/{ex_id}/result.txt")
            execution_log_file = os.path.join(result_dir, f"{domain}/{ex_id}/execution_log.json")
            error_file = os.path.join(result_dir, f"{domain}/{ex_id}/err_reason.txt")

            # --- 1. Get Score ---
            if os.path.exists(score_file):
                with open(score_file, "r") as f:
                    try:
                        score = eval(f.read()) * 100
                    except:
                        score = 0
            else:
                # If no result file exists, treat score as 0 and count as remaining task
                score = 0
            if score > 0 and os.path.exists(error_file):
                os.remove(error_file)
            if not os.path.exists(score_file) or os.path.exists(error_file):
                count_remain += 1
            
            all_scores.append(score)
            stats[domain]["score"].append(score)

            # --- 2. Check for Errors ---
            # If an error file exists, skip statistics and fail logging after recording the score.
            # This ensures we don't record cost/tokens or add it to 'fails'.
            if os.path.exists(error_file):
                print(f"Error file exists: {error_file}")
                assert score == 0, f"Score is not 0 when error file exists: {error_file}"
                count_errors += 1
                continue 

            # --- 3. Process Execution Log ---
            # Logic reaches here only if err_reason.txt does not exist.
            
            if os.path.exists(execution_log_file):
                try:
                    with open(execution_log_file, "r", encoding="utf-8") as f:
                        execution_log = json.load(f)

                    execution_stats = execution_log.get("statistics", {})
                    
                    # Extract basic data
                    cost = execution_stats.get("total_cost", 0)
                    prompt_tokens = execution_stats.get("prompt_tokens", 0) / 1e3
                    completion_tokens = execution_stats.get("completion_tokens", 0) / 1e3
                    image_count = execution_stats.get("image_count", 0)
                    execution_time = execution_stats.get("execution_time", 0)
                    
                    # Extract step data
                    if "cua_steps" in execution_stats:
                        gui_steps = execution_stats.get("cua_steps", 0)
                        code_steps = execution_stats.get("coding_steps", 0)
                    else:
                        gui_steps = execution_stats.get("total_steps", 0)
                        code_steps = 0

                    # Accumulate Model Usage
                    local_model_usage = execution_stats.get("model_usage", {})
                    for model, usage in local_model_usage.items():
                        if model not in global_model_usage:
                            global_model_usage[model] = {
                                "model_name": usage.get("model_name", "unknown"),
                                "cost": 0, "prompt_tokens": 0, "completion_tokens": 0
                            }
                        global_model_usage[model]["cost"] += usage.get("cost", 0)
                        global_model_usage[model]["prompt_tokens"] += usage.get("prompt_tokens", 0) / 1e3
                        global_model_usage[model]["completion_tokens"] += usage.get("completion_tokens", 0) / 1e3
                    
                    # --- 4. Record Statistics ---
                    # Only record these if no error occurred and execution_log exists
                    all_costs.append(cost)
                    all_prompt_tokens.append(prompt_tokens)
                    all_completion_tokens.append(completion_tokens)
                    all_image_counts.append(image_count)
                    all_execution_times.append(execution_time)
                    
                    stats[domain]["cost"].append(cost)
                    stats[domain]["gui_steps"].append(gui_steps)
                    stats[domain]["code_steps"].append(code_steps)
                    stats[domain]["total_steps"].append(gui_steps + code_steps)
                    stats[domain]["execution_time"].append(execution_time)
                    stats[domain]["prompt_tokens"].append(prompt_tokens)
                    stats[domain]["completion_tokens"].append(completion_tokens)
                    stats[domain]["image_counts"].append(image_count)
                    if gui_steps+code_steps > 50:
                        all_scores_50.append(0)
                    else:
                        all_scores_50.append(score)
                except:
                    print(f"error loading execution_log_file: {execution_log_file}")
                    continue
            else:
                if os.path.exists(score_file):
                    print(f"not found: {execution_log_file}")
                continue

    num_tasks = len(all_scores)
    num_tasks_with_log = len(all_costs)  # Number of tasks with execution_log
    avg_score = np.mean(all_scores)
    avg_score_50 = np.sum(all_scores_50) / num_tasks
    total_cost = sum(all_costs)

    # Calculate total operations and tokens
    total_gui_steps = sum(
        sum(stats[domain]["gui_steps"]) for domain in stats
    )
    total_code_steps = sum(
        sum(stats[domain]["code_steps"]) for domain in stats
    )
    total_steps = total_gui_steps + total_code_steps
    total_prompt_tokens = sum(all_prompt_tokens)
    total_completion_tokens = sum(all_completion_tokens)
    total_tokens = total_prompt_tokens + total_completion_tokens
    total_image_counts = sum(all_image_counts)
    total_execution_times = sum(all_execution_times)
    
    # Use number of tasks with execution_log to calculate averages
    avg_cost = total_cost / num_tasks_with_log if num_tasks_with_log > 0 else 0
    avg_steps = total_steps / num_tasks_with_log if num_tasks_with_log > 0 else 0
    avg_gui_steps = total_gui_steps / num_tasks_with_log if num_tasks_with_log > 0 else 0
    avg_code_steps = total_code_steps / num_tasks_with_log if num_tasks_with_log > 0 else 0
    avg_prompt_tokens = total_prompt_tokens / num_tasks_with_log if num_tasks_with_log > 0 else 0
    avg_completion_tokens = total_completion_tokens / num_tasks_with_log if num_tasks_with_log > 0 else 0
    avg_total_tokens = total_tokens / num_tasks_with_log if num_tasks_with_log > 0 else 0
    avg_image_counts = total_image_counts / num_tasks_with_log if num_tasks_with_log > 0 else 0
    avg_execution_time = total_execution_times / num_tasks_with_log if num_tasks_with_log > 0 else 0

    # Save detailed statistics as JSON
    detailed_stats = {
        "summary": {
            "score": avg_score,
            "score_50": avg_score_50,
            "total_tasks": num_tasks,
            "completed_tasks": num_tasks_with_log,
            "left_tasks": count_remain,  # All incomplete tasks
            "error_tasks": count_errors,  # Only tasks with err_reason.txt
            "total": {
                "cost": total_cost,
                "tokens": total_tokens,
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "image_counts": total_image_counts,
                "steps": total_steps,
                "cua_steps": total_gui_steps,
                "code_steps": total_code_steps,
                "execution_time": total_execution_times,
            },
            "average": {
                "score": avg_score,
                "score_50": avg_score_50,
                "cost": avg_cost,
                "tokens": avg_total_tokens,
                "prompt_tokens": avg_prompt_tokens,
                "completion_tokens": avg_completion_tokens,
                "image_counts": avg_image_counts,
                "steps": avg_steps,
                "cua_steps": avg_gui_steps,
                "code_steps": avg_code_steps,
                "execution_time": avg_execution_time,
            },
            "domain_score": {
                domain: np.mean(stats[domain]["score"]) if len(stats[domain]["score"]) > 0 else 0
                for domain in test_all_meta
            },
            "model_usage": global_model_usage,
        },
        "domain_breakdown": {
            domain: {
                "score": np.mean(stats[domain]["score"]),
                "cost": np.mean(stats[domain]["cost"]),
                "tokens": np.mean(stats[domain]["prompt_tokens"])
                + np.mean(stats[domain]["completion_tokens"]),
                "prompt_tokens": np.mean(stats[domain]["prompt_tokens"]),
                "completion_tokens": np.mean(
                    stats[domain]["completion_tokens"]
                ),
                "image_counts": np.mean(stats[domain]["image_counts"]),
                "steps": np.mean(stats[domain]["total_steps"]),
                "cua_steps": np.mean(stats[domain]["gui_steps"]),
                "code_steps": np.mean(stats[domain]["code_steps"]),
                "execution_time": np.mean(stats[domain]["execution_time"]),
            }
            for domain in test_all_meta
        },
    }

    with open(os.path.join(result_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(detailed_stats, f, indent=2, ensure_ascii=False)

    summary_stats = detailed_stats['summary']
    # print(json.dumps(summary_stats, indent=2, ensure_ascii=False))

    total_tasks = summary_stats['total_tasks']
    left_tasks = summary_stats['left_tasks']
    error_tasks = summary_stats['error_tasks']
    avg_score = summary_stats['score']
    avg_score_50 = summary_stats['score_50']
    avg_cost = summary_stats['average']['cost']
    avg_total_tokens = summary_stats['average']['tokens']
    avg_prompt_tokens = summary_stats['average']['prompt_tokens']
    avg_completion_tokens = summary_stats['average']['completion_tokens']
    avg_steps = summary_stats['average']['steps']
    avg_execution_time = summary_stats['average']['execution_time']
    print(f"Total tasks: {total_tasks}, Left tasks: {left_tasks}, Error tasks: {error_tasks}")
    print(f"score, score_50, cost, tokens, prompt_tokens, completion_tokens, steps, execution_time:\n{avg_score:.3f}\t{avg_score_50:.3f}\t{avg_cost:.3f}\t{avg_total_tokens:.3f}\t{avg_prompt_tokens:.3f}\t{avg_completion_tokens:.3f}\t{avg_steps:.3f}\t{avg_execution_time:.3f}")

    return detailed_stats

if __name__ == "__main__":
    #autoglm-os_baseline, hisa_wo_pattern
    # summary('/data1/lwm/projects/ComputerRL/results/hisa_gta1_7b_wo_step_refinement_pattern', '/data1/lwm/projects/ComputerRL/evaluation_examples/test_small.json')
    # summary('/data1/lwm/projects/ComputerRL/results/autoglm-os_baseline', '/data1/lwm/projects/ComputerRL/evaluation_examples/test_small.json')
    summary('/data1/lwm/projects/ComputerRL/results/autoglm-os_gta1_7b_recovery', '/data1/lwm/projects/ComputerRL/evaluation_examples/test_small.json')
