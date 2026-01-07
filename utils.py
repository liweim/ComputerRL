import os
import json
import numpy as np

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
            else:
                if os.path.exists(score_file):
                    print(f"not found: {execution_log_file}")
                continue

    num_tasks = len(all_scores)
    num_tasks_with_log = len(all_costs)  # Number of tasks with execution_log
    avg_score = np.mean(all_scores)
    # avg_score_50 = np.sum(all_scores_50) / num_tasks
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
            # "score_50": avg_score_50,
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
                # "score_50": avg_score_50,
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

    print(json.dumps(detailed_stats['summary'], indent=2, ensure_ascii=False))
    return detailed_stats

if __name__ == "__main__":
    summary('/data1/lwm/projects/ComputerRL/results/autoglm_computer_use/a11y_tree/autoglm-os', '/data1/lwm/projects/ComputerRL/evaluation_examples/test_small.json')