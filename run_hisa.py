#!/usr/bin/env python3
import argparse
import json
import logging
import os
import shutil
import textwrap
from typing import Dict, List, Tuple
from mm_agents.hisa.main import HiSA
import traceback
from utils import summary, save_args_to_settings, setup_logger, get_unfinished
from tqdm import tqdm


def _ensure_vm_resolution(env, width: int, height: int, logger: logging.Logger) -> None:
    if width == 1920 and height == 1080:
        return
        
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
    logger.info(f"VM resolution set to {width}x{height}")


def _attach_resolution_guard(env, width: int, height: int, logger: logging.Logger) -> None:
    """Ensure VM resolution after every env.reset call."""
    original_reset = env.reset

    def guarded_reset(*args, **kwargs):
        result = original_reset(*args, **kwargs)
        _ensure_vm_resolution(env, width, height, logger)
        return result

    env.reset = guarded_reset

def config() -> argparse.Namespace:
    from desktop_env.desktop_env import DesktopEnv

    parser = argparse.ArgumentParser(description="Run dual agent framework evaluation")
    
    # Environment config
    parser.add_argument("--path_to_vm", type=str, default="vm_data/Ubuntu0/Ubuntu0/Ubuntu0.vmx",
                       help="Path to VM file")
    parser.add_argument(
        "--provider_name",
        type=str,
        default="docker",
        help="Virtualization provider (vmware, docker, aws, azure, gcp, virtualbox)",
    )
    parser.add_argument("--snapshot_name", type=str, default="init_state")
    parser.add_argument("--screen_width", type=int, default=1280) #1920
    parser.add_argument("--screen_height", type=int, default=720) #1080
    parser.add_argument("--sleep_after_execution", type=float, default=0.5)
    parser.add_argument("--client_password", type=str, default="password",
                       help="VM client password")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--record", action="store_true", help="Record the execution process")

    # Agent config
    parser.add_argument("--global_planner_model", type=str, default="qwen3.5-9b",
                       help="Model for Global Planner agent")
    parser.add_argument("--visual_grounder_model", type=str, default="gta1-7b",
                       help="Model for Visual Grounder agent")
    parser.add_argument("--visual_grounder_scale", type=float, default=1.0,
                       help="Scale factor for visual grounder image preprocessing (default: 1.0)")
    parser.add_argument("--state_manager_model", type=str, default="qwen3.5-9b",
                       help="Model for auxiliary tasks (step abstraction, context refinement, pattern induction, etc.)")
    parser.add_argument("--max_steps", type=int, default=15,
                       help="Maximum steps for Global Planner")
    parser.add_argument("--wo_pattern", action="store_true", help="Disable pattern induction (pattern induction is enabled by default)")
    parser.add_argument("--wo_roi", action="store_true",
                       help="Disable ROI cropping (ROI cropping is enabled by default, reduces token usage)")
    parser.add_argument("--roi_margin", type=int, default=50,
                       help="Margin around ROI when cropping (default: 50)")
    parser.add_argument("--refine_period", type=int, default=5,
                       help="Period to refine (default: 5)")
    parser.add_argument("--bash_timeout", type=int, default=60,
                       help="Timeout for bash script execution in seconds (default: 300)")
    parser.add_argument("--wo_step", action="store_true",
                       help="Skip step abstraction and use full conversation history")
    parser.add_argument("--wo_refinement", action="store_true",
                       help="Disable context refinement and use sliding window")
    parser.add_argument("--sliding_window_size", type=int, default=5,
                       help="Sliding window size (number of conversation turns to keep) (default: 5)")

    # Task config
    parser.add_argument("--domain", type=str, default="all")
    parser.add_argument("--test_all_meta_path", type=str, default=os.path.join('evaluation_examples', 'test_one.json'))
    parser.add_argument("--test_config_base_dir", type=str, default="evaluation_examples/examples")
    parser.add_argument("--rerun", action="store_true", help="Rerun tests that have already been run")
    parser.add_argument("--rerun_fail", action="store_true", help="Rerun failed tests")
    parser.add_argument("--get_score", action="store_true", help="Get scores")

    # RAG config
    parser.add_argument("--rag", action='store_true', help="Enable RAG context")
    parser.add_argument("--rag_topk", type=int, default=4)
    parser.add_argument("--summarize_rag", action='store_true', help="Summarize RAG context")
    parser.add_argument("--rag_filename", type=str, default="retrieved_chunk_size_512_chunk_overlap_20_topk_4_embed_bge-large-en-v1.5.txt")
    parser.add_argument("--pattern_dir", type=str, default="D:/projects/qdrant/qdrant_storage", help="Qdrant storage directory")
    parser.add_argument("--use_qdrant_server", action="store_true", help="Use Qdrant server, otherwise use local file storage")
    parser.add_argument("--qdrant_server_url", type=str, default="http://localhost:6333", help="Qdrant server URL")

    # Output config
    parser.add_argument("--result_dir", type=str, default="./results/dual_agent",
                       help="Directory to save results")
    parser.add_argument("--log_level", type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                       default='INFO', help="Set the logging level")
    
    args = parser.parse_args()
    result_name = os.path.basename(args.result_dir)
    boot_logger = setup_logger(result_name, args.log_level)

    args.env = DesktopEnv(
        provider_name=args.provider_name,
        path_to_vm=args.path_to_vm,
        snapshot_name=args.snapshot_name,
        screen_size=(args.screen_width, args.screen_height),
        headless=args.headless,
        require_a11y_tree=False
    )
    _attach_resolution_guard(
        args.env,
        args.screen_width,
        args.screen_height,
        boot_logger,
    )
    _ensure_vm_resolution(
        args.env,
        args.screen_width,
        args.screen_height,
        boot_logger,
    )
    args.logger = boot_logger
    return args

def process_single_task(
    domain: str,
    task_id: str,
    cfg: dict,
    logger: logging.Logger,
    args: argparse.Namespace,
) -> Tuple[str, float]:
    """Process a single task with the dual agent framework."""
    # Extract parameters
    result_dir = args.result_dir
    global_planner_model = args.global_planner_model
    visual_grounder_model = args.visual_grounder_model
    visual_grounder_scale = args.visual_grounder_scale
    state_manager_model = args.state_manager_model
    screen_width = args.screen_width
    screen_height = args.screen_height
    max_steps = args.max_steps
    sleep_after_execution = args.sleep_after_execution
    client_password = args.client_password
    rag = args.rag
    rag_topk = args.rag_topk
    rag_filename = args.rag_filename
    summarize_rag = args.summarize_rag
    test_config_base_dir = args.test_config_base_dir

    logger.info(f"[Processing task] {domain}/{task_id}")
    
    # Setup result directory
    save_dir = os.path.join(result_dir, f"{domain}/{task_id}")
    
    framework = None
    try:
        # Initialize framework
        framework = HiSA(
            env=args.env,
            global_planner_model=global_planner_model,
            visual_grounder_model=visual_grounder_model,
            visual_grounder_scale=visual_grounder_scale,
            state_manager_model=state_manager_model,
            client_password=client_password,
            screen_width=screen_width,
            screen_height=screen_height,
            sleep_after_execution=sleep_after_execution,
            max_steps=max_steps,
            save_dir=save_dir,
            record=args.record,
            wo_pattern=args.wo_pattern,
            wo_roi=args.wo_roi,
            roi_margin=args.roi_margin,
            refine_period=args.refine_period,
            bash_timeout=args.bash_timeout,
            pattern_dir=args.pattern_dir,
            use_qdrant_server=args.use_qdrant_server,
            qdrant_server_url=args.qdrant_server_url,
            wo_step=args.wo_step,
            wo_refinement=args.wo_refinement,
            sliding_window_size=args.sliding_window_size
        )

        # Execute task
        logger.info(f"[Domain]: {domain}")
        logger.info(f"[Example ID]: {task_id}")
        logger.info(f"[Instruction]: {cfg['instruction']}")

        # Add domain to task config
        cfg['domain'] = domain
        score = framework.execute_task(cfg)

        # Save results
        with open(os.path.join(save_dir, "result.txt"), "w") as f:
            f.write(str(score))

        # Read execution log for statistics
        execution_log_path = os.path.join(save_dir, "execution_log.json")
        if os.path.exists(execution_log_path):
            with open(execution_log_path, "r") as f:
                execution_log = json.load(f)
                stats = execution_log.get("statistics", {})
                gui_ops = stats.get("cua_steps", 0)
                code_ops = stats.get("coding_steps", 0)
                total_cost = stats.get("total_cost", 0)

                logger.info(f"Task {domain}/{task_id} completed with score: {score}")
                logger.info(f"Total operations: {gui_ops + code_ops} (GUI: {gui_ops}, Code: {code_ops})")
                logger.info(f"Total cost: ${total_cost:.4f}")
        else:
            logger.info(f"Task {domain}/{task_id} completed with score: {score}")
        return domain, score

    except Exception as e:
        logger.error(f"Error processing task {domain}/{task_id}")
        logger.error(traceback.format_exc())
        score = 0.0

        # Save error information
        with open(os.path.join(save_dir, "result.txt"), "w") as f:
            f.write(str(score))
        with open(os.path.join(save_dir, "err_reason.txt"), "w") as f:
            f.write(f"Fatal error: {str(e)}")

        return domain, 0.0

    finally:
        # Always cleanup to release resources (especially Qdrant lock)
        if framework is not None:
            try:
                framework.cleanup()
            except Exception as cleanup_error:
                logger.warning(f"Error during cleanup: {cleanup_error}")


def filter_tasks(args, test_all_meta: dict, logger) -> List[tuple]:
    """Filter tasks based on rerun/rerun_fail flags."""
    unfinished = get_unfinished(
        args.result_dir,
        test_all_meta,
        rerun=args.rerun,
        rerun_fail=args.rerun_fail,
        logger=logger,
    )
    tasks_to_run = []
    for domain in unfinished:
        for example_id in unfinished[domain]:
            tasks_to_run.append((domain, example_id))
    return tasks_to_run

def run(args, logger=None, tasks=None):
    """
    Run evaluation tasks.
    
    Args:
        args: Command line arguments
        logger: Logger instance (optional, will create if not provided)
        tasks: List of (domain, task_id) tuples (optional, will build from file if not provided)
    """
    # Setup logging configuration
    result_name = os.path.basename(args.result_dir)
    
    # Build tasks if not provided
    if tasks is None:
        with open(args.test_all_meta_path, encoding="utf-8") as f:
            test_all_meta = json.load(f)
        
        if args.domain != "all":
            test_all_meta = {args.domain: test_all_meta[args.domain]}

        task_filter_logger = logger
        if task_filter_logger is None and hasattr(args, "logger"):
            task_filter_logger = args.logger
        tasks = filter_tasks(args, test_all_meta, task_filter_logger)
    
    if not args.get_score:
        if logger is None and hasattr(args, "logger"):
            logger = args.logger
        if logger is None:
            logger = setup_logger(result_name, args.log_level)

        save_args_to_settings(args)

        scores: Dict[str, List[float]] = {}
        
        # Execute all tasks
        if not tasks:
            logger.info("No tasks to process.")
        else:
            for domain, task_id in tqdm(tasks, desc="Processing tasks"):
                # Prepare task directory and config
                target_dir = os.path.join(args.result_dir, f"{domain}/{task_id}")
                cfg_path = os.path.join(args.test_config_base_dir, f"{domain}/{task_id}/{task_id}.json")
                if not os.path.exists(cfg_path):
                    cfg_path = os.path.join(args.test_config_base_dir, f"{domain}/{task_id}.json")
                cfg = json.load(open(cfg_path, 'r', encoding='utf-8'))
                
                # Clean up existing directory and prepare for execution
                if os.path.exists(target_dir):
                    shutil.rmtree(target_dir)
                os.makedirs(target_dir, exist_ok=True)
                
                result_domain, score = process_single_task(domain, task_id, cfg, logger, args)
                
                # Collect scores
                if result_domain not in scores:
                    scores[result_domain] = []
                scores[result_domain].append(score)

    # Calculate and display final results with cost information
    # Summary accepts tasks list directly
    if tasks:
        summary(args.result_dir, tasks)

if __name__ == "__main__":
    args = config()
    run(args)
