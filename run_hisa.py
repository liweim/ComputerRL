#!/usr/bin/env python3
import argparse
import json
import logging
import os
import shutil
import requests
from typing import Dict, List, Tuple
from mm_agents.hisa.main import HiSA
import traceback
import docker
from utils import summary, save_args_to_settings, setup_logger
from tqdm import tqdm
import run_autoglm_v
from run_autoglm_v import DesktopEnv

global_logger = None  # Will be initialized in run function


def config() -> argparse.Namespace:
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
    parser.add_argument("--screen_width", type=int, default=1920)
    parser.add_argument("--screen_height", type=int, default=1080)
    parser.add_argument("--sleep_after_execution", type=float, default=0.5)
    parser.add_argument("--client_password", type=str, default="password",
                       help="VM client password")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--record", action="store_true", help="Record the execution process")

    # Agent config
    parser.add_argument("--global_planner_model", type=str, default="autoglm-os",
                       help="Model for Global Planner agent")
    parser.add_argument("--global_planner_temperature", type=float, default=0.2,
                       help="Temperature for Global Planner model")
    parser.add_argument("--global_planner_top_p", type=float, default=0.1,
                       help="Top-p for Global Planner model")
    parser.add_argument("--global_planner_max_tokens", type=int, default=256,
                       help="Max tokens for Global Planner model")
    parser.add_argument("--global_planner_repetition_penalty", type=float, default=1.0,
                       help="Repetition penalty for Global Planner model")

    parser.add_argument("--visual_grounder_model", type=str, default="autoglm-os",
                       help="Model for Visual Grounder agent")
    parser.add_argument("--visual_grounder_temperature", type=float, default=0.2,
                       help="Temperature for Visual Grounder model")
    parser.add_argument("--visual_grounder_top_p", type=float, default=0.1,
                       help="Top-p for Visual Grounder model")
    parser.add_argument("--visual_grounder_max_tokens", type=int, default=256,
                       help="Max tokens for Visual Grounder model")
    parser.add_argument("--visual_grounder_repetition_penalty", type=float, default=1.0,
                       help="Repetition penalty for Visual Grounder model")

    parser.add_argument("--state_manager_model", type=str, default="autoglm-os",
                       help="Model for auxiliary tasks (step abstraction, context refinement, pattern induction, etc.)")
    parser.add_argument("--state_manager_temperature", type=float, default=0.2,
                       help="Temperature for State Manager model")
    parser.add_argument("--state_manager_top_p", type=float, default=0.9,
                       help="Top-p for State Manager model")
    parser.add_argument("--state_manager_max_tokens", type=int, default=2048,
                       help="Max tokens for State Manager model")
    parser.add_argument("--state_manager_repetition_penalty", type=float, default=1.0,
                       help="Repetition penalty for State Manager model")

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
                       help="Timeout for bash script execution in seconds (default: 60)")
    parser.add_argument("--wo_step", action="store_true",
                       help="Skip step abstraction and use full conversation history")
    parser.add_argument("--wo_refinement", action="store_true",
                       help="Disable context refinement and use sliding window")
    parser.add_argument("--sliding_window_size", type=int, default=5,
                       help="Sliding window size (number of conversation turns to keep) (default: 5)")
    parser.add_argument("--max_parse_retries", type=int, default=3,
                       help="Maximum number of retries for parsing LLM responses (default: 3)")
    parser.add_argument("--unify_llm", action="store_true",
                       help="Unified LLM (use same model for controller and grounder, default: False means separate models are used)")

    # Task config
    parser.add_argument("--domain", type=str, default="all")
    parser.add_argument("--test_all_meta_path", type=str, default=os.path.join('evaluation_examples', 'test_one.json'))
    parser.add_argument("--test_config_base_dir", type=str, default="evaluation_examples/examples")
    parser.add_argument("--rerun", action="store_true", help="Rerun tests that have already been run")
    parser.add_argument("--rerun_fail", action="store_true", help="Rerun failed tests")
    parser.add_argument("--get_score", action="store_true", help="Get scores")

    # pattern config
    parser.add_argument("--pattern_dir", type=str, default="D:/projects/qdrant/qdrant_storage", help="Qdrant storage directory")
    parser.add_argument("--use_qdrant_server", action="store_true", help="Use Qdrant server, otherwise use local file storage")
    parser.add_argument("--qdrant_server_url", type=str, default="http://localhost:6333", help="Qdrant server URL")

    # docker related
    parser.add_argument("--cleanup_docker", action="store_true", default=False, help="Cleanup docker containers before starting")

    # Output config
    parser.add_argument("--result_dir", type=str, default="./results/dual_agent",
                       help="Directory to save results")
    parser.add_argument("--log_level", type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                       default='INFO', help="Set the logging level")
    
    args = parser.parse_args()

    result_name = os.path.basename(args.result_dir)
    global global_logger
    global_logger = setup_logger(result_name, args.log_level)
    run_autoglm_v.logger = global_logger

    args.env = DesktopEnv(
        provider_name=args.provider_name,
        path_to_vm=args.path_to_vm,
        action_space="autoglm_computer_use",
        screen_size=(args.screen_width, args.screen_height),
        headless=args.headless,
        os_type="Ubuntu",
        require_a11y_tree=False
    )
    return args


def create_llm_function(model_name: str, temperature: float = 0.1, top_p: float = 0.9, max_tokens: int = 2048, repetition_penalty: float = 1.0):
    """Create a callable LLM function from model name string."""
    # Get API configuration from environment
    base_url = os.environ.get('OPENAI_BASE_URL', 'http://localhost:30000/v1')
    api_key = os.environ.get('OPENAI_API_KEY', 'EMPTY')

    def call_llm(messages):
        """Call LLM API with OpenAI compatible interface."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        # Prepare request data with specified parameters
        data = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "stream": False
        }

        url = f"{base_url}/chat/completions"

        try:
            response = requests.post(
                url,
                json=data,
                headers=headers,
                timeout=60.0
            )
            response.raise_for_status()

            result = response.json()
            # Return both content and usage info
            content = result['choices'][0]['message']['content']
            usage = result.get('usage', {})
            return content, usage

        except Exception as e:
            if global_logger:
                global_logger.error(f"Failed to call LLM {model_name}: {e}")
            raise

    return call_llm


class LLM:
    """Track token usage for LLM calls."""
    def __init__(self, model_name: str, temperature: float = 0.1, top_p: float = 0.9, max_tokens: int = 2048, repetition_penalty: float = 1.0):
        self.model_name = model_name
        self.call_llm = create_llm_function(model_name, temperature, top_p, max_tokens, repetition_penalty)
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.total_image_count = 0
        self.last_usage = {}

    def __call__(self, messages):
        result, usage = self.call_llm(messages)

        # Count images in the messages
        image_count = 0
        for msg in messages:
            if isinstance(msg.get('content'), list):
                for item in msg['content']:
                    if item.get('type') in ['image_url', 'input_image']:
                        image_count += 1

        # Store usage info from API response
        self.last_usage = {
            'prompt_tokens': usage.get('prompt_tokens', 0),
            'completion_tokens': usage.get('completion_tokens', 0),
            'total_tokens': usage.get('total_tokens', 0),
            'image_count': image_count,
            'cost': 0.0  # Cost calculation would need model-specific pricing
        }

        # Update total counters
        self.total_prompt_tokens += self.last_usage['prompt_tokens']
        self.total_completion_tokens += self.last_usage['completion_tokens']
        self.total_tokens += self.last_usage['total_tokens']
        self.total_image_count += image_count

        return result

    def get_last_usage(self):
        return self.last_usage

    def reset(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.total_image_count = 0


def cleanup_osworld_containers(logger, remove_running=False):
    """Clean up osworld docker containers before starting.

    Args:
        logger: Logger instance for consistent logging
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
                            logger.info(f"  Stopped and removed running container: {container.name}")
                            removed_count += 1
                        else:
                            skipped_count += 1
                    else:
                        # Remove exited/stopped containers
                        container.remove(force=True)
                        logger.info(f"  Removed exited container: {container.name}")
                        removed_count += 1
                except Exception as e:
                    logger.warning(f"  Failed to remove container {container.name}: {e}")
            logger.info(f"Cleanup completed. Removed: {removed_count}, Skipped (running): {skipped_count}")
        else:
            logger.info("No existing osworld containers found.")
    except Exception as e:
        logger.warning(f"Warning: Failed to cleanup containers: {e}")

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
    unify_llm = args.unify_llm
    max_steps = args.max_steps
    sleep_after_execution = args.sleep_after_execution
    client_password = args.client_password
    max_parse_retries = args.max_parse_retries

    # Convert model name strings to callable functions
    global_planner_model = LLM(
        args.global_planner_model,
        args.global_planner_temperature,
        args.global_planner_top_p,
        args.global_planner_max_tokens,
        args.global_planner_repetition_penalty
    )
    visual_grounder_model = LLM(
        args.visual_grounder_model,
        args.visual_grounder_temperature,
        args.visual_grounder_top_p,
        args.visual_grounder_max_tokens,
        args.visual_grounder_repetition_penalty
    ) if not unify_llm else global_planner_model
    state_manager_model = LLM(
        args.state_manager_model,
        args.state_manager_temperature,
        args.state_manager_top_p,
        args.state_manager_max_tokens,
        args.state_manager_repetition_penalty
    )

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
            state_manager_model=state_manager_model,
            client_password=client_password,
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
            sliding_window_size=args.sliding_window_size,
            max_parse_retries=max_parse_retries,
            unify_llm=unify_llm
        )

        # Execute task
        logger.info(f"[Domain]: {domain}")
        logger.info(f"[Example ID]: {task_id}")

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

def run(args, logger=None, tasks=None):
    """
    Run evaluation tasks.

    Args:
        args: Command line arguments
        logger: Logger instance (optional, will create if not provided)
        tasks: List of (domain, task_id) tuples (optional, will build from file if not provided)
    """

    # Clean up existing osworld containers before starting (docker only, if requested)
    if args.provider_name == "docker" and args.cleanup_docker:
        cleanup_osworld_containers(global_logger)

    # Build tasks if not provided
    if tasks is None:
        with open(args.test_all_meta_path, encoding="utf-8") as f:
            test_all_meta = json.load(f)
        
        if args.domain != "all":
            test_all_meta = {args.domain: test_all_meta[args.domain]}
        
        tasks = []
        for domain in test_all_meta:
            for task_id in test_all_meta[domain]:
                tasks.append((domain, task_id))
    
    if not args.get_score:
        # Use global logger for task processing
        logger = global_logger

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