#!/usr/bin/env python3
import base64
import json
import os
import logging
import traceback
from typing import Optional, Dict, List, Tuple
from utils import serialize_json, get_change_roi
from json_repair import repair_json
from utils import postprocess_action
import re
import ast
from .qdrant import QdrantManager, add_lessons_to_existing
from .embedding import EmbeddingClient
from PIL import Image
import io
import time
import glob
from PIL import Image
from io import BytesIO
from .prompt.hisa_prompt import GLOBAL_PLANNER_PROMPT, CONTEXT_REFINEMENT_PROMPT, FIX_RESPONSE_PROMPT, FIX_RESPONSE_UNIFY_PROMPT, STEP_ABSTRACTION_PROMPT, PATTERN_INDUCTION_PROMPT, PATTERN_SYNTHESIS_PROMPT
from .prompt.procedural_memory import Prompt as AutoGLMPrompt
from .prompt.grounding_agent import GroundingAgent
from .prompt.accessibility_tree_handle import linearize_accessibility_tree, trim_accessibility_tree
from .tools.package.google_chrome import BrowserTools

# ==================== PATTERN MANAGER ====================

class PatternManager:
    """Manage task execution pattern by domain."""

    def __init__(
        self,
        llm=None,
        qdrant_path: str = "D:/projects/qdrant/qdrant_storage",
        embedding_service_url: str = "http://localhost:8000",
        similarity_threshold: float = 0.7,
        use_qdrant_server: bool = False,  # Default to server mode for multi-process
        qdrant_server_url: str = "http://localhost:6333"
    ):
        self.llm = llm
        self.similarity_threshold = similarity_threshold
        self.logger = logging.getLogger("desktopenv.pattern")
        if not os.path.exists(qdrant_path):
            for json_file in glob.glob("../GUIAgent/patterns/*.json"):
                collection_name = os.path.basename(json_file).split(".")[0]
                add_lessons_to_existing(
                    json_file=json_file,
                    collection_name=collection_name,
                    use_server=False,
                    path=qdrant_path
                )
        self.qdrant = QdrantManager(
            path=qdrant_path,
            use_server=use_qdrant_server,
            server_url=qdrant_server_url
        )
        self.embedding_client = EmbeddingClient(service_url=embedding_service_url)
        mode = "server" if use_qdrant_server else "local"
        self.logger.info(f"Vector database ({mode} mode) and embedding service initialized")

    def _ensure_collection(self, collection_name: str):
        """Ensure Qdrant collection exists for a domain."""
        try:
            collections = self.qdrant.list_collections()
            if collection_name not in collections:
                self.qdrant.create_collection(
                    collection_name=collection_name,
                    vector_size=1024,
                    distance="Cosine"
                )
                self.logger.info(f"Created Qdrant collection: {collection_name}")
        except Exception as e:
            self.logger.error(f"Failed to ensure collection {collection_name}: {e}")

    def _save_pattern(self, domain: str, lessons: List[Dict]):
        """Save lessons using Qdrant vector database with deduplication.

        Args:
            domain: The domain to save lessons to
            lessons: List of lesson dicts, each with 'type' and 'lesson' fields
                    type must be: 'success' or 'failure' (determined by LLM from execution)

        Note:
            - Each lesson is vectorized and stored in Qdrant
            - Similar lessons (cosine similarity > threshold) are detected and removed
            - New lessons replace similar old ones
            - Different domains use different Qdrant collections
        """
        try:
            self._ensure_collection(domain)

            # Get current max ID from Qdrant
            try:
                count = self.qdrant.count_points(domain)
                all_points = self.qdrant.scroll_all(domain, limit=1000, with_vectors=False)
                max_id = max([p["id"] for p in all_points], default=0) if all_points else 0
                next_id = max_id + 1
            except:
                next_id = 1

            added_count = 0
            replaced_count = 0

            for lesson_obj in lessons:
                lesson_text = lesson_obj.get("lesson", "")
                lesson_type = lesson_obj.get("type", "failure")

                if not lesson_text:
                    continue

                # Generate embedding for the lesson
                try:
                    lesson_vector = self.embedding_client(lesson_text)
                except Exception as e:
                    self.logger.error(f"Failed to generate embedding: {e}")
                    continue

                # Search for similar lessons
                try:
                    similar_results = self.qdrant.search(
                        collection_name=domain,
                        query_vector=lesson_vector,
                        limit=5,
                        score_threshold=self.similarity_threshold
                    )
                except Exception as e:
                    self.logger.warning(f"Search failed: {e}, assuming no similar lessons")
                    similar_results = []

                # Filter out lessons with type="require" from deletion candidates
                # IMPORTANT: Never delete or modify lessons with type="require"
                deletable_similar = []
                for r in similar_results:
                    similar_type = r.get("payload", {}).get("type", "")
                    if similar_type != "require":
                        deletable_similar.append(r)
                    else:
                        self.logger.info(f"Skipping deletion of require-type lesson (id={r['id']}) - these are protected")

                # Delete similar old lessons (excluding require type)
                if deletable_similar:
                    deletable_ids = [r["id"] for r in deletable_similar]
                    self.logger.info(
                        f"Found {len(deletable_similar)} similar lesson(s) with similarity > {self.similarity_threshold}, "
                        f"replacing them with new lesson"
                    )
                    try:
                        self.qdrant.delete_by_ids(domain, deletable_ids)
                        replaced_count += len(deletable_ids)
                    except Exception as e:
                        self.logger.error(f"Failed to delete similar lessons: {e}")

                # Add new lesson
                try:
                    self.qdrant.insert_points(
                        collection_name=domain,
                        points=[{
                            "id": next_id,
                            "vector": lesson_vector,
                            "payload": {
                                "lesson": lesson_text,
                                "type": lesson_type,
                                "domain": domain
                            }
                        }]
                    )
                    added_count += 1
                    next_id += 1
                except Exception as e:
                    self.logger.error(f"Failed to insert lesson: {e}")

            self.logger.info(
                f"Vector DB update for domain {domain}: "
                f"added {added_count} new lesson(s), replaced {replaced_count} similar lesson(s)"
            )

        except Exception as e:
            self.logger.error(f"Failed to save pattern with vector DB: {e}")
            raise

    def _pattern_induction(self, task_instruction: str, action_logs: List[Dict]) -> List[str]:
        """Use LLM to extract key lessons from task execution.

        Returns:
            List of lesson strings
        """
        if not self.llm:
            return []

        # Use step_abstract directly (already contains step, action, result)
        step_abstracts = []
        for log in action_logs:
            if "step_abstract" in log:
                step_abstracts.append(log["step_abstract"])

        prompt = PATTERN_INDUCTION_PROMPT.format(
            task_instruction=task_instruction,
            step_abstracts='\n'.join(step_abstracts)
        )

        try:
            messages = [
                {"role": "system", "content": "You are an expert at analyzing task execution patterns and extracting the most critical, reusable lessons. Be highly selective - only extract truly valuable insights. CRITICAL: focus only on the execution process."},
                {"role": "user", "content": prompt}
            ]

            response = self.llm(messages)

            # Try to parse as JSON
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            else:
                json_str = response.strip()

            lessons = json.loads(repair_json(json_str))
            if isinstance(lessons, list):
                # Validate that each item is a dict with 'type' and 'lesson'
                validated_lessons = []
                for item in lessons[:3]:  # Max 3 lessons
                    if isinstance(item, dict) and "type" in item and "lesson" in item:
                        # Validate type is success or failure
                        if item["type"] in ["success", "failure"]:
                            validated_lessons.append(item)
                        else:
                            self.logger.warning(f"Invalid lesson type '{item['type']}', skipping")
                    else:
                        self.logger.warning(f"Invalid lesson format: {item}, skipping")
                return validated_lessons
            else:
                self.logger.warning(f"Expected list, got {type(lessons)}")
                return []

        except Exception as e:
            self.logger.error(f"Failed to extract lessons: {e}")
            return []

    def _get_relevant_pattern(self, domain: str, current_task: str) -> str:
        """Retrieve relevant patterns using vector similarity search.

        Returns:
            Actionable advice string based on relevant patterns
        """
        try:
            self._ensure_collection(domain)

            # Check if collection has any points
            try:
                count = self.qdrant.count_points(domain)
                if count == 0:
                    self.logger.info(f"No pattern found in collection {domain}")
                    return ""
            except Exception as e:
                self.logger.warning(f"Failed to check collection count: {e}")
                return ""

            # First, retrieve ALL lessons with type="require" (mandatory requirements)
            require_patterns = []
            try:
                all_require_results = self.qdrant.search_by_filter(
                    collection_name=domain,
                    filter_conditions={"type": "require"},
                    limit=100  # Get all require type lessons
                )
                for result in all_require_results:
                    payload = result["payload"]
                    lesson_text = payload.get("lesson", "")
                    entry = {"id": result["id"], "lesson": lesson_text, "score": result["score"]}
                    require_patterns.append(entry)
                
            except Exception as e:
                self.logger.warning(f"Failed to retrieve require type lessons: {e}")

            # Vectorize current task
            try:
                task_vector = self.embedding_client(current_task)
            except Exception as e:
                self.logger.error(f"Failed to generate task embedding: {e}")
                raise

            # Search for similar lessons (top 5, threshold 0.75 for high quality matching)
            try:
                search_results = self.qdrant.search(
                    collection_name=domain,
                    query_vector=task_vector,
                    limit=5,
                    score_threshold=0.5
                )
            except Exception as e:
                self.logger.error(f"Vector search failed: {e}")
                raise

            # Group by type (excluding require since we already have them all)
            success_patterns = []
            failure_patterns = []

            for result in search_results:
                payload = result["payload"]
                lesson_type = payload.get("type", "failure")
                lesson_text = payload.get("lesson", "")
                score = result["score"]

                entry = {"id": result["id"], "lesson": lesson_text, "score": score}

                # Skip require type here as we already retrieved all of them above
                if lesson_type == "require":
                    continue
                elif lesson_type == "success":
                    success_patterns.append(entry)
                else:
                    failure_patterns.append(entry)

            # Build summary
            pattern_summary = []
            if require_patterns:
                pattern_summary.append("\n--- REQUIREMENTS (MUST FOLLOW) ---")
                for pattern in require_patterns:
                    pattern_summary.append(f"[{pattern['id']}] {pattern['lesson']} (similarity: {pattern['score']:.2f})")

            if success_patterns:
                pattern_summary.append("\n--- SUCCESS Patterns ---")
                for pattern in success_patterns:
                    pattern_summary.append(f"[{pattern['id']}] {pattern['lesson']} (similarity: {pattern['score']:.2f})")

            if failure_patterns:
                pattern_summary.append("\n--- FAILURE Patterns ---")
                for pattern in failure_patterns:
                    pattern_summary.append(f"[{pattern['id']}] {pattern['lesson']} (similarity: {pattern['score']:.2f})")

            if not pattern_summary:
                return ""

            prompt = PATTERN_SYNTHESIS_PROMPT.format(
                current_task=current_task,
                pattern_summary='\n'.join(pattern_summary)
            )

            try:
                messages = [
                    {"role": "system", "content": "You are an expert at analyzing past lessons and providing actionable advice for new tasks."},
                    {"role": "user", "content": prompt}
                ]

                response = self.llm(messages)
                self.logger.info(f"Retrieved {len(pattern_summary)} relevant lesson(s) using vector search")
                return response.strip()

            except Exception as e:
                self.logger.error(f"Failed to summarize patterns: {e}")
                return '\n'.join(pattern_summary)

        except Exception as e:
            self.logger.error(f"Failed to get relevant patterns with vector DB: {e}")
            raise

# ==================== AGENT FRAMEWORK ====================

class HiSA:
    """Cognitive Memory Model Agent."""

    def __init__(
        self,
        env,
        global_planner_model=None,  # Function to call LLM for global planner
        visual_grounder_model=None,  # LLM for visual grounding (e.g., gta1-7b)
        state_manager_model=None,  # Function to call LLM for state manager
        client_password: str = "password",
        screen_width: int = 1280,
        screen_height: int = 720,
        image_width: int = 1280,
        image_height: int = 720,
        sleep_after_execution: float = 0.5,
        max_steps: int = 15,
        save_dir: str = "",
        record: bool = False,
        max_parse_retries: int = 3,
        wo_pattern: bool = False,  # If True, disable pattern induction (default: False means pattern induction is enabled)
        pattern_dir: str = "D:/projects/qdrant/qdrant_storage",
        use_qdrant_server: bool = False,  # Use server mode by default for multi-process
        qdrant_server_url: str = "http://localhost:6333",
        wo_roi: bool = False,  # If True, disable ROI cropping (default: False means ROI cropping is enabled)
        roi_margin: int = 50,  # Margin around ROI when cropping
        refine_period: int = 5,
        bash_timeout: int = 60,  # Timeout for bash script execution in seconds
        wo_step: bool = False,  # If True, skip step abstraction and use full conversation history
        wo_refinement: bool = False,  # If True, disable context refinement and use sliding window
        sliding_window_size: int = 5,  # Sliding window size (number of conversation turns to keep)
        with_image: bool = True,
        with_atree: bool = False,
        tool_in_sys_msg: bool = True,
        glm41v_format: bool = True,
    ):
        self.env = env
        self.global_planner_model = global_planner_model
        self.visual_grounder_model = visual_grounder_model
        self.state_manager_model = state_manager_model
        self.client_password = client_password
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.image_width = image_width
        self.image_height = image_height
        self.sleep_after_execution = sleep_after_execution
        self.max_steps = max_steps
        self.save_dir = save_dir
        self.record = record
        self.max_parse_retries = max_parse_retries
        self.wo_pattern = wo_pattern  # If True, disable pattern induction (default: False means pattern induction is enabled)
        self.wo_roi = wo_roi  # If True, disable ROI cropping (default: False means ROI cropping is enabled)
        self.roi_margin = roi_margin
        self.refine_period = refine_period
        self.bash_timeout = bash_timeout  # Timeout for bash script execution
        self.wo_step = wo_step  # Skip step abstraction if True
        self.wo_refinement = wo_refinement  # Disable context refinement if True
        self.sliding_window_size = sliding_window_size  # Sliding window size for conversation history
        self.with_image = with_image
        self.with_atree = with_atree
        self.tool_in_sys_msg = tool_in_sys_msg
        relative_coordinate = True
        if self.visual_grounder_model is not None:
            model_name = getattr(self.visual_grounder_model, "model_name", "")
            if isinstance(model_name, str) and model_name.startswith("gta1"):
                relative_coordinate = False
        self.relative_coordinate = relative_coordinate
        self.glm41v_format = glm41v_format
        self.grounding_agent = GroundingAgent(
            screen_width=self.screen_width,
            screen_height=self.screen_height,
            relative_coordinate=self.relative_coordinate,
        )

        # Tool list for unified LLM (same as autoglm_v)
        self.tool_list = {
            "libreoffice_calc": "CalcTools",
            "libreoffice_impress": "ImpressTools",
            "libreoffice_writer": "WriterTools",
            "code": "CodeTools",
            "vlc": "VLCTools",
            "google_chrome": "BrowserTools"
        }

        self.logger = logging.getLogger("desktopenv")

        # Initialize token usage tracking (for compatibility with existing code)
        self.global_planner_usage = {"cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "image_count": 0}
        self.visual_grounder_usage = {"cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "image_count": 0}
        self.state_manager_usage = {"cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "image_count": 0}

        # Initialize pattern manager
        if not self.wo_pattern:
            self.pattern_manager = PatternManager(
                llm=self.global_planner_model,
                qdrant_path=pattern_dir,
                similarity_threshold=0.7,
                use_qdrant_server=use_qdrant_server,
                qdrant_server_url=qdrant_server_url
            )
            self.logger.info(f"Pattern manager initialized")

        # Execution state
        self.operation_count = 0
        self.operations_dir = ""
        self.action_logs = []
        self.last_error_feedback = None  # Store error feedback for retry
        self.last_full_summary = None  # Last complete history summary
        self.last_summary_step = 0  # Step number of last summary
        self.step_token_usage = {}  # Store token usage for current step
        self.current_thought = ""  # Store current step's thought for step_abstract
        self.last_tool_output = None  # Store last tool execution result for wo_step mode

    def _call_llm(self, func, messages, usage_tracker):
        """Call LLM function and update usage statistics."""
        if hasattr(func, 'get_last_usage'):
            # TokenTracker style function
            response = func(messages)
            usage = func.get_last_usage()
            usage_tracker["cost"] += usage.get("cost", 0.0)
            usage_tracker["prompt_tokens"] += usage.get("prompt_tokens", 0)
            usage_tracker["completion_tokens"] += usage.get("completion_tokens", 0)
            usage_tracker["image_count"] += usage.get("image_count", 0)
        else:
            # Simple function that returns content
            response = func(messages)
            # For simple functions, we don't have usage info, so we skip updating
        return response

    def _get_usage_snapshot(self) -> Dict:
        """Get current token usage snapshot from all LLMs."""
        snapshot = {
            "global_planner": self.global_planner_usage.copy(),
            "state_manager": self.state_manager_usage.copy()
        }
        if self.visual_grounder_model is not None:
            snapshot["visual_grounder"] = self.visual_grounder_usage.copy()
        return snapshot

    def _calculate_usage_delta(self, before: Dict, after: Dict) -> Dict:
        """Calculate the difference in token usage between two snapshots."""
        delta = {}
        for model in before.keys():
            delta[model] = {
                "cost": after[model]["cost"] - before[model]["cost"],
                "prompt_tokens": after[model]["prompt_tokens"] - before[model]["prompt_tokens"],
                "completion_tokens": after[model]["completion_tokens"] - before[model]["completion_tokens"],
                "image_count": after[model]["image_count"] - before[model]["image_count"]
            }
        return delta

    def _get_llm_usage_dict(self, llm) -> Dict:
        """Fetch usage stats from AbstractLLM-style clients."""
        if llm is None or not hasattr(llm, "get_usage"):
            return {"cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "image_count": 0}
        cost, prompt_tokens, completion_tokens, image_count = llm.get_usage()
        return {
            "cost": cost,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "image_count": image_count,
        }

    def _ground_coordinates(self, description: str, screenshot: bytes) -> Tuple[int, int]:
        """Return grounded (x, y) using the visual grounder."""
        if self.visual_grounder_model is None:
            raise ValueError("Visual grounder model is not configured")

        usage_before = self._get_llm_usage_dict(self.visual_grounder_model)
        img = Image.open(io.BytesIO(screenshot))
        model_name = getattr(self.visual_grounder_model, "model_name", "")
        scale = 1.5 if isinstance(model_name, str) and model_name.startswith("gta1") else 1.0

        py_cmd, reasoning = self.visual_grounder_model.call_cua(
            description,
            img,
            environment="linux",
            screen_width=self.screen_width,
            screen_height=self.screen_height,
            scale=scale,
        )

        usage_after = self._get_llm_usage_dict(self.visual_grounder_model)
        usage_delta = {
            "cost": usage_after["cost"] - usage_before["cost"],
            "prompt_tokens": usage_after["prompt_tokens"] - usage_before["prompt_tokens"],
            "completion_tokens": usage_after["completion_tokens"] - usage_before["completion_tokens"],
            "image_count": usage_after["image_count"] - usage_before["image_count"],
        }
        self.visual_grounder_usage["cost"] += usage_delta["cost"]
        self.visual_grounder_usage["prompt_tokens"] += usage_delta["prompt_tokens"]
        self.visual_grounder_usage["completion_tokens"] += usage_delta["completion_tokens"]
        self.visual_grounder_usage["image_count"] += usage_delta["image_count"]

        if not py_cmd:
            raise ValueError(f"Visual Grounder failed to provide result. Reasoning: {reasoning}")

        if isinstance(py_cmd, tuple) and len(py_cmd) == 2:
            return py_cmd

        raise ValueError(f"Visual Grounder did not return coordinates: {py_cmd}")

    def _parse_agent_call(self, code: str):
        """Parse Agent.<method>(...) call and return (method, args, kwargs)."""
        tree = ast.parse(code)
        if len(tree.body) != 1:
            raise ValueError("Expected a single Agent call")
        node = tree.body[0]
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            raise ValueError("Expected a call expression")
        call = node.value
        if not isinstance(call.func, ast.Attribute):
            raise ValueError("Expected attribute call")
        if not isinstance(call.func.value, ast.Name) or call.func.value.id != "Agent":
            raise ValueError("Expected Agent.<method> call")
        method = call.func.attr
        args = [ast.literal_eval(arg) for arg in call.args]
        kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
        return method, args, kwargs

    def _resolve_agent_call(self, code: str, desc: str, screenshot: bytes):
        """Resolve Agent.click/type calls using visual grounding and return action code."""
        method, args, kwargs = self._parse_agent_call(code)

        if method == "click":
            coordinate = self._ground_coordinates(desc, screenshot)
            num_clicks = kwargs.get("num_clicks", args[1] if len(args) > 1 else 1)
            button_type = kwargs.get("button_type", args[2] if len(args) > 2 else "left")
            return self.grounding_agent.click(coordinate, num_clicks=num_clicks, button_type=button_type)
        if method == "type":
            coordinate = self._ground_coordinates(desc, screenshot)
            text = kwargs.get("text", args[1] if len(args) > 1 else "")
            overwrite = kwargs.get("overwrite", False)
            enter = kwargs.get("enter", False)
            return self.grounding_agent.type(coordinate=coordinate, text=text, overwrite=overwrite, enter=enter)

        if method == "scroll":
            coordinate = self._ground_coordinates(desc, screenshot)
            direction = kwargs.get("direction", args[1] if len(args) > 1 else "down")
            return self.grounding_agent.scroll(coordinate, direction)

        if method == "drag_and_drop":
            desc_text = desc.strip()
            start_desc = ""
            end_desc = ""
            if "->" in desc_text:
                parts = desc_text.split("->", 1)
                start_desc, end_desc = parts[0].strip(), parts[1].strip()
            elif " to " in desc_text.lower():
                parts = re.split(r"\s+to\s+", desc_text, maxsplit=1, flags=re.IGNORECASE)
                if len(parts) == 2:
                    start_desc, end_desc = parts[0].strip(), parts[1].strip()
            elif ";" in desc_text:
                parts = desc_text.split(";", 1)
                start_desc, end_desc = parts[0].strip(), parts[1].strip()

            if not start_desc:
                start_desc = f"{desc_text} (drag start)"
            if not end_desc:
                end_desc = f"{desc_text} (drop target)"

            drag_from = self._ground_coordinates(start_desc, screenshot)
            drop_on = self._ground_coordinates(end_desc, screenshot)
            return self.grounding_agent.drag_and_drop(drag_from, drop_on)

        if method == "exit":
            success = kwargs.get("success", args[0] if len(args) > 0 else True)
            return self.grounding_agent.exit(success=success)

        raise ValueError(f"Grounding not supported for Agent.{method}")


    def _get_context_refinement(self, logs: List[Dict], start_step: int, end_step: int, previous_summary: str = "") -> str:
        """Summarize a segment of action logs with context refinement."""
        
        if not logs and not previous_summary:
            return f"Steps {start_step}~{end_step}: No actions. Suggestion: Continue"

        # Build detailed history of new logs
        if self.wo_step:
            history_lines = []
            
            if logs:
                for i, log in enumerate(logs):
                    step_num = log.get("step", 0)
                    msg_idx = i * 2 
                    
                    if msg_idx + 1 < len(self.conversation_messages):
                        user_msg = self.conversation_messages[msg_idx]
                        assistant_msg = self.conversation_messages[msg_idx + 1]
                        
                        # Extract text content
                        user_text = ""
                        if isinstance(user_msg.get('content'), list):
                            for content_item in user_msg['content']:
                                if content_item.get('type') in ['text', 'input_text']:
                                    user_text = content_item.get('text', '')
                                    break
                        else:
                            user_text = user_msg.get('content', '')
                        
                        assistant_text = ""
                        if isinstance(assistant_msg.get('content'), list):
                            for content_item in assistant_msg['content']:
                                if content_item.get('type') in ['text', 'input_text']:
                                    assistant_text = content_item.get('text', '')
                                    break
                        else:
                            assistant_text = assistant_msg.get('content', '')
                        
                        history_lines.append(f"Step {step_num}:\n  User: {user_text}\n  Assistant: {assistant_text}")
        else:
            # Original step_abstract approach
            history_lines = []
            for log in logs:
                if "step_abstract" in log:
                    history_lines.append(log["step_abstract"])

        # Build complete history text
        if previous_summary:
            # Include previous summary + new logs
            if history_lines:
                history_text = f"<previous_summary>\n{previous_summary}\n</previous_summary>\n\n<new_steps>\n" + "\n".join(history_lines) + "\n</new_steps>"
            else:
                # Only previous summary, no new steps
                history_text = f"<previous_summary>\n{previous_summary}\n</previous_summary>"
        else:
            # First time, only new logs
            history_text = "\n".join(history_lines) if history_lines else ""

        if not history_text.strip():
            return f"Steps {start_step}~{end_step}: No detailed records. Suggestion: Continue"

        # Use LLM to summarize with context refinement
        prompt = CONTEXT_REFINEMENT_PROMPT.format(
            task_instruction=self.task_instruction,
            start_step=start_step,
            end_step=end_step,
            history_text=history_text
        )

        try:
            messages = [
                {"role": "user", "content": prompt}
            ]
            summary_with_context_refinement = self._call_llm(self.state_manager_model, messages, self.state_manager_usage)
            return summary_with_context_refinement.strip()
        except Exception as e:
            self.logger.error(f"Failed to summarize history segment with context refinement: {e}")
            # Fallback: combine previous summary with brief new summary
            if previous_summary:
                brief_new = f"Steps {start_step}~{end_step}: {len(logs)} actions" if logs else "no new actions"
                return f"{previous_summary} + {brief_new}. Suggestion: Continue"
            else:
                brief_summary = f"{len(logs)} actions executed"
                return f"Steps {start_step}~{end_step}: {brief_summary}. Suggestion: Continue"

    def execute_task(
        self,
        task_config: dict
    ) -> float:
        """Execute task using tool-calling loop."""
        
        # Record start time for execution time tracking
        self.start_time = time.time()

        # Reset state
        self.global_planner_usage = {"cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "image_count": 0}
        self.visual_grounder_usage = {"cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "image_count": 0}
        self.state_manager_usage = {"cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "image_count": 0}
        self.env.reset(task_config=task_config)
        try:
            from run_hisa import _ensure_vm_resolution
            _ensure_vm_resolution(self.env, self.screen_width, self.screen_height, self.logger)
        except Exception as e:
            self.logger.warning(f"Failed to reset VM resolution: {e}")
        self.operation_count = 0
        self.action_logs = []
        self.last_full_summary = None
        self.last_summary_step = 0
        self.conversation_messages = []  # Store full conversation history when wo_step=True
        self.last_tool_output = None  # Store last tool execution result for wo_step mode

        if self.record:
            self.env.controller.start_recording()

        # Setup directories
        self.operations_dir = os.path.join(self.save_dir, "operations")
        os.makedirs(self.operations_dir, exist_ok=True)

        global_planner_name = getattr(self.global_planner_model, "model_name", "unknown")
        if self.visual_grounder_model is None:
            visual_grounder_name = "autoglm-os (unified)"
        else:
            visual_grounder_name = getattr(self.visual_grounder_model, "model_name", "unknown")
        state_manager_name = getattr(self.state_manager_model, "model_name", "unknown")
        self.logger.info(f"Global Planner: {global_planner_name}")
        self.logger.info(f"Visual Grounder: {visual_grounder_name}")
        self.logger.info(f"State Manager: {state_manager_name}")
        self.logger.info(f"Max steps: {self.max_steps}")
        self.logger.info(f"wo_step: {self.wo_step}")
        
        # Initial message
        task_instruction = task_config["instruction"]

        # Save task instruction as instance variable for later use
        self.task_instruction = task_instruction

        # Load relevant pattern
        domain = task_config.get("domain", "general")
        past_pattern_text = ""
        if not self.wo_pattern:
            past_pattern_text = self.pattern_manager._get_relevant_pattern(
                domain, task_instruction
            )
            if past_pattern_text:
                self.logger.info(f"Found relevant past pattern for domain: {domain}\n{past_pattern_text}")
            else:
                self.logger.info(f"No relevant past pattern found for domain: {domain}")

        # Save past pattern as instance variable for later use
        self.past_pattern_text = past_pattern_text

        # Main execution loop
        is_infeasible = False
        infeasible_reason = ""
        try:
            while self.operation_count < self.max_steps:
                self.logger.info(f"Step {self.operation_count + 1}/{self.max_steps}")

                # Capture token usage before this step
                usage_before_step = self._get_usage_snapshot()

                # Get global planner decision
                decision = self._get_decision()

                if decision is None:
                    self.logger.error("Failed to get valid decision")
                    # Send "FAIL" action to environment for task failure
                    try:
                        self.env.step("FAIL", 0)
                    except Exception as e:
                        self.logger.warning(f"Failed to send FAIL action: {e}")
                    break

                # Capture token usage after global planner decision
                usage_after_global_planner = self._get_usage_snapshot()

                # Check termination or infeasible
                if decision.get("code") == "DONE":
                    is_infeasible = False
                    self.logger.info("Task COMPLETED")
                    break
                elif decision.get("code") == "FAIL":
                    is_infeasible = True
                    infeasible_reason = "Task failed (marked as FAIL by agent)"
                    self.logger.info(f"Task FAILED: {infeasible_reason}")
                    # Send "FAIL" action to environment so action_history ends with "FAIL"
                    # This is required for OSWorld's infeasible task evaluation
                    try:
                        self.env.step("FAIL", 0)
                    except Exception as e:
                        self.logger.warning(f"Failed to send FAIL action: {e}")
                    break
                elif decision.get("code") == "WAIT":
                    # Handle WAIT command - continue to next iteration
                    self.logger.info("Agent requested WAIT - continuing to next step")
                    continue

                # Pre-calculate global planner token usage and set step_token_usage before tool execution
                global_planner_usage = self._calculate_usage_delta(usage_before_step, usage_after_global_planner)

                # Initialize step_token_usage with global planner data (state_manager will be updated after execution)
                visual_grounder_zero = {"cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "image_count": 0}
                self.step_token_usage = {
                    "global_planner": global_planner_usage["global_planner"],
                    "visual_grounder": global_planner_usage.get("visual_grounder", visual_grounder_zero),
                    "state_manager": {"cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "image_count": 0},
                    "total": {
                        "cost": global_planner_usage["global_planner"]["cost"],
                        "prompt_tokens": global_planner_usage["global_planner"]["prompt_tokens"],
                        "completion_tokens": global_planner_usage["global_planner"]["completion_tokens"],
                        "image_count": global_planner_usage["global_planner"]["image_count"]
                    }
                }

                # Execute tool and capture execution result text
                execution_result_text = self._execute_tool(decision)
                
                if execution_result_text and self.wo_step:
                    self.last_tool_output = execution_result_text

                # Capture token usage after tool execution
                usage_after_tool = self._get_usage_snapshot()

                # Calculate state_manager token usage and update step_token_usage
                tool_usage = self._calculate_usage_delta(usage_after_global_planner, usage_after_tool)
                total_step_usage = self._calculate_usage_delta(usage_before_step, usage_after_tool)

                # Update token usage: combine state_manager usage from history summarization and step abstraction
                visual_grounder_usage = global_planner_usage.get("visual_grounder", visual_grounder_zero)
                visual_grounder_tool = tool_usage.get("visual_grounder", visual_grounder_zero)
                self.step_token_usage = {
                    "global_planner": global_planner_usage["global_planner"],
                    "visual_grounder": {
                        "cost": visual_grounder_usage["cost"] + visual_grounder_tool["cost"],
                        "prompt_tokens": visual_grounder_usage["prompt_tokens"] + visual_grounder_tool["prompt_tokens"],
                        "completion_tokens": visual_grounder_usage["completion_tokens"] + visual_grounder_tool["completion_tokens"],
                        "image_count": visual_grounder_usage["image_count"] + visual_grounder_tool["image_count"]
                    },
                    "state_manager": {
                        "cost": global_planner_usage["state_manager"]["cost"] + tool_usage["state_manager"]["cost"],
                        "prompt_tokens": global_planner_usage["state_manager"]["prompt_tokens"] + tool_usage["state_manager"]["prompt_tokens"],
                        "completion_tokens": global_planner_usage["state_manager"]["completion_tokens"] + tool_usage["state_manager"]["completion_tokens"],
                        "image_count": global_planner_usage["state_manager"]["image_count"] + tool_usage["state_manager"]["image_count"]
                    },
                    "total": {
                        "cost": total_step_usage["global_planner"]["cost"] + total_step_usage["state_manager"]["cost"] + total_step_usage.get("visual_grounder", visual_grounder_zero)["cost"],
                        "prompt_tokens": total_step_usage["global_planner"]["prompt_tokens"] + total_step_usage["state_manager"]["prompt_tokens"] + total_step_usage.get("visual_grounder", visual_grounder_zero)["prompt_tokens"],
                        "completion_tokens": total_step_usage["global_planner"]["completion_tokens"] + total_step_usage["state_manager"]["completion_tokens"] + total_step_usage.get("visual_grounder", visual_grounder_zero)["completion_tokens"],
                        "image_count": total_step_usage["global_planner"]["image_count"] + total_step_usage["state_manager"]["image_count"] + total_step_usage.get("visual_grounder", visual_grounder_zero)["image_count"]
                    }
                }

                # Update the action_log entry that was already added with complete token usage
                if self.action_logs and self.action_logs[-1]["step"] == self.operation_count + 1:
                    self.action_logs[-1]["token_usage"] = self.step_token_usage

                self.operation_count += 1

                # Continue with next iteration
                # (screenshot will be fetched in next get_decision call)

            # Check if reached max_steps without completion
            if self.operation_count >= self.max_steps and not is_infeasible:
                is_infeasible = True
                infeasible_reason = f"Reached maximum steps ({self.max_steps}) without completing the task. Task may be infeasible or requires a different approach."
                self.logger.info(f"Reached max_steps ({self.max_steps}), marking as INFEASIBLE")
                # Send "FAIL" action to environment so action_history ends with "FAIL"
                # This is required for OSWorld's infeasible task evaluation
                try:
                    self.env.step("FAIL", 0)
                except Exception as e:
                    self.logger.warning(f"Failed to send FAIL action: {e}")

            # Evaluation
            score = self._evaluate_and_save(task_config, is_infeasible, infeasible_reason)

        except Exception as e:
            self.logger.error(f"Execution error: {e}")
            self.logger.error(traceback.format_exc())
            # Send "FAIL" action to environment for unexpected task failure
            try:
                self.env.step("FAIL", 0)
            except Exception as fail_error:
                self.logger.warning(f"Failed to send FAIL action: {fail_error}")
            
            with open(os.path.join(self.save_dir, "result.txt"), "w") as f:
                f.write("0.0")
            
            # Save err_reason.txt with error details
            with open(os.path.join(self.save_dir, "err_reason.txt"), "w") as f:
                f.write(f"Fatal error: {str(e)}\n\n{traceback.format_exc()}")
            
            score = 0.0
        
        if self.record:
            self.env.controller.end_recording(os.path.join(self.save_dir, "recording.mp4"))
        
        return score
    
    def _get_decision(self) -> Optional[Dict]:
        """Get decision from global planner with retry on parsing errors."""

        for attempt in range(self.max_parse_retries):
            try:
                # Get current screenshot
                screenshot = self.env.controller.get_screenshot()
                screenshot_b64 = base64.b64encode(screenshot).decode("utf-8")

                # Get current app info for tool_commands processing
                cur_app = None
                try:
                    app_list, cur_window_id = self.env.get_current_apps()
                    if cur_window_id in app_list:
                        cur_app = app_list[cur_window_id]['app_name']
                except Exception as e:
                    self.logger.warning(f"Failed to get current app: {e}")
                    cur_app = None

                # Context Refinement
                total_logs = len(self.action_logs)
                
                # Only trigger context refinement if not disabled (wo_refinement=False)
                if not self.wo_refinement and total_logs > 0 and total_logs % self.refine_period == 0:
                    # Trigger context refinement
                    if self.last_full_summary:
                        # Not first time: use previous summary + new logs since last summary
                        logs_to_summarize = self.action_logs[self.last_summary_step:]
                        start_step = self.action_logs[0]["step"]
                        end_step = self.action_logs[-1]["step"]
                        summary = self._get_context_refinement(
                            logs_to_summarize, start_step, end_step,
                            previous_summary=self.last_full_summary
                        )
                    else:
                        # First time: summarize all logs without previous summary
                        logs_to_summarize = self.action_logs
                        start_step = logs_to_summarize[0]["step"]
                        end_step = logs_to_summarize[-1]["step"]
                        summary = self._get_context_refinement(logs_to_summarize, start_step, end_step)

                    self.last_full_summary = summary
                    self.last_summary_step = total_logs
                    
                    # Clear conversation messages and last tool output after context refinement
                    if self.wo_step:
                        self.conversation_messages = []
                        self.last_tool_output = None  # Clear observation as it's now in summary

                # ========== Build Messages ==========
                # If there's error feedback, use direct error message without context
                if self.last_error_feedback:
                    # Direct call with error feedback only, no other context
                    messages = [
                        {"role": "user", "content": self.last_error_feedback}
                    ]
                else:
                    messages = self._build_messages(screenshot_b64)

                if attempt > 0:
                    self.logger.warning(f"Retry attempt {attempt}/{self.max_parse_retries}")

                # Call global planner
                response = self._call_llm(self.global_planner_model, messages, self.global_planner_usage)

                obs_dict = {"cur_app": cur_app}
                decision = self._parse_response(response, obs_dict)

                self.logger.info(f"Code: {decision.get('code', 'N/A')} | Thought: {decision.get('thought', '')}")

                # Clear error feedback on success
                self.last_error_feedback = None
                
                # Store conversation after successful parsing
                if self.wo_step and messages and len(messages) > 1:
                    # For traditional hisa with wo_step, store the current user message
                    if messages and len(messages) > 1:  # system + user messages
                        # Store user message (last one)
                        self.conversation_messages.append(messages[-1])
                        # Store assistant response
                        self.conversation_messages.append({
                            "role": "assistant",
                            "content": response
                        })

                return decision
                
            except Exception as e:
                self.logger.error(f"Decision parsing error (attempt {attempt + 1}/{self.max_parse_retries}): {e}")
                
                # If not last attempt, set error feedback for retry
                if attempt < self.max_parse_retries - 1:
                    error_feedback = FIX_RESPONSE_UNIFY_PROMPT.format(
                        error_message=str(e),
                        response=response
                    )

                    # Store error feedback for next iteration
                    self.last_error_feedback = error_feedback

                    # Continue to next retry
                    continue
                else:
                    # Last attempt failed, return None
                    self.logger.error("All retry attempts exhausted, cannot get valid decision")
                    with open(os.path.join(self.save_dir, "err_reason.txt"), "w") as f:
                        f.write("All retry attempts exhausted, cannot get valid decision")
                    return None
        
        return None

    def error_feedback(self) -> Optional[Dict]:
        """Get decision from global planner with retry on parsing errors."""

        for attempt in range(self.max_parse_retries):
            try:
                # Get current screenshot
                screenshot = self.env.controller.get_screenshot()
                screenshot_b64 = base64.b64encode(screenshot).decode("utf-8")

                # Get current app info for tool_commands processing
                cur_app = None
                try:
                    app_list, cur_window_id = self.env.get_current_apps()
                    if cur_window_id in app_list:
                        cur_app = app_list[cur_window_id]['app_name']
                except Exception as e:
                    self.logger.warning(f"Failed to get current app: {e}")
                    cur_app = None

                # Context Refinement
                total_logs = len(self.action_logs)
                
                # Only trigger context refinement if not disabled (wo_refinement=False)
                if not self.wo_refinement and total_logs > 0 and total_logs % self.refine_period == 0:
                    # Trigger context refinement
                    if self.last_full_summary:
                        # Not first time: use previous summary + new logs since last summary
                        logs_to_summarize = self.action_logs[self.last_summary_step:]
                        start_step = self.action_logs[0]["step"]
                        end_step = self.action_logs[-1]["step"]
                        summary = self._get_context_refinement(
                            logs_to_summarize, start_step, end_step,
                            previous_summary=self.last_full_summary
                        )
                    else:
                        # First time: summarize all logs without previous summary
                        logs_to_summarize = self.action_logs
                        start_step = logs_to_summarize[0]["step"]
                        end_step = logs_to_summarize[-1]["step"]
                        summary = self._get_context_refinement(logs_to_summarize, start_step, end_step)

                    self.last_full_summary = summary
                    self.last_summary_step = total_logs
                    
                    # Clear conversation messages and last tool output after context refinement
                    if self.wo_step:
                        self.conversation_messages = []
                        self.last_tool_output = None  # Clear observation as it's now in summary

                # ========== Build Messages ==========
                # If there's error feedback, use direct error message without context
                if self.last_error_feedback:
                    # Direct call with error feedback only, no other context
                    messages = [
                        {"role": "user", "content": self.last_error_feedback}
                    ]
                else:
                    messages = self._build_messages(screenshot_b64)

                if attempt > 0:
                    self.logger.warning(f"Retry attempt {attempt}/{self.max_parse_retries}")

                # Call global planner
                response = self._call_llm(self.global_planner_model, messages, self.global_planner_usage)

                obs_dict = {"cur_app": cur_app}
                decision = self._parse_response(response, obs_dict)

                self.logger.info(f"Code: {decision.get('code', 'N/A')} | Thought: {decision.get('thought', '')}")

                # Clear error feedback on success
                self.last_error_feedback = None
                
                # Store conversation after successful parsing
                if self.wo_step and messages and len(messages) > 1:
                    # For traditional hisa with wo_step, store the current user message
                    if messages and len(messages) > 1:  # system + user messages
                        # Store user message (last one)
                        self.conversation_messages.append(messages[-1])
                        # Store assistant response
                        self.conversation_messages.append({
                            "role": "assistant",
                            "content": response
                        })

                return decision
                
            except Exception as e:
                self.logger.error(f"Decision parsing error (attempt {attempt + 1}/{self.max_parse_retries}): {e}")
                
                # If not last attempt, set error feedback for retry
                if attempt < self.max_parse_retries - 1:
                    error_feedback = FIX_RESPONSE_UNIFY_PROMPT.format(
                        error_message=str(e),
                        response=response
                    )

                    # Store error feedback for next iteration
                    self.last_error_feedback = error_feedback

                    # Continue to next retry
                    continue
                else:
                    # Last attempt failed, return None
                    self.logger.error("All retry attempts exhausted, cannot get valid decision")
                    with open(os.path.join(self.save_dir, "err_reason.txt"), "w") as f:
                        f.write("All retry attempts exhausted, cannot get valid decision")
                    return None
        
        return None

    def _build_messages(self, screenshot_b64: str) -> List[Dict]:
        """Build messages for unified LLM approach using autoglm_v prompts."""
        # Get current app and accessibility tree for autoglm_v style observation
        cur_app = None
        accessibility_tree = ""
        app_info = ""
        app_list = {}
        cur_window_id = ""

        try:
            # Get current apps info (similar to autoglm_v)
            app_list, cur_window_id = self.env.get_current_apps()
            if cur_window_id in app_list:
                cur_app = app_list[cur_window_id]['app_name']
                # Try to get app-specific info
                tool_name = cur_app.strip().lower().replace('-', '_')
                if hasattr(self.env, '_get_obs'):
                    obs = self.env._get_obs()
                    accessibility_tree = obs.get('accessibility_tree', '')
                    app_info = obs.get('app_info', '')
        except Exception as e:
            self.logger.warning(f"Failed to get app info for unified LLM: {e}")

        # ========== CHANGE: Build history messages ==========
        history_messages = []
        if self.wo_step:
            # Use conversation messages directly (with sliding window)
            conversation_to_use = self.conversation_messages
            max_messages = self.sliding_window_size * 2
            if self.wo_refinement and len(self.conversation_messages) > max_messages:
                conversation_to_use = self.conversation_messages[-max_messages:]
            
            history_messages = conversation_to_use
        else:
            # Build from action logs (with sliding window)
            logs_to_use = self.action_logs
            if self.wo_refinement and len(self.action_logs) > self.sliding_window_size:
                logs_to_use = self.action_logs[-self.sliding_window_size:]
            
            if not self.wo_refinement and self.last_full_summary:
                # Context refinement enabled: add summary as first message
                history_messages.append({
                    "role": "assistant",
                    "content": self.last_full_summary
                })
                # Add recent step summaries
                for log in self.action_logs[self.last_summary_step:]:
                    if "step_abstract" in log:
                        history_messages.append({
                            "role": "assistant",
                            "content": log["step_abstract"]
                        })
            else:
                # wo_refinement=True or no summary yet
                for log in logs_to_use:
                    if "step_abstract" in log:
                        history_messages.append({
                            "role": "assistant",
                            "content": log["step_abstract"]
                        })

        # Construct prompt using autoglm_v's Prompt class
        if cur_app:
            tool_name = cur_app.strip().lower().replace("-", "_")
            tool_name = tool_name if tool_name in self.tool_list.keys() else None
        else:
            tool_name = None

        setup_prompt, func_def_prompt, note_prompt = AutoGLMPrompt.construct_procedural_memory(
            GroundingAgent, app_name=tool_name, client_password=self.client_password,
            with_image=self.with_image, with_atree=self.with_atree,
            relative_coordinate=self.relative_coordinate, glm41v_format=self.glm41v_format,
            screen_width=self.screen_width, screen_height=self.screen_height,
        )

        if self.tool_in_sys_msg:
            system_message = setup_prompt + "\n\n" + func_def_prompt + "\n\n" + note_prompt
        else:
            system_message = setup_prompt + "\n\n" + note_prompt
        
        # ========== CHANGE: Move task instruction to system message ==========
        system_message += f"\n\n**IMPORTANT** You are asked to complete the following task: {self.task_instruction}"

        # ========== Inject past pattern into system message on first turn ==========
        if self.past_pattern_text and len(self.conversation_messages) == 0:
            system_message += f"\n\n<past_pattern>\n{self.past_pattern_text}\n</past_pattern>"

        # ========== CHANGE: Build messages list structure like reference ==========
        messages = [
            {
                "role": "system",
                "content": system_message,
            }
        ]
        messages.extend(history_messages)

        # Build current observation
        if app_list:
            app_str = "Window ID    App Name    Title\n"
            for window_id, app in app_list.items():
                app_str += f"{window_id}    {app['app_name']}    {app['title']}\n"
        else:
            app_str = "None"

        last_result = ""
        if self.last_tool_output:
            last_result = self.last_tool_output.strip()
            last_result = last_result[:2000] + "..." if len(last_result) > 2000 else last_result
            self.last_tool_output = None
        last_result = last_result if last_result else "None"

        tree = ""
        if accessibility_tree and self.with_atree:
            tree = linearize_accessibility_tree(accessibility_tree, "Ubuntu")
            tree = trim_accessibility_tree(tree, 300)

        app_info_trimmed = app_info.strip() if app_info else "None"
        app_info_trimmed = app_info_trimmed[:5000] + "..." if len(app_info_trimmed) > 5000 else app_info_trimmed

        # ========== CHANGE: Build prompt like reference (no task instruction, no summary here) ==========
        prompt = "* Apps: {}\n\n* Current App: {}{}\n\n* App Info: {}\n\n* Previous Action Result: {}".format(
            app_str.strip(),
            cur_window_id.strip() if cur_window_id in app_str else "None",
            '\n\n* A11y Tree: {}'.format(tree.strip()) if self.with_atree and tree else "",
            app_info_trimmed,
            last_result,
        ) + (
            "\n\n" + func_def_prompt if not self.tool_in_sys_msg else ""
        ) + "\n\nBased on the current screenshot and conversation history, decide the next action."

        content = [{"type": "text", "text": prompt}]
        if self.with_image and screenshot_b64:
            screenshot_bytes = base64.b64decode(screenshot_b64)
    
            img = Image.open(BytesIO(screenshot_bytes))
            img = img.resize((self.image_width, self.image_height))
            buf = BytesIO()
            img.save(buf, format='PNG')
            resized_screenshot = base64.b64encode(buf.getvalue()).decode('utf-8')

            content = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{resized_screenshot}",
                        "detail": "high",
                    },
                }
            ] + content

        messages.append({"role": "user", "content": content})

        return messages


    def _parse_response(self, response: str, obs: Dict = None) -> Dict:
        """Parse unified LLM response (autoglm_v style)."""
        # Extract code from response (similar to autoglm_v's parse_code_from_string)
        import re

        # Find code blocks
        pattern = r"```(?:\w+\s+)?(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)

        if not matches:
            raise ValueError("No code block found in response")

        code = matches[0].strip()

        # Check for special commands
        if code in ["WAIT", "DONE", "FAIL"]:
            return {"code": code, "thought": ""}

        # Process tool method calls like autoglm_v
        code = re.sub(r'^python\s*(\\+n|\n)+', '', code, flags=re.IGNORECASE)
        code = re.sub(r'^(\\+n|\n)+', '', code)
        code = re.sub(r'(\\+n|\n)+$', '', code)

        thought = re.sub(pattern, '', response, flags=re.DOTALL).strip()

        # Handle tool method calls exactly like autoglm_v
        if "Agent." in code:
            if code.startswith("Agent.exit") or code.startswith("Agent.wait"):
                action = eval(code, {"Agent": self.grounding_agent, "BrowserTools": BrowserTools})
                return {"code": action, "thought": thought}
            action = code
        elif "BrowserTools." in code:
            action = eval(code, {"Agent": self.grounding_agent, "BrowserTools": BrowserTools})
        else:
            # For regular code, handle like autoglm_v with tool_commands
            cur_app = obs.get("cur_app") if obs else None
            if cur_app:
                tool_name = cur_app.strip().lower().replace("-", "_")
                if tool_name in self.tool_list:
                    actions = self.grounding_agent.tool_commands(code, tool_name)
                    action = actions[0]
                else:
                    action = code
            else:
                action = code

        return {"code": action, "thought": thought}

    def _execute_tool(self, decision: Dict) -> str:
        """Execute tool based on decision and return execution result text."""
        # In unified LLM mode, decision should contain the Python code to execute
        code = decision.get("code", "")
        if not code:
            return "No code to execute"

        # Store thought for step_abstract
        self.current_thought = decision.get("thought", "")

        # Execute the code directly (similar to autoglm_v approach)
        return self._execute_code(code)

    def _execute_code(self, code) -> str:
        """Execute unified LLM generated code (autoglm_v style)."""
        # Record step start time
        step_start_time = time.time()
        step = self.operation_count + 1

        try:
            # Get before screenshot
            before_screenshot = self.env.controller.get_screenshot()
            screenshot_file = f"step_{step}.png"

            with open(os.path.join(self.operations_dir, screenshot_file), "wb") as f:
                f.write(before_screenshot)

            final_code = code
            if isinstance(final_code, str) and "Agent." in final_code and self.visual_grounder_model is not None:
                try:
                    final_code = self._resolve_agent_call(final_code, self.current_thought, before_screenshot)
                except Exception as e:
                    self.logger.warning(f"Agent grounding failed, falling back to direct eval: {e}")

            # Handle different types of code (tool methods already evaluated in parse phase)
            if isinstance(final_code, dict):
                # Special action dict (like OPEN_CHROME_TAB from BrowserTools methods)
                obs, *_ = self.env.step(final_code, self.sleep_after_execution)
                final_code = str(final_code)  # Convert to string for logging
            else:
                # If still a tool method string, evaluate it now
                if isinstance(final_code, str) and ("Agent." in final_code or "BrowserTools." in final_code):
                    final_code = eval(final_code, {"Agent": self.grounding_agent, "BrowserTools": BrowserTools})
                if isinstance(final_code, dict):
                    obs, *_ = self.env.step(final_code, self.sleep_after_execution)
                    final_code = str(final_code)
                else:
                    # Execute regular pyautogui code
                    obs, *_ = self.env.step(final_code, self.sleep_after_execution)

            # Wait for action to take effect
            time.sleep(10)

            # Get after screenshot
            after_screenshot = self.env.controller.get_screenshot()

            # Step abstraction for unified execution
            if self.wo_step:
                step_abstraction = ""
            else:
                step_abstraction = "Result: " + self._get_step_abstraction(
                    before_screenshot, after_screenshot, f"Executed: {final_code}",
                    wo_roi=self.wo_roi, roi_margin=self.roi_margin
                )

            # Generate step_abstract
            thought_prefix = f"Thought: {self.current_thought} | " if self.current_thought else ""
            step_abstract = f"Step {step}: unified_execution | {thought_prefix}Code: {final_code} | {step_abstraction}"

            # Calculate step execution time
            step_time = time.time() - step_start_time

            self.action_logs.append({
                "step": step,
                "type": "unified_execution",
                "execution_success": True,
                "screenshot": screenshot_file,
                "step_abstract": step_abstract,
                "step_time": round(step_time, 2),
                "token_usage": self.step_token_usage
            })

            # Return execution result text for wo_step mode
            return f"Unified Execution: {final_code}\nStatus: Success\n{step_abstraction}"

        except Exception as e:
            self.logger.error(f"Unified execution error: {e}")

            # Generate step_abstract for error
            thought_prefix = f"Thought: {self.current_thought} | " if self.current_thought else ""
            code_str = str(code) if isinstance(code, dict) else code
            step_abstract = f"Step {step}: unified_execution | {thought_prefix}Code: {code_str} | Result: Error - {str(e)}"

            # Calculate step execution time
            step_time = time.time() - step_start_time

            self.action_logs.append({
                "step": step,
                "type": "unified_execution",
                "execution_success": False,
                "screenshot": screenshot_file if 'screenshot_file' in locals() else "",
                "step_abstract": step_abstract,
                "step_time": round(step_time, 2),
                "token_usage": self.step_token_usage
            })

            # Return execution result text for wo_step mode
            code_str = str(code) if isinstance(code, dict) else code
            return f"Unified Execution: {code_str}\nStatus: Failed\nError: {str(e)}"

    def _get_step_abstraction(self, before_screenshot: bytes, after_screenshot: bytes,
            action_description: str, wo_roi: bool = False,
            roi_margin: int = 50) -> str:
        """Abstract step by comparing before/after screenshots.

        Args:
            before_screenshot: Screenshot before action
            after_screenshot: Screenshot after action
            action_description: Description of the action performed
            wo_roi: If True, disable ROI cropping (default: False means ROI cropping is enabled)
            roi_margin: Margin to add around ROI when cropping (default: 50)

        Returns:
            Step abstract text (e.g., "Succeeded. Menu opened." or "Failed. No UI change.")
        """
        try:
            # Convert screenshots to PIL Images for ROI detection
            before_img = Image.open(io.BytesIO(before_screenshot))
            after_img = Image.open(io.BytesIO(after_screenshot))
            
            # Check for size mismatch and log detailed info for debugging
            if before_img.size != after_img.size:
                self.logger.error(f"[ANOMALY] Screenshot size mismatch detected!")
                
            # Optionally crop to change ROI (enabled by default, disabled when wo_roi=True)
            if not wo_roi:
                try:
                    cropped_before, cropped_after = get_change_roi(
                        before_img, after_img,
                        margin=roi_margin,
                    )

                    # If ROI detected, use cropped images
                    if cropped_before is not None and cropped_after is not None:
                        before_img = cropped_before
                        after_img = cropped_after
                    else:
                        # No change detected - directly return without calling LLM
                        return "No change detected."
                except Exception as roi_error:
                    self.logger.warning(f"ROI detection failed, using full screenshots: {roi_error}")

            # Convert (possibly cropped) images to base64
            before_buffer = io.BytesIO()
            after_buffer = io.BytesIO()
            before_img.save(before_buffer, format="PNG")
            after_img.save(after_buffer, format="PNG")

            before_b64 = base64.b64encode(before_buffer.getvalue()).decode("utf-8")
            after_b64 = base64.b64encode(after_buffer.getvalue()).decode("utf-8")

            prompt = STEP_ABSTRACTION_PROMPT.format(
                action_description=action_description
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Before screenshot:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{before_b64}", "detail": "high"}},
                        {"type": "text", "text": "After screenshot:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{after_b64}", "detail": "high"}},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            step_abstraction = self._call_llm(self.state_manager_model, messages, self.state_manager_usage)
            return step_abstraction.strip()

        except Exception as e:
            self.logger.error(f"Failed to abstract step: {e}")
            return "Step abstraction failed due to error."

    def _get_bash_execution(self, code: str) -> str:
        """Execute bash commands or Python scripts (not pyautogui)."""
        self.logger.info(f"[bash_execution] {code}")

        # Record step start time
        step_start_time = time.time()

        step = self.operation_count + 1

        try:
            # Get before screenshot
            before_screenshot = self.env.controller.get_screenshot()

            # Call env.controller.run_bash_script instead of env.step
            output_dict = self.env.controller.run_bash_script(code, timeout=self.bash_timeout)
            exitcode = 0 if output_dict["status"] == "success" else 1
            logs = output_dict["output"]

            # Wait 10 seconds for action to take effect
            time.sleep(10)

            # Get after screenshot
            after_screenshot = self.env.controller.get_screenshot()
            screenshot_file = f"step_{step}_bash.png"

            with open(os.path.join(self.operations_dir, screenshot_file), "wb") as f:
                f.write(after_screenshot)

            # Step abstraction for bash execution
            # Skip step abstraction if wo_step is True
            if self.wo_step:
                step_abstraction = ""
            else:
                bash_description = f"Bash command: {code}\nOutput: {logs}..."  # Truncate long output
                step_abstraction = "Result: " + self._get_step_abstraction(
                    before_screenshot, after_screenshot, bash_description,
                    wo_roi=self.wo_roi, roi_margin=self.roi_margin
                )

            # Generate step_abstract summary
            thought_prefix = f"Thought: {self.current_thought} | " if self.current_thought else ""
            step_abstract = f"Step {step}: Bash execution | {thought_prefix}Code: {code} | Output: {logs} | {step_abstraction}"

            # Calculate step execution time
            step_time = time.time() - step_start_time

            self.action_logs.append({
                "step": step,
                "type": "bash_execution",
                "execution_success": exitcode == 0,
                "screenshot": screenshot_file,
                "step_abstract": step_abstract,
                "step_time": round(step_time, 2),
                "token_usage": self.step_token_usage
            })

            # Return execution result text for wo_step mode
            status_str = "Success" if exitcode == 0 else "Failed"
            return f"Bash Command: {code}\nStatus: {status_str}\nOutput:\n{logs}"
            
        except Exception as e:
            self.logger.error(f"Bash execution error: {e}")

            screenshot = self.env.controller.get_screenshot()
            screenshot_file = f"step_{step}_bash_error.png"

            with open(os.path.join(self.operations_dir, screenshot_file), "wb") as f:
                f.write(screenshot)

            # Generate step_abstract summary for error
            thought_prefix = f"Thought: {self.current_thought} | " if self.current_thought else ""
            step_abstract = f"Step {step}: Bash execution | {thought_prefix}Code: {code} | Result: Error - {str(e)}"

            self.action_logs.append({
                "step": step,
                "type": "bash_execution",
                "execution_success": False,
                "screenshot": screenshot_file,
                "step_abstract": step_abstract,
                "token_usage": self.step_token_usage
            })

            # Return execution result text for wo_step mode
            return f"Bash Command: {code}\nStatus: Failed\nError: {str(e)}"

    def _wait_function(self, seconds_str: str) -> str:
        """Wait for specified seconds and observe UI changes."""
        try:
            wait_seconds = float(seconds_str)
            # Limit wait time to reasonable range
            wait_seconds = max(5, min(wait_seconds, 60))
        except:
            self.logger.warning(f"Invalid wait time '{seconds_str}', using default 15 seconds")
            wait_seconds = 15

        self.logger.info(f"[wait] Waiting for {wait_seconds} seconds...")

        # Record step start time
        step_start_time = time.time()

        step = self.operation_count + 1

        try:
            # Get before screenshot
            before_screenshot = self.env.controller.get_screenshot()
            screenshot_file = f"step_{step}wait_function_before.png"

            with open(os.path.join(self.operations_dir, screenshot_file), "wb") as f:
                f.write(before_screenshot)

            # Wait
            time.sleep(wait_seconds)

            # Get after screenshot
            after_screenshot = self.env.controller.get_screenshot()
            after_screenshot_file = f"step_{step}wait_function_after.png"

            with open(os.path.join(self.operations_dir, after_screenshot_file), "wb") as f:
                f.write(after_screenshot)

            # Step abstraction for wait
            # Skip step abstraction if wo_step is True
            if self.wo_step:
                step_abstraction = ""
            else:
                step_abstraction = "Result: " + self._get_step_abstraction(
                    before_screenshot, after_screenshot,
                    f"Waited {wait_seconds} seconds to observe UI changes",
                    wo_roi=self.wo_roi, roi_margin=self.roi_margin
                )

            # Generate step_abstract
            thought_prefix = f"Thought: {self.current_thought} | " if self.current_thought else ""
            step_abstract = f"Step {step}: wait | {thought_prefix}Duration: {wait_seconds}s | {step_abstraction}"

            # Calculate step execution time
            step_time = time.time() - step_start_time

            self.action_logs.append({
                "step": step,
                "type": "wait",
                "execution_success": True,
                "screenshot": screenshot_file,
                "step_abstract": step_abstract,
                "step_time": round(step_time, 2),
                "token_usage": self.step_token_usage
            })

            # Return execution result text for wo_step mode
            return f"Wait: {wait_seconds}s\nStatus: Success\n{step_abstraction}"

        except Exception as e:
            self.logger.error(f"Wait execution error: {e}")

            # Generate step_abstract for error
            thought_prefix = f"Thought: {self.current_thought} | " if self.current_thought else ""
            step_abstract = f"Step {step}: wait | {thought_prefix}Duration: {wait_seconds}s | Result: Error - {str(e)}"

            # Calculate step execution time
            step_time = time.time() - step_start_time

            self.action_logs.append({
                "step": step,
                "type": "wait",
                "execution_success": False,
                "screenshot": screenshot_file if 'screenshot_file' in locals() else "",
                "step_abstract": step_abstract,
                "step_time": round(step_time, 2),
                "token_usage": self.step_token_usage
            })

            # Return execution result text for wo_step mode
            return f"Wait: {wait_seconds}s\nStatus: Failed\nError: {str(e)}"


    def _evaluate_and_save(self, task_config: dict, is_infeasible: bool = False, termination_reason: str = "") -> float:
        """Evaluate task and save results."""
        self.logger.info(f"\n{'='*80}")
        self.logger.info("Task Evaluation")
        self.logger.info("="*80)

        # Extract and save pattern BEFORE evaluating score
        # This prevents data leakage - lessons should be based on execution process only
        domain = task_config.get("domain", "general")
        task_instruction = task_config["instruction"]

        if not self.wo_pattern:
            self.logger.info("Inducing pattern...")
            key_lessons = self.pattern_manager._pattern_induction(
                task_instruction=task_instruction,
                action_logs=self.action_logs
            )

            if key_lessons:
                self.pattern_manager._save_pattern(domain, key_lessons)
                # Format lessons as numbered list for logging
                formatted_lessons = "\n".join(f"  {i+1}. [{lesson['type']}] {lesson['lesson']}" for i, lesson in enumerate(key_lessons))
                self.logger.info(f"Saved {len(key_lessons)} lesson(s):\n{formatted_lessons}")
            else:
                self.logger.info("No significant lessons to save")

        # Now evaluate score
        try:
            # self.logger.info("Closing temporary windows...")
            # self.env.step("pyautogui.press('esc')", 0.5)

            # Wait for VM HTTP service to stabilize after task execution
            self.logger.info("Waiting for VM to stabilize before evaluation...")
            time.sleep(10)
            
            # Retry evaluation with exponential backoff to handle transient VM service issues
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    score = self.env.evaluate()
                    break
                except Exception as eval_error:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 5  # 5s, 10s, 15s
                        self.logger.warning(f"Evaluation attempt {attempt + 1} failed: {eval_error}. Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        raise
        except Exception as e:
            self.logger.error(f"Evaluation failed after {max_retries} attempts: {e}")
            score = 0.0

        gui_steps = len([log for log in self.action_logs if log["type"] == "gui_action"])
        bash_steps = len([log for log in self.action_logs if log["type"] == "bash_execution"])
        wait_steps = len([log for log in self.action_logs if log["type"] == "wait"])

        global_planner_cost = self.global_planner_usage["cost"]
        global_planner_prompt = self.global_planner_usage["prompt_tokens"]
        global_planner_completion = self.global_planner_usage["completion_tokens"]
        global_planner_images = self.global_planner_usage["image_count"]

        visual_grounder_cost = self.visual_grounder_usage["cost"]
        visual_grounder_prompt = self.visual_grounder_usage["prompt_tokens"]
        visual_grounder_completion = self.visual_grounder_usage["completion_tokens"]
        visual_grounder_images = self.visual_grounder_usage["image_count"]

        state_manager_cost = self.state_manager_usage["cost"]
        state_manager_prompt = self.state_manager_usage["prompt_tokens"]
        state_manager_completion = self.state_manager_usage["completion_tokens"]
        state_manager_images = self.state_manager_usage["image_count"]

        total_cost = global_planner_cost + visual_grounder_cost + state_manager_cost
        total_images = global_planner_images + visual_grounder_images + state_manager_images

        # Calculate execution time
        execution_time = time.time() - self.start_time

        # Determine success and failure reason (score is 0 or 1)
        failure_reason = ""
        if is_infeasible:
            # Use termination_reason if provided (contains detailed infeasible explanation)
            failure_reason = termination_reason if termination_reason else "Task marked as infeasible"
        elif termination_reason:
            failure_reason = termination_reason

        execution_log = {
            "statistics": {
                "score": score,
                "total_steps": self.operation_count,
                "cua_steps": gui_steps,
                "coding_steps": bash_steps,
                "wait_steps": wait_steps,
                "image_count": total_images,
                "total_cost": total_cost,
                "prompt_tokens": global_planner_prompt + visual_grounder_prompt + state_manager_prompt,
                "completion_tokens": global_planner_completion + visual_grounder_completion + state_manager_completion,
                "execution_time": execution_time,
                "model_usage": {
                    "global_planner": {
                        "model_name": getattr(self.global_planner_model, "model_name", "unknown"),
                        "cost": global_planner_cost,
                        "prompt_tokens": global_planner_prompt,
                        "completion_tokens": global_planner_completion,
                        "image_count": global_planner_images
                    },
                    "visual_grounder": {
                        "model_name": getattr(self.visual_grounder_model, "model_name", "none"),
                        "cost": visual_grounder_cost,
                        "prompt_tokens": visual_grounder_prompt,
                        "completion_tokens": visual_grounder_completion,
                        "image_count": visual_grounder_images
                    },
                    "state_manager": {
                        "model_name": getattr(self.state_manager_model, "model_name", "unknown"),
                        "cost": state_manager_cost,
                        "prompt_tokens": state_manager_prompt,
                        "completion_tokens": state_manager_completion,
                        "image_count": state_manager_images
                    }
                }
            },
            "task_config": task_config,
            "action_logs": self.action_logs,
            "success": score == 1.0,
            "failure_reason": failure_reason
        }

        with open(os.path.join(self.save_dir, "execution_log.json"), "w") as f:
            json.dump(serialize_json(execution_log), f, indent=2)

        with open(os.path.join(self.save_dir, "result.txt"), "w") as f:
            f.write(str(score))

        self.logger.info("="*80)

        return score

    def cleanup(self):
        """Clean up resources."""
        if self.env:
            self.logger.info("Closing environment...")
            self.env.close()
            self.env = None

        # Close pattern manager to release Qdrant lock
        if hasattr(self, 'pattern_manager') and self.pattern_manager:
            self.logger.info("Closing pattern manager...")
            try:
                if hasattr(self.pattern_manager, 'qdrant') and self.pattern_manager.qdrant:
                    if hasattr(self.pattern_manager.qdrant, 'client'):
                        self.pattern_manager.qdrant.client.close()
            except Exception as e:
                self.logger.warning(f"Error closing Qdrant client: {e}")
            self.pattern_manager = None
