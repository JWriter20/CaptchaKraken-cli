"""
ActionPlanner - LLM-based captcha planning with tool support.

Supports:
1. Grid selection - which numbered squares to click
2. Tool-aware planning - can request detect() calls
3. Drag refinement - iterative adjustment with visual feedback
4. Text reading - OCR for text captchas
5. Video captchas - plan actions for video captchas

Backends: ollama, transformers
"""

import json
import os
import sys
import base64
import time
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

# Import planner types for documentation/type hinting
try:
    from .planner_types import (
        PlannerPlan, 
        PlannerDragRefineAction,
        PlannerClickAction,
        PlannerTypeAction
    )
except ImportError:
    # Fallback if types file is missing or in different environment
    PlannerPlan = Any  # type: ignore
    PlannerDragRefineAction = Any  # type: ignore
    PlannerClickAction = Any  # type: ignore
    PlannerTypeAction = Any  # type: ignore

# Timing helper (opt-in via CAPTCHA_TIMINGS=1)
from .timing import timed
from .hardware import get_recommended_model_config

# Debug flag - set via CAPTCHA_DEBUG=1 environment variable
DEBUG = os.getenv("CAPTCHA_DEBUG", "0") == "1"


# For general planning with tool support
PLAN_WITH_TOOLS_PROMPT = """Solve the captcha using ONE direct action or ONE tool call.

{instruction}

{object_list_section}

{history_section}

Direct Actions:
- {{ "action": "click", "target_description": "...", "max_items": N }} (Use plural for multiple items)
- {{ "action": "drag", "source_description": "...", "target_description": "...", "location_hint": [x, y] }}
- {{ "action": "type", "text": "...", "target_description": "..." }}
- {{ "action": "wait", "duration_ms": 500 }}
- {{ "action": "done" }}

Tool Calls:
- detect(object_class, max_items)
- simulate_drag(source, target_hint, location_hint)

Important: Respond ONLY with JSON. Limit analysis to 1 sentence.
Example Click:
{{
  "analysis": "Brief reasoning",
  "goal": "Visual goal",
  "action": {{ ... }} 
  // OR: "tool_calls": [ {{ "name": "...", "args": {{ ... }} }} ]
}}"""


# For text captchas: read the text
TEXT_READ_PROMPT = """Read the distorted text in this captcha image.
Look carefully at each character, accounting for distortion, rotation, and noise.

Typically, these captchas contain exactly 6 characters.

Respond ONLY with JSON matching this structure:
{{
  "analysis": "Brief reasoning",
  "goal": "Type the text",
  "action": {{
    "action": "type",
    "text": "the text identified",
    "target_description": "text input field"
  }}
}}"""


# For grid selection captchas
SELECT_GRID_PROMPT = """Solve the captcha grid by choosing the cells that contain the target object.

Grid: {rows}x{cols} ({total} cells)
{grid_hint}
{instruction_context}

Return JSON format ALWAYS:
{{
  "analysis": "Think step by step and solve the captcha grid.
    1. Identify the target object from the header text in the image.
    2. Analyze each numbered cell.
  "goal": "To select the target object found in the analysis section.",
  "action": {{
    "action": "click",
    "target_ids": [list of cell numbers (1-{total})]
  }}
}}

Correct Example:
{{
  "analysis": "The header asks for \\"buses\\".\\ncell 1: Highway with cars.\\ncell 2: Parked car and trees.\\ncell 3: Silver pickup truck.\\ncell 4: Palm trees and building.\\ncell 5: Road and trees.\\ncell 6: Gas station sign.\\ncell 7: Red fire hydrant on grass.\\ncell 8: Underside of a bridge.\\ncell 9: Bicycle parked near a fence.\\nWe need to identify squares containing buses that have not been selected. However none of the images appear to contain buses. The user has likely already solved the captcha, meaning all required tiles have been selected and verified. There are no remaining tiles to select.",
  "goal": "Select all tiles with buses.",
  "action": {{
    "action": "click",
    "target_ids": []
  }}
}}"""


# For drag refinement: iterative adjustment
DRAG_REFINE_PROMPT = """You are refining a drag action for a captcha puzzle.

{instruction}

Primary Goal: {primary_goal}
(Use this goal to judge success. Do NOT change this goal.)

Object to Move: "{source_desc}"
Target Destination: "{target_desc}"

CRITICAL: The destination must be DISTINCT from the source. You are never dragging something to itself.
The draggable item is marked with a LIGHT GREEN box (drawn with padding around the item).

EVALUATION CRITERIA:
1. **Vertical Position**: Is the object too high or too low? (e.g. Is the head on the stomach? It should be on the neck.)
2. **Horizontal Position**: Is it too far left or right?
3. **Connectivity**: Do the lines of the object flow smoothly into the target?

Evaluate this drag destination:
1. Describe the specific visual alignment. (e.g. "The neck lines are misaligned by...", "The head is overlapping the torso...")
2. If not perfect, what RELATIVE adjustment is needed?

The image shows:
- LIGHT GREEN box: The draggable item at its current location.

Current destination: ({target_x:.1%}, {target_y:.1%})

{history_text}

Provide adjustments as percentages of image size:
- dx: positive = move right, negative = move left (e.g., 0.05 = 5% right)
- dy: positive = move down, negative = move up (e.g., -0.02 = 2% up)

Make SMALL adjustments (typically 1-5%). We can refine iteratively.

Respond ONLY with JSON:
{{
  "analysis": "Specific critique of vertical/horizontal alignment and edge connectivity",
  "goal": "{primary_goal}",
  "action": "refine_drag",
  "decision": "accept" | "adjust",
  "dx": 0.0,
  "dy": 0.0
}}"""


class ActionPlanner:
    """
    LLM-based captcha planner with tool support.

    Supports backends: ollama, transformers
    """

    def __init__(
        self,
        backend: Literal["ollama", "transformers"] = "ollama",
        model: Optional[str] = None,
        debug_callback: Optional[Any] = None,
        **kwargs
    ):
        self.backend = backend
        self.debug_callback = debug_callback
        self.config = kwargs
        self._model = None
        self._processor = None
        self.token_usage: List[Dict[str, Any]] = []

        # Default models per backend
        if model is None:
            if backend == "transformers":
                # Get recommended model based on hardware
                rec_model, _, rec_msg = get_recommended_model_config()
                if rec_model == "API":
                    self._log(f"WARNING: {rec_msg}")
                    # Default to the full merged model if they still want to try locally
                    self.model = "Jake-Writer-Jobharvest/qwen3-vl-8b-merged-fp16"
                else:
                    self._log(f"Recommended model: {rec_model} ({rec_msg})")
                    self.model = rec_model
            elif backend == "ollama":
                self.model = "test-qwen3-vl:latest"
        else:
            self.model = model

    def _log(self, message: str) -> None:
        """Log message to callback and/or stderr."""
        # Always print to stderr if global DEBUG is set
        if DEBUG:
            print(f"[Planner DEBUG] {message}", file=sys.stderr)
        
        # Also use callback if provided (for file logging)
        if self.debug_callback:
            self.debug_callback(f"[Planner] {message}")

    # ------------------------------------------------------------------
    # Core helper: chat with image
    # ------------------------------------------------------------------
    def _chat_with_image(self, prompt: str, image_path: Union[str, List[str]]) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Send a prompt + image(s)/video(s) to the LLM backend, get response."""
        self._log(f"Backend: {self.backend}, Model: {self.model}")
        
        image_paths = [image_path] if isinstance(image_path, str) else image_path
        self._log(f"Input paths: {image_paths}")

        if self.backend == "transformers":
            # Check if we should use the tool server for warm inference
            tool_server_url = os.getenv("CAPTCHA_TOOL_SERVER")
            if tool_server_url:
                import requests
                self._log(f"Calling tool server for planning: {tool_server_url}/plan")
                try:
                    resp = requests.post(
                        f"{tool_server_url}/plan",
                        json={"image_paths": image_paths, "prompt": prompt},
                        timeout=300
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    return data.get("response", ""), None
                except Exception as e:
                    self._log(f"Tool server call failed: {e}. Falling back to local transformers.")

            from PIL import Image
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
            from qwen_vl_utils import process_vision_info
            
            if self._model is None:
                self._log(f"Loading transformers model: {self.model}")
                self._processor = AutoProcessor.from_pretrained(self.model, trust_remote_code=True)
                
                from .hardware import get_device
                device = get_device()
                
                # Use bfloat16 for CUDA, float16 for MPS
                dtype = torch.bfloat16 if device == "cuda" else torch.float16 if device == "mps" else torch.float32
                
                self._log(f"Using device: {device}, dtype: {dtype}")
                
                # Use SDPA for speed
                attn_impl = "sdpa"
                
                self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                    self.model,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                    attn_implementation=attn_impl
                ).to(device)
                
                # No longer need device_map="auto" if we force to device
            
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert captcha solver. Think carefully about the visual cues. Respond with your reasoning followed by the JSON action."
                },
                {"role": "user", "content": []}
            ]
            processed_images = []
            
            for path in image_paths:
                is_video = any(path.lower().endswith(ext) for ext in [".mp4", ".webm", ".gif", ".avi"])
                if is_video:
                    import cv2
                    cap = cv2.VideoCapture(path)
                    ret, frame = cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        processed_images.append(Image.fromarray(frame_rgb))
                        messages[1]["content"].append({"type": "image", "image": path})
                    cap.release()
                else:
                    processed_images.append(Image.open(path).convert("RGB"))
                    messages[1]["content"].append({"type": "image", "image": path})
            
            messages[1]["content"].append({"type": "text", "text": prompt})
            
            try:
                text = self._processor.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True,
                    enable_thinking=True
                )
            except:
                text = prompt
            
            image_inputs, video_inputs = process_vision_info(messages)
            
            # Optimization: Cap resolution for speed
            inputs = self._processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                min_pixels=256*256,
                max_pixels=768*768
            )
            
            # Move inputs to model device
            inputs = inputs.to(device)
            
            self._log(f"Generating response with {self.model}...")
            with torch.no_grad():
                # Use greedy decoding and reduce tokens
                generated_ids = self._model.generate(
                    **inputs, 
                    max_new_tokens=1024, # Increased for thinking
                    do_sample=False,
                    use_cache=True
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            result = self._processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            return result, None

        if self.backend == "ollama":
            import requests
            host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            
            images_b64 = []
            for path in image_paths:
                with open(path, "rb") as f:
                    images_b64.append(base64.b64encode(f.read()).decode("utf-8"))
            
            self._log(f"Generating Ollama response for model: {self.model}")
            resp = requests.post(
                f"{host}/api/chat",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt, "images": images_b64}],
                    "stream": False,
                    # Removed "format": "json" to allow thinking tags/reasoning
                },
                timeout=120
            )
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"], None

        raise ValueError(f"Unknown backend: {self.backend}")

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Extract and parse JSON from LLM response."""
        text = text.strip()

        # Remove markdown code blocks
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0]

        # Find JSON object
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            text = text[start:end]

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}

    # ------------------------------------------------------------------
    # Tool-aware planning
    # ------------------------------------------------------------------
    def plan_with_tools(
        self,
        image_path: str,
        instruction: str = "",
        objects: Optional[List[Dict[str, Any]]] = None,
        history: Optional[List[str]] = None,
        context_image_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Plan the next action, potentially requesting a tool call.

        Args:
            image_path: Path to the captcha image or video
            instruction: Optional instruction text
            objects: Optional list of detected objects
            history: Optional list of previous actions
            context_image_path: Optional path to a static image with overlays/labels

        Returns:
            Dict containing the planner's response, following the PlannerPlan schema.
        """
        object_list_section = ""
        if objects:
            lines = ["Detected Objects (ID: box [x, y, w, h]):"]
            for obj in sorted(objects, key=lambda x: x.get("id", 0)):
                lines.append(f"- Object {obj.get('id')}: {obj.get('bbox')}")
            lines.append("\\nUse these Object IDs in your analysis.")
            object_list_section = "\\n".join(lines)

        history_section = ""
        if history:
            lines = ["Action History (most recent last):"]
            for i, h in enumerate(history):
                lines.append(f"{i+1}. {h}")
            lines.append("\\nAVOID repeating failed actions.")
            history_section = "\\n".join(lines)

        prompt = PLAN_WITH_TOOLS_PROMPT.format(
            instruction=instruction or "Solve this captcha.",
            object_list_section=object_list_section,
            history_section=history_section,
        )
        
        images = [image_path]
        if context_image_path and context_image_path != image_path:
            images.append(context_image_path)
            prompt += "\n\nNOTE: Multiple images/media provided. The first item is the raw captcha (could be video), the second is a static frame with red boxes and Object IDs for reference."

        response, _ = self._chat_with_image(prompt, images)
        data = self._parse_json(response)
        
        # Validation if Pydantic is available
        if PlannerPlan is not Any:
            try:
                # We return dict for compatibility but ensure it matches the schema
                PlannerPlan.model_validate(data)
            except Exception as e:
                self._log(f"Planner response validation failed: {e}")
                
        return data

    # ------------------------------------------------------------------
    # Drag refinement
    # ------------------------------------------------------------------
    def refine_drag(
        self,
        image_path: str,
        instruction: str,
        current_target: List[float],
        history: List[Dict[str, Any]],
        source_description: str = "movable item",
        target_description: str = "matching slot",
        primary_goal: str = "Complete the puzzle",
    ) -> Dict[str, Any]:
        """
        Refine a drag destination with visual feedback.

        Args:
            image_path: Path to image with drag overlay (source box, arrow, target box)
            instruction: Original puzzle instruction
            current_target: Current target position [x, y] as percentages (0-1)
            history: List of previous refinements:
                [{"destination": [x, y], "analysis": "...", "decision": "..."}]
            source_description: Description of the item being dragged
            target_description: Description of where it should go
            primary_goal: The specific goal identified by the planner

        Returns:
            Dict following the PlannerDragRefineAction schema.
        """
        # Format history for context
        if history:
            history_lines = ["Previous attempts:"]
            for i, h in enumerate(history):
                dest = h.get("destination", [0, 0])
                analysis = h.get("analysis", "")
                decision = h.get("decision", "")
                history_lines.append(
                    f"  {i + 1}. destination: ({dest[0]:.1%}, {dest[1]:.1%}), "
                    f'analysis: "{analysis}", decision: {decision}'
                )
            history_lines.append(f"  {len(history) + 1}. destination: ({current_target[0]:.1%}, {current_target[1]:.1%}) (CURRENT)")
            history_text = "\n".join(history_lines)
        else:
            history_text = "This is the first attempt."

        prompt = DRAG_REFINE_PROMPT.format(
            instruction=instruction or "Complete the drag puzzle.",
            primary_goal=primary_goal,
            source_desc=source_description,
            target_desc=target_description,
            target_x=current_target[0],
            target_y=current_target[1],
            history_text=history_text,
        )

        response, _ = self._chat_with_image(prompt, image_path)
        result = self._parse_json(response)

        # Validation
        if PlannerDragRefineAction is not Any:
            try:
                PlannerDragRefineAction.model_validate(result)
            except Exception as e:
                self._log(f"Drag refinement validation failed: {e}")

        # Ensure we have valid adjustment values and return compatible dict
        return {
            "analysis": result.get("analysis", ""),
            "goal": result.get("goal", primary_goal),
            "action": "refine_drag",
            "decision": result.get("decision", "accept"),
            "dx": float(result.get("dx", 0)),
            "dy": float(result.get("dy", 0)),
        }


    # ------------------------------------------------------------------
    # Text reading (for text captchas)
    # ------------------------------------------------------------------
    def read_text(self, image_path: str) -> str:
        """
        Read distorted text from a text captcha.

        Returns:
            The text to type
        """
        response, _ = self._chat_with_image(TEXT_READ_PROMPT, image_path)
        data = self._parse_json(response)
        
        # Validation
        if PlannerPlan is not Any:
            try:
                PlannerPlan.model_validate(data)
            except Exception as e:
                self._log(f"Text reading validation failed: {e}")
                
        action = data.get("action", {})
        return action.get("text", "")

    # ------------------------------------------------------------------
    # Grid Selection
    # ------------------------------------------------------------------
    def get_grid_selection(self, image_path: str, rows: int, cols: int, instruction: str = "") -> List[int]:
        """
        Ask which numbers to select in the grid.
        Returns a list of selected cell numbers.
        """
        total = rows * cols

        self._log("ActionPlanner.get_grid_selection called")
        if instruction:
            self._log(f"  external instruction: '{instruction}'")
        self._log(f"  grid: {rows}x{cols}")

        grid_hint = ""
        if rows == 4 and cols == 4:
            grid_hint = "Hint: Single large image split into tiles. Select ALL parts."
        elif rows == 3 and cols == 3:
            grid_hint = "Hint: Separate images. Select only clear matches."

        instruction_context = ""
        if instruction:
            instruction_context = f"External Instruction Context: \"{instruction}\" (Use this if the image text is unclear)"

        prompt = SELECT_GRID_PROMPT.format(
            rows=rows, 
            cols=cols, 
            total=total, 
            instruction_context=instruction_context,
            grid_hint=grid_hint
        )
        
        response, _ = self._chat_with_image(prompt, image_path)
        data = self._parse_json(response)
        
        # Validation
        if PlannerPlan is not Any:
            try:
                PlannerPlan.model_validate(data)
            except Exception as e:
                self._log(f"Grid selection validation failed: {e}")

        action = data.get("action", {})
        selected = action.get("target_ids", [])
        
        # Log model analysis (if present)
        self._log(f"Analysis: {data.get('analysis', 'N/A')}")
        self._log(f"Final selection: {selected}")

        return selected
