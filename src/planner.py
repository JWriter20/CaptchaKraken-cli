"""
ActionPlanner - LLM-based captcha planning with tool support.

Supports:
1. Grid selection - which numbered squares to click
2. Tool-aware planning - can request detect() calls
3. Drag refinement - iterative adjustment with visual feedback
4. Text reading - OCR for text captchas
5. Video captchas - plan actions for video captchas

Backends: vllm, transformers
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


'''
Author's note:
For now we are prioritizing quality over speed. We may be able to have the model pass in descriptions instead of ids, but this often causes less often detections.
It is better in my opinion to do two model queries instead of one to call a detect() tool call, then use the id, rather than passing in description
directly into actions then calling detect in the background and returning immediately. 
'''

# For general planning with tool support
PLAN_WITH_TOOLS_PROMPT = """Your task is to solve the captcha using the tools provided, look at the top of the image for instructions.

{history_section}

Choose ONE action, ONE tool call, or follow the DRAG REFINEMENT process.

DRAG REFINEMENT PROCESS:
Step 1: Detect all draggable items (if not already done).
- {{ "objectDescription": "colored segment" }}

Step 2: Initial guess for destination.
- "output": [
    {{
      "Action": "simulate_drag",
      "SourceID": N,
      "DestinationDescription": "...",
      "EstimatedPosition": {{ "x": 1-1000, "y": 1-1000 }}
    }}
  ]

Step 3: Refinement (handled by a specialized LoRa).

DIRECT ACTIONS & TOOLS:
- {{ "action": "click", "target_ids": [id1, id2, ...] }}
- {{ "action": "drag", "source_id": N, "target_id": N }}
- {{ "action": "type", "text": "...", "target_id": N }}
- {{ "action": "wait", "duration_ms": 500 }}
- {{ "action": "done" }}
- detect(object_description, max_items)

CRITICAL:
1. For drag puzzles requiring precision, ALWAYS start with Step 1: {{ "objectDescription": "colored segment" }}.
2. After Step 1 is done and items are labelled, proceed to Step 2.
3. Respond ONLY with JSON.

Template:
{{
  "goal": "Explain what you see and why you are taking this action",
  "action": {{ ... }}
  // OR: "tool_calls": [ {{ "name": "...", "args": {{ ... }} }} ]
  // OR: "objectDescription": "colored segment"
  // OR: "output": [ {{ "Action": "simulate_drag", ... }} ]
}}"""


# For text captchas: read the text
TEXT_READ_PROMPT = """Read the distorted text in this captcha image.
Look carefully at each character, accounting for distortion, rotation, and noise.

Typically, these captchas contain exactly 6 characters.

Respond ONLY with JSON matching this structure:
{{
  "action": "type",
  "text": "the text identified",
  "target_id": N
}}"""


# For grid selection captchas
SELECT_GRID_PROMPT = """Solve the captcha grid by choosing the cell numbers that match the description from the captcha image prompt.

Grid: {rows}x{cols} ({total} cells)
{grid_hint}

If no tiles match the description (e.g., they have all been cleared or none were present), return an empty list for target_ids: [].

Return JSON format ALWAYS:
{{
    "action": "click",
    "target_ids": [list of cell numbers (1-{total})]
}}"""


# For drag refinement: iterative adjustment (Step 3)
DRAG_REFINE_PROMPT = """You are refining a drag action for a captcha puzzle.

Given the history of previous guesses and the current image, estimate the remaining distance to the target.

Input format:
[
  {{
    "DestinationDescription": "...",
    "SourceId": N,
    "History": [
      {{
        "guess": {{ "x": N, "y": N }},
        "estimatedVerticalDistanceFromTarget": N,
        "estimatedHorizontalDistanceFromTarget": N
      }}
    ]
  }}
]

Evaluation Criteria:
1. Vertical Position: Is the object too high (negative) or too low (positive)?
2. Horizontal Position: Is it too far left (negative) or too far right (positive)?

Respond ONLY with JSON:
[
  {{
    "SourceId": N,
    "estimatedVerticalDistanceFromTarget": Y_DISTANCE,
    "estimatedHorizontalDistanceFromTarget": X_DISTANCE
  }}
]

CRITICAL:
- Distances are on a 1-1000 scale.
- 0 means perfectly aligned.
- -100 means 100 units above/left of target.
- 100 means 100 units below/right of target."""


class ActionPlanner:
    """
    LLM-based captcha planner with tool support.

    Supports backends: vllm, transformers
    """

    def __init__(
        self,
        backend: Literal["transformers", "vllm", "captchaKrakenApi"] = "captchaKrakenApi",
        model: Optional[str] = None,
        debug_callback: Optional[Any] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ):
        self.backend = backend
        self.debug_callback = debug_callback
        self.config = kwargs
        self._model = None
        self._processor = None
        self.token_usage: List[Dict[str, Any]] = []
        
        # API config for captchaKrakenApi backend
        self.base_url = base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        self.api_key = api_key or os.getenv("CAPTCHA_KRAKEN_API_KEY") or os.getenv("VLLM_API_KEY", "EMPTY")
        
        self.chat_callback = kwargs.get("chat_callback")

        # Default models per backend
        if model is None:
            if backend == "transformers":
                # Get recommended model based on hardware
                rec_model, _, rec_msg = get_recommended_model_config()
                if rec_model == "API":
                    self._log(f"WARNING: {rec_msg}")
                    # Default to the full merged model if they still want to try locally
                    self.model = "Jake-Writer-Jobharvest/qwen3-vl-8b-merged-BF16"
                else:
                    self._log(f"Recommended model: {rec_model} ({rec_msg})")
                    self.model = rec_model
            elif backend == "vllm" or backend == "captchaKrakenApi":
                self.model = "Qwen/Qwen3-VL-8B-Instruct-FP8"
        else:
            self.model = model

        # LoRA configurations
        self.loras = {
            "general": "Jake-Writer-Jobharvest/qwen3-vl-8b-general-lora",
            "grid": "Jake-Writer-Jobharvest/qwen3-vl-8b-grid-lora"
        }

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
    def _chat_with_image(self, prompt: str, image_path: Union[str, List[str]], lora_name: Optional[str] = None) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Send a prompt + image(s)/video(s) to the LLM backend, get response."""
        self._log(f"Backend: {self.backend}, Model: {self.model}")
        
        # Use callback if provided (e.g., for docker server integration)
        if self.chat_callback:
            self._log(f"Using chat_callback...")
            return self.chat_callback(prompt, image_path, lora_name)

        image_paths = [image_path] if isinstance(image_path, str) else image_path
        self._log(f"Input paths: {image_paths}")

        if self.backend == "transformers":
            # Check if we should use the tool server for warm inference
            tool_server_url = os.getenv("CAPTCHA_TOOL_SERVER")
            if tool_server_url:
                import requests
                self._log(f"Calling tool server for planning: {tool_server_url}/plan")
                headers = {}
                api_key = os.getenv("VLLM_API_KEY")
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                
                try:
                    resp = requests.post(
                        f"{tool_server_url}/plan",
                        json={"image_paths": image_paths, "prompt": prompt, "lora": lora_name},
                        headers=headers,
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
            
            # Get device before model initialization so it's available later
            from .hardware import get_device
            device = get_device()
            
            if self._model is None:
                self._log(f"Loading transformers model: {self.model}")
                self._processor = AutoProcessor.from_pretrained(self.model, trust_remote_code=True)
                
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
                
                # Load LoRAs if using transformers
                from peft import PeftModel
                for name, path in self.loras.items():
                    self._log(f"Loading LoRA adapter: {name} from {path}")
                    try:
                        if hasattr(self._model, "add_adapter"):
                            self._model.load_adapter(path, adapter_name=name)
                        else:
                            # Fallback if load_adapter is not available (older peft)
                            self._model = PeftModel.from_pretrained(self._model, path, adapter_name=name)
                    except Exception as e:
                        self._log(f"Failed to load LoRA {name}: {e}")

            # Switch adapter if needed
            if lora_name and lora_name in self.loras:
                self._log(f"Switching to LoRA: {lora_name}")
                try:
                    self._model.set_adapter(lora_name)
                except Exception as e:
                    self._log(f"Failed to set adapter {lora_name}: {e}")
            else:
                # Use base model if no lora requested or not found
                try:
                    self._model.disable_adapters()
                except:
                    pass
            
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert captcha solver. Think carefully about the visual cues. Respond ONLY with the JSON action."
                },
                {"role": "user", "content": []}
            ]
            processed_images = []
            
            for path in image_paths:
                is_video = any(path.lower().endswith(ext) for ext in [".mp4", ".gif", ".avi"])
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
                    max_new_tokens=512, # Reduced as we no longer want analysis
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

        if self.backend == "vllm":
            try:
                from vllm import LLM, SamplingParams
                from vllm.lora.request import LoRARequest
                from PIL import Image
                
                # Patch for Qwen3-VL config issues in some vLLM/transformers combinations
                try:
                    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
                    if not hasattr(Qwen3VLTextConfig, "tie_word_embeddings"):
                        Qwen3VLTextConfig.tie_word_embeddings = False
                except ImportError:
                    pass
            except ImportError:
                raise ImportError("vLLM not installed. Please install with 'pip install vllm'")

            if self._model is None:
                self._log(f"Loading vLLM model: {self.model}")
                # Default to 0.75 to allow SAM 3 to coexist on 32GB cards
                gpu_memory_utilization = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.75"))
                quantization = os.getenv("VLLM_QUANTIZATION", "fp8")
                
                self._model = LLM(
                    model=self.model,
                    trust_remote_code=True,
                    max_model_len=8192,
                    gpu_memory_utilization=gpu_memory_utilization,
                    enable_lora=True,
                    max_loras=4,
                    quantization=quantization,
                    **self.config
                )
            
            sampling_params = SamplingParams(
                temperature=0,
                max_tokens=512,
                stop=["<|im_end|>", "<|endoftext|>"]
            )

            # Handle LoRA request
            lora_request = None
            if lora_name and lora_name in self.loras:
                lora_path = self.loras[lora_name]
                self._log(f"Using LoRA: {lora_name} ({lora_path})")
                # Use a stable integer ID for the LoRA
                lora_id = 1 if lora_name == "general" else 2
                lora_request = LoRARequest(lora_name, lora_id, lora_path)

            # vLLM multi-modal input format for Qwen2-VL style models
            mm_data = {}
            if len(image_paths) == 1:
                mm_data = {"image": Image.open(image_paths[0]).convert("RGB")}
            else:
                mm_data = {"image": [Image.open(p).convert("RGB") for p in image_paths]}

            # vLLM expects the prompt with vision tokens if using multi-modal
            # For Qwen2-VL/Qwen3-VL, it typically uses <|vision_start|><|image_pad|><|vision_end|> placeholders
            vision_tokens = "<|vision_start|><|image_pad|><|vision_end|>" * len(image_paths)
            full_prompt = f"<|im_start|>system\nYou are an expert captcha solver. Think carefully about the visual cues. Respond ONLY with the JSON action.<|im_end|>\n<|im_start|>user\n{vision_tokens}{prompt}<|im_end|>\n<|im_start|>assistant\n"

            outputs = self._model.generate(
                {
                    "prompt": full_prompt,
                    "multi_modal_data": mm_data,
                },
                sampling_params=sampling_params,
                lora_request=lora_request
            )
            
            result = outputs[0].outputs[0].text
            return result, None

        if self.backend == "captchaKrakenApi":
            import requests
            import base64
            from mimetypes import guess_type

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            content = []
            
            # Add images first
            for path in image_paths:
                is_video = any(path.lower().endswith(ext) for ext in [".mp4", ".gif", ".avi", ".webm"])
                if is_video:
                    self._log("Video not fully supported in captchaKrakenApi backend via HTTP yet (needs frame extraction). Sending prompt only.")
                    # TODO: Extract frames and send as images
                    continue
                
                with open(path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                    mime_type, _ = guess_type(path)
                    if mime_type is None: mime_type = "image/jpeg"
                    
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}"
                        }
                    })

            # Add text prompt
            content.append({
                "type": "text",
                "text": prompt
            })

            messages = [
                {
                    "role": "system",
                    "content": "You are an expert captcha solver. Think carefully about the visual cues. Respond ONLY with the JSON action."
                },
                {
                    "role": "user",
                    "content": content
                }
            ]

            # Resolve LoRA name to model ID if possible
            model_id = self.model
            if lora_name:
                if lora_name == "general":
                    model_id = "general-lora"
                elif lora_name == "grid":
                    model_id = "grid-lora"
                else:
                    model_id = lora_name

            payload = {
                "model": model_id,
                "messages": messages,
                "max_tokens": 512,
                "temperature": 0
            }
            
            self._log(f"Sending request to {self.base_url}/chat/completions with model {payload['model']}")
            
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=120
                )
                response.raise_for_status()
                data = response.json()
                
                result = data['choices'][0]['message']['content']
                usage = data.get('usage', {})
                self.token_usage.append(usage)
                
                return result, None
            except Exception as e:
                self._log(f"OpenAI API call failed: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    self._log(f"Response: {e.response.text}")
                raise e

        raise ValueError(f"Unknown backend: {self.backend}")

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Extract and parse JSON from LLM response."""
        self._log(f"Raw response to parse: {text}")
        text = text.strip()

        # Handle <think> blocks if present
        if "<think>" in text:
            text = text.split("</think>")[1].strip()

        # Remove markdown code blocks
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0]

        # Find JSON object or list
        start_obj = text.find("{")
        start_list = text.find("[")
        
        if start_obj != -1 and (start_list == -1 or start_obj < start_list):
            end = text.rfind("}") + 1
            if end > start_obj:
                text = text[start_obj:end]
        elif start_list != -1:
            end = text.rfind("]") + 1
            if end > start_list:
                text = text[start_list:end]

        self._log(f"Cleaned text to parse: {text}")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            self._log("JSON decode failed")
            return {}

    # ------------------------------------------------------------------
    # Tool-aware planning
    # ------------------------------------------------------------------
    def plan_with_tools(
        self,
        image_path: str,
        objects: Optional[List[Dict[str, Any]]] = None,
        history: Optional[List[str]] = None,
        context_image_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Plan the next action, potentially requesting a tool call.

        Args:
            image_path: Path to the captcha image or video
            objects: Optional list of detected objects
            history: Optional list of previous actions
            context_image_path: Optional path to a static image with overlays/labels

        Returns:
            Dict containing the planner's response, following the PlannerPlan schema.
        """
        object_list_section = "No objects detected yet. Use the 'detect' tool to find items."
        if objects:
            lines = []
            for obj in sorted(objects, key=lambda x: x.get("id", 0)):
                obj_id = obj.get("id")
                label = obj.get("label", "object")
                bbox = obj.get("bbox")
                lines.append(f"- Object {obj_id}: {label} at {bbox}")
            object_list_section = "\n".join(lines)

        history_section = ""
        if history:
            lines = ["Action History (most recent last):"]
            for i, h in enumerate(history):
                lines.append(f"{i+1}. {h}")
            lines.append("\\nAVOID repeating failed actions.")
            history_section = "\\n".join(lines)

        prompt = PLAN_WITH_TOOLS_PROMPT.format(
            history_section=history_section,
        )
        
        images = [image_path]
        if context_image_path and context_image_path != image_path:
            images.append(context_image_path)
            prompt += "\n\nNOTE: Multiple images/media provided. The first item is the raw captcha (could be video), the second is a static frame with red boxes and Object IDs for reference."

        response, _ = self._chat_with_image(prompt, images, lora_name="general")
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
        current_guess: Dict[str, int],
        history: List[Dict[str, Any]],
        source_id: int = 0,
        target_description: str = "matching slot",
    ) -> List[Dict[str, Any]]:
        """
        Refine a drag destination with visual feedback (Step 3).

        Args:
            image_path: Path to image with drag overlay
            current_guess: Current guess {"x": 1-1000, "y": 1-1000}
            history: List of previous refinements:
                [{"guess": {"x": N, "y": N}, "v_dist": N, "h_dist": N}]
            source_id: ID of the item being dragged
            target_description: Description of where it should go

        Returns:
            List of refinement results as per Step 3 instructions.
        """
        history_data = []
        for h in history:
            history_data.append({
                "guess": h.get("guess", {"x": 500, "y": 500}),
                "estimatedVerticalDistanceFromTarget": h.get("v_dist", 0),
                "estimatedHorizontalDistanceFromTarget": h.get("h_dist", 0)
            })
        
        # Add current guess to history for the prompt context
        history_data.append({
            "guess": current_guess,
            "estimatedVerticalDistanceFromTarget": "???",
            "estimatedHorizontalDistanceFromTarget": "???"
        })

        input_context = [
            {
                "DestinationDescription": target_description,
                "SourceId": source_id,
                "History": history_data
            }
        ]
        
        prompt = DRAG_REFINE_PROMPT + f"\n\nInput Context:\n{json.dumps(input_context, indent=2)}"

        # Use base model for drag refinement as requested
        response, _ = self._chat_with_image(prompt, image_path, lora_name=None)
        result = self._parse_json(response)
        
        # If the result is a dict (e.g. from some fallback), wrap it in a list
        if isinstance(result, dict):
            result = [result]

        return result


    # ------------------------------------------------------------------
    # Text reading (for text captchas)
    # ------------------------------------------------------------------
    def read_text(self, image_path: str) -> str:
        """
        Read distorted text from a text captcha.

        Returns:
          the text to type
        """
        response, _ = self._chat_with_image(TEXT_READ_PROMPT, image_path, lora_name="general")
        data = self._parse_json(response)
        
        # Validation against PlannerTypeAction instead of PlannerPlan
        if PlannerTypeAction is not Any:
            try:
                PlannerTypeAction.model_validate(data)
            except Exception as e:
                self._log(f"Text reading validation failed: {e}")
                
        return data.get("text", "")

    # ------------------------------------------------------------------
    # Grid Selection
    # ------------------------------------------------------------------
    def get_grid_selection(self, image_path: str, rows: int, cols: int) -> List[int]:
        """
        Ask which numbers to select in the grid.
        Returns a list of selected cell numbers.
        """
        total = rows * cols

        self._log("ActionPlanner.get_grid_selection called")
        self._log(f"  grid: {rows}x{cols}")

        grid_hint = ""
        if rows == 4 and cols == 4:
            grid_hint = "Hint: Select all tiles that make up the object in the image, even if a tile only has a small part of the object."
        elif rows == 3 and cols == 3:
            grid_hint = "Hint: Separate images. Select only clear matches."

        prompt = SELECT_GRID_PROMPT.format(
            rows=rows, 
            cols=cols, 
            total=total, 
            grid_hint=grid_hint
        )
        
        response, _ = self._chat_with_image(prompt, image_path, lora_name="grid")
        data = self._parse_json(response)
        
        # Handle case where model returns raw list [1, 2, 3] instead of object
        if isinstance(data, list):
            self._log(f"Model returned raw list: {data}. Wrapping in target_ids.")
            data = {"target_ids": data, "action": "click"}

        # Validation against PlannerClickAction instead of PlannerPlan
        if PlannerClickAction is not Any:
            try:
                PlannerClickAction.model_validate(data)
            except Exception as e:
                self._log(f"Grid selection validation failed: {e}")

        selected = data.get("target_ids", [])
        
        # Log final selection
        self._log(f"Final selection: {selected}")

        return selected
