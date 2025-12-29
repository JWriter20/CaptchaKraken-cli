"""
ActionPlanner - LLM-based captcha planning with tool support.

Supports:
1. Grid selection - which numbered squares to click
2. Tool-aware planning - can request detect() or point() calls
3. Drag refinement - iterative adjustment with visual feedback
4. Text reading - OCR for text captchas

Backends: ollama, gemini
"""

import json
import os
import sys
import base64
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

# Timing helper (opt-in via CAPTCHA_TIMINGS=1)
from .timing import timed

# Debug flag - set via CAPTCHA_DEBUG=1 environment variable
DEBUG = os.getenv("CAPTCHA_DEBUG", "0") == "1"


try:
    # New Gemini SDK (direct Gemini API, no Vertex AI dependency)
    from google import genai  # type: ignore[import-not-found]
    from google.genai.types import GenerateContentConfig, Part  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    genai = None  # type: ignore[assignment]
    GenerateContentConfig = None  # type: ignore[assignment]
    Part = None  # type: ignore[assignment]



# For general planning with tool support
PLAN_WITH_TOOLS_PROMPT = """You are an expert captcha solver. Solve this captcha image by analyzing its type and using the available tools.

{instruction}

Types of captchas you support:
- Clicking described objects (e.g. "click the two similar shapes")
- Text captchas (e.g. "type the text in the image")
- Drag puzzles (e.g. "drag the puzzle piece to the correct position", "complete the image")
- Analyzing connected lines (e.g. "which watermelon is connected to the bird?")

{object_list_section}

{history_section}

Step 1: VISUAL ANALYSIS
Briefly describe the objects you see in the image. Identify the captcha type.
If the goal involves "similar objects", "matching shapes", or "odd one out", explicitly IDENTIFY the specific shape or object class.

Step 2: GOAL IDENTIFICATION
Identify the *goal* in concrete visual terms.
BE SPECIFIC about locations and explicitly state where exactly any action should take place (e.g. "top right", "center left").
Explain WHY this is the correct target (matching rotation, shape, color, etc).

Step 3: ACTION PLANNING
Decide on the best tool or action.

You have three tools available:
1. detect(object_class, max_items) - Find instances of an object (e.g., "traffic light", "bus", "checkbox", "verify button").
   - object_class: Description of what to find.
   - max_items: (Optional) Maximum number of items to return. Use 1 if you only need one specific thing.

2. simulate_drag(source, target_hint, target_object_id, source_quality, refined_source, location_hint) - Simulate a drag operation with visual feedback.
       Use when: You need to drag an item to a target that requires precise visual alignment (e.g., "fit the puzzle piece", "complete the image").
       - source: The ID of the object to drag (e.g. "Object 4") OR a description.
       - target_hint: description of where it roughly should go (e.g., "matching slot").
       - target_object_id: (Optional) The ID of the object that acts as the target base.
       - location_hint: [x, y] (0.0-1.0) indicating the rough center of the target destination.

3. find_connected_elems(instruction) - Analyze lines or connections between elements.
       Use when: The puzzle asks which items are connected or follow a path.
       - instruction: The connection to look for (e.g., "watermelon connected to bird").

    NOTE: The image has already been segmented and objects are labeled with red boxes and IDs (if any were found).
    Use the "Detected Objects" list above to refer to specific items by their ID (e.g., "Object 1").

    If you need to interact with multiple specific items (e.g., "click the two similar shapes"), PREFER using multiple detect() calls with max_items=1.

    You can also return a direct action:
    - click: Click on something (provide target_description).
    - drag: Drag something (provide source and target descriptions). Use this ONLY for simple sliders.
    - type: Type text (provide the text to type, typically 6 characters).
    - wait: Wait for loading.
    - done: Captcha is already solved.

    Respond ONLY with JSON:
    {
      "analysis": "Your visual analysis and captcha type identification",
      "goal": "The specific goal you identified",
      "tool_calls": [
        {
          "name": "detect" | "simulate_drag" | "find_connected_elems",
          "args": {...}
        }
      ]
    }

Or return an action:
{{
  "analysis": "Your visual analysis...",
  "goal": "The specific goal...",
  "action_type": "click" | "drag" | "type" | "wait" | "done",
  "target_description": "what to click (for click)",
  "source_description": "what to drag (for drag)",
  "drag_target_description": "where to drag to (for drag)",
  "text": "text to type (for type)",
  "duration_ms": 500,
  "reasoning": "brief explanation that refers back to the puzzle goal you identified"
}}"""


# For text captchas: read the text
TEXT_READ_PROMPT = """Read the distorted text in this captcha image.
Look carefully at each character, accounting for distortion, rotation, and noise.

Typically, these captchas contain exactly 6 characters.
Only respond with more than 6 or fewer than 6 characters if you are confident that the captcha clearly shows more or fewer characters.

Respond ONLY with JSON:
{{
  "text": "the text to type"
}}"""


# For grid selection captchas
SELECT_GRID_PROMPT = """Solve the captcha grid.
Instruction: "{instruction}"
Grid: {rows}x{cols} ({total} cells)
{grid_hint}

Respond ONLY with JSON:
{{
  "analysis": "Briefly describe which cells contain the target objects",
  "selected_numbers": [list of integers (1-{total})]
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
  "conclusion": "Specific critique of vertical/horizontal alignment and edge connectivity",
  "decision": "accept" | "adjust",
  "dx": 0.0,
  "dy": 0.0
}}"""

# For verifying source crop tightness
VERIFY_CROP_PROMPT = """You are refining a selection for a drag-and-drop task.
Instruction: {instruction}

The system detected "Object {id}" as the item to drag.
Look at the RED box labeled "{id}" in the image.

Does this box represent a TIGHT isolation of the movable piece?
OR does it include a lot of background, other slots, or empty space that should be excluded?

For example:
- If the movable piece is a "white W" but the box covers a large black column containing the W, that is BAD (not tight).
- If the box tightly hugs the shape of the item, that is GOOD (tight).

Respond ONLY with JSON:
{{
  "analysis": "Describe what Object {id} contains vs what the actual movable piece is",
  "is_tight": true | false,
  "refined_prompt": "If false, provide a visual description to find the tight item (e.g. 'white letter W')"
}}"""


class ActionPlanner:
    """
    LLM-based captcha planner with tool support.

    Supports backends: ollama, gemini
    """

    def __init__(
        self,
        backend: Literal["ollama", "gemini", "openrouter"] = "gemini",
        model: Optional[str] = None,
        ollama_host: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        openrouter_key: Optional[str] = None,
        debug_callback: Optional[Any] = None,
    ):
        self.backend = backend
        self.ollama_host = ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.openrouter_key = openrouter_key or os.getenv("OPENROUTER_KEY")
        self.debug_callback = debug_callback
        self._genai_client = None
        self.token_usage: List[Dict[str, Any]] = []

        # Configure Gemini if selected
        if self.backend == "gemini":
            if not genai:
                raise ImportError("google-genai required. Install: pip install google-genai")

            if not self.gemini_api_key:
                raise ValueError("GEMINI_API_KEY is required for Gemini backend when not using Vertex AI.")

            # Simple direct Gemini client using API key
            self._genai_client = genai.Client(api_key=self.gemini_api_key)  # type: ignore[call-arg]

        # Default models per backend
        if model is None:
            if backend == "ollama":
                self.model = "qwen3-vl:4b"
            elif backend == "gemini":
                # Default Gemini model; can be overridden by caller
                self.model = "gemini-2.5-flash-lite"
            elif backend == "openrouter":
                self.model = "google/gemini-2.0-flash-lite-preview-02-05:free"
            else:
                raise ValueError(f"Unknown backend: {backend}")
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

        usage = None

        if self.backend == "ollama":
            import ollama

            self._log("Waiting for Ollama response...")
            client = ollama.Client(host=self.ollama_host)
            # Ollama chat supports multiple images in the 'images' list
            response = client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt, "images": image_paths}],
                options={
                    "temperature": 0.0,
                    "num_predict": 1024,
                },
            )
            result = response["message"]["content"]
            
            if "prompt_eval_count" in response:
                usage = {
                    "input_tokens": response["prompt_eval_count"],
                    "output_tokens": response["eval_count"],
                    "model": self.model
                }
                self.token_usage.append(usage)
            
            return result, usage

        if self.backend == "gemini":
            if not genai or not GenerateContentConfig or not Part:
                raise ValueError("google-genai not installed or incomplete. Install: pip install google-genai")

            if self._genai_client is None:
                if not self.gemini_api_key:
                    raise ValueError("GEMINI_API_KEY is required for Gemini backend when not using Vertex AI.")
                self._genai_client = genai.Client(api_key=self.gemini_api_key)  # type: ignore[call-arg]

            import mimetypes
            contents: List[Any] = []

            for path in image_paths:
                mime_type, _ = mimetypes.guess_type(path)
                if not mime_type:
                    mime_type = "image/png"
                
                with open(path, "rb") as f:
                    data = f.read()
                    contents.append(Part.from_bytes(data=data, mime_type=mime_type))
            
            contents.append(prompt)

            # Configure response as JSON (planner expects JSON it can parse)
            config = GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
            )

            response = self._genai_client.models.generate_content(  # type: ignore[union-attr]
                model=self.model,
                contents=contents,
                config=config,
            )
            result = response.text
            
            if response.usage_metadata:
                usage = {
                    "input_tokens": response.usage_metadata.prompt_token_count,
                    "output_tokens": response.usage_metadata.candidates_token_count,
                    "cached_input_tokens": getattr(response.usage_metadata, "cached_content_token_count", 0),
                    "model": self.model
                }
                self.token_usage.append(usage)

            return result, usage

        if self.backend == "openrouter":
            import requests

            if not self.openrouter_key:
                raise ValueError("OPENROUTER_KEY is required for openrouter backend.")

            import mimetypes
            message_content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]

            for path in image_paths:
                mime_type, _ = mimetypes.guess_type(path)
                if not mime_type:
                    mime_type = "image/png"

                with open(path, "rb") as f:
                    image_base64 = base64.b64encode(f.read()).decode("utf-8")
                    message_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image_base64}"},
                    })

            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.openrouter_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/jakewriter/PlaywrightCaptchaKrakenJS",
                "X-Title": "CaptchaKraken",
            }

            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": message_content}],
                "temperature": 0.0,
                "response_format": {"type": "json_object"} if "json" in prompt.lower() else None,
            }

            self._log(f"Waiting for OpenRouter response ({self.model})...")
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

            if "choices" in data and len(data["choices"]) > 0:
                result = data["choices"][0]["message"]["content"]
            else:
                self._log(f"OpenRouter Error: {data}")
                raise ValueError(f"OpenRouter returned no results: {data}")

            # ... usage ...
            if "usage" in data:
                usage = {
                    "input_tokens": data["usage"].get("prompt_tokens", 0),
                    "output_tokens": data["usage"].get("completion_tokens", 0),
                    "model": self.model,
                }
                self.token_usage.append(usage)

            return result, usage

        raise ValueError(f"Unknown backend: {self.backend}")

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
            Dict with either:
            - "tool_call": {"name": "detect"|"point", "args": {...}}
            - "action_type": "click"|"drag"|"type"|"wait"|"done" + action details
        """
        object_list_section = ""
        if objects:
            lines = ["Detected Objects (ID: box [x, y, w, h]):"]
            for obj in sorted(objects, key=lambda x: x.get("id", 0)):
                lines.append(f"- Object {obj.get('id')}: {obj.get('bbox')}")
            lines.append("\\nUse these Object IDs in your reasoning.")
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
        return self._parse_json(response)

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
                [{"destination": [x, y], "conclusion": "...", "decision": "..."}]
            source_description: Description of the item being dragged
            target_description: Description of where it should go
            primary_goal: The specific goal identified by the planner

        Returns:
            {
                "conclusion": "assessment of current position",
                "decision": "accept" | "adjust",
                "dx": float,  # relative adjustment (-1 to 1)
                "dy": float   # relative adjustment (-1 to 1)
            }
        """
        # Format history for context
        if history:
            history_lines = ["Previous attempts:"]
            for i, h in enumerate(history):
                dest = h.get("destination", [0, 0])
                conclusion = h.get("conclusion", "")
                decision = h.get("decision", "")
                history_lines.append(
                    f"  {i + 1}. destination: ({dest[0]:.1%}, {dest[1]:.1%}), "
                    f'conclusion: "{conclusion}", decision: {decision}'
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

        # Ensure we have valid adjustment values
        return {
            "conclusion": result.get("conclusion", ""),
            "decision": result.get("decision", "accept"),
            "dx": float(result.get("dx", 0)),
            "dy": float(result.get("dy", 0)),
        }

    # ------------------------------------------------------------------
    # Source Verification
    # ------------------------------------------------------------------
    def verify_source_crop(
        self,
        image_path: str,
        source_object: Dict[str, Any],
        instruction: str
    ) -> Dict[str, Any]:
        """
        Verify if the selected source object is a tight crop or needs refinement.
        """
        prompt = VERIFY_CROP_PROMPT.format(
            id=source_object.get("id"),
            instruction=instruction
        )
        response, _ = self._chat_with_image(prompt, image_path)
        return self._parse_json(response)

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
        result = self._parse_json(response)
        return result.get("text", "")

    # ------------------------------------------------------------------
    # Grid Selection
    # ------------------------------------------------------------------
    def get_grid_selection(self, image_path: str, rows: int, cols: int, instruction: str = "Solve the captcha by selecting the correct images") -> List[int]:
        """
        Ask which numbers to select in the grid.
        Returns a list of selected cell numbers.
        """
        total = rows * cols

        self._log("ActionPlanner.get_grid_selection called")
        self._log(f"  instruction: '{instruction}'")
        self._log(f"  grid: {rows}x{cols}")

        grid_hint = ""
        if rows == 4 and cols == 4:
            grid_hint = "Hint: Single large image split into tiles. Select ALL parts."
        elif rows == 3 and cols == 3:
            grid_hint = "Hint: Separate images. Select only clear matches."

        prompt = SELECT_GRID_PROMPT.format(
            rows=rows, 
            cols=cols, 
            total=total, 
            instruction=instruction,
            grid_hint=grid_hint
        )
        
        response, _ = self._chat_with_image(prompt, image_path)
        result = self._parse_json(response)

        selected = result.get("selected_numbers", [])
        
        # Log model reasoning (if present)
        self._log(f"Analysis: {result.get('analysis', 'N/A')}")
        self._log(f"Final selection: {selected}")

        return selected

    # ------------------------------------------------------------------
    # Item Selection (for SAM2 candidate filtering)
    # ------------------------------------------------------------------
    def select_items(self, image_path: str, instruction: str, candidates_count: int) -> List[int]:
        """
        Select items from a set of labeled candidates based on instruction.
        
        Args:
            image_path: Path to image with labeled candidates (numbers on objects)
            instruction: Description of what to select (e.g. "the letter W")
            candidates_count: Number of candidates available (for validation)
            
        Returns:
            List of Selected IDs (int)
        """
        prompt = f"""You are a vision system selecting objects.
The image shows {candidates_count} objects detected by a segmentation system.
Each important object is overlaid with a SEMI-TRANSPARENT colored mask and a numeric label (ID).

Task: We want to identify the segments that match this description: "{instruction}"

CRITICAL INSTRUCTIONS:
1. Ignore the color of the mask. Focus on the SHAPE and CONTEXT.
2. Each mask is a different color, and each mask is numbered with an ID.
3. It is possible that our target object is fragmented into multiple masks. If so, select all relevant IDs.
4. ONLY select the mask IDs that directly match the description, DO NOT include any backgrounds or other objects

Respond ONLY with JSON:
{{
  "analysis": "Describe the candidates that look relevant, noting their shape matches despite mask color",
  "selected_ids": [id1, id2, ...], // often only one id, don't feel the need to select multiple ids always
  "confidence": "high" | "medium" | "low",
  "reasoning": "Why these IDs match the description"
}}"""
        
        response, _ = self._chat_with_image(prompt, image_path)
        result = self._parse_json(response)
        
        self._log(f"Selection reasoning: {result.get('reasoning')}")
        
        selected_ids = result.get("selected_ids")
        valid_ids = []
        
        if isinstance(selected_ids, int):
            selected_ids = [selected_ids]
            
        if isinstance(selected_ids, list):
            for pid in selected_ids:
                if isinstance(pid, int) and 0 <= pid < candidates_count:
                    valid_ids.append(pid)
                    
        return valid_ids
