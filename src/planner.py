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
from typing import Any, Dict, List, Literal, Optional

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


# For grid selection: select numbered squares
SELECT_GRID_PROMPT = """You are solving a visual CAPTCHA puzzle.

IMPORTANT: Look at the instruction text AND the reference image at the top of the captcha.

This is a {rows}x{cols} grid of images. Each cell has a red number (1-{total}) in the top-left corner.

TASK: First analyze, then select. Think step by step:

1. What does the instruction/header text say?
2. What does the reference image (if any) show?
3. What type of thing should you be looking for?
4. Look at EACH numbered cell carefully - what does it contain?
5. Which cells match the criteria?

Respond with JSON. CRITICAL: Fill in the reasoning fields FIRST, then select numbers based on your reasoning:
{{
  "instruction_text": "copy the exact instruction text from the captcha header",
  "reference_image_shows": "describe what the reference/example image shows (e.g., 'an egg in a nest')",
  "looking_for": "what type of object/creature/thing to select (e.g., 'birds - creatures that hatch from eggs')",
  "cell_contents": {{
    "1": "elephant",
    "2": "parrots (birds)",
    "3": "pigeons (birds)"
  }},
  "selected_numbers": [2, 3],
  "reasoning": "Selected cells containing birds because birds hatch from eggs like in the reference"
}}"""


# For general planning with tool support
PLAN_WITH_TOOLS_PROMPT = """You are a captcha solver. Analyze this image and decide the next action.

{instruction}

{object_list_section}

{history_section}

Step 1: VISUAL ANALYSIS
Briefly describe the objects you see in the image.
If the goal involves "similar objects", "matching shapes", or "odd one out", explicitly IDENTIFY the specific shape or object class.
Example: "I see three wireframe cubes and one solid cube." or "I see multiple traffic lights."

Step 2: GOAL IDENTIFICATION
Identify the *goal* in concrete visual terms.
BE SPECIFIC about locations and explicitly state where exactly any action should take place (e.g. "top right", "center left").
Explain WHY this is the correct target (matching rotation, shape, color, etc).
(e.g., "drag the moveable letter 'W' in the top right to the center-right 'W' shaped outline because it matches the rotation and shape of the draggable W").

Step 3: ACTION PLANNING
Decide on the best tool or action.

You have three tools available:
1. detect(object_class) - Find all instances of an object (e.g., "traffic light", "bus", "open-frame cube")
   Use when: You need to find/click multiple instances of a specific object class.

2. point(target) - Find a single element (e.g., "checkbox", "verify button", "slider handle", "wireframe cube on the left")
   Use when: You need to click/interact with one specific element.

    3. simulate_drag(source, target_hint, target_object_id, source_quality, refined_source, location_hint) - Simulate a drag operation with visual feedback.
       Use when: You need to drag an item to a target that requires precise visual alignment (e.g., "fit the puzzle piece", "complete the line") and you don't have exact coordinates yet.
       - source: The ID of the object to drag (e.g. "Object 4") OR a description if not found.
       - target_hint: description of where it roughly should go (e.g., "matching slot", "center of Object 1")
       - target_object_id: (Optional) The ID of the object that acts as the target base (e.g. if dragging a hat to a head (Object 5), set this to 5).
       - source_quality: "properly-cropped" (if the Object ID is a tight crop of the item), "includes-source" (if the Object ID contains the item but also extra background), or "not-bounded" (if source is not an ID).
       - refined_source: If "includes-source", provide a visual description of the inner item (e.g. "white letter W").
       - location_hint: A 2-element list [x, y] (0.0-1.0) indicating the rough center of the target destination. E.g. [0.8, 0.2] is top-right.

    NOTE: The image has already been segmented and objects are labeled with red boxes and IDs (if any were found).
    Use the "Detected Objects" list above to refer to specific items by their ID (e.g., "Object 1").

    If you need to interact with multiple specific items (e.g., "click the two similar shapes"), PREFER using multiple point() or detect() calls to ensure all targets are hit.
    Example: instead of detect("wireframe cube"), use:
      point("wireframe cube on the left"),
      point("wireframe cube on the right")

    You can also return a direct action if you know exactly what to do:
    - click: Click on something (provide target_description for point() to find it). Do NOT click "Skip" or "Reload" unless you are 100% certain the puzzle is unsolvable.
    - drag: Drag something. Use this ONLY if you know the exact source and destination (e.g. "drag slider to the end"). For puzzle alignment, use simulate_drag() instead.
    - type: Type text (provide the text to type, maintain casing, typically 6 characters, ignore reflections, noise, etc.)
    - wait: Wait for loading
    - done: Captcha is already solved

    IMPORTANT:
    1. Do not give up easily. If the target is abstract or unclear, look for the BEST MATCHING shape or outline.
    2. For drag puzzles, if you see a movable piece, there IS a target. It might be a faint outline, a shadow, or a matching background pattern.
    3. If there is a MAIN BODY or BASE object (like a puzzle frame or a body needing a head), IDENTIFY IT and use it as 'target_object_id'.
    4. If you are unsure of the exact coordinates, use 'simulate_drag' with a descriptive target_hint so the vision system can find it.

    Respond ONLY with JSON. Either request tool(s):
    {{
      "analysis": "Your visual analysis of the objects and the puzzle type",
      "goal": "The specific goal you identified. MUST SPECIFY LOCATION (e.g. 'drag Object 4 to the W-shaped outline in the top-right')",
      "tool_calls": [
        {{
          "name": "detect" | "point" | "simulate_drag",
          "args": {{"object_class": "..."}} or {{"target": "..."}} or {{"source": "...", "target_hint": "...", "target_object_id": 1, "source_quality": "...", "refined_source": "...", "location_hint": [x, y]}}
        }}
      ]
    }}

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
        backend: Literal["ollama", "gemini"] = "gemini",
        model: Optional[str] = None,
        ollama_host: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        debug_callback: Optional[Any] = None,
    ):
        self.backend = backend
        self.ollama_host = ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.debug_callback = debug_callback
        self._genai_client = None

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
    def _chat_with_image(self, prompt: str, image_path: str) -> str:
        """Send a prompt + image to the LLM backend, get response."""
        self._log(f"Backend: {self.backend}, Model: {self.model}")
        self._log(f"Image path: {image_path}")
        self._log(f"=== PROMPT START ===\n{prompt}\n=== PROMPT END ===")

        if self.backend == "ollama":
            import ollama

            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt, "images": [image_path]}],
                options={"temperature": 0.0},
            )
            result = response["message"]["content"]
            self._log(f"=== RAW RESPONSE ===\n{result}\n=== END RESPONSE ===")
            return result

        if self.backend == "gemini":
            if not genai or not GenerateContentConfig or not Part:
                raise ValueError("google-genai not installed or incomplete. Install: pip install google-genai")

            if self._genai_client is None:
                if not self.gemini_api_key:
                    raise ValueError("GEMINI_API_KEY is required for Gemini backend when not using Vertex AI.")
                self._genai_client = genai.Client(api_key=self.gemini_api_key)  # type: ignore[call-arg]

            # Load image bytes
            import mimetypes

            mime_type, _ = mimetypes.guess_type(image_path)
            if not mime_type:
                mime_type = "image/png"

            with open(image_path, "rb") as f:
                image_bytes = f.read()

            # Configure response as JSON (planner expects JSON it can parse)
            config = GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
            )

            contents = [
                Part.from_bytes(data=image_bytes, mime_type=mime_type),
                prompt,
            ]

            response = self._genai_client.models.generate_content(  # type: ignore[union-attr]
                model=self.model,
                contents=contents,
                config=config,
            )
            result = response.text
            self._log(f"=== RAW RESPONSE ===\n{result}\n=== END RESPONSE ===")
            return result

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
    # Grid selection
    # ------------------------------------------------------------------
    def get_grid_selection(self, instruction: str, image_path: str, rows: int, cols: int) -> List[int]:
        """
        For grid-based image selection, ask which numbers to select.

        Args:
            instruction: The instruction text (e.g., "Select all crosswalks")
            image_path: Path to image with numbered overlay
            rows: Number of rows in grid
            cols: Number of cols in grid

        Returns:
            List of integers (1-based indices) to select.
        """
        total = rows * cols

        self._log("get_grid_selection called")
        self._log(f"  context instruction: '{instruction}'")
        self._log(f"  grid: {rows}x{cols} = {total} cells")

        prompt = SELECT_GRID_PROMPT.format(rows=rows, cols=cols, total=total)
        response = self._chat_with_image(prompt, image_path)
        result = self._parse_json(response)

        selected = result.get("selected_numbers", [])

        # Log all the detailed reasoning from the structured response
        self._log(f"Model extracted instruction: '{result.get('instruction_text', '(not provided)')}'")
        self._log(f"Reference image shows: '{result.get('reference_image_shows', '(not provided)')}'")
        self._log(f"Looking for: '{result.get('looking_for', '(not provided)')}'")

        cell_contents = result.get("cell_contents", {})
        if cell_contents:
            self._log("Cell contents analysis:")
            for cell_num, content in sorted(cell_contents.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
                self._log(f"  Cell {cell_num}: {content}")

        self._log(f"Selected numbers: {selected}")
        self._log(f"Reasoning: {result.get('reasoning', '(not provided)')}")

        return selected

    # ------------------------------------------------------------------
    # Tool-aware planning
    # ------------------------------------------------------------------
    def plan_with_tools(
        self,
        image_path: str,
        instruction: str = "",
        objects: Optional[List[Dict[str, Any]]] = None,
        history: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Plan the next action, potentially requesting a tool call.

        Args:
            image_path: Path to the captcha image
            instruction: Optional instruction text
            objects: Optional list of detected objects
            history: Optional list of previous actions

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
        response = self._chat_with_image(prompt, image_path)
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

        response = self._chat_with_image(prompt, image_path)
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
        response = self._chat_with_image(prompt, image_path)
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
        response = self._chat_with_image(TEXT_READ_PROMPT, image_path)
        result = self._parse_json(response)
        return result.get("text", "")

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
        
        response = self._chat_with_image(prompt, image_path)
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
