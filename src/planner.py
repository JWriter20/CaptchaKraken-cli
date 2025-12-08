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


def _debug_log(message: str) -> None:
    """Print debug message if debugging is enabled."""
    if DEBUG:
        print(f"[Planner DEBUG] {message}", file=sys.stderr)

try:
    import google.generativeai as genai
except ImportError:
    genai = None


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

You have two tools available:
1. detect(object_class) - Find all instances of an object (e.g., "traffic light", "bus", "bicycle")
   Use when: You need to find multiple instances of something

2. point(target) - Find a single element (e.g., "checkbox", "verify button", "slider handle")
   Use when: You need to click/interact with one specific element

You can also return a direct action if you know exactly what to do:
- click: Click on something (provide target_description for point() to find it)
- drag: Drag something (provide source_description and target_description)
- type: Type text (provide the text to type)
- wait: Wait for loading
- done: Captcha is already solved

Respond ONLY with JSON. Either request a tool:
{{
  "tool_call": {{
    "name": "detect" | "point",
    "args": {{"object_class": "..."}} or {{"target": "..."}}
  }}
}}

Or return an action:
{{
  "action_type": "click" | "drag" | "type" | "wait" | "done",
  "target_description": "what to click (for click)",
  "source_description": "what to drag (for drag)",
  "drag_target_description": "where to drag to (for drag)",
  "text": "text to type (for type)",
  "duration_ms": 500,
  "reasoning": "brief explanation"
}}"""


# For text captchas: read the text
TEXT_READ_PROMPT = """Read the distorted text in this captcha image.
Look carefully at each character, accounting for distortion, rotation, and noise.

Respond ONLY with JSON:
{{
  "text": "the text to type"
}}"""


# For drag refinement: iterative adjustment
DRAG_REFINE_PROMPT = """You are refining a drag action for a captcha puzzle.

{instruction}

The image shows:
- RED border: The source element (what we're dragging)
- RED arrow: The current proposed drag path
- DASHED border: The proposed destination

Current destination: ({target_x:.1%}, {target_y:.1%})

{history_text}

Evaluate this drag destination:
1. Is the destination correct? Will dropping here solve the puzzle?
2. If not, what RELATIVE adjustment is needed?

Provide adjustments as percentages of image size:
- dx: positive = move right, negative = move left (e.g., 0.05 = 5% right)
- dy: positive = move down, negative = move up (e.g., -0.02 = 2% up)

Make SMALL adjustments (typically 1-5%). We can refine iteratively.

Respond ONLY with JSON:
{{
  "conclusion": "brief assessment of current position (e.g., 'too far right, height looks good')",
  "decision": "accept" | "adjust",
  "dx": 0.0,
  "dy": 0.0
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
    ):
        self.backend = backend
        self.ollama_host = ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")

        # Configure Gemini if selected
        if self.backend == "gemini":
            if not genai:
                raise ImportError("google-generativeai required. Install: pip install google-generativeai")
            if self.gemini_api_key:
                genai.configure(api_key=self.gemini_api_key)

        # Default models per backend
        if model is None:
            if backend == "ollama":
                self.model = "llama3.2-vision"
            elif backend == "gemini":
                self.model = "gemini-2.5-flash-lite"
            else:
                raise ValueError(f"Unknown backend: {backend}")
        else:
            self.model = model

    # ------------------------------------------------------------------
    # Core helper: chat with image
    # ------------------------------------------------------------------
    def _chat_with_image(self, prompt: str, image_path: str) -> str:
        """Send a prompt + image to the LLM backend, get response."""
        _debug_log(f"Backend: {self.backend}, Model: {self.model}")
        _debug_log(f"Image path: {image_path}")
        _debug_log(f"=== PROMPT START ===\n{prompt}\n=== PROMPT END ===")
        
        if self.backend == "ollama":
            import ollama

            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt, "images": [image_path]}],
                options={"temperature": 0.0},
            )
            result = response["message"]["content"]
            _debug_log(f"=== RAW RESPONSE ===\n{result}\n=== END RESPONSE ===")
            return result

        if self.backend == "gemini":
            if not genai:
                raise ValueError("google-generativeai not installed")
            if self.gemini_api_key:
                genai.configure(api_key=self.gemini_api_key)

            model = genai.GenerativeModel(self.model)
            import PIL.Image

            img = PIL.Image.open(image_path)

            response = model.generate_content(
                [prompt, img],
                generation_config=genai.types.GenerationConfig(response_mime_type="application/json", temperature=0.0),
            )
            result = response.text
            _debug_log(f"=== RAW RESPONSE ===\n{result}\n=== END RESPONSE ===")
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
        
        _debug_log(f"get_grid_selection called")
        _debug_log(f"  context instruction: '{instruction}'")
        _debug_log(f"  grid: {rows}x{cols} = {total} cells")
        
        prompt = SELECT_GRID_PROMPT.format(rows=rows, cols=cols, total=total)
        response = self._chat_with_image(prompt, image_path)
        result = self._parse_json(response)
        
        selected = result.get("selected_numbers", [])
        
        # Log all the detailed reasoning from the structured response
        _debug_log(f"Model extracted instruction: '{result.get('instruction_text', '(not provided)')}'")
        _debug_log(f"Reference image shows: '{result.get('reference_image_shows', '(not provided)')}'")
        _debug_log(f"Looking for: '{result.get('looking_for', '(not provided)')}'")
        
        cell_contents = result.get("cell_contents", {})
        if cell_contents:
            _debug_log(f"Cell contents analysis:")
            for cell_num, content in sorted(cell_contents.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
                _debug_log(f"  Cell {cell_num}: {content}")
        
        _debug_log(f"Selected numbers: {selected}")
        _debug_log(f"Reasoning: {result.get('reasoning', '(not provided)')}")
        
        return selected

    # ------------------------------------------------------------------
    # Tool-aware planning
    # ------------------------------------------------------------------
    def plan_with_tools(self, image_path: str, instruction: str = "") -> Dict[str, Any]:
        """
        Plan the next action, potentially requesting a tool call.

        Args:
            image_path: Path to the captcha image
            instruction: Optional instruction text

        Returns:
            Dict with either:
            - "tool_call": {"name": "detect"|"point", "args": {...}}
            - "action_type": "click"|"drag"|"type"|"wait"|"done" + action details
        """
        prompt = PLAN_WITH_TOOLS_PROMPT.format(instruction=instruction or "Solve this captcha.")
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
    ) -> Dict[str, Any]:
        """
        Refine a drag destination with visual feedback.

        Args:
            image_path: Path to image with drag overlay (source box, arrow, target box)
            instruction: Original puzzle instruction
            current_target: Current target position [x, y] as percentages (0-1)
            history: List of previous refinements:
                [{"destination": [x, y], "conclusion": "...", "decision": "..."}]

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
            history_text = "\n".join(history_lines)
        else:
            history_text = "This is the first attempt."

        prompt = DRAG_REFINE_PROMPT.format(
            instruction=instruction or "Complete the drag puzzle.",
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
