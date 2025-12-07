"""
ActionPlanner - Stage 1 of the captcha solving pipeline.

Queries an LLM to determine what action is needed to solve the captcha.
Outputs the action type and a description of the target for Stage 2.

Enhanced with:
- Understanding of numbered overlay elements
- Multi-step reasoning and self-questioning
- Support for multi-click operations
"""

import os
import json
import base64
from typing import Optional, List, Literal, Dict, Any, Tuple
from dataclasses import dataclass, field


@dataclass
class PlannedAction:
    """Result from the ActionPlanner."""
    action_type: Literal["click", "drag", "type", "wait", "request_updated_image", "done", "verify"]
    target_description: Optional[str] = None  # e.g., "the traffic lights", "the submit button"
    text_to_type: Optional[str] = None  # For type actions
    wait_duration_ms: Optional[int] = None  # For wait actions
    drag_target_description: Optional[str] = None  # For drag actions (where to drag to)
    reasoning: Optional[str] = None  # Model's reasoning (for debugging)
    
    # Enhanced fields for intelligent solving
    target_element_ids: Optional[List[int]] = None  # For numbered element references (e.g., [2, 5, 9])
    source_element_id: Optional[int] = None  # For drag: source element ID
    target_element_id: Optional[int] = None  # For drag: target element ID
    object_class_to_detect: Optional[str] = None  # For detection (e.g., "bird", "traffic light")
    confidence: Optional[float] = None  # Confidence in the decision


# Intelligent planning prompt that understands numbered elements
PLANNING_PROMPT = """You are an expert captcha-solving AI assistant. Analyze the captcha image and determine the SINGLE NEXT ACTION needed to solve it.

IMPORTANT: The image may have NUMBERED OVERLAY BOXES on interactable elements (labeled 1, 2, 3, etc.). When you see these numbers, reference them in your response.

Your response must be a JSON object with these fields:
- "action_type": one of "click", "drag", "type", "wait", "request_updated_image", "verify", "done"
- "target_description": (for click/drag) describe WHAT to click/drag
- "target_element_ids": (optional) array of element IDs if numbered overlays are visible, e.g., [2, 5, 9]
- "source_element_id": (for drag) the element ID to drag FROM
- "target_element_id": (for drag) the element ID to drag TO
- "drag_target_description": (for drag only) describe WHERE to drag to
- "object_class_to_detect": (optional) if clicking all instances of something (e.g., "bird", "traffic light")
- "text_to_type": (for type only) the text to enter
- "wait_duration_ms": (for wait only) milliseconds to wait
- "reasoning": step-by-step explanation of your decision

Action type guidelines:
- "click": Click on one or more specific elements
  - If you see numbered overlays, specify which numbers to click via target_element_ids
  - If detecting objects (like "all images with birds"), set object_class_to_detect
- "drag": Drag an element from source to target
  - Use source_element_id and target_element_id for numbered elements
  - Or use target_description and drag_target_description for descriptions
- "type": Type text (text captchas, input fields)
- "wait": Wait for animation/loading before next action
- "verify": Click the verify/submit button to check solution
- "request_updated_image": Need a fresh screenshot
- "done": Captcha appears to be already solved

REASONING PROCESS:
1. First, describe what you see in the image
2. Identify the captcha type (reCAPTCHA, hCaptcha, slider, etc.)
3. Read any prompt/instruction text
4. Identify numbered elements if present
5. Determine the correct action
6. Specify exactly which elements to interact with

Example responses:

For image selection captcha with numbered overlays:
{
  "action_type": "click",
  "target_description": "images containing traffic lights",
  "target_element_ids": [2, 5, 7],
  "object_class_to_detect": "traffic light",
  "reasoning": "The prompt asks to select all traffic lights. I see numbered elements 1-9. Elements 2, 5, and 7 contain traffic lights."
}

For drag puzzle with numbered elements:
{
  "action_type": "drag",
  "target_description": "the puzzle piece",
  "drag_target_description": "the empty slot where the piece fits",
  "source_element_id": 1,
  "target_element_id": 2,
  "reasoning": "This is a jigsaw puzzle captcha. Element 1 is the draggable piece, element 2 is the destination slot."
}

For checkbox captcha:
{
  "action_type": "click",
  "target_description": "the checkbox next to 'I'm not a robot'",
  "reasoning": "This is a simple checkbox captcha. I need to click the unchecked checkbox to verify."
}

For verification button:
{
  "action_type": "verify",
  "target_description": "the verify button",
  "target_element_ids": [10],
  "reasoning": "I have selected all required images. Now I need to click the verify button."
}

Be SPECIFIC and reference element IDs when numbered overlays are visible.
Respond with ONLY the JSON object, no other text."""


THINKING_PROMPT = """Before deciding on the action, think step by step:

1. What type of captcha is this? (checkbox, image selection, slider, puzzle, etc.)
2. What is the captcha asking me to do? (Read any visible prompt text)
3. Are there numbered overlay boxes on the image? If so, what elements do they mark?
4. What is the current state? (Have any selections been made already?)
5. What is the SINGLE NEXT ACTION I should take?

After reasoning, provide your answer as a JSON object."""


# New specialized prompts based directly on AlgoImprovements.txt
CLASSIFICATION_PROMPT = """Classify this captcha image into one of the following types:
- "checkbox": a single checkbox such as the basic hCaptcha/reCAPTCHA check
- "split-image": a single coherent image split into many tiles (e.g., 16 squares) where multiple tiles must be selected
- "images": multiple separate images (often 9) where some should be selected based on a prompt
- "drag_puzzle": one or more draggable pieces that must be dragged to a correct location
- "text": distorted or warped text that must be typed

Also, if it is a drag puzzle, identify whether it is:
- "template_matching": drag a cut-out piece to its matching location in the same image
- "logical": drag based on semantic meaning or instructions rather than literal template matching

Respond ONLY with JSON:
{
  "captcha_kind": "checkbox" | "split-image" | "images" | "drag_puzzle" | "text",
  "drag_variant": "template_matching" | "logical" | "unknown",
  "reason": "short reason"
}"""

DETECTION_TARGET_PROMPT = """You will help pick the detection target for an image-selection captcha.
Prompt/instructions: "{prompt_text}"
Captcha kind: "{captcha_kind}"

Return ONLY JSON:
{{
  "target_to_detect": "short object phrase for detect() (e.g., traffic light, bus, motorcycle)",
  "reason": "very short note"
}}
Use 2-5 words max for target_to_detect, focused on the object class that should be highlighted."""

DRAG_STRATEGY_PROMPT = """Plan how to solve this drag puzzle. Follow AlgoImprovements.txt rules.

Decide whether this is a template-matching drag or a logical drag.
- "logical": Use this when you need to move a distinct object (like a bee, deer head) to a semantic target (strawberry, body).
- "template_matching": Use this only if dragging a puzzle piece into a matching hole/cutout in an image.

Then provide the prompts we should use:
- draggable_prompt MUST follow the pattern "{location} movable {item} {square_if_colored_tile}"
  Examples: "top segment movable square", "top movable deer head", "bottom movable deer legs", "bottom right movable bee"
- destination_prompt describes where to drop the item. Example: "top left strawberry", "top right deer body".

Respond ONLY with JSON:
{
  "drag_type": "template_matching" | "logical",
  "draggable_prompt": "string",
  "destination_prompt": "string",
  "notes": "brief reasoning"
}"""

TEXT_EXTRACTION_PROMPT = """Read the text in this captcha and return ONLY JSON:
{
  "text": "<the characters to type>"
}"""


DRAG_INITIAL_PROMPT = """You are solving a drag-and-drop puzzle.
1. The image shows a source piece highlighted in RED.
2. A green grid overlay marks 10% increments.

The goal is to move this piece to the correct location to solve the puzzle.
Instruction: {prompt_text}

Return the target center coordinates as percentages (0.0 to 1.0) of the image width/height.
Use the grid lines to estimate the coordinates precisely.

Respond ONLY with JSON:
{{
  "target_x": float, // 0.0-1.0
  "target_y": float, // 0.0-1.0
  "reasoning": "brief explanation"
}}
"""

DRAG_REFINEMENT_PROMPT = """You are refining a puzzle solution.
The image shows:
1. The draggable piece (Red Box)
2. A proposed move (Red Arrow) pointing to a dashed target box.
3. A green grid overlay where each cell is 10% of the image size.

Goal: Move the piece to {prompt_text}.

Look at the dashed target box. Is it perfectly aligned with the correct destination?
- If it is correct, return status "correct".
- If it is slightly off, use the grid to estimate precise adjustments (e.g., "move down 0.05", "move left 0.1").

Respond ONLY with JSON:
{{
  "status": "correct" | "needs_adjustment",
  "adjustment": {{
    "x_offset": float, // Negative = Move Left, Positive = Move Right (e.g., -0.05 is 5% left)
    "y_offset": float  // Negative = Move Up, Positive = Move Down (e.g., 0.1 is 10% down)
  }},
  "reasoning": "why the adjustment is needed, referencing grid lines if helpful"
}}
"""


class ActionPlanner:
    """
    Queries an LLM to determine the next action for solving a captcha.
    
    Supports multiple backends:
    - ollama: Local Ollama server with multimodal models (llava, gemma, etc.)
    - openai: OpenAI API (GPT-4V, etc.)
    
    Enhanced features:
    - Understands numbered overlay elements
    - Multi-step reasoning with self-questioning
    - Support for multi-click operations
    """
    
    def __init__(
        self,
        backend: Literal["ollama", "openai"] = "ollama",
        model: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        ollama_host: Optional[str] = None,
        thinking_enabled: bool = True,
    ):
        self.backend = backend
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openai_base_url = openai_base_url or os.getenv("OPENAI_BASE_URL")
        self.ollama_host = ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.thinking_enabled = thinking_enabled
        
        # Default models per backend
        if model is None:
            self.model = "hf.co/unsloth/gemma-3-12b-it-GGUF:Q4_K_M" if backend == "ollama" else "gpt-4o"
        else:
            self.model = model
    
    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def _encode_image_base64(self, image_path: str) -> Tuple[str, str]:
        """Encode an image for OpenAI-style requests."""
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        if image_path.lower().endswith(".png"):
            media_type = "image/png"
        elif image_path.lower().endswith((".jpg", ".jpeg")):
            media_type = "image/jpeg"
        else:
            media_type = "image/png"
        return image_data, media_type
    
    def _chat_with_image(self, prompt: str, image_path: str) -> str:
        """Send a single-turn multimodal prompt to the configured backend."""
        if self.backend == "ollama":
            import ollama
            
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt, "images": [image_path]}],
                options={"temperature": 0.1}
            )
            return response["message"]["content"]
        
        if self.backend == "openai":
            from openai import OpenAI
            
            client = OpenAI(
                api_key=self.openai_api_key,
                base_url=self.openai_base_url
            )
            image_data, media_type = self._encode_image_base64(image_path)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_data}"}},
                        ],
                    }
                ],
                temperature=0.1,
                max_tokens=600,
            )
            return response.choices[0].message.content
        
        raise ValueError(f"Unknown backend: {self.backend}")
    
    def _safe_json_loads(self, content: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Parse JSON content defensively, returning a default on failure."""
        default = default or {}
        if not content:
            return default
        
        text = content.strip()
        if "```json" in text:
            text = text.split("```json", 1)[1]
            text = text.split("```", 1)[0]
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0]
        
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            text = text[json_start:json_end]
        
        try:
            return json.loads(text)
        except Exception:
            return default
    
    # ------------------------------------------------------------------
    # Specialized planning helpers (AlgoImprovements-aligned)
    # ------------------------------------------------------------------
    def classify_captcha(
        self,
        image_path: str,
        prompt_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Classify captcha into the specialized categories."""
        prompt = CLASSIFICATION_PROMPT
        if prompt_text:
            prompt += f'\n\nInstruction text: "{prompt_text}"'
        
        raw = self._chat_with_image(prompt, image_path)
        return self._safe_json_loads(
            raw,
            {"captcha_kind": "unknown", "drag_variant": "unknown", "reason": "fallback"},
        )
    
    def plan_detection_target(
        self,
        image_path: str,
        prompt_text: Optional[str],
        captcha_kind: str = "images",
    ) -> Dict[str, Any]:
        """Choose the concise detection target for image-selection captchas."""
        prompt = DETECTION_TARGET_PROMPT.format(
            prompt_text=prompt_text or "",
            captcha_kind=captcha_kind,
        )
        raw = self._chat_with_image(prompt, image_path)
        parsed = self._safe_json_loads(
            raw,
            {"target_to_detect": prompt_text or "target object", "reason": "fallback"},
        )
        if not parsed.get("target_to_detect"):
            parsed["target_to_detect"] = prompt_text or "target object"
        return parsed
    
    def plan_drag_strategy(
        self,
        image_path: str,
        prompt_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Decide drag puzzle mode and generate prompts for draggable and destination."""
        prompt = DRAG_STRATEGY_PROMPT
        if prompt_text:
            prompt += f'\n\nPrompt/instructions: "{prompt_text}"'
        
        raw = self._chat_with_image(prompt, image_path)
        parsed = self._safe_json_loads(
            raw,
            {
                "drag_type": "logical",
                "draggable_prompt": "top movable piece",
                "destination_prompt": prompt_text or "correct location",
                "notes": "fallback",
            },
        )
        
        # Ensure mandatory fields are present
        if not parsed.get("draggable_prompt"):
            parsed["draggable_prompt"] = "top movable piece"
        if not parsed.get("destination_prompt"):
            parsed["destination_prompt"] = prompt_text or "correct location"
        if not parsed.get("drag_type"):
            parsed["drag_type"] = "logical"
        return parsed
    
    def read_text(self, image_path: str) -> str:
        """Extract text for text-based captchas."""
        raw = self._chat_with_image(TEXT_EXTRACTION_PROMPT, image_path)
        parsed = self._safe_json_loads(raw, {"text": ""})
        return parsed.get("text", "")
    
    def get_drag_destination(
        self,
        image_path: str,
        prompt_text: str,
        draggable_bbox_pct: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Ask where to move the highlighted piece using the iterative prompt."""
        prompt = DRAG_INITIAL_PROMPT.format(prompt_text=prompt_text or "")
        raw = self._chat_with_image(prompt, image_path)
        return self._safe_json_loads(
            raw,
            {"target_x": 0.5, "target_y": 0.5, "reasoning": "fallback"},
        )

    def refine_drag(self, image_path: str, prompt_text: str = "") -> Dict[str, Any]:
        """Ask for verification/adjustment of the proposed drag."""
        prompt = DRAG_REFINEMENT_PROMPT.format(prompt_text=prompt_text or "the solution")
        raw = self._chat_with_image(prompt, image_path)
        return self._safe_json_loads(
            raw,
            {"status": "needs_adjustment", "adjustment": {"x_offset": 0, "y_offset": 0}, "reasoning": "fallback"}
        )

    def plan(
        self,
        image_path: str,
        context: str = "",
        elements: Optional[List[dict]] = None,
        prompt_text: Optional[str] = None,
    ) -> PlannedAction:
        """
        Analyze the captcha image and return the next action to take.
        
        Args:
            image_path: Path to the captcha image
            context: Additional context (e.g., "This is an hCaptcha", previous actions taken)
            elements: List of numbered elements extracted from DOM (optional)
            prompt_text: The captcha's instruction text if available
        
        Returns:
            PlannedAction with the action type and target description
        """
        # Build enhanced context
        full_context = self._build_context(context, elements, prompt_text)
        
        if self.backend == "ollama":
            return self._plan_ollama(image_path, full_context)
        elif self.backend == "openai":
            return self._plan_openai(image_path, full_context)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _build_context(
        self,
        context: str,
        elements: Optional[List[dict]],
        prompt_text: Optional[str],
    ) -> str:
        """Build enhanced context with element information."""
        parts = []
        
        if prompt_text:
            parts.append(f"Captcha prompt: \"{prompt_text}\"")
        
        if elements:
            parts.append(f"\nNumbered elements visible in image ({len(elements)} total):")
            for elem in elements[:20]:  # Limit to first 20
                elem_id = elem.get('element_id', elem.get('id', '?'))
                elem_type = elem.get('element_type', 'unknown')
                parts.append(f"  - Element {elem_id}: {elem_type}")
            parts.append("\nUse these element IDs in your response when applicable.")
        
        if context:
            parts.append(f"\nAdditional context: {context}")
        
        return "\n".join(parts)
    
    def _plan_ollama(self, image_path: str, context: str) -> PlannedAction:
        """Plan using Ollama with a multimodal model."""
        import ollama
        
        prompt = PLANNING_PROMPT
        if self.thinking_enabled:
            prompt = THINKING_PROMPT + "\n\n" + prompt
        if context:
            prompt += f"\n\n{context}"
        
        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_path]
                }
            ],
            options={"temperature": 0.1}  # Low temperature for consistent outputs
        )
        
        return self._parse_response(response["message"]["content"])
    
    def _plan_openai(self, image_path: str, context: str) -> PlannedAction:
        """Plan using OpenAI API with vision model."""
        from openai import OpenAI
        
        client = OpenAI(
            api_key=self.openai_api_key,
            base_url=self.openai_base_url
        )
        
        # Encode image as base64
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        # Determine image type
        if image_path.lower().endswith(".png"):
            media_type = "image/png"
        elif image_path.lower().endswith((".jpg", ".jpeg")):
            media_type = "image/jpeg"
        else:
            media_type = "image/png"  # Default
        
        prompt = PLANNING_PROMPT
        if self.thinking_enabled:
            prompt = THINKING_PROMPT + "\n\n" + prompt
        if context:
            prompt += f"\n\n{context}"
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_data}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000,
            temperature=0.1
        )
        
        return self._parse_response(response.choices[0].message.content)
    
    def _parse_response(self, response_text: str) -> PlannedAction:
        """Parse the LLM response into a PlannedAction."""
        # Clean up response - extract JSON if wrapped in markdown
        content = response_text.strip()
        
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        # Find JSON object
        json_start = content.find("{")
        json_end = content.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            content = content[json_start:json_end]
        
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {response_text}") from e
        
        return PlannedAction(
            action_type=data.get("action_type", "click"),
            target_description=data.get("target_description"),
            text_to_type=data.get("text_to_type"),
            wait_duration_ms=data.get("wait_duration_ms"),
            drag_target_description=data.get("drag_target_description"),
            reasoning=data.get("reasoning"),
            target_element_ids=data.get("target_element_ids"),
            source_element_id=data.get("source_element_id"),
            target_element_id=data.get("target_element_id"),
            object_class_to_detect=data.get("object_class_to_detect"),
            confidence=data.get("confidence"),
        )
    
    def analyze_captcha_type(self, image_path: str) -> dict:
        """
        Analyze the captcha to determine its type and requirements.
        
        Returns:
            Dict with captcha_type, requires_selection, is_checkbox, etc.
        """
        import ollama
        
        prompt = """Analyze this captcha image and respond with a JSON object containing:
{
  "captcha_type": "recaptcha_v2" | "recaptcha_v3" | "hcaptcha" | "cloudflare" | "slider" | "puzzle" | "text" | "other",
  "is_checkbox": true/false,
  "is_image_selection": true/false,
  "is_drag_puzzle": true/false,
  "prompt_visible": true/false,
  "prompt_text": "the prompt text if visible",
  "num_selectable_elements": number or null,
  "current_state": "initial" | "selecting" | "completed" | "failed"
}

Respond with ONLY the JSON object."""
        
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt, "images": [image_path]}],
            options={"temperature": 0.1}
        )
        
        try:
            content = response["message"]["content"]
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start != -1:
                return json.loads(content[json_start:json_end])
        except Exception:
            pass
        
        return {"captcha_type": "unknown"}
    
    def question_self(self, image_path: str, question: str) -> str:
        """
        Ask a specific question about the captcha to refine understanding.
        
        Args:
            image_path: Path to the image
            question: Question to ask (e.g., "Are there any birds in element 3?")
        
        Returns:
            Answer to the question
        """
        import ollama
        
        prompt = f"""{question}

Provide a brief, direct answer. If asking about specific elements, reference them by their number."""
        
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt, "images": [image_path]}],
            options={"temperature": 0.1}
        )
        
        return response["message"]["content"]
