"""
ActionPlanner - Simplified deterministic captcha solver.

1. Classifies captcha type (checkbox, image-selection, drag-puzzle, text)
2. Executes deterministic workflow based on type
3. Uses only two tools: detect() and point()

No complex prompts or multi-step reasoning - just clean workflows.
"""

import os
import json
import base64
from typing import Optional, List, Literal, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    import google.generativeai as genai
except ImportError:
    genai = None


class CaptchaType(Enum):
    """Simple captcha type classification."""
    CHECKBOX = "checkbox"  # Simple "I'm not a robot" checkbox
    IMAGE_SELECTION = "image_selection"  # Select all images with X
    DRAG_PUZZLE = "drag_puzzle"  # Drag piece to location
    TEXT = "text"  # Type distorted text
    UNKNOWN = "unknown"


@dataclass
class PlannedAction:
    """Result from the ActionPlanner - simplified."""
    action_type: Literal["click", "drag", "type", "done"]
    
    # Click action
    click_targets: Optional[List[Tuple[float, float]]] = None  # [(x, y), ...] in pixels or percentages
    
    # Drag action
    drag_source: Optional[Tuple[float, float]] = None  # (x, y)
    drag_target: Optional[Tuple[float, float]] = None  # (x, y)
    
    # Type action
    text_to_type: Optional[str] = None
    
    # Debug info
    reasoning: Optional[str] = None
    captcha_type: Optional[str] = None


# Simple classification prompt
CLASSIFY_PROMPT = """Analyze this captcha and classify it into ONE of these types:

1. "checkbox" - Simple checkbox (e.g., "I'm not a robot")
2. "image_selection" - Select all images containing X (e.g., traffic lights, buses)
3. "drag_puzzle" - Drag a piece to complete the puzzle
4. "text" - Type the distorted text shown

Also extract any instruction text visible in the image.

Respond ONLY with JSON:
{
  "type": "checkbox" | "image_selection" | "drag_puzzle" | "text",
  "instruction": "the instruction text or prompt, if visible",
  "reasoning": "brief explanation"
}"""

# For image selection: what object to detect
DETECT_TARGET_PROMPT = """The instruction says: "{instruction}"

What single object class should we detect? Use 1-3 words only.

Examples:
- "Select all images with traffic lights" → "traffic light"
- "Click on all buses" → "bus"
- "Select motorcycles" → "motorcycle"

Respond ONLY with JSON:
{
  "target": "object class to detect"
}"""

# For drag puzzles: where to drag
DRAG_PROMPTS_PROMPT = """This is a drag puzzle. Create two prompts:
1. draggable_prompt: describe the piece to drag (e.g., "puzzle piece", "slider handle")
2. destination_prompt: describe where to drag it (e.g., "empty slot", "slider track end")

Instruction: {instruction}

Respond ONLY with JSON:
{
  "draggable_prompt": "what to drag",
  "destination_prompt": "where to drag to"
}"""

# For text captchas: read the text
TEXT_READ_PROMPT = """Read the distorted text in this captcha.

Respond ONLY with JSON:
{
  "text": "the text to type"
}"""


class ActionPlanner:
    """
    Simple deterministic captcha solver.
    
    Workflow:
    1. Classify captcha type
    2. Execute deterministic sequence based on type:
       - checkbox: point("checkbox center") → click
       - image_selection: detect(object) → click all matches
       - drag_puzzle: point(source) + point(dest) → drag
       - text: read_text() → type
    
    Supports backends: ollama, openai, gemini, deepseek
    """
    
    def __init__(
        self,
        backend: Literal["ollama", "openai", "gemini", "deepseek"] = "gemini",
        model: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        ollama_host: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
    ):
        self.backend = backend
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openai_base_url = openai_base_url or os.getenv("OPENAI_BASE_URL")
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
            elif backend == "openai":
                self.model = "gpt-4o"
            elif backend == "gemini":
                self.model = "gemini-2.0-flash-exp"
            elif backend == "deepseek":
                self.model = "deepseek-chat"
        else:
            self.model = model
    
    # ------------------------------------------------------------------
    # Core helper: chat with image
    # ------------------------------------------------------------------
    def _chat_with_image(self, prompt: str, image_path: str) -> str:
        """Send a prompt + image to the LLM backend, get JSON response."""
        if self.backend == "ollama":
            import ollama
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt, "images": [image_path]}],
                options={"temperature": 0.0}
            )
            return response["message"]["content"]
        
        if self.backend in ["openai", "deepseek"]:
            from openai import OpenAI
            
            # Encode image as base64
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            media_type = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
            
            base_url = self.openai_base_url
            if self.backend == "deepseek" and not base_url:
                base_url = "https://api.deepseek.com/v1"
            
            client = OpenAI(api_key=self.openai_api_key, base_url=base_url)
            response = client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_data}"}},
                    ],
                }],
                temperature=0.0,
                max_tokens=500,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content

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
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.0
                )
            )
            return response.text
            
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
    # Main entry point: classify and solve deterministically
    # ------------------------------------------------------------------
    def classify(self, image_path: str) -> Dict[str, Any]:
        """
        Classify the captcha type.
        
        Returns:
            {
                "type": "checkbox" | "image_selection" | "drag_puzzle" | "text",
                "instruction": "instruction text or None",
                "reasoning": "brief explanation"
            }
        """
        response = self._chat_with_image(CLASSIFY_PROMPT, image_path)
        result = self._parse_json(response)
        
        # Ensure type is valid
        if result.get("type") not in ["checkbox", "image_selection", "drag_puzzle", "text"]:
            result["type"] = "checkbox"  # Default fallback
        
        return result
    
    def get_detection_target(self, instruction: str, image_path: str) -> str:
        """
        For image_selection captchas, get the object class to detect.
        
        Example: "Select all traffic lights" → "traffic light"
        """
        prompt = DETECT_TARGET_PROMPT.format(instruction=instruction)
        response = self._chat_with_image(prompt, image_path)
        result = self._parse_json(response)
        return result.get("target", instruction)  # Fallback to instruction if parsing fails
    
    def get_drag_prompts(self, instruction: str, image_path: str) -> Dict[str, str]:
        """
        For drag_puzzle captchas, get prompts for point() tool.
        
        Returns:
            {
                "draggable_prompt": "what to drag",
                "destination_prompt": "where to drag to"
            }
        """
        prompt = DRAG_PROMPTS_PROMPT.format(instruction=instruction or "complete the puzzle")
        response = self._chat_with_image(prompt, image_path)
        result = self._parse_json(response)
        
        return {
            "draggable_prompt": result.get("draggable_prompt", "puzzle piece"),
            "destination_prompt": result.get("destination_prompt", "empty slot"),
        }
    
    def read_text(self, image_path: str) -> str:
        """
        For text captchas, read the distorted text.
        
        Returns:
            The text to type
        """
        response = self._chat_with_image(TEXT_READ_PROMPT, image_path)
        result = self._parse_json(response)
        return result.get("text", "")
