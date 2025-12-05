"""
ActionPlanner - Stage 1 of the captcha solving pipeline.

Queries an LLM to determine what action is needed to solve the captcha.
Outputs the action type and a description of the target for Stage 2.
"""

import os
import json
import base64
from typing import Optional, Tuple, Literal
from dataclasses import dataclass


@dataclass
class PlannedAction:
    """Result from the ActionPlanner."""
    action_type: Literal["click", "drag", "type", "wait", "request_updated_image", "done"]
    target_description: Optional[str] = None  # e.g., "the traffic lights", "the submit button"
    text_to_type: Optional[str] = None  # For type actions
    wait_duration_ms: Optional[int] = None  # For wait actions
    drag_target_description: Optional[str] = None  # For drag actions (where to drag to)
    reasoning: Optional[str] = None  # Model's reasoning (for debugging)


PLANNING_PROMPT = """You are a captcha-solving assistant. Analyze the captcha image and determine the SINGLE NEXT ACTION needed to solve it.

Your response must be a JSON object with these fields:
- "action_type": one of "click", "drag", "type", "wait", "request_updated_image", "done"
- "target_description": (for click/drag) describe WHAT to click/drag, e.g., "the traffic light in the top-left grid cell"
- "drag_target_description": (for drag only) describe WHERE to drag to
- "text_to_type": (for type only) the text to enter
- "wait_duration_ms": (for wait only) milliseconds to wait
- "reasoning": brief explanation of your decision

Action type guidelines:
- "click": Click on a specific element (images, checkboxes, buttons)
- "drag": Drag an element from source to target (puzzle pieces, sliders)
- "type": Type text (text captchas, input fields)
- "wait": Wait for animation/transition before next action
- "request_updated_image": Need a fresh screenshot (image is fading/changing)
- "done": Captcha appears to be already solved or no action needed

Be SPECIFIC in target_description. Don't say "click the correct image" - say exactly WHICH image based on what you see.

Example responses:
{"action_type": "click", "target_description": "the bicycle in the second row, first column", "reasoning": "The captcha asks to select all bicycles, and I see one in that cell"}
{"action_type": "drag", "target_description": "the puzzle piece on the left", "drag_target_description": "the matching gap on the right side", "reasoning": "This is a puzzle captcha requiring drag and drop"}
{"action_type": "wait", "wait_duration_ms": 500, "reasoning": "The images are still loading/fading in"}
{"action_type": "done", "reasoning": "The captcha shows a green checkmark indicating it's solved"}

Respond with ONLY the JSON object, no other text."""


class ActionPlanner:
    """
    Queries an LLM to determine the next action for solving a captcha.
    
    Supports multiple backends:
    - ollama: Local Ollama server with multimodal models (llava, etc.)
    - openai: OpenAI API (GPT-4V, etc.)
    """
    
    def __init__(
        self,
        backend: Literal["ollama", "openai"] = "ollama",
        model: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        ollama_host: Optional[str] = None,
    ):
        self.backend = backend
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openai_base_url = openai_base_url or os.getenv("OPENAI_BASE_URL")
        self.ollama_host = ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        
        # Default models per backend
        if model is None:
            self.model = "hf.co/unsloth/gemma-3-12b-it-GGUF:Q4_K_M" if backend == "ollama" else "gpt-4o"
        else:
            self.model = model
    
    def plan(self, image_path: str, context: str = "") -> PlannedAction:
        """
        Analyze the captcha image and return the next action to take.
        
        Args:
            image_path: Path to the captcha image
            context: Additional context (e.g., "This is an hCaptcha", previous actions taken)
        
        Returns:
            PlannedAction with the action type and target description
        """
        if self.backend == "ollama":
            return self._plan_ollama(image_path, context)
        elif self.backend == "openai":
            return self._plan_openai(image_path, context)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _plan_ollama(self, image_path: str, context: str) -> PlannedAction:
        """Plan using Ollama with a multimodal model."""
        import ollama
        
        prompt = PLANNING_PROMPT
        if context:
            prompt += f"\n\nAdditional context: {context}"
        
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
        if context:
            prompt += f"\n\nAdditional context: {context}"
        
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
            max_tokens=500,
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
            reasoning=data.get("reasoning")
        )

