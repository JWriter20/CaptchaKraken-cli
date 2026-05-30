"""
ActionPlanner (v2) — talks to local vLLM serving the `captcha` LoRA over the
OpenAI-compatible /v1/chat/completions endpoint.

v1 supported transformers/vllm-local/gemini/openrouter and a whole tool-using
planner with detect/segment/drag-refine; that code is preserved on the
`v1-old-architecture` branch.
"""

import base64
import json
import os
import sys
from mimetypes import guess_type
from typing import Any, Dict, List, Optional

import requests

from .timing import timed

DEBUG = os.getenv("CAPTCHA_DEBUG", "0") == "1"


# Matches the training-distribution grid prompts in
# cleanSamples/test/test_solutions.json -> grade.synthesize_instruction.
# Drifting from these costs measurable accuracy.
SELECT_GRID_PROMPT = """Solve the captcha grid by choosing the cell numbers that match the description from the captcha image prompt.

Grid: {rows}x{cols} ({total} cells)
{grid_hint}

If no tiles match the description (e.g., they have all been cleared or none were present), return an empty list for target_ids: [].

Return JSON Array: [list of cell numbers (1-{total})]"""


# Mirrors _UNIVERSAL_ACTION_PROMPT in src/testing/grade.py — used for non-grid
# puzzles where the LoRA chooses between click points and drag actions itself.
UNIVERSAL_ACTION_PROMPT = (
    "Your task is to solve the captcha. Read the instruction at the top of the image carefully.\n\n"
    "Look at the puzzle and decide what action solves it. All coordinates you return must be on a normalized 0–1000 image scale (top-left = (0, 0), bottom-right = (1000, 1000)).\n\n"
    "Choose ONE response:\n\n"
    "FOR CLICK PUZZLES:\n"
    "  Identify every position you need to click and emit them as a list of points:\n"
    "  → \"action\": { \"action\": \"click\", \"points\": [[x1, y1], [x2, y2], ...] }\n\n"
    "FOR DRAG PUZZLES:\n"
    "  Drag ONE item at a time. The source position is the centroid of the piece you are picking up; the destination position is where it should end up. If multiple drags are needed, drag the topmost item first.\n"
    "  → \"output\": [{ \"Action\": \"simulate_drag\", \"SourceDescription\": \"...\", \"SourcePosition\": { \"x\": 1-1000, \"y\": 1-1000 }, \"DestinationDescription\": \"...\", \"EstimatedPosition\": { \"x\": 1-1000, \"y\": 1-1000 } }]\n\n"
    "Respond ONLY with JSON:\n"
    "{\n"
    "  \"action\": { ... }\n"
    "  // OR \"output\": [ ... ]\n"
    "}"
)


class ActionPlanner:
    """Thin client for the vLLM `captcha` LoRA."""

    def __init__(
        self,
        model: Optional[str] = None,
        debug_callback: Optional[Any] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **_: Any,
    ):
        self.debug_callback = debug_callback
        self.token_usage: List[Dict[str, Any]] = []

        # The LoRA name registered with vLLM (see /etc/systemd/system/vllm.service).
        # Override via CAPTCHA_LORA_NAME for future MoLoRA experts.
        self.model = model or os.getenv("CAPTCHA_LORA_NAME", "captcha")
        self.base_url = base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        self.api_key = (
            api_key
            or os.getenv("CAPTCHA_KRAKEN_API_KEY")
            or os.getenv("VLLM_API_KEY")
            or "EMPTY"
        )

    def _log(self, message: str) -> None:
        if DEBUG:
            print(f"[Planner] {message}", file=sys.stderr)
        if self.debug_callback:
            self.debug_callback(f"[Planner] {message}")

    def _chat_with_image(self, prompt: str, image_path: str, max_tokens: int = 512) -> str:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        mime, _ = guess_type(image_path)
        if mime is None:
            mime = "image/png"

        messages = [
            {
                "role": "system",
                "content": "You are an expert captcha solver. Respond ONLY with the JSON action.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            # Qwen3.5's reasoning otherwise eats the token budget. `/no_think`
            # in the prompt alone is unreliable; disabling at the chat-template
            # level is the documented way.
            "chat_template_kwargs": {"enable_thinking": False},
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        url = f"{self.base_url}/chat/completions"
        self._log(f"POST {url} model={self.model} max_tokens={max_tokens}")

        with timed("planner.chat"):
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        if data.get("usage"):
            self.token_usage.append(data["usage"])

        content = data["choices"][0]["message"].get("content") or ""
        self._log(f"Raw content: {content[:300]}")
        return content

    @staticmethod
    def _parse_json(text: str) -> Any:
        text = (text or "").strip()
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0]

        start_obj = text.find("{")
        start_list = text.find("[")
        if start_list != -1 and (start_obj == -1 or start_list < start_obj):
            end = text.rfind("]") + 1
            text = text[start_list:end]
        elif start_obj != -1:
            end = text.rfind("}") + 1
            text = text[start_obj:end]

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def get_grid_selection(self, image_path: str, rows: int, cols: int) -> List[int]:
        """Return the list of 1-indexed cells the model wants to click."""
        total = rows * cols
        if rows == 4 and cols == 4:
            grid_hint = "Hint: Single large image split into tiles. Select ALL parts."
        else:
            grid_hint = "Hint: Separate images. Select only clear matches."

        prompt = SELECT_GRID_PROMPT.format(
            rows=rows, cols=cols, total=total, grid_hint=grid_hint
        )
        raw = self._chat_with_image(prompt, image_path, max_tokens=128)
        data = self._parse_json(raw)

        if isinstance(data, list):
            ids = data
        elif isinstance(data, dict):
            ids = data.get("target_ids") or data.get("action", {}).get("target_ids") or []
        else:
            ids = []

        out: List[int] = []
        for v in ids:
            try:
                iv = int(v)
                if 1 <= iv <= total:
                    out.append(iv)
            except (TypeError, ValueError):
                continue
        self._log(f"grid selection -> {out}")
        return out

    def get_universal_action(self, image_path: str) -> Dict[str, Any]:
        """For non-grid puzzles. Returns the parsed JSON (caller maps it)."""
        raw = self._chat_with_image(UNIVERSAL_ACTION_PROMPT, image_path, max_tokens=512)
        data = self._parse_json(raw)
        return data if isinstance(data, dict) else {}
