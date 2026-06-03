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
        self.base_url = base_url or os.getenv("VLLM_BASE_URL", "http://13.57.41.42:8000/v1")
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

        # Surface auth / server errors with the actual body instead of letting
        # resp.json() blow up with a cryptic "Expecting value: line 1 column 1"
        # on a non-JSON response (e.g. a 401 {"error":"Unauthorized"} or an
        # HTML error page). 401/403 almost always means the bearer token
        # (CAPTCHA_KRAKEN_API_KEY) didn't reach this process.
        if not resp.ok:
            body = (resp.text or "")[:300]
            hint = ""
            if resp.status_code in (401, 403):
                hint = (
                    " — check CAPTCHA_KRAKEN_API_KEY is set and forwarded to the CLI"
                )
            raise RuntimeError(
                f"vLLM {resp.status_code} {resp.reason} at {url}{hint}. Body: {body}"
            )

        try:
            data = resp.json()
        except ValueError:
            body = (resp.text or "")[:300]
            raise RuntimeError(
                f"vLLM returned a non-JSON body from {url} (is the server up and "
                f"is VLLM_BASE_URL correct?). Body: {body}"
            )

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

    def get_grid_selection(
        self,
        image_path: str,
        rows: int,
        cols: int,
        retry_mode: Optional[str] = None,
    ) -> List[int]:
        """Return the list of 1-indexed cells the model wants to click.

        retry_mode == "missed-tiles": the previous submission was rejected
        by the captcha vendor with an under-selection error. Append an
        explicit recovery instruction that tells the model the FULL grid
        contains at least one matching tile it didn't pick last time. This
        nudges it off the "I already covered everything" attractor.
        """
        total = rows * cols
        if rows == 4 and cols == 4:
            grid_hint = "Hint: Single large image split into tiles. Select ALL parts."
        else:
            grid_hint = "Hint: Separate images. Select only clear matches."

        prompt = SELECT_GRID_PROMPT.format(
            rows=rows, cols=cols, total=total, grid_hint=grid_hint
        )
        if retry_mode == "missed-tiles":
            prompt = (
                prompt
                + "\n\nIMPORTANT: A previous submission was rejected because not all "
                  "matching tiles were selected. Re-examine EVERY cell in the grid "
                  "carefully. There is at least one more matching tile you missed. "
                  "Return the complete list of cell numbers that match the description, "
                  "including any matches you may have overlooked."
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
