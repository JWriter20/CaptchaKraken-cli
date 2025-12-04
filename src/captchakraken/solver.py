import os
import sys
import json
import base64
from typing import List, Dict, Any, Optional
import ollama
from openai import OpenAI
from src.captchakraken.types import Solution, CaptchaAction
from src.captchakraken.parser import CaptchaParser

class CaptchaSolver:
    def __init__(self, openai_api_key: Optional[str] = None, openai_base_url: Optional[str] = None):
        self.openai_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openai_base_url = openai_base_url or os.getenv("OPENAI_BASE_URL")
        self.parser = None # Lazy load parser
        self.parsed_models = {}

    def _get_parser(self):
        if not self.parser:
            self.parser = CaptchaParser()
        return self.parser

    def _get_best_model(self, prefix: str) -> str:
        """Finds the best available model with the given prefix in Ollama."""
        if prefix in self.parsed_models:
            return self.parsed_models[prefix]
            
        try:
            models_info = ollama.list()
            # models_info['models'] is a list of dicts with 'name' or 'model'
            if 'models' in models_info:
                 model_list = models_info['models']
            else:
                 model_list = []
            
            available_models = []
            for m in model_list:
                # m might be a dict or object
                if isinstance(m, dict):
                    name = m.get('name') or m.get('model')
                else:
                    name = getattr(m, 'model', getattr(m, 'name', str(m)))
                if name:
                    available_models.append(name)
            
            # Find models starting with prefix
            matches = [m for m in available_models if prefix in m]
            
            if not matches:
                # Fallback to the prefix itself (e.g. might be pulled on demand)
                print(f"Warning: No installed model found for {prefix}, using exact name.")
                return prefix
                
            # Sort by size/version if possible? For now take the first or longest match
            # install_models.sh logic suggests we might have qwen3-vl:2b or qwen3-vl:8b
            # We prefer the one that exists.
            best = matches[0]
            self.parsed_models[prefix] = best
            print(f"Selected local model {best} for {prefix}")
            return best
        except Exception:
            # print(f"DEBUG: Error listing models: {e}", file=sys.stderr)
            return prefix

    def solve(self, image_path: str, prompt: str, strategy: str = "omniparser") -> List[CaptchaAction]:
        """
        Solves the captcha using the specified strategy.
        Returns a list of CaptchaAction objects.
        """
        if strategy == "omniparser":
            return self._solve_omniparser(image_path, prompt)
        elif strategy == "holo1.5" or strategy == "holo2":
            return self._solve_holo(image_path, prompt)
        elif strategy == "holo_mlx":
            return self._solve_holo_mlx(image_path, prompt)
        elif strategy == "holo_vllm":
            return self._solve_holo_vllm(image_path, prompt)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _solve_omniparser(self, image_path: str, task_prompt: str) -> List[CaptchaAction]:
        parser = self._get_parser()
        components, labeled_image_b64 = parser.parse(image_path)
        
        # Construct context for LLM
        component_descriptions = "\n".join([f"ID {c.id}: {c.label} ({c.type})" for c in components])
        
        system_prompt = """You are an expert AI agent that solves captchas and UI tasks.
You will be given a labeled image where UI elements are marked with numeric IDs.
You will also receive a list of these elements with their descriptions.
Your goal is to generate a sequence of actions to complete the user's task.

Output MUST be a valid JSON object with a single key "actions" containing a list of action objects.
Schema for actions:
- Click: {"action": "click", "target_id": <id>}
- Drag: {"action": "drag", "source_id": <id>, "target_id": <id>}
- Type: {"action": "type", "text": "<text>", "target_id": <id>}
- Wait: {"action": "wait", "duration_ms": <number>}
- RequestUpdatedImage: {"action": "request_updated_image"}

Example:
{
  "actions": [
    {"action": "click", "target_id": 5},
    {"action": "type", "text": "hello", "target_id": 5}
  ]
}
"""
        user_message = f"""Task: {task_prompt}

Detected Elements:
{component_descriptions}

Please provide the solution actions in JSON format.
"""
        
        # Use OpenAI if key exists, otherwise try local Qwen3 if available
        if self.openai_key:
            client = OpenAI(api_key=self.openai_key)
            response = client.chat.completions.create(
                model="gpt-4o", 
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_message},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{labeled_image_b64}"
                                }
                            }
                        ]
                    }
                ],
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
        else:
            # Check requirements before using local model
            from src.captchakraken.hardware import check_requirements
            ok, msg = check_requirements()
            if not ok:
                 raise RuntimeError(f"Local LLM requirements not met ({msg}). Please provide an OpenAI API key.")

            # Try Ollama with Qwen3-VL
            model_name = self._get_best_model('qwen3-vl')
            try:
                response = ollama.chat(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": user_message,
                            "images": [labeled_image_b64]
                        }
                    ],
                    format='json'
                )
                content = response['message']['content']
            except Exception as e:
                # If local model fails, and no openai key, we are stuck.
                # For testing purposes, we might want to return dummy actions if allowed, 
                # but better to fail so user knows setup is incomplete.
                raise RuntimeError(f"No LLM available. OpenAI key missing and Ollama failed: {e}")

        try:
            data = json.loads(content)
            return Solution(**data).actions
        except Exception as e:
            print(f"Failed to parse LLM response: {content}")
            raise e

    def _solve_holo_mlx(self, image_path: str, task_prompt: str) -> List[CaptchaAction]:
        # Check requirements before using local model
        from src.captchakraken.hardware import check_requirements
        ok, msg = check_requirements()
        if not ok:
             raise RuntimeError(f"Local LLM requirements not met ({msg}).")

        try:
            from mlx_vlm import load, generate
            from mlx_vlm.utils import load_image
        except ImportError:
            raise ImportError("mlx-vlm not installed. Please install with `pip install mlx-vlm`")
        
        # Holo2 model path
        model_path = "Hcompany/Holo2-8B"
        
        print(f"Loading {model_path} with mlx-vlm...")
        # trust_remote_code=True might be needed for Qwen-based models
        model, processor = load(model_path, trust_remote_code=True)
        
        system_prompt = """You are a UI automation agent.
Your task is to solve the captcha/UI task.
Output the solution as a JSON object with "actions".
For clicks, provide "coordinates": [x, y] where x and y are normalized (0-1000) or pixel values.
For drags, provide "source_coordinates": [x, y] and "target_coordinates": [x, y].
"""
        user_message = f"Task: {task_prompt}"
        
        formatted_prompt = processor.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message} # Image is handled separately in mlx-vlm generate usually?
                # Actually, mlx-vlm generate takes 'images' arg.
                # But the chat template might need to include the image token.
                # For Qwen2-VL (Holo2 base), usually it's automatic or we need <image> token.
            ], 
            tokenize=False, 
            add_generation_prompt=True
        )

        print(f"Generating solution for {image_path}...")
        output = generate(
            model, 
            processor, 
            image=image_path, # mlx-vlm generate handles image loading path or PIL
            prompt=formatted_prompt,
            verbose=True,
            max_tokens=512
        )
        
        print(f"Holo MLX output: {output}")
        
        try:
            # Holo output might be markdown wrapped json
            content = output.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            data = json.loads(content.strip())
            return Solution(**data).actions
        except Exception as e:
            print(f"Failed to parse Holo MLX response: {output}")
            raise e

    def _solve_holo_vllm(self, image_path: str, task_prompt: str) -> List[CaptchaAction]:
        """
        Solves using Holo2 hosted via VLLM (OpenAI compatible API).
        Requires VLLM running with: vllm serve Hcompany/Holo2-8B
        """
        # Read image
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        system_prompt = """You are a UI automation agent.
Your task is to solve the captcha/UI task.
Output the solution as a JSON object with "actions".
For clicks, provide "coordinates": [x, y] where x and y are normalized (0-1000) or pixel values.
For drags, provide "source_coordinates": [x, y] and "target_coordinates": [x, y].
"""
        
        # Configure client for VLLM
        base_url = self.openai_base_url or "http://localhost:8000/v1"
        api_key = self.openai_key or "EMPTY" # VLLM often uses EMPTY
        
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        # Determine model name (VLLM serves usually with the name passed to CLI)
        # We can try to list models or assume Hcompany/Holo2-8B
        try:
            models = client.models.list()
            model_name = models.data[0].id
        except:
            model_name = "Hcompany/Holo2-8B"

        print(f"Using VLLM model: {model_name} at {base_url}")

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Task: {task_prompt}"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}"
                                }
                            }
                        ]
                    }
                ],
                response_format={"type": "json_object"} # VLLM supports json mode if configured? Or we parse text.
                # Holo2 fine-tuned might not strictly follow json_object constraint if vllm backend doesn't enforce it easily without grammar.
                # Let's try without response_format if it fails, but for now include it as desired output.
            )
            content = response.choices[0].message.content
            
            # Clean up markdown
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
                
            data = json.loads(content.strip())
            return Solution(**data).actions
        except Exception as e:
            raise RuntimeError(f"Holo VLLM strategy failed: {e}")

    def _solve_holo(self, image_path: str, task_prompt: str) -> List[CaptchaAction]:
        # Check requirements before using local model
        from src.captchakraken.hardware import check_requirements
        ok, msg = check_requirements()
        if not ok:
             raise RuntimeError(f"Local LLM requirements not met ({msg}).")

        # Holo uses coordinate based interaction.
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
            
        system_prompt = """You are a UI automation agent.
Your task is to solve the captcha/UI task.
Output the solution as a JSON object with "actions".
For clicks, provide "coordinates": [x, y] where x and y are normalized (0-1000) or pixel values.
For drags, provide "source_coordinates": [x, y] and "target_coordinates": [x, y].
"""
        model_name = self._get_best_model('holo') # Matches holo2:8b etc
        
        try:
            response = ollama.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"Task: {task_prompt}",
                        "images": [image_b64]
                    }
                ],
                format='json'
            )
            content = response['message']['content']
            data = json.loads(content)
            return Solution(**data).actions
        except Exception as e:
             raise RuntimeError(f"Holo strategy failed: {e}")
