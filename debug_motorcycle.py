import os
import sys
from src.solver import CaptchaSolver
from src.action_types import ClickAction

# Setup
solver = CaptchaSolver(provider="gemini", api_key=os.getenv("GEMINI_API_KEY"))
image_path = "captchaimages/motorcycleImagesRecaptcha.png"
instruction = "Select all squares with motorcycles"

print(f"Solving {image_path}...")
actions = solver.solve(image_path, instruction)

selected = []
if isinstance(actions, list):
    for action in actions:
        if isinstance(action, ClickAction) and action.target_number is not None:
            selected.append(action.target_number)
selected.sort()
print(f"Selected: {selected}")

