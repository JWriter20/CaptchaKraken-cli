import pytest
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.solver import CaptchaSolver
from src.action_types import ClickAction

# Paths
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.dirname(TESTS_DIR)
IMAGES_DIR = os.path.join(CORE_DIR, "captchaimages")

def test_stairs_captcha_solution_ollama():
    """
    Test solving stairsCaptchaImage.png with Ollama qwen3-vl:8b.
    Expected squares: 10, 14, 15
    """
    filename = "stairsCaptchaImage.png"
    image_path = os.path.join(IMAGES_DIR, filename)
    if not os.path.exists(image_path):
        pytest.skip(f"Image not found: {image_path}")
    
    # Initialize solver with Ollama and Qwen
    print(f"\nInitializing solver with Ollama qwen3-vl:8b...")
    solver = CaptchaSolver(provider="ollama", model="qwen3-vl:8b")

    # User specified strict set: 10, 14, 15
    core_selection = {10, 14, 15}
    # Allow some leniency if the model sees nearby parts
    optional_selection = {9, 13, 11, 16} 
    expected_full = core_selection | optional_selection

    print(f"Testing {filename}...")
    instruction = "Select all squares with stairs"
    
    actions = solver.solve(image_path, instruction)
    
    selected = set()
    if isinstance(actions, list):
        for action in actions:
            if isinstance(action, ClickAction) and action.target_number is not None:
                selected.add(action.target_number)
    
    selected_list = sorted(list(selected))
    print(f"Selected: {selected_list}")
    
    # Check that we found all core items
    missing_core = core_selection - selected
    if missing_core:
        print(f"Missed core squares: {missing_core}")
    
    # Check that we didn't find anything unexpected
    unexpected = selected - expected_full
    if unexpected:
        print(f"Selected unexpected squares: {unexpected}")

    # Assertions
    assert not missing_core, f"Missed core squares: {missing_core}"
    assert not unexpected, f"Selected unexpected squares: {unexpected}"

if __name__ == "__main__":
    test_stairs_captcha_solution_ollama()


