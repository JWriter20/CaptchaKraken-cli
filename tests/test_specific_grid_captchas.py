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

@pytest.fixture(scope="module")
def solver():
    """
    Initialize solver. 
    Note: This requires GEMINI_API_KEY to be set in the environment.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY not set in environment. Skipping real API tests.", allow_module_level=True)
    
    return CaptchaSolver(provider="gemini", api_key=api_key)

def test_bus_captcha_solution(solver):
    """
    Test solving busImagesRecaptcha.png with REAL LLM call.
    Expected squares: 6, 7, 10, 11, 14, 15
    """
    filename = "busImagesRecaptcha.png"
    image_path = os.path.join(IMAGES_DIR, filename)
    if not os.path.exists(image_path):
        pytest.skip(f"Image not found: {image_path}")
    
    # Define core (obvious) and optional (edge/partial) squares
    core_selection = {6, 10, 11, 14}
    optional_selection = {7, 15}
    expected_full = core_selection | optional_selection
    
    print(f"\nTesting {filename}...")
    instruction = "Select all squares with buses"
    
    actions = solver.solve(image_path, instruction)
    
    # Extract selected numbers
    selected = set()
    if isinstance(actions, list):
        for action in actions:
            if isinstance(action, ClickAction) and action.target_number is not None:
                selected.add(action.target_number)
    
    selected_list = sorted(list(selected))
    print(f"Selected: {selected_list}")
    
    # Check that we found all core items
    missing_core = core_selection - selected
    assert not missing_core, f"Missed core squares: {missing_core}"
    
    # Check that we didn't find anything unexpected (outside full expected set)
    unexpected = selected - expected_full
    assert not unexpected, f"Selected unexpected squares: {unexpected}"

def test_motorcycle_captcha_solution(solver):
    """
    Test solving motorcycleImagesRecaptcha.png with REAL LLM call.
    Expected squares: 2, 3, 4, 5, 6, 7, 9, 10, 11, 13, 14
    (4, 13, 14 are edge cases often missed)
    """
    filename = "motorcycleImagesRecaptcha.png"
    image_path = os.path.join(IMAGES_DIR, filename)
    if not os.path.exists(image_path):
        pytest.skip(f"Image not found: {image_path}")
    
    # Core: The clear motorcycle parts
    core_selection = {2, 3, 5, 6, 7, 9, 10, 11}
    # Optional: Very small edges/shadows that might be missed
    optional_selection = {4, 13, 14} 
    expected_full = core_selection | optional_selection

    print(f"\nTesting {filename}...")
    instruction = "Select all squares with motorcycles"
    
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
    assert not missing_core, f"Missed core squares: {missing_core}"
    
    # Check that we didn't find anything unexpected
    unexpected = selected - expected_full
    assert not unexpected, f"Selected unexpected squares: {unexpected}"
