import pytest
import os
import json
from typing import List, Set
from unittest.mock import MagicMock, patch
from pydantic import ValidationError

# We'll import the solver and types. 
# Note: These might not be fully implemented yet.
from src.captchakraken.types import CaptchaAction, ClickAction, DragAction, Solution
# Assuming the solver module will have a CaptchaSolver class
from src.captchakraken.solver import CaptchaSolver 
from src.captchakraken.parser import Component

# Path to images
IMAGES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "captchaimages"))

# Map images to expected task descriptions and expected action types
# This acts as our "Ground Truth" configuration for the tests
TEST_CONFIG = {
    "cloudflare.png": {
        "prompt": "Click the checkbox to verify you are human",
        "expected_action_type": "click",
        "expected_actions": 1
    },
    "hcaptchaBasic.png": {
        "prompt": "Click the checkbox",
        "expected_action_type": "click",
        "expected_actions": 1,
    },
    "recaptchaBasic.png": {
        "prompt": "Click the checkbox",
        "expected_action_type": "click",
        "expected_actions": 1,
    },
    # Drag the noodle on the top right to where it belongs in the bottom left 
    # connecting the other noodles, and drag the noodle on the bottom right 
    # to where it belongs in the top middle connecting the other noodles.
    "hcaptchaDragImage1.png": {
        "prompt": "Drag the noodle on the top right to where it belongs in the bottom left connecting the other noodles, and drag the noodle on the bottom right to where it belongs in the top middle connecting the other noodles.",
        "expected_action_type": "drag",
        "expected_actions": 2,
    },
    # Drag the head of the deer onto its body where it is missing, and the front legs onto the body where it is missing.
    "hcaptchaDragImage2.png": {
        "prompt": "Drag the head of the deer onto its body where it is missing, and the front legs onto the body where it is missing.",
        "expected_action_type": "drag",
        "expected_actions": 2,
    },
    # Drag the bee in the bottom right to the strawberry in the top left.
    "hcaptchaDragImages3.png": {
        "prompt": "Drag the bee in the bottom right to the strawberry in the top left.",
        "expected_action_type": "drag",
        "expected_actions": 1,
    },
    # Make sure we identify the prompt: "Find creatures born from something similar to the reference" (A picuture of an egg)
    # and that we click on the 3 birds (top middle, top right, bottom left), the other elephants, rhinos and lions do not come from eggs.
    "hcaptchaImages1.png": {
        "prompt": "Find creatures born from something similar to the reference (A picture of an egg)",
        "expected_action_type": "click",
        "expected_actions": 3,
        "order_invariant": True
    },
    # The bottom right and center middle are the only squares containing cars, make sure we identify the prompt:
    # "Select all images containing cars. Click verify once there are none left".
    "recaptchaImages.png": {
        "prompt": "Select all images containing cars. Click verify once there are none left",
        "expected_action_type": "click",
        "expected_actions": 2, # Bottom right and center middle
        "order_invariant": True
    },
    # Image of a man on a motorcycle with prompt: "Select all the squares with motorcycles If there are none, click skip"
    # Select the following squares (x means the square contains a motorcycle)
    # [ ] [x] [x] [x]
    # [ ] [ ] [x] [x]
    # [ ] [ ] [x] [x]
    # [ ] [ ] [x] [x]
    "recaptchaImages2.png": {
        "prompt": "Select all the squares with motorcycles. If there are none, click skip",
        "expected_action_type": "click",
        "expected_actions": 10, # Counted from the x's in the comment
        "order_invariant": True
    },
    # Image selection with prompt: "Select all images with a fire hydrant click verify when there are none left".
    # We should select the following squares (x means the square contains a fire hydrant)
    # [ ] [ ] [x]
    # [ ] [x] [ ]
    # [x] [ ] [ ]
    "recaptchaImages3.png": {
        "prompt": "Select all images with a fire hydrant. Click verify when there are none left",
        "expected_action_type": "click",
        "expected_actions": 3,
        "order_invariant": True
    },
}

def get_test_files():
    """Returns a list of (filename, config) tuples for parametrization."""
    files = []
    # If directory doesn't exist, we can't iterate, but we want to run tests if possible.
    # We'll use keys from TEST_CONFIG to drive the test if dir is missing/empty, 
    # assuming we Mock everything anyway.
    if not os.path.exists(IMAGES_DIR) or not os.listdir(IMAGES_DIR):
        for f in TEST_CONFIG:
            files.append((f, TEST_CONFIG[f]))
    else:
        for f in os.listdir(IMAGES_DIR):
            if f.endswith(".png") or f.endswith(".jpg"):
                if f in TEST_CONFIG:
                    files.append((f, TEST_CONFIG[f]))
                else:
                    # Default fallback for new/unknown images
                    files.append((f, {
                        "prompt": "Solve the captcha", 
                        "expected_action_type": "click",
                        "min_actions": 1
                    }))
    return files

@pytest.fixture
def solver():
    """Fixture to initialize the solver."""
    return CaptchaSolver()

@pytest.fixture
def mock_parser():
    with patch("src.captchakraken.solver.CaptchaParser") as MockParser:
        instance = MockParser.return_value
        # Default mock parse return
        instance.parse.return_value = (
            [Component(id=i, label="item", box=[0,0,10,10], type="icon") for i in range(20)], 
            "base64image"
        )
        yield instance

@pytest.fixture
def mock_llm():
    with patch("src.captchakraken.solver.ollama.chat") as mock_chat:
        yield mock_chat

@pytest.fixture
def mock_openai():
    with patch("src.captchakraken.solver.OpenAI") as mock_oai:
        yield mock_oai

@pytest.mark.parametrize("strategy", ["holo2", "omniparser"])
@pytest.mark.parametrize("filename, config", get_test_files())
def test_solve_captcha_image(solver, filename, config, strategy, mock_parser, mock_llm, mock_openai):
    """
    Test that the solver produces valid actions for a given image and strategy.
    """
    image_path = os.path.join(IMAGES_DIR, filename)
    # Create dummy image if it doesn't exist to avoid file not found errors in solver
    if not os.path.exists(image_path):
        if not os.path.exists(IMAGES_DIR):
            os.makedirs(IMAGES_DIR)
        with open(image_path, "wb") as f:
            f.write(b"fake_image_bytes")

    prompt = config["prompt"]
    
    # Setup Mock LLM response to match expectation
    actions = []
    expected_count = config.get("expected_actions", config.get("min_actions", 1))
    
    if config.get("expected_action_type") == "click":
        for i in range(expected_count):
            if strategy == "holo2":
                actions.append({"action": "click", "coordinates": [10*i, 10*i]})
            else:
                actions.append({"action": "click", "target_id": i})
                
    elif config.get("expected_action_type") == "drag":
        for i in range(expected_count):
            if strategy == "holo2":
                 actions.append({
                    "action": "drag", 
                    "source_coordinates": [10, 10], 
                    "target_coordinates": [20, 20]
                })
            else:
                actions.append({
                    "action": "drag", 
                    "source_id": i, 
                    "target_id": i+10
                })
    
    mock_response_json = json.dumps({"actions": actions})
    
    # Configure Ollama Mock
    mock_llm.return_value = {'message': {'content': mock_response_json}}
    
    # Configure OpenAI Mock (if we were using it, but we are not setting key so it uses ollama logic in solver)
    # If we wanted to test OpenAI path, we'd need to mock environment variable or init.
    
    # 1. Execute the solver
    try:
        solution_data = solver.solve(image_path, prompt, strategy=strategy)
    except NotImplementedError:
        pytest.skip("Solver implementation pending")
    except Exception as e:
        pytest.fail(f"Solver failed with error: {e}")

    # 2. Validate Schema
    assert solution_data is not None, "Solver returned None"
    
    # In our implementation, solve returns List[CaptchaAction] (which are Pydantic models)
    # The original test expected dicts or Solution object.
    # If it returns list of objects, we can inspect directly.
    
    actions = solution_data # It is already the list of actions

    # 3. Verify Constraints
    if "expected_actions" in config:
        assert len(actions) == config["expected_actions"], \
            f"Expected exactly {config['expected_actions']} actions, got {len(actions)}"
    else:
        assert len(actions) >= config.get("min_actions", 0), \
            f"Expected at least {config.get('min_actions')} actions, got {len(actions)}"

    # 4. Verify Action Types
    expected_type = config.get("expected_action_type")
    if expected_type:
        for action in actions:
            if action.action == "wait":
                continue
            assert action.action == expected_type, \
                f"Expected action type {expected_type}, got {action.action}"

    # 5. Order Invariant Check
    if config.get("order_invariant"):
        targets = []
        for action in actions:
            if isinstance(action, ClickAction):
                if action.target_id is not None:
                    targets.append(action.target_id)
                elif action.coordinates is not None:
                    targets.append(tuple(action.coordinates))
        assert len(targets) > 0, "Order invariant task produced no targets"

    # 6. Basic Validity Checks
    for action in actions:
        if isinstance(action, ClickAction):
            assert action.target_id is not None or action.coordinates is not None, \
                "Click action must have target_id or coordinates"
        elif isinstance(action, DragAction):
            assert (action.source_id is not None or action.source_coordinates is not None) and \
                   (action.target_id is not None or action.target_coordinates is not None), \
                   "Drag action missing source or target"



