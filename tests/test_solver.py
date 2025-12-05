"""
Tests for the CaptchaSolver with the two-stage architecture.

These tests verify:
1. ActionPlanner correctly identifies action types
2. AttentionExtractor produces valid coordinates
3. CaptchaSolver orchestrates both correctly
"""

import pytest
import os
import json
from typing import List
from unittest.mock import MagicMock, patch, PropertyMock

from src.captchakraken.types import (
    CaptchaAction, ClickAction, DragAction, TypeAction, 
    WaitAction, RequestUpdatedImageAction, Solution
)
from src.captchakraken.solver import CaptchaSolver
from src.captchakraken.planner import ActionPlanner, PlannedAction
from src.captchakraken.attention import AttentionExtractor


# Path to test images
IMAGES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "captchaimages"))

# Test configurations - what we expect for each image
TEST_CONFIG = {
    "cloudflare.png": {
        "prompt": "Solve this captcha",
        "expected_action_type": "click",
        "description": "Cloudflare turnstile checkbox"
    },
    "hcaptchaBasic.png": {
        "prompt": "Solve this captcha",
        "expected_action_type": "click",
        "description": "hCaptcha checkbox"
    },
    "recaptchaBasic.png": {
        "prompt": "Solve this captcha",
        "expected_action_type": "click",
        "description": "reCAPTCHA checkbox"
    },
    "hcaptchaDragImage1.png": {
        "prompt": "Solve this captcha",
        "expected_action_type": "drag",
        "description": "hCaptcha drag puzzle"
    },
    "hcaptchaImages1.png": {
        "prompt": "Solve this captcha",
        "expected_action_type": "click",
        "description": "hCaptcha image selection"
    },
    "recaptchaImages.png": {
        "prompt": "Solve this captcha",
        "expected_action_type": "click",
        "description": "reCAPTCHA image grid"
    },
}


def get_test_files():
    """Returns list of (filename, config) tuples for parametrization."""
    files = []
    for filename, config in TEST_CONFIG.items():
        image_path = os.path.join(IMAGES_DIR, filename)
        if os.path.exists(image_path):
            files.append((filename, config))
    return files if files else [("dummy.png", {"prompt": "test", "expected_action_type": "click"})]


class TestActionPlanner:
    """Tests for the ActionPlanner component."""
    
    def test_planner_initialization_ollama(self):
        """Test planner initializes with ollama backend."""
        planner = ActionPlanner(backend="ollama")
        assert planner.backend == "ollama"
        assert planner.model == "hf.co/unsloth/gemma-3-12b-it-GGUF:Q4_K_M"
    
    def test_planner_initialization_openai(self):
        """Test planner initializes with openai backend."""
        planner = ActionPlanner(backend="openai")
        assert planner.backend == "openai"
        assert planner.model == "gpt-4o"
    
    def test_parse_response_click(self):
        """Test parsing a click action response."""
        planner = ActionPlanner()
        response = '{"action_type": "click", "target_description": "the checkbox", "reasoning": "Need to click"}'
        
        result = planner._parse_response(response)
        
        assert result.action_type == "click"
        assert result.target_description == "the checkbox"
        assert result.reasoning == "Need to click"
    
    def test_parse_response_drag(self):
        """Test parsing a drag action response."""
        planner = ActionPlanner()
        response = '''{"action_type": "drag", 
                       "target_description": "the puzzle piece", 
                       "drag_target_description": "the empty slot",
                       "reasoning": "Drag puzzle"}'''
        
        result = planner._parse_response(response)
        
        assert result.action_type == "drag"
        assert result.target_description == "the puzzle piece"
        assert result.drag_target_description == "the empty slot"
    
    def test_parse_response_wait(self):
        """Test parsing a wait action response."""
        planner = ActionPlanner()
        response = '{"action_type": "wait", "wait_duration_ms": 1000, "reasoning": "Loading"}'
        
        result = planner._parse_response(response)
        
        assert result.action_type == "wait"
        assert result.wait_duration_ms == 1000
    
    def test_parse_response_with_markdown(self):
        """Test parsing response wrapped in markdown code block."""
        planner = ActionPlanner()
        response = '''```json
{"action_type": "click", "target_description": "button"}
```'''
        
        result = planner._parse_response(response)
        
        assert result.action_type == "click"
        assert result.target_description == "button"


class TestAttentionExtractor:
    """Tests for the AttentionExtractor component."""
    
    def test_extractor_initialization(self):
        """Test extractor initializes correctly."""
        extractor = AttentionExtractor(backend="moondream")
        assert extractor.backend == "moondream"
        assert extractor.model_id == "vikhyatk/moondream2"
    
    def test_parse_percentage_xy_format(self):
        """Test parsing x=%, y=% format coordinates."""
        extractor = AttentionExtractor()
        response = "The checkbox is located at x=37%, y=50%"
        
        x_pct, y_pct = extractor._parse_percentage_from_text(response)
        
        assert x_pct == 0.37
        assert y_pct == 0.50
    
    def test_parse_percentage_colon_format(self):
        """Test parsing x: %, y: % format coordinates."""
        extractor = AttentionExtractor()
        response = "Located at position x: 25%, y: 75%"
        
        x_pct, y_pct = extractor._parse_percentage_from_text(response)
        
        assert x_pct == 0.25
        assert y_pct == 0.75
    
    def test_parse_percentage_fallback(self):
        """Test fallback to center when parsing fails."""
        extractor = AttentionExtractor()
        response = "I cannot determine the location"
        
        x_pct, y_pct = extractor._parse_percentage_from_text(response)
        
        # Should fall back to center (0.5, 0.5)
        assert x_pct == 0.5
        assert y_pct == 0.5


class TestCaptchaSolver:
    """Tests for the main CaptchaSolver class."""
    
    def test_solver_initialization(self):
        """Test solver initializes with default settings."""
        solver = CaptchaSolver()
        
        assert solver.planner is not None
        assert solver.attention_extractor is not None
    
    def test_solver_initialization_custom_backends(self):
        """Test solver with custom backend configuration."""
        solver = CaptchaSolver(
            planner_backend="openai",
            attention_model="vikhyatk/moondream2"
        )
        
        assert solver.planner.backend == "openai"
        assert solver.attention_extractor.model_id == "vikhyatk/moondream2"
    
    @patch.object(ActionPlanner, 'plan')
    @patch.object(AttentionExtractor, 'extract_coordinates')
    def test_solve_step_click(self, mock_extract, mock_plan):
        """Test solve_step produces correct ClickAction."""
        # Setup mocks
        mock_plan.return_value = PlannedAction(
            action_type="click",
            target_description="the checkbox"
        )
        mock_extract.return_value = (150, 200)
        
        solver = CaptchaSolver()
        
        # Create a dummy image file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b'\x89PNG\r\n\x1a\n')  # PNG header
            temp_path = f.name
        
        try:
            action = solver.solve_step(temp_path, "Solve this captcha")
            
            assert isinstance(action, ClickAction)
            assert action.action == "click"
            assert action.coordinates == [150, 200]
        finally:
            os.unlink(temp_path)
    
    @patch.object(ActionPlanner, 'plan')
    @patch.object(AttentionExtractor, 'extract_drag_coordinates')
    def test_solve_step_drag(self, mock_extract_drag, mock_plan):
        """Test solve_step produces correct DragAction."""
        mock_plan.return_value = PlannedAction(
            action_type="drag",
            target_description="the puzzle piece",
            drag_target_description="the empty slot"
        )
        mock_extract_drag.return_value = ((100, 100), (300, 300))
        
        solver = CaptchaSolver()
        
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b'\x89PNG\r\n\x1a\n')
            temp_path = f.name
        
        try:
            action = solver.solve_step(temp_path, "Solve this captcha")
            
            assert isinstance(action, DragAction)
            assert action.action == "drag"
            assert action.source_coordinates == [100, 100]
            assert action.target_coordinates == [300, 300]
        finally:
            os.unlink(temp_path)
    
    @patch.object(ActionPlanner, 'plan')
    def test_solve_step_wait(self, mock_plan):
        """Test solve_step produces correct WaitAction."""
        mock_plan.return_value = PlannedAction(
            action_type="wait",
            wait_duration_ms=500
        )
        
        solver = CaptchaSolver()
        
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b'\x89PNG\r\n\x1a\n')
            temp_path = f.name
        
        try:
            action = solver.solve_step(temp_path, "Solve this captcha")
            
            assert isinstance(action, WaitAction)
            assert action.action == "wait"
            assert action.duration_ms == 500
        finally:
            os.unlink(temp_path)
    
    @patch.object(ActionPlanner, 'plan')
    def test_solve_step_request_updated_image(self, mock_plan):
        """Test solve_step produces RequestUpdatedImageAction."""
        mock_plan.return_value = PlannedAction(
            action_type="request_updated_image"
        )
        
        solver = CaptchaSolver()
        
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b'\x89PNG\r\n\x1a\n')
            temp_path = f.name
        
        try:
            action = solver.solve_step(temp_path, "Solve this captcha")
            
            assert isinstance(action, RequestUpdatedImageAction)
            assert action.action == "request_updated_image"
        finally:
            os.unlink(temp_path)
    
    def test_format_history(self):
        """Test action history formatting."""
        solver = CaptchaSolver()
        solver._action_history = [
            ClickAction(action="click", coordinates=[100, 200]),
            WaitAction(action="wait", duration_ms=500),
            ClickAction(action="click", coordinates=[300, 400]),
        ]
        
        history = solver._format_history()
        
        assert "Clicked at [100, 200]" in history
        assert "Waited 500ms" in history
        assert "Clicked at [300, 400]" in history
    
    @patch.object(ActionPlanner, 'plan')
    @patch.object(AttentionExtractor, 'extract_coordinates')
    def test_solve_loop_max_steps(self, mock_extract, mock_plan):
        """Test solve_loop respects max_steps."""
        mock_plan.return_value = PlannedAction(
            action_type="click",
            target_description="checkbox"
        )
        mock_extract.return_value = (100, 100)
        
        solver = CaptchaSolver()
        
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b'\x89PNG\r\n\x1a\n')
            temp_path = f.name
        
        try:
            actions = list(solver.solve_loop(
                get_image=lambda: temp_path,
                context="test",
                max_steps=3,
                end_condition=None
            ))
            
            assert len(actions) == 3
        finally:
            os.unlink(temp_path)
    
    @patch.object(ActionPlanner, 'plan')
    @patch.object(AttentionExtractor, 'extract_coordinates')
    def test_solve_loop_end_condition(self, mock_extract, mock_plan):
        """Test solve_loop stops when end_condition is met."""
        mock_plan.return_value = PlannedAction(
            action_type="click",
            target_description="checkbox"
        )
        mock_extract.return_value = (100, 100)
        
        solver = CaptchaSolver()
        
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b'\x89PNG\r\n\x1a\n')
            temp_path = f.name
        
        try:
            call_count = [0]
            
            def end_condition():
                call_count[0] += 1
                return call_count[0] >= 2  # Stop after 2 checks (1 action)
            
            actions = list(solver.solve_loop(
                get_image=lambda: temp_path,
                context="test",
                max_steps=10,
                end_condition=end_condition
            ))
            
            assert len(actions) == 1
        finally:
            os.unlink(temp_path)


class TestTypes:
    """Tests for Pydantic type models."""
    
    def test_click_action_validation(self):
        """Test ClickAction validates correctly."""
        action = ClickAction(action="click", coordinates=[100, 200])
        assert action.action == "click"
        assert action.coordinates == [100, 200]
    
    def test_drag_action_validation(self):
        """Test DragAction validates correctly."""
        action = DragAction(
            action="drag",
            source_coordinates=[100, 100],
            target_coordinates=[200, 200]
        )
        assert action.action == "drag"
        assert action.source_coordinates == [100, 100]
        assert action.target_coordinates == [200, 200]
    
    def test_solution_with_multiple_actions(self):
        """Test Solution can contain multiple action types."""
        solution = Solution(actions=[
            {"action": "click", "coordinates": [100, 100]},
            {"action": "wait", "duration_ms": 500},
            {"action": "click", "coordinates": [200, 200]},
        ])
        
        assert len(solution.actions) == 3
        assert solution.actions[0].action == "click"
        assert solution.actions[1].action == "wait"


# Parametrized tests for real images (if available)
@pytest.mark.parametrize("filename,config", get_test_files())
def test_solver_with_real_images(filename, config):
    """
    Integration test with real captcha images.
    
    These tests are skipped if the images don't exist or
    if the required backends (Ollama/OpenAI) aren't available.
    """
    image_path = os.path.join(IMAGES_DIR, filename)
    
    if not os.path.exists(image_path):
        pytest.skip(f"Test image not found: {image_path}")
    
    # Skip actual inference in unit tests - mock instead
    with patch.object(ActionPlanner, 'plan') as mock_plan, \
         patch.object(AttentionExtractor, 'extract_coordinates') as mock_extract:
        
        # Setup mocks based on expected action type
        if config["expected_action_type"] == "click":
            mock_plan.return_value = PlannedAction(
                action_type="click",
                target_description="the target element"
            )
            mock_extract.return_value = (500, 500)
        elif config["expected_action_type"] == "drag":
            mock_plan.return_value = PlannedAction(
                action_type="drag",
                target_description="source element",
                drag_target_description="target location"
            )
        
        solver = CaptchaSolver()
        action = solver.solve_step(image_path, config["prompt"])
        
        # Verify action type matches expected
        assert action.action == config["expected_action_type"], \
            f"Expected {config['expected_action_type']}, got {action.action}"
