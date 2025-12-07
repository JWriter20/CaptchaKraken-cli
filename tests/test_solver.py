"""
Tests for the CaptchaSolver with the intelligent two-stage architecture.

These tests verify:
1. ActionPlanner correctly identifies action types and handles numbered elements
2. AttentionExtractor produces valid coordinates and supports detection
3. CaptchaSolver orchestrates both correctly with intelligent refinement
"""

import pytest
import os
import json
from typing import List
from unittest.mock import MagicMock, patch, PropertyMock
from types import SimpleNamespace

from src.action_types import (
    CaptchaAction, ClickAction, DragAction, TypeAction, 
    WaitAction, RequestUpdatedImageAction, VerifyAction, Solution
)
from src.solver import CaptchaSolver
from src.planner import ActionPlanner, PlannedAction
from src.attention import AttentionExtractor


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
    
    def test_parse_response_click_with_element_ids(self):
        """Test parsing a click action with numbered element IDs."""
        planner = ActionPlanner()
        response = '''{"action_type": "click", 
                       "target_description": "images with traffic lights",
                       "target_element_ids": [2, 5, 7],
                       "object_class_to_detect": "traffic light",
                       "reasoning": "Elements 2, 5, 7 contain traffic lights"}'''
        
        result = planner._parse_response(response)
        
        assert result.action_type == "click"
        assert result.target_element_ids == [2, 5, 7]
        assert result.object_class_to_detect == "traffic light"
    
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
    
    def test_parse_response_drag_with_element_ids(self):
        """Test parsing a drag action with source/target element IDs."""
        planner = ActionPlanner()
        response = '''{"action_type": "drag", 
                       "target_description": "element 1",
                       "drag_target_description": "element 2",
                       "source_element_id": 1,
                       "target_element_id": 2,
                       "reasoning": "Drag element 1 to element 2"}'''
        
        result = planner._parse_response(response)
        
        assert result.action_type == "drag"
        assert result.source_element_id == 1
        assert result.target_element_id == 2
    
    def test_parse_response_wait(self):
        """Test parsing a wait action response."""
        planner = ActionPlanner()
        response = '{"action_type": "wait", "wait_duration_ms": 1000, "reasoning": "Loading"}'
        
        result = planner._parse_response(response)
        
        assert result.action_type == "wait"
        assert result.wait_duration_ms == 1000
    
    def test_parse_response_verify(self):
        """Test parsing a verify action response."""
        planner = ActionPlanner()
        response = '{"action_type": "verify", "target_description": "verify button", "reasoning": "Submit solution"}'
        
        result = planner._parse_response(response)
        
        assert result.action_type == "verify"
    
    def test_parse_response_with_markdown(self):
        """Test parsing response wrapped in markdown code block."""
        planner = ActionPlanner()
        response = '''```json
{"action_type": "click", "target_description": "button"}
```'''
        
        result = planner._parse_response(response)
        
        assert result.action_type == "click"
        assert result.target_description == "button"
    
    def test_build_context_with_elements(self):
        """Test context building includes element information."""
        planner = ActionPlanner()
        
        elements = [
            {"element_id": 1, "element_type": "image_tile"},
            {"element_id": 2, "element_type": "image_tile"},
            {"element_id": 3, "element_type": "verify_button"},
        ]
        
        context = planner._build_context(
            context="Test context",
            elements=elements,
            prompt_text="Select all images with birds"
        )
        
        assert "Select all images with birds" in context
        assert "Element 1: image_tile" in context
        assert "Element 3: verify_button" in context


class TestAttentionExtractor:
    """Tests for the AttentionExtractor component."""
    
    def test_extractor_initialization(self):
        """Test extractor initializes correctly."""
        extractor = AttentionExtractor(backend="moondream")
        assert extractor.backend == "moondream"
        # Default should use moondream2 on Hugging Face
        assert "moondream" in extractor.model_id.lower()
    
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
    
    def test_map_detections_to_elements(self):
        """Test mapping detections to numbered elements."""
        extractor = AttentionExtractor()
        
        # Detections in normalized format [x_min, y_min, x_max, y_max]
        detections = [
            {"label": "bird", "bbox": [0.1, 0.1, 0.3, 0.3]},
            {"label": "bird", "bbox": [0.4, 0.4, 0.6, 0.6]},
        ]
        
        # Elements in normalized format [x_min, y_min, x_max, y_max]
        elements = [
            {"element_id": 1, "bbox": [0.0, 0.0, 0.33, 0.33]},
            {"element_id": 2, "bbox": [0.33, 0.0, 0.66, 0.33]},
            {"element_id": 3, "bbox": [0.33, 0.33, 0.66, 0.66]},
        ]
        
        mapped = extractor.map_detections_to_elements(detections, elements, iou_threshold=0.1)
        
        # First detection should overlap with element 1
        assert 1 in mapped[0]['overlapping_element_ids']
        # Second detection should overlap with element 3
        assert 3 in mapped[1]['overlapping_element_ids']


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
            model="gpt-4o",
            provider="openai",
            api_key="test-key"
        )
        
        assert solver.planner.backend == "openai"
        assert solver.attention_extractor.model_id == "vikhyatk/moondream2"
    
    @patch.object(ActionPlanner, 'classify_captcha')
    @patch.object(AttentionExtractor, 'focus')
    def test_solve_step_checkbox(self, mock_focus, mock_classify):
        """Specialized checkbox flow returns click with percent + pixels."""
        mock_classify.return_value = {"captcha_kind": "checkbox"}
        mock_focus.return_value = (0.25, 0.75)
        
        import tempfile
        from PIL import Image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = Image.new('RGB', (100, 200), color='white')
            img.save(f.name)
            temp_path = f.name
        
        try:
            solver = CaptchaSolver()
            action = solver.solve_step(temp_path, "click the checkbox")
            
            assert isinstance(action, ClickAction)
            assert action.point_percent == [0.25, 0.75]
            assert action.coordinates == [25.0, 150.0]
        finally:
            os.unlink(temp_path)
    
    @patch.object(ActionPlanner, 'plan_detection_target')
    @patch.object(ActionPlanner, 'classify_captcha')
    @patch.object(AttentionExtractor, 'detect_objects')
    def test_solve_step_image_selection(self, mock_detect, mock_classify, mock_plan_target):
        """Split-image/images flow returns bounding boxes."""
        mock_classify.return_value = {"captcha_kind": "images"}
        mock_plan_target.return_value = {"target_to_detect": "traffic light"}
        mock_detect.return_value = [
            {"bbox": [0.1, 0.1, 0.3, 0.3]},
            {"bbox": [0.5, 0.5, 0.6, 0.6]},
        ]
        
        import tempfile
        from PIL import Image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = Image.new('RGB', (100, 100), color='white')
            img.save(f.name)
            temp_path = f.name
        
        try:
            solver = CaptchaSolver()
            action = solver.solve_step(temp_path, "Select all traffic lights")
            
            assert isinstance(action, ClickAction)
            assert len(action.bounding_boxes) == 2
            assert len(action.bounding_boxes_px) == 2
            # First bounding box should map to 10-30 pixels on 100x100
            assert action.bounding_boxes_px[0][:2] == [10.0, 10.0]
        finally:
            os.unlink(temp_path)
    
    @patch.object(ActionPlanner, 'plan_drag_strategy')
    @patch.object(ActionPlanner, 'classify_captcha')
    @patch.object(AttentionExtractor, 'detect_objects')
    def test_solve_step_drag_logical(self, mock_detect, mock_classify, mock_drag_plan):
        """Logical drag uses detection for start and end."""
        mock_classify.return_value = {"captcha_kind": "drag_puzzle", "drag_variant": "logical"}
        mock_drag_plan.return_value = {
            "drag_type": "logical",
            "draggable_prompt": "top movable deer head",
            "destination_prompt": "top left strawberry",
        }
        mock_detect.side_effect = [
            [{"bbox": [0.1, 0.1, 0.2, 0.2]}],  # draggable
            [{"bbox": [0.7, 0.7, 0.8, 0.8]}],  # target
        ]
        
        import tempfile
        from PIL import Image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = Image.new('RGB', (200, 200), color='white')
            img.save(f.name)
            temp_path = f.name
        
        try:
            solver = CaptchaSolver()
            action = solver.solve_step(temp_path, "drag the piece")
            
            assert isinstance(action, DragAction)
            assert action.source_coordinates_pct == [0.15000000000000002, 0.15000000000000002]
            assert action.target_coordinates_pct == [0.75, 0.75]
        finally:
            os.unlink(temp_path)
    
    @patch.object(ActionPlanner, 'refine_drag')
    @patch.object(ActionPlanner, 'get_drag_destination')
    @patch.object(ActionPlanner, 'plan_drag_strategy')
    @patch.object(ActionPlanner, 'classify_captcha')
    @patch.object(AttentionExtractor, 'detect_objects')
    def test_solve_step_drag_template(
        self,
        mock_detect,
        mock_classify,
        mock_drag_plan,
        mock_get_dest,
        mock_refine,
    ):
        """Template drag now uses iterative visual solver."""
        mock_classify.return_value = {"captcha_kind": "drag_puzzle", "drag_variant": "template_matching"}
        mock_drag_plan.return_value = {
            "drag_type": "template_matching",
            "draggable_prompt": "top segment movable square",
            "destination_prompt": "unused",
        }
        
        # 1. Detection of draggable
        mock_detect.return_value = [{"bbox": [0.0, 0.0, 0.2, 0.2]}] # Top-left 20%
        
        # 2. Initial proposal
        mock_get_dest.return_value = {
            "target_x": 0.5,
            "target_y": 0.5,
            "reasoning": "Move to center"
        }
        
        # 3. Refinement (accepted immediately)
        mock_refine.return_value = {
            "status": "correct",
            "adjustment": {},
            "reasoning": "Looks good"
        }
        
        import tempfile
        from PIL import Image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = Image.new('RGB', (100, 100), color='white')
            img.save(f.name)
            temp_path = f.name
        
        try:
            solver = CaptchaSolver()
            action = solver.solve_step(temp_path, "solve drag puzzle")
            
            assert isinstance(action, DragAction)
            # Source center of [0,0,0.2,0.2] is [0.1, 0.1] -> 10, 10
            assert action.source_coordinates == [10.0, 10.0]
            # Target is [0.5, 0.5] -> 50, 50
            assert action.target_coordinates == [50.0, 50.0]
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @patch.object(ActionPlanner, 'read_text')
    @patch.object(ActionPlanner, 'classify_captcha')
    def test_solve_step_text(self, mock_classify, mock_read_text):
        """Text captchas return TypeAction with extracted text."""
        mock_classify.return_value = {"captcha_kind": "text"}
        mock_read_text.return_value = "ABC123"
        
        import tempfile
        from PIL import Image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = Image.new('RGB', (80, 40), color='white')
            img.save(f.name)
            temp_path = f.name
        
        try:
            solver = CaptchaSolver()
            action = solver.solve_step(temp_path, "type the text")
            
            assert isinstance(action, TypeAction)
            assert action.text == "ABC123"
        finally:
            os.unlink(temp_path)
    
    @patch.object(ActionPlanner, 'plan')
    @patch.object(ActionPlanner, 'classify_captcha')
    def test_solve_step_fallback_wait(self, mock_classify, mock_plan):
        """Unknown classification falls back to legacy planner."""
        mock_classify.return_value = {"captcha_kind": "unknown"}
        mock_plan.return_value = PlannedAction(action_type="wait", wait_duration_ms=250)
        
        import tempfile
        from PIL import Image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = Image.new('RGB', (50, 50), color='white')
            img.save(f.name)
            temp_path = f.name
        
        try:
            solver = CaptchaSolver()
            action = solver.solve_step(temp_path, "waiting")
            
            assert isinstance(action, WaitAction)
            assert action.duration_ms == 250
        finally:
            os.unlink(temp_path)
    
    def test_format_history(self):
        """Test action history formatting with bounding boxes."""
        solver = CaptchaSolver()
        solver._action_history = [
            ClickAction(action="click", bounding_boxes=[[0.1, 0.1, 0.2, 0.2]]),
            WaitAction(action="wait", duration_ms=500),
            ClickAction(action="click", coordinates=[300.0, 400.0]),
        ]
        
        history = solver._format_history()
        
        assert "Clicked 1 detected boxes" in history
        assert "Waited 500ms" in history
        assert "Clicked at [300.0, 400.0]" in history
    
    def test_format_history_with_element_ids(self):
        """Test action history formatting with element IDs."""
        solver = CaptchaSolver()
        solver._action_history = [
            ClickAction(action="click", target_ids=[1, 3, 5], all_coordinates=[[50, 50], [150, 50], [250, 50]]),
            VerifyAction(action="verify", target_id=10),
        ]
        
        history = solver._format_history()
        
        assert "Clicked elements [1, 3, 5]" in history
        assert "Clicked verify button" in history
    
    def test_normalize_elements_pixel_to_percentage(self):
        """Test element bbox normalization from pixels to percentages."""
        solver = CaptchaSolver()
        solver._current_image_size = (300, 300)
        
        elements = [
            {"element_id": 1, "bbox": [0, 0, 100, 100]},  # pixel format [x, y, w, h]
            {"element_id": 2, "bbox": [100, 0, 100, 100]},
        ]
        
        normalized = solver._normalize_elements(elements)
        
        # First element should be normalized to [0, 0, 0.333, 0.333]
        assert normalized[0]['bbox'][0] == 0.0
        assert abs(normalized[0]['bbox'][2] - 0.333) < 0.01
    
    def test_get_element_center(self):
        """Test getting element center from bbox."""
        solver = CaptchaSolver()
        solver._current_image_size = (300, 300)
        
        elements = [
            {"element_id": 1, "bbox": [0, 0, 100, 100]},  # Center should be (50, 50)
            {"element_id": 2, "bbox": [100, 100, 100, 100]},  # Center should be (150, 150)
        ]
        
        center1 = solver._get_element_center(1, elements)
        center2 = solver._get_element_center(2, elements)
        
        assert center1 == (50, 50)
        assert center2 == (150, 150)
    
    def test_normalize_elements_pixel_to_percentage(self):
        """Test element bbox normalization from pixels to percentages."""
        solver = CaptchaSolver()
        solver._current_image_size = (300, 300)
        
        elements = [
            {"element_id": 1, "bbox": [0, 0, 100, 100]},  # pixel format [x, y, w, h]
            {"element_id": 2, "bbox": [100, 0, 100, 100]},
        ]
        
        normalized = solver._normalize_elements(elements)
        
        # First element should be normalized to [0, 0, 0.333, 0.333]
        assert normalized[0]['bbox'][0] == 0.0
        assert abs(normalized[0]['bbox'][2] - 0.333) < 0.01
    
    def test_get_element_center(self):
        """Test getting element center from bbox."""
        solver = CaptchaSolver()
        solver._current_image_size = (300, 300)
        
        elements = [
            {"element_id": 1, "bbox": [0, 0, 100, 100]},  # Center should be (50, 50)
            {"element_id": 2, "bbox": [100, 100, 100, 100]},  # Center should be (150, 150)
        ]
        
        center1 = solver._get_element_center(1, elements)
        center2 = solver._get_element_center(2, elements)
        
        assert center1 == (50, 50)
        assert center2 == (150, 150)


class TestTypes:
    """Tests for Pydantic type models."""
    
    def test_click_action_validation(self):
        """Test ClickAction validates correctly."""
        action = ClickAction(action="click", coordinates=[100, 200])
        assert action.action == "click"
        assert action.coordinates == [100, 200]
    
    def test_click_action_with_element_ids(self):
        """Test ClickAction with multiple element IDs."""
        action = ClickAction(
            action="click",
            target_ids=[1, 3, 5],
            all_coordinates=[[50, 50], [150, 50], [250, 50]]
        )
        assert action.target_ids == [1, 3, 5]
        assert len(action.all_coordinates) == 3
    
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
    
    def test_drag_action_with_element_ids(self):
        """Test DragAction with element IDs."""
        action = DragAction(
            action="drag",
            source_id=1,
            target_id=2,
            source_coordinates=[50, 50],
            target_coordinates=[150, 150]
        )
        assert action.source_id == 1
        assert action.target_id == 2
    
    def test_verify_action(self):
        """Test VerifyAction validates correctly."""
        action = VerifyAction(action="verify", target_id=10)
        assert action.action == "verify"
        assert action.target_id == 10
    
    def test_solution_with_multiple_actions(self):
        """Test Solution can contain multiple action types."""
        solution = Solution(actions=[
            {"action": "click", "coordinates": [100, 100]},
            {"action": "wait", "duration_ms": 500},
            {"action": "click", "coordinates": [200, 200]},
            {"action": "verify", "target_id": 10},
        ])
        
        assert len(solution.actions) == 4
        assert solution.actions[0].action == "click"
        assert solution.actions[1].action == "wait"
        assert solution.actions[3].action == "verify"
    


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
    with patch.object(ActionPlanner, 'classify_captcha') as mock_classify, \
         patch.object(ActionPlanner, 'plan_detection_target') as mock_plan_target, \
         patch.object(ActionPlanner, 'plan_drag_strategy') as mock_drag_plan, \
         patch.object(AttentionExtractor, 'focus') as mock_focus, \
         patch.object(AttentionExtractor, 'detect_objects') as mock_detect:
        
        # Determine classification based on filename
        if "Drag" in filename:
            mock_classify.return_value = {"captcha_kind": "drag_puzzle", "drag_variant": "logical"}
            mock_drag_plan.return_value = {
                "drag_type": "logical",
                "draggable_prompt": "top movable piece",
                "destination_prompt": "target slot",
            }
            mock_detect.side_effect = [
                [{"bbox": [0.2, 0.2, 0.3, 0.3]}],
                [{"bbox": [0.7, 0.7, 0.8, 0.8]}],
            ]
        elif "images" in filename.lower() or "images" in config["description"].lower():
            mock_classify.return_value = {"captcha_kind": "images"}
            mock_plan_target.return_value = {"target_to_detect": "traffic light"}
            mock_detect.return_value = [{"bbox": [0.1, 0.1, 0.2, 0.2]}]
        else:
            mock_classify.return_value = {"captcha_kind": "checkbox"}
            mock_focus.return_value = (0.5, 0.5)
            mock_detect.return_value = []
        
        solver = CaptchaSolver()
        action = solver.solve_step(image_path, config["prompt"])
        
        # Verify action type matches expected
        assert action.action == config["expected_action_type"], \
            f"Expected {config['expected_action_type']}, got {action.action}"
