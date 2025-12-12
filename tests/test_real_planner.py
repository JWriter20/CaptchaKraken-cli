"""
Tests for ActionPlanner with real LLM calls.

Tests:
1. Grid selection - verifies planner can select correct numbered cells
2. Tool-aware planning - verifies planner returns appropriate tool calls or actions
3. Drag refinement - verifies planner provides reasonable adjustments
"""

import pytest
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from src.planner import ActionPlanner
from src.grid_planner import GridPlanner
from src.overlay import add_overlays_to_image
from src.imagePreprocessing import get_grid_bounding_boxes

# Paths
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.dirname(TESTS_DIR)
IMAGES_DIR = os.path.join(CORE_DIR, "captchaimages")


@pytest.fixture(scope="module")
def planner():
    return ActionPlanner(backend="gemini")


class TestPlanWithTools:
    """Test plan_with_tools method."""
    
    # (filename, instruction, expected_result)
    TOOL_CASES = [
        # Checkbox - should return click action or point tool call
        ("cloudflare.png", "Solve this captcha", 
         {"expect_action": ["click"], "expect_tool": ["point"]}),
        
        ("hcaptchaBasic.png", "Solve this captcha",
         {"expect_action": ["click"], "expect_tool": ["point"]}),
        
        # Drag puzzles - should return drag action
        ("hcaptchaDragImages3.png", "Drag the bee to the strawberry",
         {"expect_action": ["drag"]}),
    ]
    
    @pytest.mark.parametrize("filename,instruction,expectations", TOOL_CASES)
    def test_plan_with_tools(self, planner, filename, instruction, expectations):
        """Verify planner returns appropriate tool calls or actions."""
        image_path = os.path.join(IMAGES_DIR, filename)
        if not os.path.exists(image_path):
            pytest.skip(f"Image not found: {image_path}")
        
        print(f"\nTesting plan_with_tools on {filename}...")
        
        result = planner.plan_with_tools(image_path, instruction)
        print(f"  Result: {result}")
        
        # Check if we got a tool call
        if result.get("tool_calls"):
            # Handle list of tool calls, take the first one for this test
            tool_call = result["tool_calls"][0]
            tool_name = tool_call["name"]
            expected_tools = expectations.get("expect_tool", [])
            if expected_tools:
                assert tool_name in expected_tools, \
                    f"Expected tool in {expected_tools}, got '{tool_name}'"
            print(f"  Tool call: {tool_name}")
        elif result.get("tool_call"):
            # Legacy support
            tool_name = result["tool_call"]["name"]
            expected_tools = expectations.get("expect_tool", [])
            if expected_tools:
                assert tool_name in expected_tools, \
                    f"Expected tool in {expected_tools}, got '{tool_name}'"
            print(f"  Tool call: {tool_name}")
        else:
            # Check if we got an action
            action_type = result.get("action_type")
            expected_actions = expectations.get("expect_action", [])
            if expected_actions:
                assert action_type in expected_actions, \
                    f"Expected action in {expected_actions}, got '{action_type}'"
            print(f"  Action: {action_type}")


class TestGridSelection:
    """Test grid selection on real grid captcha images."""
    
    def test_grid_selection_hcaptcha(self, planner, tmp_path):
        """Test grid selection on hCaptcha image grid."""
        image_path = os.path.join(IMAGES_DIR, "hcaptchaImages1.png")
        if not os.path.exists(image_path):
            pytest.skip(f"Image not found: {image_path}")
        
        # Check if it's actually a grid
        grid_boxes = get_grid_bounding_boxes(image_path)
        if not grid_boxes:
            pytest.skip("No grid detected in image")
        
        print(f"\nDetected grid with {len(grid_boxes)} cells")
        
        # Create overlay
        from PIL import Image
        with Image.open(image_path) as img:
            w, h = img.size
        
        overlays = []
        for i, (x1, y1, x2, y2) in enumerate(grid_boxes):
            bw = x2 - x1
            bh = y2 - y1
            overlays.append({
                'bbox': [x1, y1, bw, bh],
                'number': i + 1,
                'color': '#E74C3C'
            })
        
        overlay_path = str(tmp_path / "grid_overlay.png")
        add_overlays_to_image(image_path, overlays, output_path=overlay_path)
        
        # Test selection
        instruction = "Select all images containing birds"
        n = len(grid_boxes)
        rows = cols = int(n ** 0.5)
        
        grid_planner = GridPlanner(backend=planner.backend, gemini_api_key=planner.gemini_api_key)
        selected, _ = grid_planner.get_grid_selection(overlay_path, rows=rows, cols=cols, instruction=instruction)
        print(f"  Selected numbers: {selected}")
        
        # Basic validation
        assert isinstance(selected, list), "Should return a list"
        assert all(isinstance(n, int) for n in selected), "All selections should be integers"
        assert all(1 <= n <= len(grid_boxes) for n in selected), "Selections should be valid cell numbers"


class TestDragRefinement:
    """Test drag refinement functionality."""
    
    def test_drag_refine_provides_adjustments(self, planner, tmp_path):
        """Test that drag refinement returns valid adjustments."""
        from src.overlay import add_drag_overlay
        import shutil
        
        image_path = os.path.join(IMAGES_DIR, "hcaptchaDragImages3.png")
        if not os.path.exists(image_path):
            pytest.skip(f"Image not found: {image_path}")
        
        # Create work image with drag overlay
        work_path = str(tmp_path / "drag_work.png")
        shutil.copy(image_path, work_path)
        
        # Draw a sample drag overlay
        # Source: bottom-right area (where bee typically is)
        # Target: somewhere in middle (wrong, should need adjustment)
        source_bbox = [300, 350, 400, 450]  # Approximate
        target_bbox = [200, 200, 300, 300]
        target_center = (250, 250)
        
        add_drag_overlay(work_path, source_bbox, target_bbox=target_bbox, target_center=target_center)
        
        # Test refinement
        current_target = [0.5, 0.5]  # Middle of image
        history = []
        
        result = planner.refine_drag(
            work_path,
            "Drag the bee to the strawberry",
            current_target,
            history
        )
        
        print(f"\nDrag refinement result: {result}")
        
        # Validate structure
        assert "conclusion" in result, "Should have conclusion"
        assert "decision" in result, "Should have decision"
        assert result["decision"] in ["accept", "adjust"], "Decision should be accept or adjust"
        assert "dx" in result, "Should have dx adjustment"
        assert "dy" in result, "Should have dy adjustment"
        
        # Adjustments should be reasonable (within -0.5 to 0.5)
        assert -0.5 <= result["dx"] <= 0.5, f"dx out of range: {result['dx']}"
        assert -0.5 <= result["dy"] <= 0.5, f"dy out of range: {result['dy']}"
    
    def test_drag_refine_with_history(self, planner, tmp_path):
        """Test that drag refinement uses history context."""
        from src.overlay import add_drag_overlay
        import shutil
        
        image_path = os.path.join(IMAGES_DIR, "hcaptchaDragImages3.png")
        if not os.path.exists(image_path):
            pytest.skip(f"Image not found: {image_path}")
        
        work_path = str(tmp_path / "drag_work2.png")
        shutil.copy(image_path, work_path)
        
        source_bbox = [300, 350, 400, 450]
        target_bbox = [100, 100, 200, 200]
        target_center = (150, 150)
        
        add_drag_overlay(work_path, source_bbox, target_bbox=target_bbox, target_center=target_center)
        
        # Provide history of previous attempts
        history = [
            {"destination": [0.3, 0.3], "conclusion": "too far left", "decision": "adjust"},
            {"destination": [0.35, 0.3], "conclusion": "still too far left", "decision": "adjust"},
        ]
        
        current_target = [0.4, 0.3]
        
        result = planner.refine_drag(
            work_path,
            "Drag the bee to the strawberry",
            current_target,
            history
        )
        
        print(f"\nDrag refinement with history: {result}")
        
        # Just verify it returns valid structure
        assert "decision" in result
        assert "dx" in result
        assert "dy" in result


class TestTextReading:
    """Test text captcha reading."""
    
    def test_read_text_captcha(self, planner):
        """Test reading distorted text from a text captcha."""
        image_path = os.path.join(IMAGES_DIR, "textCaptcha.png")
        if not os.path.exists(image_path):
            pytest.skip(f"Image not found: {image_path}")
        
        text = planner.read_text(image_path)
        print(f"\nRead text: '{text}'")
        
        # Basic validation
        assert isinstance(text, str), "Should return a string"
        assert len(text) > 0, "Should read some text"
        # Text captchas usually have alphanumeric characters
        assert any(c.isalnum() for c in text), "Should contain alphanumeric characters"
