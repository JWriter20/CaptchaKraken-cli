
import unittest
from unittest.mock import MagicMock, patch
from src.solver import CaptchaSolver
from src.action_types import ClickAction

class TestSolverGridFix(unittest.TestCase):
    def setUp(self):
        self.solver = CaptchaSolver(provider="gemini", api_key="test_key")
        self.solver.debug = MagicMock()
        self.solver.grid_planner = MagicMock()
        self.solver.image_processor = MagicMock()
        # Mock image size
        self.solver._image_size = (100, 100)
        self.solver._temp_files = []

    @patch('src.solver.add_overlays_to_image')
    @patch('src.solver.os.path.exists')
    @patch('src.solver.Image.open')
    def test_solve_grid_handles_string_numbers(self, mock_open, mock_exists, mock_add_overlays):
        mock_exists.return_value = True
        
        # Setup grid boxes
        # 3x3 grid
        grid_boxes = [(0,0,10,10) for _ in range(9)]
        
        # Mock planner response with STRINGS
        # This simulates the failure case: "selected_numbers": ["6"]
        self.solver.grid_planner.get_grid_selection.return_value = (["6"], False)
        
        # Mock image processor to return empty list for already selected cells
        self.solver.image_processor.detect_selected_cells.return_value = []

        actions = self.solver._solve_grid("test.png", "instruction", grid_boxes)
        
        self.assertTrue(len(actions) > 0)
        self.assertIsInstance(actions[0], ClickAction)
        self.assertEqual(actions[0].target_number, 6) # Pydantic converts "6" to 6
        
    @patch('src.solver.add_overlays_to_image')
    @patch('src.solver.os.path.exists')
    @patch('src.solver.Image.open')
    def test_solve_grid_handles_int_numbers(self, mock_open, mock_exists, mock_add_overlays):
        mock_exists.return_value = True
        grid_boxes = [(0,0,10,10) for _ in range(9)]
        
        # Mock planner response with INTEGERS (normal case)
        self.solver.grid_planner.get_grid_selection.return_value = ([6], False)
        self.solver.image_processor.detect_selected_cells.return_value = []

        actions = self.solver._solve_grid("test.png", "instruction", grid_boxes)
        
        self.assertTrue(len(actions) > 0)
        self.assertEqual(actions[0].target_number, 6)

    @patch('src.solver.add_overlays_to_image')
    @patch('src.solver.os.path.exists')
    @patch('src.solver.Image.open')
    def test_solve_grid_handles_mixed_and_invalid(self, mock_open, mock_exists, mock_add_overlays):
        mock_exists.return_value = True
        grid_boxes = [(0,0,10,10) for _ in range(9)]
        
        # Mixed int, string, and invalid garbage
        self.solver.grid_planner.get_grid_selection.return_value = ([1, "2", "invalid"], False)
        self.solver.image_processor.detect_selected_cells.return_value = []

        actions = self.solver._solve_grid("test.png", "instruction", grid_boxes)
        
        self.assertEqual(len(actions), 2)
        # Should have processed 1 and "2"
        # We don't guarantee order if implementation changes, but usually it preserves order
        targets = [a.target_number for a in actions]
        self.assertIn(1, targets)
        self.assertIn(2, targets) # Pydantic converts "2" to 2
        self.assertNotIn("invalid", targets)

if __name__ == '__main__':
    unittest.main()

