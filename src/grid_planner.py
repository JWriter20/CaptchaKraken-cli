"""
GridPlanner - Specialized planner for grid-based captchas (3x3, 4x4).
Inherits from ActionPlanner to reuse backend connection.
"""

from typing import List, Tuple
from .planner import ActionPlanner

# Concise prompt for grid selection
SELECT_GRID_PROMPT = """You are a captcha solver.
Task: Select cells in the {rows}x{cols} grid (1-{total}) that match the instruction.

Instruction: "{instruction}"
{grid_hint}

1. Analyze the instruction.
2. For each cell (1-{total}), determine:
   - Content: What is visible?
   - Match: Does it contain ANY PART of the target object? (Even small edges/slivers count for 4x4)

3. Return the list of cells that MATCH the instruction.
   - For 4x4 (single large image): Include cells with ANY part of the object. If a matching object touches a cell border, SELECT THE NEIGHBORING CELL too if it contains any continuation.
   - For 3x3 (separate images): Include only clear matches.

Respond JSON ONLY:
{{
  "instruction_analysis": "what to find",
  "cell_states": {{
    "1": "content description",
    ...
  }},
  "loading_cells": [],
  "selected_numbers": [2, 3] // Cells that MATCH
}}"""

class GridPlanner(ActionPlanner):
    """
    Specialized planner for grid selection tasks.
    """

    def get_grid_selection(self, instruction: str, image_path: str, rows: int, cols: int) -> Tuple[List[int], bool]:
        """
        Ask which numbers to select in the grid.
        """
        total = rows * cols

        self._log("GridPlanner.get_grid_selection called")
        self._log(f"  instruction: '{instruction}'")
        self._log(f"  grid: {rows}x{cols}")

        grid_hint = ""
        if rows == 4 and cols == 4:
            grid_hint = "Hint: Single large image split into tiles. Select ALL parts."
        elif rows == 3 and cols == 3:
            grid_hint = "Hint: Separate images. Select only clear matches."

        prompt = SELECT_GRID_PROMPT.format(
            rows=rows, 
            cols=cols, 
            total=total, 
            instruction=instruction,
            grid_hint=grid_hint
        )
        
        response = self._chat_with_image(prompt, image_path)
        result = self._parse_json(response)

        selected = result.get("selected_numbers", [])
        loading_cells = result.get("loading_cells", [])

        # Log reasoning
        self._log(f"Analysis: {result.get('instruction_analysis', 'N/A')}")
        
        # Only wait if we have loading cells AND no valid selections
        should_wait = len(loading_cells) > 0 and len(selected) == 0

        if should_wait:
            self._log(f"Loading cells detected: {loading_cells}. Suggesting wait.")

        self._log(f"Final selection: {selected}")

        return selected, should_wait

