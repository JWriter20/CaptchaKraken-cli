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

1. Define the target object's visual core (e.g., "stairs" = steps/risers).
2. Evaluate each cell (1-{total}):
   - Visible content?
   - Contains the CORE target object?
   - NEGATIVE CONSTRAINT: Do NOT select cells containing only "associated" items (e.g., railings, poles) or "context" (trees, walls) if the core object (steps) is absent.

3. Selection:
   - Select ALL cells containing the target object, including those with only a part of it.
   - For 4x4: If the object spans cells, ensure you select the entire object. But strictly respect the negative constraint: associated items (like railings) alone do NOT count.
   - CHECK CORNERS: Often the object extends into corners (like bottom steps).

Respond JSON ONLY:
{{
  "instruction_analysis": "target core features",
  "cell_states": {{
    "1": "content - match/no match",
    ...
  }},
  "loading_cells": [],
  "selected_numbers": [2, 3]
}}"""

class GridPlanner(ActionPlanner):
    """
    Specialized planner for grid selection tasks.
    """

    def get_grid_selection(self, image_path: str, rows: int, cols: int, instruction: str = "Solve the captcha by selecting the correct images, DO NOT click on cells you are not sure about") -> Tuple[List[int], bool]:
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
