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

1. Deconstruct the Instruction:
   - Identify the "Core Target" (e.g. for "buses", core = vehicle body/wheels/roof; for "traffic lights", core = the HEAD/HOUSING containing the lights).
   - Identify "Associated Items" to EXCLUDE (e.g. for "buses", exclude road/asphalt; for "traffic lights", exclude the POLE/SUPPORT structure if it extends away from the light housing).

2. Evaluate each cell (1-{total}):
   - Describe the visual content.
   - CHECK: Is there a checkmark overlay?
     * LOADING: Large blue/white circle/spinner OR Large blue circle with checkmark in the CENTER of the cell OR Blank/White/Grey empty cell.
     * SELECTED: Small blue checkmark in the TOP-LEFT corner of the cell.
   - CHECK: Is the Core Target visible? (Even a small edge/corner counts).
   - CHECK: Is it a structural continuation of the Core Target? (e.g. vehicle body, roof, traffic light housing). NOTE: Poles are associated items, NOT continuations.
   - CHECK: Is it ONLY an Associated Item? (e.g. only a pole, only a railing, only a road).
   - CONSTRAINT: If it is ONLY an Associated Item, do NOT select it.

3. Final Selection:
   - Select ALL cells containing the Core Target OR structural continuations (e.g. vehicle parts, traffic light housing parts).
   - For 4x4: Be precise but inclusive of edges. 
   - STRICTLY REJECT: Cells containing ONLY associated items (e.g. poles with no lights, railings, roads, sky).
   - LOADING CELLS: Only add cells to "loading_cells" if they match the LOADING state (Center overlay or Blank).
   - IGNORE SELECTED: If a cell is ALREADY SELECTED (Top-Left checkmark), do NOT add to "selected_numbers" and do NOT add to "loading_cells".

Respond JSON ONLY:
{{
  "analysis": "Core: [Describe], Associated/Excluded: [Describe]",
  "cell_states": {{
    "1": "Description. Checkmark? [NONE/LOADING/SELECTED]. Core Visible? [YES/NO]. Associated Only? [YES/NO]. -> [MATCH/NO MATCH/LOADING/ALREADY SELECTED]",
    ...
  }},
  "loading_cells": [list of cells with CENTER LOADING checkmarks or Blank cells],
  "selected_numbers": [list of integers representing cells to click]
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
        cell_states = result.get("cell_states", {})

        # Filter out cells that the model analysis identified as already selected
        # This fixes cases where the model correctly identifies "ALREADY SELECTED" in analysis
        # but fails to exclude it from the final list.
        filtered_selected = []
        for num in selected:
            state_desc = cell_states.get(str(num), "")
            
            # Check for indicators of selection in the analysis text
            is_already_selected = (
                "ALREADY SELECTED" in state_desc or 
                "Checkmark? SELECTED" in state_desc
            )
            
            if is_already_selected:
                self._log(f"Filtering cell {num} because analysis identified it as ALREADY SELECTED.")
            else:
                filtered_selected.append(num)
        
        selected = filtered_selected

        # Log reasoning
        self._log(f"Analysis: {result.get('analysis', 'N/A')}")
        
        # Only wait if we have loading cells AND no valid selections
        should_wait = len(loading_cells) > 0 and len(selected) == 0

        if should_wait:
            self._log(f"Loading cells detected: {loading_cells}. Suggesting wait.")

        self._log(f"Final selection: {selected}")

        return selected, should_wait
