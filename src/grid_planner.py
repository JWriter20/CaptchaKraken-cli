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
   - Status: Does it have a VISIBLE CHECKMARK? (If yes, mark as 'checked', else 'unchecked')
   - Match: Does it contain ANY PART of the target object? (Even small edges/slivers count for 4x4)

3. Return the list of cells that MATCH the instruction but are currently UNCHECKED.
   - For 4x4 (single large image): Include cells with ANY part of the object. If a matching object touches a cell border, SELECT THE NEIGHBORING CELL too if it contains any continuation.
   - For 3x3 (separate images): Include only clear matches.

Respond JSON ONLY:
{{
  "instruction_analysis": "what to find",
  "cell_states": {{
    "1": "content description - checked/unchecked",
    ...
  }},
  "loading_cells": [],
  "already_selected": [1], // Cells that HAVE VISIBLE CHECKMARKS
  "selected_numbers": [2, 3] // Cells that MATCH but are UNCHECKED
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
        already_selected = set(result.get("already_selected", []))
        loading_cells = result.get("loading_cells", [])
        cell_states = result.get("cell_states", {})

        # Robustness: Check cell_states for 'selected' or 'checked' and add to already_selected
        for cell_num_str, state_desc in cell_states.items():
            state_lower = state_desc.lower()
            # "checked" is a substring of "unchecked", so we must ensure "unchecked" is NOT present
            # when looking for "checked".
            is_checked = "checked" in state_lower and "unchecked" not in state_lower
            is_selected = "selected" in state_lower and "unselected" not in state_lower
            
            if is_checked or is_selected:
                try:
                    already_selected.add(int(cell_num_str))
                except ValueError:
                    pass

        # Log reasoning
        self._log(f"Analysis: {result.get('instruction_analysis', 'N/A')}")
        
        # Double check "already selected" logic
        # If the model put a number in 'selected_numbers' but it is actually already selected, remove it
        # This prevents accidental deselection (toggling off)
        selected = [s for s in selected if s not in already_selected]
        
        # Only wait if we have loading cells AND no valid selections
        should_wait = len(loading_cells) > 0 and len(selected) == 0

        if should_wait:
            self._log(f"Loading cells detected: {loading_cells}. Suggesting wait.")

        self._log(f"Final selection: {selected}")

        return selected, should_wait

