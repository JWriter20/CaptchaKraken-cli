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

1. Analyze the instruction and reference image (if any).
2. Inspect each numbered cell.
   - If a cell is LOADING (blank/white/spinner), mark as "loading".
   - If a cell is ALREADY SELECTED (checkmark/highlight), mark as "selected".
3. Decide which UNCHECKED cells contain the target.
   - For 4x4 (single large image): Select ALL tiles containing ANY part of the object.
   - For 3x3 (separate images): Select ONLY tiles with the CLEAR object.

Respond JSON ONLY:
{{
  "instruction_analysis": "what to find",
  "cell_states": {{
    "1": "content - state (checked/loading/normal)",
    ...
  }},
  "loading_cells": [1],
  "already_selected": [2],
  "selected_numbers": [3, 4] // ONLY new selections. Do not include already_selected.
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
            if "selected" in state_desc.lower() or "checked" in state_desc.lower():
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

