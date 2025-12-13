"""
GridPlanner - Specialized planner for grid-based captchas (3x3, 4x4).
Inherits from ActionPlanner to reuse backend connection.
"""

from typing import List, Tuple
from .planner import ActionPlanner

# Detailed prompt for grid selection (Tuned for precision)
SELECT_GRID_PROMPT = """You are an expert captcha solver.
Task: Identify grid cells (1-{total}) that contain the "{instruction}".

Image Grid: {rows}x{cols}
{grid_hint}

Rules:
1. TARGET IDENTIFICATION: Focus on the main object specified.
2. INCLUSION: Select cells containing ANY visible part of the target object.
   - For 4x4 (single large image): Select ALL cells with any part of the object to maintain connectivity.
   - For 3x3 (separate images): Select only cells with a clear view of the object.
3. EXCLUSION: DO NOT select cells that contain ONLY:
   - Background (road, sky, trees, buildings)
   - Supporting structures (poles, wires, signs) - UNLESS they are integral to the object (e.g. crosswalk lines).
   - Riders/People (if target is vehicle).
   - Shadows/Reflections.
   - ALREADY SELECTED cells: Look for a blue/green checkmark in the corner or a faded/white overlay. If present, DO NOT SELECT.

Instructions for specific targets:
- "traffic lights": Select ONLY the signal housing (box) and lights. IGNORE the pole, arm, wires, and pedestrian signals.
- "bus" / "motorcycle" / "vehicle" / "bicycle": Select vehicle parts (body, wheels, windows, handlebars). IGNORE riders/drivers.
- "stairs": Select the tread/risers. IGNORE independent walls or railings not attached to steps.
- "crosswalk": Select the painted road lines.
- "fire hydrant": Select the hydrant body.

Process:
1. Scan the image to locate the target object(s).
2. For each cell 1-{total}, decide if it contains the target.
3. Return the list of selected numbers.

Respond JSON ONLY:
{{
  "reasoning": "Brief analysis",
  "cell_analysis": {{
    "1": "Description. Match? [Yes/No]",
    ...
  }},
  "selected_numbers": [list of integers]
}}"""

class GridPlanner(ActionPlanner):
    """
    Specialized planner for grid selection tasks.
    """

    def get_grid_selection(self, image_path: str, rows: int, cols: int, instruction: str = "Solve the captcha by selecting the correct images") -> List[int]:
        """
        Ask which numbers to select in the grid.
        Returns a list of selected cell numbers.
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
        
        # Log reasoning
        self._log(f"Analysis: {result.get('analysis', 'N/A')}")
        self._log(f"Cell States: {result.get('cell_states', 'N/A')}")
        self._log(f"Final selection: {selected}")

        return selected
