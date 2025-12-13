"""
GridPlanner - Specialized planner for grid-based captchas (3x3, 4x4).
Inherits from ActionPlanner to reuse backend connection.
"""

from typing import List, Tuple
from .planner import ActionPlanner

# Detailed prompt for grid selection (Tuned for precision)
SELECT_GRID_PROMPT = """You are an expert captcha solver.
Task: Solve the captcha puzzle by selecting the correct grid cells (1-{total}).
Follow the instruction text shown inside the captcha image itself. You may also use the provided instruction: "{instruction}".

Image Grid: {rows}x{cols}
{grid_hint}

CRITICAL NEGATIVE CONSTRAINTS (MUST FOLLOW):
1. IGNORE ALREADY SELECTED CELLS: If a cell has a BLUE CIRCLE with a CHECKMARK (usually top-left), or a thick white overlay, it is ALREADY SELECTED.
   - DO NOT SELECT IT AGAIN.
   - Even if it contains the target, if it has the blue checkmark, exclude it.
2. IGNORE RIDERS/PEOPLE: If the target is a vehicle (motorcycle, bicycle, bus), DO NOT select cells that contain ONLY the rider (helmet, body, legs) or driver. Select ONLY the machine parts.

Rules:
1. TARGET IDENTIFICATION: Focus on the main object specified.
2. INCLUSION (PRIORITY: HIGH RECALL):
   - Select cells containing ANY visible part of the target object.
   - INCLUDE: edges, corners, bumpers, tires, windshields, traffic light hoods/frames, handle-bars.
   - Even small fragments count if they are part of the object structure.
   - For 4x4 (single large image): Select ALL cells with any part of the object to maintain connectivity.
   - For 3x3 (separate images): Select only cells with a clear view of the object.
3. EXCLUSION: DO NOT select cells that contain ONLY:
   - Background (road, sky, trees, buildings)
   - Supporting structures (poles, wires, signs) - UNLESS they are integral to the object (e.g. crosswalk lines).
   - Shadows/Reflections (unless the object itself is also present).

Instructions for specific targets:
- "traffic lights": Select the signal housing (box) and lights. INCLUDE the surrounding hood/shield. IGNORE the pole, arm, wires.
- "bus" / "motorcycle" / "vehicle" / "bicycle": Select vehicle parts (body, wheels, windows, handlebars, exhaust). IGNORE riders/drivers.
- "stairs": Select the tread/risers. IGNORE independent walls or railings not attached to steps.
- "crosswalk": Select the painted road lines.
- "fire hydrant": Select the hydrant body.
- "motorcycle": Select the motorcycle body, wheels, handlebars, and exhaust, ignore the rider, any coverings on the motorcycle etc.

Process:
1. Scan the image to locate the target object(s).
2. Check for "Already Selected" badges (Blue Checkmarks) - these are FORBIDDEN.
3. For each remaining cell 1-{total}, decide if it contains a visible part of the target.
4. Return the list of selected numbers.

Respond JSON ONLY:
{{
  "analysis": "Brief analysis",
  "cell_states": {{
    "1": "Description. Contains target? [Yes/No]. Already selected? [Yes/No]",
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
        
        # Log model reasoning (if present)
        self._log(f"Analysis: {result.get('analysis', 'N/A')}")
        self._log(f"Cell States: {result.get('cell_states', 'N/A')}")
        self._log(f"Final selection: {selected}")

        return selected
