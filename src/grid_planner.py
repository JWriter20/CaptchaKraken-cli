"""
GridPlanner - Specialized planner for grid-based captchas (3x3, 4x4).
Inherits from ActionPlanner to reuse backend connection.
"""

from typing import List, Tuple
from .planner import ActionPlanner

# Detailed prompt for grid selection (Restored & Tuned)
SELECT_GRID_PROMPT = """You are a captcha solver.
Task: Select cells in the {rows}x{cols} grid (1-{total}) that match the instruction.

Instruction: "{instruction}"
{grid_hint}

1. Deconstruct the Instruction:
   - Identify the "Core Target" (e.g. for "buses", core = vehicle body/wheels/roof; for "traffic lights", core = the HEAD/HOUSING containing the lights; for "motorcycles", core = vehicle parts/wheels/engine; for "stairs", core = steps/treads).
   - Identify "Associated Items" to EXCLUDE (e.g. for "buses", exclude road/asphalt; for "traffic lights", exclude the POLE/SUPPORT structure if it extends away from the light housing).
   - EXCLUDE RIDERS/OPERATORS: If the target is a vehicle/object, EXCLUDE any person operating or riding it (riders, drivers, passengers). The rider (helmet, body, arms) is NOT the vehicle.
   - EXCLUDE SHADOWS: Shadows cast by the object are NOT the object. Do not select cells that contain ONLY the shadow.

2. Evaluate each cell (1-{total}):
   - Describe the visual content.
   - CHECK: Is the Core Target visible? (Even a small edge/corner counts). NOTE: A rider or shadow is NOT the Core Target.
   - CHECK: Is it a structural continuation of the Core Target? (e.g. vehicle body, roof, traffic light housing). NOTE: Poles, riders, and shadows are NOT continuations.
   - CHECK: Is it ONLY an Associated Item, Rider, or Shadow? (e.g. only a pole, only a railing, only a road, only a rider's body/helmet without the vehicle part, only a shadow).
   - CONSTRAINT: If it is ONLY an Associated Item, Rider, or Shadow, do NOT select it.

3. Final Selection:
   - Select ALL cells containing the Core Target OR structural continuations (e.g. vehicle parts, traffic light housing parts).
   - For 4x4: Be precise but inclusive of edges. 
   - STRICTLY REJECT: Cells containing ONLY associated items (e.g. poles with no lights, railings, roads, sky), ONLY riders/people, or ONLY shadows.
   
Respond JSON ONLY:
{{
  "analysis": "Core: [Describe], Associated/Excluded: [Describe]",
  "cell_states": {{
    "1": "Description. Core Visible? [YES/NO]. Associated Only? [YES/NO]. -> [MATCH/NO MATCH]",
    ...
  }},
  "selected_numbers": [list of integers representing cells to click]
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
