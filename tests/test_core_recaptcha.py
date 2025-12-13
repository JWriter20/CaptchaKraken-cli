import json
import os
import unittest
from pathlib import Path
from src.solver import CaptchaSolver
from src.action_types import ClickAction, DoneAction, WaitAction

# Instructions mapping - adjust as needed based on image content
INSTRUCTIONS = {
    "busImagesRecaptcha.png": "Select all squares with buses",
    "fireHydrantCaptcha.png": "Select all squares with fire hydrants",
    "motorcycleImagesRecaptcha.png": "Select all squares with motorcycles",
    "recaptchaImages.png": "Select all squares with traffic lights",
    "recaptchaImages2.png": "Select all squares with bicycles", 
    "recaptchaImages3.png": "Select all squares with palm trees", 
    "selectedDisappearingRecaptcha.png": "Select all squares with traffic lights",
    "selectedFieldsRecaptcha.png": "Select all squares with traffic lights",
    "stairsCaptchaImage.png": "Select all squares with stairs",
    "streetLightsRecaptcha.png": "Select all squares with traffic lights",
    "recaptchaBusSolved.png": "Select all squares with buses",
    "recaptchaTrafficLightsSolved.png": "Select all squares with traffic lights"
}

class TestCoreRecaptcha(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load answers
        with open("captchaimages/coreRecaptcha/recaptchaAnswers.json", "r") as f:
            cls.answers = json.load(f)
        
        # Check API key
        cls.api_key = os.getenv("GEMINI_API_KEY")
        if not cls.api_key:
            print("WARNING: GEMINI_API_KEY not set. Tests will likely fail or need mocking.")

    def setUp(self):
        self.solver = CaptchaSolver(provider="gemini", api_key=self.api_key)

    def test_all_images(self):
        failures = []
        
        base_dir = Path("captchaimages/coreRecaptcha")
        
        for filename, expected in self.answers.items():
            if filename not in INSTRUCTIONS:
                print(f"Skipping {filename} - no instruction mapped")
                continue
                
            image_path = base_dir / filename
            if not image_path.exists():
                print(f"Skipping {filename} - file not found")
                continue

            print(f"\nTesting {filename}...")
            instruction = INSTRUCTIONS[filename]
            
            try:
                # Run solver
                # We expect _solve_grid to return List[ClickAction] or Done/Wait
                # Since we are calling solve(), it might call _solve_grid internally if grid is detected.
                # solve() returns Union[CaptchaAction, List[ClickAction]]
                
                result = self.solver.solve(str(image_path), instruction)
                
                selected_numbers = []
                if isinstance(result, list):
                    # List of ClickActions
                    for action in result:
                        if isinstance(action, ClickAction) and action.target_number:
                            selected_numbers.append(int(action.target_number))
                elif isinstance(result, ClickAction) and result.target_number:
                     selected_numbers.append(int(result.target_number))
                elif isinstance(result, (DoneAction, WaitAction)):
                    selected_numbers = [] # None selected
                
                # Validation
                required = set(expected.get("required", []))
                optional = set(expected.get("optional", []))
                selected = set(selected_numbers)
                
                # Logic:
                # 1. All required must be selected.
                # 2. Selected must be within required + optional.
                
                missing_required = required - selected
                extra_selected = selected - (required | optional)
                
                if missing_required or extra_selected:
                    error_msg = f"{filename}: Failed. "
                    if missing_required:
                        error_msg += f"Missing required: {missing_required}. "
                    if extra_selected:
                        error_msg += f"Selected extras: {extra_selected}. "
                    error_msg += f"Got {selected}, Expected required {required}"
                    print(error_msg)
                    failures.append(error_msg)
                else:
                    print(f"{filename}: Passed. Selected {selected}")
                    
            except Exception as e:
                error_msg = f"{filename}: Exception {e}"
                print(error_msg)
                failures.append(error_msg)
                
        if failures:
            self.fail("\n".join(failures))

if __name__ == "__main__":
    unittest.main()

