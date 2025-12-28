import json
import os
import sys
import unittest
import importlib.util
from pathlib import Path

# Add project root to path before imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.solver import CaptchaSolver
from src.action_types import ClickAction, DoneAction, WaitAction
from src.tool_calls.find_grid import find_grid, detect_selected_cells

SYSTEM_INSTRUCTION = (
    "Select the image tiles needed to solve the captcha puzzle. "
    "Follow the instruction text shown in the captcha image itself. "
    "Don't select any already-selected tiles (indicated by a blue circle with a checkmark in the top-left)."
)


def _extract_selected_numbers(result):
    selected_numbers = []

    if isinstance(result, list):
        # List of ClickActions
        for action in result:
            if isinstance(action, ClickAction) and action.target_number:
                selected_numbers.append(int(action.target_number))
        return selected_numbers

    if isinstance(result, ClickAction) and result.target_number:
        return [int(result.target_number)]

    if isinstance(result, (DoneAction, WaitAction)):
        return []

    return []


def _score_prediction(required, optional, selected):
    """
    Score a single image prediction.

    Design goals (aligned to prompt tuning):
    - Full solves are rewarded more heavily.
    - Missing required is punished more than selecting extras.
    - Selecting optional is a small positive, but never compensates for missing required.
    """
    required = set(required or [])
    optional = set(optional or [])
    selected = set(selected or [])

    required_hits = len(selected & required)
    optional_hits = len(selected & optional)
    missing_required = len(required - selected)
    extras = len(selected - (required | optional))

    # Full solve bonus: no missing required and no extras.
    full_solve = (missing_required == 0 and extras == 0)

    # Normalize by number of required; treat empty-required cases as 1 for stability.
    r_denom = max(1, len(required))

    # Base scoring: heavily weight required, punish missing required strongly.
    # Keep extras penalty meaningful but smaller than missing-required penalty.
    base = (
        (2.0 * required_hits) / r_denom
        + (0.25 * optional_hits) / max(1, len(optional) or 1)
        - (3.0 * missing_required) / r_denom
        - (0.75 * extras) / max(1, (len(required) + len(optional)) or 1)
    )

    # Clamp base into [0, 1] so we can do weighted averaging cleanly.
    base = max(0.0, min(1.0, base))

    # Add a small bonus for full solves (still clamps at 1.0).
    if full_solve:
        base = min(1.0, base + 0.35)

    # Weight: more required tiles => more important; full solve also increases weight.
    weight = 1.0 + float(len(required))
    if full_solve:
        weight *= 1.75

    details = {
        "required_hits": required_hits,
        "optional_hits": optional_hits,
        "missing_required": missing_required,
        "extras": extras,
        "full_solve": full_solve,
        "weight": weight,
        "score": base,
    }

    return base, weight, details

class TestCoreRecaptcha(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load answers
        with open("captchaimages/coreRecaptcha/recaptchaAnswers.json", "r") as f:
            cls.answers = json.load(f)
        
        # Check API key
        cls.api_key = os.getenv("GEMINI_API_KEY")
        try:
            has_google_genai = importlib.util.find_spec("google.genai") is not None
        except ModuleNotFoundError:
            has_google_genai = False
        if not has_google_genai:
            raise unittest.SkipTest("google-genai is not installed. Install it to run real Gemini-backed tests.")
        if not cls.api_key:
            raise unittest.SkipTest("GEMINI_API_KEY not set. Skipping real Gemini-backed tests.")

    def setUp(self):
        self.solver = CaptchaSolver(provider="gemini", api_key=self.api_key)

    def test_all_images(self):
        # If you want this test to gate CI, set RECAPTCHA_MIN_SCORE (0.0 - 1.0).
        min_score = float(os.getenv("RECAPTCHA_MIN_SCORE", "0.0"))

        total_weight = 0.0
        weighted_score_sum = 0.0
        per_image_rows = []
        
        base_dir = Path("captchaimages/coreRecaptcha")
        
        for filename, expected in self.answers.items():
            image_path = base_dir / filename
            if not image_path.exists():
                print(f"Skipping {filename} - file not found")
                continue

            print(f"\nTesting {filename}...")
            instruction = SYSTEM_INSTRUCTION
            
            try:
                result = self.solver.solve(str(image_path), instruction)
                
                selected_numbers = _extract_selected_numbers(result)

                # --- HYBRID FILTERING START ---
                # Use find_grid/detect_selected_cells to detect already-selected cells and filter them out.
                # This mimics the "post-processing" step we want to validate.
                try:
                    grid_boxes = find_grid(str(image_path))
                    if grid_boxes:
                        already_selected_indices, _ = detect_selected_cells(str(image_path), grid_boxes)
                        if already_selected_indices:
                            original_count = len(selected_numbers)
                            # Remove any LLM-selected number that is also visually detected as already selected
                            selected_numbers = [n for n in selected_numbers if n not in already_selected_indices]
                            if len(selected_numbers) < original_count:
                                print(f"  [Hybrid Filter] Removed {original_count - len(selected_numbers)} tiles detected as already selected: {set(already_selected_indices)}")
                except Exception as cv_e:
                     print(f"  [Hybrid Filter] Warning: CV check failed: {cv_e}")
                # --- HYBRID FILTERING END ---
                
                # Validation
                required = set(expected.get("required", []))
                optional = set(expected.get("optional", []))
                selected = set(selected_numbers)

                score, weight, details = _score_prediction(required, optional, selected)
                weighted_score_sum += score * weight
                total_weight += weight

                missing_required = required - selected
                extra_selected = selected - (required | optional)
                per_image_rows.append(
                    {
                        "filename": filename,
                        "score": score,
                        "weight": weight,
                        "full_solve": details["full_solve"],
                        "missing_required": sorted(missing_required),
                        "extra_selected": sorted(extra_selected),
                        "selected": sorted(selected),
                        "required": sorted(required),
                        "optional": sorted(optional),
                    }
                )

                status = "FULL" if details["full_solve"] else "PARTIAL"
                print(
                    f"{filename}: {status} | score={score:.3f} (w={weight:.2f}) | "
                    f"selected={sorted(selected)} | missing_required={sorted(missing_required)} | extras={sorted(extra_selected)}"
                )
                    
            except Exception as e:
                # Count exceptions as a hard 0 for the image (still contributes weight).
                required = set(expected.get("required", []))
                optional = set(expected.get("optional", []))
                score, weight, details = _score_prediction(required, optional, selected=set())
                weighted_score_sum += 0.0 * weight
                total_weight += weight

                per_image_rows.append(
                    {
                        "filename": filename,
                        "score": 0.0,
                        "weight": weight,
                        "full_solve": False,
                        "missing_required": sorted(required),
                        "extra_selected": [],
                        "selected": [],
                        "required": sorted(required),
                        "optional": sorted(optional),
                        "exception": str(e),
                    }
                )

                print(f"{filename}: EXCEPTION {e} | score=0.000 (w={weight:.2f})")
                
        overall = (weighted_score_sum / total_weight) if total_weight > 0 else 0.0

        full_solves = sum(1 for r in per_image_rows if r.get("full_solve"))
        total_images = len(per_image_rows)

        print("\n=== Core Recaptcha Overall Score ===")
        print(f"Images scored: {total_images}")
        print(f"Full solves: {full_solves}/{total_images}")
        print(f"Overall weighted score: {overall:.4f}")
        print(f"Minimum required score (RECAPTCHA_MIN_SCORE): {min_score:.4f}")

        # Fail only if a threshold is configured.
        self.assertGreaterEqual(
            overall,
            min_score,
            msg=(
                f"Overall score {overall:.4f} is below threshold {min_score:.4f}. "
                f"Set RECAPTCHA_MIN_SCORE=0.0 to disable gating."
            ),
        )

if __name__ == "__main__":
    unittest.main()

