import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

try:
    from src.captchakraken.solver import CaptchaSolver
    print("Successfully imported CaptchaSolver")
    solver = CaptchaSolver()
    print("Successfully instantiated CaptchaSolver")
except Exception as e:
    import traceback
    traceback.print_exc()

