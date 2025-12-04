import sys
import os

# Add project root to path to allow importing from src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.captchakraken.hardware import check_requirements
except ImportError:
    # Fallback if src is not found (e.g. strange structure)
    print("Error: Could not import hardware check module.")
    sys.exit(1)

if __name__ == "__main__":
    print("[*] Verifying system requirements...")
    ok, msg = check_requirements()
    if ok:
        print(f"    Check Passed: {msg}")
        sys.exit(0)
    else:
        print(f"    Check Failed: {msg}")
        sys.exit(1)
