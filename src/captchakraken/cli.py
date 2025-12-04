import argparse
import json
import sys
import os

# Add src to path if running directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.captchakraken.solver import CaptchaSolver

def main():
    parser = argparse.ArgumentParser(description="CaptchaKraken CLI")
    parser.add_argument("image_path", help="Path to the captcha image")
    parser.add_argument("prompt", help="Task prompt")
    parser.add_argument("--strategy", default="omniparser", help="Strategy to use (omniparser/holo2/holo_mlx/holo_vllm)")
    parser.add_argument("--api_key", help="OpenAI API Key (optional)")
    parser.add_argument("--api_base", help="OpenAI Base URL (optional, for VLLM)")
    
    args = parser.parse_args()
    
    solver = CaptchaSolver(openai_api_key=args.api_key, openai_base_url=args.api_base)
    try:
        actions = solver.solve(args.image_path, args.prompt, strategy=args.strategy)
        # Convert actions to dicts for JSON output
        # Pydantic models have model_dump
        actions_data = [action.model_dump() for action in actions]
        print(json.dumps(actions_data))
    except Exception as e:
        # Print error to stderr so stdout only contains JSON result or is empty
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

