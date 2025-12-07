"""
CaptchaKraken CLI - Command-line interface for captcha solving.

Usage:
    python -m src.captchakraken.cli image.png model_name api_provider [api_key]
"""

import argparse
import json
import sys
import os

# Add src to path if running directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.captchakraken.solver import CaptchaSolver


def main():
    parser = argparse.ArgumentParser(
        description="CaptchaKraken - AI-powered captcha solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with Ollama (no API key needed)
  python -m src.captchakraken.cli captcha.png llama3.2:3b ollama

  # Use OpenAI
  python -m src.captchakraken.cli captcha.png gpt-4o openai sk-your-api-key

  # Use Gemini
  python -m src.captchakraken.cli captcha.png gemini-2.0-flash-exp gemini your-gemini-key
        """
    )

    parser.add_argument(
        "image_path",
        help="Path to the captcha image"
    )
    parser.add_argument(
        "model",
        help="AI model to use for solving"
    )
    parser.add_argument(
        "api_provider",
        choices=["ollama", "openai", "gemini", "deepseek"],
        help="API provider to use"
    )
    parser.add_argument(
        "api_key",
        nargs="?",
        help="API key (not required for ollama)"
    )

    args = parser.parse_args()

    # Validate image exists
    if not os.path.exists(args.image_path):
        print(json.dumps({"error": f"Image not found: {args.image_path}"}), file=sys.stderr)
        sys.exit(1)

    # Validate API key requirement
    if args.api_provider != "ollama" and not args.api_key:
        print(json.dumps({"error": f"API key required for {args.api_provider} provider"}), file=sys.stderr)
        sys.exit(1)

    try:
        # Map API provider to solver parameters
        solver_kwargs = {
            "planner_backend": args.api_provider,
            "planner_model": args.model,
        }

        if args.api_provider == "openai":
            solver_kwargs["openai_api_key"] = args.api_key
        elif args.api_provider == "gemini":
            solver_kwargs["gemini_api_key"] = args.api_key
        elif args.api_provider == "deepseek":
            # DeepSeek uses OpenAI-compatible API
            solver_kwargs["openai_api_key"] = args.api_key
            solver_kwargs["openai_base_url"] = "https://api.deepseek.com/v1"

        solver = CaptchaSolver(**solver_kwargs)

        # Solve the captcha - use a default context since the user wants to simplify
        action = solver.solve_step(
            args.image_path,
            "Solve this captcha"  # Default context
        )

        # Convert to dict for JSON output
        action_data = action.model_dump()
        print(json.dumps(action_data))

    except Exception as e:
        # Always print traceback for debugging purposes now
        import traceback
        traceback.print_exc()
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
