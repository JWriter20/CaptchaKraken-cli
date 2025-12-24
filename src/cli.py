"""
CaptchaKraken CLI - Command-line interface for captcha solving.

Usage:
    python -m src.cli image.png model_name api_provider [api_key]
"""

import argparse
import json
import os
import sys

from src.solver import CaptchaSolver
from src.timing import timed


def main():
    # Handle start-tool-server command
    if len(sys.argv) > 1 and sys.argv[1] == "start-tool-server":
        from src.server import start_tool_server
        port = 8000
        if len(sys.argv) > 2:
            try:
                port = int(sys.argv[2])
            except ValueError:
                print(f"Invalid port '{sys.argv[2]}', using default 8000", file=sys.stderr)
        start_tool_server(port)
        return

    # Handle stop-tool-server command
    if len(sys.argv) > 1 and sys.argv[1] == "stop-tool-server":
        import urllib.request
        port = 8000
        if len(sys.argv) > 2:
            try:
                port = int(sys.argv[2])
            except ValueError:
                print(f"Invalid port '{sys.argv[2]}', using default 8000", file=sys.stderr)
        
        try:
            req = urllib.request.Request(f"http://localhost:{port}/shutdown", method="POST")
            with urllib.request.urlopen(req) as response:
                if response.status == 200:
                    print("Server shutdown request sent successfully.", file=sys.stderr)
                else:
                    print(f"Server returned status {response.status}", file=sys.stderr)
        except Exception as e:
            print(f"Failed to stop server: {e}", file=sys.stderr)
            print("Is the server running?", file=sys.stderr)
        return

    # Handle check-movement command
    if len(sys.argv) > 1 and sys.argv[1] == "check-movement":
        if len(sys.argv) < 4:
            print(json.dumps({"error": "Usage: python -m src.cli check-movement img1.png img2.png [threshold]"}), file=sys.stderr)
            sys.exit(1)
        
        from src.image_processor import ImageProcessor
        img1 = sys.argv[2]
        img2 = sys.argv[3]
        threshold = 0.005
        if len(sys.argv) > 4:
            try:
                threshold = float(sys.argv[4])
            except ValueError:
                pass
        
        has_movement = ImageProcessor.detect_movement(img1, img2, threshold)
        print(json.dumps({"has_movement": has_movement}))
        return

    parser = argparse.ArgumentParser(
        description="CaptchaKraken - AI-powered captcha solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with Ollama (no API key needed)
  python -m src.cli captcha.png qwen3-vl:4b ollama

  # Use Gemini
  python -m src.cli captcha.png gemini-2.0-flash-exp gemini your-gemini-key
        """,
    )

    parser.add_argument("image_path", help="Path to the captcha image")
    parser.add_argument(
        "model", help="AI model to use for solving ie. gemini-2.0-flash-exp, qwen3-vl:4b"
    )
    parser.add_argument(
        "api_provider",
        choices=["ollama", "gemini", "openrouter"],
        help="API provider to use one of: ollama, gemini, openrouter",
    )
    parser.add_argument("api_key", nargs="?", help="API key (not required for ollama)")

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
        with timed("cli.total"):
            # Create solver with simplified interface
            with timed("cli.init_solver", extra=f"provider={args.api_provider}, model={args.model}"):
                solver = CaptchaSolver(
                    model=args.model,
                    provider=args.api_provider,
                    api_key=args.api_key,
                )

            # Solve the captcha
            with timed("cli.solve"):
                result = solver.solve(args.image_path, "Solve this captcha")

        # Convert to dict for JSON output
        # solve() returns either a single action or a list of ClickActions (for grids)
        if isinstance(result, list):
            action_data = [action.model_dump() for action in result]
        elif hasattr(result, "model_dump"):
            action_data = result.model_dump()
        else:
            action_data = result

        # Combine token usage from both planners
        all_token_usage = solver.planner.token_usage + solver.grid_planner.token_usage
        
        output = {
            "actions": action_data,
            "token_usage": all_token_usage
        }
        print(json.dumps(output))

    except Exception as e:
        # Always print traceback for debugging purposes now
        import traceback

        traceback.print_exc()
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
