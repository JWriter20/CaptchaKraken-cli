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
        choices=["ollama", "gemini"],
        help="API provider to use one of: ollama, gemini",
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
        # Create solver with simplified interface
        solver = CaptchaSolver(
            model=args.model,
            provider=args.api_provider,
            api_key=args.api_key,
        )

        # Solve the captcha
        result = solver.solve(args.image_path, "Solve this captcha")

        # Convert to dict for JSON output
        # solve() returns either a single action or a list of ClickActions (for grids)
        if isinstance(result, list):
            action_data = [action.model_dump() for action in result]
        else:
            action_data = result.model_dump()
        print(json.dumps(action_data))

    except Exception as e:
        # Always print traceback for debugging purposes now
        import traceback

        traceback.print_exc()
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
