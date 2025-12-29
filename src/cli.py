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

    # Handle individual tool calls
    if len(sys.argv) > 1 and sys.argv[1] in ["detect", "find-checkbox", "find-grid", "segment", "simulate-drag", "find-connected-elems", "detect-selected", "get-numbered-grid"]:
        command = sys.argv[1]
        if len(sys.argv) < 3:
            print(json.dumps({"error": f"Usage: python -m src.cli {command} image.png [args...]"}), file=sys.stderr)
            sys.exit(1)
        
        image_path = sys.argv[2]
        if not os.path.exists(image_path):
            print(json.dumps({"error": f"Image not found: {image_path}"}), file=sys.stderr)
            sys.exit(1)

        try:
            result = None
            if command == "find-grid":
                from src.tool_calls.find_grid import find_grid
                result = find_grid(image_path)
            
            elif command == "detect-selected":
                from src.tool_calls.find_grid import find_grid, detect_selected_cells
                grid_boxes = find_grid(image_path)
                if not grid_boxes:
                    result = {"error": "No grid detected"}
                else:
                    selected, loading = detect_selected_cells(image_path, grid_boxes)
                    result = {"selected": selected, "loading": loading}

            elif command == "get-numbered-grid":
                from src.tool_calls.find_grid import find_grid, get_numbered_grid_overlay
                grid_boxes = find_grid(image_path)
                if not grid_boxes:
                    result = {"error": "No grid detected"}
                else:
                    overlay_path = get_numbered_grid_overlay(image_path, grid_boxes)
                    result = {"overlay_image": overlay_path}
            
            elif command == "find-checkbox":
                from src.tool_calls.find_checkbox import find_checkbox
                result = find_checkbox(image_path)
            
            elif command == "segment":
                from src.solver import CaptchaSolver
                from src.tool_calls.segment import segment
                # Need a solver instance for image_processor and attention
                solver = CaptchaSolver(provider="ollama", model="qwen3-vl:4b")
                labeled_path, objects = segment(solver.image_processor, solver._get_attention(), image_path, solver.debug)
                result = {"labeled_image": labeled_path, "objects": objects}
            
            elif command == "detect":
                if len(sys.argv) < 4:
                    print(json.dumps({"error": "Usage: python -m src.cli detect image.png object_class"}), file=sys.stderr)
                    sys.exit(1)
                object_class = sys.argv[3]
                from src.solver import CaptchaSolver
                from src.tool_calls.detect import detect
                solver = CaptchaSolver(provider="ollama", model="qwen3-vl:4b")
                result = detect(solver._get_attention(), image_path, object_class)
            
            elif command == "simulate-drag":
                if len(sys.argv) < 5:
                    print(json.dumps({"error": "Usage: python -m src.cli simulate-drag image.png source_desc target_desc"}), file=sys.stderr)
                    sys.exit(1)
                source_desc = sys.argv[3]
                target_desc = sys.argv[4]
                from src.solver import CaptchaSolver
                from src.tool_calls.simulate_drag import simulate_drag
                # Note: This might need more setup for a full drag simulation (instruction, etc.)
                solver = CaptchaSolver(provider="ollama", model="qwen3-vl:4b")
                result = simulate_drag(solver, image_path, "Simulated drag", source_desc, target_desc)
            
            elif command == "find-connected-elems":
                if len(sys.argv) < 4:
                    print(json.dumps({"error": "Usage: python -m src.cli find-connected-elems image.png instruction"}), file=sys.stderr)
                    sys.exit(1)
                instruction = sys.argv[3]
                from src.tool_calls.find_connected_elems import find_connected_elems
                result = find_connected_elems(image_path, instruction)

            print(json.dumps(result))
            return
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(json.dumps({"error": str(e)}), file=sys.stderr)
            sys.exit(1)

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

        # Token usage from planner
        all_token_usage = solver.planner.token_usage
        
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
