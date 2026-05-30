"""
CaptchaKraken CLI (v2).

Modes:
  python -m src.cli image.png [model_name] [api_provider] [api_key]
        Solve a captcha image / video. `api_provider` is kept for v1
        compat; only `captchaKrakenApi` is supported and is the default.

  python -m src.cli check-movement   img1.png img2.png [threshold]
  python -m src.cli check-movement-batch threshold img1 img2 [img3 ...]
        Frame-diff helpers used by the Playwright lib's video flow.

  python -m src.cli find-grid       image.png
  python -m src.cli detect-selected image.png
  python -m src.cli get-numbered-grid image.png
  python -m src.cli find-checkbox   image.png
        OpenCV tool calls.
"""

import argparse
import json
import os
import sys

from .solver import CaptchaSolver
from .timing import timed


def _handle_movement_commands() -> bool:
    if len(sys.argv) <= 1:
        return False

    cmd = sys.argv[1]

    if cmd == "check-movement":
        if len(sys.argv) < 4:
            print(
                json.dumps({"error": "Usage: python -m src.cli check-movement img1.png img2.png [threshold]"}),
                file=sys.stderr,
            )
            sys.exit(1)

        from .image_processor import ImageProcessor

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
        return True

    if cmd == "check-movement-batch":
        if len(sys.argv) < 5:
            print(
                json.dumps({"error": "Usage: python -m src.cli check-movement-batch threshold img1 img2 [img3 ...]"}),
                file=sys.stderr,
            )
            sys.exit(1)

        import cv2

        try:
            threshold = float(sys.argv[2])
        except ValueError:
            threshold = 0.003
        paths = sys.argv[3:]

        imgs = [cv2.imread(p) for p in paths]
        valid = [(p, im) for p, im in zip(paths, imgs) if im is not None]
        if len(valid) < 2:
            print(json.dumps({"has_movement": False, "max_ratio": 0.0, "valid_samples": len(valid)}))
            return True

        max_ratio = 0.0
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                a, b = valid[i][1], valid[j][1]
                if a.shape != b.shape:
                    max_ratio = 1.0
                    break
                diff = cv2.absdiff(a, b)
                gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                _, thr = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
                ratio = cv2.countNonZero(thr) / (thr.shape[0] * thr.shape[1])
                if ratio > max_ratio:
                    max_ratio = ratio
            if max_ratio >= 1.0:
                break

        print(
            json.dumps(
                {
                    "has_movement": max_ratio > threshold,
                    "max_ratio": max_ratio,
                    "valid_samples": len(valid),
                }
            )
        )
        return True

    return False


def _handle_tool_commands() -> bool:
    if len(sys.argv) <= 1:
        return False
    cmd = sys.argv[1]
    if cmd not in {"find-grid", "find-checkbox", "detect-selected", "get-numbered-grid"}:
        return False

    if len(sys.argv) < 3:
        print(json.dumps({"error": f"Usage: python -m src.cli {cmd} image.png"}), file=sys.stderr)
        sys.exit(1)

    image_path = sys.argv[2]
    if not os.path.exists(image_path):
        print(json.dumps({"error": f"Image not found: {image_path}"}), file=sys.stderr)
        sys.exit(1)

    try:
        if cmd == "find-grid":
            from .tool_calls.find_grid import find_grid

            result = find_grid(image_path)
        elif cmd == "detect-selected":
            from .tool_calls.find_grid import detect_selected_cells, find_grid

            grid_boxes = find_grid(image_path)
            if not grid_boxes:
                result = {"error": "No grid detected"}
            else:
                selected, loading = detect_selected_cells(image_path, grid_boxes)
                result = {"selected": selected, "loading": loading}
        elif cmd == "get-numbered-grid":
            from .tool_calls.find_grid import find_grid, get_numbered_grid_overlay

            grid_boxes = find_grid(image_path)
            if not grid_boxes:
                result = {"error": "No grid detected"}
            else:
                overlay_path = get_numbered_grid_overlay(image_path, grid_boxes)
                result = {"overlay_image": overlay_path}
        else:
            from .tool_calls.find_checkbox import find_checkbox

            result = find_checkbox(image_path)

        print(json.dumps(result))
        return True
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


def main():
    if _handle_movement_commands():
        return
    if _handle_tool_commands():
        return

    parser = argparse.ArgumentParser(description="CaptchaKraken v2 (vLLM)")
    parser.add_argument("image_path", help="Path to the captcha image or video")
    parser.add_argument(
        "model",
        nargs="?",
        default=None,
        help="LoRA name registered with vLLM (default: 'captcha').",
    )
    parser.add_argument(
        "api_provider",
        nargs="?",
        default="captchaKrakenApi",
        choices=["captchaKrakenApi"],
        help="Kept for v1 argv compatibility; only captchaKrakenApi is supported in v2.",
    )
    parser.add_argument(
        "api_key",
        nargs="?",
        default=None,
        help="Bearer token (or set VLLM_API_KEY / CAPTCHA_KRAKEN_API_KEY).",
    )
    parser.add_argument(
        "--puzzle-source",
        default="unknown",
        choices=["hcaptcha", "recaptcha", "unknown"],
        help="Vendor hint from the Playwright wrapper. hCaptcha skips grid detection "
        "(find_grid false-positives on the header/footer bands of click puzzles).",
    )

    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(json.dumps({"error": f"Image not found: {args.image_path}"}), file=sys.stderr)
        sys.exit(1)

    try:
        with timed("cli.total"):
            solver = CaptchaSolver(model=args.model, api_key=args.api_key)
            result = solver.solve(args.image_path, puzzle_source=args.puzzle_source)

        if isinstance(result, list):
            action_data = [a.model_dump() for a in result]
        elif hasattr(result, "model_dump"):
            action_data = result.model_dump()
        else:
            action_data = result

        print(json.dumps({"actions": action_data, "token_usage": solver.planner.token_usage}))
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
