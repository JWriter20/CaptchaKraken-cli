"""
CaptchaKraken CLI - Command-line interface for captcha solving.

Usage:
    python -m src.captchakraken.cli captcha.png "Solve this captcha"
    python -m src.captchakraken.cli captcha.png "Click the checkbox" --planner ollama --attention transformers
"""

import argparse
import json
import sys
import os

# Add src to path if running directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.captchakraken.solver import CaptchaSolver


def main():
    parser = argparse.ArgumentParser(
        description="CaptchaKraken - AI-powered captcha solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with Ollama
  python -m src.captchakraken.cli captcha.png "Solve this captcha"
  
  # Use OpenAI for planning
  python -m src.captchakraken.cli captcha.png "Click the traffic lights" --planner openai
  
  # Use Gemini for planning
  python -m src.captchakraken.cli captcha.png "Click the traffic lights" --planner gemini
  
  # Visualize attention (debug mode)
  python -m src.captchakraken.cli captcha.png "Find the checkbox" --visualize attention_map.png
        """
    )
    
    parser.add_argument(
        "image_path",
        help="Path to the captcha image"
    )
    parser.add_argument(
        "context",
        help="Context/instructions for solving (e.g., 'Solve this captcha', 'Click the traffic lights')"
    )
    
    # Backend options
    parser.add_argument(
        "--planner",
        choices=["ollama", "openai", "gemini"],
        default="ollama",
        help="Backend for action planning (default: ollama)"
    )
    parser.add_argument(
        "--planner-model",
        default=None,
        help="Model for planning (default: hf.co/unsloth/gemma-3-12b-it-GGUF:Q4_K_M for ollama, gpt-4o for openai, gemini-2.0-flash-exp for gemini)"
    )
    parser.add_argument(
        "--attention-model",
        default=None,
        help="HuggingFace model for attention extraction (default: vikhyatk/moondream2)"
    )
    
    # API options
    parser.add_argument(
        "--api-key",
        help="OpenAI API key (or set OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--gemini-api-key",
        help="Gemini API key (or set GEMINI_API_KEY env var)"
    )
    parser.add_argument(
        "--api-base",
        help="OpenAI base URL (or set OPENAI_BASE_URL env var)"
    )
    
    # Extracted info options
    parser.add_argument("--prompt-text", help="Extracted prompt text from the captcha")
    parser.add_argument("--prompt-image-url", help="URL of the prompt image")
    parser.add_argument("--challenge-element-selector", help="Selector for the main challenge element (informational)")
    parser.add_argument("--elements", help="JSON string of detected interactable elements")

    # Output options
    parser.add_argument(
        "--visualize",
        metavar="OUTPUT_PATH",
        help="Save attention visualization to this path"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate image exists
    if not os.path.exists(args.image_path):
        print(json.dumps({"error": f"Image not found: {args.image_path}"}), file=sys.stderr)
        sys.exit(1)
    
    try:
        solver = CaptchaSolver(
            planner_backend=args.planner,
            planner_model=args.planner_model,
            attention_model=args.attention_model,
            openai_api_key=args.api_key,
            openai_base_url=args.api_base,
            gemini_api_key=args.gemini_api_key,
        )
        
        # Visualize mode
        if args.visualize:
            print(f"Generating attention visualization...", file=sys.stderr)
            solver.visualize_attention(
                args.image_path,
                args.context,
                args.visualize
            )
            print(json.dumps({"visualization": args.visualize}))
            sys.exit(0)
        
        # Enrich context with extracted info
        context = args.context
        if args.prompt_text:
            context += f"\nPrompt Text: {args.prompt_text}"
        if args.prompt_image_url:
            context += f"\nPrompt Image URL: {args.prompt_image_url}"
        
        # Parse elements if provided
        elements = None
        if args.elements:
            try:
                elements = json.loads(args.elements)
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse elements JSON", file=sys.stderr)

        # Normal solve mode
        action = solver.solve_step(
            args.image_path,
            context,
            elements=elements,
            prompt_text=args.prompt_text
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
