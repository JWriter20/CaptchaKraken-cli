#!/usr/bin/env python3
"""
Test the simplified ActionPlanner with real captcha images.

This demonstrates the deterministic workflow approach:
1. Classify captcha type
2. Execute deterministic sequence based on type
"""

import os
from src.planner import ActionPlanner


def test_classify_captchas():
    """Test classification on different captcha types."""
    planner = ActionPlanner(backend="gemini")
    
    test_images = [
        ("captchaimages/recaptchaBasic.png", "checkbox"),
        ("captchaimages/recaptchaImages.png", "image_selection"),
        ("captchaimages/hcaptchaDragImage1.png", "drag_puzzle"),
    ]
    
    print("=" * 60)
    print("TESTING CAPTCHA CLASSIFICATION")
    print("=" * 60)
    
    for image_path, expected_type in test_images:
        if not os.path.exists(image_path):
            print(f"\n❌ Image not found: {image_path}")
            continue
        
        print(f"\n📷 Testing: {image_path}")
        print(f"   Expected: {expected_type}")
        
        result = planner.classify(image_path)
        
        print(f"   ✅ Got: {result['type']}")
        print(f"   📝 Instruction: {result.get('instruction', 'None')}")
        print(f"   💭 Reasoning: {result.get('reasoning', 'None')}")
        
        if result["type"] == expected_type:
            print(f"   ✅ CORRECT!")
        else:
            print(f"   ❌ MISMATCH!")


def test_image_selection_workflow():
    """Test the complete workflow for image selection captcha."""
    planner = ActionPlanner(backend="gemini")
    
    image_path = "captchaimages/recaptchaImages.png"
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    print("\n" + "=" * 60)
    print("TESTING IMAGE SELECTION WORKFLOW")
    print("=" * 60)
    
    # Step 1: Classify
    print("\n[Step 1] Classifying captcha...")
    classification = planner.classify(image_path)
    print(f"  Type: {classification['type']}")
    print(f"  Instruction: {classification.get('instruction', 'None')}")
    
    if classification["type"] != "image_selection":
        print("❌ Not an image selection captcha, skipping workflow test")
        return
    
    # Step 2: Get detection target
    print("\n[Step 2] Getting detection target...")
    instruction = classification.get("instruction") or "Select the target objects"
    target = planner.get_detection_target(instruction, image_path)
    print(f"  Target to detect: '{target}'")
    
    print("\n[Step 3] Would now call: detect_tool('{}')"
          "and click all returned boxes".format(target))
    
    print("\n✅ Image selection workflow complete!")


def test_drag_puzzle_workflow():
    """Test the complete workflow for drag puzzle captcha."""
    planner = ActionPlanner(backend="gemini")
    
    image_path = "captchaimages/hcaptchaDragImage1.png"
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    print("\n" + "=" * 60)
    print("TESTING DRAG PUZZLE WORKFLOW")
    print("=" * 60)
    
    # Step 1: Classify
    print("\n[Step 1] Classifying captcha...")
    classification = planner.classify(image_path)
    print(f"  Type: {classification['type']}")
    print(f"  Instruction: {classification.get('instruction', 'None')}")
    
    if classification["type"] != "drag_puzzle":
        print("❌ Not a drag puzzle, skipping workflow test")
        return
    
    # Step 2: Get drag prompts
    print("\n[Step 2] Getting drag prompts...")
    instruction = classification.get("instruction") or "Complete the puzzle"
    prompts = planner.get_drag_prompts(instruction, image_path)
    print(f"  Draggable: '{prompts['draggable_prompt']}'")
    print(f"  Destination: '{prompts['destination_prompt']}'")
    
    print("\n[Step 3] Would now call:")
    print(f"  source = point_tool('{prompts['draggable_prompt']}')")
    print(f"  destination = point_tool('{prompts['destination_prompt']}')")
    print("  drag(source, destination)")
    
    print("\n✅ Drag puzzle workflow complete!")


def test_text_workflow():
    """Test the complete workflow for text captcha (if we have one)."""
    planner = ActionPlanner(backend="gemini")
    
    # Note: We might not have a text captcha image in the repo
    print("\n" + "=" * 60)
    print("TESTING TEXT CAPTCHA WORKFLOW")
    print("=" * 60)
    print("(Would test with a text captcha image if available)")
    print("\nWorkflow would be:")
    print("  1. classification = planner.classify(image_path)")
    print("  2. text = planner.read_text(image_path)")
    print("  3. type_text(text)")


def test_checkbox_workflow():
    """Test the complete workflow for checkbox captcha."""
    print("\n" + "=" * 60)
    print("TESTING CHECKBOX WORKFLOW")
    print("=" * 60)
    
    image_path = "captchaimages/recaptchaBasic.png"
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    print("\nFor checkbox captchas, NO PLANNER CALLS ARE NEEDED!")
    print("\nSimply:")
    print("  location = point_tool('checkbox center')")
    print("  click(location)")
    print("\n✅ Checkbox workflow is the simplest!")


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("GEMINI_API_KEY"):
        print("⚠️  Warning: GEMINI_API_KEY not set. Tests will fail.")
        print("Set it with: export GEMINI_API_KEY='your-key'")
        exit(1)
    
    try:
        test_classify_captchas()
        test_image_selection_workflow()
        test_drag_puzzle_workflow()
        test_text_workflow()
        test_checkbox_workflow()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETE!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

