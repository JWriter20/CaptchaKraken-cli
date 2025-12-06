import pytest
import os
from patchright.sync_api import sync_playwright
from captchakraken_playwright.extraction import extract_captcha_info
from captchakraken.overlay import add_overlays_to_image

@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        yield browser
        browser.close()

def test_recaptcha_extraction_local(browser):
    page = browser.new_page()
    try:
        # Construct absolute path to the test file
        test_file_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            "../../tests/html/recaptchaImages/recaptchaImages.html"
        ))
        
        print(f"Loading file: file://{test_file_path}")
        page.goto(f"file://{test_file_path}")
        page.wait_for_load_state("networkidle")
        
        print("\n--- Testing Extraction Logic ---")
        
        # Simulate detection results
        captcha_type = 'recaptcha'
        captcha_subtype = 'challenge'
        target_frame = page.main_frame
        
        print(f"Simulating detection: Type={captcha_type}, Subtype={captcha_subtype}")
        
        # Extract info
        info = extract_captcha_info(target_frame, captcha_type, captcha_subtype)
        
        print("\nExtracted Info:")
        print(f"Prompt Text: {info.get('prompt_text')}")
        print(f"Boxes found: {len(info.get('boxes', []))}")
        
        boxes = info.get('boxes', [])
        assert len(boxes) > 0, "No boxes extracted!"
        
        # Take a screenshot
        screenshot_path = os.path.join(os.path.dirname(__file__), "recaptcha_overlay_test.png")
        page.screenshot(path=screenshot_path)
        print(f"Original screenshot saved to: {screenshot_path}")
        
        # Apply Python-based overlay
        print("Applying Python overlays...")
        add_overlays_to_image(screenshot_path, boxes)
        print(f"Overlay applied to: {screenshot_path}")

    finally:
        page.close()
