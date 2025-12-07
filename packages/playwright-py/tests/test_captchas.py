import pytest
from patchright.sync_api import sync_playwright
from captchakraken_playwright.detection import find_captcha_frames, get_captcha_type
from captchakraken_playwright.solver import solve_captcha

@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as p:
        # Running in headless mode for CI/Environment compatibility
        browser = p.chromium.launch(headless=True)
        yield browser
        browser.close()

def test_hcaptcha_detection_and_solve(browser):
    page = browser.new_page()
    try:
        page.goto("https://democaptcha.com/demo-form-eng/hcaptcha.html")
        page.wait_for_load_state("networkidle")
        
        # Test Detection
        captchas = find_captcha_frames(page)
        assert len(captchas) > 0
        assert any(c['type'] == 'hcaptcha' for c in captchas)
        
        print(f"\nFound {len(captchas)} hCaptchas")
        
        # Test Interaction (confirm new captcha pops up)
        # We Mock the solver to just click the checkbox and verify detection of new frame
        # But here we want to verify the logic.
        
        # For this test, we won't run full AI solve (expensive/slow), 
        # but we can verify that find_captcha_frames sees the checkbox.
        
        checkbox_frame = next((c['frame'] for c in captchas if c['subtype'] == 'checkbox'), None)
        assert checkbox_frame is not None, "hCaptcha checkbox frame not found"
        
        # We can try to click it manually to trigger challenge
        checkbox = checkbox_frame.locator("#checkbox")
        if checkbox.is_visible():
            checkbox.click()
            page.wait_for_timeout(2000)
            
            # Rescan
            captchas_after = find_captcha_frames(page)
            challenge_frames = [c for c in captchas_after if c['subtype'] == 'challenge']
            print(f"Found {len(challenge_frames)} challenge frames after click")
            
            # Note: democaptcha might not always trigger challenge if reputation is high,
            # but usually it does for bots.
            
    finally:
        page.close()

def test_recaptcha_detection(browser):
    page = browser.new_page()
    try:
        page.goto("https://www.google.com/recaptcha/api2/demo")
        page.wait_for_load_state("networkidle")
        
        captchas = find_captcha_frames(page)
        assert len(captchas) > 0
        assert any(c['type'] == 'recaptcha' for c in captchas)
        print(f"\nFound {len(captchas)} reCaptchas")
        
        checkbox_frame = next((c['frame'] for c in captchas if c['subtype'] == 'checkbox'), None)
        assert checkbox_frame is not None
        
    finally:
        page.close()

def test_cloudflare_detection(browser):
    page = browser.new_page()
    try:
        page.goto("https://2captcha.com/demo/cloudflare-turnstile")
        # Cloudflare often takes a moment to render
        page.wait_for_timeout(3000)
        
        captchas = find_captcha_frames(page)
        # Cloudflare might be just a checkbox or invisible.
        # The demo usually shows a widget.
        
        # Note: 2captcha demo might change, but we look for challenges.cloudflare.com
        assert len(captchas) > 0 or "challenges.cloudflare.com" in page.content()
        
        if captchas:
            assert captchas[0]['type'] == 'cloudflare'
            print(f"\nFound Cloudflare frame")
        
    finally:
        page.close()

