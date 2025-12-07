# CaptchaKraken Playwright (Python)

Seamless integration of CaptchaKraken with Playwright/Patchright for Python.

## Features
- **Auto-Detection**: Automatically finds captcha frames (reCAPTCHA, hCaptcha, Cloudflare, etc.).
- **Smart Solving**: Uses the core CaptchaKraken AI to solve challenges.
- **Robustness**: Handles checkbox clicks, popup challenges, and verification loops.

## Installation

```bash
pip install captchakraken-playwright
```

## Usage

```python
from patchright.sync_api import sync_playwright
from captchakraken_playwright import solve_captcha

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    
    page.goto("https://example.com/captcha-page")
    
    # Solve all captchas on the page automatically
    success = solve_captcha(page)
    
    if success:
        print("Captcha solved!")
        # Proceed with login/submission...
    else:
        print("Failed to solve captcha.")
        
    browser.close()
```

## API

### `solve_captcha(page, solver=None, **kwargs)`
Main entry point. Automatically detects and solves captchas.
- **page**: Playwright `Page` object.
- **solver**: Optional pre-configured `CaptchaSolver` instance.
- **kwargs**: Arguments passed to `CaptchaSolver` if creating a new one.

### `solve_single_captcha(frame, selector, ...)`
Lower-level function to solve a specific captcha frame if you want more control.

## Requirements
- `captchakraken` core package.
- `patchright` or `playwright`.
