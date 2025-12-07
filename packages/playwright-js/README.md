# CaptchaKraken Playwright (JS/TS)

Seamless integration of CaptchaKraken with Playwright/Patchright for TypeScript and JavaScript.

## Features
- **Auto-Detection**: Automatically finds captcha frames (reCAPTCHA, hCaptcha, Cloudflare, etc.).
- **Smart Solving**: Uses the core CaptchaKraken AI to solve challenges.
- **Robustness**: Handles checkbox clicks, popup challenges, and verification loops.

## Installation

```bash
npm install captchakraken-playwright
# Ensure you have the core Python package installed as well!
```

## Usage

```typescript
import { chromium } from 'playwright'; // or 'patchright'
import { solveCaptcha } from 'captchakraken-playwright';

(async () => {
  const browser = await chromium.launch({ headless: false });
  const page = await browser.newPage();
  
  await page.goto('https://example.com/captcha-page');

  // Solve all captchas on the page automatically
  const success = await solveCaptcha(page);
  
  if (success) {
    console.log('Captcha solved!');
    // Proceed with login/submission...
  } else {
    console.log('Failed to solve captcha.');
  }

  await browser.close();
})();
```

## API

### `solveCaptcha(page, options?)`
Main entry point. Automatically detects and solves captchas.
- **page**: Playwright `Page` object.
- **options**: Optional configuration.

### `solveCaptchaLoop(frame, selector, prompt, ...)`
Lower-level function to solve a specific captcha frame if you want more control.

## Requirements
- Python environment with `captchakraken` core installed.
- Node.js 18+.

