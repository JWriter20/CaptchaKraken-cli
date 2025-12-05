# CaptchaKraken Monorepo

This repository contains the CaptchaKraken ecosystem, split into isolated modules:

## Packages

### 1. Core (`packages/core`)
The core CaptchaKraken library containing the AI solver logic and generic solvers.
- **Python**: `captchakraken` (PyPI)
- **TypeScript**: `captchakraken` (NPM) - Wrapper around the Python CLI.

### 2. Playwright Python (`packages/playwright-py`)
Integration with Playwright/Patchright for Python.
- Uses `patchright` and `captchakraken` core.
- Provides `solve_captcha(page, selector)` helper.

### 3. Playwright TypeScript (`packages/playwright-js`)
Integration with Playwright/Patchright for TypeScript/JavaScript.
- Uses `patchright` and `captchakraken` core.
- Provides `solveCaptchaLoop(page, selector)` helper.

## Setup

Each package is isolated. See their respective directories for installation and usage instructions.

