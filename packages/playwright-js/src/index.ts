import { Page, Frame, ElementHandle } from 'patchright';
import { solveCaptcha, CaptchaAction, SolveOptions, applyOverlays } from 'captchakraken';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { findCaptchaFrames, waitForNewCaptcha, DetectedCaptcha } from './detection';
import { extractCaptchaInfo, Box } from './extraction';

export { findCaptchaFrames, DetectedCaptcha };

/**
 * Find and solve all captchas on the page, handling popup challenges.
 */
export async function solveCaptchasOnPage(
  page: Page,
  options: SolveOptions = {}
): Promise<boolean> {
  while (true) {
    // 1. Find captchas
    const captchas = await findCaptchaFrames(page);
    if (captchas.length === 0) {
      console.log("No captchas found.");
      return true;
    }

    console.log(`Found ${captchas.length} captcha frames.`);

    // 2. Prioritize
    let target: DetectedCaptcha | undefined;

    // Look for visible challenges first
    for (const c of captchas) {
      if (c.subtype === 'challenge') {
        const body = c.frame.locator('body').first();
        if (await body.isVisible()) {
          target = c;
          break;
        }
      }
    }

    // Then checkboxes
    if (!target) {
      target = captchas.find(c => c.subtype === 'checkbox');
    }

    // Fallback
    if (!target) {
      target = captchas[0];
    }

    console.log(`Targeting captcha: ${target.type} (${target.subtype})`);

    // 3. Solve
    const success = await solveCaptchaLoop(
      target.frame,
      'body',
      `Solve this ${target.type} ${target.subtype}`,
      options,
      target.subtype === 'checkbox' ? 5 : 15,
      target.type,
      target.subtype
    );

    if (!success) {
      console.log("Failed to solve target.");
      return false;
    }

    // 4. Check for new challenges if we solved a checkbox
    if (target.subtype === 'checkbox') {
      console.log("Solved checkbox, checking for new challenges...");
      const knownFrames = page.frames();
      const newFrame = await waitForNewCaptcha(page, knownFrames, 3000);

      if (newFrame) {
        console.log("New challenge appeared!");
        continue;
      } else {
        console.log("No new challenge appeared. Assuming done.");
        return true;
      }
    } else {
      return true;
    }
  }
}

/**
 * Solve a captcha on a Playwright/Patchright page or frame.
 * 
 * @param pageOrFrame - The page or frame containing the captcha
 * @param selector - CSS selector for the captcha container element
 * @param prompt - Context/instructions for solving
 * @param options - Solver configuration options
 * @param maxSteps - Maximum steps to attempt
 * @returns True if solved, False if failed or max steps reached
 */
export async function solveCaptchaLoop(
  pageOrFrame: Page | Frame,
  selector: string,
  prompt: string = "Solve this captcha",
  options: SolveOptions = {},
  maxSteps: number = 10,
  captchaType?: string,
  captchaSubtype?: string
): Promise<boolean> {
  const tmpDir = os.tmpdir();
  const tmpPath = path.join(tmpDir, `captcha-${Date.now()}.png`);

  try {
    let currentSelector = selector;
    let boxes: Box[] = [];

    for (let i = 0; i < maxSteps; i++) {
      // Extract info and get boxes if we have type info
      if (captchaType && captchaSubtype && 'page' in pageOrFrame) {
        const info = await extractCaptchaInfo(pageOrFrame as Frame, captchaType, captchaSubtype);

        if (info.challengeElementSelector) {
          currentSelector = info.challengeElementSelector;
        }

        // Store boxes for overlay
        boxes = info.boxes || [];

        // Merge options
        options = {
          ...options,
          promptText: info.promptText || undefined,
          promptImageUrl: info.promptImageUrl || undefined,
          challengeElementSelector: info.challengeElementSelector || undefined
        };
      }

      const locator = pageOrFrame.locator(currentSelector).first();

      // Ensure visible
      if (await locator.count() === 0) {
        // Fallback to original selector if new one fails?
        // Or just fail this step and retry/continue
        if (currentSelector !== selector) {
          currentSelector = selector;
          continue;
        }
        throw new Error(`Element not found: ${currentSelector}`);
      }

      await locator.waitFor({ state: 'visible', timeout: 5000 });

      // Take screenshot
      await locator.screenshot({ path: tmpPath });

      // Apply overlays if we have boxes
      if (boxes.length > 0) {
        console.log(`Applying overlays for ${boxes.length} elements...`);
        await applyOverlays(tmpPath, boxes, options.pythonPath);
      }

      // Get action from solver
      const action = await solveCaptcha(tmpPath, prompt, options);

      console.log(`Step ${i + 1}/${maxSteps}: Executing ${action.action}`);

      if (action.action === 'click') {
        // Coordinates are relative to the screenshot (element)
        await locator.click({
          position: { x: action.coordinates[0], y: action.coordinates[1] }
        });
      }
      else if (action.action === 'drag') {
        const source = action.source_coordinates;
        const target = action.target_coordinates;

        const box = await locator.boundingBox();
        if (box) {
          const absSx = box.x + source[0];
          const absSy = box.y + source[1];
          const absTx = box.x + target[0];
          const absTy = box.y + target[1];

          // Use mouse on the page
          const page = 'page' in pageOrFrame ? pageOrFrame.page() : pageOrFrame;
          await page.mouse.move(absSx, absSy);
          await page.mouse.down();
          await page.mouse.move(absTx, absTy, { steps: 10 });
          await page.mouse.up();
        }
      }
      else if (action.action === 'type') {
        await locator.type(action.text);
      }
      else if (action.action === 'wait') {
        if (action.duration_ms === 0) {
          console.log("Solver indicated completion.");
          return true;
        }
        await new Promise(r => setTimeout(r, action.duration_ms));
      }
      else if (action.action === 'request_updated_image') {
        // Just continue loop
        continue;
      }
    }

    console.log("Max steps reached.");
    return false;

  } catch (error) {
    console.error("Error solving captcha:", error);
    return false;
  } finally {
    // Cleanup
    if (fs.existsSync(tmpPath)) {
      fs.unlinkSync(tmpPath);
    }
  }
}
