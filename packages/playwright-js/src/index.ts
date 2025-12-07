import { Page, Frame, ElementHandle } from 'patchright';
import { solveCaptcha as solveCaptchaCore, CaptchaAction, SolveOptions, applyOverlays } from 'captchakraken';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { findCaptchaFrames, waitForNewCaptcha, DetectedCaptcha } from './detection';
import { extractCaptchaInfo, extractDragElements, Box, InteractableElement, ExtractedInfo } from './extraction';

export { findCaptchaFrames, DetectedCaptcha, extractCaptchaInfo, InteractableElement };

/**
 * Find and solve all captchas on the page, handling popup challenges.
 */
export async function solveCaptcha(
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
 * @param captchaType - Type of captcha (recaptcha, hcaptcha, etc.)
 * @param captchaSubtype - Subtype (checkbox, challenge)
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
    let elements: InteractableElement[] = [];
    let promptText: string | null = null;

    for (let i = 0; i < maxSteps; i++) {
      console.log(`\n${'='.repeat(40)}`);
      console.log(`Step ${i + 1}/${maxSteps}`);
      console.log(`${'='.repeat(40)}`);

      // Extract info and get boxes/elements if we have type info
      if (captchaType && captchaSubtype && 'page' in pageOrFrame) {
        const info = await extractCaptchaInfo(pageOrFrame as Frame, captchaType, captchaSubtype);

        if (info.challengeElementSelector) {
          currentSelector = info.challengeElementSelector;
        }

        // Store extracted data
        boxes = info.boxes || [];
        elements = info.elements || [];
        promptText = info.promptText;

        // Build enhanced prompt
        let enhancedPrompt = prompt;
        if (info.promptText) {
          enhancedPrompt += `\nPrompt Text: ${info.promptText}`;
        }
        if (info.promptImageUrl) {
          enhancedPrompt += `\nPrompt Image URL: ${info.promptImageUrl}`;
        }

        // Check for drag elements
        if (captchaSubtype === 'slider' || captchaSubtype === 'puzzle') {
          const dragInfo = await extractDragElements(pageOrFrame as Frame);
          if (dragInfo.source) {
            elements.push(dragInfo.source);
          }
          if (dragInfo.target) {
            elements.push(dragInfo.target);
          }
        }

        // Update options with extracted info
        options = {
          ...options,
          promptText: info.promptText || undefined,
          promptImageUrl: info.promptImageUrl || undefined,
          challengeElementSelector: info.challengeElementSelector || undefined,
          elements: elements.length > 0 ? elements : undefined,
        };
      }

      const locator = pageOrFrame.locator(currentSelector).first();

      // Ensure visible
      if (await locator.count() === 0) {
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

      // Get action from solver with element info
      const action = await solveCaptchaCore(tmpPath, prompt, {
        ...options,
        elements: elements,
        promptText: promptText || undefined,
      });

      console.log(`Executing action: ${action.action}`);

      if (action.action === 'click') {
        await executeClick(locator, action, elements);
      }
      else if (action.action === 'drag') {
        await executeDrag(pageOrFrame, locator, action);
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
      else if (action.action === 'verify') {
        await executeVerify(pageOrFrame, locator, action, elements);
        // Wait for verification
        await new Promise(r => setTimeout(r, 1000));
      }
      else if (action.action === 'request_updated_image') {
        // Just continue loop
        continue;
      }

      // Small delay between actions
      await new Promise(r => setTimeout(r, 300));
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

/**
 * Execute a click action, supporting multi-click and element IDs.
 */
async function executeClick(
  locator: ReturnType<Page['locator']>,
  action: CaptchaAction,
  elements: InteractableElement[]
): Promise<void> {
  if (action.action !== 'click') return;

  const box = await locator.boundingBox();
  if (!box) {
    console.error("Could not get bounding box for click target");
    return;
  }

  // Use percentages if available - more reliable than pixel coordinates calculated by Python
  if (action.all_coordinates_pct && action.all_coordinates_pct.length > 0) {
    for (const coordPct of action.all_coordinates_pct) {
      const x = box.width * coordPct[0];
      const y = box.height * coordPct[1];
      console.log(`  Clicking pct at (${coordPct[0]}, ${coordPct[1]}) -> (${x}, ${y})`);
      await locator.click({ position: { x, y } });
      await new Promise(r => setTimeout(r, 200));
    }
  }
  // Fallback to all_coordinates (likely pixels)
  else if (action.all_coordinates) {
    for (const coords of action.all_coordinates) {
      console.log(`  Clicking at (${coords[0]}, ${coords[1]})`);
      await locator.click({ position: { x: coords[0], y: coords[1] } });
      await new Promise(r => setTimeout(r, 200));
    }
  }
  // Multi-click via target_ids
  else if (action.target_ids && action.target_ids.length > 0) {
    for (const targetId of action.target_ids) {
      const elem = elements.find(e => e.element_id === targetId);
      if (elem) {
        // element bbox is usually percentage in InteractableElement if normalized? 
        // Need to check how extraction works. 
        // But here we might just click based on relative bbox if available.
        const x = elem.bbox[0] + elem.bbox[2] / 2;
        const y = elem.bbox[1] + elem.bbox[3] / 2;
        console.log(`  Clicking element ${targetId} at (${x}, ${y})`);
        await locator.click({ position: { x, y } });
        await new Promise(r => setTimeout(r, 200));
      }
    }
  }
  // Single click with percentage (Preferred)
  else if (action.point_percent) {
    const x = box.width * action.point_percent[0];
    const y = box.height * action.point_percent[1];
    console.log(`  Clicking at ${action.point_percent[0] * 100}%, ${action.point_percent[1] * 100}% -> (${x}, ${y})`);
    await locator.click({
      position: { x, y }
    });
  }
  // Single click with coordinates (Fallback)
  else if (action.coordinates) {
    console.log(`  Clicking at (${action.coordinates[0]}, ${action.coordinates[1]})`);
    await locator.click({
      position: { x: action.coordinates[0], y: action.coordinates[1] }
    });
  }
}

/**
 * Execute a drag action.
 */
async function executeDrag(
  pageOrFrame: Page | Frame,
  locator: ReturnType<Page['locator']>,
  action: CaptchaAction
): Promise<void> {
  if (action.action !== 'drag') return;

  const box = await locator.boundingBox();
  if (!box) {
    console.error("Could not get bounding box for drag target");
    return;
  }

  let sourceX, sourceY, targetX, targetY;

  // Prefer percentages
  if (action.source_coordinates_pct && action.target_coordinates_pct) {
    sourceX = box.x + (box.width * action.source_coordinates_pct[0]);
    sourceY = box.y + (box.height * action.source_coordinates_pct[1]);
    targetX = box.x + (box.width * action.target_coordinates_pct[0]);
    targetY = box.y + (box.height * action.target_coordinates_pct[1]);
    console.log(`  Dragging using percentages: (${action.source_coordinates_pct}) -> (${action.target_coordinates_pct})`);
  } else {
    // Fallback to absolute pixels (relative to element top-left)
    const source = action.source_coordinates;
    const target = action.target_coordinates;

    if (!source || !target) {
      console.log("  Missing drag coordinates");
      return;
    }

    sourceX = box.x + source[0];
    sourceY = box.y + source[1];
    targetX = box.x + target[0];
    targetY = box.y + target[1];
  }

  console.log(`  Dragging from (${sourceX}, ${sourceY}) to (${targetX}, ${targetY})`);

  // Use mouse on the page
  const page = 'page' in pageOrFrame ? pageOrFrame.page() : pageOrFrame;
  await page.mouse.move(sourceX, sourceY);
  await page.mouse.down();

  // Move in steps for smoother drag
  const steps = 20;
  for (let i = 1; i <= steps; i++) {
    const t = i / steps;
    const currentX = sourceX + (targetX - sourceX) * t;
    const currentY = sourceY + (targetY - sourceY) * t;
    await page.mouse.move(currentX, currentY);
    await new Promise(r => setTimeout(r, 20));
  }

  await page.mouse.up();
}

/**
 * Execute a verify/submit action.
 */
async function executeVerify(
  pageOrFrame: Page | Frame,
  locator: ReturnType<Page['locator']>,
  action: CaptchaAction,
  elements: InteractableElement[]
): Promise<void> {
  if (action.action !== 'verify') return;

  // Try to find verify button from elements
  const verifyElem = elements.find(e =>
    e.element_type === 'verify_button' || e.element_type === 'submit_button'
  );

  if (verifyElem) {
    const x = verifyElem.bbox[0] + verifyElem.bbox[2] / 2;
    const y = verifyElem.bbox[1] + verifyElem.bbox[3] / 2;
    console.log(`  Clicking verify button at (${x}, ${y})`);
    await locator.click({ position: { x, y } });
    return;
  }

  // Fall back to common verify button selectors
  const verifySelectors = [
    '.rc-button-default',
    '#recaptcha-verify-button',
    '.verify-button',
    '.submit-button',
    '.button-submit',
    "button[type='submit']",
  ];

  for (const sel of verifySelectors) {
    const btn = pageOrFrame.locator(sel).first();
    if (await btn.count() > 0 && await btn.isVisible()) {
      console.log(`  Found verify button with selector: ${sel}`);
      await btn.click();
      return;
    }
  }

  console.log("  Could not find verify button");
}
