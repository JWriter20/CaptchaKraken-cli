/**
 * Extraction utilities for captcha information from Playwright frames.
 * 
 * Extracts:
 * - Prompt text (instructions)
 * - Prompt image URL (if any)
 * - Challenge element selector
 * - Numbered interactable elements with bounding boxes
 */

import { Frame } from 'patchright';

export interface Box {
  text: string;
  bbox: [number, number, number, number]; // x, y, width, height
}

export interface InteractableElement {
  element_id: number;
  text: string;
  bbox: [number, number, number, number];
  element_type: string;
  is_selected?: boolean;
  has_animation?: boolean;
}

export interface ExtractedInfo {
  promptText: string | null;
  promptImageUrl: string | null;
  challengeElementSelector: string | null;
  boxes: Box[];
  elements: InteractableElement[];
  captchaType?: string;
  captchaSubtype?: string;
}

export interface DragElements {
  source: InteractableElement | null;
  target: InteractableElement | null;
}

/**
 * Extract detailed information from a captcha frame.
 */
export async function extractCaptchaInfo(
  frame: Frame,
  type: string,
  subtype: string
): Promise<ExtractedInfo> {
  const info: ExtractedInfo = {
    promptText: null,
    promptImageUrl: null,
    challengeElementSelector: null,
    boxes: [],
    elements: [],
    captchaType: type,
    captchaSubtype: subtype,
  };

  try {
    if (type === 'recaptcha' && subtype === 'challenge') {
      await extractRecaptchaChallenge(frame, info);
    } else if (type === 'recaptcha' && subtype === 'checkbox') {
      await extractRecaptchaCheckbox(frame, info);
    } else if (type === 'hcaptcha' && subtype === 'challenge') {
      await extractHcaptchaChallenge(frame, info);
    } else if (type === 'hcaptcha' && subtype === 'checkbox') {
      await extractHcaptchaCheckbox(frame, info);
    } else if (type === 'cloudflare') {
      await extractCloudflare(frame, info);
    } else {
      await extractGeneric(frame, info);
    }
  } catch (e) {
    console.error("Error extracting captcha info:", e);
  }

  return info;
}

/**
 * Extract info from reCAPTCHA image challenge.
 */
async function extractRecaptchaChallenge(frame: Frame, info: ExtractedInfo): Promise<void> {
  // Prompt text - try multiple selectors
  const promptSelectors = [
    '.rc-imageselect-desc-no-canonical',
    '.rc-imageselect-instructions',
    '.rc-imageselect-desc',
  ];

  for (const selector of promptSelectors) {
    const instructions = await frame.$(selector);
    if (instructions) {
      const text = await instructions.textContent();
      if (text) {
        info.promptText = text.trim();
        break;
      }
    }
  }

  // Prompt image (some reCAPTCHA show example images)
  const promptImg = await frame.$('.rc-imageselect-desc-wrapper img, .rc-imageselect-example img');
  if (promptImg) {
    info.promptImageUrl = await promptImg.getAttribute('src');
  }

  // Challenge element selector
  if (await frame.$('.rc-imageselect-challenge')) {
    info.challengeElementSelector = '.rc-imageselect-challenge';
  } else if (await frame.$('.rc-imageselect-payload')) {
    info.challengeElementSelector = '.rc-imageselect-payload';
  } else {
    info.challengeElementSelector = 'body';
  }

  // Extract numbered tiles
  const elementsData = await frame.evaluate(() => {
    const tiles = document.querySelectorAll('td.rc-imageselect-tile');
    const results: InteractableElement[] = [];

    tiles.forEach((tile, index) => {
      const target = tile.querySelector('.rc-image-tile-wrapper') || tile;
      const rect = target.getBoundingClientRect();
      const isSelected = tile.classList.contains('rc-imageselect-tileselected');

      results.push({
        element_id: index + 1,
        text: (index + 1).toString(),
        bbox: [rect.x, rect.y, rect.width, rect.height] as [number, number, number, number],
        element_type: 'image_tile',
        is_selected: isSelected,
        has_animation: tile.querySelector('.rc-image-tile-overlay') !== null,
      });
    });

    // Also check for verify button
    const verifyBtn = document.querySelector('.rc-button-default, #recaptcha-verify-button');
    if (verifyBtn) {
      const rect = verifyBtn.getBoundingClientRect();
      results.push({
        element_id: results.length + 1,
        text: 'VERIFY',
        bbox: [rect.x, rect.y, rect.width, rect.height] as [number, number, number, number],
        element_type: 'verify_button',
        is_selected: false,
      });
    }

    return results;
  });

  info.elements = elementsData;
  info.boxes = elementsData.map(e => ({ text: e.text, bbox: e.bbox }));
}

/**
 * Extract info from reCAPTCHA checkbox.
 */
async function extractRecaptchaCheckbox(frame: Frame, info: ExtractedInfo): Promise<void> {
  info.promptText = "Click the checkbox to verify you're human";
  info.challengeElementSelector = '.recaptcha-checkbox-border, #recaptcha-anchor';

  const elementsData = await frame.evaluate(() => {
    const checkbox = document.querySelector('.recaptcha-checkbox-border, #recaptcha-anchor');
    if (checkbox) {
      const rect = checkbox.getBoundingClientRect();
      const isChecked = checkbox.getAttribute('aria-checked') === 'true';
      return [{
        element_id: 1,
        text: '1',
        bbox: [rect.x, rect.y, rect.width, rect.height] as [number, number, number, number],
        element_type: 'checkbox',
        is_selected: isChecked,
      }];
    }
    return [];
  });

  info.elements = elementsData;
  info.boxes = elementsData.map(e => ({ text: e.text, bbox: e.bbox }));
}

/**
 * Extract info from hCaptcha image challenge.
 */
async function extractHcaptchaChallenge(frame: Frame, info: ExtractedInfo): Promise<void> {
  // Prompt text
  const prompt = await frame.$('.prompt-text');
  if (prompt) {
    info.promptText = await prompt.textContent();
  }

  // Prompt image (usually in .prompt-image with background-image)
  const pImg = await frame.$('.prompt-image');
  if (pImg) {
    const style = await pImg.getAttribute('style');
    if (style && style.includes('url(')) {
      const match = style.match(/url\(["\']?(.+?)["\']?\)/);
      if (match) info.promptImageUrl = match[1];
    } else {
      info.promptImageUrl = await pImg.getAttribute('src');
    }
  }

  // Challenge element selector
  if (await frame.$('.challenge-view')) {
    info.challengeElementSelector = '.challenge-view';
  } else if (await frame.$('.task-grid')) {
    info.challengeElementSelector = '.task-grid';
  } else if (await frame.$('.image-grid')) {
    info.challengeElementSelector = '.image-grid';
  } else {
    info.challengeElementSelector = 'body';
  }

  // Extract task images
  const elementsData = await frame.evaluate(() => {
    // Try multiple selectors for hCaptcha grid images
    let images = document.querySelectorAll('.task-image');
    if (images.length === 0) {
      images = document.querySelectorAll('.image-grid .image, .task-grid .task');
    }

    const results: InteractableElement[] = [];

    images.forEach((img, index) => {
      const rect = img.getBoundingClientRect();
      const isSelected = img.classList.contains('selected') ||
        img.closest('.selected') !== null ||
        img.getAttribute('aria-pressed') === 'true';

      results.push({
        element_id: index + 1,
        text: (index + 1).toString(),
        bbox: [rect.x, rect.y, rect.width, rect.height] as [number, number, number, number],
        element_type: 'image_tile',
        is_selected: isSelected,
      });
    });

    // Check for submit button
    const submitBtn = document.querySelector('.submit-button, .button-submit, [data-action="submit"]');
    if (submitBtn) {
      const rect = submitBtn.getBoundingClientRect();
      results.push({
        element_id: results.length + 1,
        text: 'SUBMIT',
        bbox: [rect.x, rect.y, rect.width, rect.height] as [number, number, number, number],
        element_type: 'submit_button',
        is_selected: false,
      });
    }

    // Check for skip button
    const skipBtn = document.querySelector('.skip-button, .button-skip, [data-action="skip"]');
    if (skipBtn) {
      const rect = skipBtn.getBoundingClientRect();
      results.push({
        element_id: results.length + 1,
        text: 'SKIP',
        bbox: [rect.x, rect.y, rect.width, rect.height] as [number, number, number, number],
        element_type: 'skip_button',
        is_selected: false,
      });
    }

    return results;
  });

  info.elements = elementsData;
  info.boxes = elementsData.map(e => ({ text: e.text, bbox: e.bbox }));
}

/**
 * Extract info from hCaptcha checkbox.
 */
async function extractHcaptchaCheckbox(frame: Frame, info: ExtractedInfo): Promise<void> {
  info.promptText = "Click the checkbox to verify you're human";
  info.challengeElementSelector = '#checkbox, .checkbox';

  const elementsData = await frame.evaluate(() => {
    const checkbox = document.querySelector('#checkbox, .checkbox, [role="checkbox"]');
    if (checkbox) {
      const rect = checkbox.getBoundingClientRect();
      const isChecked = checkbox.getAttribute('aria-checked') === 'true';
      return [{
        element_id: 1,
        text: '1',
        bbox: [rect.x, rect.y, rect.width, rect.height] as [number, number, number, number],
        element_type: 'checkbox',
        is_selected: isChecked,
      }];
    }
    return [];
  });

  info.elements = elementsData;
  info.boxes = elementsData.map(e => ({ text: e.text, bbox: e.bbox }));
}

/**
 * Extract info from Cloudflare Turnstile.
 */
async function extractCloudflare(frame: Frame, info: ExtractedInfo): Promise<void> {
  info.promptText = "Verify you are human";
  info.challengeElementSelector = 'body';

  const elementsData = await frame.evaluate(() => {
    const results: InteractableElement[] = [];

    // Cloudflare checkbox/challenge area
    const checkbox = document.querySelector('input[type="checkbox"], .cf-turnstile, #turnstile-wrapper') as HTMLInputElement | null;
    if (checkbox) {
      const rect = checkbox.getBoundingClientRect();
      results.push({
        element_id: 1,
        text: '1',
        bbox: [rect.x, rect.y, rect.width, rect.height] as [number, number, number, number],
        element_type: 'checkbox',
        is_selected: checkbox.checked || false,
      });
    }

    // Also look for any clickable areas
    const clickable = document.querySelector('[role="button"], button, .challenge-button');
    if (clickable && clickable !== checkbox) {
      const rect = clickable.getBoundingClientRect();
      results.push({
        element_id: results.length + 1,
        text: (results.length + 1).toString(),
        bbox: [rect.x, rect.y, rect.width, rect.height] as [number, number, number, number],
        element_type: 'button',
        is_selected: false,
      });
    }

    return results;
  });

  info.elements = elementsData;
  info.boxes = elementsData.map(e => ({ text: e.text, bbox: e.bbox }));
}

/**
 * Generic extraction for unknown captcha types.
 */
async function extractGeneric(frame: Frame, info: ExtractedInfo): Promise<void> {
  info.challengeElementSelector = 'body';

  // Try to find any visible text that looks like instructions
  const textContent = await frame.evaluate(() => {
    const textElements = document.querySelectorAll('h1, h2, h3, p, .prompt, .instructions, .title');
    for (const el of textElements) {
      const text = el.textContent?.trim();
      if (text && text.length > 10 && text.length < 200) {
        return text;
      }
    }
    return null;
  });

  if (textContent) {
    info.promptText = textContent;
  }

  // Find all clickable/interactive elements
  const elementsData = await frame.evaluate(() => {
    const clickables = document.querySelectorAll(
      'button, input[type="checkbox"], input[type="submit"], ' +
      '[role="button"], [onclick], .clickable, img[style*="cursor"], ' +
      '[draggable="true"], .draggable'
    );

    const results: InteractableElement[] = [];
    let idx = 1;

    clickables.forEach((el) => {
      const rect = el.getBoundingClientRect();
      // Skip tiny or invisible elements
      if (rect.width < 10 || rect.height < 10) return;
      if (rect.width === 0 || rect.height === 0) return;

      const isDraggable = (el as HTMLElement).draggable || el.classList.contains('draggable');

      results.push({
        element_id: idx,
        text: idx.toString(),
        bbox: [rect.x, rect.y, rect.width, rect.height] as [number, number, number, number],
        element_type: isDraggable ? 'draggable' : 'clickable',
        is_selected: false,
      });
      idx++;
    });

    return results;
  });

  info.elements = elementsData;
  info.boxes = elementsData.map(e => ({ text: e.text, bbox: e.bbox }));
}

/**
 * Specifically extract drag-related elements for slider/puzzle captchas.
 */
export async function extractDragElements(frame: Frame): Promise<DragElements> {
  return await frame.evaluate(() => {
    const result: DragElements = { source: null, target: null };

    // Look for draggable elements
    const draggable = document.querySelector(
      '[draggable="true"], .puzzle-piece, .slider-handle, ' +
      '.drag-source, .draggable, .captcha-slider'
    );

    if (draggable) {
      const rect = draggable.getBoundingClientRect();
      result.source = {
        element_id: 1,
        text: '1',
        bbox: [rect.x, rect.y, rect.width, rect.height] as [number, number, number, number],
        element_type: 'draggable',
      };
    }

    // Look for drop targets
    const dropTarget = document.querySelector(
      '.drop-target, .puzzle-slot, .slider-track, ' +
      '.drag-target, .destination, .captcha-track'
    );

    if (dropTarget) {
      const rect = dropTarget.getBoundingClientRect();
      result.target = {
        element_id: 2,
        text: '2',
        bbox: [rect.x, rect.y, rect.width, rect.height] as [number, number, number, number],
        element_type: 'drop_target',
      };
    }

    return result;
  });
}
