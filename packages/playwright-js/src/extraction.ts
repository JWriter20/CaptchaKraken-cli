import { Frame } from 'patchright';

export interface Box {
  text: string;
  bbox: [number, number, number, number]; // x, y, width, height
}

export interface ExtractedInfo {
  promptText: string | null;
  promptImageUrl: string | null;
  challengeElementSelector: string | null;
  boxes: Box[];
}

export async function extractCaptchaInfo(frame: Frame, type: string, subtype: string): Promise<ExtractedInfo> {
  let promptText: string | null = null;
  let promptImageUrl: string | null = null;
  let challengeElementSelector: string | null = null;
  let boxes: Box[] = [];

  try {
    if (type === 'recaptcha' && subtype === 'challenge') {
      // ReCAPTCHA
      // Prompt text
      const instructions = await frame.$('.rc-imageselect-desc-no-canonical, .rc-imageselect-instructions');
      if (instructions) {
        promptText = await instructions.textContent();
      }

      // Prompt image - usually none for reCAPTCHA (text based), but check for optional image
      const promptImg = await frame.$('.rc-imageselect-desc-wrapper img');
      if (promptImg) {
        promptImageUrl = await promptImg.getAttribute('src');
      }

      // Main challenge element - use the payload wrapper or challenge wrapper
      const challenge = await frame.$('.rc-imageselect-challenge');
      if (challenge) {
        challengeElementSelector = '.rc-imageselect-challenge';
      } else {
        challengeElementSelector = '.rc-imageselect-payload'; // Fallback
      }

      // Collect bounding boxes
      boxes = await frame.evaluate(() => {
        const tiles = document.querySelectorAll('td.rc-imageselect-tile');
        const results: { text: string; bbox: [number, number, number, number] }[] = [];
        tiles.forEach((tile, index) => {
          const target = tile.querySelector('.rc-image-tile-wrapper') || tile;
          const rect = target.getBoundingClientRect();
          results.push({
            text: (index + 1).toString(),
            bbox: [rect.x, rect.y, rect.width, rect.height]
          });
        });
        return results;
      });

    } else if (type === 'hcaptcha' && subtype === 'challenge') {
      // hCaptcha
      // Prompt
      const prompt = await frame.$('.prompt-text');
      if (prompt) {
        promptText = await prompt.textContent();
      }

      // Prompt Image
      const pImg = await frame.$('.prompt-image');
      if (pImg) {
        // Could be background image
        const style = await pImg.getAttribute('style');
        if (style && style.includes('url(')) {
          // extract url
          const match = style.match(/url\("?(.+?)"?\)/);
          if (match) promptImageUrl = match[1];
        } else {
          promptImageUrl = await pImg.getAttribute('src');
        }
      }

      // Main Challenge
      if (await frame.$('.challenge-view')) {
        challengeElementSelector = '.challenge-view';
      } else if (await frame.$('.image-grid')) {
        challengeElementSelector = '.image-grid';
      } else if (await frame.$('.task-image')) {
        // If just task images found but no container known, try to find a container
        challengeElementSelector = 'body';
      }

      // Collect bounding boxes
      boxes = await frame.evaluate(() => {
        const gridImages = document.querySelectorAll('.task-image .image, .task-image');
        const results: { text: string; bbox: [number, number, number, number] }[] = [];
        if (gridImages.length > 0) {
          gridImages.forEach((img, index) => {
            const rect = img.getBoundingClientRect();
            results.push({
              text: (index + 1).toString(),
              bbox: [rect.x, rect.y, rect.width, rect.height]
            });
          });
        }
        return results;
      });
    }
  } catch (e) {
    console.error("Error extracting captcha info:", e);
  }

  return { promptText, promptImageUrl, challengeElementSelector, boxes };
}
