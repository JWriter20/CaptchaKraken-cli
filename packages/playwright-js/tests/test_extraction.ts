import { chromium } from 'patchright';
import { extractCaptchaInfo } from '../src/extraction';
import { applyOverlays } from 'captchakraken';
import * as path from 'path';
import * as fs from 'fs';

async function testRecaptchaExtraction() {
  console.log('Starting reCAPTCHA extraction test...\n');

  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();

  try {
    // Construct path to test file
    const testFilePath = path.resolve(__dirname, '../../tests/html/recaptchaImages/recaptchaImages.html');
    console.log(`Loading file: file://${testFilePath}`);

    await page.goto(`file://${testFilePath}`);
    await page.waitForLoadState('networkidle');

    console.log('\n--- Testing Extraction Logic ---');

    // Simulate detection results
    const captchaType = 'recaptcha';
    const captchaSubtype = 'challenge';
    const targetFrame = page.mainFrame();

    console.log(`Simulating detection: Type=${captchaType}, Subtype=${captchaSubtype}`);

    // Extract info
    const info = await extractCaptchaInfo(targetFrame, captchaType, captchaSubtype);

    console.log('\nExtracted Info:');
    console.log(`Prompt Text: ${info.promptText}`);
    console.log(`Prompt Image URL: ${info.promptImageUrl}`);
    console.log(`Challenge Selector: ${info.challengeElementSelector}`);
    console.log(`Boxes found: ${info.boxes.length}`);

    // Verify boxes
    if (info.boxes.length === 0) {
      throw new Error('No boxes extracted!');
    }

    console.log('\nBox details:');
    info.boxes.forEach((box, i) => {
      console.log(`  Box ${i + 1}: text="${box.text}", bbox=[${box.bbox.join(', ')}]`);
    });

    // Take screenshot
    const screenshotPath = path.join(__dirname, 'recaptcha_overlay_test.png');
    await page.screenshot({ path: screenshotPath });
    console.log(`\nOriginal screenshot saved to: ${screenshotPath}`);

    // Apply Python overlays
    console.log('Applying Python overlays...');
    await applyOverlays(screenshotPath, info.boxes);
    console.log(`Overlay applied to: ${screenshotPath}`);

    // Verify prompt text contains expected content
    if (!info.promptText?.includes('Select all images with')) {
      throw new Error('Prompt text not extracted correctly');
    }

    if (!info.promptText?.includes('bus')) {
      throw new Error('Prompt should contain "bus"');
    }

    console.log('\n✓ All tests passed!');

  } catch (error) {
    console.error('\n✗ Test failed:', error);
    process.exit(1);
  } finally {
    await browser.close();
  }
}

testRecaptchaExtraction();

