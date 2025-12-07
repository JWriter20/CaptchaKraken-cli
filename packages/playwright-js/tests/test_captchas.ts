import { chromium } from 'patchright';
import { findCaptchaFrames, solveCaptcha } from '../src/index';
import * as path from 'path';
import * as dotenv from 'dotenv';

// Load environment variables from the root .env file
dotenv.config({ path: path.resolve(__dirname, '../../../.env') });

const SOLVER_OPTIONS = {
  planner: 'gemini' as const,
  geminiApiKey: process.env.GEMINI_API_KEY,
  plannerModel: 'gemini-2.0-flash',
  pythonPath: path.resolve(__dirname, '../../../venv/bin/python')
};

if (!process.env.GEMINI_API_KEY) {
  console.warn("WARNING: GEMINI_API_KEY is not set. Solver might fail if using gemini backend.");
}

async function testHCaptcha() {
  console.log('\nTesting hCaptcha Detection & Solving...');
  // Headed mode as requested
  const browser = await chromium.launch({ headless: false });
  const page = await browser.newPage();

  try {
    await page.goto('https://democaptcha.com/demo-form-eng/hcaptcha.html');
    await page.waitForLoadState('networkidle');

    console.log("Solving hCaptcha...");
    const success = await solveCaptcha(page, SOLVER_OPTIONS);

    if (success) {
      console.log('✅ Successfully solved hCaptcha');
    } else {
      console.error('❌ Failed to solve hCaptcha');
    }

  } catch (e) {
    console.error('Error:', e);
  } finally {
    await browser.close();
  }
}

async function testReCaptcha() {
  console.log('\nTesting reCaptcha Detection & Solving...');
  const browser = await chromium.launch({ headless: false });
  const page = await browser.newPage();

  try {
    await page.goto('https://www.google.com/recaptcha/api2/demo');
    await page.waitForLoadState('networkidle');

    console.log("Solving reCaptcha...");
    const success = await solveCaptcha(page, SOLVER_OPTIONS);

    if (success) {
      console.log('✅ Successfully solved reCaptcha');
    } else {
      console.error('❌ Failed to solve reCaptcha');
    }
  } catch (e) {
    console.error('Error:', e);
  } finally {
    await browser.close();
  }
}

async function testCloudflare() {
  console.log('\nTesting Cloudflare Detection & Solving...');
  const browser = await chromium.launch({ headless: false });
  const page = await browser.newPage();

  try {
    await page.goto('https://2captcha.com/demo/cloudflare-turnstile');
    // Wait for cloudflare to appear
    await page.waitForTimeout(3000);

    console.log("Solving Cloudflare...");
    const success = await solveCaptcha(page, SOLVER_OPTIONS);

    if (success) {
      console.log('✅ Successfully solved Cloudflare');
    } else {
      console.error('❌ Failed to solve Cloudflare');
    }
  } catch (e) {
    console.error('Error:', e);
  } finally {
    await browser.close();
  }
}

async function runTests() {
  // await testHCaptcha();
  await testReCaptcha();
  // await testCloudflare();
}

runTests();
