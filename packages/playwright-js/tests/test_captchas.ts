import { chromium } from 'patchright';
import { findCaptchaFrames, solveCaptchasOnPage } from '../src/index';

async function testHCaptcha() {
  console.log('\nTesting hCaptcha Detection...');
  // Headless true for CI environment
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  
  try {
    await page.goto('https://democaptcha.com/demo-form-eng/hcaptcha.html');
    await page.waitForLoadState('networkidle');
    
    const captchas = await findCaptchaFrames(page);
    console.log(`Found ${captchas.length} frames.`);
    
    if (captchas.some(c => c.type === 'hcaptcha')) {
      console.log('✅ Identified hCaptcha');
    } else {
      console.error('❌ Failed to identify hCaptcha');
    }

    const checkbox = captchas.find(c => c.subtype === 'checkbox');
    if (checkbox) {
        console.log('✅ Found Checkbox iframe');
        // Simulate click to trigger popup check
        const box = checkbox.frame.locator('#checkbox');
        if (await box.isVisible()) {
            await box.click();
            console.log('Clicked checkbox, waiting for popup...');
            await page.waitForTimeout(3000);
            
            const captchasAfter = await findCaptchaFrames(page);
            const challenge = captchasAfter.find(c => c.subtype === 'challenge');
            if (challenge) {
                console.log('✅ Found Challenge popup');
            } else {
                console.log('⚠️ No challenge popup found (might be one-click solve)');
            }
        }
    }

  } catch (e) {
    console.error('Error:', e);
  } finally {
    await browser.close();
  }
}

async function testReCaptcha() {
  console.log('\nTesting reCaptcha Detection...');
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  
  try {
    await page.goto('https://www.google.com/recaptcha/api2/demo');
    await page.waitForLoadState('networkidle');
    
    const captchas = await findCaptchaFrames(page);
    console.log(`Found ${captchas.length} frames.`);
    
    if (captchas.some(c => c.type === 'recaptcha')) {
      console.log('✅ Identified reCaptcha');
    } else {
      console.error('❌ Failed to identify reCaptcha');
    }
  } catch (e) {
    console.error('Error:', e);
  } finally {
    await browser.close();
  }
}

async function testCloudflare() {
  console.log('\nTesting Cloudflare Detection...');
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  
  try {
    await page.goto('https://2captcha.com/demo/cloudflare-turnstile');
    await page.waitForTimeout(3000);
    
    const captchas = await findCaptchaFrames(page);
    console.log(`Found ${captchas.length} frames.`);
    
    if (captchas.some(c => c.type === 'cloudflare')) {
      console.log('✅ Identified Cloudflare');
    } else {
      console.error('❌ Failed to identify Cloudflare');
    }
  } catch (e) {
    console.error('Error:', e);
  } finally {
    await browser.close();
  }
}

async function runTests() {
  await testHCaptcha();
  await testReCaptcha();
  await testCloudflare();
}

runTests();

