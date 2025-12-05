import { Page, Frame } from 'patchright';

const CAPTCHA_SIGNATURES: Record<string, string[]> = {
  "hcaptcha": ["hcaptcha.com", "assets.hcaptcha.com"],
  "recaptcha": ["google.com/recaptcha", "gstatic.com/recaptcha"],
  "cloudflare": ["challenges.cloudflare.com"],
};

export interface DetectedCaptcha {
  type: string;
  subtype: 'checkbox' | 'challenge' | 'generic';
  frame: Frame;
}

export function getCaptchaType(url: string): string | null {
  for (const [type, domains] of Object.entries(CAPTCHA_SIGNATURES)) {
    if (domains.some(domain => url.includes(domain))) {
      return type;
    }
  }
  return null;
}

export async function isFrameVisible(frame: Frame): Promise<boolean> {
  try {
    if (frame.isDetached()) return false;

    const element = await frame.frameElement();
    if (!await element.isVisible()) return false;

    const box = await element.boundingBox();
    if (!box || box.width === 0 || box.height === 0) return false;

    return true;
  } catch {
    return false;
  }
}

export async function findCaptchaFrames(page: Page): Promise<DetectedCaptcha[]> {
  const captchas: DetectedCaptcha[] = [];

  for (const frame of page.frames()) {
    if (!await isFrameVisible(frame)) continue;

    const type = getCaptchaType(frame.url());
    if (type) {
      let subtype: DetectedCaptcha['subtype'] = 'generic';
      const url = frame.url();

      if (type === 'hcaptcha') {
        if (url.includes('checkbox')) subtype = 'checkbox';
        else if (url.includes('h5') || url.includes('challenge')) subtype = 'challenge';
      } else if (type === 'recaptcha') {
        if (url.includes('anchor')) subtype = 'checkbox';
        else if (url.includes('bframe')) subtype = 'challenge';
      }

      captchas.push({ type, subtype, frame });
    }
  }

  return captchas;
}

export async function waitForNewCaptcha(
  page: Page,
  knownFrames: Frame[],
  timeout = 5000
): Promise<Frame | null> {
  const start = Date.now();
  while (Date.now() - start < timeout) {
    for (const frame of page.frames()) {
      if (!knownFrames.includes(frame) && getCaptchaType(frame.url())) {
        if (await isFrameVisible(frame)) {
          return frame;
        }
      }
    }
    await page.waitForTimeout(200);
  }
  return null;
}

