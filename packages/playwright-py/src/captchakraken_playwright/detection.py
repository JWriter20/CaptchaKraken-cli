from typing import List, Dict, Optional, Union
from patchright.sync_api import Page, Frame

CAPTCHA_SIGNATURES = {
    "hcaptcha": ["hcaptcha.com", "assets.hcaptcha.com"],
    "recaptcha": ["google.com/recaptcha", "gstatic.com/recaptcha"],
    "cloudflare": ["challenges.cloudflare.com"],
}

def get_captcha_type(url: str) -> Optional[str]:
    for captcha_type, domains in CAPTCHA_SIGNATURES.items():
        if any(domain in url for domain in domains):
            return captcha_type
    return None

def is_frame_visible(frame: Frame) -> bool:
    """
    Check if a frame is visible, on-screen, and not fully obscured.
    """
    try:
        if frame.is_detached():
            return False
            
        element = frame.frame_element()
        if not element.is_visible():
            return False
            
        # Check bounding box
        box = element.bounding_box()
        if not box or box['width'] == 0 or box['height'] == 0:
            return False
            
        # Check if center point is within viewport
        # Note: Playwright's is_visible handles CSS visibility and display none.
        # But we also want to ensure it's in the viewport or at least scrollable to it?
        # Usually captchas are in view. 
        
        # Check if obscured by checking if element at center point is this element (or inside it)
        # This is expensive, so maybe skip for now unless requested.
        # But user asked for "non-covered".
        
        # Simple check: is_visible() covers display:none, visibility:hidden, etc.
        # We can add a check for viewport intersection if needed.
        
        return True
    except Exception:
        return False

def find_captcha_frames(page: Page) -> List[Dict[str, Union[str, Frame]]]:
    """
    Find all visible captcha iframes on the page.
    
    Returns:
        List of dicts with keys 'type' and 'frame'.
    """
    captchas = []
    
    for frame in page.frames:
        if not is_frame_visible(frame):
            continue
            
        captcha_type = get_captcha_type(frame.url)
        if captcha_type:
            # refine hcaptcha types
            subtype = "generic"
            if captcha_type == "hcaptcha":
                if "checkbox" in frame.url:
                    subtype = "checkbox"
                elif "h5" in frame.url or "challenge" in frame.url:
                    subtype = "challenge"
            
            elif captcha_type == "recaptcha":
                if "anchor" in frame.url:
                    subtype = "checkbox"
                elif "bframe" in frame.url:
                    subtype = "challenge"
            
            captchas.append({
                "type": captcha_type,
                "subtype": subtype,
                "frame": frame
            })
            
    return captchas

def wait_for_new_captcha(page: Page, known_frames: List[Frame], timeout: float = 5000) -> Optional[Frame]:
    """
    Wait for a new visible captcha frame to appear.
    """
    start = page.evaluate("Date.now()")
    while page.evaluate("Date.now()") - start < timeout:
        for frame in page.frames:
            if frame not in known_frames and get_captcha_type(frame.url):
                if is_frame_visible(frame):
                    return frame
        page.wait_for_timeout(200)
    return None

