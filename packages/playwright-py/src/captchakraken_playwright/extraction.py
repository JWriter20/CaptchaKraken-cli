from patchright.sync_api import Frame
from typing import Optional, Dict
import re

def extract_captcha_info(frame: Frame, captcha_type: str, subtype: str) -> Dict[str, Optional[str]]:
    info = {
        "prompt_text": None,
        "prompt_image_url": None,
        "challenge_element_selector": None,
        "boxes": []  # List of boxes to draw on the screenshot
    }
    
    try:
        if captcha_type == "recaptcha" and subtype == "challenge":
            # ReCAPTCHA
            instructions = frame.locator(".rc-imageselect-desc-no-canonical, .rc-imageselect-instructions").first
            if instructions.count() > 0:
                info["prompt_text"] = instructions.text_content()
                
            prompt_img = frame.locator(".rc-imageselect-desc-wrapper img").first
            if prompt_img.count() > 0:
                info["prompt_image_url"] = prompt_img.get_attribute("src")
                
            if frame.locator(".rc-imageselect-challenge").count() > 0:
                info["challenge_element_selector"] = ".rc-imageselect-challenge"
            else:
                info["challenge_element_selector"] = ".rc-imageselect-payload"
                
            # Collect bounding boxes for overlay
            boxes = frame.evaluate("""
                () => {
                    const tiles = document.querySelectorAll('td.rc-imageselect-tile');
                    const results = [];
                    tiles.forEach((tile, index) => {
                        const target = tile.querySelector('.rc-image-tile-wrapper') || tile;
                        const rect = target.getBoundingClientRect();
                        results.push({
                            text: (index + 1).toString(),
                            bbox: [rect.x, rect.y, rect.width, rect.height]
                        });
                    });
                    return results;
                }
            """)
            info["boxes"] = boxes
            
        elif captcha_type == "hcaptcha" and subtype == "challenge":
            # hCaptcha
            prompt = frame.locator(".prompt-text").first
            if prompt.count() > 0:
                info["prompt_text"] = prompt.text_content()
                
            p_img = frame.locator(".prompt-image").first
            if p_img.count() > 0:
                style = p_img.get_attribute("style") or ""
                if "url(" in style:
                    match = re.search(r'url\("?(.+?)"?\)', style)
                    if match:
                        info["prompt_image_url"] = match.group(1)
                else:
                    info["prompt_image_url"] = p_img.get_attribute("src")
                    
            if frame.locator(".challenge-view").count() > 0:
                info["challenge_element_selector"] = ".challenge-view"
            elif frame.locator(".image-grid").count() > 0:
                info["challenge_element_selector"] = ".image-grid"
            elif frame.locator(".task-image").count() > 0:
                # If just task images found but no container known, try to find a container
                info["challenge_element_selector"] = "body"

            # Collect bounding boxes for overlay
            boxes = frame.evaluate("""
                () => {
                    const gridImages = document.querySelectorAll('.task-image .image, .task-image');
                    const results = [];
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
                }
            """)
            info["boxes"] = boxes

    except Exception as e:
        print(f"Error extracting info: {e}")
        
    return info
