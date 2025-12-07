"""
Extraction utilities for captcha information from Playwright frames.

Extracts:
- Prompt text (instructions)
- Prompt image URL (if any)
- Challenge element selector
- Numbered interactable elements with bounding boxes
"""

from patchright.sync_api import Frame
from typing import Optional, Dict, List, Any
import re


def extract_captcha_info(frame: Frame, captcha_type: str, subtype: str) -> Dict[str, Any]:
    """
    Extract detailed information from a captcha frame.
    
    Args:
        frame: The Playwright frame containing the captcha
        captcha_type: Type of captcha (recaptcha, hcaptcha, cloudflare)
        subtype: Subtype (checkbox, challenge)
    
    Returns:
        Dict containing:
        - prompt_text: The instruction text
        - prompt_image_url: URL of prompt image if any
        - challenge_element_selector: CSS selector for main challenge area
        - boxes: List of numbered elements with bounding boxes
        - elements: List of InteractableElement-like dicts for the solver
    """
    info = {
        "prompt_text": None,
        "prompt_image_url": None,
        "challenge_element_selector": None,
        "boxes": [],  # For overlay drawing
        "elements": [],  # For intelligent solver
        "captcha_type": captcha_type,
        "captcha_subtype": subtype,
    }
    
    try:
        if captcha_type == "recaptcha" and subtype == "challenge":
            _extract_recaptcha_challenge(frame, info)
        elif captcha_type == "recaptcha" and subtype == "checkbox":
            _extract_recaptcha_checkbox(frame, info)
        elif captcha_type == "hcaptcha" and subtype == "challenge":
            _extract_hcaptcha_challenge(frame, info)
        elif captcha_type == "hcaptcha" and subtype == "checkbox":
            _extract_hcaptcha_checkbox(frame, info)
        elif captcha_type == "cloudflare":
            _extract_cloudflare(frame, info)
        else:
            # Generic extraction
            _extract_generic(frame, info)
            
    except Exception as e:
        print(f"Error extracting captcha info: {e}")
        
    return info


def _extract_recaptcha_challenge(frame: Frame, info: Dict[str, Any]) -> None:
    """Extract info from reCAPTCHA image challenge."""
    
    # Prompt text - try multiple selectors
    for selector in [
        ".rc-imageselect-desc-no-canonical",
        ".rc-imageselect-instructions",
        ".rc-imageselect-desc",
    ]:
        instructions = frame.locator(selector).first
        if instructions.count() > 0:
            text = instructions.text_content()
            if text:
                info["prompt_text"] = text.strip()
                break
    
    # Prompt image (some reCAPTCHA show example images)
    prompt_img = frame.locator(".rc-imageselect-desc-wrapper img, .rc-imageselect-example img").first
    if prompt_img.count() > 0:
        info["prompt_image_url"] = prompt_img.get_attribute("src")
    
    # Challenge element selector
    if frame.locator(".rc-imageselect-challenge").count() > 0:
        info["challenge_element_selector"] = ".rc-imageselect-challenge"
    elif frame.locator(".rc-imageselect-payload").count() > 0:
        info["challenge_element_selector"] = ".rc-imageselect-payload"
    else:
        info["challenge_element_selector"] = "body"
    
    # Extract numbered tiles
    boxes_and_elements = frame.evaluate("""
        () => {
            const tiles = document.querySelectorAll('td.rc-imageselect-tile');
            const results = [];
            
            tiles.forEach((tile, index) => {
                const target = tile.querySelector('.rc-image-tile-wrapper') || tile;
                const rect = target.getBoundingClientRect();
                const isSelected = tile.classList.contains('rc-imageselect-tileselected');
                
                results.push({
                    element_id: index + 1,
                    text: (index + 1).toString(),
                    bbox: [rect.x, rect.y, rect.width, rect.height],
                    element_type: 'image_tile',
                    is_selected: isSelected,
                    has_animation: tile.querySelector('.rc-image-tile-overlay') !== null
                });
            });
            
            // Also check for verify button
            const verifyBtn = document.querySelector('.rc-button-default, #recaptcha-verify-button');
            if (verifyBtn) {
                const rect = verifyBtn.getBoundingClientRect();
                results.push({
                    element_id: results.length + 1,
                    text: 'VERIFY',
                    bbox: [rect.x, rect.y, rect.width, rect.height],
                    element_type: 'verify_button',
                    is_selected: false
                });
            }
            
            return results;
        }
    """)
    
    info["boxes"] = [{"text": e["text"], "bbox": e["bbox"]} for e in boxes_and_elements]
    info["elements"] = boxes_and_elements


def _extract_recaptcha_checkbox(frame: Frame, info: Dict[str, Any]) -> None:
    """Extract info from reCAPTCHA checkbox."""
    
    info["prompt_text"] = "Click the checkbox to verify you're human"
    info["challenge_element_selector"] = ".recaptcha-checkbox-border, #recaptcha-anchor"
    
    # Extract checkbox element
    checkbox_info = frame.evaluate("""
        () => {
            const checkbox = document.querySelector('.recaptcha-checkbox-border, #recaptcha-anchor');
            if (checkbox) {
                const rect = checkbox.getBoundingClientRect();
                const isChecked = checkbox.getAttribute('aria-checked') === 'true';
                return [{
                    element_id: 1,
                    text: '1',
                    bbox: [rect.x, rect.y, rect.width, rect.height],
                    element_type: 'checkbox',
                    is_selected: isChecked
                }];
            }
            return [];
        }
    """)
    
    info["boxes"] = [{"text": e["text"], "bbox": e["bbox"]} for e in checkbox_info]
    info["elements"] = checkbox_info


def _extract_hcaptcha_challenge(frame: Frame, info: Dict[str, Any]) -> None:
    """Extract info from hCaptcha image challenge."""
    
    # Prompt text
    prompt = frame.locator(".prompt-text").first
    if prompt.count() > 0:
        info["prompt_text"] = prompt.text_content()
    
    # Prompt image (usually in .prompt-image with background-image)
    p_img = frame.locator(".prompt-image").first
    if p_img.count() > 0:
        style = p_img.get_attribute("style") or ""
        if "url(" in style:
            match = re.search(r'url\(["\']?(.+?)["\']?\)', style)
            if match:
                info["prompt_image_url"] = match.group(1)
        else:
            info["prompt_image_url"] = p_img.get_attribute("src")
    
    # Challenge element selector
    if frame.locator(".challenge-view").count() > 0:
        info["challenge_element_selector"] = ".challenge-view"
    elif frame.locator(".task-grid").count() > 0:
        info["challenge_element_selector"] = ".task-grid"
    elif frame.locator(".image-grid").count() > 0:
        info["challenge_element_selector"] = ".image-grid"
    else:
        info["challenge_element_selector"] = "body"
    
    # Extract task images
    boxes_and_elements = frame.evaluate("""
        () => {
            // Try multiple selectors for hCaptcha grid images
            let images = document.querySelectorAll('.task-image');
            if (images.length === 0) {
                images = document.querySelectorAll('.image-grid .image, .task-grid .task');
            }
            
            const results = [];
            
            images.forEach((img, index) => {
                const rect = img.getBoundingClientRect();
                const isSelected = img.classList.contains('selected') || 
                                   img.closest('.selected') !== null ||
                                   img.getAttribute('aria-pressed') === 'true';
                
                results.push({
                    element_id: index + 1,
                    text: (index + 1).toString(),
                    bbox: [rect.x, rect.y, rect.width, rect.height],
                    element_type: 'image_tile',
                    is_selected: isSelected
                });
            });
            
            // Check for submit button
            const submitBtn = document.querySelector('.submit-button, .button-submit, [data-action="submit"]');
            if (submitBtn) {
                const rect = submitBtn.getBoundingClientRect();
                results.push({
                    element_id: results.length + 1,
                    text: 'SUBMIT',
                    bbox: [rect.x, rect.y, rect.width, rect.height],
                    element_type: 'submit_button',
                    is_selected: false
                });
            }
            
            // Check for skip button
            const skipBtn = document.querySelector('.skip-button, .button-skip, [data-action="skip"]');
            if (skipBtn) {
                const rect = skipBtn.getBoundingClientRect();
                results.push({
                    element_id: results.length + 1,
                    text: 'SKIP',
                    bbox: [rect.x, rect.y, rect.width, rect.height],
                    element_type: 'skip_button',
                    is_selected: false
                });
            }
            
            return results;
        }
    """)
    
    info["boxes"] = [{"text": e["text"], "bbox": e["bbox"]} for e in boxes_and_elements]
    info["elements"] = boxes_and_elements


def _extract_hcaptcha_checkbox(frame: Frame, info: Dict[str, Any]) -> None:
    """Extract info from hCaptcha checkbox."""
    
    info["prompt_text"] = "Click the checkbox to verify you're human"
    info["challenge_element_selector"] = "#checkbox, .checkbox"
    
    checkbox_info = frame.evaluate("""
        () => {
            const checkbox = document.querySelector('#checkbox, .checkbox, [role="checkbox"]');
            if (checkbox) {
                const rect = checkbox.getBoundingClientRect();
                const isChecked = checkbox.getAttribute('aria-checked') === 'true';
                return [{
                    element_id: 1,
                    text: '1',
                    bbox: [rect.x, rect.y, rect.width, rect.height],
                    element_type: 'checkbox',
                    is_selected: isChecked
                }];
            }
            return [];
        }
    """)
    
    info["boxes"] = [{"text": e["text"], "bbox": e["bbox"]} for e in checkbox_info]
    info["elements"] = checkbox_info


def _extract_cloudflare(frame: Frame, info: Dict[str, Any]) -> None:
    """Extract info from Cloudflare Turnstile."""
    
    info["prompt_text"] = "Verify you are human"
    info["challenge_element_selector"] = "body"
    
    element_info = frame.evaluate("""
        () => {
            // Cloudflare checkbox/challenge area
            const checkbox = document.querySelector('input[type="checkbox"], .cf-turnstile, #turnstile-wrapper');
            const results = [];
            
            if (checkbox) {
                const rect = checkbox.getBoundingClientRect();
                results.push({
                    element_id: 1,
                    text: '1',
                    bbox: [rect.x, rect.y, rect.width, rect.height],
                    element_type: 'checkbox',
                    is_selected: checkbox.checked
                });
            }
            
            // Also look for any clickable areas
            const clickable = document.querySelector('[role="button"], button, .challenge-button');
            if (clickable && clickable !== checkbox) {
                const rect = clickable.getBoundingClientRect();
                results.push({
                    element_id: results.length + 1,
                    text: (results.length + 1).toString(),
                    bbox: [rect.x, rect.y, rect.width, rect.height],
                    element_type: 'button',
                    is_selected: false
                });
            }
            
            return results;
        }
    """)
    
    info["boxes"] = [{"text": e["text"], "bbox": e["bbox"]} for e in element_info]
    info["elements"] = element_info


def _extract_generic(frame: Frame, info: Dict[str, Any]) -> None:
    """Generic extraction for unknown captcha types."""
    
    info["challenge_element_selector"] = "body"
    
    # Try to find any visible text that looks like instructions
    text_content = frame.evaluate("""
        () => {
            const textElements = document.querySelectorAll('h1, h2, h3, p, .prompt, .instructions, .title');
            for (const el of textElements) {
                const text = el.textContent.trim();
                if (text.length > 10 && text.length < 200) {
                    return text;
                }
            }
            return null;
        }
    """)
    
    if text_content:
        info["prompt_text"] = text_content
    
    # Find all clickable/interactive elements
    element_info = frame.evaluate("""
        () => {
            const clickables = document.querySelectorAll(
                'button, input[type="checkbox"], input[type="submit"], ' +
                '[role="button"], [onclick], .clickable, img[style*="cursor"], ' +
                '[draggable="true"], .draggable'
            );
            
            const results = [];
            let idx = 1;
            
            clickables.forEach((el) => {
                const rect = el.getBoundingClientRect();
                // Skip tiny or invisible elements
                if (rect.width < 10 || rect.height < 10) return;
                if (rect.width === 0 || rect.height === 0) return;
                
                const isDraggable = el.draggable || el.classList.contains('draggable');
                
                results.push({
                    element_id: idx,
                    text: idx.toString(),
                    bbox: [rect.x, rect.y, rect.width, rect.height],
                    element_type: isDraggable ? 'draggable' : 'clickable',
                    is_selected: false
                });
                idx++;
            });
            
            return results;
        }
    """)
    
    info["boxes"] = [{"text": e["text"], "bbox": e["bbox"]} for e in element_info]
    info["elements"] = element_info


def extract_drag_elements(frame: Frame) -> Dict[str, Any]:
    """
    Specifically extract drag-related elements for slider/puzzle captchas.
    
    Returns:
        Dict with 'source' and 'target' element info
    """
    return frame.evaluate("""
        () => {
            const result = {source: null, target: null};
            
            // Look for draggable elements
            const draggable = document.querySelector(
                '[draggable="true"], .puzzle-piece, .slider-handle, ' +
                '.drag-source, .draggable, .captcha-slider'
            );
            
            if (draggable) {
                const rect = draggable.getBoundingClientRect();
                result.source = {
                    element_id: 1,
                    bbox: [rect.x, rect.y, rect.width, rect.height],
                    element_type: 'draggable'
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
                    bbox: [rect.x, rect.y, rect.width, rect.height],
                    element_type: 'drop_target'
                };
            }
            
            return result;
        }
    """)
