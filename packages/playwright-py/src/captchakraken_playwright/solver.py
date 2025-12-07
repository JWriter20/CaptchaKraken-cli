"""
Playwright/Patchright integration for CaptchaKraken.

Provides functions to solve captchas in browser automation contexts.
"""

from typing import Optional, Union, Callable, List
import os
import time
import tempfile
from patchright.sync_api import Page, Frame, Locator
from captchakraken.solver import CaptchaSolver
from captchakraken.action_types import (
    ClickAction,
    DragAction,
    TypeAction,
    WaitAction,
    RequestUpdatedImageAction,
    VerifyAction,
)
from captchakraken.overlay import add_overlays_to_image
from .detection import find_captcha_frames, wait_for_new_captcha
from .extraction import extract_captcha_info, extract_drag_elements


def solve_captcha(
    page: Page,
    solver: Optional[CaptchaSolver] = None,
    **solver_kwargs
) -> bool:
    """
    Find and solve all captchas on the page, handling popup challenges.
    
    Args:
        page: Playwright Page object
        solver: Optional pre-configured CaptchaSolver instance
        **solver_kwargs: Arguments to pass to CaptchaSolver if creating new one
    
    Returns:
        True if captchas were solved successfully, False otherwise
    """
    if solver is None:
        solver = CaptchaSolver(**solver_kwargs)
        
    while True:
        # 1. Find captchas
        captchas = find_captcha_frames(page)
        if not captchas:
            print("No captchas found.")
            return True
            
        print(f"Found {len(captchas)} captcha frames.")
        
        # 2. Prioritize: Solve visible challenges first, then checkboxes
        target = None
        
        # Look for visible challenges first
        for c in captchas:
            if c['subtype'] == 'challenge':
                if c['frame'].locator('body').is_visible():
                    target = c
                    break
        
        # If no challenge, look for checkbox
        if not target:
            for c in captchas:
                if c['subtype'] == 'checkbox':
                    target = c
                    break
        
        # If still no target, default to first
        if not target:
            target = captchas[0]
            
        print(f"Targeting captcha: {target['type']} ({target['subtype']})")
        
        # 3. Solve the target
        success = solve_single_captcha(
            page_or_frame=target['frame'],
            selector="body",
            solver=solver,
            context=f"Solve this {target['type']} {target['subtype']}",
            max_steps=5 if target['subtype'] == 'checkbox' else 15,
            captcha_type=target['type'],
            captcha_subtype=target['subtype']
        )
        
        if not success:
            print("Failed to solve target.")
            return False
            
        # 4. If we solved a checkbox, wait to see if a challenge appears
        if target['subtype'] == 'checkbox':
            print("Solved checkbox, checking for new challenges...")
            known_frames = page.frames
            new_frame = wait_for_new_captcha(page, known_frames, timeout=3000)
            
            if new_frame:
                print("New challenge appeared!")
                continue
            else:
                print("No new challenge appeared. Assuming done.")
                return True
        else:
            return True


def solve_single_captcha(
    page_or_frame: Union[Page, Frame],
    selector: str,
    solver: Optional[CaptchaSolver] = None,
    context: str = "Solve this captcha",
    max_steps: int = 10,
    captcha_type: Optional[str] = None,
    captcha_subtype: Optional[str] = None,
    use_intelligent_detection: bool = True,
    **solver_kwargs
) -> bool:
    """
    Solve a captcha on a Playwright/Patchright page or frame.
    
    Args:
        page_or_frame: The page or frame containing the captcha
        selector: CSS selector for the captcha container element
        solver: Existing CaptchaSolver instance (optional)
        context: Instructions for the solver
        max_steps: Maximum steps to attempt
        captcha_type: Type of captcha (recaptcha, hcaptcha, etc.)
        captcha_subtype: Subtype (checkbox, challenge)
        use_intelligent_detection: Whether to use moondream detection for refinement
        **solver_kwargs: Arguments to pass to CaptchaSolver constructor if solver is None
    
    Returns:
        True if solved (based on solver finishing), False if max steps reached
    """
    if solver is None:
        solver = CaptchaSolver(**solver_kwargs)
    
    # Store boxes and elements for overlay and intelligent solving
    boxes = []
    elements = []
    prompt_text = None
    
    # Create a temporary file for screenshots
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Extract info if we have type information
        if captcha_type and captcha_subtype and isinstance(page_or_frame, Frame):
            info = extract_captcha_info(page_or_frame, captcha_type, captcha_subtype)
            
            if info.get("challenge_element_selector"):
                selector = info["challenge_element_selector"]
                
            prompt_text = info.get("prompt_text")
            if prompt_text:
                context += f"\nPrompt Text: {prompt_text}"
            
            if info.get("prompt_image_url"):
                context += f"\nPrompt Image URL: {info['prompt_image_url']}"
                
            boxes = info.get("boxes", [])
            elements = info.get("elements", [])
            
            # Check for drag elements
            if captcha_subtype in ['slider', 'puzzle']:
                drag_info = extract_drag_elements(page_or_frame)
                if drag_info.get('source'):
                    elements.append(drag_info['source'])
                if drag_info.get('target'):
                    elements.append(drag_info['target'])
        
        element = page_or_frame.locator(selector).first
        
        for step in range(max_steps):
            print(f"\n{'='*40}")
            print(f"Step {step + 1}/{max_steps}")
            print(f"{'='*40}")
            
            # Re-extract element info each step (selections may have changed)
            if captcha_type and captcha_subtype and isinstance(page_or_frame, Frame):
                info = extract_captcha_info(page_or_frame, captcha_type, captcha_subtype)
                boxes = info.get("boxes", [])
                elements = info.get("elements", [])
            
            # Ensure element is stable/visible
            try:
                element.wait_for(state="visible", timeout=5000)
            except Exception:
                print("Element not visible, trying original selector...")
                element = page_or_frame.locator("body").first
                element.wait_for(state="visible", timeout=5000)
            
            # Take screenshot of the specific element
            element.screenshot(path=tmp_path)
            
            # Apply overlays if boxes were found
            if boxes:
                print(f"Applying overlays for {len(boxes)} elements...")
                add_overlays_to_image(tmp_path, boxes)
            
            # Get action using intelligent detection if enabled
            if use_intelligent_detection and elements:
                action = solver.solve_step_intelligent(
                    tmp_path,
                    context=context,
                    elements=elements,
                    prompt_text=prompt_text,
                )
            else:
                action = solver.solve_step(
                    tmp_path,
                    context=context,
                    elements=elements if elements else None,
                    prompt_text=prompt_text,
                )
            
            print(f"Executing action: {action.action}")
            
            # Execute the action
            if isinstance(action, ClickAction):
                _execute_click(element, action, elements)
                
            elif isinstance(action, DragAction):
                _execute_drag(page_or_frame, element, action)
            
            elif isinstance(action, TypeAction):
                if action.target_id is not None and elements:
                    # Type into specific element if possible
                    target_elem = _find_element_by_id(page_or_frame, elements, action.target_id)
                    if target_elem:
                        target_elem.type(action.text)
                    else:
                        element.type(action.text)
                else:
                    element.type(action.text)
                    
            elif isinstance(action, WaitAction):
                if action.duration_ms == 0:
                    print("Solver indicates completion.")
                    return True
                time.sleep(action.duration_ms / 1000.0)
                
            elif isinstance(action, RequestUpdatedImageAction):
                # Just continue to next iteration
                continue
            
            elif isinstance(action, VerifyAction):
                _execute_verify(page_or_frame, element, action, elements)
                # After verify, check if we're done
                time.sleep(1.0)  # Wait for verification
                
            # Small delay between actions
            time.sleep(0.3)
        
        print("Max steps reached.")
        return False
        
    except Exception as e:
        print(f"Error solving captcha: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _execute_click(element: Locator, action: ClickAction, elements: List[dict]) -> None:
    """Execute a click action."""
    
    # Multi-click support
    if action.all_coordinates:
        for coords in action.all_coordinates:
            x, y = coords[0], coords[1]
            print(f"  Clicking at ({x}, {y})")
            element.click(position={"x": x, "y": y})
            time.sleep(0.2)  # Small delay between clicks
    
    elif action.target_ids:
        # Click multiple elements by ID
        for target_id in action.target_ids:
            # Find element center from elements list
            for elem in elements:
                eid = elem.get('element_id', elem.get('id'))
                if eid == target_id:
                    bbox = elem.get('bbox', [])
                    if len(bbox) >= 4:
                        x = bbox[0] + bbox[2] / 2
                        y = bbox[1] + bbox[3] / 2
                        print(f"  Clicking element {target_id} at ({x}, {y})")
                        element.click(position={"x": x, "y": y})
                        time.sleep(0.2)
                    break
    
    elif action.coordinates:
        x, y = action.coordinates[0], action.coordinates[1]
        print(f"  Clicking at ({x}, {y})")
        element.click(position={"x": x, "y": y})


def _execute_drag(page_or_frame: Union[Page, Frame], element: Locator, action: DragAction) -> None:
    """Execute a drag action."""
    sx, sy = action.source_coordinates[0], action.source_coordinates[1]
    tx, ty = action.target_coordinates[0], action.target_coordinates[1]
    
    # Get bounding box to calculate absolute coordinates
    box = element.bounding_box()
    if box:
        # Calculate absolute coordinates
        abs_sx = box['x'] + sx
        abs_sy = box['y'] + sy
        abs_tx = box['x'] + tx
        abs_ty = box['y'] + ty
        
        print(f"  Dragging from ({abs_sx}, {abs_sy}) to ({abs_tx}, {abs_ty})")
        
        # Get page object for mouse operations
        page = page_or_frame if isinstance(page_or_frame, Page) else page_or_frame.page
        
        # Perform drag with smooth movement
        page.mouse.move(abs_sx, abs_sy)
        page.mouse.down()
        time.sleep(0.1)
        
        # Move in steps for smoother drag
        steps = 20
        for i in range(1, steps + 1):
            t = i / steps
            current_x = abs_sx + (abs_tx - abs_sx) * t
            current_y = abs_sy + (abs_ty - abs_sy) * t
            page.mouse.move(current_x, current_y)
            time.sleep(0.02)
        
        page.mouse.up()


def _execute_verify(
    page_or_frame: Union[Page, Frame],
    element: Locator,
    action: VerifyAction,
    elements: List[dict]
) -> None:
    """Execute a verify/submit action."""
    
    # Try to find verify button from elements
    verify_elem = None
    
    for elem in elements:
        elem_type = elem.get('element_type', '')
        if elem_type in ['verify_button', 'submit_button']:
            bbox = elem.get('bbox', [])
            if len(bbox) >= 4:
                x = bbox[0] + bbox[2] / 2
                y = bbox[1] + bbox[3] / 2
                print(f"  Clicking verify button at ({x}, {y})")
                element.click(position={"x": x, "y": y})
                return
    
    # Fall back to common verify button selectors
    verify_selectors = [
        ".rc-button-default",
        "#recaptcha-verify-button",
        ".verify-button",
        ".submit-button",
        ".button-submit",
        "button[type='submit']",
    ]
    
    for sel in verify_selectors:
        btn = page_or_frame.locator(sel).first
        if btn.count() > 0 and btn.is_visible():
            print(f"  Found verify button with selector: {sel}")
            btn.click()
            return
    
    print("  Could not find verify button")


def _find_element_by_id(
    page_or_frame: Union[Page, Frame],
    elements: List[dict],
    element_id: int
) -> Optional[Locator]:
    """Find a Playwright locator for an element by its ID."""
    for elem in elements:
        eid = elem.get('element_id', elem.get('id'))
        if eid == element_id:
            bbox = elem.get('bbox', [])
            if len(bbox) >= 4:
                # Use position-based clicking since we have bbox
                x = bbox[0] + bbox[2] / 2
                y = bbox[1] + bbox[3] / 2
                # Return a positioned locator
                return page_or_frame.locator(f":nth-match(*, 1)").first
    return None
