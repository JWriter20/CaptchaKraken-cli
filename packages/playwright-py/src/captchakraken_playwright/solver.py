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
    RequestUpdatedImageAction
)
from captchakraken.overlay import add_overlays_to_image
from .detection import find_captcha_frames, wait_for_new_captcha
from .extraction import extract_captcha_info

def solve_captchas_on_page(
    page: Page,
    solver: Optional[CaptchaSolver] = None,
    **solver_kwargs
) -> bool:
    """
    Find and solve all captchas on the page, handling popup challenges.
    """
    if solver is None:
        solver = CaptchaSolver(**solver_kwargs)
        
    while True:
        # 1. Find captchas
        captchas = find_captcha_frames(page)
        if not captchas:
            print("No captchas found.")
            return True # Or False?
            
        print(f"Found {len(captchas)} captcha frames.")
        
        # 2. Prioritize: Solve checkboxes first, then challenges
        # Usually challenges only appear after checkboxes are clicked.
        # But if a challenge is already present, we should solve it.
        
        target = None
        
        # Look for visible challenges first
        for c in captchas:
            if c['subtype'] == 'challenge':
                # Check visibility
                if c['frame'].locator('body').is_visible():
                    target = c
                    break
        
        # If no challenge, look for checkbox
        if not target:
            for c in captchas:
                if c['subtype'] == 'checkbox':
                    target = c
                    break
        
        if not target:
            # Maybe generic frames?
            target = captchas[0]
            
        print(f"Targeting captcha: {target['type']} ({target['subtype']})")
        
        # 3. Solve the target
        # We pass the frame and select 'body' as the element to screenshot/interact
        success = solve_captcha(
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
                # Loop will restart and find this new frame
                continue
            else:
                print("No new challenge appeared. Assuming done.")
                return True
        else:
            # If we solved a challenge, we are likely done or need to check checkbox status?
            # For now, assume done.
            return True

def solve_captcha(
    page_or_frame: Union[Page, Frame],
    selector: str,
    solver: Optional[CaptchaSolver] = None,
    context: str = "Solve this captcha",
    max_steps: int = 10,
    captcha_type: Optional[str] = None,
    captcha_subtype: Optional[str] = None,
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
        **solver_kwargs: Arguments to pass to CaptchaSolver constructor if solver is None
    
    Returns:
        True if solved (based on solver finishing), False if max steps reached
    """
    if solver is None:
        solver = CaptchaSolver(**solver_kwargs)
    
    # Store boxes for overlay
    boxes = []
    
    # Extract info and inject boxes if type provided
    if captcha_type and captcha_subtype and isinstance(page_or_frame, Frame):
        info = extract_captcha_info(page_or_frame, captcha_type, captcha_subtype)
        
        if info["challenge_element_selector"]:
            selector = info["challenge_element_selector"]
            
        if info["prompt_text"]:
            context += f"\nPrompt Text: {info['prompt_text']}"
        if info["prompt_image_url"]:
            context += f"\nPrompt Image URL: {info['prompt_image_url']}"
            
        if info.get("boxes"):
            boxes = info["boxes"]
            
    # Create a temporary file for screenshots
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        element = page_or_frame.locator(selector).first
        
        # Define image getter for the loop
        def get_image():
            # Ensure element is stable/visible
            element.wait_for(state="visible", timeout=5000)
            # Take screenshot of the specific element
            element.screenshot(path=tmp_path)
            
            # Apply overlays if boxes were found
            if boxes:
                print(f"Applying overlays to {tmp_path}")
                add_overlays_to_image(tmp_path, boxes)
                
            return tmp_path
        
        # Define end condition (optional - user can provide custom logic via callback in future)
        # For now, we rely on the planner saying "done"
        
        # Run the solver loop
        for action in solver.solve_loop(get_image, context, max_steps=max_steps):
            print(f"Executing action: {action}")
            
            if isinstance(action, ClickAction):
                # Coordinates are relative to the screenshot (the element)
                # Playwright expects relative to element for element.click?
                # element.click(position={x, y}) clicks relative to the element's top-left.
                x, y = action.coordinates
                element.click(position={"x": x, "y": y})
                
            elif isinstance(action, DragAction):
                sx, sy = action.source_coordinates
                tx, ty = action.target_coordinates
                
                # Perform drag using mouse API
                # We need absolute page coordinates for mouse.move usually, 
                # but if we use element handles, we can use relative?
                # Easier to use element bounding box to calculate page coordinates
                box = element.bounding_box()
                if box:
                    # Calculate absolute coordinates
                    abs_sx = box['x'] + sx
                    abs_sy = box['y'] + sy
                    abs_tx = box['x'] + tx
                    abs_ty = box['y'] + ty
                    
                    page = page_or_frame if isinstance(page_or_frame, Page) else page_or_frame.page
                    page.mouse.move(abs_sx, abs_sy)
                    page.mouse.down()
                    page.mouse.move(abs_tx, abs_ty, steps=10) # smooth drag
                    page.mouse.up()
            
            elif isinstance(action, TypeAction):
                if action.target_id is not None:
                    # If target_id specific, might need more complex logic
                    # For now, type into the main element or active element
                    element.type(action.text)
                else:
                    element.type(action.text)
                    
            elif isinstance(action, WaitAction):
                time.sleep(action.duration_ms / 1000.0)
                
            elif isinstance(action, RequestUpdatedImageAction):
                # Just loop, image will be retaken
                pass
                
        return True
        
    except Exception as e:
        print(f"Error solving captcha: {e}")
        return False
        
    finally:
        # Cleanup
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


