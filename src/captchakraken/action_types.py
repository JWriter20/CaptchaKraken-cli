from typing import List, Optional, Literal, Union
from pydantic import BaseModel, Field


class Action(BaseModel):
    action: str


class ClickAction(Action):
    """Click on one or more elements/coordinates."""
    action: Literal["click"]
    target_id: Optional[int] = None  # Single element ID
    target_ids: Optional[List[int]] = None  # Multiple element IDs for multi-click
    # Pixel coordinates for single click (fallback for downstream executors)
    coordinates: Optional[List[float]] = None  # [x, y] as percentages or pixels
    all_coordinates: Optional[List[List[float]]] = None  # Multiple coordinates for multi-click
    all_coordinates_pct: Optional[List[List[float]]] = None  # Multiple normalized coordinates for multi-click
    # Normalized coordinates for single-click targets
    point_percent: Optional[List[float]] = None  # [x_pct, y_pct] in [0, 1]
    # Bounding boxes for multi-select captchas (normalized and absolute variants)
    bounding_boxes: Optional[List[List[float]]] = None  # [[x1, y1, x2, y2]] in [0, 1]
    bounding_boxes_px: Optional[List[List[float]]] = None  # [[x1, y1, x2, y2]] in pixels


class DragAction(Action):
    """Drag from source to target."""
    action: Literal["drag"]
    source_id: Optional[int] = None
    source_coordinates: Optional[List[float]] = None
    source_coordinates_pct: Optional[List[float]] = None
    target_id: Optional[int] = None
    target_coordinates: Optional[List[float]] = None
    target_coordinates_pct: Optional[List[float]] = None
    template_match_confidence: Optional[float] = None


class TypeAction(Action):
    """Type text into an input."""
    action: Literal["type"]
    text: str
    target_id: Optional[int] = None


class WaitAction(Action):
    """Wait for a specified duration."""
    action: Literal["wait"]
    duration_ms: int


class RequestUpdatedImageAction(Action):
    """Request a fresh screenshot."""
    action: Literal["request_updated_image"]


class VerifyAction(Action):
    """Verify/submit the current solution."""
    action: Literal["verify"]
    target_id: Optional[int] = None  # Verify/submit button ID if numbered


CaptchaAction = Union[ClickAction, DragAction, TypeAction, WaitAction, RequestUpdatedImageAction, VerifyAction]


class Solution(BaseModel):
    """Complete solution with ordered actions."""
    actions: List[CaptchaAction]


# Rich context passed from playwright packages
class CaptchaContext(BaseModel):
    """Rich context about the captcha for intelligent solving."""
    prompt_text: Optional[str] = None  # "Select all images with traffic lights"
    prompt_image_url: Optional[str] = None  # URL of prompt image if any
    captcha_type: Optional[str] = None  # recaptcha, hcaptcha, cloudflare
    captcha_subtype: Optional[str] = None  # checkbox, challenge
    
    # Numbered interactable elements from DOM extraction
    elements: Optional[List["InteractableElement"]] = None
    has_numbered_overlays: bool = False  # True if image has numbered overlay boxes


class InteractableElement(BaseModel):
    """An interactable element detected in the captcha."""
    element_id: int  # The numbered ID (1, 2, 3, ...)
    bbox: List[float]  # [x, y, width, height] relative to screenshot
    element_type: Optional[str] = None  # "image", "checkbox", "button", "slider"
    description: Optional[str] = None  # Optional description


class DetectedObject(BaseModel):
    """Object detected by moondream detect()."""
    label: str
    bbox: List[float]  # [x_min, y_min, x_max, y_max] as percentages
    confidence: Optional[float] = None
    overlapping_element_ids: Optional[List[int]] = None  # Element IDs this detection overlaps with
