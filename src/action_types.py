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
