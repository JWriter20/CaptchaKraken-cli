from typing import List, Literal, Optional, Union

from pydantic import BaseModel


class Action(BaseModel):
    action: str


class ClickAction(Action):
    action: Literal["click"]
    target_number: Optional[int] = None  # Only for grid selection captchas (1-9) or (1-16)
    target_bounding_box: Optional[List[float]] = None  # From detect() method (x1, y1, x2, y2) in percentages
    target_coordinates: Optional[List[float]] = None  # From point() method (x, y) in percentages


class DragAction(Action):
    action: Literal["drag"]
    source_number: Optional[int] = None  # Only for grid selection captchas (1-9) or (1-16)
    source_bounding_box: Optional[List[float]] = None  # From detect() method (x1, y1, x2, y2) in percentages
    source_coordinates: Optional[List[float]] = None  # From point() method (x, y) in percentages
    target_number: Optional[int] = None  # Only for grid selection captchas (1-9) or (1-16)
    target_bounding_box: Optional[List[float]] = None  # From detect() method (x1, y1, x2, y2) in percentages
    target_coordinates: Optional[List[float]] = None  # From point() method (x, y) in percentages


class TypeAction(Action):
    """Type text into an input."""

    action: Literal["type"]
    text: str
    target_bounding_box: Optional[List[float]] = None  # From detect() method (x1, y1, x2, y2) in percentages
    target_coordinates: Optional[List[float]] = None  # From point() method (x, y) in percentages


class WaitAction(Action):
    """Wait for a specified duration."""

    action: Literal["wait"]
    duration_ms: int


CaptchaAction = Union[ClickAction, DragAction, TypeAction, WaitAction]
