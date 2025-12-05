from typing import List, Optional, Literal, Union
from pydantic import BaseModel, Field

class Action(BaseModel):
    action: str

class ClickAction(Action):
    action: Literal["click"]
    target_id: Optional[int] = None
    coordinates: Optional[List[int]] = None # [x, y]

class DragAction(Action):
    action: Literal["drag"]
    source_id: Optional[int] = None
    source_coordinates: Optional[List[int]] = None
    target_id: Optional[int] = None
    target_coordinates: Optional[List[int]] = None

class TypeAction(Action):
    action: Literal["type"]
    text: str
    target_id: Optional[int] = None

class WaitAction(Action):
    action: Literal["wait"]
    duration_ms: int

class RequestUpdatedImageAction(Action):
    action: Literal["request_updated_image"]

CaptchaAction = Union[ClickAction, DragAction, TypeAction, WaitAction, RequestUpdatedImageAction]

class Solution(BaseModel):
    actions: List[CaptchaAction]
