from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field


class PlannerAction(BaseModel):
    analysis: str = Field(description="Brief reasoning for the action")
    goal: str = Field(description="Description of the visual goal")


class PlannerClickAction(PlannerAction):
    action: Literal["click"]
    target_description: Optional[str] = Field(None, description="Object ID (e.g. 'Object 1') or description (e.g. 'the traffic light') to click")
    target_ids: Optional[List[int]] = Field(None, description="List of object IDs or grid cell numbers to click")
    max_items: int = Field(default=1, description="Maximum number of items to click if description matches multiple objects")


class PlannerDragAction(PlannerAction):
    action: Literal["drag"]
    source_description: Optional[str] = Field(None, description="Object ID or description of item to drag")
    source_id: Optional[int] = Field(None, description="Object ID of item to drag")
    target_description: str = Field(description="Description of where it should go (e.g. 'matching slot')")
    target_id: Optional[int] = Field(None, description="Object ID of destination")
    location_hint: Optional[List[float]] = Field(None, description="[x, y] (0.0-1.0) rough destination center if known")


class PlannerTypeAction(PlannerAction):
    action: Literal["type"]
    text: str = Field(description="Text to type into the input")
    target_description: Optional[str] = Field(None, description="Object ID or description of the input field")
    target_id: Optional[int] = Field(None, description="Object ID of the input field")


class PlannerWaitAction(PlannerAction):
    action: Literal["wait"]
    duration_ms: int = Field(default=500, description="Duration to wait in milliseconds")


class PlannerDragRefineAction(PlannerAction):
    action: Literal["refine_drag"]
    decision: Literal["accept", "adjust"] = Field(description="'accept' if aligned, 'adjust' if refinement needed")
    dx: float = Field(default=0.0, description="Horizontal adjustment (-1.0 to 1.0)")
    dy: float = Field(default=0.0, description="Vertical adjustment (-1.0 to 1.0)")


class PlannerDoneAction(PlannerAction):
    action: Literal["done"]


class PlannerToolCall(BaseModel):
    name: Literal["detect", "simulate_drag"]
    args: dict


class PlannerPlan(BaseModel):
    analysis: str
    goal: str
    tool_calls: Optional[List[PlannerToolCall]] = None
    # We allow the planner to return either a direct action or tool calls
    action: Optional[Union[PlannerClickAction, PlannerDragAction, PlannerTypeAction, PlannerWaitAction, PlannerDoneAction, PlannerDragRefineAction]] = None

