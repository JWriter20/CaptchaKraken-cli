from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field, model_validator


class PlannerAction(BaseModel):
    goal: Optional[str] = Field(None, description="Description of the visual goal")


class PlannerClickAction(PlannerAction):
    action: Literal["click"]
    target_ids: List[int] = Field(description="List of object IDs or grid cell numbers to click")


class PlannerDragAction(PlannerAction):
    action: Literal["drag"]
    source_id: int = Field(description="Object ID of item to drag")
    target_id: int = Field(description="Object ID of destination")


class PlannerTypeAction(PlannerAction):
    action: Literal["type"]
    text: str = Field(description="Text to type into the input")
    target_id: int = Field(description="Object ID of the input field")


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
    goal: str
    tool_calls: Optional[List[PlannerToolCall]] = None
    # We allow the planner to return either a direct action or tool calls
    action: Optional[Union[PlannerClickAction, PlannerDragAction, PlannerTypeAction, PlannerWaitAction, PlannerDoneAction, PlannerDragRefineAction]] = None

    @model_validator(mode="after")
    def check_exclusive(self) -> "PlannerPlan":
        if self.action and self.tool_calls:
            raise ValueError("Provide EITHER 'action' OR 'tool_calls', not both.")
        if not self.action and not self.tool_calls:
            raise ValueError("Provide EITHER 'action' OR 'tool_calls'.")
        return self

