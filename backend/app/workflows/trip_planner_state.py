"""State schema used by the trip planner LangGraph workflow."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

from ..models.schemas import Attraction, Hotel, TripPlan, TripRequest, WeatherInfo


def update_step(prev: str, new: str) -> str:
    """Reducer for current_step channel."""
    return new or prev


class TripPlannerState(TypedDict):
    """Workflow state shared across graph nodes."""

    # Input
    request: TripRequest
    user_input: str

    # Task decomposition result
    task_breakdown: Dict[str, Any]

    # Tool/agent outputs
    attractions: List[Attraction]
    weather_info: List[WeatherInfo]
    hotels: List[Hotel]
    abnormal_alerts: List[str]

    # Conversation/debug messages
    messages: Annotated[List[Dict[str, Any]], add_messages]

    # Final output / error
    trip_plan: Optional[TripPlan]
    error: Optional[str]
    current_step: Annotated[str, update_step]


def create_initial_state(request: TripRequest, user_input: str = "") -> TripPlannerState:
    """Create initial graph state."""
    return {
        "request": request,
        "user_input": user_input,
        "task_breakdown": {},
        "attractions": [],
        "weather_info": [],
        "hotels": [],
        "abnormal_alerts": [],
        "messages": [],
        "trip_plan": None,
        "error": None,
        "current_step": "started",
    }


def has_error(state: TripPlannerState) -> bool:
    """Whether current state has an error."""
    return state.get("error") is not None
