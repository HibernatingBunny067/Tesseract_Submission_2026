"""Multi-agent design state schema and messaging utilities.

Defines the shared DesignState TypedDict flowing through the agent graph,
along with structured message creation for the live UI thought stream.
"""
from __future__ import annotations

from typing import TypedDict, Optional, Any, List, Dict
from dataclasses import dataclass, field, asdict
from datetime import datetime


AGENT_INFO: Dict[str, Dict[str, str]] = {
    "clinical_interpreter": {"emoji": "🧑‍⚕️", "display_name": "Clinical Interpreter"},
    "materials_advisor": {"emoji": "🔬", "display_name": "Materials Advisor"},
    "optimization_controller": {"emoji": "⚙️", "display_name": "Optimization Controller"},
    "validation_auditor": {"emoji": "📋", "display_name": "Validation Auditor"},
}


@dataclass
class AgentMessage:
    """A single structured message emitted by an agent node."""
    agent_name: str
    agent_emoji: str
    agent_display_name: str
    content: str
    message_type: str
    data: Optional[Dict[str, Any]] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def create_message(
    agent_name: str,
    content: str,
    message_type: str,
    data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a structured agent message dictionary for the UI stream and audit log."""
    info: Dict[str, str] = AGENT_INFO.get(
        agent_name,
        {"emoji": "🤖", "display_name": agent_name.replace("_", " ").title()},
    )
    msg = AgentMessage(
        agent_name=agent_name,
        agent_emoji=info["emoji"],
        agent_display_name=info["display_name"],
        content=content,
        message_type=message_type,
        data=data,
    )
    msg_dict = msg.to_dict()

    # Automatically persist to dedicated agent audit log
    try:
        from src.utils.logger import log_agent_message
        log_agent_message(
            agent_name=agent_name,
            display_name=info["display_name"],
            emoji=info["emoji"],
            message_type=message_type,
            content=content,
            data=data
        )
    except Exception:
        pass

    return msg_dict


class DesignState(TypedDict):
    """Shared typed state that flows through every node in the agent graph."""
    surgeon_prompt: str
    clinical_profile: Optional[Dict[str, Any]]
    design_spec: Optional[Dict[str, Any]]
    optimization_result: Optional[Dict[str, Any]]
    validation_report: Optional[Dict[str, Any]]
    corrections: Optional[Dict[str, Any]]
    attempt: int
    max_attempts: int
    max_steps: Optional[int]
    messages: List[Dict[str, Any]]
    fem_client: Optional[Any]
    geometry_client: Optional[Any]
    bend_y_array: Optional[List[float]]
    bend_z_array: Optional[List[float]]
    final_design: Optional[Dict[str, Any]]
