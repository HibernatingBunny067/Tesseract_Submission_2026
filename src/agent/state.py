from __future__ import annotations
from typing import TypedDict, Optional, Any, List, Dict
from dataclasses import dataclass, field, asdict
from datetime import datetime

AGENT_INFO = {
    'clinical_interpreter': {'emoji': '🧑‍⚕️', 'display_name': 'Clinical Interpreter'},
    'materials_advisor': {'emoji': '🔬', 'display_name': 'Materials Advisor'},
    'optimization_controller': {'emoji': '⚙️', 'display_name': 'Optimization Controller'},
    'validation_auditor': {'emoji': '📋', 'display_name': 'Validation Auditor'}
}

@dataclass
class AgentMessage:
    agent_name: str
    agent_emoji: str
    agent_display_name: str
    content: str
    message_type: str
    data: Optional[Dict[str, Any]] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def create_message(agent_name: str, content: str, message_type: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    info = AGENT_INFO.get(agent_name, {'emoji': '🤖', 'display_name': agent_name.replace('_', ' ').title()})
    msg = AgentMessage(
        agent_name=agent_name,
        agent_emoji=info['emoji'],
        agent_display_name=info['display_name'],
        content=content,
        message_type=message_type,
        data=data
    )
    return msg.to_dict()

class DesignState(TypedDict):
    surgeon_prompt: str
    clinical_profile: Optional[Dict[str, Any]]
    design_spec: Optional[Dict[str, Any]]
    optimization_result: Optional[Dict[str, Any]]
    validation_report: Optional[Dict[str, Any]]
    corrections: Optional[Dict[str, Any]]
    attempt: int
    max_attempts: int
    messages: List[Dict[str, Any]]
    fem_client: Optional[Any]
    geometry_client: Optional[Any]
    final_design: Optional[Dict[str, Any]]
