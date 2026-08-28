"""
Integration test for Multi-Agent LangGraph Biomechanics System.
"""

import os
import sys
import io

# Force UTF-8 output on Windows consoles
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent.state import DesignState, create_message
from src.agent.llm_provider import get_provider
from src.agent.graph import build_design_graph, run_design_agent


def test_graph_compilation():
    print("Test 1: Compiling LangGraph StateGraph...")
    graph = build_design_graph()
    assert graph is not None, "Graph compilation failed"
    print(" [PASS] StateGraph compiled successfully!\n")


def test_agent_deliberation_flow():
    print("Test 2: Running End-to-End Multi-Agent Design Cycle...")
    prompt = "22-year-old competitive skier with midshaft spiral femur fracture needing high torsional rigidity"
    
    agent_message_count = 0
    final_state = None
    
    for event in run_design_agent(prompt, max_attempts=2, stream=True):
        if event["type"] == "agent_message":
            msg = event["message"]
            agent_message_count += 1
            print(f"[{msg.get('agent_emoji', '')} {msg.get('agent_display_name', '')} - {msg.get('message_type','').upper()}]:")
            print(f"{msg.get('content', '')}\n" + "-"*60)
        elif event["type"] == "final_result":
            final_state = event["state"]

    assert agent_message_count >= 4, f"Expected at least 4 agent messages, got {agent_message_count}"
    assert final_state is not None, "Final state should not be None"
    assert final_state.get("clinical_profile") is not None, "Clinical profile missing"
    assert final_state.get("design_spec") is not None, "Design spec missing"
    assert final_state.get("optimization_result") is not None, "Optimization result missing"
    assert final_state.get("validation_report") is not None, "Validation report missing"
    
    print("\n [PASS] Multi-Agent End-to-End Execution Passed!")
    print("Final Design Summary:", final_state.get("final_design"))


if __name__ == "__main__":
    test_graph_compilation()
    test_agent_deliberation_flow()
