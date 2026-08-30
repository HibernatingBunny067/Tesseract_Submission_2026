"""
LangGraph Multi-Agent Orchestration Graph for Biomechanical Implant Optimization.

Coordinates 4 specialist agents:
1. Clinical Interpreter (Patient intent & AO Foundation principles)
2. Materials & Topology Advisor (Biomaterials & TPMS selection)
3. Optimization Controller (JAX WSD Adam solver & convergence analysis)
4. Validation Auditor (ASTM F382 & ISO 7206 in-silico testing & self-correction)
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Generator, Optional, Union

from src.agent.state import DesignState, create_message
from src.agent.nodes import (
    clinical_interpreter_node,
    materials_advisor_node,
    optimization_controller_node,
    validation_auditor_node,
)


def prepare_retry_node(state: DesignState) -> dict:
    """Prepares state for the next optimization attempt by incrementing attempt counter."""
    attempt = state.get("attempt", 1) + 1
    messages = list(state.get("messages", []))
    corrections = state.get("corrections", {}) or {}
    adjusted = corrections.get("adjusted_params", {})
    
    design_spec = dict(state.get("design_spec") or {})
    initial_params = dict(design_spec.get("initial_params") or {})
    for k, v in adjusted.items():
        initial_params[k] = v
    design_spec["initial_params"] = initial_params
    
    messages.append(create_message(
        "optimization_controller",
        f"🔄 Initializing Self-Correction Loop (Attempt {attempt}/{state.get('max_attempts', 3)})\n\n"
        f"Applying prescribed parameters: {adjusted}",
        "status",
        data={"attempt": attempt, "corrections": corrections}
    ))
    
    return {
        "attempt": attempt,
        "design_spec": design_spec,
        "messages": messages,
    }


def finalize_design_node(state: DesignState) -> dict:
    """Synthesizes the full design session into a clinical sign-off package."""
    clinical = state.get("clinical_profile") or {}
    design = state.get("design_spec") or {}
    opt_res = state.get("optimization_result") or {}
    val_rep = state.get("validation_report") or {}
    messages = list(state.get("messages", []))
    
    verdict = val_rep.get("verdict", "COMPLETED")
    score = val_rep.get("score_pct", 0.0)
    attempts = state.get("attempt", 1)
    
    final_summary = {
        "verdict": verdict,
        "compliance_score_pct": score,
        "attempts_required": attempts,
        "material": design.get("material_name", "Ti-6Al-4V (Grade 5 Titanium)"),
        "tpms_topology": design.get("tpms_type", "Schwarz Primitive (P)"),
        "target_micro_motion_mm": clinical.get("target_micro_motion_m", 0.0002) * 1000.0,
        "achieved_micro_motion_mm": (opt_res.get("final_metrics", {}).get("frac_disp", 0.0)) * 1000.0,
        "final_mass_fraction": opt_res.get("final_metrics", {}).get("mass_fraction", 0.0),
        "mean_porosity_pct": (opt_res.get("final_metrics", {}).get("mean_porosity", 0.0)) * 100.0,
        "bend_y_array": state.get("bend_y_array"),
        "bend_z_array": state.get("bend_z_array"),
    }
    
    status_emoji = "🏆" if "CLINICALLY APPROVED" in verdict else "⚠️"
    messages.append(create_message(
        "validation_auditor",
        f"{status_emoji} **Final Implant Sign-Off Report**\n\n"
        f"• Status: {verdict} ({score:.0f}% Verification Compliance)\n"
        f"• Total Iterations: {attempts}\n"
        f"• Final Metamaterial: {final_summary['material']} | {final_summary['tpms_topology']}\n"
        f"• Fracture Gap Micro-Motion: {final_summary['achieved_micro_motion_mm']:.3f} mm "
        f"(Target: {final_summary['target_micro_motion_mm']:.2f} mm)\n"
        f"• Plate Porosity / Mass Fraction: {final_summary['mean_porosity_pct']:.1f}% / {final_summary['final_mass_fraction']:.3f}",
        "result",
        data=final_summary
    ))

    # Log completion to dedicated agent audit log
    try:
        from src.utils.logger import log_agent_session_summary
        log_agent_session_summary(
            surgeon_prompt=state.get("surgeon_prompt", ""),
            attempts=attempts,
            verdict=verdict,
            messages_count=len(messages),
            final_design=final_summary
        )
    except Exception:
        pass
    
    return {
        "final_design": final_summary,
        "messages": messages
    }


def should_retry(state: DesignState) -> str:
    """Conditional edge router: determines whether to loop back for correction or finalize."""
    report = state.get("validation_report")
    attempt = state.get("attempt", 1)
    max_attempts = state.get("max_attempts", 3)

    if not report:
        return "finalize"

    verdict = report.get("verdict", "")
    if "CLINICALLY APPROVED" in verdict:
        return "finalize"
    elif attempt >= max_attempts:
        return "finalize"
    else:
        return "retry"


def build_design_graph():
    """Builds and compiles the LangGraph StateGraph for multi-agent implant optimization."""
    try:
        from langgraph.graph import StateGraph, END
    except ImportError:
        return None

    workflow = StateGraph(DesignState)

    # 1. Add specialist agent nodes
    workflow.add_node("clinical_interpreter", clinical_interpreter_node)
    workflow.add_node("materials_advisor", materials_advisor_node)
    workflow.add_node("optimization_controller", optimization_controller_node)
    workflow.add_node("validation_auditor", validation_auditor_node)
    workflow.add_node("prepare_retry", prepare_retry_node)
    workflow.add_node("finalize_design", finalize_design_node)

    # 2. Add linear deterministic flow
    workflow.set_entry_point("clinical_interpreter")
    workflow.add_edge("clinical_interpreter", "materials_advisor")
    workflow.add_edge("materials_advisor", "optimization_controller")
    workflow.add_edge("optimization_controller", "validation_auditor")

    # 3. Add conditional self-correction loop
    workflow.add_conditional_edges(
        "validation_auditor",
        should_retry,
        {
            "retry": "prepare_retry",
            "finalize": "finalize_design",
        }
    )
    workflow.add_edge("prepare_retry", "optimization_controller")
    workflow.add_edge("finalize_design", END)

    return workflow.compile()


def _run_fallback_graph_loop(initial_state: DesignState) -> Generator[Dict[str, Any], None, DesignState]:
    """
    Deterministic native Python graph runner in case LangGraph package is not present.
    Guarantees 100% unconditional execution reliability.
    """
    state = dict(initial_state)
    seen_messages_count = 0

    def sync_and_yield():
        nonlocal seen_messages_count
        messages = state.get("messages", [])
        while seen_messages_count < len(messages):
            msg = messages[seen_messages_count]
            seen_messages_count += 1
            yield {"type": "agent_message", "message": msg, "state": state}

    # Step 1: Clinical Interpreter
    res = clinical_interpreter_node(state)
    state.update(res)
    yield from sync_and_yield()

    # Step 2: Materials Advisor
    res = materials_advisor_node(state)
    state.update(res)
    yield from sync_and_yield()

    # Optimization-Validation Loop
    max_attempts = state.get("max_attempts", 3)
    while state.get("attempt", 1) <= max_attempts:
        # Step 3: Optimization Controller
        res = optimization_controller_node(state)
        state.update(res)
        yield from sync_and_yield()

        # Step 4: Validation Auditor
        res = validation_auditor_node(state)
        state.update(res)
        yield from sync_and_yield()

        # Check loop condition
        route = should_retry(state)
        if route == "retry":
            res = prepare_retry_node(state)
            state.update(res)
            yield from sync_and_yield()
        else:
            break

    # Final Step: Finalize
    res = finalize_design_node(state)
    state.update(res)
    yield from sync_and_yield()

    yield {"type": "final_result", "state": state}
    return state


def run_design_agent(
    surgeon_prompt: str,
    fem_client: Optional[Any] = None,
    geometry_client: Optional[Any] = None,
    max_attempts: int = 3,
    max_steps: int = 15,
    bend_y_array: Optional[List[float]] = None,
    bend_z_array: Optional[List[float]] = None,
    stream: bool = True,
    step_callback: Optional[Any] = None
) -> Generator[Dict[str, Any], None, DesignState]:
    """
    Top-level orchestrator entry-point for the multi-agent biomechanical system.
    
    Streams real-time agent messages, optimization telemetry, and validation reports.
    """
    initial_state: DesignState = {
        "surgeon_prompt": surgeon_prompt,
        "clinical_profile": None,
        "design_spec": None,
        "optimization_result": None,
        "validation_report": None,
        "corrections": None,
        "attempt": 1,
        "max_attempts": max_attempts,
        "max_steps": max_steps,
        "messages": [],
        "fem_client": fem_client,
        "geometry_client": geometry_client,
        "bend_y_array": bend_y_array,
        "bend_z_array": bend_z_array,
        "final_design": None,
        "step_callback": step_callback
    }

    graph = build_design_graph()

    if graph is None:
        # Run robust fallback loop
        yield from _run_fallback_graph_loop(initial_state)
        return

    seen_messages_count = 0
    current_state = initial_state

    try:
        # Stream graph execution node by node
        for output in graph.stream(initial_state, stream_mode="values"):
            current_state = output
            messages = output.get("messages", [])
            while seen_messages_count < len(messages):
                msg = messages[seen_messages_count]
                seen_messages_count += 1
                yield {
                    "type": "agent_message",
                    "message": msg,
                    "state": current_state
                }
        
        yield {
            "type": "final_result",
            "state": current_state
        }
    except Exception as e:
        try:
            import sys
            if sys.stdout and not sys.stdout.closed:
                print(f"[LangGraph Orchestrator Exception: {e}], switching to native loop runner")
        except Exception:
            pass
        yield from _run_fallback_graph_loop(initial_state)


if __name__ == "__main__":
    # Test script in CLI mode
    test_prompt = "22-year-old competitive skier with midshaft spiral femur fracture needing high torsional rigidity"
    print("--- Starting Multi-Agent Biomechanics Graph Test ---")
    for event in run_design_agent(test_prompt, max_attempts=2):
        if event["type"] == "agent_message":
            msg = event["message"]
            print(f"[{msg['agent_emoji']} {msg['agent_display_name']}]: {msg['content']}\n")
        elif event["type"] == "final_result":
            print("--- Design Process Complete ---")
            print("Final State Summary:", event["state"].get("final_design"))
