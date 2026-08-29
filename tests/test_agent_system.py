"""Integration tests for the unified agent system.

Validates that all new modules can be imported and their key
interfaces are accessible, without requiring a running FEM solver
or LLM API keys.
"""
from __future__ import annotations

import sys
import os
import importlib

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_import_state() -> None:
    from src.agent.state import DesignState, AgentMessage, create_message, AGENT_INFO
    assert "clinical_interpreter" in AGENT_INFO
    msg = create_message("clinical_interpreter", "test content", "status")
    assert msg["agent_name"] == "clinical_interpreter"
    assert msg["content"] == "test content"
    assert msg["message_type"] == "status"
    print("[PASS] state.py imports and message creation OK")


def test_import_prompts() -> None:
    from src.agent.prompts import (
        CLINICAL_INTERPRETER_PROMPT,
        MATERIALS_ADVISOR_PROMPT,
        OPTIMIZATION_CONTROLLER_PROMPT,
        VALIDATION_AUDITOR_PROMPT,
    )
    assert len(CLINICAL_INTERPRETER_PROMPT) > 100
    assert len(MATERIALS_ADVISOR_PROMPT) > 100
    assert len(OPTIMIZATION_CONTROLLER_PROMPT) > 100
    assert len(VALIDATION_AUDITOR_PROMPT) > 100
    print("[PASS] prompts.py imports OK")


def test_import_llm_provider() -> None:
    from src.agent.llm_provider import (
        LLMProvider,
        GeminiProvider,
        GroqProvider,
        OllamaProvider,
        get_provider,
        generate_with_fallback,
    )
    assert issubclass(GeminiProvider, LLMProvider)
    assert issubclass(GroqProvider, LLMProvider)
    assert issubclass(OllamaProvider, LLMProvider)
    print("[PASS] llm_provider.py imports OK")


def test_import_nodes() -> None:
    from src.agent.nodes import (
        clinical_interpreter_node,
        materials_advisor_node,
        optimization_controller_node,
        validation_auditor_node,
    )
    assert callable(clinical_interpreter_node)
    assert callable(materials_advisor_node)
    assert callable(optimization_controller_node)
    assert callable(validation_auditor_node)
    print("[PASS] nodes.py imports OK")


def test_import_graph() -> None:
    from src.agent.graph import run_design_agent
    assert callable(run_design_agent)
    print("[PASS] graph.py imports OK")


def test_import_optimize_cad() -> None:
    from src.agent.optimize_cad import run_cad_shape_optimization
    assert callable(run_cad_shape_optimization)
    print("[PASS] optimize_cad.py imports OK")


def test_import_morph() -> None:
    from src.geometry.morph import apply_pygem_ffd, build_ffd_warper, _bernstein
    import numpy as np
    # Verify Bernstein basis polynomial at t=0.5 for degree 2, index 1
    t = np.array([0.5])
    b = _bernstein(2, 1, t)
    assert abs(b[0] - 0.5) < 1e-10, f"Expected 0.5, got {b[0]}"
    print("[PASS] morph.py imports and Bernstein basis OK")


def test_import_forward_rebuild() -> None:
    from src.fem.forward import rebuild_for_morphed_mesh
    assert callable(rebuild_for_morphed_mesh)
    print("[PASS] forward.py rebuild_for_morphed_mesh OK")


if __name__ == "__main__":
    test_import_state()
    test_import_prompts()
    test_import_llm_provider()
    test_import_nodes()
    test_import_graph()
    test_import_optimize_cad()
    test_import_morph()
    test_import_forward_rebuild()
    print("\n[ALL PASS] All integration import tests succeeded.")
