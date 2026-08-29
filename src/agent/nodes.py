"""
LangGraph node functions for the multi-agent biomechanical design system.

Each node is a specialist agent that reads from and writes to the shared DesignState.
Nodes call into existing project modules (optimize.py, validation.py, materials.py)
without modifying them.
"""

from __future__ import annotations

import json
import traceback
from typing import Any, Dict

from src.agent.state import DesignState, create_message
from src.agent.prompts import (
    CLINICAL_INTERPRETER_PROMPT,
    MATERIALS_ADVISOR_PROMPT,
    OPTIMIZATION_CONTROLLER_PROMPT,
    VALIDATION_AUDITOR_PROMPT,
)
from src.agent.llm_provider import get_provider, generate_with_fallback, LLMProvider


def _safe_log(msg: str):
    """Safely log messages to stdout without crashing if the stream was closed."""
    try:
        import sys
        if sys.stdout and not sys.stdout.closed:
            print(msg)
    except Exception:
        pass


def _get_providers(role: str) -> list[LLMProvider]:
    """Build a fallback chain of LLM providers for a given role."""
    import os
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    providers: list[LLMProvider] = []
    gemini_key = os.environ.get("GEMINI_API_KEY")
    groq_key = os.environ.get("GROQ_API_KEY")

    if role == "materials_advisor":
        # Groq first (fast structured output), then Gemini, then Ollama
        if groq_key:
            from src.agent.llm_provider import GroqProvider
            providers.append(GroqProvider(api_key=groq_key))
        if gemini_key:
            from src.agent.llm_provider import GeminiProvider
            providers.append(GeminiProvider(api_key=gemini_key))
        from src.agent.llm_provider import OllamaProvider
        providers.append(OllamaProvider())
    else:
        # Gemini first (strong reasoning), then Groq, then Ollama
        if gemini_key:
            from src.agent.llm_provider import GeminiProvider
            providers.append(GeminiProvider(api_key=gemini_key))
        if groq_key:
            from src.agent.llm_provider import GroqProvider
            providers.append(GroqProvider(api_key=groq_key))
        from src.agent.llm_provider import OllamaProvider
        providers.append(OllamaProvider())

    if not providers:
        raise RuntimeError(
            "No LLM providers available. Set GEMINI_API_KEY or GROQ_API_KEY, "
            "or ensure Ollama is running locally."
        )
    return providers


def _llm_call(role: str, system_prompt: str, user_message: str) -> dict:
    """Make an LLM call and parse the JSON response."""
    providers = _get_providers(role)
    raw = generate_with_fallback(
        providers=providers,
        messages=[{"role": "user", "content": user_message}],
        system_prompt=system_prompt,
        temperature=0.1,
        json_mode=True,
    )
    # Strip markdown code fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)
    return json.loads(cleaned)


# ---------------------------------------------------------------------------
# Node 1: Clinical Interpreter
# ---------------------------------------------------------------------------

def clinical_interpreter_node(state: DesignState) -> dict:
    """Translate surgeon prompt into a structured clinical profile."""
    prompt = state["surgeon_prompt"]
    messages = list(state.get("messages", []))

    messages.append(create_message(
        "clinical_interpreter",
        f"Analyzing clinical case: \"{prompt}\"",
        "status",
    ))

    try:
        result = _llm_call(
            role="clinical_interpreter",
            system_prompt=CLINICAL_INTERPRETER_PROMPT,
            user_message=f"Analyze this clinical case and extract structured parameters:\n\n\"{prompt}\"",
        )

        # Validate and clamp target micro-motion to physical bounds
        target = float(result.get("target_micro_motion_m", 0.0002))
        if target > 1.0:
            target *= 1e-6  # microns → meters
        elif target > 0.001:
            target *= 1e-3  # mm → meters
        target = max(0.00008, min(0.00040, target))
        result["target_micro_motion_m"] = target

        reasoning = result.get("clinical_reasoning", "Clinical analysis complete.")
        messages.append(create_message(
            "clinical_interpreter",
            f"{reasoning}\n\n"
            f"• Target micro-motion: {target * 1000:.2f} mm\n"
            f"• Loading pattern: {result.get('loading_pattern', 'N/A')}\n"
            f"• Objective: {result.get('clinical_objective', 'N/A')}",
            "result",
            data=result,
        ))

        return {"clinical_profile": result, "messages": messages}

    except Exception as e:
        # Fallback to existing local NLP parser
        _safe_log(f"[ClinicalInterpreter] LLM failed ({e}), falling back to local NLP parser")
        try:
            from src.agent.agent import parse_design_request
            fallback = parse_design_request(prompt)
            result = {
                "target_micro_motion_m": fallback.target_fracture_displacement,
                "loading_pattern": "axial_bending",
                "healing_timeline_months": 6,
                "patient_demographics": "standard adult",
                "clinical_objective": fallback.objective,
                "contraindications": [],
                "clinical_reasoning": fallback.clinical_rationale or "Parsed via local NLP fallback.",
            }
            # Infer loading pattern from TPMS recommendation
            if "Diamond" in fallback.recommended_tpms:
                result["loading_pattern"] = "high_torsion"
            elif "Gyroid" in fallback.recommended_tpms:
                result["loading_pattern"] = "combined_shear"

            messages.append(create_message(
                "clinical_interpreter",
                f"[Fallback NLP] {result['clinical_reasoning']}\n\n"
                f"• Target: {result['target_micro_motion_m'] * 1000:.2f} mm\n"
                f"• Objective: {result['clinical_objective']}",
                "result",
                data=result,
            ))
            return {"clinical_profile": result, "messages": messages}
        except Exception as fallback_err:
            _safe_log(f"[ClinicalInterpreter] Fallback also failed: {fallback_err}")
            # Ultimate fallback: safe defaults
            result = {
                "target_micro_motion_m": 0.0002,
                "loading_pattern": "axial_bending",
                "healing_timeline_months": 6,
                "patient_demographics": "standard adult",
                "clinical_objective": "Callus Stimulation & Mass Minimization",
                "contraindications": [],
                "clinical_reasoning": "Using safe clinical defaults due to parsing failure.",
            }
            messages.append(create_message(
                "clinical_interpreter",
                "[Default Fallback] Using standard parameters: 0.20mm target, axial loading.",
                "result",
                data=result,
            ))
            return {"clinical_profile": result, "messages": messages}


# ---------------------------------------------------------------------------
# Node 2: Materials & Topology Advisor
# ---------------------------------------------------------------------------

def materials_advisor_node(state: DesignState) -> dict:
    """Select optimal biomaterial and TPMS architecture."""
    clinical = state["clinical_profile"]
    messages = list(state.get("messages", []))

    messages.append(create_message(
        "materials_advisor",
        f"Selecting material and TPMS topology for: {clinical.get('clinical_objective', 'N/A')} "
        f"(loading: {clinical.get('loading_pattern', 'N/A')})",
        "status",
    ))

    try:
        result = _llm_call(
            role="materials_advisor",
            system_prompt=MATERIALS_ADVISOR_PROMPT,
            user_message=(
                f"Select the optimal biomaterial and TPMS architecture for this clinical profile:\n\n"
                f"Clinical Objective: {clinical.get('clinical_objective', 'N/A')}\n"
                f"Target Micro-Motion: {clinical.get('target_micro_motion_m', 0.0002) * 1000:.2f} mm\n"
                f"Loading Pattern: {clinical.get('loading_pattern', 'axial_bending')}\n"
                f"Patient Demographics: {clinical.get('patient_demographics', 'standard adult')}\n"
                f"Healing Timeline: {clinical.get('healing_timeline_months', 6)} months\n"
                f"Contraindications: {clinical.get('contraindications', [])}"
            ),
        )

        # Validate material name
        valid_materials = ["Ti-6Al-4V (Grade 5 Titanium)", "316L Stainless Steel"]
        mat_name = result.get("material_name", "")
        # Map variations to canonical names
        if "ti" in mat_name.lower() or "titanium" in mat_name.lower() or "grade 5" in mat_name.lower():
            result["material_name"] = "Ti-6Al-4V (Grade 5 Titanium)"
        elif "steel" in mat_name.lower() or "316" in mat_name.lower():
            result["material_name"] = "316L Stainless Steel"
        else:
            result["material_name"] = "Ti-6Al-4V (Grade 5 Titanium)"

        # Validate TPMS type
        tpms = result.get("tpms_type", "")
        if "diamond" in tpms.lower():
            result["tpms_type"] = "Schwarz Diamond (D)"
        elif "gyroid" in tpms.lower():
            result["tpms_type"] = "Schoen Gyroid (G)"
        else:
            result["tpms_type"] = "Schwarz Primitive (P)"

        reasoning = result.get("material_reasoning", "Material selection complete.")
        messages.append(create_message(
            "materials_advisor",
            f"{reasoning}\n\n"
            f"• Material: {result['material_name']}\n"
            f"• TPMS: {result['tpms_type']}",
            "result",
            data=result,
        ))

        return {"design_spec": result, "messages": messages}

    except Exception as e:
        _safe_log(f"[MaterialsAdvisor] LLM failed ({e}), using heuristic fallback")
        # Heuristic fallback based on clinical profile
        loading = clinical.get("loading_pattern", "axial_bending")
        tpms_map = {
            "high_torsion": "Schwarz Diamond (D)",
            "combined_shear": "Schoen Gyroid (G)",
            "axial_bending": "Schwarz Primitive (P)",
            "max_rigidity": "Schwarz Diamond (D)",
        }
        target = clinical.get("target_micro_motion_m", 0.0002)
        # Rigid designs need lower bridge porosity
        tau_bridge = 0.35 if target < 0.00016 else (0.70 if target > 0.00025 else 0.50)

        result = {
            "material_name": "Ti-6Al-4V (Grade 5 Titanium)",
            "tpms_type": tpms_map.get(loading, "Schwarz Primitive (P)"),
            "initial_params": {
                "cell_size_m": 0.005,
                "tau_bridge": tau_bridge,
                "tau_anchors": 0.20,
                "tau_transitions": 0.35,
                "skin_thickness_m": 0.0005,
                "screw_spacing_m": 0.0145,
                "bridge_span_m": 0.030,
                "fillet_radius_m": 0.0012,
            },
            "material_reasoning": f"Heuristic selection: Ti-6Al-4V + {tpms_map.get(loading, 'Primitive')} for {loading} loading.",
        }
        messages.append(create_message(
            "materials_advisor",
            f"[Heuristic Fallback] {result['material_reasoning']}",
            "result",
            data=result,
        ))
        return {"design_spec": result, "messages": messages}


# ---------------------------------------------------------------------------
# Node 3: Optimization Controller
# ---------------------------------------------------------------------------

def optimization_controller_node(state: DesignState) -> dict:
    """Run the gradient-based optimization using existing optimize.py."""
    clinical = state["clinical_profile"]
    design = state["design_spec"]
    corrections = state.get("corrections")
    attempt = state.get("attempt", 1)
    messages = list(state.get("messages", []))

    # Look up material properties from existing database
    from src.fem.materials import BIOMATERIALS
    mat_name = design.get("material_name", "Ti-6Al-4V (Grade 5 Titanium)")
    material = BIOMATERIALS.get(mat_name, BIOMATERIALS["Ti-6Al-4V (Grade 5 Titanium)"])

    target_disp = clinical.get("target_micro_motion_m", 0.0002)
    objective = clinical.get("clinical_objective", "Callus Stimulation & Mass Minimization")

    # Determine initial parameters
    init_params = design.get("initial_params", {})

    # Default optimization kwargs
    opt_kwargs = {
        "target_fracture_displacement": target_disp,
        "objective": objective,
        "max_mass": 0.60,
        "material_modulus_gpa": material.youngs_modulus_gpa,
        "tpms_ga_exponent": material.tpms_ga_exponent,
        "yield_strength_mpa": material.yield_strength_mpa,
        "init_cell_size": init_params.get("cell_size_m", 0.005),
        "init_t_top": init_params.get("skin_thickness_m", 0.0004),
        "init_t_bot": init_params.get("skin_thickness_m", 0.0004),
        "init_screw_spacing": init_params.get("screw_spacing_m", 0.0145),
        "init_bridge_span": init_params.get("bridge_span_m", 0.030),
        "init_fillet_radius": init_params.get("fillet_radius_m", 0.0012),
        "max_steps": state.get("max_steps", 15) or 15,
        "fem_client": state.get("fem_client"),
        "geometry_client": state.get("geometry_client"),
    }

    # If this is a correction run, apply adjustments
    if corrections and attempt > 1:
        adjusted = corrections.get("adjusted_params", {})
        # Map correction params to optimizer kwargs
        param_map = {
            "tau_bridge": None,          # handled via objective heuristics
            "tau_anchors": None,         # handled via objective heuristics
            "cell_size_m": "init_cell_size",
            "skin_thickness_m": "init_t_top",
            "t_top_m": "init_t_top",
            "t_bot_m": "init_t_bot",
            "screw_spacing_m": "init_screw_spacing",
            "bridge_span_m": "init_bridge_span",
            "fillet_radius_m": "init_fillet_radius",
        }
        for param, value in adjusted.items():
            kwarg_name = param_map.get(param)
            if kwarg_name and kwarg_name in opt_kwargs:
                opt_kwargs[kwarg_name] = float(value)
            # Also update skin_thickness for t_bot if t_top is set
            if param == "skin_thickness_m":
                opt_kwargs["init_t_bot"] = float(value)

        # Adjust step count for correction runs
        opt_kwargs["max_steps"] = corrections.get("adjusted_max_steps", state.get("max_steps", 15) or 15)

        messages.append(create_message(
            "optimization_controller",
            f"Correction run (attempt {attempt}/{state.get('max_attempts', 3)}). "
            f"Applying adjustments: {adjusted}. Running {opt_kwargs['max_steps']} steps.",
            "status",
        ))
    else:
        messages.append(create_message(
            "optimization_controller",
            f"Starting optimization: target {target_disp * 1000:.2f}mm, "
            f"{mat_name}, {opt_kwargs['max_steps']} steps.",
            "status",
        ))

    # Run the existing optimizer — unchanged
    from src.agent.optimize import run_optimization

    step_history = []
    final_step = None

    try:
        for step_data in run_optimization(**opt_kwargs):
            step_history.append(step_data)
            final_step = step_data

        if final_step is None:
            raise RuntimeError("Optimization produced no output steps")

        # Build optimization result
        opt_result = {
            "final_theta": [
                final_step.get("cell_size_mm", 5.0),
                final_step.get("tau_p_anc", 0.35),
                final_step.get("tau_p_tra", 0.45),
                final_step.get("tau_bridge", 0.55),
                final_step.get("tau_d_tra", 0.45),
                final_step.get("tau_d_anc", 0.35),
                final_step.get("sigma_blend", 0.015) / 0.010,  # denormalize
                final_step.get("t_top_mm", 0.5),
                final_step.get("t_bottom_mm", 0.5),
                final_step.get("screw_spacing_mm", 14.5),
                final_step.get("bridge_span_mm", 30.0),
                final_step.get("fillet_radius_mm", 1.2),
            ],
            "final_metrics": {
                "loss": final_step.get("loss", 0.0),
                "frac_disp": final_step.get("frac_disp", 0.0),
                "compliance": final_step.get("compliance", 0.0),
                "mean_porosity": final_step.get("mean_porosity", 0.0),
                "mass_fraction": final_step.get("mass_fraction", 0.0),
            },
            "steps_taken": len(step_history),
            "converged": len(step_history) < opt_kwargs["max_steps"],
            "step_history": step_history,
        }

        # Convergence analysis via LLM (best-effort, falls back to heuristic)
        convergence_analysis = _analyze_convergence(step_history, target_disp)
        opt_result["convergence_analysis"] = convergence_analysis

        achieved_mm = final_step.get("frac_disp", 0.0) * 1000
        messages.append(create_message(
            "optimization_controller",
            f"Optimization complete in {len(step_history)} steps.\n\n"
            f"• Micro-motion: {achieved_mm:.3f} mm (target: {target_disp * 1000:.2f} mm)\n"
            f"• Mass fraction: {final_step.get('mass_fraction', 0):.3f}\n"
            f"• Mean porosity: {final_step.get('mean_porosity', 0):.1%}\n"
            f"• {convergence_analysis}",
            "result",
            data={"steps_taken": len(step_history), "achieved_mm": achieved_mm},
        ))

        return {"optimization_result": opt_result, "messages": messages}

    except Exception as e:
        _safe_log(f"[OptimizationController] Error during optimization: {e}")
        messages.append(create_message(
            "optimization_controller",
            f"Optimization failed: {str(e)}",
            "status",
        ))
        # Return empty result so validation can still run and report failure
        return {
            "optimization_result": {
                "final_theta": [5.0, 0.35, 0.45, 0.55, 0.45, 0.35, 1.5, 0.5, 0.5, 14.5, 30.0, 1.2],
                "final_metrics": {"loss": 999.0, "frac_disp": 0.0, "compliance": 0.0, "mean_porosity": 0.0, "mass_fraction": 0.0},
                "steps_taken": 0,
                "converged": False,
                "step_history": [],
                "convergence_analysis": f"Optimization failed: {str(e)}",
            },
            "messages": messages,
        }


def _analyze_convergence(step_history: list[dict], target_disp: float) -> str:
    """Heuristic convergence analysis — no LLM needed."""
    if not step_history:
        return "No optimization steps recorded."

    losses = [s.get("loss", 0.0) for s in step_history]
    n = len(losses)

    if n < 3:
        return f"Short run ({n} steps). Insufficient data for convergence analysis."

    # Check for plateau
    final_loss = losses[-1]
    mid_loss = losses[n // 2] if n > 4 else losses[0]
    initial_loss = losses[0]

    reduction_pct = (1.0 - final_loss / (initial_loss + 1e-9)) * 100
    late_reduction = abs(losses[-1] - losses[-3]) / (abs(losses[-3]) + 1e-9) * 100

    achieved = step_history[-1].get("frac_disp", 0.0)
    error_pct = abs(achieved - target_disp) / (target_disp + 1e-9) * 100

    parts = [f"Loss reduced {reduction_pct:.0f}% over {n} steps."]

    if late_reduction < 1.0:
        parts.append("Converged (loss plateau detected in final steps).")
    elif late_reduction < 5.0:
        parts.append("Near convergence (small improvements in final steps).")
    else:
        parts.append("Still improving — additional steps may help.")

    if error_pct < 10:
        parts.append(f"Target tracking: excellent ({error_pct:.1f}% error).")
    elif error_pct < 20:
        parts.append(f"Target tracking: acceptable ({error_pct:.1f}% error).")
    else:
        parts.append(f"Target tracking: poor ({error_pct:.1f}% error) — may need re-optimization.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Node 4: Validation Auditor
# ---------------------------------------------------------------------------

def validation_auditor_node(state: DesignState) -> dict:
    """Run validation suite and diagnose failures."""
    opt_result = state["optimization_result"]
    clinical = state["clinical_profile"]
    design = state["design_spec"]
    attempt = state.get("attempt", 1)
    messages = list(state.get("messages", []))

    # Look up material
    from src.fem.materials import BIOMATERIALS
    mat_name = design.get("material_name", "Ti-6Al-4V (Grade 5 Titanium)")
    material = BIOMATERIALS.get(mat_name, BIOMATERIALS["Ti-6Al-4V (Grade 5 Titanium)"])

    target_disp = clinical.get("target_micro_motion_m", 0.0002)
    metrics = opt_result.get("final_metrics", {})
    theta = opt_result.get("final_theta", [])

    # Extract tau values for validation
    tau_values = theta[1:6] if len(theta) >= 6 else [0.35, 0.45, 0.55, 0.45, 0.35]
    achieved_disp = metrics.get("frac_disp", 0.0)
    porosity = metrics.get("mean_porosity", 0.5) * 100  # as percentage

    messages.append(create_message(
        "validation_auditor",
        f"Running ASTM F382 & ISO 7206 validation battery (attempt {attempt})...",
        "status",
    ))

    # Run the existing validation suite — unchanged
    from src.fem.validation import run_insilico_validation_suite
    report = run_insilico_validation_suite(
        tau_values=tau_values,
        target_disp_m=target_disp,
        achieved_disp_m=achieved_disp,
        avg_porosity_pct=porosity,
        material=material,
    )

    # Build validation report dict
    failed_tests = [t.name for t in report.tests if not t.passed]
    test_dicts = [
        {
            "name": t.name,
            "standard": t.standard,
            "measured_value": t.measured_value,
            "target_criteria": t.target_criteria,
            "passed": t.passed,
            "safety_margin": t.safety_margin,
        }
        for t in report.tests
    ]

    validation_dict = {
        "verdict": report.overall_verdict,
        "score_pct": report.overall_score_pct,
        "tests": test_dicts,
        "failed_tests": failed_tests,
        "raw_report": report,  # pass through for app.py
    }

    # Build result message
    test_lines = []
    for t in report.tests:
        emoji = "✅" if t.passed else "❌"
        test_lines.append(f"  {emoji} {t.name}: {t.measured_value} ({t.target_criteria})")
    test_summary = "\n".join(test_lines)

    if not failed_tests:
        # All passed — no corrections needed
        messages.append(create_message(
            "validation_auditor",
            f"**{report.overall_verdict}**\n\n{test_summary}\n\n"
            f"All {len(report.tests)} tests passed. Design is clinically approved.",
            "result",
            data=validation_dict,
        ))
        return {
            "validation_report": validation_dict,
            "corrections": None,
            "messages": messages,
        }

    # Tests failed — generate correction prescription
    messages.append(create_message(
        "validation_auditor",
        f"**{report.overall_verdict}**\n\n{test_summary}\n\n"
        f"Failed: {', '.join(failed_tests)}. Generating correction prescription...",
        "status",
        data=validation_dict,
    ))

    # Try LLM-based diagnosis, fall back to heuristic
    corrections = _generate_corrections(
        failed_tests=failed_tests,
        test_results=test_dicts,
        current_theta=theta,
        target_disp=target_disp,
        achieved_disp=achieved_disp,
        porosity=porosity,
        material=material,
    )

    validation_dict["diagnosis"] = corrections.get("diagnosis", "")
    validation_dict["correction_prescription"] = corrections

    messages.append(create_message(
        "validation_auditor",
        f"Diagnosis: {corrections.get('diagnosis', 'N/A')}\n\n"
        f"Corrections: {corrections.get('reasoning', 'N/A')}\n"
        f"Risk: {corrections.get('risk_assessment', 'N/A')}",
        "correction",
        data=corrections,
    ))

    return {
        "validation_report": validation_dict,
        "corrections": corrections,
        "messages": messages,
    }


def _generate_corrections(
    failed_tests: list[str],
    test_results: list[dict],
    current_theta: list[float],
    target_disp: float,
    achieved_disp: float,
    porosity: float,
    material: Any,
) -> dict:
    """Generate correction prescription — tries LLM first, falls back to heuristics."""

    # Try LLM-based diagnosis
    try:
        result = _llm_call(
            role="validation_auditor",
            system_prompt=VALIDATION_AUDITOR_PROMPT,
            user_message=(
                f"The following validation tests FAILED: {failed_tests}\n\n"
                f"Full test results:\n{json.dumps(test_results, indent=2)}\n\n"
                f"Current parameters (theta):\n"
                f"  cell_size: {current_theta[0]:.2f} mm\n"
                f"  tau_anchors: {current_theta[1]:.3f}, {current_theta[5]:.3f}\n"
                f"  tau_transitions: {current_theta[2]:.3f}, {current_theta[4]:.3f}\n"
                f"  tau_bridge: {current_theta[3]:.3f}\n"
                f"  skin_top: {current_theta[7]:.2f} mm, skin_bot: {current_theta[8]:.2f} mm\n"
                f"  screw_spacing: {current_theta[9]:.1f} mm\n"
                f"  bridge_span: {current_theta[10]:.1f} mm\n"
                f"  fillet_radius: {current_theta[11]:.2f} mm\n\n"
                f"Target micro-motion: {target_disp * 1000:.3f} mm\n"
                f"Achieved micro-motion: {achieved_disp * 1000:.3f} mm\n"
                f"Average porosity: {porosity:.1f}%\n"
                f"Material: {material.name} (σ_y={material.yield_strength_mpa} MPa)\n\n"
                f"Prescribe specific parameter corrections to fix the failed tests."
            ),
        )
        # Ensure the correction has the right structure
        if "correction_prescription" in result:
            corrections = result["correction_prescription"]
            corrections["diagnosis"] = result.get("diagnosis", "")
            corrections["risk_assessment"] = result.get("risk_assessment", "")
            return corrections
        else:
            result.setdefault("adjusted_params", {})
            result.setdefault("adjusted_max_steps", 15)
            result.setdefault("reasoning", result.get("diagnosis", "LLM correction"))
            return result

    except Exception as e:
        _safe_log(f"[ValidationAuditor] LLM correction failed ({e}), using heuristic")

    # Heuristic fallback corrections
    adjusted_params = {}
    reasoning_parts = []

    for test_name in failed_tests:
        if "Yield" in test_name or "Safety" in test_name:
            # Low FoS → increase density, thicken skins
            adjusted_params["skin_thickness_m"] = min(
                (current_theta[7] + 0.15) * 1e-3, 0.002
            )
            reasoning_parts.append(
                "Increasing skin thickness to improve bending resistance and raise FoS."
            )
        elif "Stress Shielding" in test_name:
            # Too stiff → increase porosity
            adjusted_params["tau_bridge"] = min(current_theta[3] + 0.10, 1.40)
            reasoning_parts.append(
                "Increasing bridge porosity to reduce plate stiffness and stress shielding."
            )
        elif "Micro-Motion" in test_name:
            error = achieved_disp - target_disp
            if error > 0:  # overshoot (too flexible)
                adjusted_params["tau_bridge"] = max(current_theta[3] - 0.10, 0.15)
                reasoning_parts.append("Reducing bridge porosity to decrease micro-motion.")
            else:  # undershoot (too rigid)
                adjusted_params["tau_bridge"] = min(current_theta[3] + 0.10, 1.40)
                reasoning_parts.append("Increasing bridge porosity to increase micro-motion.")
        elif "Fatigue" in test_name:
            adjusted_params["skin_thickness_m"] = min(
                (current_theta[7] + 0.10) * 1e-3, 0.002
            )
            reasoning_parts.append("Reinforcing skins to improve fatigue endurance margin.")

    return {
        "adjusted_params": adjusted_params,
        "adjusted_max_steps": 15,
        "reasoning": " ".join(reasoning_parts) if reasoning_parts else "Applying conservative structural reinforcement.",
        "diagnosis": f"Failed tests: {', '.join(failed_tests)}. Applying heuristic corrections.",
        "risk_assessment": "Heuristic corrections may affect other test margins. Monitor all 4 tests after re-optimization.",
    }
