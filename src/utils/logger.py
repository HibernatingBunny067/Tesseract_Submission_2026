"""
Comprehensive Clinical & Engineering Audit Logger.

Records complete audit trails for every surgical design request:
1. User Raw Input Prompt
2. LLM / Agent Parsed Response & Clinical Rationale
3. Differentiable Adjoint Optimization Dynamics
4. In-Silico Verification Results (ASTM F382 / ISO 7206)
5. Final Manufacturing & Biomechanical Outcome
"""

import os
import json
import datetime
from typing import Dict, Any, Optional, Tuple, List, Union
from src.agent.agent import DesignRequest
from src.fem.materials import Biomaterial
from src.fem.validation import ClinicalValidationReport

LOGS_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../logs"))
os.makedirs(LOGS_DIR, exist_ok=True)

TEXT_LOG_PATH: str = os.path.join(LOGS_DIR, "clinical_audit.log")
JSONL_LOG_PATH: str = os.path.join(LOGS_DIR, "session_history.jsonl")


def log_user_prompt_and_llm_response(
    user_prompt: str,
    design_req: DesignRequest,
    engine: str = "Groq LPU / Local NLP"
) -> None:
    """Logs the raw NLP user request and structured agent response."""
    timestamp: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Formatted human-readable audit entry
    entry: str = f"""
================================================================================
[AUDIT LOG] NLP DESIGN REQUEST & AGENT INTERPRETATION
Timestamp: {timestamp}
Engine:    {engine}
--------------------------------------------------------------------------------
USER INPUT PROMPT:
"{user_prompt}"

AGENT STRUCTURED TRANSLATION:
• Clinical Objective:      {design_req.objective}
• Target Micro-Motion:     {design_req.target_fracture_displacement*1000:.3f} mm ({design_req.target_fracture_displacement*1e6:.0f} µm)
• Upper Mass Limit:        {design_req.max_mass*100:.0f}%
• Recommended Biomaterial: {design_req.recommended_material}
• Recommended Topology:    {getattr(design_req, 'recommended_tpms', 'Schwarz Primitive (P)')}
• Clinical Rationale:
  {design_req.clinical_rationale}
================================================================================
"""
    with open(TEXT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(entry + "\n")
        
    # Structured JSONL record
    json_entry: Dict[str, Any] = {
        "event": "nlp_request_parsed",
        "timestamp": timestamp,
        "engine": engine,
        "user_prompt": user_prompt,
        "objective": design_req.objective,
        "target_micro_motion_mm": design_req.target_fracture_displacement * 1000.0,
        "max_mass_fraction": design_req.max_mass,
        "recommended_material": design_req.recommended_material,
        "recommended_tpms": getattr(design_req, 'recommended_tpms', 'Schwarz Primitive (P)'),
        "clinical_rationale": design_req.clinical_rationale
    }
    with open(JSONL_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(json_entry) + "\n")


def log_full_optimization_and_validation(
    user_prompt: str,
    design_req: DesignRequest,
    material: Biomaterial,
    fidelity_mode: str,
    total_steps: int,
    initial_loss: float,
    final_loss: float,
    final_disp_mm: float,
    target_disp_mm: float,
    avg_porosity_pct: float,
    solid_mass_g: float,
    optimized_mass_g: float,
    tau_values: Union[Tuple[float, ...], List[float]],
    validation_report: ClinicalValidationReport
) -> None:
    """
    Logs complete end-to-end outcome including adjoint optimization stats
    and ASTM/ISO in-silico verification results.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mass_reduction_pct = ((solid_mass_g - optimized_mass_g) / solid_mass_g) * 100.0
    
    test_lines = ""
    for t in validation_report.tests:
        res_tag = "[PASS]" if t.passed else "[FAIL]"
        test_lines += f"  • {res_tag:<6} {t.name:<32} Measured: {t.measured_value:<25} Criteria: {t.target_criteria}\n"
        
    entry = f"""
================================================================================
[AUDIT LOG] COMPLETE BIOMECHANICAL OPTIMIZATION & IN-SILICO VERIFICATION RUN
Timestamp:        {timestamp}
Simulation Mode:  {fidelity_mode}
Material:         {material.name} ({material.code}, E={material.youngs_modulus_gpa} GPa)
--------------------------------------------------------------------------------
1. USER REQUEST & CLINICAL TARGET:
   • Prompt:        "{user_prompt}"
   • Objective:     {design_req.objective}
   • Target Motion: {target_disp_mm:.2f} mm

2. ADJOINT OPTIMIZATION METRICS:
   • Steps to Convergence: {total_steps}
   • Objective Loss:       {initial_loss:.4f} -> {final_loss:.4f}
   • Final TPMS Values:    tau_prox={tau_values[0]:.3f}, tau_bridge={tau_values[1]:.3f}, tau_dist={tau_values[2]:.3f}
   • Achieved Porosity:    {avg_porosity_pct:.1f}%

3. BIOMECHANICAL OUTCOME:
   • Baseline Solid Mass:  {solid_mass_g:.1f} g
   • Optimized TPMS Mass:  {optimized_mass_g:.1f} g (-{mass_reduction_pct:.1f}% Reduction)
   • Achieved Micro-Motion:{final_disp_mm:.3f} mm (Target: {target_disp_mm:.2f} mm)

4. IN-SILICO CLINICAL VERIFICATION (ASTM F382 / ISO 7206):
   • Overall Verdict:      {validation_report.overall_verdict} ({validation_report.overall_score_pct:.0f}%)
{test_lines}
================================================================================
"""
    with open(TEXT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(entry + "\n")
        
    # Structured JSONL entry
    json_record = {
        "event": "optimization_run_completed",
        "timestamp": timestamp,
        "fidelity_mode": fidelity_mode,
        "material": {
            "name": material.name,
            "code": material.code,
            "modulus_gpa": material.youngs_modulus_gpa,
            "density": material.density_g_cm3
        },
        "user_prompt": user_prompt,
        "objective": design_req.objective,
        "target_micro_motion_mm": target_disp_mm,
        "achieved_micro_motion_mm": final_disp_mm,
        "avg_porosity_pct": avg_porosity_pct,
        "solid_mass_g": solid_mass_g,
        "optimized_mass_g": optimized_mass_g,
        "mass_reduction_pct": mass_reduction_pct,
        "tpms_tau_values": [float(tau_values[0]), float(tau_values[1]), float(tau_values[2])],
        "optimization_steps": total_steps,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "verification_verdict": validation_report.overall_verdict,
        "verification_score_pct": validation_report.overall_score_pct,
        "verification_tests": [
            {
                "name": t.name,
                "standard": t.standard,
                "measured": t.measured_value,
                "criteria": t.target_criteria,
                "passed": t.passed
            } for t in validation_report.tests
        ]
    }
    with open(JSONL_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(json_record) + "\n")


AGENT_TEXT_LOG_PATH: str = os.path.join(LOGS_DIR, "agent_deliberation.log")
AGENT_JSONL_PATH: str = os.path.join(LOGS_DIR, "agent_session.jsonl")


def log_agent_message(
    agent_name: str,
    display_name: str,
    emoji: str,
    message_type: str,
    content: str,
    data: Optional[Dict[str, Any]] = None
) -> None:
    """
    Dedicated logger for real-time agent thoughts, clinical interpretations,
    material recommendations, and audit corrections.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 1. Human-readable text audit entry
    text_entry = (
        f"[{timestamp}] {emoji} [{display_name.upper()}] ({message_type.upper()}):\n"
        f"{content}\n"
    )
    if data:
        formatted_data = json.dumps(data, indent=2, default=str)
        text_entry += f"--- Attached Payload ---\n{formatted_data}\n"
    text_entry += "-" * 75 + "\n"
    
    try:
        with open(AGENT_TEXT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(text_entry)
    except Exception:
        pass

    # 2. Structured JSONL entry
    json_entry = {
        "timestamp": timestamp,
        "agent_name": agent_name,
        "display_name": display_name,
        "emoji": emoji,
        "message_type": message_type,
        "content": content,
        "data": data
    }
    try:
        with open(AGENT_JSONL_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(json_entry, default=str) + "\n")
    except Exception:
        pass


def log_agent_session_summary(
    surgeon_prompt: str,
    attempts: int,
    verdict: str,
    messages_count: int,
    final_design: Optional[Dict[str, Any]] = None
) -> None:
    """Logs the final high-level agent session sign-off."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_entry = f"""
================================================================================
🤖 MULTI-AGENT DELIBERATION SESSION COMPLETED
Timestamp:        {timestamp}
Surgeon Prompt:   "{surgeon_prompt}"
Attempts Taken:   {attempts}
Final Verdict:    {verdict}
Total Messages:   {messages_count}
================================================================================
"""
    try:
        with open(AGENT_TEXT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(summary_entry)
    except Exception:
        pass
