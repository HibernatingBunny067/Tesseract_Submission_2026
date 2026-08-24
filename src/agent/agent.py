# Natural Language Agent: Surgeon ke prompt se mathematical parameters extract karta hai
# Groq LLM API use karega agar key mili, nahi toh local regex/NLP fallback chalaega

import os
import sys
import re
import json
import requests
from typing import Optional
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# .env file se environment variables load karo
try:
    from dotenv import load_dotenv
    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.env"))
    load_dotenv(env_path)
except Exception:
    pass

# Manual fallback agar dotenv miss ho jaye
if not os.getenv("GROQ_API_KEY"):
    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.env"))
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    clean_val = v.strip().strip('"').strip("'")
                    os.environ[k.strip()] = clean_val


class DesignRequest(BaseModel):
    objective: str = Field(description="Clinical optimization objective name")
    target_fracture_displacement: float = Field(description="Target fracture site micro-motion in meters")
    max_mass: float = Field(description="Upper mass fraction limit (0.0 - 1.0)")
    recommended_material: str = Field(default="Ti-6Al-4V (Grade 5 Titanium)", description="Suggested biomaterial")
    recommended_tpms: str = Field(default="Schwarz Primitive (P)", description="Suggested minimal surface metamaterial architecture")
    fillet_radius_mm: float = Field(default=1.2, description="Edge fillet radius in mm")
    screw_spacing_mm: float = Field(default=15.0, description="Screw hole pitch/spacing in mm")
    clinical_rationale: str = Field(default="", description="Clinical biomechanical reasoning")


SYSTEM_PROMPT = """You are an expert Orthopaedic Biomechanics AI Agent specializing in patient-specific implant co-design.
Your task is to analyze surgeon requirements and translate natural language prompts into precise numerical parameters for a Differentiable FEM Metamaterial Optimizer.

Key Biomechanical Principles:
1. Interfragmentary Micro-Motion (Secondary Healing / Callus Stimulation):
   - Optimal Callus Formation Window: 0.15mm to 0.35mm (0.00015m to 0.00035m).
   - Rigid fixation (Athletes / Segmental fractures / Young patients): 0.08mm to 0.15mm.
   - Osteoporotic bone / Elderly / Compliant: 0.30mm to 0.40mm with maximal porosity to prevent stress shielding.

2. Certified Orthopaedic Biomaterials:
   - "Ti-6Al-4V (Grade 5 Titanium)" (Gold standard biocompatible alloy, E=110 GPa, density=4.43 g/cm³, yield=880 MPa)
   - "316L Stainless Steel" (Standard trauma fixation / cost-effective high ductility, E=193 GPa, density=8.00 g/cm³, yield=220 MPa)

3. Minimal Surface Metamaterial Architectures:
   - "Schwarz Primitive (P)" (Highest fluid permeability & nutrient transport, optimal for bone callus stimulation and standard secondary healing).
   - "Schoen Gyroid (G)" (Exceptional shear resistance & isotropic compliance, optimal for oblique fractures, dynamic athletic gait, and active patients).
   - "Schwarz Diamond (D)" (Maximum torsional stiffness & multi-axial bending stability, optimal for spiral fractures, athletes, and heavy trauma).

4. Fixation Geometry:
   - "fillet_radius_mm": Edge rounding radius (default 1.2 mm, range 0.4 - 2.5 mm).
   - "screw_spacing_mm": Distance between consecutive screw hole centers (default 15.0 mm, range 10.0 - 18.0 mm).

You must output ONLY a valid JSON object with these exact keys:
{
  "objective": "Concise objective title (e.g. 'Cost-Effective Rigid Trauma Fixation' or 'Callus Stimulation & Mass Minimization')",
  "target_fracture_displacement": float in meters (e.g. 0.0002 for 0.2mm, 0.00012 for 0.12mm, 0.00030 for 0.30mm),
  "max_mass": float between 0.2 and 1.0 (e.g. 0.55 for 55% mass),
  "recommended_material": "Exact material name from the list above",
  "recommended_tpms": "Exact TPMS architecture name: 'Schwarz Primitive (P)' OR 'Schoen Gyroid (G)' OR 'Schwarz Diamond (D)'",
  "fillet_radius_mm": float between 0.4 and 2.5 (default 1.2),
  "screw_spacing_mm": float between 10.0 and 18.0 (default 15.0),
  "clinical_rationale": "Clear 2-sentence biomechanical rationale for the surgeon explaining the material and TPMS choice"
}
"""


def _parse_with_backend_llm(prompt: str, api_key: str) -> Optional[DesignRequest]:
    # Groq API pe call lagao, model fail hua toh automatic fallback list me se doosra try karo
    clean_key = api_key.strip().strip('"').strip("'")
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {clean_key}",
        "Content-Type": "application/json"
    }
    
    candidate_models = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.6-27b",
        "groq/compound-mini",
        "llama-3.3-70b-versatile"
    ]
    
    for model in candidate_models:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Analyze this clinical request: '{prompt}'"}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1
        }
        try:
            print(f"[Backend LLM Agent] ⚡ Querying Groq ({model}) for: '{prompt}'...")
            resp = requests.post(url, headers=headers, json=payload, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                parsed_json = json.loads(content)
                print(f"[Backend LLM Agent] ✅ Groq Success ({model}): {parsed_json}")
                
                # Material mapping clean karo (Only Titanium & Steel)
                mat = parsed_json.get("recommended_material", "Ti-6Al-4V (Grade 5 Titanium)")
                if "steel" in mat.lower() or "316" in mat.lower() or "cheap" in prompt.lower() or "cost" in prompt.lower() or "affordable" in prompt.lower():
                    mat = "316L Stainless Steel"
                else:
                    mat = "Ti-6Al-4V (Grade 5 Titanium)"
                    
                # TPMS lattice architecture clean karo
                tpms_raw = parsed_json.get("recommended_tpms", "Schwarz Primitive (P)")
                if "gyroid" in tpms_raw.lower():
                    tpms = "Schoen Gyroid (G)"
                elif "diamond" in tpms_raw.lower() or "spiral" in prompt.lower() or "torsion" in prompt.lower():
                    tpms = "Schwarz Diamond (D)"
                else:
                    tpms = "Schwarz Primitive (P)"
                    
                # Fillet radius aur screw spacing parse karo
                f_rad = float(parsed_json.get("fillet_radius_mm", 1.2))
                s_spac = float(parsed_json.get("screw_spacing_mm", 15.0))
                f_rad = min(max(f_rad, 0.4), 2.5)
                s_spac = min(max(s_spac, 10.0), 18.0)
                
                return DesignRequest(
                    objective=parsed_json.get("objective", "Callus Stimulation & Mass Minimization"),
                    target_fracture_displacement=float(parsed_json.get("target_fracture_displacement", 0.0002)),
                    max_mass=float(parsed_json.get("max_mass", 0.60)),
                    recommended_material=mat,
                    recommended_tpms=tpms,
                    fillet_radius_mm=f_rad,
                    screw_spacing_mm=s_spac,
                    clinical_rationale=f"⚡ [Groq LLM / {model}]: " + parsed_json.get("clinical_rationale", "")
                )
        except Exception as e:
            print(f"[Backend LLM Agent Exception on {model}]: {e}")
            
    return None


def _parse_with_local_nlp(user_prompt: str) -> DesignRequest:
    # Internet/API na ho toh local rule-based parsing engine chalao
    prompt_lower = user_prompt.lower()
    target_disp_m = 0.00020
    
    # Millimeter ya micron value dhundo regex se
    mm_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:mm|millimeters|millimetre)", prompt_lower)
    um_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:um|µm|microns|micrometers)", prompt_lower)
    
    if mm_match:
        target_disp_m = float(mm_match.group(1)) * 1e-3
    elif um_match:
        target_disp_m = float(um_match.group(1)) * 1e-6
    elif "rigid" in prompt_lower or "high-strength" in prompt_lower or "athlete" in prompt_lower:
        target_disp_m = 0.00012
    elif "osteoporotic" in prompt_lower or "elderly" in prompt_lower or "compliant" in prompt_lower:
        target_disp_m = 0.00030
    elif "callus" in prompt_lower or "secondary healing" in prompt_lower:
        target_disp_m = 0.00020
        
    recommended_tpms = "Schwarz Primitive (P)"
    if "gyroid" in prompt_lower or "shear" in prompt_lower or "oblique" in prompt_lower:
        recommended_tpms = "Schoen Gyroid (G)"
    elif "diamond" in prompt_lower or "torsion" in prompt_lower or "spiral" in prompt_lower or "athlete" in prompt_lower:
        recommended_tpms = "Schwarz Diamond (D)"
    
    if "rigid" in prompt_lower or "athlete" in prompt_lower or "high-strength" in prompt_lower or "trauma" in prompt_lower:
        objective = "Rigid Fixation & High-Strength Stability"
        clinical_rationale = (
            f"Enforces high structural rigidity ({target_disp_m*1000:.2f}mm limit) using a {recommended_tpms} lattice "
            "to withstand athletic load cycles and prevent fixation failure."
        )
        recommended_mat = "Ti-6Al-4V (Grade 5 Titanium)"
        max_mass = 0.85
    elif "osteoporotic" in prompt_lower or "elderly" in prompt_lower or "stress shielding" in prompt_lower:
        objective = "Osteoporotic Compliance & Porosity Maximization"
        clinical_rationale = (
            f"Targets elevated micro-motion ({target_disp_m*1000:.2f}mm) with a high-permeability {recommended_tpms} architecture "
            "to match low bone mineral density and eliminate cortical resorption."
        )
        recommended_mat = "Ti-6Al-4V (Grade 5 Titanium)"
        max_mass = 0.45
    elif "cheap" in prompt_lower or "cost" in prompt_lower or "affordable" in prompt_lower or "steel" in prompt_lower:
        objective = "Cost-Effective Secondary Bone Fixation"
        clinical_rationale = (
            f"Selects medical-grade stainless steel with a {recommended_tpms} lattice to optimize manufacturing costs "
            f"while maintaining physiological micro-motion ({target_disp_m*1000:.2f}mm)."
        )
        recommended_mat = "316L Stainless Steel"
        max_mass = 0.70
    else:
        objective = "Callus Stimulation & Mass Minimization"
        clinical_rationale = (
            f"Optimizes for the physiological secondary bone healing window ({target_disp_m*1000:.2f}mm micro-motion) "
            f"using a permeable {recommended_tpms} architecture while minimizing foreign implant mass."
        )
        recommended_mat = "Ti-6Al-4V (Grade 5 Titanium)"
        max_mass = 0.60
        
    mass_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:mass|weight)", prompt_lower)
    porosity_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:porosity|porous)", prompt_lower)
    
    if mass_match:
        max_mass = float(mass_match.group(1)) / 100.0
    elif porosity_match:
        max_mass = 1.0 - (float(porosity_match.group(1)) / 100.0)
        
    if "steel" in prompt_lower or "316" in prompt_lower or "cheap" in prompt_lower or "cost" in prompt_lower or "affordable" in prompt_lower:
        recommended_mat = "316L Stainless Steel"
    else:
        recommended_mat = "Ti-6Al-4V (Grade 5 Titanium)"

    # Fillet radius aur screw spacing parsing
    fillet_mm = 1.2
    fillet_match = re.search(r"(?:fillet|radius|rounding)\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*mm", prompt_lower)
    if not fillet_match:
        fillet_match = re.search(r"(\d+(?:\.\d+)?)\s*mm\s*(?:fillet|radius|rounding)", prompt_lower)
    if fillet_match:
        fillet_mm = min(max(float(fillet_match.group(1)), 0.4), 2.5)
        
    spacing_mm = 15.0
    spacing_match = re.search(r"(?:hole|screw|pitch)?\s*spacing\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*mm", prompt_lower)
    if not spacing_match:
        spacing_match = re.search(r"(\d+(?:\.\d+)?)\s*mm\s*(?:hole|screw|pitch)?\s*spacing", prompt_lower)
    if spacing_match:
        spacing_mm = min(max(float(spacing_match.group(1)), 10.0), 18.0)

    return DesignRequest(
        objective=objective,
        target_fracture_displacement=target_disp_m,
        max_mass=max_mass,
        recommended_material=recommended_mat,
        recommended_tpms=recommended_tpms,
        fillet_radius_mm=fillet_mm,
        screw_spacing_mm=spacing_mm,
        clinical_rationale="🔍 [Local Biomechanical NLP]: " + clinical_rationale
    )


def parse_design_request(user_prompt: str) -> DesignRequest:
    # Main function: Pehle Groq API try karega fir local fallback pe jaega
    from src.utils.logger import log_user_prompt_and_llm_response
    
    # Reload from .env if needed
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.env"))
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    if "GROQ_API_KEY" in line and "=" in line:
                        groq_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        os.environ["GROQ_API_KEY"] = groq_key
                        break
                        
    if groq_key and len(groq_key.strip()) > 5:
        llm_res = _parse_with_backend_llm(user_prompt, groq_key)
        if llm_res is not None:
            log_user_prompt_and_llm_response(user_prompt, llm_res, engine="Groq LPU (Llama 3.3 70B)")
            return llm_res
            
    local_res = _parse_with_local_nlp(user_prompt)
    log_user_prompt_and_llm_response(user_prompt, local_res, engine="Local Biomechanical NLP")
    return local_res


if __name__ == "__main__":
    test_p = "A 22 year old athlete with spiral femur fracture needing maximum torsion resistance"
    req = parse_design_request(test_p)
    print(f"\nResult for '{test_p}':")
    print(req)
