# Natural Language Biomechanical Agent: Parses clinical prompts into mathematical optimization parameters
# Uses high-speed Groq LPU inference with automatic local NLP regex fallback

import os
import sys
import re
import json
import requests
from typing import Optional
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Load environment variables
try:
    from dotenv import load_dotenv
    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.env"))
    load_dotenv(env_path)
except Exception:
    pass

# Manual fallback if dotenv is unavailable
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
    is_gibberish: bool = Field(default=False, description="True if prompt was identified as gibberish or off-topic")
    warning_message: Optional[str] = Field(default=None, description="Warning message if non-clinical input was detected")


def detect_prompt_anomaly(prompt: str) -> tuple[bool, str]:
    """
    Analyzes prompt for gibberish, non-clinical intent, keyboard smashes, or extreme brevity.
    Returns (is_anomalous: bool, reason_message: str).
    """
    if not prompt or not prompt.strip():
        return True, "Empty or blank prompt provided."

    clean = prompt.strip()
    clean_lower = clean.lower()

    # 1. Extreme brevity
    if len(clean) < 4:
        return True, f"Prompt '{clean}' is too short (< 4 characters) to extract clinical intent."

    words = [w for w in re.split(r"\s+", clean_lower) if w]
    
    # 2. Repeated character spam (e.g. "aaaaa", "zzzzzz", "asdasdasd")
    if re.search(r"(.)\1{4,}", clean_lower):
        return True, "Repetitive character spam detected."

    # 3. Non-alphanumeric noise ratio
    alpha_chars = sum(1 for c in clean if c.isalnum())
    if len(clean) >= 6 and (alpha_chars / len(clean)) < 0.40:
        return True, "Excessive special characters/punctuation noise detected."

    # 4. Long words without vowels (keyboard smash e.g. "asdfghjkl", "qwrtpsdfg")
    vowel_pattern = re.compile(r"[aeiouy]")
    for w in words:
        clean_w = re.sub(r"[^a-z]", "", w)
        if len(clean_w) >= 6 and not vowel_pattern.search(clean_w):
            return True, f"Unintelligible consonant sequence '{w}' detected."

    # 5. Common known off-topic / filler strings
    off_topic_patterns = [
        "lorem ipsum", "dolor sit amet", "asdf", "qwerty", "foo bar", "test test",
        "hello world", "drop table", "select * from", "make me a", "write a poem",
        "recipe for", "tell me a joke", "who are you", "what is the meaning of",
        "blah blah", "xyz123"
    ]
    for otp in off_topic_patterns:
        if otp in clean_lower and len(words) <= 6:
            return True, f"Non-clinical filler text '{otp}' detected."

    # 6. Domain vocabulary presence check for non-trivial inputs
    ORTHOPAEDIC_ONTOLOGY = {
        "bone", "bones", "femur", "femoral", "cortex", "cortical", "diaphyseal", "diaphysis",
        "callus", "osteoporotic", "osteoporosis", "osteopenia", "osteopenic",
        "trabecular", "cancellous", "periosteal", "periosteum", "fracture", "fractures", "gap",
        "knee", "hip", "tka", "tha", "stem", "periprosthetic", "segmental",
        "transverse", "oblique", "spiral", "comminuted", "refracture", "non-union",
        "delayed", "union", "osteotomy", "osteosarcoma", "sarcoma", "bisphosphonate",
        "patient", "elderly", "athlete", "athletic", "trauma", "dialysis", "radiation",
        "irradiated", "weight", "obese", "obesity", "gait", "leg", "limb", "healing",
        "biology", "biological", "tissue", "implant", "implants", "plate", "plates", "screw", "screws",
        "fixation", "osteosynthesis", "stiff", "stiffness", "rigid", "rigidity",
        "compliant", "compliance", "flexible", "flexure", "motion", "micro-motion", "micromotion",
        "displacement", "deflection", "strain", "stress", "shielding", "load", "loading",
        "torsion", "torsional", "shear", "bending", "axial", "pullout", "yield",
        "fatigue", "safety", "fos", "strength", "ductile", "ductility", "endurance",
        "titanium", "steel", "ti-6al-4v", "316l", "alloy", "metal", "tpms", "lattice",
        "porosity", "porous", "density", "mass", "cell", "primitive", "gyroid", "diamond",
        "fillet", "pitch", "spacing", "span", "bridge", "skin", "thickness", "schwarz", "schoen",
        "cost", "cheap", "affordable"
    }

    token_set = set(re.findall(r"[a-z0-9\-\_]+", clean_lower))
    has_domain_term = bool(token_set & ORTHOPAEDIC_ONTOLOGY)
    has_numerical_spec = bool(re.search(r"\b\d+(?:\.\d+)?\s*(?:mm|um|µm|%|gpa|mpa|n|kn|yr|yo|years?)\b", clean_lower))

    if len(words) >= 3 and not has_domain_term and not has_numerical_spec:
        return True, "Input lacks identifiable orthopaedic, anatomical, or biomechanical context."

    if len(words) < 2 and not has_domain_term and not has_numerical_spec:
        return True, f"Single-word non-clinical prompt '{clean}' lacks biomechanical context."

    return False, ""


def create_safe_baseline_request(prompt: str, reason: str) -> DesignRequest:
    """Returns a certified safe default design request when anomalous input is provided."""
    snippet = prompt.strip()[:40] + ("..." if len(prompt.strip()) > 40 else "")
    return DesignRequest(
        objective="Safe Clinical Baseline (Callus Stimulation)",
        target_fracture_displacement=0.00020,
        max_mass=0.60,
        recommended_material="Ti-6Al-4V (Grade 5 Titanium)",
        recommended_tpms="Schwarz Primitive (P)",
        fillet_radius_mm=1.2,
        screw_spacing_mm=14.5,
        clinical_rationale=(
            f"🛡️ [Gibberish / Non-Clinical Guardrail Activated]: {reason} "
            f"Safely defaulted to standard secondary callus fixation baseline (0.20mm micro-motion, Ti-6Al-4V Grade 5, Schwarz Primitive lattice) to ensure patient safety."
        ),
        is_gibberish=True,
        warning_message=f"Non-clinical or unintelligible input: {reason} Defaulted to certified AO Foundation clinical baseline."
    )


SYSTEM_PROMPT = """You are an expert Orthopaedic Biomechanics AI Agent specializing in patient-specific implant co-design.
Your task is to analyze surgeon requirements and translate natural language prompts into precise numerical parameters for a Differentiable FEM Metamaterial Optimizer.

Key Biomechanical Principles:
1. Interfragmentary Micro-Motion (Secondary Healing / Callus Stimulation):
   - Optimal Callus Formation Window: 0.15mm to 0.25mm (0.00015m to 0.00025m, default 0.00020m).
   - Rigid fixation (Athletes / Segmental fractures / Young patients): 0.08mm to 0.15mm (default 0.00012m).
   - Osteoporotic bone / Elderly / Compliant: 0.25mm to 0.35mm (default 0.00030m).

2. Certified Orthopaedic Biomaterials:
   - "Ti-6Al-4V (Grade 5 Titanium)" (Gold standard biocompatible alloy, E=110 GPa, density=4.43 g/cm³, yield=880 MPa)
   - "316L Stainless Steel" (Standard trauma fixation / cost-effective high ductility, E=193 GPa, density=8.00 g/cm³, yield=220 MPa)

3. Minimal Surface Metamaterial Architectures:
   - "Schwarz Primitive (P)" (Highest fluid permeability & nutrient transport, optimal for bone callus stimulation and standard secondary healing).
   - "Schoen Gyroid (G)" (Exceptional shear resistance & isotropic compliance, optimal for oblique fractures, dynamic athletic gait, and active patients).
   - "Schwarz Diamond (D)" (Maximum torsional stiffness & multi-axial bending stability, optimal for spiral fractures, athletes, and heavy trauma).

4. Fixation & Sandwich Geometry:
   - "max_mass": Float between 0.45 and 0.85 (default 0.60 for optimal balance of porosity and strength).
   - "fillet_radius_mm": Edge rounding radius (default 1.2 mm, range 0.4 - 2.5 mm).
   - "screw_spacing_mm": Distance between consecutive screw hole centers (default 14.5 mm, range 10.0 - 16.0 mm).

You must output ONLY a valid JSON object with these exact keys:
{
  "objective": "Concise objective title (e.g. 'Cost-Effective Rigid Trauma Fixation' or 'Callus Stimulation & Mass Minimization')",
  "target_fracture_displacement": float in meters (e.g. 0.0002 for 0.2mm, 0.00012 for 0.12mm, 0.00030 for 0.30mm),
  "max_mass": float between 0.45 and 0.85 (default 0.60),
  "recommended_material": "Exact material name from the list above",
  "recommended_tpms": "Exact TPMS architecture name: 'Schwarz Primitive (P)' OR 'Schoen Gyroid (G)' OR 'Schwarz Diamond (D)'",
  "fillet_radius_mm": float between 0.4 and 2.5 (default 1.2),
  "screw_spacing_mm": float between 10.0 and 16.0 (default 14.5),
  "clinical_rationale": "Clear 2-sentence biomechanical rationale for the surgeon explaining the material and TPMS choice"
}
"""


def _parse_with_backend_llm(prompt: str, api_key: str) -> Optional[DesignRequest]:
    clean_key = api_key.strip().strip('"').strip("'")
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {clean_key}",
        "Content-Type": "application/json"
    }
    
    candidate_models = [
        "qwen/qwen3.8-27b",
        "qwen/qwen3.6-27b",
        "openai/gpt-oss-120b"
    ]
    
    for model in candidate_models:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Analyze this clinical request and extract optimization parameters: '{prompt}'"}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=7)
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                parsed_json = json.loads(content)
                
                # Material mapping validation (Ti-6Al-4V Grade 5 Titanium or 316L Stainless Steel)
                mat = parsed_json.get("recommended_material", "Ti-6Al-4V (Grade 5 Titanium)")
                if "steel" in mat.lower() or "316" in mat.lower() or "cheap" in prompt.lower() or "cost" in prompt.lower() or "affordable" in prompt.lower():
                    mat = "316L Stainless Steel"
                else:
                    mat = "Ti-6Al-4V (Grade 5 Titanium)"
                    
                # TPMS lattice architecture validation
                tpms_raw = parsed_json.get("recommended_tpms", "Schwarz Primitive (P)")
                if "gyroid" in tpms_raw.lower() or "shear" in prompt.lower() or "oblique" in prompt.lower():
                    tpms = "Schoen Gyroid (G)"
                elif "diamond" in tpms_raw.lower() or "spiral" in prompt.lower() or "torsion" in prompt.lower() or "athlete" in prompt.lower():
                    tpms = "Schwarz Diamond (D)"
                else:
                    tpms = "Schwarz Primitive (P)"
                    
                # Parameter bounds validation
                f_rad = float(parsed_json.get("fillet_radius_mm", 1.2))
                s_spac = float(parsed_json.get("screw_spacing_mm", 14.5))
                f_rad = min(max(f_rad, 0.4), 2.5)
                s_spac = min(max(s_spac, 10.0), 16.0)
                
                raw_disp = float(parsed_json.get("target_fracture_displacement", 0.00020))
                # Handle micron vs meter unit confusion from LLM
                if raw_disp > 1.0:
                    target_disp = raw_disp * 1e-6
                elif raw_disp > 0.001:
                    target_disp = raw_disp * 1e-3
                else:
                    target_disp = raw_disp
                target_disp = min(max(target_disp, 0.00008), 0.00035)
                
                raw_mass = float(parsed_json.get("max_mass", 0.60))
                if raw_mass > 1.0:
                    raw_mass /= 100.0
                max_mass = min(max(raw_mass, 0.45), 0.85)
                
                return DesignRequest(
                    objective=parsed_json.get("objective", "Callus Stimulation & Mass Minimization"),
                    target_fracture_displacement=target_disp,
                    max_mass=max_mass,
                    recommended_material=mat,
                    recommended_tpms=tpms,
                    fillet_radius_mm=f_rad,
                    screw_spacing_mm=s_spac,
                    clinical_rationale=f"⚡ [Groq LLM / {model}]: " + parsed_json.get("clinical_rationale", "")
                )
        except Exception:
            pass
            
    return None


def _parse_with_gemini_llm(prompt: str, api_key: str) -> Optional[DesignRequest]:
    clean_key = api_key.strip().strip('"').strip("'")
    try:
        from google import genai
        from google.genai import types
        
        client = genai.Client(api_key=clean_key)
        config = types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
            system_instruction=SYSTEM_PROMPT
        )
        
        for model in ["gemini-3.6-flash", "gemini-3.5-flash"]:
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=f"Analyze this clinical request and extract optimization parameters: '{prompt}'",
                    config=config
                )
                if response.text:
                    parsed_json = json.loads(response.text)
                    
                    mat = parsed_json.get("recommended_material", "Ti-6Al-4V (Grade 5 Titanium)")
                    if "steel" in mat.lower() or "316" in mat.lower() or "cheap" in prompt.lower() or "cost" in prompt.lower() or "affordable" in prompt.lower():
                        mat = "316L Stainless Steel"
                    else:
                        mat = "Ti-6Al-4V (Grade 5 Titanium)"
                        
                    tpms_raw = parsed_json.get("recommended_tpms", "Schwarz Primitive (P)")
                    if "gyroid" in tpms_raw.lower() or "shear" in prompt.lower() or "oblique" in prompt.lower():
                        tpms = "Schoen Gyroid (G)"
                    elif "diamond" in tpms_raw.lower() or "spiral" in prompt.lower() or "torsion" in prompt.lower() or "athlete" in prompt.lower():
                        tpms = "Schwarz Diamond (D)"
                    else:
                        tpms = "Schwarz Primitive (P)"
                        
                    f_rad = float(parsed_json.get("fillet_radius_mm", 1.2))
                    s_spac = float(parsed_json.get("screw_spacing_mm", 14.5))
                    f_rad = min(max(f_rad, 0.4), 2.5)
                    s_spac = min(max(s_spac, 10.0), 16.0)
                    
                    raw_disp = float(parsed_json.get("target_fracture_displacement", 0.00020))
                    if raw_disp > 1.0:
                        target_disp = raw_disp * 1e-6
                    elif raw_disp > 0.001:
                        target_disp = raw_disp * 1e-3
                    else:
                        target_disp = raw_disp
                    target_disp = min(max(target_disp, 0.00008), 0.00035)
                    
                    raw_mass = float(parsed_json.get("max_mass", 0.60))
                    if raw_mass > 1.0:
                        raw_mass /= 100.0
                    max_mass = min(max(raw_mass, 0.45), 0.85)
                    
                    return DesignRequest(
                        objective=parsed_json.get("objective", "Callus Stimulation & Mass Minimization"),
                        target_fracture_displacement=target_disp,
                        max_mass=max_mass,
                        recommended_material=mat,
                        recommended_tpms=tpms,
                        fillet_radius_mm=f_rad,
                        screw_spacing_mm=s_spac,
                        clinical_rationale=f"✨ [Google Gemini / {model}]: " + parsed_json.get("clinical_rationale", "")
                    )
            except Exception:
                continue
    except Exception:
        pass
        
    return None


def _parse_with_local_nlp(user_prompt: str) -> DesignRequest:
    # Rule-based NLP parsing engine
    prompt_lower = user_prompt.lower()
    target_disp_m = 0.00020
    
    # 1. Micro-motion / displacement parsing with explicit keyword matching
    disp_explicit = re.search(r"(?:motion|displacement|movement|deflection|strain|flexure)\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*(?:mm|um|µm|microns?)", prompt_lower)
    if not disp_explicit:
        disp_explicit = re.search(r"(\d+(?:\.\d+)?)\s*(?:mm|um|µm|microns?)\s*(?:of\s*)?(?:micro-?motion|displacement|movement|deflection|motion)", prompt_lower)
    
    if disp_explicit:
        val = float(disp_explicit.group(1))
        match_str = disp_explicit.group(0).lower()
        if "um" in match_str or "µm" in match_str or "micron" in match_str or val > 10.0:
            target_disp_m = val * 1e-6
        else:
            target_disp_m = val * 1e-3
    else:
        mm_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:mm|millimeters?|millimetres?)", prompt_lower)
        um_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:um|µm|microns?|micrometers?)", prompt_lower)
        if mm_match and float(mm_match.group(1)) <= 1.0:
            target_disp_m = float(mm_match.group(1)) * 1e-3
        elif um_match:
            target_disp_m = float(um_match.group(1)) * 1e-6
        elif "rigid" in prompt_lower or "high-strength" in prompt_lower or "athlete" in prompt_lower:
            target_disp_m = 0.00012
        elif "osteoporotic" in prompt_lower or "elderly" in prompt_lower or "compliant" in prompt_lower:
            target_disp_m = 0.00030
        elif "callus" in prompt_lower or "secondary healing" in prompt_lower:
            target_disp_m = 0.00020
            
    target_disp_m = min(max(target_disp_m, 0.00008), 0.00035)
        
    # 2. TPMS lattice architecture parsing
    recommended_tpms = "Schwarz Primitive (P)"
    if "gyroid" in prompt_lower or "shear" in prompt_lower or "oblique" in prompt_lower:
        recommended_tpms = "Schoen Gyroid (G)"
    elif "diamond" in prompt_lower or "torsion" in prompt_lower or "spiral" in prompt_lower or "athlete" in prompt_lower or "rotat" in prompt_lower:
        recommended_tpms = "Schwarz Diamond (D)"
    
    # 3. Clinical objective & rationale
    if "rigid" in prompt_lower or "athlete" in prompt_lower or "high-strength" in prompt_lower or "trauma" in prompt_lower:
        objective = "Rigid Fixation & High-Strength Stability"
        clinical_rationale = (
            f"Enforces high structural rigidity ({target_disp_m*1000:.2f}mm limit) using a {recommended_tpms} lattice "
            "to withstand athletic load cycles and prevent fixation failure."
        )
        recommended_mat = "Ti-6Al-4V (Grade 5 Titanium)"
        max_mass = 0.70
    elif "osteoporotic" in prompt_lower or "elderly" in prompt_lower or "stress shielding" in prompt_lower:
        objective = "Osteoporotic Compliance & Porosity Maximization"
        clinical_rationale = (
            f"Targets elevated micro-motion ({target_disp_m*1000:.2f}mm) with a high-permeability {recommended_tpms} architecture "
            "to match low bone mineral density and eliminate cortical resorption."
        )
        recommended_mat = "Ti-6Al-4V (Grade 5 Titanium)"
        max_mass = 0.50
    elif "cheap" in prompt_lower or "cost" in prompt_lower or "affordable" in prompt_lower or "steel" in prompt_lower:
        objective = "Cost-Effective Secondary Bone Fixation"
        clinical_rationale = (
            f"Selects medical-grade stainless steel with a {recommended_tpms} lattice to optimize manufacturing costs "
            f"while maintaining physiological micro-motion ({target_disp_m*1000:.2f}mm)."
        )
        recommended_mat = "316L Stainless Steel"
        max_mass = 0.65
    else:
        objective = "Callus Stimulation & Mass Minimization"
        clinical_rationale = (
            f"Optimizes for the physiological secondary bone healing window ({target_disp_m*1000:.2f}mm micro-motion) "
            f"using a permeable {recommended_tpms} architecture while minimizing foreign implant mass."
        )
        recommended_mat = "Ti-6Al-4V (Grade 5 Titanium)"
        max_mass = 0.60
        
    # 4. Mass and porosity parsing
    mass_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:mass|weight)", prompt_lower)
    porosity_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:porosity|porous)", prompt_lower)
    if mass_match:
        max_mass = min(max(float(mass_match.group(1)) / 100.0, 0.45), 0.85)
    elif porosity_match:
        max_mass = min(max(1.0 - (float(porosity_match.group(1)) / 100.0), 0.45), 0.85)
        
    # 5. Biomaterial selection
    if "steel" in prompt_lower or "316" in prompt_lower or "cheap" in prompt_lower or "cost" in prompt_lower or "affordable" in prompt_lower:
        recommended_mat = "316L Stainless Steel"
    else:
        recommended_mat = "Ti-6Al-4V (Grade 5 Titanium)"

    # 6. Fillet radius parsing
    fillet_mm = 1.2
    fillet_match = re.search(r"(?:fillet|rounding|edge\s*radius)\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*mm", prompt_lower)
    if not fillet_match:
        fillet_match = re.search(r"(\d+(?:\.\d+)?)\s*mm\s*(?:fillet|rounding|edge\s*radius)", prompt_lower)
    if fillet_match:
        fillet_mm = min(max(float(fillet_match.group(1)), 0.4), 2.5)
        
    # 7. Screw pitch / spacing parsing
    spacing_mm = 14.5
    pitch_match = re.search(r"(?:pitch|spacing|hole\s*distance)\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*mm", prompt_lower)
    if not pitch_match:
        pitch_match = re.search(r"(\d+(?:\.\d+)?)\s*mm\s*(?:pitch|spacing|screw\s*pitch)", prompt_lower)
    if pitch_match:
        spacing_mm = min(max(float(pitch_match.group(1)), 10.0), 16.0)

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
    """
    Parses a clinical user prompt with robust multi-tiered fallback and gibberish protection:
    0. Gibberish / Non-Clinical Anomaly Guardrail
    1. Groq LPU (Ultra-fast structured output with Llama 3.3 70B)
    2. Google Gemini (Advanced multi-modal reasoning with Gemini 2.5 Flash)
    3. Local Biomechanical NLP (Deterministic regex parameter extraction)
    """
    from src.utils.logger import log_user_prompt_and_llm_response

    # 0. Pre-flight Gibberish & Non-Clinical Anomaly Guardrail
    is_anomalous, anomaly_reason = detect_prompt_anomaly(user_prompt)
    if is_anomalous:
        safe_req = create_safe_baseline_request(user_prompt, anomaly_reason)
        log_user_prompt_and_llm_response(user_prompt, safe_req, engine="Gibberish Guardrail (Safety Fallback)")
        return safe_req
    
    # Reload from .env if needed
    groq_key = os.getenv("GROQ_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    if not groq_key or not gemini_key:
        env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.env"))
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    if "GROQ_API_KEY" in line and "=" in line:
                        groq_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        os.environ["GROQ_API_KEY"] = groq_key
                    elif ("GEMINI_API_KEY" in line or "GOOGLE_API_KEY" in line) and "=" in line:
                        gemini_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        os.environ["GEMINI_API_KEY"] = gemini_key

    # 1. First attempt: Groq LPU
    if groq_key and len(groq_key.strip()) > 8 and not groq_key.startswith("YOUR_"):
        llm_res = _parse_with_backend_llm(user_prompt, groq_key)
        if llm_res is not None:
            log_user_prompt_and_llm_response(user_prompt, llm_res, engine="Groq LPU (Llama 3.3 70B)")
            return llm_res

    # 2. Second attempt: Google Gemini
    if gemini_key and len(gemini_key.strip()) > 8 and not gemini_key.startswith("YOUR_"):
        gem_res = _parse_with_gemini_llm(user_prompt, gemini_key)
        if gem_res is not None:
            log_user_prompt_and_llm_response(user_prompt, gem_res, engine="Google Gemini")
            return gem_res

    # 3. Third attempt: Local Biomechanical Regex NLP
    local_res = _parse_with_local_nlp(user_prompt)
    log_user_prompt_and_llm_response(user_prompt, local_res, engine="Local Biomechanical Regex NLP")
    return local_res


if __name__ == "__main__":
    test_p = "A 22 year old athlete with spiral femur fracture needing maximum torsion resistance"
    req = parse_design_request(test_p)
    print(f"\nResult for '{test_p}':")
    print(req)
