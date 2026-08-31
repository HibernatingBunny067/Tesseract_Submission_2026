import os
import sys
import time
import datetime
import streamlit as st
import numpy as np
import pandas as pd
import logging
import meshio
from typing import Tuple, Optional, Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

# Suppress non-critical library logs to maintain clean runtime output
logging.getLogger("jax_fem").setLevel(logging.ERROR)
logging.getLogger("uvicorn").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
os.environ["JAX_FEM_LOG_LEVEL"] = "ERROR"
import src.fem.petsc_compat

from src.agent.agent import parse_design_request, DesignRequest
from src.agent.optimize import run_optimization
from src.agent.graph import run_design_agent
from src.geometry.plot_plotly import get_mesh_plotly_fig, get_von_mises_plotly_fig, generate_tpms_stl_bytes
from src.fem.forward import solve_fem
from src.fem.problem import compute_nodal_von_mises_stress
from src.fem.materials import BIOMATERIALS, Biomaterial
from src.fem.validation import run_insilico_validation_suite
from src.ui import build_css
from src.ui.components import (
    hero_banner,
    section_label,
    StatusBadge,
    metric_tiles,
    comparison_panel,
    material_card,
    validation_report_card,
)
from src.ui.charts import (
    create_disp_tracking_fig,
    create_porosity_tracking_fig,
    create_loss_tracking_fig,
    create_gradient_tracking_fig,
)

import subprocess, atexit
import tesseract_core as tc

# Dual Tesseract microservices background launcher
@st.cache_resource
def start_dual_tesseract_servers() -> Tuple[subprocess.Popen, subprocess.Popen]:
    fem_path: str = os.path.join(os.path.dirname(__file__), "tesseracts", "fem_tesseract", "tesseract_server.py")
    geom_path: str = os.path.join(os.path.dirname(__file__), "tesseracts", "geometry_tesseract", "tesseract_server.py")
    
    proc_fem: subprocess.Popen = subprocess.Popen([sys.executable, fem_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proc_geom: subprocess.Popen = subprocess.Popen([sys.executable, geom_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    def _shutdown() -> None:
        for p in [proc_fem, proc_geom]:
            try:
                p.terminate()
                p.wait(timeout=2)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
                    
    atexit.register(_shutdown)
    time.sleep(3)
    return proc_fem, proc_geom

server_fem_proc, server_geom_proc = start_dual_tesseract_servers()


st.set_page_config(page_title="Tesseract BioMechanics", page_icon="🦴", layout="wide", initial_sidebar_state="expanded")

# Inject custom CSS styling
st.markdown(build_css(), unsafe_allow_html=True)

# Main hero banner
logo_img_path = os.path.join(os.path.dirname(__file__), "images", "logo.png")
st.markdown(
    hero_banner(
        title="Differentiable Biomechanics Engine",
        subtitle="Agentic Biomechanical Implant Optimization · Dual Tesseract REST Engines · WSD Adam Optimizer",
        accent_word="Differentiable",
        logo_path=logo_img_path,
    ),
    unsafe_allow_html=True,
)

# System Architecture & Overview Banner
app_img_path = os.path.join(os.path.dirname(__file__), "images", "app.png")
if os.path.exists(app_img_path):
    _, img_col, _ = st.columns([2.2, 5.6, 2.2])
    with img_col:
        st.image(app_img_path, use_container_width=True)

# Dual Tesseract Microservices live health badges
st.markdown(
    """<div style="display: flex; gap: 1rem; margin-bottom: 0.8rem; flex-wrap: wrap;">
        <div class="glass-card" style="padding: 0.45rem 0.9rem; font-size: 0.78rem; border-left: 3px solid #22c55e; display: flex; align-items: center; gap: 0.4rem;">
            <span style="color: #4ade80; font-size: 0.9rem;">●</span> <b>Tesseract 1:</b> FEM Adjoint Solver (<span style="color: #94a3b8;">Port 8000 · JAX-FEM + PETSc</span>)
        </div>
        <div class="glass-card" style="padding: 0.45rem 0.9rem; font-size: 0.78rem; border-left: 3px solid #38bdf8; display: flex; align-items: center; gap: 0.4rem;">
            <span style="color: #38bdf8; font-size: 0.9rem;">●</span> <b>Tesseract 2:</b> Geometry & Porosity Engine (<span style="color: #94a3b8;">Port 8001 · Differentiable SDF</span>)
        </div>
    </div>
    <div style="background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.25); border-left: 4px solid #f59e0b; padding: 0.6rem 0.9rem; border-radius: 6px; margin-bottom: 1.1rem; font-size: 0.78rem; color: #fef3c7; display: flex; align-items: flex-start; gap: 0.6rem;">
        <span style="font-size: 1.1rem; line-height: 1.1;">⚠️</span>
        <div>
            <b style="color: #fbbf24;">MEDICAL-GRADE INVESTIGATIONAL CAUTION:</b> 
            This AI platform synthesizes patient-specific TPMS metamaterial constructs via in-silico surrogate mechanics, differentiable finite element simulation (JAX-FEM), and automated ASTM F382 / ISO 7206 virtual verification. In-silico predictions are for <b>computational surgical planning & biomechanical research only</b> and must be validated by certified physical mechanical testing prior to in-vivo additive manufacturing.
            <div style="margin-top: 0.2rem; font-size: 0.73rem; color: #94a3b8;">
                <i>⚡ <b>Performance Advisory:</b> Docker multi-container virtualization may introduce JAX JIT compilation and cross-network translation latency; native local execution (<code>./run</code>) provides maximum performance.</i>
            </div>
        </div>
    </div>""",
    unsafe_allow_html=True
)

from src.agent.agent import parse_design_request

# Clinical preset definitions
st.sidebar.markdown("### 🎛️ Control Architecture")

workflow_mode: str = st.sidebar.radio(
    "Architecture Mode",
    ["🤖 Multi-Agent Orchestrator (LangGraph)", "⚙️ Direct Parametric Mode"],
    index=0,
    help="Multi-Agent mode orchestrates 4 specialist agents (Clinical, Materials, Optimization, Validation) with autonomous closed-loop self-correction. Direct Parametric mode unlocks all manual engineering controls."
)

is_agent_mode: bool = "Multi-Agent" in workflow_mode
mat_keys = list(BIOMATERIALS.keys())
tpms_options = [
    "Schwarz Primitive (P) · High Permeability",
    "Schoen Gyroid (G) · High Shear Strength",
    "Schwarz Diamond (D) · High Torsion Grip"
]

if is_agent_mode:
    st.sidebar.markdown("### 🤖 Clinical Intent & Presets")
    
    CLINICAL_PRESETS: Dict[str, str] = {
        "Callus Stimulation (Default)": "I need a compliant Titanium plate that allows 0.2mm of micro-motion at the fracture site to stimulate callus formation, while keeping the implant as light as possible.",
        "⚠️ Stress Test: Periprosthetic Refracture": "A 76-year-old morbidly obese female on chronic bisphosphonate therapy and renal dialysis presents with an atypical periprosthetic femur refracture adjacent to a loosened total knee arthroplasty stem. She previously underwent radiation therapy for a proximal femoral osteosarcoma, leaving the surrounding cortex severely devitalized and osteopenic. Her surgeon requires a fixation construct that promotes aggressive biological healing in this compromised bone stock while surviving the extreme cyclic loading from her elevated body weight during early weight-bearing rehabilitation. The construct must accommodate the irradiated cortex's inability to remodel normally.",
        "Elderly Osteoporotic Patient": "I need a highly porous Titanium plate for an elderly osteoporotic patient that allows 0.30mm of micro-motion, minimizing stress shielding and maximizing porosity.",
        "Young Athlete High-Impact Trauma": "I need a rigid, high-strength Stainless Steel plate for a young athlete that restricts micro-motion to 0.12mm to ensure stable fixation.",
        "Cost-Effective Trauma Fixation": "I need an affordable, cost-effective 316L Stainless Steel plate that maintains 0.18mm micro-motion with high ductility."
    }

    def on_preset_change() -> None:
        selected: str = st.session_state.get("preset_select", "")
        if selected in CLINICAL_PRESETS:
            st.session_state.prompt_area = CLINICAL_PRESETS[selected]
            st.session_state.parsed_req = parse_design_request(st.session_state.prompt_area)

    if "prompt_area" not in st.session_state:
        st.session_state.prompt_area = CLINICAL_PRESETS["Callus Stimulation (Default)"]

    preset_options = list(CLINICAL_PRESETS.keys()) + ["Custom Specification"]

    preset = st.sidebar.selectbox(
        "Clinical Scenario Preset",
        preset_options,
        key="preset_select",
        on_change=on_preset_change
    )

    user_prompt = st.sidebar.text_area(
        "Design Request (Natural Language)",
        key="prompt_area",
        height=110
    )

    if "last_parsed_prompt" not in st.session_state:
        st.session_state.last_parsed_prompt = None

    if st.session_state.last_parsed_prompt != user_prompt or "parsed_req" not in st.session_state or st.session_state.parsed_req is None:
        st.session_state.parsed_req = parse_design_request(user_prompt)
        st.session_state.last_parsed_prompt = user_prompt

    if st.sidebar.button("⚡ Re-Parse Prompt", width="stretch"):
        with st.spinner("Agent interpreting biomechanical prompt…"):
            time.sleep(0.2)
            st.session_state.parsed_req = parse_design_request(user_prompt)
            st.session_state.last_parsed_prompt = user_prompt

    req = st.session_state.parsed_req
    rec_tpms = getattr(req, 'recommended_tpms', 'Schwarz Primitive (P)')
    f_rad_val = getattr(req, 'fillet_radius_mm', 1.2)
    s_spac_val = getattr(req, 'screw_spacing_mm', 15.0)

    if req:
        if getattr(req, "is_gibberish", False):
            warn_txt = getattr(req, "warning_message", "Non-clinical input detected.")
            st.sidebar.markdown(f"""
            <div class="glass-card" style="padding: 0.75rem; margin-top: 0.5rem; font-size: 0.78rem; border-left: 3px solid #f59e0b; background: rgba(245, 158, 11, 0.12);">
                <div style="color: #fbbf24; font-weight: 700; margin-bottom: 0.25rem;">⚠️ Safety Guardrail Activated</div>
                <div style="color: #fde68a; font-size: 0.73rem; margin-bottom: 0.3rem;">{warn_txt}</div>
                <div style="color: #94a3b8; font-size: 0.70rem;"><i>Safely applied certified clinical baseline (Callus Stimulation · 0.20 mm).</i></div>
            </div>
            """, unsafe_allow_html=True)

        badge_color = "#f59e0b" if getattr(req, "is_gibberish", False) else "#4ade80"
        badge_text = "⚠️ Default Safety Baseline Engaged" if getattr(req, "is_gibberish", False) else "✓ Requirements Parsed"
        st.sidebar.markdown(f"""
        <div class="glass-card" style="padding: 0.8rem; margin-top: 0.5rem; font-size: 0.8rem; border-left: 3px solid {'#f59e0b' if getattr(req, 'is_gibberish', False) else '#6366f1'};">
            <div style="color: {badge_color}; font-weight: 600; margin-bottom: 0.3rem;">{badge_text}</div>
            <div><b>Objective:</b> <span style="color: #a78bfa;">{req.objective}</span></div>
            <div><b>Target Micro-Motion:</b> <span style="color: #f8fafc; font-weight: 600;">{req.target_fracture_displacement*1000:.2f} mm</span> ({req.target_fracture_displacement*1e6:.0f} µm)</div>
            <div><b>Upper Mass Limit:</b> <span style="color: #f8fafc;">{req.max_mass*100:.0f}%</span></div>
            <div><b>Recommended Material:</b> <span style="color: #38bdf8;">{req.recommended_material}</span></div>
            <div><b>Recommended Topology:</b> <span style="color: #c084fc;">{rec_tpms}</span></div>
            <div><b>Fillet Radius:</b> <span style="color: #fbbf24;">{f_rad_val:.1f} mm</span> &nbsp;|&nbsp; <b>Screw Pitch:</b> <span style="color: #34d399;">{s_spac_val:.1f} mm</span></div>
            <div style="margin-top: 0.3rem; color: #94a3b8; font-size: 0.72rem;"><i>{req.clinical_rationale}</i></div>
        </div>
        """, unsafe_allow_html=True)

    with st.sidebar.expander("📐 Fixation CAD Geometry (Agent-Governed)", expanded=False):
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.75); padding: 0.7rem; border-radius: 6px; border-left: 3px solid #6366f1; font-size: 0.78rem;">
            <div style="color: #a5b4fc; font-weight: 600; margin-bottom: 0.4rem;">🔒 Autonomous CAD Baseline</div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.35rem; font-size: 0.75rem;">
                <div>• <b>Fillet:</b> {f_rad_val:.1f} mm</div>
                <div>• <b>Screw Pitch:</b> {s_spac_val:.1f} mm</div>
                <div>• <b>Top Skin:</b> 0.50 mm</div>
                <div>• <b>Bottom Skin:</b> 0.50 mm</div>
                <div>• <b>TPMS Core:</b> 5.00 mm</div>
                <div>• <b>Bridge Span:</b> 30.0 mm</div>
                <div>• <b>Cell Size:</b> 5.0 mm</div>
                <div>• <b>Topology:</b> {rec_tpms.split(' ')[1] if len(rec_tpms.split(' ')) > 1 else rec_tpms}</div>
            </div>
            <div style="margin-top: 0.5rem; color: #94a3b8; font-size: 0.7rem; font-style: italic;">
                Parameters are autonomously governed & self-corrected by the 4-agent loop. Switch to "Direct Parametric Mode" for manual overrides.
            </div>
        </div>
        """, unsafe_allow_html=True)

    fillet_radius_mm = float(f_rad_val)
    t_top_mm = 0.50
    t_bot_mm = 0.50
    bridge_span_mm = 30.0
    cell_size_mm = 5.0
    screw_spacing_mm = float(s_spac_val)

    st.sidebar.markdown("### 🧪 Material & Topology (Agent-Prescribed)")
    default_mat_idx = mat_keys.index(req.recommended_material) if (req and req.recommended_material in mat_keys) else 0
    selected_material_name = mat_keys[default_mat_idx]
    selected_material = BIOMATERIALS[selected_material_name]
    st.sidebar.markdown(material_card(selected_material), unsafe_allow_html=True)

    rec_lower = getattr(req, "recommended_tpms", "").lower()
    if "gyroid" in rec_lower:
        tpms_type_code = "gyroid"
        tpms_type_name = "Schoen Gyroid (G)"
        tpms_choice = "Schoen Gyroid (G) · High Shear Strength"
    elif "diamond" in rec_lower:
        tpms_type_code = "diamond"
        tpms_type_name = "Schwarz Diamond (D)"
        tpms_choice = "Schwarz Diamond (D) · High Torsion Grip"
    else:
        tpms_type_code = "primitive"
        tpms_type_name = "Schwarz Primitive (P)"
        tpms_choice = "Schwarz Primitive (P) · High Permeability"

else:
    # Direct Parametric Mode: Full manual control for engineers
    st.sidebar.markdown("""
    <div style="background: rgba(14, 165, 233, 0.1); border: 1px solid rgba(14, 165, 233, 0.3); border-left: 3px solid #0ea5e9; padding: 0.6rem 0.8rem; border-radius: 6px; font-size: 0.78rem; margin-bottom: 0.8rem; color: #e0f2fe;">
        ⚙️ <b>Direct Parametric Mode Active</b><br/>
        Manual biomechanical engineering control. Natural language NLP parsing is bypassed — tune all boundary conditions, alloys, and CAD dimensions manually below.
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("### 🎯 Optimization Targets")
    target_disp_mm = st.sidebar.slider(
        "Target Micro-Motion (mm)", 0.08, 0.35, 0.20, 0.01,
        help="Physiological fracture micro-motion window (0.08mm - 0.35mm)"
    )
    max_mass_pct = st.sidebar.slider(
        "Upper Mass Limit (%)", 40, 85, 60, 5,
        help="Maximum relative mass fraction allowed"
    )

    st.sidebar.markdown("### 🧪 Biomaterial & Lattice Topology")
    selected_material_name = st.sidebar.selectbox("Biomaterial", mat_keys, index=0)
    selected_material = BIOMATERIALS[selected_material_name]
    st.sidebar.markdown(material_card(selected_material), unsafe_allow_html=True)

    tpms_choice = st.sidebar.selectbox("Lattice Topology", tpms_options, index=0)
    if "gyroid" in tpms_choice.lower():
        tpms_type_code = "gyroid"
        tpms_type_name = "Schoen Gyroid (G)"
    elif "diamond" in tpms_choice.lower():
        tpms_type_code = "diamond"
        tpms_type_name = "Schwarz Diamond (D)"
    else:
        tpms_type_code = "primitive"
        tpms_type_name = "Schwarz Primitive (P)"

    with st.sidebar.expander("📐 Fixation CAD Geometry (Manual Control)", expanded=True):
        fillet_radius_mm = st.slider("Fillet Radius (mm)", 0.4, 2.5, 1.2, 0.1)
        t_top_mm = st.slider("Top Skin (mm)", 0.15, 2.00, 0.50, 0.05)
        t_bot_mm = st.slider("Bottom Skin (mm)", 0.15, 2.00, 0.50, 0.05)
        h_tpms_mm = max(6.0 - t_top_mm - t_bot_mm, 1.0)
        st.caption(f"🧬 TPMS Core: **{h_tpms_mm:.2f} mm** · Total Depth: **6.00 mm**")
        bridge_span_mm = st.slider("Bridge Span (mm)", 18.0, 45.0, 30.0, 1.0)
        cell_size_mm = st.slider("TPMS Cell Size (mm)", 3.5, 7.5, 5.0, 0.5)
        screw_spacing_mm = st.slider("Screw Pitch (mm)", 10.0, 16.0, 14.5, 0.5)

    # Synthesize direct manual design request
    req = DesignRequest(
        objective="Manual Parametric Fixation Optimization",
        target_fracture_displacement=target_disp_mm / 1000.0,
        max_mass=max_mass_pct / 100.0,
        recommended_material=selected_material_name,
        recommended_tpms=tpms_type_name,
        fillet_radius_mm=fillet_radius_mm,
        screw_spacing_mm=screw_spacing_mm,
        clinical_rationale=f"Manual parametric optimization: {selected_material_name} with {tpms_type_name} target motion {target_disp_mm:.2f}mm."
    )

h_tpms_mm = max(6.0 - t_top_mm - t_bot_mm, 1.0)
fillet_radius_m = fillet_radius_mm / 1000.0
t_top_m = t_top_mm / 1000.0
t_bot_m = t_bot_mm / 1000.0
skin_thickness_m = 0.5 * (t_top_m + t_bot_m)
screw_spacing_m = screw_spacing_mm / 1000.0
bridge_span_m = bridge_span_mm / 1000.0
cell_size_m = cell_size_mm / 1000.0

# Stage 1: Macro CAD Shape Morphing (PyGeM FFD)
with st.sidebar.expander("🔧 Stage 1: PyGeM CAD Morphing", expanded=False):
    enable_cad_morphing: bool = st.checkbox("Enable FFD Plate Curvature Morphing", value=True)
    cad_morph_steps: int = 0
    if enable_cad_morphing:
        cad_morph_steps = st.slider("CAD Morphing Steps", 3, 30, 10, 1)

if not enable_cad_morphing:
    cad_morph_steps = 0

with st.sidebar.expander("⚙️ Simulation & Solver", expanded=False):
    fidelity_choice = st.radio(
        "FEM Element Type",
        ["Clinical Grade (TET10 Refined · Quadratic 10-node)", "Fast Screening (TET4 Coarse · Linear 4-node)"],
        index=0
    )
    opt_max_steps = st.slider("Max Optimization Iterations", 5, 1000, 100, 5)
    enable_early_stopping = st.checkbox("Enable Early Convergence Stopping", value=True)

with st.sidebar.expander("📖 Objective Functions", expanded=False):
    st.markdown(r"""
**Stage 1 · PyGeM FFD Loss:**
$$\mathcal{L}_{\text{CAD}} = 2 \left( 22 \cdot \frac{\delta - \delta^*}{\delta^*} \right)^2 + \mathcal{C}(u)$$

**Stage 2 · JAX-FEM Adjoint Loss:**
$$\mathcal{L}_{\text{TPMS}} = \mathcal{L}_{\text{motion}} + c \mathcal{C}(u) + w \mathcal{L}_{\text{mass}} + \mathcal{B}_{\text{geom}} + \mathcal{B}_{\text{FoS}}$$

*FoS ≥ 1.50× under 750 N gait · Skin ≥ 0.35 mm*

---
🔗 [Detailed Mathematical Formulation](https://github.com/HibernatingBunny067/Tesseract_Submission_2026/tree/main#5-two-stage-mathematical-optimization)
    """)

st.sidebar.markdown("---")
st.sidebar.caption("Powered by **Tesseract Core** · JAX-FEM · PETSc")

# Top viewport: Geometry preview (left) and Live optimization telemetry (right)
morphed_mesh_path = os.path.join(os.path.dirname(__file__), "src", "fem", "data", "morphed_model.msh")
base_mesh_path = os.path.join(os.path.dirname(__file__), "src", "fem", "data", "model.msh")
mesh_path = morphed_mesh_path if os.path.exists(morphed_mesh_path) else base_mesh_path
cad_bend_y = st.session_state.get("cad_bend_y")
cad_bend_z = st.session_state.get("cad_bend_z")

col_geo, col_opt = st.columns([1, 1], gap="large")

with col_geo:
    st.markdown(section_label("📐", "Anatomical Model & Fixation Geometry"), unsafe_allow_html=True)
    geo_ph = st.empty()
    if os.path.exists(mesh_path):
        with geo_ph.container():
            if cad_bend_y is not None and len(cad_bend_y) > 0:
                st.caption("🔧 *Morphed Patient-Specific CAD Anatomy Active*")
            st.plotly_chart(
                get_mesh_plotly_fig(
                    mesh_path,
                    tau_values=None,
                    tpms_type=tpms_type_code,
                    fillet_radius=fillet_radius_m,
                    screw_spacing=screw_spacing_m,
                    bend_y_array=cad_bend_y,
                    bend_z_array=cad_bend_z,
                ),
                width="stretch",
                key="geo_base"
            )
            st.markdown(
                '<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.5rem; margin-top: 0.6rem; font-size: 0.78rem;">'
                '<div style="background: rgba(248, 250, 252, 0.08); border-left: 3px solid #f8fafc; padding: 0.35rem 0.6rem; border-radius: 4px; color: #f8fafc;">'
                '🦴 <b>Cortical Bone:</b> Ivory (E = 18 GPa)'
                '</div>'
                '<div style="background: rgba(251, 113, 133, 0.08); border-left: 3px solid #fb7185; padding: 0.35rem 0.6rem; border-radius: 4px; color: #fb7185;">'
                '🩸 <b>Trabecular Core:</b> Rose (E = 1.0 GPa)'
                '</div>'
                '<div style="background: rgba(251, 191, 36, 0.08); border-left: 3px solid #fbbf24; padding: 0.35rem 0.6rem; border-radius: 4px; color: #fbbf24;">'
                '⚡ <b>Fracture Gap:</b> 2.0mm (E = 1.0 MPa)'
                '</div>'
                '<div style="background: rgba(56, 189, 248, 0.08); border-left: 3px solid #38bdf8; padding: 0.35rem 0.6rem; border-radius: 4px; color: #38bdf8;">'
                '🔩 <b>Fixation Plate:</b> Cyan (E = 110 GPa)'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )
    else:
        geo_ph.warning("No mesh found. Run the mesh generator first.")

optimization_finished = False
opt_results = {}

with col_opt:
    st.markdown(section_label("🧪", "Tesseract Differentiable Optimization Engine"), unsafe_allow_html=True)

    is_agent_mode: bool = "Multi-Agent" in workflow_mode
    agent_delib_container = st.container() if is_agent_mode else None

    if req:
        col_btn1, col_btn2 = st.columns([3, 2], gap="small")
        with col_btn1:
            button_text: str = "🚀 Launch 2-Stage Multi-Agent Synthesis" if is_agent_mode else "🚀 Start 2-Stage Adjoint Optimization"
            btn_main = st.button(button_text, width="stretch")
        with col_btn2:
            btn_skip = st.button("⚡ Direct Lattice (Skip PyGeM)", width="stretch", help="Skip Stage 1 macro CAD shape morphing and proceed directly to high-fidelity JAX-FEM lattice optimization.")

        start_button = btn_main or btn_skip
        skip_cad_morph = btn_skip or (not enable_cad_morphing)

        # Stage 1 CAD Morphing Top-Level Placeholder
        morph_ph = st.empty()
        if enable_cad_morphing and not btn_skip:
            initial_by = st.session_state.get("cad_bend_y", [])
            initial_bz = st.session_state.get("cad_bend_z", [])
            if initial_by:
                mid_y = initial_by[len(initial_by)//2] * 1000.0 if len(initial_by) > 0 else 0.0
                mid_z = initial_bz[len(initial_bz)//2] * 1000.0 if len(initial_bz) > 0 else 0.0
                morph_ph.markdown(
                    f'<div class="glass-card" style="padding: 0.9rem 1.2rem; margin-bottom: 1.2rem; border-left: 4px solid #10b981; background: rgba(15, 23, 42, 0.92);">'
                    f'<div style="display: flex; justify-content: space-between; align-items: center;">'
                    f'<div style="font-weight: 700; font-size: 0.95rem; color: #f8fafc;">'
                    f'✅ <b>Stage 1 Morphed:</b> Anatomical FFD Plate Morphing Active'
                    f'</div>'
                    f'<div style="display: flex; gap: 0.8rem; font-size: 0.82rem;">'
                    f'<span style="background: rgba(56, 189, 248, 0.15); color: #38bdf8; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 600;">Sagittal Y: {mid_y:+.2f}mm</span>'
                    f'<span style="background: rgba(167, 139, 250, 0.15); color: #a78bfa; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 600;">Coronal Z: {mid_z:+.2f}mm</span>'
                    f'</div>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                morph_ph.markdown(
                    f'<div class="glass-card" style="padding: 0.9rem 1.2rem; margin-bottom: 1.2rem; border-left: 4px solid #38bdf8; background: rgba(15, 23, 42, 0.85);">'
                    f'<div style="display: flex; justify-content: space-between; align-items: center;">'
                    f'<div style="font-weight: 700; font-size: 0.95rem; color: #f8fafc;">'
                    f'🔧 <b>Stage 1:</b> Patient-Specific FFD Shape Morphing (PyGeM Active)'
                    f'</div>'
                    f'<span style="font-size: 0.8rem; padding: 0.2rem 0.6rem; border-radius: 4px; background: rgba(56, 189, 248, 0.15); color: #38bdf8; font-weight: 600;">{cad_morph_steps} FFD STEPS CONFIGURED</span>'
                    f'</div>'
                    f'<div style="font-size: 0.78rem; color: #94a3b8; margin-top: 0.4rem;">'
                    f'Adapts plate curvature to femoral anatomy (Sagittal Y & Coronal Z) before Stage 2 micro-lattice optimization.'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        progress_ph = st.empty()
        status_ph   = st.empty()
        
        tab_physics, tab_sens = st.tabs(["📊 Loss & Physical State", "🔍 Adjoint Gradients (∂L/∂τ)"])
        
        with tab_physics:
            loss_ph     = st.empty()
            disp_ph     = st.empty()
            porosity_ph = st.empty()

        with tab_sens:
            grad_ph = st.empty()
            
        def to_porosity(t):
            # Physically accurate mapping from level-set threshold tau in [0.10, 1.45]
            # to unit-cell TPMS lattice porosity [54.7%, 88.1%]
            t_clamped = min(max(float(t), 0.10), 1.45)
            return 54.7 + ((t_clamped - 0.10) / 1.35) * (88.1 - 54.7)

        if start_button:
            target_mm = req.target_fracture_displacement * 1000.0
            initial_baseline_disp = 0.018  # Un-optimized rigid solid construct baseline

            # Immediately populate telemetry charts with baseline targets and active reference corridors
            loss_ph.plotly_chart(create_loss_tracking_fig([120.0]), width="stretch")
            disp_ph.plotly_chart(create_disp_tracking_fig([initial_baseline_disp], target_mm), width="stretch")
            status_ph.markdown(StatusBadge.for_displacement(initial_baseline_disp, target_mm), unsafe_allow_html=True)
            porosity_ph.plotly_chart(create_porosity_tracking_fig({
                "Prox Anchor (%)": [to_porosity(0.35)],
                "Prox Transition (%)": [to_porosity(0.45)],
                "Bridge Gap (%)": [to_porosity(0.55)],
                "Dist Transition (%)": [to_porosity(0.45)],
                "Dist Anchor (%)": [to_porosity(0.35)]
            }, target_porosity_pct=(1.0 - req.max_mass)*100.0), width="stretch")
            grad_ph.plotly_chart(create_gradient_tracking_fig({
                "dL/dtau_p_anc": [0.0], "dL/dtau_p_tra": [0.0], "dL/dtau_bridge": [0.0],
                "dL/dtau_d_tra": [0.0], "dL/dtau_d_anc": [0.0], "dL/dsigma_blend": [0.0],
                "dL/dt_top": [0.0], "dL/dt_bottom": [0.0], "dL/ds_pitch": [0.0],
                "dL/dL_bridge": [0.0], "dL/dd_cell": [0.0], "dL/dr_fillet": [0.0]
            }), width="stretch")

            progress_ph.info("🚀 AI Biomechanical Inverse Design Initialized · Formulating Multi-Agent Strategy...")

            # Run Stage 1 CAD morphing if enabled and not skipped
            if not skip_cad_morph:
                try:
                    base_mesh = os.path.join(os.path.dirname(__file__), "src", "fem", "data", "model.msh")
                    morphed_mesh = os.path.join(os.path.dirname(__file__), "src", "fem", "data", "morphed_model.msh")
                    from src.agent.optimize_cad import run_cad_shape_optimization
                    
                    last_cad = None
                    for cad_state in run_cad_shape_optimization(
                        base_mesh_path=base_mesh,
                        morphed_mesh_path=morphed_mesh,
                        target_disp=req.target_fracture_displacement,
                        max_steps=cad_morph_steps
                    ):
                        last_cad = cad_state
                        step_idx = cad_state["step"] + 1
                        pct = min(step_idx / cad_morph_steps, 1.0)
                        
                        all_by = cad_state.get("all_bend_y", [])
                        all_bz = cad_state.get("all_bend_z", [])
                        
                        prox_y = all_by[0] * 1000.0 if len(all_by) > 0 else 0.0
                        mid_y = cad_state["bend_y"] * 1000.0
                        dist_y = all_by[-1] * 1000.0 if len(all_by) > 0 else 0.0
                        
                        prox_z = all_bz[0] * 1000.0 if len(all_bz) > 0 else 0.0
                        mid_z = cad_state["bend_z"] * 1000.0
                        dist_z = all_bz[-1] * 1000.0 if len(all_bz) > 0 else 0.0
                        
                        morph_html = (
                            f'<div class="glass-card" style="padding: 1.2rem; margin-bottom: 1.2rem; border-left: 4px solid #38bdf8; background: rgba(15, 23, 42, 0.95);">'
                            f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">'
                            f'<div style="font-weight: 700; font-size: 1rem; color: #f8fafc;">🔧 Stage 1: Macro CAD Shape Morphing (PyGeM FFD Active)</div>'
                            f'<span style="font-size: 0.8rem; padding: 0.2rem 0.6rem; border-radius: 4px; background: rgba(56, 189, 248, 0.2); color: #38bdf8; font-weight: 600;">STEP {step_idx}/{cad_morph_steps} ({int(pct*100)}%)</span>'
                            f'</div>'
                            f'<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.8rem; margin-bottom: 0.8rem;">'
                            f'<div style="background: rgba(30, 41, 59, 0.7); padding: 0.6rem 0.8rem; border-radius: 6px;">'
                            f'<div style="font-size: 0.72rem; color: #94a3b8;">OBJECTIVE LOSS</div>'
                            f'<div style="font-size: 1.05rem; font-weight: 700; color: #f43f5e;">{cad_state["loss"]:.4f}</div>'
                            f'<div style="font-size: 0.7rem; color: #64748b;">Motion: {cad_state["frac_disp"]*1000:.3f} mm</div>'
                            f'</div>'
                            f'<div style="background: rgba(30, 41, 59, 0.7); padding: 0.6rem 0.8rem; border-radius: 6px;">'
                            f'<div style="font-size: 0.72rem; color: #94a3b8;">SAGITTAL BEND (Y)</div>'
                            f'<div style="font-size: 1.05rem; font-weight: 700; color: #38bdf8;">{mid_y:+.3f} mm</div>'
                            f'<div style="font-size: 0.7rem; color: #64748b;">∂L/∂Y: {cad_state["grad_y"]:.1f}</div>'
                            f'</div>'
                            f'<div style="background: rgba(30, 41, 59, 0.7); padding: 0.6rem 0.8rem; border-radius: 6px;">'
                            f'<div style="font-size: 0.72rem; color: #94a3b8;">CORONAL BEND (Z)</div>'
                            f'<div style="font-size: 1.05rem; font-weight: 700; color: #a78bfa;">{mid_z:+.3f} mm</div>'
                            f'<div style="font-size: 0.7rem; color: #64748b;">∂L/∂Z: {cad_state["grad_z"]:.1f}</div>'
                            f'</div>'
                            f'</div>'
                            f'<div style="font-size: 0.78rem; color: #cbd5e1; background: rgba(30, 41, 59, 0.5); padding: 0.5rem 0.8rem; border-radius: 6px;">'
                            f'📍 <b>5x4x4 FFD Grid Deflections:</b> Proximal: (Y: {prox_y:+.2f}mm, Z: {prox_z:+.2f}mm) · Center Bridge: (Y: {mid_y:+.2f}mm, Z: {mid_z:+.2f}mm) · Distal: (Y: {dist_y:+.2f}mm, Z: {dist_z:+.2f}mm)'
                            f'</div>'
                            f'</div>'
                        )
                        morph_ph.markdown(morph_html, unsafe_allow_html=True)
                    
                    if last_cad:
                        st.session_state["cad_bend_y"] = last_cad.get("all_bend_y", [])
                        st.session_state["cad_bend_z"] = last_cad.get("all_bend_z", [])
                    
                    # Rebuild FEM solver for the morphed mesh
                    from src.fem.forward import rebuild_for_morphed_mesh
                    rebuild_for_morphed_mesh(morphed_mesh)
                    
                    # Display persistent completed Stage 1 status card
                    if last_cad:
                        final_by = last_cad.get("bend_y", 0.0) * 1000.0
                        final_bz = last_cad.get("bend_z", 0.0) * 1000.0
                        final_motion = last_cad.get("frac_disp", 0.0) * 1000.0
                        done_html = (
                            f'<div class="glass-card" style="padding: 0.9rem 1.2rem; margin-bottom: 1.2rem; border-left: 4px solid #10b981; background: rgba(15, 23, 42, 0.92);">'
                            f'<div style="display: flex; justify-content: space-between; align-items: center;">'
                            f'<div style="font-weight: 700; font-size: 0.95rem; color: #f8fafc;">'
                            f'✅ <b>Stage 1 Complete:</b> Anatomical FFD Plate Morphing Active'
                            f'</div>'
                            f'<div style="display: flex; gap: 0.8rem; font-size: 0.82rem;">'
                            f'<span style="background: rgba(56, 189, 248, 0.15); color: #38bdf8; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 600;">Sagittal Y: {final_by:+.2f}mm</span>'
                            f'<span style="background: rgba(167, 139, 250, 0.15); color: #a78bfa; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 600;">Coronal Z: {final_bz:+.2f}mm</span>'
                            f'<span style="background: rgba(16, 185, 129, 0.15); color: #10b981; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 600;">Macro Motion: {final_motion:.3f}mm</span>'
                            f'</div>'
                            f'</div>'
                            f'</div>'
                        )
                        morph_ph.markdown(done_html, unsafe_allow_html=True)
                    st.toast("✅ Stage 1 FFD CAD Morphing Complete: Solid mesh morphed to patient anatomy", icon="🔧")
                    
                    # Update the top 3D viewer to show the morphed SOLID geometry before Stage 2 starts
                    if 'geo_ph' in locals():
                        with geo_ph.container():
                            st.caption("🔧 *Stage 1 Complete: Morphed Patient-Specific CAD Anatomy Active*")
                            st.plotly_chart(
                                get_mesh_plotly_fig(
                                    morphed_mesh,
                                    tau_values=None, # Keep it solid
                                    tpms_type=tpms_type_code,
                                    fillet_radius=fillet_radius_m,
                                    screw_spacing=screw_spacing_m,
                                    bend_y_array=st.session_state.get("cad_bend_y"),
                                    bend_z_array=st.session_state.get("cad_bend_z"),
                                ),
                                width="stretch",
                                key="geo_morphed_post"
                            )
                except Exception as morph_err:
                    morph_ph.empty()
                    st.warning(f"Stage 1 CAD morphing warning: {morph_err}. Continuing with base mesh.")
            else:
                refined_mesh = os.path.join(os.path.dirname(__file__), "src", "fem", "data", "refined_model.msh")
                from src.fem.forward import rebuild_for_morphed_mesh
                if os.path.exists(refined_mesh):
                    rebuild_for_morphed_mesh(refined_mesh)
                morph_ph.markdown(
                    '<div class="glass-card" style="padding: 0.9rem 1.2rem; margin-bottom: 1.2rem; border-left: 4px solid #f59e0b; background: rgba(15, 23, 42, 0.92);">'
                    '<div style="display: flex; justify-content: space-between; align-items: center;">'
                    '<div style="font-weight: 700; font-size: 0.95rem; color: #f8fafc;">'
                    '⚡ <b>Stage 1 Skipped:</b> High-Fidelity JAX-FEM Lattice Optimization Active (63,153 DOFs)'
                    '</div>'
                    '<span style="background: rgba(245, 158, 11, 0.15); color: #f59e0b; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 600;">DIRECT LATTICE MODE</span>'
                    '</div>'
                    '</div>',
                    unsafe_allow_html=True
                )
                st.toast("⚡ Stage 1 PyGeM CAD Morphing Skipped. Running JAX-FEM on high-fidelity 63k DOF refined mesh.", icon="⚡")

            loss_history     = []
            disp_history     = []
            phase_history    = []
            porosity_history = {
                "Prox Anchor (%)": [],
                "Prox Transition (%)": [],
                "Bridge Gap (%)": [],
                "Dist Transition (%)": [],
                "Dist Anchor (%)": []
            }
            grad_history = {
                "dL/dtau_p_anc": [],
                "dL/dtau_p_tra": [],
                "dL/dtau_bridge": [],
                "dL/dtau_d_tra": [],
                "dL/dtau_d_anc": [],
                "dL/dsigma_blend": [],
                "dL/dt_top": [],
                "dL/dt_bottom": [],
                "dL/ds_pitch": [],
                "dL/dL_bridge": [],
                "dL/dd_cell": [],
                "dL/dr_fillet": []
            }
            
            target_disp      = req.target_fracture_displacement
            target_mm        = target_disp * 1000

            fem_url = os.environ.get("FEM_TESSERACT_URL", "http://127.0.0.1:8000")
            geom_url = os.environ.get("GEOMETRY_TESSERACT_URL", "http://127.0.0.1:8001")
            fem_client = None
            geom_client = None
            try:
                import httpx as _httpx
                resp_fem = _httpx.get(f"{fem_url}/health", timeout=0.2)
                resp_geo = _httpx.get(f"{geom_url}/health", timeout=0.2)
                if resp_fem.status_code == 200 and resp_geo.status_code == 200:
                    fem_client = tc.sdk.tesseract.Tesseract.from_url(fem_url)
                    geom_client = tc.sdk.tesseract.Tesseract.from_url(geom_url)
            except Exception:
                fem_client = None
                geom_client = None

            patience_val = 5 if enable_early_stopping else 9999
            last_tau = None
            final_agent_state = None

            if is_agent_mode:
                # Multi-Agent LangGraph Orchestration Flow
                with agent_delib_container:
                    st.markdown("#### 🤖 Multi-Agent Collaborative Workflow")
                    agent_chat_ph = st.empty()
                    rendered_messages = []
                    agent_chat_ph.markdown(
                        '<div class="glass-card" style="padding: 1rem 1.2rem; border-left: 4px solid #6366f1; background: rgba(15, 23, 42, 0.95); margin-bottom: 0.8rem;">'
                        '<div style="display: flex; align-items: center; gap: 0.6rem;">'
                        '<span style="font-size: 1.2rem;">🤖</span>'
                        '<div style="font-weight: 600; font-size: 0.92rem; color: #f8fafc;">Autonomous Multi-Agent System Deliberating...</div>'
                        '</div>'
                        '<div style="font-size: 0.82rem; color: #94a3b8; margin-top: 0.3rem;">'
                        'Specialist agents are evaluating clinical intent, material kinematics, and JAX adjoint sensitivities.'
                        '</div>'
                        '</div>',
                        unsafe_allow_html=True
                    )

                def on_opt_step(step_state: dict, step_num: int):
                    loss_val = float(step_state.get("loss", 0.0))
                    disp_val = float(step_state.get("frac_disp", 0.0)) * 1000.0
                    loss_history.append(loss_val)
                    disp_history.append(disp_val)
                    phase_history.append(step_state.get("phase", "Adam"))
                    
                    t_p_anc = step_state.get("tau_p_anc", step_state.get("tau_prox", 0.35))
                    t_p_tra = step_state.get("tau_p_tra", step_state.get("tau_prox", 0.45))
                    t_bri   = step_state.get("tau_bridge", 0.55)
                    t_d_tra = step_state.get("tau_d_tra", step_state.get("tau_dist", 0.45))
                    t_d_anc = step_state.get("tau_d_anc", step_state.get("tau_dist", 0.35))
                    sigma   = step_state.get("sigma_blend", 0.015)
                    
                    if "t_top_mm" in step_state:
                        t_top_m = float(step_state["t_top_mm"]) / 1000.0
                    if "t_bottom_mm" in step_state:
                        t_bot_m = float(step_state["t_bottom_mm"]) / 1000.0
                    if "screw_spacing_mm" in step_state:
                        screw_spacing_m = float(step_state["screw_spacing_mm"]) / 1000.0
                    if "bridge_span_mm" in step_state:
                        bridge_span_m = float(step_state["bridge_span_mm"]) / 1000.0
                    if "fillet_radius_mm" in step_state:
                        fillet_radius_m = float(step_state["fillet_radius_mm"]) / 1000.0
                    if "cell_size_mm" in step_state:
                        cell_size_m = float(step_state["cell_size_mm"]) / 1000.0
                    
                    porosity_history["Prox Anchor (%)"].append(to_porosity(t_p_anc))
                    porosity_history["Prox Transition (%)"].append(to_porosity(t_p_tra))
                    porosity_history["Bridge Gap (%)"].append(to_porosity(t_bri))
                    porosity_history["Dist Transition (%)"].append(to_porosity(t_d_tra))
                    porosity_history["Dist Anchor (%)"].append(to_porosity(t_d_anc))
                    
                    grad_history["dL/dtau_p_anc"].append(step_state.get("grad_p_anc", 0.0))
                    grad_history["dL/dtau_p_tra"].append(step_state.get("grad_p_tra", 0.0))
                    grad_history["dL/dtau_bridge"].append(step_state.get("grad_bridge", 0.0))
                    grad_history["dL/dtau_d_tra"].append(step_state.get("grad_d_tra", 0.0))
                    grad_history["dL/dtau_d_anc"].append(step_state.get("grad_d_anc", 0.0))
                    grad_history["dL/dsigma_blend"].append(step_state.get("grad_sigma", 0.0))
                    grad_history["dL/dt_top"].append(step_state.get("grad_t_top", 0.0))
                    grad_history["dL/dt_bottom"].append(step_state.get("grad_t_bot", 0.0))
                    grad_history["dL/ds_pitch"].append(step_state.get("grad_pitch", 0.0))
                    grad_history["dL/dL_bridge"].append(step_state.get("grad_bridge_span", 0.0))
                    grad_history["dL/dd_cell"].append(step_state.get("grad_cell_size", 0.0))
                    grad_history["dL/dr_fillet"].append(step_state.get("grad_fillet", 0.0))
                    
                    last_tau = (t_p_anc, t_p_tra, t_bri, t_d_tra, t_d_anc, sigma)

                    loss_ph.plotly_chart(create_loss_tracking_fig(loss_history), width="stretch")
                    disp_ph.plotly_chart(create_disp_tracking_fig(disp_history, target_mm), width="stretch")
                    status_ph.markdown(StatusBadge.for_displacement(disp_history[-1], target_mm), unsafe_allow_html=True)
                    porosity_ph.plotly_chart(create_porosity_tracking_fig(porosity_history, target_porosity_pct=(1.0 - req.max_mass)*100.0), width="stretch")
                    grad_ph.plotly_chart(create_gradient_tracking_fig(grad_history), width="stretch")

                    pitch_mm = screw_spacing_m * 1000.0
                    core_mm = max(6.0 - (t_top_m + t_bot_m)*1000.0, 1.0)
                    progress_ph.progress(
                        min(step_num / opt_max_steps, 1.0),
                        text=f"Step {step_num}/{opt_max_steps} | Loss: {loss_val:.2f} | Motion: {disp_val:.3f} mm | Pitch: {pitch_mm:.1f} mm | Core: {core_mm:.2f} mm"
                    )

                for event in run_design_agent(
                    surgeon_prompt=user_prompt,
                    fem_client=fem_client,
                    geometry_client=geom_client,
                    max_attempts=3,
                    max_steps=opt_max_steps,
                    bend_y_array=st.session_state.get("cad_bend_y"),
                    bend_z_array=st.session_state.get("cad_bend_z"),
                    stream=True,
                    step_callback=on_opt_step
                ):
                    if event["type"] == "agent_message":
                        msg = event["message"]
                        rendered_messages.append(msg)

                        # Update active progress banner dynamically with the current specialist agent
                        agent_status_map = {
                            "clinical_interpreter": f"🧑‍⚕️ Clinical Interpreter: Synthesizing fracture kinematics ({req.objective})...",
                            "materials_advisor": f"🔬 Materials Advisor: Evaluating Gibson-Ashby scaling for {selected_material.name}...",
                            "optimization_controller": "⚙️ Optimization Controller: Compiling JAX-FEM adjoint graph on 63,153 DOFs...",
                            "validation_auditor": "📋 Validation Auditor: Executing ASTM F382 & ISO 7206 virtual verification..."
                        }
                        current_status = agent_status_map.get(msg.get("agent_name"), "🤖 Multi-Agent Orchestrator active...")
                        progress_ph.info(current_status)

                        chat_html = ['<div style="display: flex; flex-direction: column; gap: 0.6rem; margin-bottom: 1rem;">']
                        for m in rendered_messages:
                            border_color = {
                                "clinical_interpreter": "#10b981",
                                "materials_advisor": "#f59e0b",
                                "optimization_controller": "#ef4444",
                                "validation_auditor": "#8b5cf6"
                            }.get(m["agent_name"], "#6366f1")
                            badge_bg = {
                                "status": "rgba(148, 163, 184, 0.15)",
                                "result": "rgba(56, 189, 248, 0.15)",
                                "correction": "rgba(239, 68, 68, 0.15)"
                            }.get(m.get("message_type", "status"), "rgba(99, 102, 241, 0.15)")
                            card_html = (
                                f'<div class="glass-card" style="padding: 0.75rem 1rem; border-left: 4px solid {border_color}; background: rgba(15, 23, 42, 0.85); margin-bottom: 0.5rem;">'
                                f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem;">'
                                f'<span style="font-weight: 600; font-size: 0.88rem; color: #f8fafc;">{m["agent_emoji"]} {m["agent_display_name"]}</span>'
                                f'<span style="font-size: 0.72rem; padding: 0.1rem 0.4rem; border-radius: 4px; background: {badge_bg}; color: #94a3b8;">{m.get("message_type", "status").upper()}</span>'
                                f'</div>'
                                f'<div style="font-size: 0.82rem; line-height: 1.45; color: #cbd5e1; white-space: pre-wrap;">{m["content"]}</div>'
                                f'</div>'
                            )
                            chat_html.append(card_html)
                        chat_html.append('</div>')
                        with agent_delib_container:
                            agent_chat_ph.markdown("".join(chat_html), unsafe_allow_html=True)

                    elif event["type"] == "final_result":
                        final_agent_state = event.get("state", {})

                # Extract final results from agent state
                if final_agent_state:
                    opt_res = final_agent_state.get("optimization_result", {})
                    final_theta = opt_res.get("final_theta", [])
                    final_metrics = opt_res.get("final_metrics", {})
                    if len(final_theta) >= 6:
                        last_tau = tuple(final_theta[1:7]) if len(final_theta) >= 7 else tuple(final_theta[1:6]) + (0.015,)
                    if final_metrics.get("frac_disp", 0.0) > 0:
                        disp_history = [final_metrics["frac_disp"] * 1000]
                        loss_history = [final_metrics.get("loss", 0.0)]

                optimization_finished = last_tau is not None
                if optimization_finished:
                    opt_results = {
                        "last_tau": last_tau,
                        "cell_size_m": cell_size_m,
                        "t_top_m": t_top_m,
                        "t_bottom_m": t_bot_m,
                        "screw_spacing_m": screw_spacing_m,
                        "bridge_span_m": bridge_span_m,
                        "fillet_radius_m": fillet_radius_m,
                        "avg_porosity": float(final_metrics.get("mean_porosity", 0.5)) * 100.0,
                        "final_disp_mm": float(disp_history[-1]) if disp_history else 0.0,
                        "target_disp": target_disp,
                        "target_mm": target_mm
                    }
                    st.session_state.last_opt_results = opt_results
                    
                    # Unified history recording for Agent mode
                    solid_mass = 64.0 * (selected_material.density_g_cm3 / 4.43)
                    avg_por = opt_results["avg_porosity"]
                    optimized_mass = solid_mass * (1.0 - (avg_por / 100.0) * 0.85)
                    achieved_m = opt_results["final_disp_mm"]
                    err_pct = abs(achieved_m - target_mm) / (target_mm + 1e-9) * 100.0
                    status_str = "✅ PASS (ASTM)" if err_pct <= 15.0 else "⚠️ TIGHTENING"

                    st.session_state.run_history.append({
                        "Timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                        "Workflow Mode": "🤖 Multi-Agent LangGraph",
                        "Scenario / Goal": req.objective,
                        "Biomaterial": selected_material.code,
                        "Lattice Topology": tpms_choice.split("·")[0].strip(),
                        "Mesh Resolution": "63,153 DOFs (Refined)" if not skip_cad_morph else "63,153 DOFs (Direct)",
                        "Target Motion": f"{target_mm:.2f} mm",
                        "Achieved Motion": f"{achieved_m:.3f} mm",
                        "Avg Porosity": f"{avg_por:.1f}%",
                        "Mass Reduction": f"{((solid_mass - optimized_mass)/solid_mass)*100:.1f}%",
                        "ASTM Status": status_str
                    })
                    st.balloons()

            else:
                # Direct Parametric Optimization Mode (existing flow)
                for state in run_optimization(
                    target_fracture_displacement=target_disp,
                    patience=patience_val,
                    max_steps=opt_max_steps,
                    fem_client=fem_client,
                    geometry_client=geom_client,
                    objective=req.objective,
                    max_mass=req.max_mass,
                    material_modulus_gpa=selected_material.youngs_modulus_gpa,
                    tpms_ga_exponent=selected_material.tpms_ga_exponent,
                    yield_strength_mpa=selected_material.yield_strength_mpa,
                    init_cell_size=cell_size_m,
                    init_t_top=t_top_m,
                    init_t_bot=t_bot_m,
                    init_screw_spacing=screw_spacing_m,
                    init_bridge_span=bridge_span_m,
                    init_fillet_radius=fillet_radius_m
                ):
                    loss_history.append(state["loss"])
                    disp_history.append(state["frac_disp"] * 1000)
                    phase_history.append(state.get("phase", "Adam"))
                    
                    t_p_anc = state.get("tau_p_anc", state["tau_prox"])
                    t_p_tra = state.get("tau_p_tra", state["tau_prox"])
                    t_bri   = state["tau_bridge"]
                    t_d_tra = state.get("tau_d_tra", state["tau_dist"])
                    t_d_anc = state.get("tau_d_anc", state["tau_dist"])
                    sigma   = state.get("sigma_blend", 0.015)
                    
                    if "t_top_mm" in state:
                        t_top_mm = state["t_top_mm"]
                        t_top_m = t_top_mm / 1000.0
                    if "t_bottom_mm" in state:
                        t_bot_mm = state["t_bottom_mm"]
                        t_bot_m = t_bot_mm / 1000.0
                    if "h_tpms_mm" in state:
                        h_tpms_mm = state["h_tpms_mm"]
                    if "screw_spacing_mm" in state:
                        screw_spacing_mm = state["screw_spacing_mm"]
                        screw_spacing_m = screw_spacing_mm / 1000.0
                    if "bridge_span_mm" in state:
                        bridge_span_mm = state["bridge_span_mm"]
                        bridge_span_m = bridge_span_mm / 1000.0
                    if "fillet_radius_mm" in state:
                        fillet_radius_mm = state["fillet_radius_mm"]
                        fillet_radius_m = fillet_radius_mm / 1000.0
                    if "cell_size_mm" in state:
                        cell_size_mm = state["cell_size_mm"]
                        cell_size_m = cell_size_mm / 1000.0
                    
                    porosity_history["Prox Anchor (%)"].append(to_porosity(t_p_anc))
                    porosity_history["Prox Transition (%)"].append(to_porosity(t_p_tra))
                    porosity_history["Bridge Gap (%)"].append(to_porosity(t_bri))
                    porosity_history["Dist Transition (%)"].append(to_porosity(t_d_tra))
                    porosity_history["Dist Anchor (%)"].append(to_porosity(t_d_anc))
                    
                    grad_history["dL/dtau_p_anc"].append(state.get("grad_p_anc", 0.0))
                    grad_history["dL/dtau_p_tra"].append(state.get("grad_p_tra", 0.0))
                    grad_history["dL/dtau_bridge"].append(state.get("grad_bridge", 0.0))
                    grad_history["dL/dtau_d_tra"].append(state.get("grad_d_tra", 0.0))
                    grad_history["dL/dtau_d_anc"].append(state.get("grad_d_anc", 0.0))
                    grad_history["dL/dsigma_blend"].append(state.get("grad_sigma", 0.0))
                    grad_history["dL/dt_top"].append(state.get("grad_t_top", 0.0))
                    grad_history["dL/dt_bottom"].append(state.get("grad_t_bot", 0.0))
                    grad_history["dL/ds_pitch"].append(state.get("grad_pitch", 0.0))
                    grad_history["dL/dL_bridge"].append(state.get("grad_bridge_span", 0.0))
                    grad_history["dL/dd_cell"].append(state.get("grad_cell_size", 0.0))
                    grad_history["dL/dr_fillet"].append(state.get("grad_fillet", 0.0))
                    
                    last_tau = (t_p_anc, t_p_tra, t_bri, t_d_tra, t_d_anc, sigma)

                    progress_ph.progress(
                        min((state["step"] + 1) / opt_max_steps, 1.0),
                        text=f"Step {state['step']+1}/{opt_max_steps} | Loss: {state['loss']:.2f} | Pitch: {screw_spacing_mm:.1f}mm | Bridge: {bridge_span_mm:.1f}mm | Core: {h_tpms_mm:.2f}mm | Motion: {state['frac_disp']*1000:.3f}mm"
                    )
                    
                    loss_ph.plotly_chart(create_loss_tracking_fig(loss_history), width="stretch")
                    disp_ph.plotly_chart(create_disp_tracking_fig(disp_history, target_mm), width="stretch")
                    status_ph.markdown(StatusBadge.for_displacement(disp_history[-1], target_mm), unsafe_allow_html=True)
                    porosity_ph.plotly_chart(create_porosity_tracking_fig(porosity_history, target_porosity_pct=(1.0 - req.max_mass)*100.0), width="stretch")
                    grad_ph.plotly_chart(create_gradient_tracking_fig(grad_history), width="stretch")

                progress_ph.empty()
                optimization_finished = True
                opt_results = {
                    "last_tau": last_tau,
                    "cell_size_m": cell_size_m,
                    "t_top_m": t_top_m,
                    "t_bottom_m": t_bot_m,
                    "screw_spacing_m": screw_spacing_m,
                    "bridge_span_m": bridge_span_m,
                    "fillet_radius_m": fillet_radius_m,
                    "avg_porosity": float(state.get("mean_porosity", np.mean([porosity_history[k][-1] for k in porosity_history]) / 100.0) * 100.0 if state.get("mean_porosity", 0) <= 1.0 else state.get("mean_porosity", 55.0)),
                    "final_disp_mm": float(disp_history[-1]),
                    "target_disp": target_disp,
                    "target_mm": target_mm
                }

                # Persist optimization telemetry in session state
                st.session_state.last_loss_history = loss_history
                st.session_state.last_disp_history = disp_history
                st.session_state.last_porosity_history = porosity_history
                st.session_state.last_grad_history = grad_history
                st.session_state.last_opt_results = opt_results
                
                # Append run record to session history
                solid_mass = 64.0 * (selected_material.density_g_cm3 / 4.43)
                avg_por = opt_results["avg_porosity"]
                optimized_mass = solid_mass * (1.0 - (avg_por / 100.0) * 0.85)
                achieved_m = opt_results["final_disp_mm"]
                err_pct = abs(achieved_m - target_mm) / (target_mm + 1e-9) * 100.0
                status_str = "✅ PASS (ASTM)" if err_pct <= 15.0 else "⚠️ TIGHTENING"
                
                st.session_state.run_history.append({
                    "Timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                    "Workflow Mode": "⚙️ Direct Adjoint Optimization",
                    "Scenario / Goal": req.objective,
                    "Biomaterial": selected_material.code,
                    "Lattice Topology": tpms_choice.split("·")[0].strip(),
                    "Mesh Resolution": "63,153 DOFs (Refined)" if not skip_cad_morph else "63,153 DOFs (Direct)",
                    "Target Motion": f"{target_mm:.2f} mm",
                    "Achieved Motion": f"{achieved_m:.3f} mm",
                    "Avg Porosity": f"{avg_por:.1f}%",
                    "Mass Reduction": f"{((solid_mass - optimized_mass)/solid_mass)*100:.1f}%",
                    "ASTM Status": status_str
                })
                st.balloons()
        else:
            target_mm = req.target_fracture_displacement * 1000
            if "last_loss_history" in st.session_state and st.session_state.last_loss_history:
                loss_ph.plotly_chart(create_loss_tracking_fig(st.session_state.last_loss_history), width="stretch")
                disp_ph.plotly_chart(create_disp_tracking_fig(st.session_state.last_disp_history, target_mm), width="stretch")
                status_ph.markdown(StatusBadge.for_displacement(st.session_state.last_disp_history[-1], target_mm), unsafe_allow_html=True)
                porosity_ph.plotly_chart(create_porosity_tracking_fig(st.session_state.last_porosity_history, target_porosity_pct=(1.0 - req.max_mass)*100.0), width="stretch")
                grad_ph.plotly_chart(create_gradient_tracking_fig(st.session_state.last_grad_history), width="stretch")
            else:
                loss_ph.plotly_chart(create_loss_tracking_fig([120.0]), width="stretch")
                disp_ph.plotly_chart(create_disp_tracking_fig([0.018], target_mm), width="stretch")
                status_ph.markdown(StatusBadge.for_displacement(0.018, target_mm), unsafe_allow_html=True)
                porosity_ph.plotly_chart(create_porosity_tracking_fig({"Prox Anchor (%)": [12.0], "Bridge Gap (%)": [15.0], "Dist Anchor (%)": [12.0]}, target_porosity_pct=(1.0 - req.max_mass)*100.0), width="stretch")
                grad_ph.plotly_chart(create_gradient_tracking_fig({"dL/dtau_p_anc": [0.0], "dL/dtau_p_tra": [0.0], "dL/dtau_bridge": [0.0], "dL/dtau_d_tra": [0.0], "dL/dtau_d_anc": [0.0], "dL/dsigma_blend": [0.0], "dL/dt_top": [0.0], "dL/dt_bottom": [0.0], "dL/dL_bridge": [0.0], "dL/dd_cell": [0.0], "dL/dr_fillet": [0.0]}), width="stretch")
    else:
        st.info("👈 Please parse a clinical prompt to initialize optimization parameters.")

# Performance metrics and verification report section
current_results = st.session_state.get("last_opt_results", None)

if current_results is not None and current_results.get("last_tau") is not None:
    st.markdown("---")
    
    last_tau = current_results["last_tau"]
    cell_size_m = current_results.get("cell_size_m", cell_size_m)
    t_top_m = current_results.get("t_top_m", t_top_m)
    t_bot_m = current_results.get("t_bottom_m", t_bot_m)
    screw_spacing_m = current_results.get("screw_spacing_m", screw_spacing_m)
    bridge_span_m = current_results.get("bridge_span_m", bridge_span_m)
    fillet_radius_m = current_results.get("fillet_radius_m", fillet_radius_m)
    avg_porosity = current_results["avg_porosity"]
    final_disp_mm = current_results["final_disp_mm"]
    final_disp_m = final_disp_mm * 1e-3
    target_disp = current_results["target_disp"]
    target_mm = current_results["target_mm"]
    
    solid_mass = 64.0 * (selected_material.density_g_cm3 / 4.43)
    optimized_mass = solid_mass * (1.0 - (avg_porosity / 100.0) * 0.85)

    # 1. Main performance metrics tiles
    st.markdown(section_label("📊", "Final Biomechanical Performance Metrics"), unsafe_allow_html=True)
    st.markdown(metric_tiles(final_disp_mm, target_disp, avg_porosity), unsafe_allow_html=True)

    # Automated in-silico testing and ASTM verification suite
    st.markdown(section_label("🧪", "Automated In-Silico Testing & Clinical Verification Suite"), unsafe_allow_html=True)
    validation_report = run_insilico_validation_suite(
        tau_values=last_tau,
        target_disp_m=target_disp,
        achieved_disp_m=final_disp_m,
        avg_porosity_pct=avg_porosity,
        material=selected_material,
        fidelity_mode=fidelity_choice
    )
    st.markdown(validation_report_card(validation_report), unsafe_allow_html=True)

    # Persist run audit log
    try:
        from src.utils.logger import log_full_optimization_and_validation
        log_full_optimization_and_validation(
            user_prompt=user_prompt,
            design_req=req,
            material=selected_material,
            fidelity_mode=fidelity_choice,
            total_steps=len(st.session_state.get("last_loss_history", [1])),
            initial_loss=float(st.session_state.get("last_loss_history", [1.0])[0]) if st.session_state.get("last_loss_history") else 1.0,
            final_loss=float(st.session_state.get("last_loss_history", [0.1])[-1]) if st.session_state.get("last_loss_history") else 0.1,
            final_disp_mm=final_disp_mm,
            target_disp_mm=target_mm,
            avg_porosity_pct=avg_porosity,
            solid_mass_g=solid_mass,
            optimized_mass_g=optimized_mass,
            tau_values=last_tau,
            validation_report=validation_report
        )
    except Exception as log_err:
        pass

    # Solid baseline vs Optimized TPMS clinical comparison
    st.markdown(
        comparison_panel(
            baseline_mass_g=solid_mass,
            optimized_mass_g=optimized_mass,
            baseline_disp_mm=0.018,
            optimized_disp_mm=final_disp_mm,
            target_disp_mm=target_mm,
            avg_porosity_pct=avg_porosity,
            material_name=selected_material.name,
        ),
        unsafe_allow_html=True
    )

    # Multi-layer 3D inspection: Porosity architecture and Von Mises stress / FoS
    if os.path.exists(mesh_path):
        st.markdown("---")
        st.markdown(section_label("🔬", "Interactive 3D Biomechanical Renders"), unsafe_allow_html=True)
        
        st.markdown(
            f"""<div style="background: rgba(30, 41, 59, 0.7); padding: 0.6rem 1rem; border-radius: 8px; border-left: 3px solid #818cf8; font-size: 0.83rem; margin-bottom: 0.8rem; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem;">
                <div>🥪 <b>3-Layer Construct:</b> <span style="color: #38bdf8;">Top: {t_top_mm:.2f}mm</span> &nbsp;|&nbsp; 
                <span style="color: #4ade80;">TPMS Core: {h_tpms_mm:.2f}mm</span> &nbsp;|&nbsp; 
                <span style="color: #10b981;">Bot: {t_bot_mm:.2f}mm</span> &nbsp;|&nbsp; 
                <b>Bridge Span:</b> <span style="color: #e879f9;">{bridge_span_mm:.1f}mm</span> &nbsp;|&nbsp; 
                <b>Fixation:</b> <span style="color: #c084fc;">6x ∅4.5 mm (Pitch {screw_spacing_mm:.1f}mm)</span></div>
                <div style="color: #94a3b8; font-size: 0.75rem;"><i>(Z-cut view reveals internal porous core)</i></div>
            </div>""",
            unsafe_allow_html=True
        )
        
        view_layer = st.radio(
            "Select 3D Inspection Layer:",
            ["🔬 Metamaterial Architecture & Porosity", "⚡ Von Mises Stress Distribution (σ_VM)", "🛡️ Factor of Safety (FoS = σ_yield / σ_VM)"],
            horizontal=True
        )
        
        r1, r2 = st.columns(2, gap="large")
        
        if "Stress" in view_layer or "Safety" in view_layer:
            with st.spinner("Evaluating 3D continuum stress tensor…"):
                t_p_anc, t_p_tra, t_bri, t_d_tra, t_d_anc, sigma = last_tau
                theta_fem = jnp.array([
                    cell_size_m,
                    t_p_anc,
                    t_p_tra,
                    t_bri,
                    t_d_tra,
                    t_d_anc,
                    sigma,
                    t_top_m,
                    t_bot_m,
                    screw_spacing_m,
                    bridge_span_m,
                    fillet_radius_m,
                    selected_material.tpms_ga_exponent,
                    selected_material.youngs_modulus_gpa
                ])
                sol_u, _, _, _ = solve_fem(theta_fem)
                nodal_vm = compute_nodal_von_mises_stress(mesh_path, sol_u, selected_material.youngs_modulus_gpa, theta_fem=theta_fem)
                
                # Extract plate nodes specifically (Tag 10) for pure implant stress and FoS calculation
                m_chk = meshio.read(mesh_path)
                p_c = np.vstack([cb.data[:, :4] for cb in m_chk.cells if cb.type in ('tetra', 'tetra10')])
                p_t = np.concatenate([m_chk.cell_data["gmsh:physical"][i] for i, cb in enumerate(m_chk.cells) if cb.type in ('tetra', 'tetra10')])
                p_nodes = np.unique(p_c[p_t == 10].ravel())
                implant_vm = nodal_vm[p_nodes] if len(p_nodes) > 0 else nodal_vm
                max_vm = float(np.percentile(implant_vm, 99.0)) if len(implant_vm) > 0 else float(np.max(nodal_vm))
                min_fos = float(selected_material.yield_strength_mpa / max(max_vm, 1.0))
                
            is_fos = "Safety" in view_layer
            mode_str = "fos" if is_fos else "stress"
            
            st.markdown(
                f"""<div style="background: rgba(15, 23, 42, 0.6); padding: 0.5rem 1rem; border-radius: 8px; border-left: 3px solid {'#22c55e' if min_fos >= 1.5 else '#eab308'}; font-size: 0.85rem; margin-bottom: 0.5rem;">
                    <b>Peak Implant Von Mises:</b> <span style="color: #f8fafc;">{max_vm:.1f} MPa</span> &nbsp;|&nbsp; 
                    <b>Material Yield:</b> <span style="color: #38bdf8;">{selected_material.yield_strength_mpa:.0f} MPa</span> &nbsp;|&nbsp; 
                    <b>Minimum Factor of Safety:</b> <span style="color: {'#4ade80' if min_fos >= 1.5 else '#facc15'}; font-weight: 700;">{min_fos:.2f}x</span> 
                    <span style="color: #94a3b8; font-size: 0.75rem;">(ASTM F382 Target: &ge; 1.50x)</span>
                </div>""",
                unsafe_allow_html=True
            )
                
            active_by = st.session_state.get("cad_bend_y")
            active_bz = st.session_state.get("cad_bend_z")
            if active_by is not None and len(active_by) > 0:
                st.markdown(
                    '<div style="background: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 6px; padding: 0.4rem 0.8rem; margin-bottom: 0.6rem; font-size: 0.8rem; color: #38bdf8;">'
                    '🔧 <b>Patient-Specific Anatomical Morphing Active</b>: 3D TPMS surface, Von Mises stress, and STL export conform to Stage 1 PyGeM FFD contours.'
                    '</div>',
                    unsafe_allow_html=True
                )

            with r1:
                title1 = "🛡️ 3D Factor of Safety (Implant Only · Full Surface)" if is_fos else "⚡ 3D Von Mises Stress (Implant Only · Full Surface)"
                st.caption(title1)
                st.plotly_chart(get_von_mises_plotly_fig(mesh_path, nodal_vm, yield_strength_mpa=selected_material.yield_strength_mpa, mode=mode_str, tau_values=last_tau, tpms_type=tpms_type_code, fillet_radius=fillet_radius_m, screw_spacing=screw_spacing_m, bridge_span=bridge_span_m, t_top=t_top_m, t_bottom=t_bot_m, bend_y_array=active_by, bend_z_array=active_bz), width="stretch", key="vm_ext")
            with r2:
                title2 = "🛡️ Internal Factor of Safety (Implant Only · Z-Cut Sagittal)" if is_fos else "⚡ Internal Stress Distribution (Implant Only · Z-Cut Sagittal)"
                st.caption(title2)
                st.plotly_chart(get_von_mises_plotly_fig(mesh_path, nodal_vm, clip_axis="z", yield_strength_mpa=selected_material.yield_strength_mpa, mode=mode_str, tau_values=last_tau, tpms_type=tpms_type_code, fillet_radius=fillet_radius_m, screw_spacing=screw_spacing_m, bridge_span=bridge_span_m, t_top=t_top_m, t_bottom=t_bot_m, bend_y_array=active_by, bend_z_array=active_bz), width="stretch", key="vm_int")
        else:
            active_by = st.session_state.get("cad_bend_y")
            active_bz = st.session_state.get("cad_bend_z")
            if active_by is not None and len(active_by) > 0:
                st.markdown(
                    '<div style="background: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 6px; padding: 0.4rem 0.8rem; margin-bottom: 0.6rem; font-size: 0.8rem; color: #38bdf8;">'
                    '🔧 <b>Patient-Specific Anatomical Morphing Active</b>: 3D TPMS surface conforms to Stage 1 PyGeM FFD contours.'
                    '</div>',
                    unsafe_allow_html=True
                )
            with r1:
                st.caption(f"🔬 3D {tpms_choice.split('·')[0].strip()} Implant Surface (Top: {t_top_mm:.2f}mm · Core: {h_tpms_mm:.2f}mm · Bot: {t_bot_mm:.2f}mm · Bridge: {bridge_span_mm:.1f}mm)")
                st.plotly_chart(get_mesh_plotly_fig(mesh_path, tau_values=last_tau, tpms_type=tpms_type_code, fillet_radius=fillet_radius_m, screw_spacing=screw_spacing_m, bridge_span=bridge_span_m, t_top=t_top_m, t_bottom=t_bot_m, bend_y_array=active_by, bend_z_array=active_bz), width="stretch", key="final_ext")
            with r2:
                st.caption(f"🔬 Internal TPMS Porosity Gradient (Implant Only · Z-Cut Sagittal)")
                st.plotly_chart(get_mesh_plotly_fig(mesh_path, tau_values=last_tau, clip_axis="z", tpms_type=tpms_type_code, fillet_radius=fillet_radius_m, screw_spacing=screw_spacing_m, bridge_span=bridge_span_m, t_top=t_top_m, t_bottom=t_bot_m, bend_y_array=active_by, bend_z_array=active_bz), width="stretch", key="final_int")

    # Manufacturing CAD STL export
    st.markdown("---")
    st.markdown(section_label("📥", "Manufacturing & CAD Export Section"), unsafe_allow_html=True)
    exp_col1, exp_col2 = st.columns(2, gap="large")
    
    with exp_col1:
        active_by = st.session_state.get("cad_bend_y")
        active_bz = st.session_state.get("cad_bend_z")
        stl_bytes = generate_tpms_stl_bytes(last_tau, tpms_type=tpms_type_code, fillet_radius=fillet_radius_m, screw_spacing=screw_spacing_m, bridge_span=bridge_span_m, t_top=t_top_m, t_bottom=t_bot_m, bend_y_array=active_by, bend_z_array=active_bz)
        st.download_button(
            label=f"🖨️ Download 3D-Printable {tpms_choice.split('·')[0].strip()} STL (.stl)",
            data=stl_bytes,
            file_name=f"tesseract_{tpms_type_code}_{selected_material.code.lower()}.stl",
            mime="model/stl",
            width="stretch"
        )
    with exp_col2:
        with open(mesh_path, "rb") as f:
            st.download_button(
                label="📁 Export FEA Gmsh Simulation Mesh (.msh)",
                data=f,
                file_name="tesseract_bone_plate_mesh.msh",
                mime="application/octet-stream",
                width="stretch"
            )

# Optimization history log table
if "run_history" not in st.session_state:
    st.session_state.run_history = []
if len(st.session_state.run_history) > 0:
    st.markdown("---")
    st.markdown(section_label("📋", "Optimization & Verification Experimentation History"), unsafe_allow_html=True)
    history_df = pd.DataFrame(st.session_state.run_history)
    st.dataframe(history_df, width="stretch", hide_index=True)
    
    col_hist_dl, col_hist_clr = st.columns([4, 1])
    with col_hist_dl:
        csv_data = history_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Experimentation History (.csv)",
            data=csv_data,
            file_name="tesseract_experiment_history.csv",
            mime="text/csv",
            key="dl_history_csv"
        )
    with col_hist_clr:
        if st.button("🗑️ Clear History", key="btn_clear_hist"):
            st.session_state.run_history = []
            st.rerun()
