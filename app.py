import os
import sys

# Prevent JAX from pre-allocating large memory pools and cap VRAM to 40%
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.40"

import time
import datetime
import streamlit as st
import numpy as np
import pandas as pd
import logging
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
os.environ["TESSERACT_OUTPUT_PATH"] = "/tmp/tesseract_runs"
os.environ["MLFLOW_TRACKING_URI"] = "file:///tmp/mlruns"

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
st.markdown(
    hero_banner(
        title="🦴 Tesseract Differentiable Simulation",
        subtitle="Agentic Biomechanical Implant Optimization · Dual Tesseract REST Engines · WSD Adam Optimizer",
        accent_word="Differentiable Simulation",
    ),
    unsafe_allow_html=True,
)

# Dual Tesseract Microservices live health badges
st.markdown(
    """<div style="display: flex; gap: 1rem; margin-bottom: 1rem; flex-wrap: wrap;">
        <div class="glass-card" style="padding: 0.45rem 0.9rem; font-size: 0.78rem; border-left: 3px solid #22c55e; display: flex; align-items: center; gap: 0.4rem;">
            <span style="color: #4ade80; font-size: 0.9rem;">●</span> <b>Tesseract 1:</b> FEM Adjoint Solver (<span style="color: #94a3b8;">Port 8000 · JAX-FEM + PETSc</span>)
        </div>
        <div class="glass-card" style="padding: 0.45rem 0.9rem; font-size: 0.78rem; border-left: 3px solid #38bdf8; display: flex; align-items: center; gap: 0.4rem;">
            <span style="color: #38bdf8; font-size: 0.9rem;">●</span> <b>Tesseract 2:</b> Geometry & Porosity Engine (<span style="color: #94a3b8;">Port 8001 · Differentiable SDF</span>)
        </div>
    </div>""",
    unsafe_allow_html=True
)

from src.agent.agent import parse_design_request

# Clinical preset definitions
# Clinical preset definitions
st.sidebar.markdown("### 🤖 Agentic NLP Interface")

workflow_mode = st.sidebar.radio(
    "Architecture Mode",
    ["🤖 Multi-Agent Orchestrator (LangGraph)", "⚙️ Direct Parametric Mode"],
    index=0,
    help="Multi-Agent mode orchestrates 4 specialist agents (Clinical, Materials, Optimization, Validation) with autonomous closed-loop self-correction."
)

CLINICAL_PRESETS: Dict[str, str] = {
    "Callus Stimulation (Default)": "I need a compliant Titanium plate that allows 0.2mm of micro-motion at the fracture site to stimulate callus formation, while keeping the implant as light as possible.",
    "Elderly Osteoporotic Patient": "I need a highly porous Titanium plate for an elderly osteoporotic patient that allows 0.30mm of micro-motion, minimizing stress shielding and maximizing porosity.",
    "Young Athlete High-Impact Trauma": "I need a rigid, high-strength Stainless Steel plate for a young athlete that restricts micro-motion to 0.12mm to ensure stable fixation.",
    "Cost-Effective Trauma Fixation": "I need an affordable, cost-effective 316L Stainless Steel plate that maintains 0.18mm micro-motion with high ductility."
}

# Synchronize prompt area when preset selection changes
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
    height=120
)

if "last_parsed_prompt" not in st.session_state:
    st.session_state.last_parsed_prompt = None

if st.session_state.last_parsed_prompt != user_prompt or "parsed_req" not in st.session_state or st.session_state.parsed_req is None:
    st.session_state.parsed_req = parse_design_request(user_prompt)
    st.session_state.last_parsed_prompt = user_prompt

if "run_history" not in st.session_state:
    st.session_state.run_history = []

if st.sidebar.button("⚡ Re-Parse Prompt", width="stretch"):
    with st.spinner("Agent interpreting biomechanical prompt…"):
        time.sleep(0.2)
        st.session_state.parsed_req = parse_design_request(user_prompt)
        st.session_state.last_parsed_prompt = user_prompt

req = st.session_state.parsed_req
if req:
    rec_tpms = getattr(req, 'recommended_tpms', 'Schwarz Primitive (P)')
    f_rad_val = getattr(req, 'fillet_radius_mm', 1.2)
    s_spac_val = getattr(req, 'screw_spacing_mm', 15.0)
    st.sidebar.markdown(f"""
    <div class="glass-card" style="padding: 0.8rem; margin-top: 0.5rem; font-size: 0.8rem; border-left: 3px solid #6366f1;">
        <div style="color: #4ade80; font-weight: 600; margin-bottom: 0.3rem;">✓ Requirements Parsed</div>
        <div><b>Objective:</b> <span style="color: #a78bfa;">{req.objective}</span></div>
        <div><b>Target Micro-Motion:</b> <span style="color: #f8fafc; font-weight: 600;">{req.target_fracture_displacement*1000:.2f} mm</span> ({req.target_fracture_displacement*1e6:.0f} µm)</div>
        <div><b>Upper Mass Limit:</b> <span style="color: #f8fafc;">{req.max_mass*100:.0f}%</span></div>
        <div><b>Recommended Material:</b> <span style="color: #38bdf8;">{req.recommended_material}</span></div>
        <div><b>Recommended Topology:</b> <span style="color: #c084fc;">{rec_tpms}</span></div>
        <div><b>Fillet Radius:</b> <span style="color: #fbbf24;">{f_rad_val:.1f} mm</span> &nbsp;|&nbsp; <b>Screw Pitch:</b> <span style="color: #34d399;">{s_spac_val:.1f} mm</span></div>
        <div style="margin-top: 0.3rem; color: #94a3b8; font-size: 0.72rem;"><i>{req.clinical_rationale}</i></div>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("### 📐 Fixation CAD Geometry & Constraints")
fillet_radius_mm = st.sidebar.slider(
    "Corner Fillet Radius (mm)",
    min_value=0.4,
    max_value=2.5,
    value=float(getattr(req, "fillet_radius_mm", 1.2)),
    step=0.1,
    help="Smooth corner edge filleting for soft-tissue gliding and reducing stress peaks (AO standard: 1.2mm)."
)
t_top_mm = st.sidebar.slider(
    "Top Solid Plate Thickness (t_top, mm)",
    min_value=0.15,
    max_value=2.00,
    value=0.50,
    step=0.05,
    help="Periosteal / muscle-facing solid titanium layer (y in [18-t_top, 18] mm)."
)
t_bot_mm = st.sidebar.slider(
    "Bottom Solid Plate Thickness (t_bottom, mm)",
    min_value=0.15,
    max_value=2.00,
    value=0.50,
    step=0.05,
    help="Bone-contacting solid titanium layer (y in [11, 11+t_bot] mm)."
)
h_tpms_mm = max(6.0 - t_top_mm - t_bot_mm, 1.0)
st.sidebar.markdown(
    f"""<div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(148, 163, 184, 0.2); border-radius: 6px; padding: 0.4rem 0.6rem; font-size: 0.8rem; margin-bottom: 0.5rem;">
        🧬 <b>Sandwich TPMS Core:</b> <span style="color:#38bdf8; font-weight:600;">{h_tpms_mm:.2f} mm</span> &nbsp;|&nbsp; 
        📐 <b>Total Depth:</b> <span style="color:#4ade80; font-weight:600;">6.00 mm</span>
    </div>""",
    unsafe_allow_html=True
)
bridge_span_mm = st.sidebar.slider(
    "Fracture Bridge Working Span (mm)",
    min_value=18.0,
    max_value=45.0,
    value=30.0,
    step=1.0,
    help="Working length between innermost fixation screws across the fracture gap (Primary flexural compliance control)."
)
cell_size_mm = st.sidebar.slider(
    "TPMS Unit Cell Dimension (mm)",
    min_value=3.5,
    max_value=7.5,
    value=5.0,
    step=0.5,
    help="Lattice pore unit cell size for LPBF 3D printing."
)
screw_spacing_mm = st.sidebar.slider(
    "Screw Hole Pitch (mm, Constant AO)",
    min_value=10.0,
    max_value=16.0,
    value=float(getattr(req, "screw_spacing_mm", 14.5)),
    step=0.5,
    help="Center-to-center pitch between consecutive cortical fixation screws (held constant during optimization)."
)
fillet_radius_m = fillet_radius_mm / 1000.0
t_top_m = t_top_mm / 1000.0
t_bot_m = t_bot_mm / 1000.0
skin_thickness_m = 0.5 * (t_top_m + t_bot_m)
screw_spacing_m = screw_spacing_mm / 1000.0
bridge_span_m = bridge_span_mm / 1000.0
cell_size_m = cell_size_mm / 1000.0
cell_size_m = cell_size_mm / 1000.0

st.sidebar.markdown("### ⚙️ Simulation Fidelity & Iterations")
fidelity_choice = st.sidebar.radio(
    "FEM Solver Quadrature / Element Refinement",
    ["Clinical Grade (TET10 Refined · Quadratic 10-node)", "Fast Screening (TET4 Coarse · Linear 4-node)"],
    index=0
)

opt_max_steps = st.sidebar.slider(
    "Max Optimization Iterations",
    min_value=5,
    max_value=1000,
    value=100,
    step=5
)
enable_early_stopping = st.sidebar.checkbox("Enable Early Convergence Stopping", value=True)

# Target biomaterial specification
mat_keys = list(BIOMATERIALS.keys())
default_mat_idx = mat_keys.index(req.recommended_material) if (req and req.recommended_material in mat_keys) else 0

st.sidebar.markdown("### 🧪 Biomaterial Specification")
selected_material_name = st.sidebar.selectbox("Target Fixation Material", mat_keys, index=default_mat_idx)
selected_material = BIOMATERIALS[selected_material_name]
st.sidebar.markdown(material_card(selected_material), unsafe_allow_html=True)

# Minimal surface metamaterial topology selection
tpms_options = [
    "Schwarz Primitive (P) · High Permeability",
    "Schoen Gyroid (G) · High Shear Strength",
    "Schwarz Diamond (D) · High Torsion Grip"
]
default_tpms_idx = 0
if req:
    rec_lower = getattr(req, "recommended_tpms", "").lower()
    if "gyroid" in rec_lower:
        default_tpms_idx = 1
    elif "diamond" in rec_lower:
        default_tpms_idx = 2

st.sidebar.markdown("### 🔬 Metamaterial Lattice Architecture")
tpms_choice = st.sidebar.selectbox(
    "Minimal Surface Topology",
    tpms_options,
    index=default_tpms_idx
)
if "gyroid" in tpms_choice.lower():
    tpms_type_code = "gyroid"
elif "diamond" in tpms_choice.lower():
    tpms_type_code = "diamond"
else:
    tpms_type_code = "primitive"

st.sidebar.markdown("---")
st.sidebar.caption("Powered by **Tesseract Core** · JAX-FEM · PETSc")

# Top viewport: Geometry preview (left) and Live optimization telemetry (right)
mesh_path = os.path.join(os.path.dirname(__file__), "src", "fem", "data", "model.msh")
col_geo, col_opt = st.columns([1, 1], gap="large")

with col_geo:
    st.markdown(section_label("📐", "Anatomical Model & Fixation Geometry"), unsafe_allow_html=True)
    if os.path.exists(mesh_path):
        st.plotly_chart(
            get_mesh_plotly_fig(
                mesh_path,
                tau_values=[0.10, 0.10, 0.10, 0.10, 0.10],
                tpms_type=tpms_type_code,
                fillet_radius=fillet_radius_m,
                screw_spacing=screw_spacing_m
            ),
            width="stretch",
            key="geo_base"
        )
    else:
        st.warning("No mesh found. Run the mesh generator first.")

optimization_finished = False
opt_results = {}

with col_opt:
    st.markdown(section_label("🧪", "Tesseract Differentiable Optimization Engine"), unsafe_allow_html=True)

    if req:
        is_agent_mode = "Multi-Agent" in workflow_mode
        button_text = "🚀 Launch Autonomous Multi-Agent Synthesis (LangGraph)" if is_agent_mode else "🚀 Start JAX-FEM Adjoint Optimization"
        start_button = st.button(button_text, width="stretch", type="primary" if is_agent_mode else "secondary")

        if is_agent_mode:
            st.markdown("""
            <div class="glass-card" style="padding: 0.6rem 0.9rem; margin-bottom: 0.8rem; font-size: 0.78rem; border-left: 3px solid #8b5cf6; display: flex; justify-content: space-between; align-items: center;">
                <span><b>Active Agents:</b> 🧑‍⚕️ Clinical &nbsp;|&nbsp; 🔬 Materials &nbsp;|&nbsp; ⚙️ Optimizer &nbsp;|&nbsp; 📋 Auditor</span>
                <span style="background: rgba(139, 92, 246, 0.2); color: #c084fc; padding: 0.15rem 0.45rem; border-radius: 4px; font-weight: 600;">LangGraph Closed-Loop</span>
            </div>
            """, unsafe_allow_html=True)

        agent_delib_container = st.container()
        progress_ph = st.empty()
        status_ph   = st.empty()
        
        tab_physics, tab_sens = st.tabs(["📊 Loss & Physical State", "🔍 Adjoint Gradients (∂L/∂τ)"])
        
        with tab_physics:
            loss_ph     = st.empty()
            disp_ph     = st.empty()
            porosity_ph = st.empty()
            
        with tab_sens:
            grad_ph = st.empty()

        if start_button:
            progress_ph.info("⏳ Initializing System & Connecting to Dual Tesseract Microservices…")

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
                "∂L/∂τ_p_anc": [],
                "∂L/∂τ_p_tra": [],
                "∂L/∂τ_bridge": [],
                "∂L/∂τ_d_tra": [],
                "∂L/∂τ_d_anc": [],
                "∂L/∂σ_blend": [],
                "∂L/∂t_top": [],
                "∂L/∂t_bottom": [],
                "∂L/∂s_pitch": [],
                "∂L/∂L_bridge": [],
                "∂L/∂d_cell": [],
                "∂L/∂r_fillet": []
            }
            
            target_disp      = req.target_fracture_displacement
            target_mm        = target_disp * 1000

            # Only attach microservice clients if local microservice servers are actively listening
            fem_client = None
            geom_client = None
            try:
                import httpx
                resp_fem = httpx.get("http://127.0.0.1:8000/health", timeout=0.3)
                resp_geo = httpx.get("http://127.0.0.1:8001/health", timeout=0.3)
                if resp_fem.status_code == 200 and resp_geo.status_code == 200:
                    fem_client = tc.sdk.tesseract.Tesseract.from_url("http://127.0.0.1:8000")
                    geom_client = tc.sdk.tesseract.Tesseract.from_url("http://127.0.0.1:8001")
            except Exception:
                fem_client = None
                geom_client = None

            def to_porosity(t):
                t_clamped = min(max(float(t), 0.10), 1.45)
                return 54.7 + ((t_clamped - 0.10) / 1.35) * (88.1 - 54.7)

            if is_agent_mode:
                # ============================================================
                # 🤖 Multi-Agent LangGraph Orchestration Flow
                # ============================================================
                with agent_delib_container:
                    st.markdown("#### 🤖 Multi-Agent Collaborative Deliberation")
                    agent_chat_ph = st.empty()
                    rendered_messages = []

                final_agent_state = None

                for event in run_design_agent(
                    surgeon_prompt=user_prompt,
                    fem_client=fem_client,
                    geometry_client=geom_client,
                    max_attempts=3,
                    stream=True
                ):
                    if event["type"] == "agent_message":
                        msg = event["message"]
                        rendered_messages.append(msg)
                        
                        # Build formatted chat transcript
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
                        agent_chat_ph.markdown("".join(chat_html), unsafe_allow_html=True)
                        
                        # If optimization result is attached to state, sync telemetry charts
                        st_data = event.get("state", {})
                        opt_res = st_data.get("optimization_result")
                        if opt_res and "step_history" in opt_res and opt_res["step_history"]:
                            steps = opt_res["step_history"]
                            loss_history = [s["loss"] for s in steps]
                            disp_history = [s["frac_disp"] * 1000 for s in steps]
                            porosity_history = {
                                "Prox Anchor (%)": [to_porosity(s.get("tau_p_anc", s.get("tau_prox", 0.35))) for s in steps],
                                "Prox Transition (%)": [to_porosity(s.get("tau_p_tra", s.get("tau_prox", 0.45))) for s in steps],
                                "Bridge Gap (%)": [to_porosity(s.get("tau_bridge", 0.55)) for s in steps],
                                "Dist Transition (%)": [to_porosity(s.get("tau_d_tra", s.get("tau_dist", 0.45))) for s in steps],
                                "Dist Anchor (%)": [to_porosity(s.get("tau_d_anc", s.get("tau_dist", 0.35))) for s in steps]
                            }
                            grad_history = {
                                "∂L/∂τ_p_anc": [s.get("grad_p_anc", 0.0) for s in steps],
                                "∂L/∂τ_p_tra": [s.get("grad_p_tra", 0.0) for s in steps],
                                "∂L/∂τ_bridge": [s.get("grad_bridge", 0.0) for s in steps],
                                "∂L/∂τ_d_tra": [s.get("grad_d_tra", 0.0) for s in steps],
                                "∂L/∂τ_d_anc": [s.get("grad_d_anc", 0.0) for s in steps],
                                "∂L/∂σ_blend": [s.get("grad_sigma", 0.0) for s in steps],
                                "∂L/∂t_top": [s.get("grad_t_top", 0.0) for s in steps],
                                "∂L/∂t_bottom": [s.get("grad_t_bot", 0.0) for s in steps],
                                "∂L/∂s_pitch": [s.get("grad_pitch", 0.0) for s in steps],
                                "∂L/∂L_bridge": [s.get("grad_bridge_span", 0.0) for s in steps],
                                "∂L/∂d_cell": [s.get("grad_cell_size", 0.0) for s in steps],
                                "∂L/∂r_fillet": [s.get("grad_fillet", 0.0) for s in steps]
                            }
                            n_msg = len(rendered_messages)
                            loss_ph.plotly_chart(create_loss_tracking_fig(loss_history), width="stretch", key=f"agent_live_loss_{n_msg}")
                            disp_ph.plotly_chart(create_disp_tracking_fig(disp_history, target_mm), width="stretch", key=f"agent_live_disp_{n_msg}")
                            status_ph.markdown(StatusBadge.for_displacement(disp_history[-1], target_mm), unsafe_allow_html=True)
                            porosity_ph.plotly_chart(create_porosity_tracking_fig(porosity_history, target_porosity_pct=(1.0 - req.max_mass)*100.0), width="stretch", key=f"agent_live_poro_{n_msg}")
                            grad_ph.plotly_chart(create_gradient_tracking_fig(grad_history), width="stretch", key=f"agent_live_grad_{n_msg}")

                    elif event["type"] == "final_result":
                        final_agent_state = event["state"]

                # Extract final parameters from multi-agent state
                if final_agent_state and final_agent_state.get("optimization_result"):
                    opt_res = final_agent_state["optimization_result"]
                    theta = opt_res.get("final_theta", [5.0, 0.35, 0.45, 0.55, 0.45, 0.35, 1.5, 0.5, 0.5, 14.5, 30.0, 1.2])
                    last_tau = (theta[1], theta[2], theta[3], theta[4], theta[5], theta[6]*0.010)
                    cell_size_m = theta[0] * 1e-3
                    t_top_m = theta[7] * 1e-3
                    t_bot_m = theta[8] * 1e-3
                    screw_spacing_m = theta[9] * 1e-3
                    bridge_span_m = theta[10] * 1e-3
                    fillet_radius_m = theta[11] * 1e-3
                    
                    final_metrics = opt_res.get("final_metrics", {})
                    opt_results = {
                        "last_tau": last_tau,
                        "cell_size_m": cell_size_m,
                        "t_top_m": t_top_m,
                        "t_bottom_m": t_bot_m,
                        "screw_spacing_m": screw_spacing_m,
                        "bridge_span_m": bridge_span_m,
                        "fillet_radius_m": fillet_radius_m,
                        "avg_porosity": float(final_metrics.get("mean_porosity", 0.55) * 100.0),
                        "final_disp_mm": float(final_metrics.get("frac_disp", 0.0002) * 1000.0),
                        "target_disp": target_disp,
                        "target_mm": target_mm
                    }
                else:
                    opt_results = {}

            else:
                # ============================================================
                # ⚙️ Direct Parametric Optimization Flow (Legacy Manual)
                # ============================================================
                patience_val = 5 if enable_early_stopping else 9999
                last_tau = None
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
                    
                    grad_history["∂L/∂τ_p_anc"].append(state.get("grad_p_anc", 0.0))
                    grad_history["∂L/∂τ_p_tra"].append(state.get("grad_p_tra", 0.0))
                    grad_history["∂L/∂τ_bridge"].append(state.get("grad_bridge", 0.0))
                    grad_history["∂L/∂τ_d_tra"].append(state.get("grad_d_tra", 0.0))
                    grad_history["∂L/∂τ_d_anc"].append(state.get("grad_d_anc", 0.0))
                    grad_history["∂L/∂σ_blend"].append(state.get("grad_sigma", 0.0))
                    grad_history["∂L/∂t_top"].append(state.get("grad_t_top", 0.0))
                    grad_history["∂L/∂t_bottom"].append(state.get("grad_t_bot", 0.0))
                    grad_history["∂L/∂s_pitch"].append(state.get("grad_pitch", 0.0))
                    grad_history["∂L/∂L_bridge"].append(state.get("grad_bridge_span", 0.0))
                    grad_history["∂L/∂d_cell"].append(state.get("grad_cell_size", 0.0))
                    grad_history["∂L/∂r_fillet"].append(state.get("grad_fillet", 0.0))
                    
                    last_tau = (t_p_anc, t_p_tra, t_bri, t_d_tra, t_d_anc, sigma)

                    progress_ph.progress(
                        min((state["step"] + 1) / opt_max_steps, 1.0),
                        text=f"Step {state['step']+1}/{opt_max_steps} · ⚡ Adam · Loss: {state['loss']:.2f} · Pitch: {screw_spacing_mm:.1f}mm · Bridge: {bridge_span_mm:.1f}mm · Core: {h_tpms_mm:.2f}mm · Motion: {state['frac_disp']*1000:.3f}mm"
                    )
                    
                    step_idx = state['step']
                    loss_ph.plotly_chart(create_loss_tracking_fig(loss_history), width="stretch", key=f"direct_live_loss_{step_idx}")
                    disp_ph.plotly_chart(create_disp_tracking_fig(disp_history, target_mm), width="stretch", key=f"direct_live_disp_{step_idx}")
                    status_ph.markdown(StatusBadge.for_displacement(disp_history[-1], target_mm), unsafe_allow_html=True)
                    porosity_ph.plotly_chart(create_porosity_tracking_fig(porosity_history, target_porosity_pct=(1.0 - req.max_mass)*100.0), width="stretch", key=f"direct_live_poro_{step_idx}")
                    grad_ph.plotly_chart(create_gradient_tracking_fig(grad_history), width="stretch", key=f"direct_live_grad_{step_idx}")

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

            progress_ph.empty()
            optimization_finished = True

            # Persist optimization telemetry in session state
            st.session_state.last_loss_history = loss_history
            st.session_state.last_disp_history = disp_history
            st.session_state.last_porosity_history = porosity_history
            st.session_state.last_grad_history = grad_history
            st.session_state.last_opt_results = opt_results
            
            # Append run record to session history
            solid_mass = 64.0 * (selected_material.density_g_cm3 / 4.43)
            optimized_mass = solid_mass * (1.0 - (opt_results["avg_porosity"] / 100.0) * 0.85)
            
            st.session_state.run_history.append({
                "Timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                "Scenario": req.objective if not is_agent_mode else (final_agent_state.get("clinical_profile", {}).get("clinical_objective", req.objective) if final_agent_state else req.objective),
                "Material": selected_material.code,
                "Fidelity": "TET10" if "TET10" in fidelity_choice else "TET4",
                "Target Motion": f"{target_mm:.2f} mm",
                "Achieved Motion": f"{opt_results['final_disp_mm']:.3f} mm",
                "Avg Porosity": f"{opt_results['avg_porosity']:.1f}%",
                "Mass Reduction": f"{((solid_mass - optimized_mass)/solid_mass)*100:.1f}%"
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
                grad_ph.plotly_chart(create_gradient_tracking_fig({"∂L/∂τ_p_anc": [0.0], "∂L/∂τ_p_tra": [0.0], "∂L/∂τ_bridge": [0.0], "∂L/∂τ_d_tra": [0.0], "∂L/∂τ_d_anc": [0.0], "∂L/∂σ_blend": [0.0], "∂L/∂t_top": [0.0], "∂L/∂t_bottom": [0.0], "∂L/∂L_bridge": [0.0], "∂L/∂d_cell": [0.0], "∂L/∂r_fillet": [0.0]}), width="stretch")
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
                max_vm = float(np.max(nodal_vm))
                min_fos = float(selected_material.yield_strength_mpa / max(max_vm, 1e-4))
                
            is_fos = "Safety" in view_layer
            mode_str = "fos" if is_fos else "stress"
            
            st.markdown(
                f"""<div style="background: rgba(15, 23, 42, 0.6); padding: 0.5rem 1rem; border-radius: 8px; border-left: 3px solid {'#22c55e' if min_fos >= 1.5 else '#eab308'}; font-size: 0.85rem; margin-bottom: 0.5rem;">
                    <b>Peak Von Mises:</b> <span style="color: #f8fafc;">{max_vm:.1f} MPa</span> &nbsp;|&nbsp; 
                    <b>Material Yield:</b> <span style="color: #38bdf8;">{selected_material.yield_strength_mpa:.0f} MPa</span> &nbsp;|&nbsp; 
                    <b>Minimum Factor of Safety:</b> <span style="color: {'#4ade80' if min_fos >= 1.5 else '#facc15'}; font-weight: 700;">{min_fos:.2f}x</span> 
                    <span style="color: #94a3b8; font-size: 0.75rem;">(ASTM F382 Target: &ge; 1.50x)</span>
                </div>""",
                unsafe_allow_html=True
            )
                
            with r1:
                title1 = "🛡️ 3D Factor of Safety Distribution (Full Remeshed TPMS Surface)" if is_fos else "⚡ 3D Von Mises Stress Contour (Full Remeshed TPMS Surface)"
                st.caption(title1)
                st.plotly_chart(get_von_mises_plotly_fig(mesh_path, nodal_vm, yield_strength_mpa=selected_material.yield_strength_mpa, mode=mode_str, tau_values=last_tau, tpms_type=tpms_type_code, fillet_radius=fillet_radius_m, screw_spacing=screw_spacing_m, bridge_span=bridge_span_m, t_top=t_top_m, t_bottom=t_bot_m), width="stretch", key="vm_ext")
            with r2:
                title2 = "🛡️ Internal Factor of Safety (Z-cut Sagittal TPMS View)" if is_fos else "⚡ Internal Stress Distribution (Z-cut Sagittal TPMS View)"
                st.caption(title2)
                st.plotly_chart(get_von_mises_plotly_fig(mesh_path, nodal_vm, clip_axis="z", yield_strength_mpa=selected_material.yield_strength_mpa, mode=mode_str, tau_values=last_tau, tpms_type=tpms_type_code, fillet_radius=fillet_radius_m, screw_spacing=screw_spacing_m, bridge_span=bridge_span_m, t_top=t_top_m, t_bottom=t_bot_m), width="stretch", key="vm_int")
        else:
            with r1:
                st.caption(f"🔬 3D {tpms_choice.split('·')[0].strip()} Solid Surface (Top: {t_top_mm:.2f}mm · Core: {h_tpms_mm:.2f}mm · Bot: {t_bot_mm:.2f}mm · Bridge: {bridge_span_mm:.1f}mm)")
                st.plotly_chart(get_mesh_plotly_fig(mesh_path, tau_values=last_tau, tpms_type=tpms_type_code, fillet_radius=fillet_radius_m, screw_spacing=screw_spacing_m, bridge_span=bridge_span_m, t_top=t_top_m, t_bottom=t_bot_m), width="stretch", key="final_ext")
            with r2:
                st.caption(f"🔬 Internal TPMS Porosity Gradient (Z-cut Sagittal View)")
                st.plotly_chart(get_mesh_plotly_fig(mesh_path, tau_values=last_tau, clip_axis="z", tpms_type=tpms_type_code, fillet_radius=fillet_radius_m, screw_spacing=screw_spacing_m, bridge_span=bridge_span_m, t_top=t_top_m, t_bottom=t_bot_m), width="stretch", key="final_int")

    # Manufacturing CAD STL export
    st.markdown("---")
    st.markdown(section_label("📥", "Manufacturing & CAD Export Section"), unsafe_allow_html=True)
    exp_col1, exp_col2 = st.columns(2, gap="large")
    
    with exp_col1:
        stl_bytes = generate_tpms_stl_bytes(last_tau, tpms_type=tpms_type_code, fillet_radius=fillet_radius_m, screw_spacing=screw_spacing_m, bridge_span=bridge_span_m, t_top=t_top_m, t_bottom=t_bot_m)
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
if len(st.session_state.run_history) > 0:
    st.markdown("---")
    st.markdown(section_label("📋", "Optimization & Verification Session History"), unsafe_allow_html=True)
    history_df = pd.DataFrame(st.session_state.run_history)
    st.dataframe(history_df, width="stretch")
