"""
Interactive Plotly Real-Time Tracking Charts with Dotted Target Reference Lines.
"""

import plotly.graph_objects as go
from typing import List, Dict, Any


def create_disp_tracking_fig(disp_history: List[float], target_mm: float) -> go.Figure:
    """
    Plots real-time fracture micro-motion vs steps with a dotted horizontal target line.
    """
    steps = list(range(1, len(disp_history) + 1))
    fig = go.Figure()

    # Achieved micro-motion trajectory
    fig.add_trace(go.Scatter(
        x=steps,
        y=disp_history,
        mode="lines+markers",
        name="Achieved Motion (mm)",
        line=dict(color="#38bdf8", width=3),
        marker=dict(size=6, color="#60a5fa")
    ))

    # Dotted horizontal target line
    fig.add_hline(
        y=target_mm,
        line_dash="dash",
        line_color="#f87171",
        line_width=2,
        annotation_text=f"Target Goal ({target_mm:.2f} mm)",
        annotation_position="top right",
        annotation_font=dict(color="#f87171", size=11)
    )

    # Tolerance bands (+/- 15%)
    fig.add_hrect(
        y0=target_mm * 0.85,
        y1=target_mm * 1.15,
        fillcolor="rgba(34, 197, 94, 0.08)",
        line_width=0,
        annotation_text="±15% Callus Band",
        annotation_position="bottom right",
        annotation_font=dict(color="#4ade80", size=9)
    )

    fig.update_layout(
        title=dict(text="Fracture Site Micro-Motion (mm)", font=dict(size=13, color="#e2e8f0")),
        xaxis=dict(title="Optimization Step", color="#94a3b8", gridcolor="rgba(255,255,255,0.05)", dtick=1),
        yaxis=dict(title="Displacement (mm)", color="#94a3b8", gridcolor="rgba(255,255,255,0.05)"),
        margin=dict(l=10, r=10, b=10, t=35),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.4)",
        height=200,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10, color="#cbd5e1")),
        hovermode="x unified"
    )
    return fig


def create_porosity_tracking_fig(porosity_history: Dict[str, List[float]], target_porosity_pct: float = 50.0) -> go.Figure:
    """
    Plots 5-zone anatomical TPMS lattice porosity with a dotted horizontal design limit.
    """
    first_key = list(porosity_history.keys())[0] if porosity_history else ""
    n_steps = len(porosity_history[first_key]) if first_key else 0
    steps = list(range(1, n_steps + 1))
    fig = go.Figure()

    colors = {
        "Prox Anchor (%)": "#60a5fa",
        "Prox Transition (%)": "#818cf8",
        "Bridge Gap (%)": "#c084fc",
        "Dist Transition (%)": "#f472b6",
        "Dist Anchor (%)": "#38bdf8",
        # Backward compatibility aliases
        "Proximal (%)": "#818cf8",
        "Bridge (%)": "#c084fc",
        "Distal (%)": "#38bdf8"
    }

    for key, values in porosity_history.items():
        fig.add_trace(go.Scatter(
            x=steps,
            y=values,
            mode="lines+markers",
            name=key,
            line=dict(color=colors.get(key, "#94a3b8"), width=2),
            marker=dict(size=4)
        ))

    # Dotted target porosity line
    fig.add_hline(
        y=target_porosity_pct,
        line_dash="dot",
        line_color="#fbbf24",
        line_width=2,
        annotation_text=f"Design Limit ({target_porosity_pct:.0f}%)",
        annotation_position="top left",
        annotation_font=dict(color="#fbbf24", size=10)
    )

    fig.update_layout(
        title=dict(text="5-Zone Anatomical TPMS Porosity (%)", font=dict(size=13, color="#e2e8f0")),
        xaxis=dict(title="Optimization Step", color="#94a3b8", gridcolor="rgba(255,255,255,0.05)", dtick=1),
        yaxis=dict(title="Porosity (%)", color="#94a3b8", gridcolor="rgba(255,255,255,0.05)", range=[0, 100]),
        margin=dict(l=10, r=10, b=10, t=35),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.4)",
        height=200,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10, color="#cbd5e1")),
        hovermode="x unified"
    )
    return fig


def create_loss_tracking_fig(loss_history: List[float]) -> go.Figure:
    """
    Plots objective loss trajectory with a zero/convergence baseline.
    """
    steps = list(range(1, len(loss_history) + 1))
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=steps,
        y=loss_history,
        mode="lines+markers",
        name="Objective Loss",
        line=dict(color="#f43f5e", width=2.5),
        marker=dict(size=5, color="#fb7185")
    ))

    # Dotted target convergence baseline
    fig.add_hline(
        y=0.0,
        line_dash="dash",
        line_color="#4ade80",
        line_width=1.5,
        annotation_text="Optimal Baseline (0.0)",
        annotation_position="bottom right",
        annotation_font=dict(color="#4ade80", size=10)
    )

    fig.update_layout(
        title=dict(text="Total Objective Loss", font=dict(size=13, color="#e2e8f0")),
        xaxis=dict(title="Optimization Step", color="#94a3b8", gridcolor="rgba(255,255,255,0.05)", dtick=1),
        yaxis=dict(title="Loss Value", color="#94a3b8", gridcolor="rgba(255,255,255,0.05)"),
        margin=dict(l=10, r=10, b=10, t=35),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.4)",
        height=180,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10, color="#cbd5e1")),
        hovermode="x unified"
    )
    return fig


def create_gradient_tracking_fig(grad_history: Dict[str, List[float]]) -> go.Figure:
    """
    Plots adjoint sensitivity gradients for all design parameters with a dotted zero-equilibrium line.
    """
    first_key = list(grad_history.keys())[0] if grad_history else ""
    n_steps = len(grad_history[first_key]) if first_key else 0
    steps = list(range(1, n_steps + 1))
    fig = go.Figure()

    colors = {
        "∂L/∂τ_p_anc": "#60a5fa",
        "∂L/∂τ_p_tra": "#818cf8",
        "∂L/∂τ_bridge": "#c084fc",
        "∂L/∂τ_d_tra": "#f472b6",
        "∂L/∂τ_d_anc": "#38bdf8",
        "∂L/∂σ_blend": "#fbbf24",
        "∂L/∂t_top": "#34d399",
        "∂L/∂t_bottom": "#10b981",
        "∂L/∂s_pitch": "#f97316",
        "∂L/∂L_bridge": "#e879f9",
        "∂L/∂d_cell": "#2dd4bf",
        "∂L/∂r_fillet": "#fb7185",
        # Backward compatibility aliases
        "∂L/∂t_skin": "#34d399",
        "∂L/∂τ_prox": "#60a5fa",
        "∂L/∂τ_dist": "#38bdf8"
    }

    for key, values in grad_history.items():
        if len(values) > 0:
            fig.add_trace(go.Scatter(
                x=steps,
                y=values,
                mode="lines+markers",
                name=key,
                line=dict(color=colors.get(key, "#94a3b8"), width=2),
                marker=dict(size=4)
            ))

    # Dotted zero-gradient equilibrium line
    fig.add_hline(
        y=0.0,
        line_dash="dash",
        line_color="#f87171",
        line_width=1.5,
        annotation_text="Stationary Zero Gradient (∂L/∂θ = 0)",
        annotation_position="top right",
        annotation_font=dict(color="#f87171", size=10)
    )

    fig.update_layout(
        title=dict(text="Adjoint Sensitivity Derivatives (∂L/∂θ · All Design Parameters)", font=dict(size=13, color="#e2e8f0")),
        xaxis=dict(title="Optimization Step", color="#94a3b8", gridcolor="rgba(255,255,255,0.05)", dtick=1),
        yaxis=dict(title="Gradient ∂L/∂θ", color="#94a3b8", gridcolor="rgba(255,255,255,0.05)"),
        margin=dict(l=10, r=10, b=10, t=35),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.4)",
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10, color="#cbd5e1")),
        hovermode="x unified"
    )
    return fig
