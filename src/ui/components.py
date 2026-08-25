"""
Reusable HTML component builders for the Streamlit dashboard.

All functions return clean, unindented HTML to avoid Markdown code-block escaping.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from src.fem.materials import Biomaterial
from src.fem.validation import ClinicalValidationReport


# ---------------------------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------------------------
def hero_banner(title: str, subtitle: str, accent_word: str = "") -> str:
    """Full-width gradient hero section."""
    if accent_word:
        title = title.replace(accent_word, f'<span class="accent">{accent_word}</span>')
    return (
        f'<div class="hero-banner">'
        f'<h1>{title}</h1>'
        f'<p>{subtitle}</p>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Section Label
# ---------------------------------------------------------------------------
def section_label(icon: str, text: str) -> str:
    """Uppercase section label above a content block."""
    return f'<p class="section-label">{icon} {text}</p>'


# ---------------------------------------------------------------------------
# Status Badge
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StatusBadge:
    """Clinical micro-motion indicator."""

    @staticmethod
    def optimal(disp_mm: float) -> str:
        return (
            f'<div class="status-badge status-optimal">'
            f'🟢 Optimal Zone ({disp_mm:.3f}mm)</div>'
        )

    @staticmethod
    def too_stiff(disp_mm: float) -> str:
        return (
            f'<div class="status-badge status-danger">'
            f'🔴 Too Stiff ({disp_mm:.3f}mm) — Risk of Non-Union</div>'
        )

    @staticmethod
    def unstable(disp_mm: float) -> str:
        return (
            f'<div class="status-badge status-danger">'
            f'🔴 Unstable ({disp_mm:.3f}mm) — Risk of Failure</div>'
        )

    @staticmethod
    def for_displacement(disp_mm: float, target_mm: float) -> str:
        if disp_mm < target_mm * 0.5:
            return StatusBadge.too_stiff(disp_mm)
        elif disp_mm > target_mm * 1.5:
            return StatusBadge.unstable(disp_mm)
        return StatusBadge.optimal(disp_mm)


# ---------------------------------------------------------------------------
# Metric Tile Row
# ---------------------------------------------------------------------------
def metric_tiles(final_disp: float, target_disp_m: float, avg_porosity: float) -> str:
    """Three-column summary row shown after optimization completes."""
    target_mm = target_disp_m * 1000.0
    delta_mm = final_disp - target_mm
    on_target = abs(delta_mm) < 0.05

    delta_cls = "delta-good" if on_target else "delta-bad"
    delta_txt = "✓ On target" if on_target else f"{delta_mm:+.3f} mm from target"

    return (
        f'<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.2rem; margin: 1rem 0;">'
        f'<div class="metric-tile">'
        f'<div class="value">{final_disp:.3f} mm</div>'
        f'<div class="label">Fracture Micro-Motion</div>'
        f'<div class="delta {delta_cls}">{delta_txt}</div>'
        f'</div>'
        f'<div class="metric-tile">'
        f'<div class="value">{avg_porosity:.1f}%</div>'
        f'<div class="label">Average Porosity</div>'
        f'<div class="delta delta-good">↑ vs Solid Plate</div>'
        f'</div>'
        f'<div class="metric-tile">'
        f'<div class="value">{100 - avg_porosity:.0f}%</div>'
        f'<div class="label">Relative Density</div>'
        f'<div class="delta delta-good">-{avg_porosity:.0f}% Mass Reduction</div>'
        f'</div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Biomechanics Comparison Panel
# ---------------------------------------------------------------------------
def comparison_panel(
    baseline_mass_g: float,
    optimized_mass_g: float,
    baseline_disp_mm: float,
    optimized_disp_mm: float,
    target_disp_mm: float,
    avg_porosity_pct: float,
    material_name: str
) -> str:
    """Side-by-side comparative table: Standard Solid Plate vs. Tesseract TPMS Plate."""
    mass_reduction_pct = ((baseline_mass_g - optimized_mass_g) / baseline_mass_g) * 100.0
    
    return (
        f'<div class="glass-card" style="margin-top: 1.2rem;">'
        f'<h3 style="color: #6366f1; margin-bottom: 1rem; font-size: 1.05rem;">🔬 Clinical Translation Comparison: Solid vs. Optimized TPMS</h3>'
        f'<table style="width: 100%; border-collapse: collapse; font-size: 0.88rem; color: #e2e8f0;">'
        f'<thead>'
        f'<tr style="border-bottom: 1px solid rgba(99, 102, 241, 0.3); text-align: left;">'
        f'<th style="padding: 0.7rem; color: #94a3b8;">Parameter</th>'
        f'<th style="padding: 0.7rem; color: #f87171;">Standard Solid Plate</th>'
        f'<th style="padding: 0.7rem; color: #4ade80;">Tesseract TPMS Plate ({material_name})</th>'
        f'<th style="padding: 0.7rem; color: #a78bfa;">Clinical Impact</th>'
        f'</tr>'
        f'</thead>'
        f'<tbody>'
        f'<tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05);">'
        f'<td style="padding: 0.6rem; font-weight: 500;">Implant Mass</td>'
        f'<td style="padding: 0.6rem;">{baseline_mass_g:.1f} g</td>'
        f'<td style="padding: 0.6rem; font-weight: 600; color: #4ade80;">{optimized_mass_g:.1f} g (-{mass_reduction_pct:.1f}%)</td>'
        f'<td style="padding: 0.6rem; color: #94a3b8;">Reduces foreign body burden and soft tissue irritation</td>'
        f'</tr>'
        f'<tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05);">'
        f'<td style="padding: 0.6rem; font-weight: 500;">Fracture Micro-Motion</td>'
        f'<td style="padding: 0.6rem; color: #f87171;">{baseline_disp_mm:.3f} mm (Too Stiff)</td>'
        f'<td style="padding: 0.6rem; font-weight: 600; color: #4ade80;">{optimized_disp_mm:.3f} mm (Target: {target_disp_mm:.2f}mm)</td>'
        f'<td style="padding: 0.6rem; color: #4ade80;">Stimulates biological secondary bone healing (callus)</td>'
        f'</tr>'
        f'<tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05);">'
        f'<td style="padding: 0.6rem; font-weight: 500;">Stress Shielding Risk</td>'
        f'<td style="padding: 0.6rem; color: #f87171;">HIGH (Cortical Resorption)</td>'
        f'<td style="padding: 0.6rem; color: #4ade80;">MINIMAL (Graded Elasticity)</td>'
        f'<td style="padding: 0.6rem; color: #94a3b8;">Prevents post-removal re-fracture</td>'
        f'</tr>'
        f'<tr>'
        f'<td style="padding: 0.6rem; font-weight: 500;">Lattice Pore Interconnectivity</td>'
        f'<td style="padding: 0.6rem;">0.0% (Solid)</td>'
        f'<td style="padding: 0.6rem; font-weight: 600; color: #4ade80;">{avg_porosity_pct:.1f}% TPMS Schwarz-P</td>'
        f'<td style="padding: 0.6rem; color: #94a3b8;">Accelerates vascularized bone tissue ingrowth</td>'
        f'</tr>'
        f'</tbody>'
        f'</table>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Material Specification Card
# ---------------------------------------------------------------------------
def material_card(mat: Biomaterial) -> str:
    """Card displaying current biomaterial properties and clinical standard."""
    return (
        f'<div class="glass-card" style="margin-bottom: 0.8rem; padding: 0.9rem 1.1rem;">'
        f'<div style="display: flex; justify-content: space-between; align-items: center;">'
        f'<span style="font-weight: 600; font-size: 0.88rem; color: #f8fafc;">🏷️ {mat.name}</span>'
        f'<span style="font-size: 0.72rem; color: #a78bfa; background: rgba(99, 102, 241, 0.18); padding: 0.2rem 0.5rem; border-radius: 6px;">{mat.code}</span>'
        f'</div>'
        f'<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem; margin-top: 0.5rem; font-size: 0.75rem;">'
        f'<div><span style="color: #94a3b8;">Young\'s Modulus:</span> <b style="color: #e2e8f0;">{mat.youngs_modulus_gpa} GPa</b></div>'
        f'<div><span style="color: #94a3b8;">Density:</span> <b style="color: #e2e8f0;">{mat.density_g_cm3} g/cm³</b></div>'
        f'<div><span style="color: #94a3b8;">Yield:</span> <b style="color: #e2e8f0;">{mat.yield_strength_mpa} MPa</b></div>'
        f'</div>'
        f'<div style="font-size: 0.72rem; color: #94a3b8; margin-top: 0.4rem; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 0.3rem;">'
        f'<span style="color: #4ade80;">✓ {mat.biocompatibility}</span> — <i>{mat.clinical_note}</i>'
        f'</div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# In-Silico Clinical Validation Report Card
# ---------------------------------------------------------------------------
def validation_report_card(report: ClinicalValidationReport) -> str:
    """Renders the comprehensive ASTM / ISO validation testing battery report without indentation escaping."""
    verdict_color = "#4ade80" if "APPROVED" in report.overall_verdict else "#fbbf24" if "CONDITIONAL" in report.overall_verdict else "#f87171"
    verdict_bg = "rgba(34, 197, 94, 0.12)" if "APPROVED" in report.overall_verdict else "rgba(251, 191, 36, 0.12)"
    
    rows_html = ""
    for t in report.tests:
        status_badge = (
            f'<span style="background: rgba(34, 197, 94, 0.18); color: #4ade80; border: 1px solid rgba(34,197,94,0.3); padding: 0.25rem 0.55rem; border-radius: 6px; font-weight: 700; font-size: 0.75rem;">✓ PASS</span>'
            if t.passed else
            f'<span style="background: rgba(239, 68, 68, 0.18); color: #f87171; border: 1px solid rgba(239,68,68,0.3); padding: 0.25rem 0.55rem; border-radius: 6px; font-weight: 700; font-size: 0.75rem;">✗ FAIL</span>'
        )
        rows_html += (
            f'<tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05);">'
            f'<td style="padding: 0.7rem 0.6rem;">'
            f'<div style="font-weight: 600; color: #f8fafc; font-size: 0.88rem;">{t.name}</div>'
            f'<div style="font-size: 0.72rem; color: #94a3b8;">{t.standard}</div>'
            f'</td>'
            f'<td style="padding: 0.7rem 0.6rem; color: #e2e8f0; font-size: 0.88rem;"><b>{t.measured_value}</b></td>'
            f'<td style="padding: 0.7rem 0.6rem; color: #94a3b8; font-size: 0.82rem;">{t.target_criteria}</td>'
            f'<td style="padding: 0.7rem 0.6rem;">{status_badge}</td>'
            f'<td style="padding: 0.7rem 0.6rem; font-size: 0.80rem; color: #cbd5e1;"><i>{t.clinical_implication}</i></td>'
            f'</tr>'
        )
        
    return (
        f'<div class="glass-card" style="margin-top: 1.2rem; border-left: 4px solid {verdict_color};">'
        f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">'
        f'<div>'
        f'<h3 style="color: #f8fafc; margin: 0; font-size: 1.15rem; font-weight: 700;">📋 In-Silico Clinical Verification Report</h3>'
        f'<span style="font-size: 0.78rem; color: #94a3b8;">Solver Discretization: <b style="color: #a78bfa;">{report.fidelity_mode}</b></span>'
        f'</div>'
        f'<div style="background: {verdict_bg}; border: 1px solid {verdict_color}; color: {verdict_color}; padding: 0.45rem 0.9rem; border-radius: 8px; font-weight: 700; font-size: 0.88rem;">'
        f'{report.overall_verdict} ({report.overall_score_pct:.0f}%)'
        f'</div>'
        f'</div>'
        f'<p style="font-size: 0.85rem; color: #cbd5e1; margin-bottom: 0.9rem; line-height: 1.4;">{report.summary_text}</p>'
        f'<table style="width: 100%; border-collapse: collapse; font-size: 0.85rem; color: #e2e8f0;">'
        f'<thead>'
        f'<tr style="border-bottom: 1px solid rgba(99, 102, 241, 0.3); text-align: left; font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">'
        f'<th style="padding: 0.6rem;">Verification Test & Standard</th>'
        f'<th style="padding: 0.6rem;">Measured Value</th>'
        f'<th style="padding: 0.6rem;">Acceptance Criteria</th>'
        f'<th style="padding: 0.6rem;">Result</th>'
        f'<th style="padding: 0.6rem;">Clinical Verdict</th>'
        f'</tr>'
        f'</thead>'
        f'<tbody>'
        f'{rows_html}'
        f'</tbody>'
        f'</table>'
        f'</div>'
    )
