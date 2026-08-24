"""
Design tokens and CSS theme for the Tesseract BioMechanics dashboard.

All visual constants are centralized here so `app.py` stays clean.
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Design Tokens
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Colors:
    """Curated colour palette (HSL-derived)."""
    bg_primary:   str = "#0f172a"
    bg_secondary: str = "#1e293b"
    bg_sidebar:   str = "#1e1b4b"
    accent:       str = "#6366f1"
    accent_light: str = "#a78bfa"
    text_primary: str = "#f8fafc"
    text_muted:   str = "#94a3b8"
    text_body:    str = "#e2e8f0"
    success:      str = "#4ade80"
    danger:       str = "#f87171"
    warning:      str = "#fbbf24"
    download:     str = "#10b981"


@dataclass(frozen=True)
class Typography:
    """Font stack and sizing."""
    family:           str = "'Inter', sans-serif"
    google_fonts_url: str = "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap"
    section_label:    str = "0.7rem"
    badge:            str = "0.8rem"
    metric_value:     str = "1.6rem"
    metric_label:     str = "0.75rem"


@dataclass(frozen=True)
class Spacing:
    """Reusable spacing / radius values."""
    card_radius:   str = "12px"
    banner_radius: str = "16px"
    button_radius: str = "8px"
    badge_radius:  str = "20px"
    card_padding:  str = "1.25rem 1.5rem"


# ---------------------------------------------------------------------------
# Composite Theme
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Theme:
    colors:     Colors     = field(default_factory=Colors)
    typography: Typography = field(default_factory=Typography)
    spacing:    Spacing    = field(default_factory=Spacing)


# Singleton
THEME = Theme()


# ---------------------------------------------------------------------------
# CSS Builder
# ---------------------------------------------------------------------------
def build_css(t: Theme = THEME) -> str:
    """Return the full CSS string, parameterised by the theme tokens."""
    c, ty, sp = t.colors, t.typography, t.spacing
    return f"""
<link href="{ty.google_fonts_url}" rel="stylesheet">
<style>
    /* ── Global ─────────────────────────────────────── */
    .stApp {{ font-family: {ty.family}; }}

    /* ── Hero Banner ────────────────────────────────── */
    .hero-banner {{
        background: linear-gradient(135deg, {c.bg_primary} 0%, {c.bg_secondary} 50%, {c.bg_primary} 100%);
        border-radius: {sp.banner_radius};
        padding: 2.5rem 2rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(99, 102, 241, 0.2);
        text-align: center;
    }}
    .hero-banner h1 {{
        color: {c.text_primary};
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
    }}
    .hero-banner p {{
        color: {c.text_muted};
        font-size: 0.95rem;
        margin: 0;
    }}
    .hero-banner .accent {{
        background: linear-gradient(90deg, {c.accent}, #8b5cf6, {c.accent_light});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}

    /* ── Glass Cards ─────────────────────────────────── */
    .glass-card {{
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: {sp.card_radius};
        padding: {sp.card_padding};
        margin-bottom: 1rem;
    }}
    .glass-card h3 {{
        color: {c.text_body};
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 0 0 0.75rem 0;
        font-weight: 600;
    }}

    /* ── Section Labels ──────────────────────────────── */
    .section-label {{
        color: {c.text_muted};
        font-size: {ty.section_label};
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }}

    /* ── Status Badges ───────────────────────────────── */
    .status-badge {{
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.4rem 0.8rem;
        border-radius: {sp.badge_radius};
        font-size: {ty.badge};
        font-weight: 500;
    }}
    .status-optimal {{
        background: rgba(34, 197, 94, 0.15);
        color: {c.success};
        border: 1px solid rgba(34, 197, 94, 0.3);
    }}
    .status-warning {{
        background: rgba(251, 191, 36, 0.15);
        color: {c.warning};
        border: 1px solid rgba(251, 191, 36, 0.3);
    }}
    .status-danger {{
        background: rgba(239, 68, 68, 0.15);
        color: {c.danger};
        border: 1px solid rgba(239, 68, 68, 0.3);
    }}

    /* ── Metric Tiles ────────────────────────────────── */
    .metric-tile {{
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6), rgba(30, 41, 59, 0.3));
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: {sp.card_radius};
        padding: 1.25rem;
        text-align: center;
    }}
    .metric-tile .value {{
        font-size: {ty.metric_value};
        font-weight: 700;
        color: {c.text_primary};
    }}
    .metric-tile .label {{
        font-size: {ty.metric_label};
        color: {c.text_muted};
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.25rem;
    }}
    .metric-tile .delta {{
        font-size: {ty.metric_label};
        margin-top: 0.25rem;
    }}
    .delta-good {{ color: {c.success}; }}
    .delta-bad  {{ color: {c.danger};  }}

    /* ── Sidebar ─────────────────────────────────────── */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {c.bg_primary}, {c.bg_sidebar}) !important;
    }}
    [data-testid="stSidebar"] * {{
        color: {c.text_body} !important;
    }}

    /* ── Buttons ─────────────────────────────────────── */
    .stButton>button {{
        background: linear-gradient(90deg, {c.accent} 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: {sp.button_radius} !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: 600 !important;
        transition: all 0.25s ease !important;
    }}
    .stButton>button:hover {{
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.35) !important;
    }}

    /* ── Download Button ─────────────────────────────── */
    .stDownloadButton>button {{
        background: linear-gradient(90deg, #059669 0%, {c.download} 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: {sp.button_radius} !important;
    }}

    /* ── Dividers ────────────────────────────────────── */
    hr {{
        border-color: rgba(99, 102, 241, 0.15) !important;
    }}
</style>
"""
