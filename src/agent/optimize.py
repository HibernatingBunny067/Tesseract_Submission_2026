import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
import optax
import sys
import os
from typing import Optional, Dict, Any, Tuple, Generator, Callable, List, Union
import tesseract_jax as tjax
import tesseract_core as tc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


def create_loss_function(
    target_fracture_displacement: float = 0.0002,
    fem_client: Optional[Any] = None,
    geometry_client: Optional[Any] = None,
    objective: str = "Callus Stimulation & Mass Minimization",
    max_mass: float = 0.60,
    material_modulus_gpa: float = 110.0,
    tpms_ga_exponent: float = 1.6,
    yield_strength_mpa: float = 880.0
) -> Callable[[jnp.ndarray], Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]]:
    # Material elasticity scale factor (only used for legacy 12-parameter remote microservice)
    # If using local solver, material properties are natively handled via 14-parameter theta.
    mat_scale: float = jnp.clip(110.0 / material_modulus_gpa, 0.3, 30.0) if fem_client is not None else 1.0
    
    def loss_fn(theta: jnp.ndarray) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        # 11 parameter normalized unitless vector:
        # [cell_size_mm, tau_1..5, sigma_norm (10mm), t_top_mm, t_bot_mm, screw_pitch_mm, bridge_span_mm]
        cell_size = theta[0] * 1e-3   # meters
        t_p_anc   = theta[1]
        t_p_tra   = theta[2]
        t_bridge  = theta[3]
        t_d_tra   = theta[4]
        t_d_anc   = theta[5]
        sigma     = theta[6] * 0.010  # meters
        t_top     = theta[7] * 1e-3   # meters
        t_bot     = theta[8] * 1e-3   # meters
        screw_s   = theta[9] * 1e-3   # meters
        bridge_s  = theta[10] * 1e-3  # meters
        r_fillet  = theta[11] * 1e-3 if len(theta) > 11 else 0.0012
        
        # 1. Evaluate Tesseract 1: JAX-FEM Biomechanical Continuum Engine
        if fem_client is not None:
            fem_inputs = {
                'cell_size': cell_size,
                'tau_prox_anchor': t_p_anc,
                'tau_prox_trans': t_p_tra,
                'tau_bridge': t_bridge,
                'tau_dist_trans': t_d_tra,
                'tau_dist_anchor': t_d_anc,
                'sigma_blend': sigma,
                't_top': t_top,
                't_bottom': t_bot,
                'screw_spacing': screw_s,
                'bridge_span': bridge_s,
                'fillet_radius': r_fillet
            }
            fem_out = tjax.apply_tesseract(fem_client, fem_inputs)
            compliance = fem_out['compliance']
            max_disp = fem_out['max_displacement']
            raw_frac_disp = fem_out['fracture_displacement']
        else:
            from src.fem.forward import solve_fem_differentiable
            theta_phys = jnp.array([cell_size, t_p_anc, t_p_tra, t_bridge, t_d_tra, t_d_anc, sigma, t_top, t_bot, screw_s, bridge_s, r_fillet, tpms_ga_exponent, material_modulus_gpa])
            compliance, max_disp, raw_frac_disp = solve_fem_differentiable(theta_phys)

        # 2. Evaluate Tesseract 2: Geometry & Porosity Metamaterial Engine
        if geometry_client is not None:
            geom_inputs = {
                'cell_size': cell_size,
                'tau_prox_anchor': t_p_anc,
                'tau_prox_trans': t_p_tra,
                'tau_bridge': t_bridge,
                'tau_dist_trans': t_d_tra,
                'tau_dist_anchor': t_d_anc,
                'sigma_blend': sigma,
                't_top': t_top,
                't_bottom': t_bot,
                'screw_spacing': screw_s,
                'bridge_span': bridge_s,
                'fillet_radius': r_fillet
            }
            geom_out = tjax.apply_tesseract(geometry_client, geom_inputs)
            mean_porosity = geom_out['mean_porosity']
            bridge_porosity = geom_out['bridge_porosity']
            mass_fraction = geom_out['mass_fraction']
        else:
            from tesseracts.geometry_tesseract.tesseract_api import evaluate_geometry_metrics
            theta_phys = jnp.array([cell_size, t_p_anc, t_p_tra, t_bridge, t_d_tra, t_d_anc, sigma, t_top, t_bot, screw_s, bridge_s, r_fillet])
            mean_porosity, bridge_porosity, mass_fraction, _ = evaluate_geometry_metrics(theta_phys)

        # Apply material stiffness scaling
        frac_disp = raw_frac_disp * mat_scale
        eff_compliance = compliance * mat_scale
        
        # Dimensionless scale-invariant relative displacement error
        err_rel = (frac_disp - target_fracture_displacement) / (target_fracture_displacement + 1e-9)
        
        # Anatomical symmetry regularization: Proximal and Distal anchor parameters should stay balanced
        symmetry_loss = 15.0 * ((t_p_anc - t_d_anc)**2 + (t_p_tra - t_d_tra)**2)
        
        # Quadratic mass budget hard penalty — no competing soft pull toward a fixed value.
        # The optimizer will naturally minimize mass subject to the displacement constraint.
        mass_loss = 12.0 * jnp.maximum(0.0, mass_fraction - max_mass)**2
        
        # Strict Anatomical Screw Boundary & Clearance Constraints:
        x_mid = 0.080
        x3 = x_mid - bridge_s / 2.0
        x1 = x3 - 2.0 * screw_s
        x4 = x_mid + bridge_s / 2.0
        x6 = x4 + 2.0 * screw_s
        # 1. Outermost screws x1 and x6 must maintain at least 5.0 mm margin from plate ends (x=0.030, x=0.130)
        # 2. Innermost screws x3 and x4 must stay outside bridge crack zone (bridge_s >= 20.0 mm)
        # 3. Screw pitch bounds: [10.0 mm, 16.0 mm]
        margin_barrier = 80.0 * (
            jnp.maximum(0.0, 0.035 - x1)**2 +
            jnp.maximum(0.0, x6 - 0.125)**2 +
            jnp.maximum(0.0, 0.020 - bridge_s)**2 +
            jnp.maximum(0.0, bridge_s - 0.045)**2 +
            jnp.maximum(0.0, 0.010 - screw_s)**2 +
            jnp.maximum(0.0, screw_s - 0.016)**2
        )
        
        # 3-Layer Sandwich & Structural Strength Barriers (Total Depth = 6.0mm):
        # 1. Structural skin thickness barriers: Minimum outer solid skin of 0.35 mm (0.00035 m)
        top_barrier = 60.0 * (jnp.maximum(0.0, 0.35 - theta[7])**2 + jnp.maximum(0.0, theta[7] - 2.00)**2)
        bot_barrier = 60.0 * (jnp.maximum(0.0, 0.35 - theta[8])**2 + jnp.maximum(0.0, theta[8] - 2.00)**2)
        core_barrier = 50.0 * jnp.maximum(0.0, (theta[7] + theta[8]) - 4.5)**2
        cell_barrier = 50.0 * (jnp.maximum(0.0, 3.5 - theta[0])**2 + jnp.maximum(0.0, theta[0] - 7.5)**2)
        fillet_barrier = 50.0 * (jnp.maximum(0.0, 0.4 - theta[11])**2 + jnp.maximum(0.0, theta[11] - 2.5)**2) if len(theta) > 11 else 0.0
        
        # 2. Anchor density barrier: Screws need solid metal grip (tau <= 0.60) to prevent notch yielding
        anchor_barrier = 40.0 * (jnp.maximum(0.0, t_p_anc - 0.60)**2 + jnp.maximum(0.0, t_d_anc - 0.60)**2)
        sandwich_barrier = top_barrier + bot_barrier + core_barrier + cell_barrier + fillet_barrier + anchor_barrier
        
        # 3. Differentiable Von Mises Peak Stress & FoS Barrier:
        # Under 750 N gait loading (bending moment M = 750 N * 0.035 m = 26.25 N·m)
        t_skin_eff = 0.5 * (t_top + t_bot)  # in meters
        z_eff = 9.6e-8 * (0.35 + 0.65 * (t_skin_eff / 0.0010)) * (0.50 + 0.50 * mass_fraction)
        bending_moment = 26.25  # N·m
        sigma_peak_mpa = (bending_moment / z_eff) * 1e-6 * 2.2 / jnp.maximum(mass_fraction, 0.20)
        
        # Factor of Safety against material yield strength (guarantee FoS >= 1.20)
        fos_est = yield_strength_mpa / (sigma_peak_mpa + 1e-6)
        target_fos = 1.75
        fos_penalty = 75.0 * jnp.maximum(0.0, target_fos - fos_est)**2
        
        # Objective-specific loss weighting:
        if "rigid" in objective.lower() or "strength" in objective.lower():
            # Rigid fixation: penalize excessive compliance strongly, allow stiff settling
            disp_penalty = jnp.where(
                err_rel > 0.0,
                (err_rel * 22.0)**2 * 3.0,
                (err_rel * 22.0)**2 * 1.5
            )
            stiffness_term = eff_compliance * 1.0
            total_loss = disp_penalty + symmetry_loss + mass_loss * 0.5 + stiffness_term + margin_barrier + sandwich_barrier + fos_penalty
            
        elif "osteoporotic" in objective.lower() or "porosity" in objective.lower():
            # Osteoporotic fixation: penalize under-compliance (too stiff / stress shielding) strongly
            disp_penalty = jnp.where(
                err_rel < 0.0,
                (err_rel * 22.0)**2 * 3.0,
                (err_rel * 22.0)**2 * 1.5
            )
            total_loss = disp_penalty + symmetry_loss + mass_loss * 1.5 + margin_barrier + sandwich_barrier + fos_penalty
            
        else:
            # Standard Callus Stimulation: strong symmetric target tracking
            disp_penalty = (err_rel * 22.0)**2 * 2.0
            total_loss = disp_penalty + symmetry_loss + mass_loss + margin_barrier + sandwich_barrier + fos_penalty
            
        return total_loss, (frac_disp, eff_compliance, mean_porosity, mass_fraction)

    return loss_fn


def run_optimization(
    target_fracture_displacement: float = 0.0002,
    patience: int = 6,
    max_steps: int = 100,
    fem_client: Optional[Any] = None,
    geometry_client: Optional[Any] = None,
    tesseract_client: Optional[Any] = None,
    objective: str = "Callus Stimulation & Mass Minimization",
    max_mass: float = 0.60,
    material_modulus_gpa: float = 110.0,
    tpms_ga_exponent: float = 1.6,
    yield_strength_mpa: float = 880.0,
    init_cell_size: float = 0.005,
    init_t_top: float = 0.0004,
    init_t_bot: float = 0.0004,
    init_skin_thickness: Optional[float] = None,
    init_screw_spacing: float = 0.0145,
    init_bridge_span: float = 0.030,
    init_fillet_radius: float = 0.0012,
    init_tau_bridge: Optional[float] = None,
    init_tau_anchors: Optional[float] = None,
    init_tau_transitions: Optional[float] = None,
    adam_steps: int = 25
) -> Generator[Dict[str, Any], None, Tuple[List[float], Dict[str, Any]]]:
    if init_skin_thickness is not None:
        init_t_top = init_skin_thickness
        init_t_bot = init_skin_thickness
        
    if fem_client is None and tesseract_client is not None:
        fem_client = tesseract_client

    loss_fn = create_loss_function(
        target_fracture_displacement=target_fracture_displacement,
        fem_client=fem_client,
        geometry_client=geometry_client,
        objective=objective,
        max_mass=max_mass,
        material_modulus_gpa=material_modulus_gpa,
        tpms_ga_exponent=tpms_ga_exponent,
        yield_strength_mpa=yield_strength_mpa
    )
    
    value_and_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    
    if "rigid" in objective.lower():
        default_anc = 0.20
        default_tra = 0.25
        default_bri = 0.30
    elif "osteoporotic" in objective.lower():
        default_anc = 0.50
        default_tra = 0.65
        default_bri = 0.80
    else:
        default_anc = 0.35
        default_tra = 0.45
        default_bri = 0.55
        
    init_anc = float(init_tau_anchors) if init_tau_anchors is not None else default_anc
    init_tra = float(init_tau_transitions) if init_tau_transitions is not None else default_tra
    init_bri = float(init_tau_bridge) if init_tau_bridge is not None else default_bri
        
    # Scale-balanced 12-parameter vector:
    theta = jnp.array([
        init_cell_size * 1000.0,      # cell_size (mm)
        init_anc, init_tra, init_bri, init_tra, init_anc,
        1.5,                          # sigma_norm (15mm)
        init_t_top * 1000.0,          # t_top (mm)
        init_t_bot * 1000.0,          # t_bot (mm)
        init_screw_spacing * 1000.0,  # screw_pitch (mm) - CONSTANT
        init_bridge_span * 1000.0,    # bridge_span (mm)
        init_fillet_radius * 1000.0   # fillet_radius (mm)
    ])
    
    # High-Convergence Warmup-Stable-Decay (WSD) Scheme:
    # 1. Warmup (steps 0-3): Rapid ramp-up from 0.03 -> 0.09 to avoid gradient shock on step 0
    # 2. Stable High-Velocity Plateau (steps 3 to ~75% max_steps): Constant 0.09 LR gives sustained
    #    gradient descent power to reach target displacement without premature slowing.
    # 3. Precision Anneal (final 25% of steps): Cosine decay down to 20% of peak for sub-micron settling.
    warmup_steps = 3
    decay_start = max(warmup_steps + 5, int(max_steps * 0.75))
    decay_steps = max(max_steps - decay_start, 5)

    wsd_schedule = optax.join_schedules(
        schedules=[
            optax.warmup_constant_schedule(init_value=0.03, peak_value=0.09, warmup_steps=warmup_steps),
            optax.cosine_decay_schedule(init_value=0.09, decay_steps=decay_steps, alpha=0.20)
        ],
        boundaries=[decay_start]
    )

    # Fast-Adapting Engineering Adam Configuration:
    # - b1 = 0.85: Lower momentum inertia prevents overshoot when reversing directions across boundary constraints.
    # - b2 = 0.98: Fast variance adaptation (effective ~50 step horizon instead of 1000 steps with 0.999),
    #   allowing immediate per-parameter second-moment scaling in short 30-50 step runs.
    # - clip_by_global_norm(0.12): Allows decisive step sizes in normalized parameter space.
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.12),
        optax.adam(learning_rate=wsd_schedule, b1=0.85, b2=0.98, eps=1e-7)
    )
    opt_state = optimizer.init(theta)
    
    best_loss = float('inf')
    prev_loss = float('inf')
    patience_counter = 0
    current_phase = "Warmup"
    
    for step in range(max_steps):
        (loss, aux), grads = value_and_grad_fn(theta)
        frac_disp, compliance, mean_porosity, mass_fraction = aux
        
        # All 12 parameters (including screw pitch, bridge span, fillet radius, and skin thickness)
        # are actively optimized with exact adjoint gradients
        masked_grads = grads
        
        # ==========================================
        # Per-Parameter Gradient Scaling
        # ==========================================
        # The parameter vector spans 2 orders of magnitude in physical range:
        #   bridge_span: [18,45]mm (range 27) vs tau: [0.1,1.45] (range 1.35)
        # clip_by_global_norm(0.08) would otherwise allocate almost all budget to
        # bridge_span, starving tau and fillet gradients.
        # Solution: scale each gradient by its parameter range → Adam sees normalized
        # [0,1]-equivalent sensitivities → divide update back to physical space.
        theta_range = jnp.array([
            4.0,   # cell_size: [3.5, 7.5] mm
            1.35,  # tau_p_anc: [0.10, 1.45]
            1.35,  # tau_p_tra
            1.35,  # tau_bridge
            1.35,  # tau_d_tra
            1.35,  # tau_d_anc
            1.8,   # sigma_norm: [1.0, 2.8]
            1.85,  # t_top: [0.15, 2.00] mm
            1.85,  # t_bot: [0.15, 2.00] mm
            6.0,   # screw_pitch: [10, 16] mm
            27.0,  # bridge_span: [18, 45] mm
            2.1,   # fillet_radius: [0.4, 2.5] mm
        ])
        # ∂L/∂θ_norm = ∂L/∂θ_phys × range  (chain rule for normalization)
        scaled_grads = masked_grads * theta_range
        # grad_norm in normalized space — comparable across all parameters
        grad_norm = float(jnp.linalg.norm(scaled_grads[1:]))
        
        t_top_val = float(theta[7])
        t_bot_val = float(theta[8])
        h_tpms_val = float(max(6.0 - t_top_val - t_bot_val, 1.0))
        
        if step < warmup_steps:
            current_phase = "Warmup"
        elif step < decay_start:
            current_phase = "Adam (Plateau)"
        else:
            current_phase = "Precision Anneal"

        # Yield rich real-time state for UI telemetry
        yield {
            "step": step,
            "phase": current_phase,
            "loss": float(loss),
            "frac_disp": float(frac_disp),
            "compliance": float(compliance),
            "mean_porosity": float(mean_porosity),
            "mass_fraction": float(mass_fraction),
            "cell_size_mm": float(theta[0]),
            "tau_p_anc": float(theta[1]),
            "tau_p_tra": float(theta[2]),
            "tau_bridge": float(theta[3]),
            "tau_d_tra": float(theta[4]),
            "tau_d_anc": float(theta[5]),
            "sigma_blend": float(theta[6] * 0.010),
            "t_top_mm": t_top_val,
            "t_bottom_mm": t_bot_val,
            "h_tpms_mm": h_tpms_val,
            "skin_thickness_mm": float(0.5 * (t_top_val + t_bot_val)),
            "screw_spacing_mm": float(theta[9]),
            "bridge_span_mm": float(theta[10]),
            "fillet_radius_mm": float(theta[11]),
            # Parameter gradients for UI tracking (in normalized space for comparable magnitudes)
            "grad_cell_size": float(scaled_grads[0]),
            "grad_p_anc": float(scaled_grads[1]),
            "grad_p_tra": float(scaled_grads[2]),
            "grad_bridge": float(scaled_grads[3]),
            "grad_d_tra": float(scaled_grads[4]),
            "grad_d_anc": float(scaled_grads[5]),
            "grad_sigma": float(scaled_grads[6]),
            "grad_t_top": float(scaled_grads[7]),
            "grad_t_bot": float(scaled_grads[8]),
            "grad_skin": float(0.5 * (scaled_grads[7] + scaled_grads[8])),
            "grad_pitch": float(scaled_grads[9]),
            "grad_bridge_span": float(scaled_grads[10]),
            "grad_fillet": float(scaled_grads[11]),
            # Telemetry aliases
            "tau_prox": float(theta[1]),
            "tau_dist": float(theta[5]),
            "grad_prox": float(scaled_grads[1]),
            "grad_dist": float(scaled_grads[5]),
            "grad_norm": grad_norm
        }
        
        # ==========================================
        # Parameter Update: Normalized-Space Adam Descent
        # ==========================================
        # Adam operates on normalized gradients; descale updates back to physical space.
        updates_norm, opt_state = optimizer.update(scaled_grads, opt_state, params=theta)
        updates_phys = updates_norm / theta_range
        theta = optax.apply_updates(theta, updates_phys)
            
        # ==========================================
        # Physical & Clinical Box Constraints
        # ==========================================
        # Unit cell size: [3.5mm, 7.5mm]
        theta = theta.at[0].set(jnp.clip(theta[0], 3.5, 7.5))
        # Density bounds: [0.10, 1.45]
        theta = theta.at[1:6].set(jnp.clip(theta[1:6], 0.10, 1.45))
        # Blend spread: [1.0, 2.8] (10mm to 28mm)
        theta = theta.at[6].set(jnp.clip(theta[6], 1.0, 2.8))
        # Top solid plate thickness: [0.15mm, 2.00mm]
        theta = theta.at[7].set(jnp.clip(theta[7], 0.15, 2.00))
        # Bottom solid plate thickness: [0.15mm, 2.00mm]
        theta = theta.at[8].set(jnp.clip(theta[8], 0.15, 2.00))
        # Screw pitch bounds: [10.0mm, 16.0mm] with outer periphery clearance enforcement
        max_pitch_mm = jnp.maximum(10.0, (45.0 - theta[10] / 2.0) / 2.0)
        theta = theta.at[9].set(jnp.clip(theta[9], 10.0, jnp.minimum(16.0, max_pitch_mm)))
        # Bridge working span bounds: [18.0mm, 45.0mm]
        theta = theta.at[10].set(jnp.clip(theta[10], 18.0, 45.0))
        # Fillet radius bounds: [0.4mm, 2.5mm]
        theta = theta.at[11].set(jnp.clip(theta[11], 0.4, 2.5))
              
        # Early stopping — requires min 15 steps (exploring the plateau) before declaring convergence.
        loss_val = float(loss)
        if loss_val < best_loss - 1e-4:
            best_loss = loss_val
            patience_counter = 0
        else:
            patience_counter += 1
            
        prev_loss = loss_val
            
        if patience_counter >= patience and step >= 15:
            break


if __name__ == "__main__":
    for step in run_optimization(target_fracture_displacement=0.0002, max_steps=3):
        print("Step:", step)
