# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Tuple, Union, List
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
from pydantic import BaseModel, Field

# pyrefly: ignore [missing-import]
from tesseract_core.runtime import (
    Array, Differentiable, Float64
)
import src.fem.petsc_compat

from src.fem.problem import evaluate_sandwich_and_screw_masks, evaluate_tpms_field


#
# Differentiable Metamaterial Geometry & Porosity Synthesizer
#

def evaluate_geometry_metrics(
    theta: Union[jnp.ndarray, np.ndarray, List[float]]
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # theta: [cell_size, tau_p_anc, tau_p_tra, tau_bridge, tau_d_tra, tau_d_anc, sigma_blend, t_top, t_bottom, screw_spacing, bridge_span, fillet_radius]
    cell_size = theta[0]
    t_p_anc   = theta[1]
    t_p_tra   = theta[2]
    t_bridge  = theta[3]
    t_d_tra   = theta[4]
    t_d_anc   = theta[5]
    sigma     = jnp.clip(theta[6], 0.008, 0.035)
    t_top     = theta[7] if len(theta) > 7 else 0.0002
    t_bot     = theta[8] if len(theta) > 8 else 0.0002
    screw_s   = theta[9] if len(theta) > 9 else 0.015
    bridge_s  = theta[10] if len(theta) > 10 else 0.030
    fillet_r  = theta[11] if len(theta) > 11 else 0.0012
    
    # 3D sampling lattice grid over plate envelope
    x = jnp.linspace(0.030, 0.130, 25)
    y = jnp.linspace(0.011, 0.017, 9)
    z = jnp.linspace(-0.008, 0.008, 9)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
    
    # 5-zone Gaussian continuous field
    w_p_anc  = jnp.exp(- ((X - 0.035) / sigma)**2)
    w_p_tra  = jnp.exp(- ((X - 0.057) / sigma)**2)
    w_bridge = jnp.exp(- ((X - 0.080) / sigma)**2)
    w_d_tra  = jnp.exp(- ((X - 0.103) / sigma)**2)
    w_d_anc  = jnp.exp(- ((X - 0.125) / sigma)**2)
    w_sum    = w_p_anc + w_p_tra + w_bridge + w_d_tra + w_d_anc + 1e-6
    
    tau_field = (
        t_p_anc  * w_p_anc +
        t_p_tra  * w_p_tra +
        t_bridge * w_bridge +
        t_d_tra  * w_d_tra +
        t_d_anc  * w_d_anc
    ) / w_sum
    tau_field = jnp.clip(tau_field, 0.10, 1.45)
    
    k = 2.0 * jnp.pi / cell_size
    F = evaluate_tpms_field(k, X, Y, Z, "primitive")
    field = F - tau_field
    rho_lattice = 1.0 / (1.0 + jnp.exp(-10.0 * field))
    
    is_skin, is_hole = evaluate_sandwich_and_screw_masks(X, Y, Z, screw_spacing=screw_s, bridge_span=bridge_s, fillet_radius=fillet_r, t_top=t_top, t_bot=t_bot)
    rho_sandwich = (1.0 - is_skin) * rho_lattice + is_skin * 1.0
    rho_final = (1.0 - is_hole) * rho_sandwich + is_hole * 0.001
    
    # Metrics
    mean_density = jnp.mean(rho_final)
    mean_porosity = 1.0 - mean_density
    
    # Bridge porosity (x in [0.070, 0.090])
    bridge_mask = (X >= 0.070) & (X <= 0.090)
    bridge_porosity = 1.0 - (jnp.sum(rho_final * bridge_mask) / (jnp.sum(bridge_mask) + 1e-6))
    
    # Mass fraction relative to solid block
    mass_fraction = mean_density
    
    # Effective compliance scaling factor (~ integral of rho^-2)
    effective_compliance_factor = jnp.mean(1.0 / (jnp.clip(rho_final, 0.05, 1.0)**2))
    
    return mean_porosity, bridge_porosity, mass_fraction, effective_compliance_factor


#
# Schemas
#

class InputSchema(BaseModel):
    cell_size: Differentiable[Array[(), Float64]] = Field(
        default=0.005,
        description="Metamaterial Lattice unit cell size in meters"
    )
    tau_prox_anchor: Differentiable[Array[(), Float64]] = Field(
        default=0.45,
        description="Lattice density threshold (Zone 1: Far Proximal Screw Anchor)"
    )
    tau_prox_trans: Differentiable[Array[(), Float64]] = Field(
        default=0.45,
        description="Lattice density threshold (Zone 2: Proximal Stress Transition)"
    )
    tau_bridge: Differentiable[Array[(), Float64]] = Field(
        default=0.45,
        description="Lattice density threshold (Zone 3: Fracture Gap Center)"
    )
    tau_dist_trans: Differentiable[Array[(), Float64]] = Field(
        default=0.45,
        description="Lattice density threshold (Zone 4: Distal Stress Transition)"
    )
    tau_dist_anchor: Differentiable[Array[(), Float64]] = Field(
        default=0.45,
        description="Lattice density threshold (Zone 5: Far Distal Screw Anchor)"
    )
    sigma_blend: Differentiable[Array[(), Float64]] = Field(
        default=0.015,
        description="Gaussian transition blend spread in meters"
    )
    t_top: Differentiable[Array[(), Float64]] = Field(
        default=0.0002,
        description="Top solid titanium plate thickness in meters (muscle-facing)"
    )
    t_bottom: Differentiable[Array[(), Float64]] = Field(
        default=0.0002,
        description="Bottom solid titanium plate thickness in meters (bone-contact)"
    )
    screw_spacing: Differentiable[Array[(), Float64]] = Field(
        default=0.015,
        description="Center-to-center pitch between consecutive cortical fixation screws in meters"
    )
    bridge_span: Differentiable[Array[(), Float64]] = Field(
        default=0.030,
        description="Central working bridge span across fracture gap in meters"
    )
    fillet_radius: Differentiable[Array[(), Float64]] = Field(
        default=0.0012,
        description="Top corner edge fillet radius in meters"
    )


class OutputSchema(BaseModel):
    mean_porosity: Differentiable[Array[(), Float64]] = Field(
        description="Volume-averaged porous void fraction (0.0 to 1.0)"
    )
    bridge_porosity: Differentiable[Array[(), Float64]] = Field(
        description="Porous void fraction in central fracture bridge zone (0.0 to 1.0)"
    )
    mass_fraction: Differentiable[Array[(), Float64]] = Field(
        description="Mass fraction relative to solid titanium implant (0.0 to 1.0)"
    )
    effective_compliance_factor: Differentiable[Array[(), Float64]] = Field(
        description="Dimensionless structural compliance factor"
    )


#
# Endpoints
#

def apply(inputs: InputSchema) -> OutputSchema:
    t_top_val = float(np.asarray(getattr(inputs, "t_top", 0.0002)))
    t_bot_val = float(np.asarray(getattr(inputs, "t_bottom", 0.0002)))
    bridge_s = float(np.asarray(getattr(inputs, "bridge_span", 0.030)))
    fillet_r = float(np.asarray(getattr(inputs, "fillet_radius", 0.0012)))
    
    theta = jnp.array([
        float(np.asarray(inputs.cell_size)),
        float(np.asarray(inputs.tau_prox_anchor)),
        float(np.asarray(inputs.tau_prox_trans)),
        float(np.asarray(inputs.tau_bridge)),
        float(np.asarray(inputs.tau_dist_trans)),
        float(np.asarray(inputs.tau_dist_anchor)),
        float(np.asarray(inputs.sigma_blend)),
        t_top_val,
        t_bot_val,
        float(np.asarray(inputs.screw_spacing)),
        bridge_s,
        fillet_r
    ])

    m_por, b_por, m_frac, eff_comp = evaluate_geometry_metrics(theta)
    
    return OutputSchema(
        mean_porosity=np.asarray(m_por, dtype=np.float64),
        bridge_porosity=np.asarray(b_por, dtype=np.float64),
        mass_fraction=np.asarray(m_frac, dtype=np.float64),
        effective_compliance_factor=np.asarray(eff_comp, dtype=np.float64)
    )


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, np.typing.ArrayLike]
) -> dict[str, np.typing.ArrayLike]:
    
    t_top_val = float(np.asarray(getattr(inputs, "t_top", 0.0002)))
    t_bot_val = float(np.asarray(getattr(inputs, "t_bottom", 0.0002)))
    bridge_s = float(np.asarray(getattr(inputs, "bridge_span", 0.030)))
    fillet_r = float(np.asarray(getattr(inputs, "fillet_radius", 0.0012)))
    
    theta = jnp.array([
        float(np.asarray(inputs.cell_size)),
        float(np.asarray(inputs.tau_prox_anchor)),
        float(np.asarray(inputs.tau_prox_trans)),
        float(np.asarray(inputs.tau_bridge)),
        float(np.asarray(inputs.tau_dist_trans)),
        float(np.asarray(inputs.tau_dist_anchor)),
        float(np.asarray(inputs.sigma_blend)),
        t_top_val,
        t_bot_val,
        float(np.asarray(inputs.screw_spacing)),
        bridge_s,
        fillet_r
    ])
    
    _, vjp_func = jax.vjp(evaluate_geometry_metrics, theta)
    
    cotangents = (
        jnp.array(cotangent_vector.get("mean_porosity", 0.0)),
        jnp.array(cotangent_vector.get("bridge_porosity", 0.0)),
        jnp.array(cotangent_vector.get("mass_fraction", 0.0)),
        jnp.array(cotangent_vector.get("effective_compliance_factor", 0.0))
    )
    
    (grad_theta,) = vjp_func(cotangents)
    grad_np = np.asarray(grad_theta)
    
    out_grads = {}
    if "cell_size" in vjp_inputs:
        out_grads["cell_size"] = np.array(grad_np[0], dtype=np.float64)
    if "tau_prox_anchor" in vjp_inputs:
        out_grads["tau_prox_anchor"] = np.array(grad_np[1], dtype=np.float64)
    if "tau_prox_trans" in vjp_inputs:
        out_grads["tau_prox_trans"] = np.array(grad_np[2], dtype=np.float64)
    if "tau_bridge" in vjp_inputs:
        out_grads["tau_bridge"] = np.array(grad_np[3], dtype=np.float64)
    if "tau_dist_trans" in vjp_inputs:
        out_grads["tau_dist_trans"] = np.array(grad_np[4], dtype=np.float64)
    if "tau_dist_anchor" in vjp_inputs:
        out_grads["tau_dist_anchor"] = np.array(grad_np[5], dtype=np.float64)
    if "sigma_blend" in vjp_inputs:
        out_grads["sigma_blend"] = np.array(grad_np[6], dtype=np.float64)
    if "t_top" in vjp_inputs:
        out_grads["t_top"] = np.array(grad_np[7] if len(grad_np) > 7 else 0.0, dtype=np.float64)
    if "t_bottom" in vjp_inputs:
        out_grads["t_bottom"] = np.array(grad_np[8] if len(grad_np) > 8 else 0.0, dtype=np.float64)
    if "screw_spacing" in vjp_inputs:
        out_grads["screw_spacing"] = np.array(grad_np[9] if len(grad_np) > 9 else 0.0, dtype=np.float64)
    if "bridge_span" in vjp_inputs:
        out_grads["bridge_span"] = np.array(grad_np[10] if len(grad_np) > 10 else 0.0, dtype=np.float64)
    if "fillet_radius" in vjp_inputs:
        out_grads["fillet_radius"] = np.array(grad_np[11] if len(grad_np) > 11 else 0.0, dtype=np.float64)
        
    return out_grads


def abstract_eval(
    abstract_inputs: InputSchema
) -> dict[str, Any]:
    return {
        "mean_porosity": {"shape": (), "dtype": "float64"},
        "bridge_porosity": {"shape": (), "dtype": "float64"},
        "mass_fraction": {"shape": (), "dtype": "float64"},
        "effective_compliance_factor": {"shape": (), "dtype": "float64"}
    }
