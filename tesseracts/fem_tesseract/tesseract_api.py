# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
from pydantic import BaseModel, Field

# pyrefly: ignore [missing-import]
from tesseract_core.runtime import (
    Array, Differentiable, Float64
)

from src.fem.forward import solve_fem_differentiable

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
    compliance: Differentiable[Array[(), Float64]] = Field(
        description="Structural Compliance"
    )
    max_displacement: Differentiable[Array[(), Float64]] = Field(
        description="Maximum absolute displacement in meters"
    )
    fracture_displacement: Differentiable[Array[(), Float64]] = Field(
        description="Relative displacement between fracture ends in meters"
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

    comp, max_disp, frac_disp = solve_fem_differentiable(theta)
    
    return OutputSchema(
        compliance=np.asarray(comp, dtype=np.float64),
        max_displacement=np.asarray(max_disp, dtype=np.float64),
        fracture_displacement=np.asarray(frac_disp, dtype=np.float64)
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
    
    # Trace solve_fem_differentiable and compute the VJP!
    _, vjp_func = jax.vjp(solve_fem_differentiable, theta)
    
    # Pack cotangents corresponding to the outputs
    cotangents = (
        jnp.array(cotangent_vector.get("compliance", 0.0)),
        jnp.array(cotangent_vector.get("max_displacement", 0.0)),
        jnp.array(cotangent_vector.get("fracture_displacement", 0.0))
    )
    
    # Pull back through JAX-FEM adjoint
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
    inputs: InputSchema
) -> dict[str, Any]:
    return {
        "compliance": {"shape": (), "dtype": "float64"},
        "max_displacement": {"shape": (), "dtype": "float64"},
        "fracture_displacement": {"shape": (), "dtype": "float64"}
    }
