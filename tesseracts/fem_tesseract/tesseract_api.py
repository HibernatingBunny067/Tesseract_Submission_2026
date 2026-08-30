# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import Any, Tuple
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

def _extract_theta_array(inputs: InputSchema) -> jnp.ndarray:
    return jnp.array([
        float(np.asarray(inputs.cell_size)),
        float(np.asarray(inputs.tau_prox_anchor)),
        float(np.asarray(inputs.tau_prox_trans)),
        float(np.asarray(inputs.tau_bridge)),
        float(np.asarray(inputs.tau_dist_trans)),
        float(np.asarray(inputs.tau_dist_anchor)),
        float(np.asarray(inputs.sigma_blend)),
        float(np.asarray(getattr(inputs, "t_top", 0.0002))),
        float(np.asarray(getattr(inputs, "t_bottom", 0.0002))),
        float(np.asarray(getattr(inputs, "screw_spacing", 0.015))),
        float(np.asarray(getattr(inputs, "bridge_span", 0.030))),
        float(np.asarray(getattr(inputs, "fillet_radius", 0.0012)))
    ], dtype=jnp.float64)


def apply(inputs: InputSchema) -> OutputSchema:
    theta = _extract_theta_array(inputs)
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
    theta = _extract_theta_array(inputs)
    
    _, vjp_func = jax.vjp(solve_fem_differentiable, theta)
    
    cotangents = (
        jnp.array(float(np.asarray(cotangent_vector.get("compliance", 0.0)))),
        jnp.array(float(np.asarray(cotangent_vector.get("max_displacement", 0.0)))),
        jnp.array(float(np.asarray(cotangent_vector.get("fracture_displacement", 0.0))))
    )
    
    (grad_theta,) = vjp_func(cotangents)
    grad_np = np.asarray(grad_theta)
    
    out_grads = {}
    param_keys = [
        "cell_size", "tau_prox_anchor", "tau_prox_trans", "tau_bridge",
        "tau_dist_trans", "tau_dist_anchor", "sigma_blend", "t_top",
        "t_bottom", "screw_spacing", "bridge_span", "fillet_radius"
    ]
    for idx, key in enumerate(param_keys):
        if key in vjp_inputs:
            out_grads[key] = np.array(grad_np[idx], dtype=np.float64)
        
    return out_grads


def abstract_eval(
    abstract_inputs: InputSchema
) -> dict[str, Any]:
    return {
        "compliance": {"shape": (), "dtype": "float64"},
        "max_displacement": {"shape": (), "dtype": "float64"},
        "fracture_displacement": {"shape": (), "dtype": "float64"}
    }
