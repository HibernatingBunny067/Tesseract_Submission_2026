# JAX-FEM Forward and Adjoint Biomechanical Solver Setup
import os
import sys
from unittest.mock import MagicMock
if "petsc4py" not in sys.modules:
    sys.modules["petsc4py"] = MagicMock()
    sys.modules["petsc4py.PETSc"] = MagicMock()

from typing import Tuple, Union, List, Any, Optional
import numpy as np
import jax.numpy as jnp
from jax_fem.solver import solver, ad_wrapper
from src.fem.problem import build_problem, BiomechanicsProblem

# Accelerated High-Performance Sparse Solver:
# Native JAX BiCGSTAB with Jacobi preconditioning provides robust convergence
# and native differentiability across CPU/GPU devices without external C dependencies.
SOLVER_OPTIONS = {
    "jax_solver": {"precond": True},
    "tol": 1e-3,
    "rel_tol": 1e-3
}

ADJOINT_SOLVER_OPTIONS = {
    "jax_solver": {"precond": True},
    "tol": 1e-3,
    "rel_tol": 1e-3
}

mesh_path: str = os.path.join(os.path.dirname(__file__), "data", "model.msh")
if os.path.exists(mesh_path):
    problem: Optional[BiomechanicsProblem] = build_problem(mesh_path)
    # Differentiable forward and adjoint solver wrapper
    fwd_pred = ad_wrapper(
        problem,
        solver_options=SOLVER_OPTIONS,
        adjoint_solver_options=ADJOINT_SOLVER_OPTIONS
    )
else:
    problem = None
    fwd_pred = None


def solve_fem(theta: Union[List[float], jnp.ndarray, np.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Executes a forward non-linear FEM simulation to evaluate the displacement and compliance field.
    """
    theta_arr: jnp.ndarray = jnp.asarray(theta)
    problem.set_params(theta_arr)

    sol_list = solver(
        problem,
        solver_options=SOLVER_OPTIONS,
    )

    points = problem.fes[0].points
    u: jnp.ndarray = sol_list[0]

    compliance: jnp.ndarray = problem.compute_compliance(u)
    max_displacement: jnp.ndarray = jnp.max(jnp.abs(u))
    
    # Interfragmentary micro-motion measured across the 2.0 mm fracture gap (x=79mm to x=81mm)
    prox_mask = jnp.logical_and(jnp.abs(points[:, 0] - 0.079) < 1e-4, jnp.sqrt(points[:, 1]**2 + points[:, 2]**2) <= 0.012)
    dist_mask = jnp.logical_and(jnp.abs(points[:, 0] - 0.081) < 1e-4, jnp.sqrt(points[:, 1]**2 + points[:, 2]**2) <= 0.012)
    
    u_prox = jnp.sum(u * prox_mask[:, None], axis=0) / (jnp.sum(prox_mask) + 1e-10)
    u_dist = jnp.sum(u * dist_mask[:, None], axis=0) / (jnp.sum(dist_mask) + 1e-10)
    
    fracture_displacement: jnp.ndarray = jnp.linalg.norm(u_prox - u_dist)

    return (
        u,
        compliance,
        max_displacement,
        fracture_displacement
    )


def solve_fem_differentiable(theta: Union[List[float], jnp.ndarray, np.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Differentiable forward solve invoked within jax.grad() for reverse-mode adjoint VJP sensitivity.
    """
    theta_arr: jnp.ndarray = jnp.asarray(theta)
    sol_list = fwd_pred(theta_arr)

    u: jnp.ndarray = sol_list[0]
    points = problem.fes[0].points
    
    compliance: jnp.ndarray = problem.compute_compliance(u)
    max_displacement: jnp.ndarray = jnp.max(jnp.abs(u))
    
    prox_mask = jnp.logical_and(jnp.abs(points[:, 0] - 0.079) < 1e-4, jnp.sqrt(points[:, 1]**2 + points[:, 2]**2) <= 0.012)
    dist_mask = jnp.logical_and(jnp.abs(points[:, 0] - 0.081) < 1e-4, jnp.sqrt(points[:, 1]**2 + points[:, 2]**2) <= 0.012)
    
    u_prox = jnp.sum(u * prox_mask[:, None], axis=0) / (jnp.sum(prox_mask) + 1e-10)
    u_dist = jnp.sum(u * dist_mask[:, None], axis=0) / (jnp.sum(dist_mask) + 1e-10)
    
    fracture_displacement: jnp.ndarray = jnp.linalg.norm(u_prox - u_dist)

    return compliance, max_displacement, fracture_displacement