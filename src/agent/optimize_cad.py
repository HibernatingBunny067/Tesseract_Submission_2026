import os
import numpy as np
import optax
import jax
import jax.numpy as jnp
from src.geometry.morph import apply_pygem_ffd, FFD_NX
from src.fem.problem import build_problem
from jax_fem.solver import solver

SOLVER_OPTIONS = {"spsolve_solver": {}}

def evaluate_cad_loss(bend_y_array, bend_z_array, base_mesh_path, temp_mesh_path, target_disp=0.0002):
    """
    Evaluates the compliance and micro-motion for a given PyGeM FFD morphing state.
    Rebuilds the JAX-FEM problem for the morphed mesh to compute physics.
    """
    # 1. Morph the mesh
    apply_pygem_ffd(base_mesh_path, temp_mesh_path, bend_y_array, bend_z_array)
    
    # 2. Build FEM Problem for the new morphed geometry
    problem = build_problem(temp_mesh_path)
    
    # Set default uniform parameters (just to evaluate global shape, not the lattice yet)
    # Using a solid titanium-like plate for CAD shape optimization
    theta = np.array([
        0.005, 1.45, 1.45, 1.45, 1.45, 1.45, 0.015, 0.002, 0.002, 0.015, 0.030, 0.0012, 1.6, 110.0
    ])
    problem.set_params(theta)
    
    # 3. Solve forward FEM
    sol_list = solver(problem, solver_options=SOLVER_OPTIONS)
    u = sol_list[0]
    
    # 4. Compute metrics
    compliance = problem.compute_compliance(u)
    
    points = problem.fes[0].points
    prox_mask = np.logical_and(np.abs(points[:, 0] - 0.079) < 1e-4, np.sqrt(points[:, 1]**2 + points[:, 2]**2) <= 0.012)
    dist_mask = np.logical_and(np.abs(points[:, 0] - 0.081) < 1e-4, np.sqrt(points[:, 1]**2 + points[:, 2]**2) <= 0.012)
    
    u_prox = np.sum(u * prox_mask[:, None], axis=0) / (np.sum(prox_mask) + 1e-10)
    u_dist = np.sum(u * dist_mask[:, None], axis=0) / (np.sum(dist_mask) + 1e-10)
    frac_disp = float(np.linalg.norm(u_prox - u_dist))
    compliance = float(compliance)
    
    # 5. Loss formulation (similar to main workflow)
    err_rel = (frac_disp - target_disp) / (target_disp + 1e-9)
    disp_penalty = (err_rel * 22.0)**2 * 2.0
    loss = disp_penalty + compliance * 1.0
    
    # Clear JAX compilation caches to release the JIT closure over the problem object
    jax.clear_caches()
    import gc
    gc.collect()
    return loss, frac_disp, compliance

def run_cad_shape_optimization(
    base_mesh_path, 
    morphed_mesh_path, 
    target_disp=0.0002, 
    max_steps=15
):
    """
    Initial stage optimization for CAD global shape using PyGeM + Optax Adam.
    Computes gradients using Finite Differences since PyGeM is non-differentiable by JAX.
    Now supports a highly dense {FFD_NX}x4x4 grid (8 internal slices) for high fidelity control.
    """
    NX = FFD_NX
    n_interior = NX - 2
    # PyGeM FFD parameters: [y1..y8, z1..z8]
    theta = jnp.zeros(2 * n_interior)
    
    # Setup Optax Adam (same logic as workflow)
    optimizer = optax.adam(learning_rate=0.001)
    opt_state = optimizer.init(theta)
    
    eps = 1e-4 # Finite difference step size
    
    print(f"Starting Initial CAD Shape Optimization (PyGeM) with highly dense {NX}x4x4 grid...")
    for step in range(max_steps):
        bend_y_array = np.array(theta[0:n_interior])
        bend_z_array = np.array(theta[n_interior:2*n_interior])
        
        grads = np.zeros(2 * n_interior)
        
        # Central difference for all parameters
        for i in range(n_interior):
            # Grad for y_i
            y_plus = bend_y_array.copy(); y_plus[i] += eps
            y_minus = bend_y_array.copy(); y_minus[i] -= eps
            loss_y_plus, _, _ = evaluate_cad_loss(y_plus, bend_z_array, base_mesh_path, morphed_mesh_path, target_disp)
            loss_y_minus, _, _ = evaluate_cad_loss(y_minus, bend_z_array, base_mesh_path, morphed_mesh_path, target_disp)
            grads[i] = (loss_y_plus - loss_y_minus) / (2 * eps)
            
            # Grad for z_i
            z_plus = bend_z_array.copy(); z_plus[i] += eps
            z_minus = bend_z_array.copy(); z_minus[i] -= eps
            loss_z_plus, _, _ = evaluate_cad_loss(bend_y_array, z_plus, base_mesh_path, morphed_mesh_path, target_disp)
            loss_z_minus, _, _ = evaluate_cad_loss(bend_y_array, z_minus, base_mesh_path, morphed_mesh_path, target_disp)
            grads[i + n_interior] = (loss_z_plus - loss_z_minus) / (2 * eps)
        
        # Evaluate current loss
        loss, frac_disp, compliance = evaluate_cad_loss(bend_y_array, bend_z_array, base_mesh_path, morphed_mesh_path, target_disp)
        
        grads_jnp = jnp.array(grads)
        
        # Update via Adam
        updates, opt_state = optimizer.update(grads_jnp, opt_state, params=theta)
        theta = optax.apply_updates(theta, updates)
        
        # Bound constraints for safety
        theta = jnp.clip(theta, -0.015, 0.015)
        
        # --- NEW: Detailed logging for each control point slice ---
        print(f"\n[PyGeM] --- STEP {step+1}/{max_steps} DETAILED CONTROL POINT LOGS ---")
        for idx in range(n_interior):
            print(f"[PyGeM] Slice {idx+1}/{n_interior} | "
                  f"Bend Y: {float(theta[idx])*1000:7.3f} mm (Grad: {float(grads_jnp[idx]):7.1f}) | "
                  f"Bend Z: {float(theta[n_interior + idx])*1000:7.3f} mm (Grad: {float(grads_jnp[n_interior + idx]):7.1f})")
        print(f"[PyGeM] -----------------------------------------------------\n")
        
        # Select the exact middle slice for UI representation
        mid = n_interior // 2
        yield {
            "step": step,
            "loss": loss,
            "frac_disp": frac_disp,
            "compliance": compliance,
            "bend_y": float(theta[mid]), 
            "bend_z": float(theta[n_interior + mid]),
            "grad_y": float(grads_jnp[mid]),
            "grad_z": float(grads_jnp[n_interior + mid]),
            "all_bend_y": theta[0:n_interior].tolist(),
            "all_bend_z": theta[n_interior:2*n_interior].tolist()

        }
        
