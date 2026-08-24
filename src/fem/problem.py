import os
import numpy as np, jax
import jax.numpy as jnp
import meshio
from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh
from jax_fem.utils import save_sol

# Gmsh mesh ke physical group tags:
# 1: Proximal_Cortical, 2: Proximal_Trabecular
# 3: Distal_Cortical, 4: Distal_Trabecular
# 5: Gap_Cortical, 6: Gap_Trabecular
# 10: Plate


def evaluate_tpms_field(k, x, y, z, tpms_type="primitive"):
    # Minimal surface field evaluate karo
    tpms_lower = str(tpms_type).lower()
    if "gyroid" in tpms_lower or "g" in tpms_lower:
        # Schoen Gyroid (G): shear strength ke liye mast
        return 1.5 * (jnp.sin(k * x) * jnp.cos(k * y) + jnp.sin(k * y) * jnp.cos(k * z) + jnp.sin(k * z) * jnp.cos(k * x))
    elif "diamond" in tpms_lower or "d" in tpms_lower:
        # Schwarz Diamond (D): twisting/torsion resistance ke liye best
        return 1.8 * (jnp.cos(k * x) * jnp.cos(k * y) * jnp.cos(k * z) - jnp.sin(k * x) * jnp.sin(k * y) * jnp.sin(k * z))
    else:
        # Schwarz Primitive (P): maximum fluid permeability aur bone callus growth
        return jnp.cos(k * x) + jnp.cos(k * y) + jnp.cos(k * z)


def evaluate_sandwich_and_screw_masks(x, y, z, screw_spacing=0.015, bridge_span=0.030, fillet_radius=0.0012, t_top=0.0002, t_bot=0.0002):
    # 3-Layer Sandwich Parameterization:
    # Total implant depth is strictly fixed at 6.0 mm (y in [0.011, 0.017]).
    # Top solid plate: thickness t_top (0.15mm - 2.0mm)
    # Bottom solid plate: thickness t_bot (0.15mm - 2.0mm)
    # Middle porous TPMS core: thickness h_tpms = 6.0mm - (t_top + t_bot)
    skin_sharpness = 2500.0
    t_top = jnp.clip(t_top, 0.00015, 0.0020)
    t_bot = jnp.clip(t_bot, 0.00015, 0.0020)
    t_perimeter = jnp.minimum(t_top, t_bot)
    rf = jnp.clip(fillet_radius, 0.0004, 0.0025)
    
    # Outer skin envelope distances
    y_top_dist   = y - (0.017 - t_top)           # Top muscle-facing solid plate
    y_bot_dist   = (0.011 + t_bot) - y           # Bottom bone-contact solid plate
    z_wall_dist  = jnp.abs(z) - (0.008 - t_perimeter)  # Side perimeter walls
    x_left_dist  = (0.030 + t_perimeter) - x     # Proximal tip
    x_right_dist = x - (0.130 - t_perimeter)     # Distal tip
    
    # Top corner fillet envelope
    dy_top = jnp.maximum(y - (0.017 - rf), 0.0)
    dz_side = jnp.maximum(jnp.abs(z) - (0.008 - rf), 0.0)
    corner_fillet = jnp.sqrt(dy_top**2 + dz_side**2 + 1e-12) - rf
    s_corner = 1.0 / (1.0 + jnp.exp(skin_sharpness * corner_fillet))
    
    s_top = 1.0 / (1.0 + jnp.exp(-skin_sharpness * y_top_dist))
    s_bot = 1.0 / (1.0 + jnp.exp(-skin_sharpness * y_bot_dist))
    s_z   = 1.0 / (1.0 + jnp.exp(-skin_sharpness * z_wall_dist))
    s_x_l = 1.0 / (1.0 + jnp.exp(-skin_sharpness * x_left_dist))
    s_x_r = 1.0 / (1.0 + jnp.exp(-skin_sharpness * x_right_dist))
    
    is_skin = jnp.clip(s_top + s_bot + s_z + s_x_l + s_x_r, 0.0, 1.0) * s_corner
    
    # Dynamic 6-Hole Standard AO LCP Screw Holes (4.5 mm diameter, 2.25 mm radius)
    # Constrained within plate anchor zones
    x_mid = 0.080
    x3 = x_mid - bridge_span / 2.0
    x2 = x3 - screw_spacing
    x1 = x2 - screw_spacing
    x4 = x_mid + bridge_span / 2.0
    x5 = x4 + screw_spacing
    x6 = x5 + screw_spacing
    screw_centers_x = [x1, x2, x3, x4, x5, x6]
    
    hole_masks = []
    for sc_x in screw_centers_x:
        # Countersink flare: top 2.5mm flares from 2.25mm to 3.15mm
        flare = 0.00090 * jnp.clip((y - 0.0155) / 0.0025, 0.0, 1.0)
        hole_r = 0.00225 + flare
        r_dist = jnp.sqrt((x - sc_x)**2 + z**2 + 1e-12)
        h_val = 1.0 / (1.0 + jnp.exp(-skin_sharpness * (hole_r - r_dist)))
        hole_masks.append(h_val)
    is_hole = jnp.clip(jnp.sum(jnp.stack(hole_masks, axis=0), axis=0), 0.0, 1.0)
    
    return is_skin, is_hole


def lattice_density_jax(
    points,
    cell_size,
    tau_prox_anchor,
    tau_prox_trans,
    tau_bridge,
    tau_dist_trans,
    tau_dist_anchor,
    sigma_blend=0.015,
    tpms_type="primitive",
    t_top=0.0002,
    t_bottom=0.0002,
    screw_spacing=0.015,
    bridge_span=0.030,
    fillet_radius=0.0012
):
    # 5-zone Gaussian smooth blend field
    k = 2.0 * jnp.pi / cell_size
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    
    # Gaussian sigma spread safe bounds me rakho
    sigma = jnp.clip(sigma_blend, 0.008, 0.035)
    
    # 5 Anatomical Centers (meters me)
    w_p_anc  = jnp.exp(- ((x - 0.035) / sigma)**2)
    w_p_tra  = jnp.exp(- ((x - 0.057) / sigma)**2)
    w_bridge = jnp.exp(- ((x - 0.080) / sigma)**2)
    w_d_tra  = jnp.exp(- ((x - 0.103) / sigma)**2)
    w_d_anc  = jnp.exp(- ((x - 0.125) / sigma)**2)
    
    w_sum = w_p_anc + w_p_tra + w_bridge + w_d_tra + w_d_anc + 1e-6
    tau = (
        tau_prox_anchor * w_p_anc +
        tau_prox_trans  * w_p_tra +
        tau_bridge      * w_bridge +
        tau_dist_trans  * w_d_tra +
        tau_dist_anchor * w_d_anc
    ) / w_sum
    
    # Tau ko clip karo taaki strut kabhi gayab na ho
    tau = jnp.clip(tau, 0.10, 1.45)
    
    F = evaluate_tpms_field(k, x, y, z, tpms_type)
    field = F - tau
    sharpness = 10.0
    rho_lattice = 1.0 / (1.0 + jnp.exp(-sharpness * field))
    
    # 3-Layer Sandwich architecture: apply top and bottom solid plates & edge filleting
    is_skin, is_hole = evaluate_sandwich_and_screw_masks(
        x, y, z, screw_spacing=screw_spacing, bridge_span=bridge_span, fillet_radius=fillet_radius, t_top=t_top, t_bot=t_bottom
    )
    
    # 1. Apply solid outer skins
    rho_sandwich = (1.0 - is_skin) * rho_lattice + is_skin * 1.0
    
    # 2. Subtract countersunk screw holes
    rho_final = (1.0 - is_hole) * rho_sandwich + is_hole * 0.001
    
    return rho_final

def density_to_youngs_modulus(rho, cell_tags, E_solid=110e9, ga_exponent=1.6):
    # Standard biomechanical tissue elastic moduli (Pa me)
    E_cortical = 18e9
    E_trabecular = 1e9
    E_gap = 1e6
    
    # Gibson-Ashby homogenization for TPMS lattices:
    # Exponent is empirically validated per material.
    # Floor of 0.001 prevents zero-pivot in PETSc direct solver.
    E_plate = E_solid * (0.001 + rho**ga_exponent * (1.0 - 0.001))
    
    cell_tags_expanded = cell_tags[:, None]
    
    is_prox_cort = (cell_tags_expanded == 1)
    is_prox_trab = (cell_tags_expanded == 2)
    is_dist_cort = (cell_tags_expanded == 3)
    is_dist_trab = (cell_tags_expanded == 4)
    is_gap_cort = (cell_tags_expanded == 5)
    is_gap_trab = (cell_tags_expanded == 6)
    is_plate = (cell_tags_expanded == 10)
    
    E = (
        (is_prox_cort | is_dist_cort) * E_cortical +
        (is_prox_trab | is_dist_trab) * E_trabecular +
        (is_gap_cort | is_gap_trab) * E_gap +
        is_plate * E_plate
    )
    return E


class BiomechanicsProblem(Problem):

    def custom_init(self):
        self.fe = self.fes[0]

    def set_params(self, theta):
        theta = jnp.array(theta)
        
        # 12-parameter contract (legacy) or 14-parameter contract (with material params):
        # theta = [cell_size(m), tau_p_anc, tau_p_tra, tau_bridge, tau_d_tra, tau_d_anc,
        #          sigma_blend(m), t_top(m), t_bottom(m), screw_spacing(m), bridge_span(m), fillet_radius(m),
        #          ga_exponent, material_modulus_gpa]
        if theta.shape[0] not in (12, 14):
            raise ValueError(
                f"BiomechanicsProblem.set_params expects 12 or 14 parameters, got {theta.shape[0]}."
            )
        
        if theta.shape[0] == 14:
            ga_exponent = theta[12]
            E_solid = theta[13] * 1e9
        else:
            ga_exponent = 1.6
            E_solid = 110e9
        
        cell_size     = theta[0]
        t_p_anc       = theta[1]
        t_p_tra       = theta[2]
        t_bridge      = theta[3]
        t_d_tra       = theta[4]
        t_d_anc       = theta[5]
        sigma         = theta[6]
        t_top         = theta[7]
        t_bottom      = theta[8]
        screw_spacing = theta[9]
        bridge_span   = theta[10]
        fillet_radius = theta[11]

        points = self.fe.get_physical_quad_points()
        rho = lattice_density_jax(
            points, cell_size, t_p_anc, t_p_tra, t_bridge, t_d_tra, t_d_anc, sigma,
            t_top=t_top,
            t_bottom=t_bottom,
            screw_spacing=screw_spacing,
            bridge_span=bridge_span,
            fillet_radius=fillet_radius
        )

        E_field = density_to_youngs_modulus(rho, self.cell_tags, E_solid=E_solid, ga_exponent=ga_exponent)
        self.internal_vars = [E_field]
        
        self.rho = rho
        self.E_field = E_field
        self.is_plate = (self.cell_tags[:, None] == 10)

    def get_tensor_map(self):
        # Linear elastic Hooke's law constitutive relation
        def stress(U_grad, E):
            nu = 0.32
            mu = E / (2.0 * (1.0 + nu))
            lmbda = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
            epsilon = 0.5 * (U_grad + U_grad.T)
            sigma = lmbda * jnp.trace(epsilon) * jnp.eye(self.dim) + 2.0 * mu * epsilon
            return sigma
        return stress

    def get_surface_maps(self):
        # Distal right face (x = 0.160m) pe downward gait traction lagao
        def surface_map(u, x):
            return jnp.array([
                0.0,
                0.0,
                -1.0e6,
            ])
        return [surface_map]

    def compute_compliance(self, sol):
        return compliance_from_solution(self, sol)

def compliance_from_solution(problem, sol):
    fe = problem.fes[0]
    boundary_inds = problem.boundary_inds_list[0]
    _, nanson_scale = fe.get_face_shape_grads(boundary_inds)
    
    u_face = (
        sol[fe.cells][boundary_inds[:, 0]][:, None, :, :] *
        fe.face_shape_vals[boundary_inds[:, 1]][:, :, :, None]
    )
    u_face = jnp.sum(u_face, axis=2)
    
    subset_quad_points = problem.physical_surface_quad_points[0]
    
    traction = -jax.vmap(
        jax.vmap(
            lambda u, x: jnp.array([0.0, 0.0, -1.0e6])
        )
    )(u_face, subset_quad_points)
    
    # Boundary traction * displacement ka integral = Total Work Done (Compliance)
    compliance = jnp.sum(traction * u_face * nanson_scale[:, :, None])
    return compliance


# Boundary conditions: left end ko clamp karo fixed cantilever
def left(point):
    return jnp.logical_and(point[0] < 1e-4, point[1]**2 + point[2]**2 <= 0.013**2)

# Right end jaha traction lagegi
def right(point):
    return jnp.logical_and(point[0] > 0.160 - 1e-4, point[1]**2 + point[2]**2 <= 0.013**2)

def zero(point):
    return 0.0

dirichlet_bc_info = [
    [left, left, left],
    [0, 1, 2],
    [zero, zero, zero],
]

location_fns = [
    right
]

def build_problem(mesh_path):
    # Gmsh .msh file padhke JAX-FEM problem structure banao
    meshio_mesh = meshio.read(mesh_path)
    
    cells_list = []
    cell_tags_list = []
    
    for i, cell_block in enumerate(meshio_mesh.cells):
        if cell_block.type in ("tetra", "tetra10"):
            cells_list.append(cell_block.data)
            if "gmsh:physical" in meshio_mesh.cell_data:
                cell_tags_list.append(meshio_mesh.cell_data["gmsh:physical"][i])
            else:
                cell_tags_list.append(np.ones(len(cell_block.data), dtype=np.int32))
            
    if not cells_list:
        raise ValueError("No tetrahedral cells found in mesh.")
        
    cells = np.vstack(cells_list)
    cell_tags = np.concatenate(cell_tags_list)
    ele_type = "TET10" if cells.shape[1] == 10 else "TET4"

    mesh = Mesh(meshio_mesh.points, cells)

    problem = BiomechanicsProblem(
        mesh, vec=3, dim=3, ele_type=ele_type,
        dirichlet_bc_info=dirichlet_bc_info,
        location_fns=location_fns
    )
    
    problem.cell_tags = jnp.array(cell_tags)
    return problem


def compute_nodal_von_mises_stress(mesh_path, sol_u, youngs_modulus_gpa=110.0):
    # Har node pe smooth continuous Von Mises stress (MPa) evaluate karo
    meshio_mesh = meshio.read(mesh_path)
    points = np.asarray(meshio_mesh.points)
    
    tet_cells = []
    tet_tags = []
    for i, cb in enumerate(meshio_mesh.cells):
        if cb.type in ("tetra", "tetra10"):
            tet_cells.append(cb.data[:, :4])
            if "gmsh:physical" in meshio_mesh.cell_data:
                tet_tags.append(meshio_mesh.cell_data["gmsh:physical"][i])
            else:
                tet_tags.append(np.ones(len(cb.data), dtype=np.int32))
                
    if not tet_cells:
        return np.zeros(len(points), dtype=np.float32)
        
    cells = np.vstack(tet_cells)
    tags = np.concatenate(tet_tags)
    
    u = np.asarray(sol_u)
    if u.ndim == 1:
        u = u.reshape(-1, 3)
        
    # Element-wise elastic moduli (Pa me)
    E_dict = {
        1: 18e9, 2: 1e9, 3: 18e9, 4: 1e9, 5: 1e6, 6: 1e6,
        10: youngs_modulus_gpa * 1e9
    }
    
    nu = 0.32
    nodal_vm_sum = np.zeros(len(points), dtype=np.float64)
    nodal_count = np.zeros(len(points), dtype=np.float64)
    
    for elem_idx in range(len(cells)):
        c_nodes = cells[elem_idx]
        x_e = points[c_nodes]
        u_e = u[c_nodes]
        tag = tags[elem_idx]
        
        E = E_dict.get(int(tag), 18e9)
        mu = E / (2.0 * (1.0 + nu))
        lmbda = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
        
        # Tetrahedral deformation gradient
        D_x = np.column_stack((x_e[1] - x_e[0], x_e[2] - x_e[0], x_e[3] - x_e[0]))
        D_u = np.column_stack((u_e[1] - u_e[0], u_e[2] - u_e[0], u_e[3] - u_e[0]))
        
        try:
            inv_Dx = np.linalg.inv(D_x)
            grad_u = D_u @ inv_Dx
            eps = 0.5 * (grad_u + grad_u.T)
            tr_eps = np.trace(eps)
            sigma = lmbda * tr_eps * np.eye(3) + 2.0 * mu * eps
            s_dev = sigma - (1.0 / 3.0) * np.trace(sigma) * np.eye(3)
            # Deviatoric stress se Von Mises MPa nikalo
            vm_mpa = np.sqrt(1.5 * np.sum(s_dev * s_dev)) / 1e6
        except Exception:
            vm_mpa = 0.0
            
        nodal_vm_sum[c_nodes] += vm_mpa
        nodal_count[c_nodes] += 1.0
        
    nodal_vm = nodal_vm_sum / np.maximum(nodal_count, 1.0)
    return nodal_vm.astype(np.float32)