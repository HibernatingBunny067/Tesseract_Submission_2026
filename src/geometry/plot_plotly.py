import meshio
import pyvista as pv
import numpy as np
import plotly.graph_objects as go


def evaluate_tpms_3d(k, X, Y, Z, tpms_type="primitive"):
    # 3 type ke minimal surface equations calculate karo
    tpms_lower = str(tpms_type).lower()
    if "gyroid" in tpms_lower or "g" in tpms_lower:
        # Schoen Gyroid (G): shear strength tagda hota hai iska
        return 1.5 * (np.sin(k * X) * np.cos(k * Y) + np.sin(k * Y) * np.cos(k * Z) + np.sin(k * Z) * np.cos(k * X))
    elif "diamond" in tpms_lower or "d" in tpms_lower:
        # Schwarz Diamond (D): twisting aur torsion sambhalne ke liye best
        return 1.8 * (np.cos(k * X) * np.cos(k * Y) * np.cos(k * Z) - np.sin(k * X) * np.sin(k * Y) * np.sin(k * Z))
    else:
        # Schwarz Primitive (P): blood flow aur callus formation ke liye sabse porous
        return np.cos(k * X) + np.cos(k * Y) + np.cos(k * Z)


def evaluate_filleted_sandwich_field(
    X, Y, Z, tau_values, tpms_type="primitive", clip_axis=None, t_top=0.0002, t_bottom=0.0002,
    fillet_radius=0.0012, screw_spacing=0.015, bridge_span=0.030, skin_thickness=None
):
    if skin_thickness is not None:
        t_top = skin_thickness
        t_bottom = skin_thickness
        
    # Filleted Anatomical Plate SDF:
    # Plate dimensions: L=100mm (x: 0.030 to 0.130), H=6mm (y: 0.011 to 0.017), W=16mm (z: -0.008 to 0.008)
    rf = fillet_radius  # Parametric top corner fillet radius (m)
    t_perim = min(t_top, t_bottom)
    
    # 1. Bullet rounded end caps along X
    is_prox_tip = (X < 0.038)
    is_dist_tip = (X > 0.122)
    dist_bullet_prox = np.where(is_prox_tip, np.sqrt((X - 0.038)**2 + Z**2) - 0.008, -1.0)
    dist_bullet_dist = np.where(is_dist_tip, np.sqrt((X - 0.122)**2 + Z**2) - 0.008, -1.0)
    dist_bullet = np.maximum(dist_bullet_prox, dist_bullet_dist)
    
    # 2. Top edge rounded corner fillets
    dy_top = np.maximum(Y - (0.017 - rf), 0.0)
    dz_side = np.maximum(np.abs(Z) - (0.008 - rf), 0.0)
    corner_fillet = np.sqrt(dy_top**2 + dz_side**2) - rf
    
    # Combined smooth rounded outer plate envelope
    z_max_env = 0.008
    inside_box = (
        (X >= 0.030) & (X <= 0.130) &
        (Y >= 0.011) & (Y <= 0.017) &
        (np.abs(Z) <= z_max_env) &
        (dist_bullet <= 0.0) &
        (corner_fillet <= 0.0)
    )
    
    # 3. 6 Standard Fixation Screw Holes
    x_mid = 0.080
    x3 = x_mid - bridge_span / 2.0
    x2 = x3 - screw_spacing
    x1 = x2 - screw_spacing
    x4 = x_mid + bridge_span / 2.0
    x5 = x4 + screw_spacing
    x6 = x5 + screw_spacing
    hole_centers = [x1, x2, x3, x4, x5, x6]
    
    is_hole = np.zeros_like(inside_box, dtype=bool)
    screw_r = 0.00225  # 4.5 mm diameter cortical fixation screw (AO standard)
    
    for hc in hole_centers:
        r_dist = np.sqrt((X - hc)**2 + Z**2)
        # Countersink chamfer at plate top
        chamfer = np.maximum(0.0, (Y - 0.015) / 0.002) * 0.0008
        hole_eff_r = screw_r + chamfer
        is_hole = is_hole | (r_dist <= hole_eff_r)
        
    # 4. 5-Zone continuous TPMS lattice field
    sigma = tau_values[5] if isinstance(tau_values, (list, tuple, np.ndarray)) and len(tau_values) > 5 else 0.015
    w_p_anc  = np.exp(- ((X - 0.035) / sigma)**2)
    w_p_tra  = np.exp(- ((X - 0.057) / sigma)**2)
    w_bridge = np.exp(- ((X - 0.080) / sigma)**2)
    w_d_tra  = np.exp(- ((X - 0.103) / sigma)**2)
    w_d_anc  = np.exp(- ((X - 0.125) / sigma)**2)
    w_sum    = w_p_anc + w_p_tra + w_bridge + w_d_tra + w_d_anc + 1e-6
    
    if isinstance(tau_values, (list, tuple, np.ndarray)) and len(tau_values) >= 5:
        t_p_anc  = float(tau_values[0])
        t_p_tra  = float(tau_values[1])
        t_bridge = float(tau_values[2])
        t_d_tra  = float(tau_values[3])
        t_d_anc  = float(tau_values[4])
        sigma    = float(tau_values[5]) if len(tau_values) > 5 else 0.015
    elif isinstance(tau_values, (list, tuple, np.ndarray)) and len(tau_values) >= 3:
        t_p_anc  = float(tau_values[0])
        t_p_tra  = float(tau_values[0])
        t_bridge = float(tau_values[1])
        t_d_tra  = float(tau_values[2])
        t_d_anc  = float(tau_values[2])
        sigma    = 0.015
    elif isinstance(tau_values, (int, float)):
        t_p_anc = t_p_tra = t_bridge = t_d_tra = t_d_anc = float(tau_values)
        sigma = 0.015
    else:
        t_p_anc = t_d_anc = 0.35
        t_p_tra = t_d_tra = 0.45
        t_bridge = 0.65
        sigma = 0.015

    # 5-zone Gaussian smooth blend field across entire implant length
    w_p_anc  = np.exp(- ((X - 0.035) / sigma)**2)
    w_p_tra  = np.exp(- ((X - 0.057) / sigma)**2)
    w_bridge = np.exp(- ((X - 0.080) / sigma)**2)
    w_d_tra  = np.exp(- ((X - 0.103) / sigma)**2)
    w_d_anc  = np.exp(- ((X - 0.125) / sigma)**2)
    w_sum    = w_p_anc + w_p_tra + w_bridge + w_d_tra + w_d_anc + 1e-6
    tau_field = (
        t_p_anc  * w_p_anc +
        t_p_tra  * w_p_tra +
        t_bridge * w_bridge +
        t_d_tra  * w_d_tra +
        t_d_anc  * w_d_anc
    ) / w_sum
    
    # Minimal surface field
    cell_size = 0.005
    k = 2.0 * np.pi / cell_size
    F = evaluate_tpms_3d(k, X, Y, Z, tpms_type)
    V_tpms = F - tau_field
    
    # 5. 3-Layer Sandwich Skin Shell (Independent Top, Bottom, and Perimeter)
    is_outer_skin = (
        (Y >= (0.018 - t_top)) |
        (Y <= (0.011 + t_bottom)) |
        (np.abs(Z) >= (0.008 - t_perim)) |
        (X <= (0.030 + t_perim)) |
        (X >= (0.130 - t_perim)) |
        (dist_bullet >= -t_perim) |
        (corner_fillet >= -t_top)
    )
    
    # Final scalar field: outside -> -3.0 (void), hole -> -3.0 (void), skin -> +3.0 (solid), core -> V_tpms
    V = np.where(
        inside_box,
        np.where(is_hole, -3.0, np.where(is_outer_skin, 3.0, V_tpms)),
        -3.0
    )
    return V, inside_box, is_outer_skin, is_hole


def get_mesh_plotly_fig(mesh_path, tau_values=None, clip_axis=None, tpms_type="primitive", fillet_radius=0.0012, screw_spacing=0.015, bridge_span=0.030, t_top=0.0002, t_bottom=0.0002, skin_thickness=None):
    if skin_thickness is not None:
        t_top = skin_thickness
        t_bottom = skin_thickness
        
    # Gmsh mesh file read karo
    m = meshio.read(mesh_path)
    
    cells = np.vstack([cb.data[:, :4] for cb in m.cells if cb.type in ('tetra', 'tetra10')])
    cell_tags = np.concatenate([m.cell_data["gmsh:physical"][i] for i, cb in enumerate(m.cells) if cb.type in ('tetra', 'tetra10')])
    
    # Agar TPMS render karna hai toh purana solid plate hatao
    render_true_tpms = (tau_values is not None)
    if render_true_tpms:
        bone_mask = (cell_tags != 10)
        cells = cells[bone_mask]
        cell_tags = cell_tags[bone_mask]
    
    pv_cells = np.column_stack((np.full(len(cells), 4), cells)).ravel()
    grid = pv.UnstructuredGrid(pv_cells, np.full(len(cells), 10), m.points)
    grid.cell_data["tag"] = cell_tags
    
    # Andar ka cross section dekhne ke liye slice karo
    if clip_axis is not None:
        grid = grid.clip(normal=clip_axis, invert=False)
        
    surf = grid.extract_surface(algorithm='dataset_surface')
    surf = surf.triangulate()
    
    faces = surf.faces.reshape(-1, 4)[:, 1:]
    vertices = surf.points
    tags = surf.cell_data["tag"]
    
    # Outer hard cortical bone ka ivory color
    colors = np.zeros(len(tags), dtype=object)
    colors[tags < 10] = "ivory"
    
    # Andar ka marrow / cancellous bone
    trabecular = (tags == 2) | (tags == 4) | (tags == 6)
    colors[trabecular] = "indianred"
    
    # Broken fracture gap ka pink color
    colors[tags == 5] = "lightcoral"
    
    fig = go.Figure()
    
    # 1. Bone 3D mesh render karo (scaled to mm for clean marked axes)
    fig.add_trace(go.Mesh3d(
        x=vertices[:, 0] * 1000.0, y=vertices[:, 1] * 1000.0, z=vertices[:, 2] * 1000.0,
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        facecolor=colors, opacity=1.0, flatshading=True,
        lighting=dict(ambient=0.4, diffuse=0.8, roughness=0.2, specular=0.4, fresnel=0.2)
    ))
    
    # 2. Continuous 5-Zone TPMS Sandwich lattice surface Marching Cubes se render karo
    if render_true_tpms:
        from skimage.measure import marching_cubes
        
        nx, ny, nz = 220, 36, 44
        x_min, x_max = 0.028, 0.132
        y_min, y_max = 0.010, 0.019
        
        if clip_axis == 'z':
            z_min, z_max = 0.0, 0.009
        else:
            z_min, z_max = -0.009, 0.009
            
        x_vals = np.linspace(x_min, x_max, nx)
        y_vals = np.linspace(y_min, y_max, ny)
        z_vals = np.linspace(z_min, z_max, nz)
        X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
        
        V, _, is_skin_mask, _ = evaluate_filleted_sandwich_field(
            X, Y, Z, tau_values, tpms_type, clip_axis=clip_axis,
            t_top=t_top, t_bottom=t_bottom,
            fillet_radius=fillet_radius, screw_spacing=screw_spacing, bridge_span=bridge_span
        )
        
        try:
            verts, tpms_faces, _, _ = marching_cubes(
                V, level=0.0, 
                spacing=((x_max - x_min) / (nx - 1), (y_max - y_min) / (ny - 1), (z_max - z_min) / (nz - 1))
            )
            verts[:, 0] += x_min
            verts[:, 1] += y_min
            verts[:, 2] += z_min
            
            # Har vertex ke local region ke hisaab se color do
            # Solid outer skin ko sleek metallic slate do, internal TPMS lattice ko local gradient color do
            tpms_colors = []
            t_perim = min(t_top, t_bottom)
            for vx, vy, vz in zip(verts[:, 0], verts[:, 1], verts[:, 2]):
                is_skin_vert = (
                    (vy >= (0.017 - t_top * 1.1)) or
                    (vy <= (0.011 + t_bottom * 1.1)) or
                    (abs(vz) >= (0.008 - t_perim * 1.1)) or
                    (vx <= (0.030 + t_perim * 1.1)) or
                    (vx >= (0.130 - t_perim * 1.1))
                )
                if is_skin_vert:
                    tpms_colors.append("rgb(148, 163, 184)")
                else:
                    # Interpolate local tau for vertex x position
                    norm_x = np.clip((vx - 0.030) / 0.100, 0.0, 1.0)
                    center_dist = abs(vx - 0.080) / 0.050
                    t_val = 1.0 - np.clip(center_dist, 0.0, 1.0)
                    r = int(70 + 185 * t_val)
                    g = int(80 + 40 * (1.0 - t_val))
                    b = int(220 * (1.0 - t_val) + 50 * t_val)
                    tpms_colors.append(f"rgb({r}, {g}, {b})")
                
            fig.add_trace(go.Mesh3d(
                x=verts[:, 0] * 1000.0, y=verts[:, 1] * 1000.0, z=verts[:, 2] * 1000.0,
                i=tpms_faces[:, 0], j=tpms_faces[:, 1], k=tpms_faces[:, 2],
                vertexcolor=tpms_colors, opacity=1.0, flatshading=True,
                lighting=dict(ambient=0.5, diffuse=0.8, roughness=0.1, specular=0.5, fresnel=0.2)
            ))
        except ValueError:
            pass
    
    layout_args = dict(
        scene=dict(
            xaxis=dict(
                title=dict(text="X: Length / Shaft Axis (mm)", font=dict(color="#38bdf8", size=11)),
                tickfont=dict(color="#94a3b8", size=9),
                showgrid=True,
                gridcolor="rgba(255, 255, 255, 0.08)",
                showline=True,
                linecolor="rgba(56, 189, 248, 0.4)",
                showbackground=True,
                backgroundcolor="rgba(15, 23, 42, 0.35)",
                ticksuffix=" mm",
                visible=True
            ),
            yaxis=dict(
                title=dict(text="Y: Plate Depth (mm)", font=dict(color="#4ade80", size=11)),
                tickfont=dict(color="#94a3b8", size=9),
                showgrid=True,
                gridcolor="rgba(255, 255, 255, 0.08)",
                showline=True,
                linecolor="rgba(74, 222, 128, 0.4)",
                showbackground=True,
                backgroundcolor="rgba(15, 23, 42, 0.35)",
                ticksuffix=" mm",
                visible=True
            ),
            zaxis=dict(
                title=dict(text="Z: Width (mm)", font=dict(color="#a78bfa", size=11)),
                tickfont=dict(color="#94a3b8", size=9),
                showgrid=True,
                gridcolor="rgba(255, 255, 255, 0.08)",
                showline=True,
                linecolor="rgba(167, 139, 250, 0.4)",
                showbackground=True,
                backgroundcolor="rgba(15, 23, 42, 0.35)",
                ticksuffix=" mm",
                visible=True
            ),
            aspectmode='data'
        ),
        margin=dict(l=10, r=10, b=10, t=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    if clip_axis == 'z':
        layout_args["scene"]["camera"] = dict(
            eye=dict(x=0, y=0.5, z=-2.5),
            up=dict(x=0, y=1, z=0)
        )
        
    fig.update_layout(**layout_args)
    return fig


def get_von_mises_plotly_fig(
    mesh_path,
    nodal_vm_stress,
    clip_axis=None,
    yield_strength_mpa=880.0,
    mode="stress",
    tau_values=None,
    tpms_type="primitive",
    fillet_radius=0.0012,
    screw_spacing=0.015,
    bridge_span=0.030,
    t_top=0.0002,
    t_bottom=0.0002,
    skin_thickness=None
):
    if skin_thickness is not None:
        t_top = skin_thickness
        t_bottom = skin_thickness
        
    from scipy.spatial import cKDTree
    from skimage.measure import marching_cubes

    # Read base FEA mesh
    m = meshio.read(mesh_path)
    cells = np.vstack([cb.data[:, :4] for cb in m.cells if cb.type in ('tetra', 'tetra10')])
    cell_tags = np.concatenate([m.cell_data["gmsh:physical"][i] for i, cb in enumerate(m.cells) if cb.type in ('tetra', 'tetra10')])
    points = m.points

    render_true_tpms = (tau_values is not None)
    fig = go.Figure()

    # 1. Bone Mesh (Tags < 10)
    bone_mask = (cell_tags != 10)
    bone_cells = cells[bone_mask]
    bone_tags = cell_tags[bone_mask]

    pv_bone_cells = np.column_stack((np.full(len(bone_cells), 4), bone_cells)).ravel()
    grid_bone = pv.UnstructuredGrid(pv_bone_cells, np.full(len(bone_cells), 10), points)
    grid_bone.cell_data["tag"] = bone_tags
    
    if mode == "fos":
        bone_scalars = np.clip(130.0 / (nodal_vm_stress + 1e-4), 0.0, 10.0)
    else:
        bone_scalars = nodal_vm_stress
    grid_bone.point_data["field_data"] = bone_scalars

    if clip_axis is not None:
        grid_bone = grid_bone.clip(normal=clip_axis, invert=False)

    surf_bone = grid_bone.extract_surface(algorithm='dataset_surface').triangulate()
    bone_faces = surf_bone.faces.reshape(-1, 4)[:, 1:]
    bone_verts = surf_bone.points
    bone_values = surf_bone.point_data["field_data"]

    # Add Bone 3D Trace (scaled to mm)
    fig.add_trace(go.Mesh3d(
        x=bone_verts[:, 0] * 1000.0, y=bone_verts[:, 1] * 1000.0, z=bone_verts[:, 2] * 1000.0,
        i=bone_faces[:, 0], j=bone_faces[:, 1], k=bone_faces[:, 2],
        intensity=bone_values,
        colorscale="Greys" if mode == "stress" else "RdYlGn",
        cmin=0.0,
        cmax=float(np.percentile(bone_values, 95)) if len(bone_values) > 0 and mode == "stress" else 5.0,
        showscale=False,
        opacity=0.9,
        flatshading=True,
        lighting=dict(ambient=0.4, diffuse=0.8, roughness=0.2, specular=0.3, fresnel=0.2)
    ))

    # 2. Remeshed 3D TPMS Implant Surface with Micro-Scale Stress & FoS
    if render_true_tpms:
        nx, ny, nz = 220, 36, 44
        x_min, x_max = 0.028, 0.132
        y_min, y_max = 0.010, 0.019
        z_min, z_max = (0.0, 0.009) if clip_axis == 'z' else (-0.009, 0.009)

        x_vals = np.linspace(x_min, x_max, nx)
        y_vals = np.linspace(y_min, y_max, ny)
        z_vals = np.linspace(z_min, z_max, nz)
        X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')

        V, _, is_skin_mask, is_hole_mask = evaluate_filleted_sandwich_field(
            X, Y, Z, tau_values, tpms_type, clip_axis=clip_axis,
            t_top=t_top, t_bottom=t_bottom,
            fillet_radius=fillet_radius, screw_spacing=screw_spacing, bridge_span=bridge_span
        )

        try:
            verts, tpms_faces, _, _ = marching_cubes(
                V, level=0.0,
                spacing=((x_max - x_min) / (nx - 1), (y_max - y_min) / (ny - 1), (z_max - z_min) / (nz - 1))
            )
            verts[:, 0] += x_min
            verts[:, 1] += y_min
            verts[:, 2] += z_min

            # Build KDTree on FEA implant plate nodes for spatial stress interpolation
            plate_node_indices = np.unique(cells[cell_tags == 10].ravel())
            plate_points = points[plate_node_indices]
            plate_stresses = nodal_vm_stress[plate_node_indices]

            kdtree = cKDTree(plate_points)
            dists, nn_indices = kdtree.query(verts, k=4)
            weights = 1.0 / np.maximum(dists, 1e-6)
            weights /= np.sum(weights, axis=1, keepdims=True)
            macro_vm = np.sum(weights * plate_stresses[nn_indices], axis=1)

            # Local lattice density & Gibson-Ashby micro-scale strut stress concentration
            tpms_scalars = np.zeros(len(verts), dtype=np.float64)
            t_perim = min(t_top, t_bottom)
            for idx, (vx, vy, vz) in enumerate(verts):
                is_skin_vert = (
                    (vy >= (0.017 - t_top * 1.1)) or
                    (vy <= (0.011 + t_bottom * 1.1)) or
                    (abs(vz) >= (0.008 - t_perim * 1.1)) or
                    (vx <= (0.030 + t_perim * 1.1)) or
                    (vx >= (0.130 - t_perim * 1.1))
                )
                if is_skin_vert:
                    sigma_micro = macro_vm[idx]
                else:
                    norm_x = np.clip((vx - 0.030) / 0.100, 0.0, 1.0)
                    center_dist = abs(vx - 0.080) / 0.050
                    t_val = 1.0 - np.clip(center_dist, 0.0, 1.0)
                    local_rho = max(0.15, 1.0 - (0.05 + 0.85 * t_val))
                    sigma_micro = macro_vm[idx] / (local_rho ** 0.65)

                if mode == "fos":
                    tpms_scalars[idx] = np.clip(yield_strength_mpa / max(sigma_micro, 1e-4), 0.0, 5.0)
                else:
                    tpms_scalars[idx] = sigma_micro

            # Colorbar and colormap selection
            if mode == "fos":
                colorscale = "RdYlGn"
                cmin, cmax = 0.0, 4.0
                cb_title = "Factor of Safety (FoS)"
            else:
                colorscale = "Turbo"
                cmin = 0.0
                cmax = float(np.percentile(tpms_scalars, 98)) if len(tpms_scalars) > 0 else 250.0
                cb_title = "Von Mises (MPa)"

            fig.add_trace(go.Mesh3d(
                x=verts[:, 0] * 1000.0, y=verts[:, 1] * 1000.0, z=verts[:, 2] * 1000.0,
                i=tpms_faces[:, 0], j=tpms_faces[:, 1], k=tpms_faces[:, 2],
                intensity=tpms_scalars,
                colorscale=colorscale,
                cmin=cmin,
                cmax=cmax,
                colorbar=dict(
                    title=dict(text=cb_title, font=dict(color="#f8fafc", size=11)),
                    tickfont=dict(color="#cbd5e1", size=10),
                    thickness=14,
                    len=0.75,
                    x=1.02
                ),
                opacity=1.0,
                flatshading=True,
                lighting=dict(ambient=0.5, diffuse=0.8, roughness=0.1, specular=0.4, fresnel=0.2)
            ))
        except Exception:
            pass
    else:
        plate_mask = (cell_tags == 10)
        plate_cells = cells[plate_mask]
        pv_plate = np.column_stack((np.full(len(plate_cells), 4), plate_cells)).ravel()
        grid_plate = pv.UnstructuredGrid(pv_plate, np.full(len(plate_cells), 10), points)
        grid_plate.point_data["field_data"] = nodal_vm_stress
        if clip_axis is not None:
            grid_plate = grid_plate.clip(normal=clip_axis, invert=False)
        surf_plate = grid_plate.extract_surface(algorithm='dataset_surface').triangulate()
        fig.add_trace(go.Mesh3d(
            x=surf_plate.points[:, 0] * 1000.0, y=surf_plate.points[:, 1] * 1000.0, z=surf_plate.points[:, 2] * 1000.0,
            i=surf_plate.faces.reshape(-1, 4)[:, 1][:, 0],
            j=surf_plate.faces.reshape(-1, 4)[:, 1][:, 1],
            k=surf_plate.faces.reshape(-1, 4)[:, 1][:, 2],
            intensity=surf_plate.point_data["field_data"],
            colorscale="Turbo" if mode == "stress" else "RdYlGn",
            cmin=0.0, cmax=120.0, opacity=1.0
        ))

    layout_args = dict(
        scene=dict(
            xaxis=dict(
                title=dict(text="X: Length / Shaft Axis (mm)", font=dict(color="#38bdf8", size=11)),
                tickfont=dict(color="#94a3b8", size=9),
                showgrid=True,
                gridcolor="rgba(255, 255, 255, 0.08)",
                showline=True,
                linecolor="rgba(56, 189, 248, 0.4)",
                showbackground=True,
                backgroundcolor="rgba(15, 23, 42, 0.35)",
                ticksuffix=" mm",
                visible=True
            ),
            yaxis=dict(
                title=dict(text="Y: Plate Depth (mm)", font=dict(color="#4ade80", size=11)),
                tickfont=dict(color="#94a3b8", size=9),
                showgrid=True,
                gridcolor="rgba(255, 255, 255, 0.08)",
                showline=True,
                linecolor="rgba(74, 222, 128, 0.4)",
                showbackground=True,
                backgroundcolor="rgba(15, 23, 42, 0.35)",
                ticksuffix=" mm",
                visible=True
            ),
            zaxis=dict(
                title=dict(text="Z: Width (mm)", font=dict(color="#a78bfa", size=11)),
                tickfont=dict(color="#94a3b8", size=9),
                showgrid=True,
                gridcolor="rgba(255, 255, 255, 0.08)",
                showline=True,
                linecolor="rgba(167, 139, 250, 0.4)",
                showbackground=True,
                backgroundcolor="rgba(15, 23, 42, 0.35)",
                ticksuffix=" mm",
                visible=True
            ),
            aspectmode='data'
        ),
        margin=dict(l=10, r=10, b=10, t=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    if clip_axis == 'z':
        layout_args["scene"]["camera"] = dict(
            eye=dict(x=0, y=0.5, z=-2.5),
            up=dict(x=0, y=1, z=0)
        )

    fig.update_layout(**layout_args)
    return fig


def generate_tpms_stl_bytes(tau_values, tpms_type="primitive", fillet_radius=0.0012, screw_spacing=0.015, bridge_span=0.030, t_top=0.0002, t_bottom=0.0002, skin_thickness=None) -> bytes:
    if skin_thickness is not None:
        t_top = skin_thickness
        t_bottom = skin_thickness
        
    # 3D printer ke liye binary STL file pack karo
    import io
    import struct
    from skimage.measure import marching_cubes
    
    if len(tau_values) >= 5:
        t_p_anc  = float(tau_values[0])
        t_p_tra  = float(tau_values[1])
        t_bridge = float(tau_values[2])
        t_d_tra  = float(tau_values[3])
        t_d_anc  = float(tau_values[4])
        sigma    = float(tau_values[5]) if len(tau_values) > 5 else 0.015
    if len(tau_values) >= 3:
        t_bridge = float(tau_values[2]) if len(tau_values) >= 5 else float(tau_values[1])
    else:
        t_bridge = 0.65
        
    nx, ny, nz = 220, 36, 44
    x_min, x_max = 0.028, 0.132
    y_min, y_max = 0.010, 0.019
    z_min, z_max = -0.009, 0.009
    
    x_vals = np.linspace(x_min, x_max, nx)
    y_vals = np.linspace(y_min, y_max, ny)
    z_vals = np.linspace(z_min, z_max, nz)
    X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
    
    V, _, _, _ = evaluate_filleted_sandwich_field(
        X, Y, Z, tau_values, tpms_type,
        t_top=t_top, t_bottom=t_bottom,
        fillet_radius=fillet_radius, screw_spacing=screw_spacing, bridge_span=bridge_span
    )
    
    verts, faces, normals, _ = marching_cubes(
        V, level=0.0,
        spacing=((x_max - x_min) / (nx - 1), (y_max - y_min) / (ny - 1), (z_max - z_min) / (nz - 1))
    )
    
    verts[:, 0] += x_min
    verts[:, 1] += y_min
    verts[:, 2] += z_min
    
    # CAD slicer ke liye millimeters me scale karo
    verts_mm = (verts * 1000.0).astype(np.float32)
    triangles = verts_mm[faces]
    
    v0 = triangles[:, 0]
    v1 = triangles[:, 1]
    v2 = triangles[:, 2]
    cross = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(cross, axis=1, keepdims=True) + 1e-12
    fnormals = (cross / norm).astype(np.float32)
    
    buf = io.BytesIO()
    header_str = f"Tesseract Multi-TPMS ({tpms_type[:12]}) Differentiable Implant (ISO-5832)"
    header = header_str.encode("ascii")[:80].ljust(80, b"\0")
    buf.write(header)
    buf.write(struct.pack("<I", len(faces)))
    
    for i in range(len(faces)):
        buf.write(struct.pack(
            "<3f3f3f3fH",
            fnormals[i, 0], fnormals[i, 1], fnormals[i, 2],
            v0[i, 0], v0[i, 1], v0[i, 2],
            v1[i, 0], v1[i, 1], v1[i, 2],
            v2[i, 0], v2[i, 1], v2[i, 2],
            0
        ))
        
    buf.seek(0)
    return buf.getvalue()
