
# --- GLOBAL CAD MORPHING CONFIGURATION ---
# Change these values to adjust the number of FFD control points across the entire application.
# NX must be >= 3 (first and last slices are anchored, NX-2 are moving interior slices).
FFD_NX, FFD_NY, FFD_NZ = 5, 4, 4
# -----------------------------------------

import meshio
import numpy as np
from pygem.ffd import FFD
import os

def apply_pygem_ffd(input_mesh_path, output_mesh_path, bend_y_array, bend_z_array):
    """
    Applies Free Form Deformation (FFD) using PyGeM to morph the global CAD/mesh shape.
    Supports highly dense control point grids.
    """
    if not os.path.exists(input_mesh_path):
        raise FileNotFoundError(f"Input mesh {input_mesh_path} not found.")

    mesh = meshio.read(input_mesh_path)
    points = np.array(mesh.points, dtype=np.float64)

    ffd = FFD()
    # Define bounding box for the plate/bone assembly
    ffd.box_length = [0.18, 0.05, 0.05]
    ffd.box_origin = [-0.01, -0.025, -0.025]
    
    # Highly dense 10x4x4 grid (8 interior moving slices)
    
    ffd.n_control_points = [FFD_NX, FFD_NY, FFD_NZ]
    
    # Apply bending to the interior control points
    bend_y_array = np.atleast_1d(bend_y_array)
    bend_z_array = np.atleast_1d(bend_z_array)
    
    for i in range(FFD_NX - 2):
        ffd.array_mu_y[i+1, :, :] = bend_y_array[i]
        ffd.array_mu_z[i+1, :, :] = bend_z_array[i]
    
    # Morph the points
    morphed_points = ffd(points)

    
    # Save the morphed mesh
    mesh.points = morphed_points
    mesh.write(output_mesh_path, file_format="gmsh")
    
    return output_mesh_path
