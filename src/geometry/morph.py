"""Free-Form Deformation (FFD) macro CAD shape morphing.

Uses PyGeM when available for high-fidelity FFD, with an automatic
pure-NumPy Bernstein polynomial fallback so the system never fails
if PyGeM C-extensions are missing.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import meshio
import numpy as np

# Global FFD grid dimensions (first and last X-slices are anchored,
# NX - 2 interior slices carry the bending DOFs)
FFD_NX: int = 5
FFD_NY: int = 4
FFD_NZ: int = 4

# FFD bounding box covering the full bone-plate assembly
FFD_BOX_ORIGIN: np.ndarray = np.array([-0.01, -0.025, -0.025])
FFD_BOX_LENGTH: np.ndarray = np.array([0.18, 0.05, 0.05])

# Try importing PyGeM at module level
try:
    from pygem.ffd import FFD as PyGeMFFD
    _HAS_PYGEM: bool = True
except ImportError:
    _HAS_PYGEM = False


def _bernstein(n: int, i: int, t: np.ndarray) -> np.ndarray:
    """Evaluate the i-th Bernstein basis polynomial of degree n at parameter t."""
    from math import comb
    return comb(n, i) * (t ** i) * ((1.0 - t) ** (n - i))


def _ffd_numpy_fallback(
    points: np.ndarray,
    bend_y_array: np.ndarray,
    bend_z_array: np.ndarray,
) -> np.ndarray:
    """Pure-NumPy FFD deformation using tensor-product Bernstein polynomials.

    Replicates PyGeM FFD behaviour for the special case where only
    the interior X-slices carry uniform Y and Z displacements.
    """
    n_interior: int = FFD_NX - 2
    bend_y_array = np.atleast_1d(bend_y_array)[:n_interior]
    bend_z_array = np.atleast_1d(bend_z_array)[:n_interior]

    # Map physical coordinates to [0, 1] parametric space
    s: np.ndarray = (points - FFD_BOX_ORIGIN) / FFD_BOX_LENGTH

    # Clamp to [0, 1]
    s = np.clip(s, 0.0, 1.0)

    # Build displacement field from Bernstein basis along X axis
    # Only interior control points (indices 1..NX-2) carry displacement
    disp = np.zeros_like(points)
    nx_degree: int = FFD_NX - 1

    for i_interior in range(n_interior):
        cp_index: int = i_interior + 1  # skip the anchored first slice
        basis_x: np.ndarray = _bernstein(nx_degree, cp_index, s[:, 0])
        disp[:, 1] += basis_x * bend_y_array[i_interior] * FFD_BOX_LENGTH[1]
        disp[:, 2] += basis_x * bend_z_array[i_interior] * FFD_BOX_LENGTH[2]

    return points + disp


def apply_pygem_ffd(
    input_mesh_path: str,
    output_mesh_path: str,
    bend_y_array: np.ndarray,
    bend_z_array: np.ndarray,
) -> str:
    """Apply Free-Form Deformation to morph the global CAD mesh shape.

    Parameters
    ----------
    input_mesh_path : str
        Path to the base Gmsh mesh file.
    output_mesh_path : str
        Path where the morphed mesh will be written.
    bend_y_array : np.ndarray
        Y-axis bending displacements for interior FFD control point slices.
    bend_z_array : np.ndarray
        Z-axis bending displacements for interior FFD control point slices.

    Returns
    -------
    str
        The output_mesh_path (for chaining).
    """
    if not os.path.exists(input_mesh_path):
        raise FileNotFoundError(f"Input mesh {input_mesh_path} not found.")

    mesh = meshio.read(input_mesh_path)
    points: np.ndarray = np.array(mesh.points, dtype=np.float64)

    bend_y_array = np.atleast_1d(bend_y_array)
    bend_z_array = np.atleast_1d(bend_z_array)
    n_interior: int = FFD_NX - 2

    if _HAS_PYGEM:
        ffd = PyGeMFFD()
        ffd.box_length = FFD_BOX_LENGTH.tolist()
        ffd.box_origin = FFD_BOX_ORIGIN.tolist()
        ffd.n_control_points = [FFD_NX, FFD_NY, FFD_NZ]

        for i in range(n_interior):
            ffd.array_mu_y[i + 1, :, :] = bend_y_array[i]
            ffd.array_mu_z[i + 1, :, :] = bend_z_array[i]

        morphed_points: np.ndarray = ffd(points)
    else:
        morphed_points = _ffd_numpy_fallback(points, bend_y_array, bend_z_array)

    mesh.points = morphed_points
    mesh.write(output_mesh_path, file_format="gmsh")

    return output_mesh_path


def build_ffd_warper(
    bend_y_array: np.ndarray,
    bend_z_array: np.ndarray,
) -> callable:
    """Return a callable that warps arbitrary 3D point arrays via FFD.

    Used by plot_plotly.py to warp Marching Cubes vertices for
    visualization and STL export.
    """
    bend_y_array = np.atleast_1d(bend_y_array)
    bend_z_array = np.atleast_1d(bend_z_array)

    if _HAS_PYGEM:
        ffd = PyGeMFFD()
        ffd.box_length = FFD_BOX_LENGTH.tolist()
        ffd.box_origin = FFD_BOX_ORIGIN.tolist()
        ffd.n_control_points = [FFD_NX, FFD_NY, FFD_NZ]
        n_interior: int = FFD_NX - 2
        for i in range(n_interior):
            ffd.array_mu_y[i + 1, :, :] = bend_y_array[i]
            ffd.array_mu_z[i + 1, :, :] = bend_z_array[i]
        return ffd
    else:
        def _warp(pts: np.ndarray) -> np.ndarray:
            return _ffd_numpy_fallback(pts, bend_y_array, bend_z_array)
        return _warp
