"""
Shared simulation-grid resize utilities.

Both the SIMIND and OpenGATE stages accept an ``xyz_spacing_mm`` parameter
that specifies a target voxel spacing (in mm) for downsampling the CT and
segmentation before simulation.  This module provides the single source of
truth for resolving that parameter into concrete per-axis target voxel counts
and scale factors.

The caller is responsible for applying the resulting scales using its own
backend (scipy.ndimage.zoom for SIMIND, SimpleITK for OpenGATE).
"""

from __future__ import annotations

from typing import Optional, Tuple


def resolve_simulation_grid(
    xyz_spacing_mm: Optional[object],
    current_size_xyz: Tuple[int, int, int],
    current_spacing_xyz_mm: Tuple[float, float, float],
) -> Optional[Tuple[Tuple[int, int, int], Tuple[float, float, float]]]:
    """
    Resolve an ``xyz_spacing_mm`` specification to target voxel counts and
    per-axis scale factors.

    Parameters
    ----------
    xyz_spacing_mm : list[float] | None
        Target voxel spacing in mm as ``[sx, sy, sz]`` (x, y, z order).
        Each value must be >= the corresponding native CT spacing (i.e. only
        coarser grids are allowed).  ``None`` means no resampling.

    current_size_xyz : (int, int, int)
        Current voxel counts in ``(x, y, z)`` order.

    current_spacing_xyz_mm : (float, float, float)
        Native voxel spacing in mm in ``(x, y, z)`` order.

    Returns
    -------
    ((nx, ny, nz), (scale_x, scale_y, scale_z))
        Target voxel counts and the per-axis scale factors applied to reach
        them.  Each scale is ``target_voxels / current_voxels`` for that axis.
    None
        Returned when ``xyz_spacing_mm`` is ``None`` or when the resolved
        target voxel counts are >= the current counts on every axis (no
        downsampling required).

    Raises
    ------
    ValueError
        If any target spacing is finer than the corresponding native spacing.
    """
    if xyz_spacing_mm is None:
        return None

    sx, sy, sz = int(current_size_xyz[0]), int(current_size_xyz[1]), int(current_size_xyz[2])
    spx, spy, spz = float(current_spacing_xyz_mm[0]), float(current_spacing_xyz_mm[1]), float(current_spacing_xyz_mm[2])

    target_spx = float(xyz_spacing_mm[0])
    target_spy = float(xyz_spacing_mm[1])
    target_spz = float(xyz_spacing_mm[2])

    _tol = 1e-6
    if target_spx < spx - _tol:
        raise ValueError(
            f"xyz_spacing_mm[0]={target_spx} mm is finer than native CT x-spacing "
            f"{spx:.4f} mm. Only coarser (larger) spacing values are allowed."
        )
    if target_spy < spy - _tol:
        raise ValueError(
            f"xyz_spacing_mm[1]={target_spy} mm is finer than native CT y-spacing "
            f"{spy:.4f} mm. Only coarser (larger) spacing values are allowed."
        )
    if target_spz < spz - _tol:
        raise ValueError(
            f"xyz_spacing_mm[2]={target_spz} mm is finer than native CT z-spacing "
            f"{spz:.4f} mm. Only coarser (larger) spacing values are allowed."
        )

    # Physical extent is fixed; target voxel counts derived from target spacing.
    phys_x = sx * spx
    phys_y = sy * spy
    phys_z = sz * spz

    nx = max(1, round(phys_x / target_spx))
    ny = max(1, round(phys_y / target_spy))
    nz = max(1, round(phys_z / target_spz))

    if nx >= sx and ny >= sy and nz >= sz:
        return None  # no downsampling needed

    scale_x = nx / sx if sx > 0 else 1.0
    scale_y = ny / sy if sy > 0 else 1.0
    scale_z = nz / sz if sz > 0 else 1.0

    return (nx, ny, nz), (scale_x, scale_y, scale_z)
