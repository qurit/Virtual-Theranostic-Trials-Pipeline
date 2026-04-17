"""
Shared simulation-grid resize utilities.

Both the SIMIND and OpenGATE stages accept an ``xyz_dim`` parameter that
specifies a target voxel grid for downsampling the CT and segmentation before
simulation.  This module provides the single source of truth for resolving that
parameter into concrete per-axis target sizes and scale factors.

The caller is responsible for applying the resulting scales using its own
backend (scipy.ndimage.zoom for SIMIND, SimpleITK for OpenGATE).
"""

from __future__ import annotations

from typing import Optional, Tuple


def resolve_simulation_grid(
    xyz_dim: Optional[object],
    current_size_xyz: Tuple[int, int, int],
) -> Optional[Tuple[Tuple[int, int, int], Tuple[float, float, float]]]:
    """
    Resolve an ``xyz_dim`` specification to target voxel counts and per-axis
    scale factors.

    Parameters
    ----------
    xyz_dim : list[int] | int | None
        Target grid specification.

        - ``[nx, ny, nz]`` — explicit per-axis target voxel counts.
        - ``int`` — legacy scalar; the in-plane dimensions (x, y) are scaled
          so that the larger of the two reaches the target, and the same
          isotropic factor is applied to z.
        - ``None`` — no resizing; returns ``None``.

    current_size_xyz : (int, int, int)
        Current voxel counts in ``(x, y, z)`` order (NIfTI / SimpleITK
        convention).

    Returns
    -------
    ((nx, ny, nz), (scale_x, scale_y, scale_z))
        Target voxel counts and the per-axis scale factors applied to reach
        them.  Each scale is ``target / current`` for that axis.
    None
        Returned when ``xyz_dim`` is ``None`` or when every target axis is
        already at or above the current size (no downsampling required).

    Notes
    -----
    Scale factors are always positive.  When a ``current`` dimension is zero
    the corresponding scale defaults to ``1.0`` to avoid division by zero.
    """
    if xyz_dim is None:
        return None

    sx, sy, sz = int(current_size_xyz[0]), int(current_size_xyz[1]), int(current_size_xyz[2])

    if isinstance(xyz_dim, (list, tuple)):
        nx = max(1, int(xyz_dim[0]))
        ny = max(1, int(xyz_dim[1]))
        nz = max(1, int(xyz_dim[2]))
    else:
        # Legacy scalar: isotropic scale so the larger in-plane dimension
        # reaches the target.
        target = int(xyz_dim)
        if sx <= target and sy <= target:
            return None
        s = min(target / sx, target / sy) if (sx > 0 and sy > 0) else 1.0
        nx = max(1, round(sx * s))
        ny = max(1, round(sy * s))
        nz = max(1, round(sz * s))

    if nx >= sx and ny >= sy and nz >= sz:
        return None  # target is at or above current size on every axis

    scale_x = nx / sx if sx > 0 else 1.0
    scale_y = ny / sy if sy > 0 else 1.0
    scale_z = nz / sz if sz > 0 else 1.0

    return (nx, ny, nz), (scale_x, scale_y, scale_z)
