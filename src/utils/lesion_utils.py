"""
Geometry and sampling helpers for synthetic lesion placement.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt


def phys_dist_mm(
    idx1_zyx: Sequence[int],
    idx2_zyx: Sequence[int],
    spacing_zyx_mm: np.ndarray,
) -> float:
    """Physical distance (mm) between two voxel indices in zyx indexing."""
    d = (np.array(idx1_zyx, dtype=np.float64) - np.array(idx2_zyx, dtype=np.float64)) * spacing_zyx_mm
    return float(np.sqrt(np.sum(d * d)))


def candidate_weights(
    mask_zyx: np.ndarray,
    cand_zyx: np.ndarray,
    spacing_zyx_mm: np.ndarray,
    prob: str,
    sigma_mm: Optional[float] = None,
    tom_map_zyx: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return nonnegative weights for candidate center voxels."""
    prob = prob.lower()
    n = cand_zyx.shape[0]

    if prob in ("uniform", "user_defined"):
        return np.ones(n, dtype=np.float64)

    if prob == "gaussian":
        if sigma_mm is None:
            raise ValueError("sigma_mm required for prob='gaussian'")
        roi_pts = np.argwhere(mask_zyx)
        if roi_pts.size == 0:
            raise ValueError("ROI mask is empty; cannot compute gaussian centroid.")
        mu = roi_pts.mean(axis=0)

        dz = (cand_zyx[:, 0] - mu[0]) * spacing_zyx_mm[0]
        dy = (cand_zyx[:, 1] - mu[1]) * spacing_zyx_mm[1]
        dx = (cand_zyx[:, 2] - mu[2]) * spacing_zyx_mm[2]
        r2 = dx * dx + dy * dy + dz * dz
        w = np.exp(-0.5 * r2 / (float(sigma_mm) ** 2))
        return w.astype(np.float64)

    if prob == "tom":
        raise NotImplementedError(
            "prob='tom' selected, but TOM integration is TODO. "
            "Use prob='uniform' or prob='gaussian' for now."
        )

    raise ValueError(f"Unknown prob choice: {prob}")


def compute_distance_to_boundary_mm(mask_zyx: np.ndarray, spacing_zyx_mm: np.ndarray) -> np.ndarray:
    """Distance transform (mm) inside the ROI mask, 0 outside the ROI."""
    return distance_transform_edt(mask_zyx.astype(np.uint8), sampling=spacing_zyx_mm)


def find_auto_radius_start_mm(
    mask_zyx: np.ndarray,
    spacing_zyx_mm: np.ndarray,
    n_lesions: int,
    dist_mm: np.ndarray,
    margin_mm: float,
    *,
    auto_start_frac: float,
) -> float:
    """Compute an initial auto radius guess in mm."""
    max_r_mm = max(0.0, float(dist_mm.max()) - float(margin_mm))
    roi_vox = int(mask_zyx.sum())
    roi_vol_mm3 = float(roi_vox) * float(np.prod(spacing_zyx_mm))
    r_eq = float((3.0 * roi_vol_mm3 / (4.0 * np.pi)) ** (1.0 / 3.0)) if roi_vol_mm3 > 0 else 0.0
    scale = max(1.0, float(n_lesions) ** (1.0 / 3.0))
    r_start = min(max_r_mm, float(auto_start_frac) * (r_eq / scale))
    return max(0.0, float(r_start))


def sample_auto_radii_mm(
    r_start_mm: float,
    n_lesions: int,
    spacing_zyx_mm: np.ndarray,
    seed: int,
    *,
    eps_radius_vox_frac: float,
) -> List[float]:
    """Sample an initial set of lesion radii for auto mode."""
    r_start = float(r_start_mm)
    if n_lesions <= 0:
        raise ValueError("n_lesions must be > 0")

    eps_mm = float(eps_radius_vox_frac * float(np.min(spacing_zyx_mm)))
    r_hi = max(r_start, 0.0)
    if r_hi <= 0.0:
        return [0.0 for _ in range(n_lesions)]

    rng = np.random.default_rng(int(seed))
    u = rng.random(n_lesions) ** 2
    radii = (eps_mm + (r_hi - eps_mm) * u).astype(np.float64)

    if eps_mm >= r_hi:
        radii = np.full(n_lesions, max(0.5 * r_hi, 1e-6), dtype=np.float64)

    return sorted([float(r) for r in radii], reverse=True)


def place_lesion_centers(
    mask_zyx: np.ndarray,
    dist_mm: np.ndarray,
    radii_mm: List[float],
    spacing_zyx_mm: np.ndarray,
    *,
    prob: str = "uniform",
    sigma_mm: Optional[float] = None,
    margin_mm: float = 1.0,
    seed: int = 0,
    max_attempts_per_lesion: int = 4000,
    tom_map_zyx: Optional[np.ndarray] = None,
    user_centers_zyx: Optional[List[Tuple[int, int, int]]] = None,
) -> Tuple[List[Tuple[int, int, int]], List[float]]:
    """Place lesion centers inside a mask with boundary and overlap constraints."""
    prob_l = str(prob).lower()
    rng = np.random.default_rng(int(seed))

    centers: List[Tuple[int, int, int]] = []
    placed_r: List[float] = []

    if prob_l == "user_defined":
        if user_centers_zyx is None:
            raise ValueError("prob='user_defined' but user_centers_zyx=None")
        if len(user_centers_zyx) != len(radii_mm):
            raise ValueError("user_centers_zyx length must match radii_mm length")

        for c_in, r in zip(user_centers_zyx, radii_mm):
            c = tuple(map(int, c_in))
            r = float(r)

            if not mask_zyx[c]:
                raise ValueError(f"User center {c} not inside ROI")
            if float(dist_mm[c]) < (r + float(margin_mm)):
                raise ValueError(f"User center {c} too close to ROI boundary for radius {r} mm")

            for cj, rj in zip(centers, placed_r):
                if phys_dist_mm(c, cj, spacing_zyx_mm) < (r + rj + float(margin_mm)):
                    raise ValueError(f"User center {c} overlaps existing lesion at {cj}")

            centers.append(c)
            placed_r.append(r)

        return centers, placed_r

    for i, r in enumerate(radii_mm, start=1):
        r = float(r)
        cand_mask = dist_mm >= (r + float(margin_mm))
        cand = np.argwhere(cand_mask)
        if cand.shape[0] == 0:
            raise RuntimeError(
                f"No valid candidate centers for radius={r:.3f} mm (after margin). "
                "Try smaller radii or smaller margin_mm."
            )

        w = candidate_weights(
            mask_zyx=mask_zyx,
            cand_zyx=cand,
            spacing_zyx_mm=spacing_zyx_mm,
            prob=prob_l,
            sigma_mm=sigma_mm,
            tom_map_zyx=tom_map_zyx,
        )
        w = np.maximum(w, 0.0)
        w_sum = float(w.sum())
        if w_sum <= 0.0:
            raise RuntimeError("All candidate weights are zero. Check gaussian sigma / probability map.")
        p = w / w_sum

        placed = False
        for _ in range(int(max_attempts_per_lesion)):
            k = int(rng.choice(cand.shape[0], p=p))
            c = tuple(map(int, cand[k]))

            ok = True
            for cj, rj in zip(centers, placed_r):
                if phys_dist_mm(c, cj, spacing_zyx_mm) < (r + rj + float(margin_mm)):
                    ok = False
                    break

            if ok:
                centers.append(c)
                placed_r.append(r)
                placed = True
                break

        if not placed:
            raise RuntimeError(
                f"Failed to place lesion {i}/{len(radii_mm)} (r={r:.3f} mm) after {max_attempts_per_lesion} attempts. "
                "Try reducing radii, margin_mm, or switching prob='uniform'."
            )

    return centers, placed_r


def auto_place_lesions(
    roi_name: str,
    organ_mask_zyx: np.ndarray,
    spacing_zyx_mm: np.ndarray,
    dist_mm: np.ndarray,
    n_lesions: int,
    prob: str,
    sigma_mm: Optional[float],
    margin_mm: float,
    seed: int,
    *,
    auto_start_frac: float,
    auto_shrink_factor: float,
    auto_max_shrink_iters: int,
    eps_radius_vox_frac: float,
    max_attempts_per_lesion: int,
) -> Tuple[List[Tuple[int, int, int]], List[float]]:
    """Auto-place lesions by sampling radii, then shrinking on failure."""
    r_start = find_auto_radius_start_mm(
        mask_zyx=organ_mask_zyx,
        spacing_zyx_mm=spacing_zyx_mm,
        n_lesions=n_lesions,
        dist_mm=dist_mm,
        margin_mm=margin_mm,
        auto_start_frac=auto_start_frac,
    )

    if r_start <= 0.0:
        raise RuntimeError(
            f"[{roi_name}] No room for lesions: r_start <= 0 (distance-to-boundary too small vs margin_mm)."
        )

    radii_try = sample_auto_radii_mm(
        r_start_mm=r_start,
        n_lesions=n_lesions,
        spacing_zyx_mm=spacing_zyx_mm,
        seed=seed,
        eps_radius_vox_frac=eps_radius_vox_frac,
    )
    eps_mm = float(eps_radius_vox_frac * float(np.min(spacing_zyx_mm)))

    last_err: Optional[Exception] = None
    for shrink_i in range(int(auto_max_shrink_iters)):
        try:
            return place_lesion_centers(
                mask_zyx=organ_mask_zyx,
                dist_mm=dist_mm,
                radii_mm=radii_try,
                spacing_zyx_mm=spacing_zyx_mm,
                prob=prob,
                sigma_mm=sigma_mm,
                margin_mm=margin_mm,
                seed=seed + shrink_i,
                max_attempts_per_lesion=max_attempts_per_lesion,
                tom_map_zyx=None,
                user_centers_zyx=None,
            )
        except RuntimeError as exc:
            last_err = exc
            radii_try = sorted(
                [float(r) * float(auto_shrink_factor) for r in radii_try],
                reverse=True,
            )
            if float(max(radii_try)) <= eps_mm + 1e-9:
                break

    raise RuntimeError(f"[{roi_name}] Auto placement failed after shrinking. Last error: {last_err}")


def build_lesion_labelmap_zyx(
    mask_zyx: np.ndarray,
    centers_zyx: List[Tuple[int, int, int]],
    radii_mm: List[float],
    spacing_zyx_mm: np.ndarray,
) -> np.ndarray:
    """Create a per-ROI lesion label map in zyx order."""
    z_size, y_size, x_size = mask_zyx.shape
    labels = np.zeros((z_size, y_size, x_size), dtype=np.uint16)

    for lbl, (center, radius_mm) in enumerate(zip(centers_zyx, radii_mm), start=1):
        z0, y0, x0 = center
        radius_mm = float(radius_mm)

        rz = int(np.ceil(radius_mm / float(spacing_zyx_mm[0])))
        ry = int(np.ceil(radius_mm / float(spacing_zyx_mm[1])))
        rx = int(np.ceil(radius_mm / float(spacing_zyx_mm[2])))

        zmin, zmax = max(0, z0 - rz), min(z_size, z0 + rz + 1)
        ymin, ymax = max(0, y0 - ry), min(y_size, y0 + ry + 1)
        xmin, xmax = max(0, x0 - rx), min(x_size, x0 + rx + 1)

        zz, yy, xx = np.ogrid[zmin:zmax, ymin:ymax, xmin:xmax]
        dz = (zz - z0) * float(spacing_zyx_mm[0])
        dy = (yy - y0) * float(spacing_zyx_mm[1])
        dx = (xx - x0) * float(spacing_zyx_mm[2])

        sphere = (dx * dx + dy * dy + dz * dz) <= (radius_mm * radius_mm)
        sphere &= mask_zyx[zmin:zmax, ymin:ymax, xmin:xmax]
        labels[zmin:zmax, ymin:ymax, xmin:xmax][sphere] = np.uint16(lbl)

    return labels
