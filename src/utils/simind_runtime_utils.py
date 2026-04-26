"""
SIMIND runtime/path helpers used by the simulation stage.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Dict, Sequence

import numpy as np


def hu_to_mu(
    hu_arr: np.ndarray,
    pixel_size_cm: float,
    mu_water: float,
    mu_bone: float,
) -> np.ndarray:
    """
    Convert HU CT values to a linear attenuation map (mu) in per-pixel units.

    Two-segment model:
      HU <= 0 (soft tissue/air): mu = mu_water * (1 + HU/1000)
      HU  > 0 (bone):           mu = mu_water + (HU/1000) * (mu_bone - mu_water)
    """
    mu_water_px = mu_water * pixel_size_cm
    mu_bone_px = mu_bone * pixel_size_cm
    mu_map = np.zeros_like(hu_arr, dtype=np.float32)
    soft = hu_arr <= 0
    bone = hu_arr > 0
    mu_map[soft] = mu_water_px * (1 + hu_arr[soft] / 1000.0)
    mu_map[bone] = mu_water_px + (hu_arr[bone] / 1000.0) * (mu_bone_px - mu_water_px)
    return mu_map


def configure_simind_environment(simind_dir: str) -> None:
    """Set environment variables required by SIMIND at runtime."""
    smc_dir = os.path.join(simind_dir, "smc_dir")
    if not os.path.isdir(smc_dir):
        raise FileNotFoundError(f"SMC_DIR folder not found: {smc_dir}")
    if not smc_dir.endswith(os.sep):
        smc_dir += os.sep
    os.environ["SMC_DIR"] = smc_dir
    os.environ["PATH"] = simind_dir + os.pathsep + os.environ.get("PATH", "")


def copy_simind_templates(repo_root: str, work_dir: str, prefix: str) -> None:
    """Copy SIMIND template files into the work directory."""
    shutil.copyfile(
        os.path.join(repo_root, "data", "scattwin.win"),
        os.path.join(work_dir, f"{prefix}.win"),
    )
    shutil.copyfile(
        os.path.join(repo_root, "data", "smc.smc"),
        os.path.join(work_dir, f"{prefix}.smc"),
    )


def get_projection_paths_for_organ(base_dir: str, prefix: str, organ_name: str) -> Dict[str, str]:
    """Return output projection paths (w1/w2/w3) for a single organ."""
    return {
        "w1": os.path.join(base_dir, f"{prefix}_{organ_name}_tot_w1.a00"),
        "w2": os.path.join(base_dir, f"{prefix}_{organ_name}_tot_w2.a00"),
        "w3": os.path.join(base_dir, f"{prefix}_{organ_name}_tot_w3.a00"),
    }


def get_summed_projection_paths(output_dir: str, prefix: str) -> Dict[str, str]:
    """Return output paths for summed (across all ROIs) projection totals."""
    return {
        "w1": os.path.join(output_dir, f"{prefix}_tot_w1.a00"),
        "w2": os.path.join(output_dir, f"{prefix}_tot_w2.a00"),
        "w3": os.path.join(output_dir, f"{prefix}_tot_w3.a00"),
    }


def build_projection_path_dict(base_dir: str, prefix: str, roi_list: Sequence[str]) -> Dict[str, Dict[str, str]]:
    """Return output projection paths for all organs."""
    return {
        organ: get_projection_paths_for_organ(base_dir, prefix, organ)
        for organ in roi_list
    }


def organ_totals_exist(work_dir: str, prefix: str, organ_name: str) -> bool:
    """Return True if all three energy window total files exist for this organ."""
    return all(
        os.path.exists(path)
        for path in get_projection_paths_for_organ(work_dir, prefix, organ_name).values()
    )


def organ_headers_exist(work_dir: str, prefix: str, organ_name: str) -> bool:
    """Return True if the SIMIND header for core 0 exists."""
    return os.path.exists(os.path.join(work_dir, f"{prefix}_{organ_name}_0_tot_w2.h00"))


def calibration_exists(output_dir: str) -> bool:
    """Return True if ``calib.res`` exists in the output directory."""
    return os.path.exists(os.path.join(output_dir, "calib.res"))


def run_jaszczak_calibration(
    output_dir: str,
    repo_root: str,
    simind_exe: str,
    isotope: str,
    collimator: str,
) -> None:
    """Run Jaszczak calibration in SIMIND to produce ``calib.res``."""
    if calibration_exists(output_dir):
        return

    jaszak_file = os.path.join(repo_root, "data", "jaszak.smc")
    shutil.copyfile(jaszak_file, os.path.join(output_dir, "jaszak.smc"))

    cmd = (
        f"{simind_exe} jaszak calib"
        f"/fi:{isotope}"
        f"/cc:{collimator}"
        "/29:1"
        "/15:5"
        "/fa:11"
        "/fa:15"
        "/fa:14"
    )
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=output_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        if len(stderr) > 1000:
            stderr = stderr[-1000:]
        detail = f"\nSIMIND stderr:\n{stderr}" if stderr else ""
        raise RuntimeError(
            f"SIMIND Jaszczak calibration failed with exit code {result.returncode}: {cmd}{detail}"
        )


def derive_input_geometry(
    arr_shape: Sequence[int],
    arr_px_spacing_cm: Sequence[float],
    *,
    output_slice_width: float,
    detector_width: float,
    detector_length: float,
) -> Dict[str, float]:
    """Derive SIMIND input/output geometry values from preprocessing outputs."""
    input_slice_width = float(arr_px_spacing_cm[0])
    input_pixel_width = float(arr_px_spacing_cm[1])
    input_half_length = float(input_slice_width * arr_shape[0] / 2.0)
    output_img_length = float(input_slice_width * arr_shape[0] / float(output_slice_width))
    detector_width_cm = float(detector_width)
    detector_length_cm = (
        float(arr_shape[0] * input_slice_width)
        if float(detector_length) == 0.0
        else float(detector_length)
    )
    return {
        "input_slice_width": input_slice_width,
        "input_pixel_width": input_pixel_width,
        "input_half_length": input_half_length,
        "output_img_length": output_img_length,
        "detector_width_cm": detector_width_cm,
        "detector_length_cm": detector_length_cm,
    }


def build_simind_switches(
    atn_name: str,
    act_name: str,
    arr_shape: Sequence[int],
    geometry: Dict[str, float],
    scale_factor: float,
    *,
    collimator: str,
    isotope: str,
    energy_window_width: float,
    output_pixel_width: float,
    num_projections: int,
    detector_distance: float,
    output_img_size: int,
) -> str:
    """Build the SIMIND command-line switch string for a single organ simulation."""
    return (
        f"/fd:{atn_name}"
        f"/fs:{act_name}"
        "/in:x22,3x"
        f"/nn:{scale_factor}"
        f"/cc:{collimator}"
        f"/fi:{isotope}"
        f"/02:{geometry['input_half_length']}"
        f"/05:{geometry['input_half_length']}"
        f"/08:{geometry['detector_length_cm']:.2f}"
        f"/10:{geometry['detector_width_cm']:.2f}"
        "/14:-7"
        "/15:-7"
        f"/20:{-1 * energy_window_width}"
        f"/21:{-1 * energy_window_width}"
        f"/28:{output_pixel_width}"
        f"/29:{num_projections}"
        f"/31:{geometry['input_pixel_width']}"
        f"/34:{arr_shape[0]}"
        f"/42:{detector_distance}"
        f"/76:{output_img_size}"
        f"/77:{geometry['output_img_length']}"
        f"/78:{arr_shape[1]}"
        f"/79:{arr_shape[2]}"
    )


def aggregate_core_projection_totals(
    work_dir: str,
    prefix: str,
    organ_name: str,
    *,
    num_cpu: int,
    mode: str,
) -> None:
    """Average projection totals across SIMIND runs and write organ-level totals."""
    xtot_w1 = xtot_w2 = xtot_w3 = 0.0

    for core_id in range(num_cpu):
        p1 = os.path.join(work_dir, f"{prefix}_{organ_name}_{core_id}_tot_w1.a00")
        p2 = os.path.join(work_dir, f"{prefix}_{organ_name}_{core_id}_tot_w2.a00")
        p3 = os.path.join(work_dir, f"{prefix}_{organ_name}_{core_id}_tot_w3.a00")

        xtot_w1 += np.fromfile(p1, dtype=np.float32)
        xtot_w2 += np.fromfile(p2, dtype=np.float32)
        xtot_w3 += np.fromfile(p3, dtype=np.float32)

        if mode == "PRODUCTION":
            for path in (p1, p2, p3):
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass

    xtot_w1 /= num_cpu
    xtot_w2 /= num_cpu
    xtot_w3 /= num_cpu

    organ_paths = get_projection_paths_for_organ(work_dir, prefix, organ_name)
    np.asarray(xtot_w1, dtype=np.float32).tofile(organ_paths["w1"])
    np.asarray(xtot_w2, dtype=np.float32).tofile(organ_paths["w2"])
    np.asarray(xtot_w3, dtype=np.float32).tofile(organ_paths["w3"])


def copy_simind_headers_to_dir(work_dir: str, header_dir: str, prefix: str, first_roi: str) -> None:
    """Copy the SIMIND headers that downstream reconstruction needs."""
    extensions = [
        f"_{first_roi}_0_tot_w1.h00",
        f"_{first_roi}_0_tot_w2.h00",
        f"_{first_roi}_0_tot_w3.h00",
        f"_{first_roi}_0.cor",
        f"_{first_roi}_0.hct",
        f"_{first_roi}_0.ict",
    ]
    for ext in extensions:
        src = os.path.join(work_dir, f"{prefix}{ext}")
        dst = os.path.join(header_dir, f"{prefix}{ext}")
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copyfile(src, dst)


def sum_projections_across_organs(
    work_dir: str,
    output_dir: str,
    prefix: str,
    roi_list: Sequence[str],
) -> Dict[str, str]:
    """Sum per-organ projection totals into a single total per energy window."""
    summed_paths = get_summed_projection_paths(output_dir, prefix)
    if all(os.path.exists(path) for path in summed_paths.values()):
        return summed_paths

    sum_w1 = sum_w2 = sum_w3 = None
    for organ_name in roi_list:
        organ_paths = get_projection_paths_for_organ(work_dir, prefix, organ_name)
        w1 = np.fromfile(organ_paths["w1"], dtype=np.float32)
        w2 = np.fromfile(organ_paths["w2"], dtype=np.float32)
        w3 = np.fromfile(organ_paths["w3"], dtype=np.float32)
        sum_w1 = w1.copy() if sum_w1 is None else sum_w1 + w1
        sum_w2 = w2.copy() if sum_w2 is None else sum_w2 + w2
        sum_w3 = w3.copy() if sum_w3 is None else sum_w3 + w3

    if sum_w1 is not None:
        np.asarray(sum_w1, dtype=np.float32).tofile(summed_paths["w1"])
        np.asarray(sum_w2, dtype=np.float32).tofile(summed_paths["w2"])
        np.asarray(sum_w3, dtype=np.float32).tofile(summed_paths["w3"])

    return summed_paths
