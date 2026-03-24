"""
SIMIND Simulation Stage for the TDT pipeline.

This stage runs SIMIND to produce per-organ SPECT projection totals (three energy windows) using:
- An attenuation map binary produced by preprocessing (`context.atn_av_path`)
- Per-organ binary ROI source maps (`context.binary_roi_act_map_paths`)
- SIMIND template files (.win, .smc) copied into a working directory

High-level workflow
-------------------
1) Validate required context fields (labels, spacing, files).
2) Configure SIMIND environment variables (SMC_DIR, PATH).
3) Copy SIMIND template files into a per-CT work directory.
4) For each ROI:
   - Copy the source map binary into the work directory.
   - Run SIMIND in parallel using `num_cores` processes (each with a unique /rr: seed).
   - Aggregate per-core projection totals into a single per-organ file per energy window.
5) Run a Jaszczak-based calibration (if not already present) to produce `calib.res`.

Expected Context interface
--------------------------
Incoming `context` is expected to provide:
- context.config["phase_2"]["simind_stage"] : dict
- context.subdir_paths["phase_2"] : str
- context.mode : str  ("DEBUG" or "PRODUCTION")
- context.class_seg : dict[str, int]
- context.arr_shape_new : tuple[int, int, int]  (z, y, x)
- context.arr_px_spacing_cm : tuple[float, float, float]  (z, y, x)
- context.binary_roi_act_map_paths : dict[str, str]
- context.mask_roi_body : dict[int, np.ndarray]
- context.atn_av_path : str

On success, this stage sets:
- context.spect_sim_output_dir : str
- context.simind_stage_output_dir : str
- context.simind_work_dir : str
- context.simind_metadata_path : str
- context.simind_calibration_path : str
- context.simind_projection_paths : dict[str, dict[str, str]]  ({roi: {w1/w2/w3: path}})
- context.simind_num_cores : int
- context.simind_geometry : dict[str, float]
- context.simind_total_num_voxels : int
- context.simind_scale_factor : float
- context.simind_switches_by_organ : dict[str, str]

Maintainer / contact: pyazdi@bccrc.ca
"""

from __future__ import annotations

import os
import json
import shutil
import subprocess
from typing import Any, Dict, List

import numpy as np


class SimindSimulationStage:
    """
    Run SIMIND simulations (per-organ, parallel cores) and save per-organ projection totals.

    Parameters
    ----------
    context : Context-like
        Pipeline context containing config and preprocessing outputs.
    """

    def __init__(self, context: Any) -> None:
        self.context = context
        self.config: Dict[str, Any] = context.config

        # Repository root: one level above this stage file (src/stages/ -> repo root).
        self.repo_root: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        self.phase_output_dir: str = context.subdir_paths["phase_2"]
        self.stage_cfg: Dict[str, Any] = context.config["phase_2"]["simind_stage"]
        self.output_dir: str = self.phase_output_dir
        self.stage_output_dir: str = os.path.join(
            self.phase_output_dir,
            self.stage_cfg.get("sub_dir_name", "simind_simulation"),
        )
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.stage_output_dir, exist_ok=True)

        self.work_dir: str = os.path.join(self.stage_output_dir, "work_dir")
        os.makedirs(self.work_dir, exist_ok=True)

        self.metadata_path: str = os.path.join(self.work_dir, "simind_metadata.json")
        self.calibration_path: str = os.path.join(self.output_dir, "calib.res")

        self.prefix: str = self.stage_cfg["file_prefix"]
        self.mode: str = self.context.mode

        # SIMIND acquisition parameters from config
        self.collimator: str = self.stage_cfg["Collimator"]
        self.isotope: str = self.stage_cfg["Isotope"]
        self.num_projections: int = self.stage_cfg["NumProjections"]
        self.detector_distance: float = self.stage_cfg["DetectorDistance"]
        self.output_img_size: int = self.stage_cfg["OutputImgSize"]
        self.output_pixel_width: float = self.stage_cfg["OutputPixelWidth"]
        self.output_slice_width: float = self.stage_cfg["OutputSliceWidth"]
        self.num_photons: float = self.stage_cfg["NumPhotons"]
        self.simind_dir: str = self.stage_cfg["SIMINDDirectory"]
        self.energy_window_width: float = self.stage_cfg["EnergyWindowWidth"]
        self.detector_width: float = self.stage_cfg["DetectorWidth"]
        self.detector_length: float = self.stage_cfg["DetectorLength"]

        # CPU count: 0 or invalid -> use all available cores
        num_cores = self.stage_cfg["NumCores"]
        max_cores = os.cpu_count() or 1
        if isinstance(num_cores, bool) or not isinstance(num_cores, int) or num_cores < 0 or num_cores > max_cores:
            self.num_cores = max_cores
        elif num_cores == 0:
            self.num_cores = max_cores
        else:
            self.num_cores = num_cores

        # SIMIND executable (supports .exe suffix on Windows)
        simind_exe = os.path.join(self.simind_dir, "simind")
        if not os.path.exists(simind_exe) and os.path.exists(simind_exe + ".exe"):
            simind_exe = simind_exe + ".exe"
        self.simind_exe: str = simind_exe

        if not os.path.exists(self.simind_exe):
            raise FileNotFoundError(f"SIMIND executable not found: {self.simind_exe}")

    # -----------------------------
    # helpers
    # -----------------------------

    def _set_simind_environment(self) -> None:
        """
        Set environment variables required by SIMIND at runtime.

        SMC_DIR must point to the `smc_dir` resource folder and end with os.sep
        (SIMIND requires this trailing separator).
        """
        smc_dir = os.path.join(self.simind_dir, "smc_dir")
        if not os.path.isdir(smc_dir):
            raise FileNotFoundError(f"SMC_DIR folder not found: {smc_dir}")
        if not smc_dir.endswith(os.sep):
            smc_dir += os.sep
        os.environ["SMC_DIR"] = smc_dir
        os.environ["PATH"] = self.simind_dir + os.pathsep + os.environ.get("PATH", "")

    def _copy_templates(self) -> None:
        """
        Copy SIMIND template files from <repo_root>/data into the work directory.

        Templates required:
        - scattwin.win  -> <work_dir>/<prefix>.win
        - smc.smc       -> <work_dir>/<prefix>.smc
        """
        shutil.copyfile(
            os.path.join(self.repo_root, "data", "scattwin.win"),
            os.path.join(self.work_dir, f"{self.prefix}.win"),
        )
        shutil.copyfile(
            os.path.join(self.repo_root, "data", "smc.smc"),
            os.path.join(self.work_dir, f"{self.prefix}.smc"),
        )

    def _get_projection_paths_for_organ(self, organ_name: str) -> Dict[str, str]:
        """Return output projection paths (w1/w2/w3) for a single organ."""
        return {
            "w1": os.path.join(self.output_dir, f"{self.prefix}_{organ_name}_tot_w1.a00"),
            "w2": os.path.join(self.output_dir, f"{self.prefix}_{organ_name}_tot_w2.a00"),
            "w3": os.path.join(self.output_dir, f"{self.prefix}_{organ_name}_tot_w3.a00"),
        }

    def _build_projection_path_dict(self, roi_list: List[str]) -> Dict[str, Dict[str, str]]:
        """Return output projection paths for all organs."""
        return {organ: self._get_projection_paths_for_organ(organ) for organ in roi_list}

    def _organ_totals_exist(self, organ_name: str) -> bool:
        """Return True if all three energy window total files exist for this organ."""
        return all(
            os.path.exists(p)
            for p in self._get_projection_paths_for_organ(organ_name).values()
        )

    def _organ_headers_exist(self, organ_name: str) -> bool:
        """Return True if the SIMIND header for core 0 exists (w2 photopeak window)."""
        return os.path.exists(
            os.path.join(self.work_dir, f"{self.prefix}_{organ_name}_0_tot_w2.h00")
        )

    def _calibration_exists(self) -> bool:
        """Return True if `calib.res` exists in the output directory."""
        return os.path.exists(os.path.join(self.output_dir, "calib.res"))

    def _run_jaszczak_calibration(self) -> None:
        """
        Run Jaszczak calibration in SIMIND to produce `calib.res`.

        No-op if `calib.res` already exists.
        Requires `jaszak.smc` (note: SIMIND uses this spelling) in <repo_root>/data.
        """
        if self._calibration_exists():
            return

        jaszak_file = os.path.join(self.repo_root, "data", "jaszak.smc") 
        shutil.copyfile(jaszak_file, os.path.join(self.output_dir, "jaszak.smc"))

        cmd = (
            f"{self.simind_exe} jaszak calib"
            f"/fi:{self.isotope}"
            f"/cc:{self.collimator}"
            f"/29:1"
            f"/15:5"
            f"/fa:11"
            f"/fa:15"
            f"/fa:14"
        )
        subprocess.run(cmd, shell=True, cwd=self.output_dir, stdout=subprocess.DEVNULL)

    def _get_input_geometry(
        self,
        arr_shape: tuple,
        arr_px_spacing_cm: tuple,
    ) -> Dict[str, float]:
        """
        Derive SIMIND input/output geometry values from preprocessing outputs.

        All lengths in cm; lengths match the (z, y, x) array convention.
        """
        input_slice_width = float(arr_px_spacing_cm[0])
        input_pixel_width = float(arr_px_spacing_cm[1])
        input_half_length = float(input_slice_width * arr_shape[0] / 2.0)
        output_img_length = float(input_slice_width * arr_shape[0] / self.output_slice_width)
        detector_width_cm = float(self.detector_width)
        detector_length_cm = (
            float(arr_shape[0] * input_slice_width)
            if self.detector_length == 0
            else float(self.detector_length)
        )
        return {
            "input_slice_width": input_slice_width,
            "input_pixel_width": input_pixel_width,
            "input_half_length": input_half_length,
            "output_img_length": output_img_length,
            "detector_width_cm": detector_width_cm,
            "detector_length_cm": detector_length_cm,
        }

    def _build_simind_switches(
        self,
        atn_name: str,
        act_name: str,
        arr_shape: tuple,
        geometry: Dict[str, float],
        scale_factor: float,
    ) -> str:
        """
        Build the SIMIND command-line switch string for a single organ simulation.

        Key switches
        ------------
        /fd  attenuation map filename
        /fs  activity source map filename
        /nn  photons per voxel (scaled by num_cores)
        /cc  collimator
        /fi  isotope
        /02,/05  input half-length (z extent)
        /08,/10  detector length and width
        /14,/15  energy window lower bounds
        /20,/21  energy window widths (signed negative = % of photopeak)
        /28  output pixel width
        /29  number of projections
        /31  input pixel width
        /34  input z pixels
        /42  detector-to-patient distance
        /76  output image size
        /77  output image length
        /78,/79  output z and x pixels
        """
        return (
            f"/fd:{atn_name}"
            f"/fs:{act_name}"
            f"/in:x22,3x"
            f"/nn:{scale_factor}"
            f"/cc:{self.collimator}"
            f"/fi:{self.isotope}"
            f"/02:{geometry['input_half_length']}"
            f"/05:{geometry['input_half_length']}"
            f"/08:{geometry['detector_length_cm']:.2f}"
            f"/10:{geometry['detector_width_cm']:.2f}"
            f"/14:-7"
            f"/15:-7"
            f"/20:{-1 * self.energy_window_width}"
            f"/21:{-1 * self.energy_window_width}"
            f"/28:{self.output_pixel_width}"
            f"/29:{self.num_projections}"
            f"/31:{geometry['input_pixel_width']}"
            f"/34:{arr_shape[0]}"
            f"/42:{self.detector_distance}"
            f"/76:{self.output_img_size}"
            f"/77:{geometry['output_img_length']}"
            f"/78:{arr_shape[1]}"
            f"/79:{arr_shape[2]}"
        )

    def _run_simind_for_organ_cores(self, organ_name: str, simind_switches: str) -> None:
        """
        Launch one SIMIND process per core in parallel for a single organ.

        Each process uses a unique /rr:<core_id> random seed so contributions are
        statistically independent. Core 0 outputs to stdout; others are silenced.
        """
        processes: List[subprocess.Popen] = []
        for j in range(self.num_cores):
            cmd = (
                f"{self.simind_exe} {self.prefix} {self.prefix}_{organ_name}_{j} "
                + simind_switches
                + f"/rr:{j}"
            )
            stdout = None if j == 0 else subprocess.DEVNULL
            processes.append(subprocess.Popen(cmd, shell=True, cwd=self.work_dir, stdout=stdout))

        for p in processes:
            p.wait()

    def _aggregate_core_totals_for_organ(self, organ_name: str) -> None:
        """
        Average projection totals across `num_cores` SIMIND runs and write to output_dir.

        Reads per-core files:  <work_dir>/<prefix>_<organ>_<j>_tot_w{1,2,3}.a00
        Writes averaged totals: <output_dir>/<prefix>_<organ>_tot_w{1,2,3}.a00

        Units after averaging: counts/MB/s (SIMIND convention).
        In PRODUCTION mode, per-core files are deleted to save disk space.
        """
        xtot_w1 = xtot_w2 = xtot_w3 = 0.0

        for j in range(self.num_cores):
            p1 = os.path.join(self.work_dir, f"{self.prefix}_{organ_name}_{j}_tot_w1.a00")
            p2 = os.path.join(self.work_dir, f"{self.prefix}_{organ_name}_{j}_tot_w2.a00")
            p3 = os.path.join(self.work_dir, f"{self.prefix}_{organ_name}_{j}_tot_w3.a00")

            xtot_w1 += np.fromfile(p1, dtype=np.float32)
            xtot_w2 += np.fromfile(p2, dtype=np.float32)
            xtot_w3 += np.fromfile(p3, dtype=np.float32)

            if self.mode == "PRODUCTION":
                for p in (p1, p2, p3):
                    try:
                        os.remove(p)
                    except FileNotFoundError:
                        pass

        # Average across cores
        xtot_w1 /= self.num_cores
        xtot_w2 /= self.num_cores
        xtot_w3 /= self.num_cores

        organ_paths = self._get_projection_paths_for_organ(organ_name)
        np.asarray(xtot_w1, dtype=np.float32).tofile(organ_paths["w1"])
        np.asarray(xtot_w2, dtype=np.float32).tofile(organ_paths["w2"])
        np.asarray(xtot_w3, dtype=np.float32).tofile(organ_paths["w3"])

    def _save_stage_metadata(
        self,
        simind_projection_paths: Dict[str, Dict[str, str]],
        roi_list: List[str],
        geometry: Dict[str, float],
        total_num_voxels: int,
        scale_factor: float,
        simind_switches_by_organ: Dict[str, str],
        organ_act_paths: Dict[str, str],
        atn_av_path: str,
    ) -> None:
        """Save stage-specific metadata for debugging / provenance."""
        metadata: Dict[str, Any] = {
            "stage": "simind_stage",
            "phase_output_dir": self.phase_output_dir,
            "output_dir": self.output_dir,
            "stage_output_dir": self.stage_output_dir,
            "work_dir": self.work_dir,
            "file_prefix": self.prefix,
            "simind_exe": self.simind_exe,
            "simind_dir": self.simind_dir,
            "collimator": self.collimator,
            "isotope": self.isotope,
            "num_projections": self.num_projections,
            "num_photons": self.num_photons,
            "detector_distance": self.detector_distance,
            "detector_width": self.detector_width,
            "detector_length": self.detector_length,
            "output_img_size": self.output_img_size,
            "output_pixel_width": self.output_pixel_width,
            "output_slice_width": self.output_slice_width,
            "energy_window_width": self.energy_window_width,
            "num_cores": self.num_cores,
            "roi_list": roi_list,
            "geometry": geometry,
            "total_num_voxels": total_num_voxels,
            "scale_factor": scale_factor,
            "simind_switches_by_organ": simind_switches_by_organ,
            "binary_roi_act_map_paths": organ_act_paths,
            "atn_av_path": atn_av_path,
            "simind_projection_paths": simind_projection_paths,
        }
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

    # -----------------------------
    # main
    # -----------------------------
    def run(self) -> Any:
        """
        Execute SIMIND simulation for all organs and save per-organ projection totals.

        Returns
        -------
        context : Context-like
        """
        self.context.require(
            "class_seg",
            "arr_shape_new",
            "arr_px_spacing_cm",
            "binary_roi_act_map_paths",
            "mask_roi_body",
            "atn_av_path",
        )

        class_seg = self.context.class_seg
        arr_shape = self.context.arr_shape_new
        arr_px_spacing_cm = self.context.arr_px_spacing_cm
        organ_act_paths = self.context.binary_roi_act_map_paths
        masks = self.context.mask_roi_body
        atn_av_path = self.context.atn_av_path

        if not os.path.exists(atn_av_path):
            raise FileNotFoundError(f"Attenuation map not found: {atn_av_path}")

        roi_list = [roi_name for roi_name in class_seg.keys() if roi_name in organ_act_paths]
        if not roi_list:
            raise ValueError("No ROI binary source maps found for SIMIND simulation.")

        simind_projection_paths = self._build_projection_path_dict(roi_list)
        geometry = self._get_input_geometry(arr_shape, arr_px_spacing_cm)

        self._set_simind_environment()
        self._copy_templates()

        # Copy attenuation map into work_dir (SIMIND resolves input paths relative to cwd).
        atn_work_name = f"{self.prefix}_atn_av.bin"
        atn_work_path = os.path.join(self.work_dir, atn_work_name)
        if not os.path.exists(atn_work_path):
            shutil.copyfile(atn_av_path, atn_work_path)

        # Scale factor: photons per source voxel per core.
        total_num_voxels = int(np.sum([np.sum(mask) for mask in masks.values()]))
        if total_num_voxels <= 0:
            raise ValueError("Total source voxels is zero; cannot compute SIMIND scale factor.")

        scale_factor = float(self.num_photons / total_num_voxels / self.num_cores)
        if scale_factor < 1:
            print(f"Not enough photons for this patient/num_cores. Requested: {self.num_photons}")
            print(f"Increasing to: {total_num_voxels * self.num_cores}")
            scale_factor = 1.0

        simind_switches_by_organ: Dict[str, str] = {}

        for organ_name in roi_list:
            act_path = organ_act_paths[organ_name]
            act_work_name = f"{self.prefix}_{organ_name}_act_av.bin"
            act_work_path = os.path.join(self.work_dir, act_work_name)
            if not os.path.exists(act_work_path):
                shutil.copyfile(act_path, act_work_path)

            if not os.path.exists(act_path):
                raise FileNotFoundError(f"Binary ROI source map not found: {act_path}")

            simind_switches = self._build_simind_switches(
                atn_name=atn_work_name,
                act_name=act_work_name,
                arr_shape=arr_shape,
                geometry=geometry,
                scale_factor=scale_factor,
            )
            simind_switches_by_organ[organ_name] = simind_switches

            if self._organ_totals_exist(organ_name) and self._organ_headers_exist(organ_name):
                continue

            self._run_simind_for_organ_cores(organ_name, simind_switches)
            self._aggregate_core_totals_for_organ(organ_name)

        self._run_jaszczak_calibration()

        self._save_stage_metadata(
            simind_projection_paths=simind_projection_paths,
            roi_list=roi_list,
            geometry=geometry,
            total_num_voxels=total_num_voxels,
            scale_factor=scale_factor,
            simind_switches_by_organ=simind_switches_by_organ,
            organ_act_paths=organ_act_paths,
            atn_av_path=atn_av_path,
        )

        self.context.spect_sim_output_dir = self.output_dir
        self.context.simind_stage_output_dir = self.stage_output_dir
        self.context.simind_work_dir = self.work_dir
        self.context.simind_metadata_path = self.metadata_path
        self.context.simind_calibration_path = self.calibration_path
        self.context.simind_projection_paths = simind_projection_paths
        self.context.simind_num_cores = self.num_cores
        self.context.simind_geometry = geometry
        self.context.simind_total_num_voxels = total_num_voxels
        self.context.simind_scale_factor = scale_factor
        self.context.simind_switches_by_organ = simind_switches_by_organ

        return self.context