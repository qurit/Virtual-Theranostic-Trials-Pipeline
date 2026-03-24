"""
SPECT reconstruction stage for the Theranostic Digital Twin (TDT) pipeline.

Reconstructs quantitative SPECT images from PBPK-weighted windowed projection totals
using PyTomography (OSEM + TEW scatter correction).

Inputs (from previous stages)
------------------------------
- PBPK per-frame window totals: <pbpk_stage>/<prefix>_<t_hr>_tot_w{1,2,3}.nii.gz
- SIMIND headers in the simulation work dir: .h00, .cor (optional), .hct
- SIMIND calibration file: calib.res

Outputs
-------
- Reconstructed SPECT images (phase 3 root): reconstructed_SPECT_<t_hr>.nii.gz
- Attenuation map in recon grid (reconstruction stage dir): recon_atn_img.nii.gz
- Stage metadata (work dir): reconstruction_metadata.json

Maintainer / contact: pyazdi@bccrc.ca
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import os
import json
import numpy as np
import SimpleITK as sitk
import torch
import pytomography
import shutil

from pytomography.algorithms import OSEM
from pytomography.io.SPECT import simind
from pytomography.io.shared import get_header_value
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform


class SpectReconstructionStage:
    """
    Reconstruct SPECT images from PBPK-weighted totals using PyTomography.

    Parameters
    ----------
    context : Any
        Pipeline context. Must provide PBPK timing, projection paths, and SIMIND headers.
    """

    def __init__(self, context: Any) -> None:
        self.context = context

        self.phase_output_dir: str = context.subdir_paths["phase_3"]
        self.stage_cfg: Dict[str, Any] = context.config["phase_3"]["reconstruction_stage"]

        self.stage_output_dir: str = os.path.join(
            self.phase_output_dir,
            self.stage_cfg.get("sub_dir_name", "reconstruction_stage"),
        )
        os.makedirs(self.phase_output_dir, exist_ok=True)
        os.makedirs(self.stage_output_dir, exist_ok=True)

        self.work_dir: str = os.path.join(self.stage_output_dir, "work_dir")
        os.makedirs(self.work_dir, exist_ok=True)
        self.metadata_path: str = os.path.join(self.work_dir, "reconstruction_metadata.json")

        self.mode = context.mode

        self.pbpk_projection_paths: Dict[str, Dict[str, str]] = getattr(
            context, "pbpk_projection_paths", None
        )
        if self.pbpk_projection_paths is None:
            raise AttributeError("Context missing required field: pbpk_projection_paths")

        self.header_dir: str = getattr(context, "simind_work_dir", None)
        if self.header_dir is None:
            raise AttributeError("Context missing required field: simind_work_dir")

        self.calibration_file: str = getattr(context, "simind_calibration_path", None) or os.path.join(
            context.subdir_paths["phase_2"], "calib.res"
        )

        self.simind_prefix: str = context.config["phase_2"]["simind_stage"]["file_prefix"]
        self.output_pixel_width: float = context.config["phase_2"]["simind_stage"]["OutputPixelWidth"]
        self.output_slice_width: float = context.config["phase_2"]["simind_stage"]["OutputSliceWidth"]
        self.detector_distance: float = context.config["phase_2"]["simind_stage"]["DetectorDistance"]

        self.frame_start: Sequence[float] = context.config["phase_3"]["pbpk_stage"]["FrameStartTimes"]
        self.frame_durations: Sequence[float] = context.config["phase_3"]["pbpk_stage"]["FrameDurations"]

        self.iterations: int = context.config["phase_3"]["reconstruction_stage"]["Iterations"]
        self.subsets: int = context.config["phase_3"]["reconstruction_stage"]["Subsets"]

        # (x, y, z) spacing tuple for SimpleITK images built from arrays
        self.output_tuple: Tuple[float, float, float] = (
            self.output_pixel_width,
            self.output_pixel_width,
            self.output_slice_width,
        )

        self.recon_algorithm: str = context.config["phase_3"]["reconstruction_stage"]["ReconstructionAlgorithm"]

    # -----------------------------
    # helpers
    # -----------------------------

    def _get_sensitivity_from_calibration_file(self, calibration_file: str) -> float:
        """
        Read sensitivity (counts/s/MBq) from SIMIND `calib.res`.

        SIMIND writes sensitivity on a fixed line (index 70, 0-based).
        """
        with open(calibration_file, "r") as f:
            lines = f.readlines()
        return float(lines[70].strip().split(":")[-1].strip().split()[0])

    def _get_metadata_from_header(
        self, photopeak_path: str, lower_path: str, upper_path: str
    ) -> Tuple[int, int, int, float, float, float]:
        """
        Parse SIMIND header files and return projection dimensions + energy window widths.

        Returns
        -------
        (proj_dim1, proj_dim2, num_proj, ww_peak, ww_lower, ww_upper)
        """
        with open(photopeak_path, "r") as f:
            headerdata = np.array(f.readlines())

        proj_dim1 = get_header_value(headerdata, "matrix size [1]", int)
        proj_dim2 = get_header_value(headerdata, "matrix size [2]", int)
        num_proj = get_header_value(headerdata, "total number of images", int)

        ww_peak, ww_lower, ww_upper = [
            simind.get_energy_window_width(p) for p in (photopeak_path, lower_path, upper_path)
        ]
        return proj_dim1, proj_dim2, num_proj, ww_peak, ww_lower, ww_upper

    def _get_cor_data(self, cor_path: str) -> np.ndarray:
        """Load COR file and sanitize to 1D if needed (writes back to disk in place)."""
        cor_data = np.loadtxt(cor_path).astype(float)
        if cor_data.ndim == 2:
            cor_data = cor_data[:, 0]
            np.savetxt(cor_path, cor_data)
        return cor_data

    def _read_projection_nifti(self, projection_path: str) -> np.ndarray:
        """Read a PBPK projection NIfTI and return as float32 array."""
        if not os.path.exists(projection_path):
            raise FileNotFoundError(f"Missing projection file: {projection_path}")
        return sitk.GetArrayFromImage(sitk.ReadImage(projection_path)).astype(np.float32)

    def _convert_counts_to_mbq_per_ml(
        self,
        reconstructed_image: torch.Tensor,
        sensitivity: float,
        frame_duration: float,
    ) -> torch.Tensor:
        """
        Convert reconstructed counts to MBq/mL using SIMIND sensitivity and voxel geometry.

        counts / (counts/s/MBq) / s / cm^2 / cm = MBq/cm^3 = MBq/mL
        """
        return (
            reconstructed_image.cpu().T
            / sensitivity
            / frame_duration
            / (self.output_pixel_width ** 2)
            / self.output_slice_width
        )

    def _get_recon_img(
        self,
        likelihood: PoissonLogLikelihood,
        sensitivity: float,
        frame_duration: float,
    ) -> sitk.Image:
        """Run the configured reconstruction algorithm and return a SimpleITK image (MBq/mL)."""
        if self.recon_algorithm.lower() == "osem":
            recon_algorithm = OSEM(likelihood)
        else:
            raise ValueError(f"Unsupported reconstruction algorithm: {self.recon_algorithm}")

        reconstructed_image = recon_algorithm(n_iters=self.iterations, n_subsets=self.subsets)
        recon_arr = self._convert_counts_to_mbq_per_ml(reconstructed_image, sensitivity, frame_duration)

        recon_img = sitk.GetImageFromArray(recon_arr)
        recon_img.SetSpacing(self.output_tuple)
        return recon_img

    def _write_recon_atn_img(self, amap: torch.Tensor) -> Tuple[sitk.Image, str]:
        """Write the attenuation map (from SIMIND .hct header) as a NIfTI in recon grid."""
        recon_atn_path = os.path.join(self.stage_output_dir, "recon_atn_img.nii.gz")
        if os.path.exists(recon_atn_path):
            return sitk.ReadImage(recon_atn_path), recon_atn_path

        recon_atn_img = sitk.GetImageFromArray(amap.cpu().T)
        recon_atn_img.SetSpacing(self.output_tuple)
        sitk.WriteImage(recon_atn_img, recon_atn_path, imageIO="NiftiImageIO")
        return recon_atn_img, recon_atn_path

    def _save_stage_metadata(
        self,
        recon_paths: Dict[str, str],
        recon_atn_path: str,
        sensitivity: float,
        photopeak_h: str,
        lower_h: str,
        upper_h: str,
        cor_path: Optional[str],
        proj_dim1: int,
        proj_dim2: int,
        num_proj: int,
        ww_peak: float,
        ww_lower: float,
        ww_upper: float,
    ) -> None:
        """Save reconstruction-stage metadata for debugging / provenance."""
        metadata: Dict[str, Any] = {
            "stage": "reconstruction_stage",
            "phase_output_dir": self.phase_output_dir,
            "stage_output_dir": self.stage_output_dir,
            "work_dir": self.work_dir,
            "pbpk_projection_paths": self.pbpk_projection_paths,
            "reconstructed_spect_paths": recon_paths,
            "recon_atn_path": recon_atn_path,
            "calibration_file": self.calibration_file,
            "sensitivity_counts_per_s_per_mbq": float(sensitivity),
            "header_dir": self.header_dir,
            "photopeak_h00": photopeak_h,
            "lower_h00": lower_h,
            "upper_h00": upper_h,
            "cor_path": cor_path,
            "iterations": int(self.iterations),
            "subsets": int(self.subsets),
            "reconstruction_algorithm": self.recon_algorithm,
            "output_pixel_width_cm": float(self.output_pixel_width),
            "output_slice_width_cm": float(self.output_slice_width),
            "detector_distance_cm": float(self.detector_distance),
            "proj_dim1": int(proj_dim1),
            "proj_dim2": int(proj_dim2),
            "num_proj": int(num_proj),
            "energy_window_widths": {
                "peak": float(ww_peak),
                "lower": float(ww_lower),
                "upper": float(ww_upper),
            },
            "frame_start_times_min": np.asarray(self.frame_start, dtype=float).tolist(),
            "frame_durations_s": np.asarray(self.frame_durations, dtype=float).tolist(),
        }
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

    # -----------------------------
    # main
    # -----------------------------
    def run(self) -> Any:
        """
        Run reconstruction for each PBPK frame and write NIfTI recon outputs.

        Returns
        -------
        context : Context-like
        """
        self.context.require("class_seg", "pbpk_projection_paths", "simind_work_dir")
        roi_list = list(self.context.class_seg.keys())

        if not os.path.exists(self.calibration_file):
            raise FileNotFoundError(f"Calibration file not found: {self.calibration_file}")
        sensitivity = self._get_sensitivity_from_calibration_file(self.calibration_file)

        # Locate SIMIND headers from phase-2 work_dir using the first ROI as a representative.
        photopeak_h = os.path.join(self.header_dir, f"{self.simind_prefix}_{roi_list[0]}_0_tot_w2.h00")
        lower_h = os.path.join(self.header_dir, f"{self.simind_prefix}_{roi_list[0]}_0_tot_w1.h00")
        upper_h = os.path.join(self.header_dir, f"{self.simind_prefix}_{roi_list[0]}_0_tot_w3.h00")

        for p in (photopeak_h, lower_h, upper_h):
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing SIMIND header file: {p}")

        cor_path: Optional[str] = None
        if self.detector_distance < 0:
            # Patient contouring enabled (negative detector distance) -> COR file required.
            cor_path = os.path.join(self.header_dir, f"{self.simind_prefix}_{roi_list[0]}_0.cor")
            if not os.path.exists(cor_path):
                raise FileNotFoundError(f"Missing COR file: {cor_path}")
            _ = self._get_cor_data(cor_path)
            object_meta, proj_meta = simind.get_metadata(photopeak_h, cor_path)
        else:
            object_meta, proj_meta = simind.get_metadata(photopeak_h) 

        proj_dim1, proj_dim2, num_proj, ww_peak, ww_lower, ww_upper = self._get_metadata_from_header(
            photopeak_h, lower_h, upper_h
        )

        path_amap = os.path.join(self.header_dir, f"{self.simind_prefix}_{roi_list[0]}_0.hct")
        if not os.path.exists(path_amap):
            raise FileNotFoundError(f"Missing attenuation map header (.hct): {path_amap}")

        amap = simind.get_attenuation_map(path_amap)

        att_transform = SPECTAttenuationTransform(amap)
        psf_meta = simind.get_psfmeta_from_header(photopeak_h)
        psf_transform = SPECTPSFTransform(psf_meta)

        system_matrix = SPECTSystemMatrix(
            obj2obj_transforms=[att_transform, psf_transform],
            proj2proj_transforms=[],
            object_meta=object_meta,
            proj_meta=proj_meta,
        )

        recon_paths: Dict[str, str] = {}
        pbpk_keys = [f"{float(t)/60.0:.6f}".rstrip("0").rstrip(".") for t in self.frame_start]

        for time_index, time in enumerate(self.frame_start):
            frame_label = f"{float(time)/60.0:.6f}".rstrip("0").rstrip(".")
            recon_output_path = os.path.join(
                self.phase_output_dir, f"reconstructed_SPECT_{frame_label}.nii.gz"
            )

            if os.path.exists(recon_output_path):
                recon_paths[frame_label] = recon_output_path
                continue

            if frame_label not in self.pbpk_projection_paths:
                raise KeyError(f"Missing PBPK projection paths for frame: {frame_label}")

            lower = torch.tensor(
                self._read_projection_nifti(self.pbpk_projection_paths[frame_label]["w1"]).copy()
            ).to(pytomography.device)
            photopeak = torch.tensor(
                self._read_projection_nifti(self.pbpk_projection_paths[frame_label]["w2"]).copy()
            ).to(pytomography.device)
            upper = torch.tensor(
                self._read_projection_nifti(self.pbpk_projection_paths[frame_label]["w3"]).copy()
            ).to(pytomography.device)

            # Add Poisson noise to produce one stochastic realization per frame.
            photopeak_realization = torch.poisson(photopeak)
            lower_realization = torch.poisson(lower)
            upper_realization = torch.poisson(upper)

            scatter_estimate_tew = simind.compute_EW_scatter(
                lower_realization, upper_realization, ww_lower, ww_upper, ww_peak
            )

            likelihood = PoissonLogLikelihood(
                system_matrix=system_matrix,
                projections=photopeak_realization,
                additive_term=scatter_estimate_tew,
            )

            recon_img = self._get_recon_img(likelihood, sensitivity, self.frame_durations[time_index])
            sitk.WriteImage(recon_img, recon_output_path, imageIO="NiftiImageIO")
            recon_paths[frame_label] = recon_output_path

        _, recon_atn_path = self._write_recon_atn_img(amap)

        self._save_stage_metadata(
            recon_paths=recon_paths,
            recon_atn_path=recon_atn_path,
            sensitivity=sensitivity,
            photopeak_h=photopeak_h,
            lower_h=lower_h,
            upper_h=upper_h,
            cor_path=cor_path,
            proj_dim1=proj_dim1,
            proj_dim2=proj_dim2,
            num_proj=num_proj,
            ww_peak=ww_peak,
            ww_lower=ww_lower,
            ww_upper=ww_upper,
        )

        # In PRODUCTION mode, clean up the SIMIND work_dir once all frames are reconstructed.
        all_frames_exist = all(
            os.path.exists(os.path.join(self.phase_output_dir, f"reconstructed_SPECT_{f}.nii.gz"))
            for f in pbpk_keys
        )
        if self.mode == "PRODUCTION" and all_frames_exist:
            work_dir = getattr(self.context, "simind_work_dir", None)
            if work_dir:
                work_dir_abs = os.path.abspath(work_dir)
                out_abs = os.path.abspath(self.phase_output_dir)
                if work_dir_abs != out_abs and os.path.exists(work_dir_abs):
                    shutil.rmtree(work_dir_abs, ignore_errors=True)

        self.context.reconstruction_output_dir = self.phase_output_dir
        return self.context