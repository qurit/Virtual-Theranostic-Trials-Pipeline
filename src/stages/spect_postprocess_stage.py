"""
SPECT post-processing for the VTT pipeline.

This stage combines PBPK TAC weighting of SIMIND projections with OSEM+TEW
reconstruction using PyTomography.

Core responsibilities
---------------------
- For each ROI present in SIMIND projections, read per-organ TAC from Phase 1 PBPK.
- Fold any PBPK organ TACs that are absent from SIMIND's roi_subset into the
  remaining_body TAC so that activity from excluded organs is not lost.
- Interpolate TAC values at configured frame start times.
- Build per-frame activity maps and apply TAC-derived activity to SIMIND projections.
- Optionally apply Poisson noise to projections.
- Optionally reconstruct quantitative SPECT images via OSEM + TEW scatter correction.
- Save PBPK-weighted projections and reconstructed SPECT images.

Controlled by config flags:
- apply_tac: whether to weight projections by PBPK TACs
- apply_poisson_noise: whether to add Poisson noise
- apply_reconstruction: whether to run PyTomography OSEM+TEW
- apply_frame_duration: whether to scale projections by frame duration

Expected Context interface
--------------------------
Incoming `context` is expected to provide:
- context.subdir_paths["phase_3"] : str
- context.config["phase_3"]["spect_postprocess_stage"] : dict
- context.config["phase_2"]["simind_stage"] : dict
- context.simind_projection_paths : dict
- context.simind_header_dir : str
- context.simind_calibration_path : str
- context.pbpk_tac_time : np.ndarray
- context.pbpk_tac_values : dict[str, np.ndarray]
- context.pbpk_vois : list[str]
- context.roi_body_seg_arr, context.mask_roi_body, context.class_seg
- context.arr_px_spacing_cm
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import os
import json
import numpy as np
import SimpleITK as sitk
import torch
import pytomography

from src.io.rerun_guard import (
    assert_stage_rerun_safe,
    build_stage_metadata,
    build_spect_rerun_snapshot,
    fingerprint_optional_file,
    stage_metadata_path,
    write_json,
)
from src.utils.nifti_utils import voxel_volume_ml
from src.utils.simind_projection_utils import (
    build_frame_projection_paths,
    load_cor_data,
    read_header_dims_and_windows,
    read_projection_bin,
    read_projection_nifti,
    read_sensitivity_from_calibration_file,
    read_simind_header_dims,
    write_activity_map_nifti,
    write_projection_nifti,
)

from pytomography.algorithms import OSEM
from pytomography.io.SPECT import simind
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform


class SpectPostprocessStage:
    """
    SPECT post-processing: TAC weighting + Poisson noise + OSEM+TEW reconstruction.
    """

    def __init__(self, context: Any) -> None:
        context.require(
            "subdir_paths",
            "config",
            "output_folder_path",
            "ct_input_identity",
        )
        self.context = context
        self.debug: bool = getattr(context, "mode", "").upper() == "DEBUG"             

        self.phase_output_dir: str = context.subdir_paths["phase_3"]
        self.stage_cfg: Dict[str, Any] = context.config["phase_3"]["spect_postprocess_stage"] 

        self.output_dir: str = os.path.join(
            self.phase_output_dir,
            self.stage_cfg.get("sub_dir_name", "spect_postprocess"),                   
        )
        os.makedirs(self.output_dir, exist_ok=True)
        self.work_dir: str = os.path.join(self.output_dir, "work_dir")
        os.makedirs(self.work_dir, exist_ok=True)
        self.metadata_path: str = stage_metadata_path(context.output_folder_path, "spect_postprocess_stage")
        self.ct_input_identity: Dict[str, Any] = context.ct_input_identity

        self.prefix: str = self.stage_cfg.get("file_prefix", "spect_postprocess")      
        self.mode: str = context.mode

        # Post-processing control flags
        self.apply_tac: bool = bool(self.stage_cfg.get("apply_tac", True))
        self.apply_poisson_noise: bool = bool(self.stage_cfg.get("apply_poisson_noise", True))
        self.apply_reconstruction: bool = bool(self.stage_cfg.get("apply_reconstruction", True))
        # NOTE: apply_frame_duration MUST be True for correct units.
        # SIMIND projections are in counts/MBq/s; organ_sum is in MBq.
        # Without the frame duration (seconds), the weighted projections are count
        # rates (counts/s) rather than counts — reconstruction would receive wrong inputs.
        self.apply_frame_duration: bool = bool(self.stage_cfg.get("apply_frame_duration", True))

        # Frame timing from post-process config 
        self.frame_start: Sequence[float] = self.stage_cfg["FrameStartTimes"]          
        self.frame_durations_s: Sequence[float] = self.stage_cfg["FrameDurations"]     

        # SIMIND config for projection reading 
        self.simind_cfg: Dict[str, Any] = context.config["phase_2"]["simind_stage"]    
        self.simind_num_projections: int = int(self.simind_cfg["NumProjections"])
        self.simind_output_img_size: int = int(self.simind_cfg["OutputImgSize"])
        self.simind_output_pixel_width_cm: float = float(self.simind_cfg["OutputPixelWidth"])
        self.simind_output_slice_width_cm: float = float(self.simind_cfg["OutputSliceWidth"])
        self.simind_prefix: str = str(self.simind_cfg["file_prefix"])
        self.detector_distance: float = float(self.simind_cfg["DetectorDistance"])      

        # Header dir (survives PRODUCTION cleanup) 
        self.header_dir: Optional[str] = getattr(context, "simind_header_dir", None)   
        if self.header_dir is None:                                                    
            self.header_dir = getattr(context, "simind_work_dir", None)                

        self.calibration_file: str = getattr(context, "simind_calibration_path", None) or os.path.join(
            context.subdir_paths["phase_2"], "calib.res"                               
        )

        self.proj_dim1: Optional[int] = None
        self.proj_dim2: Optional[int] = None
        self.num_proj: Optional[int] = None

        # Reconstruction settings
        self.iterations: int = int(self.stage_cfg.get("Iterations", 4))                
        self.subsets: int = int(self.stage_cfg.get("Subsets", 8))                       
        self.recon_algorithm: str = self.stage_cfg.get("ReconstructionAlgorithm", "OSEM") 

        # (x, y, z) spacing tuple for SimpleITK images
        self.output_tuple: Tuple[float, float, float] = (
            self.simind_output_pixel_width_cm,
            self.simind_output_pixel_width_cm,
            self.simind_output_slice_width_cm,
        )

    def _frame_label(self, i: int) -> str:
        """Return a compact minute-valued string label for frame index i."""
        return f"{float(self.frame_start[i]) / 60.0:.6f}".rstrip("0").rstrip(".")

    def _rerun_config_snapshot(self) -> Dict[str, Any]:
        """Return the SPECT post-processing settings that must match for reruns."""
        return build_spect_rerun_snapshot(self.context.config)

    def _current_dependency_fingerprints(self) -> Dict[str, Any]:
        """Return fingerprints for the upstream artifacts that feed this stage."""
        return {
            "simind_stage_metadata": fingerprint_optional_file(self.context.simind_metadata_path),
            "pbpk_tac_json": fingerprint_optional_file(self.context.pbpk_tac_json_path),
            "pbpk_tac_npz": fingerprint_optional_file(self.context.pbpk_tac_npz_path),
        }

    def _cache_marker_paths(self) -> List[str]:
        """Return representative output files whose presence means rerun metadata must match."""
        markers: List[str] = []
        for window_paths in build_frame_projection_paths(self.output_dir, self.prefix, self.frame_start).values():
            markers.extend(window_paths.values())
        if self.apply_reconstruction:
            for i in range(len(self.frame_start)):
                markers.append(os.path.join(self.phase_output_dir, f"reconstructed_SPECT_{self._frame_label(i)}.nii.gz"))
            markers.append(os.path.join(self.output_dir, "recon_atn_img.nii.gz"))
        return markers

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
            / (self.simind_output_pixel_width_cm ** 2)
            / self.simind_output_slice_width_cm
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
        recon_atn_path = os.path.join(self.output_dir, "recon_atn_img.nii.gz")         
        if os.path.exists(recon_atn_path):
            return sitk.ReadImage(recon_atn_path), recon_atn_path

        recon_atn_img = sitk.GetImageFromArray(amap.cpu().T)
        recon_atn_img.SetSpacing(self.output_tuple)
        sitk.WriteImage(recon_atn_img, recon_atn_path, imageIO="NiftiImageIO")
        return recon_atn_img, recon_atn_path

    def _save_stage_metadata(                                                          
        self,
        pbpk_projection_paths: Dict[str, Dict[str, str]],
        recon_paths: Dict[str, str],
    ) -> None:
        """Save post-processing stage metadata for debugging / provenance."""
        outputs: Dict[str, Any] = {
            "pbpk_projection_paths": pbpk_projection_paths,
            "reconstructed_spect_paths": recon_paths,
        }
        extra: Dict[str, Any] = {
            "stage": "spect_postprocess_stage",                                        
            "phase_output_dir": self.phase_output_dir,
            "output_dir": self.output_dir,
            "work_dir": self.work_dir,
            "file_prefix": self.prefix,
            "apply_tac": self.apply_tac,                                               
            "apply_poisson_noise": self.apply_poisson_noise,                           
            "apply_reconstruction": self.apply_reconstruction,                         
            "apply_frame_duration": self.apply_frame_duration,                         
            "frame_start_times_min": np.asarray(self.frame_start, dtype=float).tolist(),
            "frame_durations_s": np.asarray(self.frame_durations_s, dtype=float).tolist(),
            "iterations": self.iterations,                                             
            "subsets": self.subsets,                                                    
            "reconstruction_algorithm": self.recon_algorithm,                          
            "pbpk_projection_paths": pbpk_projection_paths,                            
            "reconstructed_spect_paths": recon_paths,                                  
            "header_dir": self.header_dir,                                             
            "calibration_file": self.calibration_file,                                 
        }
        metadata = build_stage_metadata(
            stage_name="spect_postprocess_stage",
            config_snapshot=self._rerun_config_snapshot(),
            ct_identity=self.ct_input_identity,
            upstream_fingerprints=self._current_dependency_fingerprints(),
            outputs=outputs,
            extra=extra,
        )
        write_json(self.metadata_path, metadata)

    # ------------------------------------------------------------------
    # main
    # ------------------------------------------------------------------
    def run(self) -> Any:
        """
        Execute SPECT post-processing: TAC weighting + optional reconstruction.

        Returns
        -------
        context : Context-like
        """
        assert_stage_rerun_safe(
            stage_name="spect_postprocess_stage",
            metadata_path=self.metadata_path,
            required_outputs=self._cache_marker_paths(),
            current_config_snapshot=self._rerun_config_snapshot(),
            current_ct_identity=self.ct_input_identity,
            current_upstream_fingerprints=self._current_dependency_fingerprints(),
            context=self.context,
        )
        if self.context.stage_skipped:
            self.context.reconstruction_output_dir = self.phase_output_dir
            return self.context

        self.context.require(
            "class_seg",
            "arr_px_spacing_cm",
            "simind_projection_paths",
            "simind_header_dir",
            "pbpk_tac_time",
            "pbpk_tac_values",
            "mask_roi_body",
            "roi_body_seg_arr",
        )

        # Remove "background" label if present (not a real ROI).  
        class_seg = {k: v for k, v in self.context.class_seg.items() if k != "background"}
        voxel_vol_ml = voxel_volume_ml(self.context.arr_px_spacing_cm)

        roi_list = list(self.context.simind_projection_paths.keys())
        if not roi_list:
            raise ValueError("No phase-2 SIMIND projection paths found in context.")

        # Locate SIMIND headers from header_dir 
        photopeak_h = os.path.join(self.header_dir, f"{self.simind_prefix}_{roi_list[0]}_0_tot_w2.h00") 
        lower_h = os.path.join(self.header_dir, f"{self.simind_prefix}_{roi_list[0]}_0_tot_w1.h00") 
        upper_h = os.path.join(self.header_dir, f"{self.simind_prefix}_{roi_list[0]}_0_tot_w3.h00") 
        self.proj_dim1, self.proj_dim2, self.num_proj = read_simind_header_dims(
            photopeak_h, lower_h, upper_h
        )

        # Get TAC data from Phase 1
        tac_time = self.context.pbpk_tac_time
        # Work on a local copy so we can safely patch remaining_body without
        # mutating context for other stages.
        tac_values: Dict[str, np.ndarray] = dict(self.context.pbpk_tac_values)

        # Aggregate excluded-organ TACs into remaining_body.
        # SIMIND's roi_subset may be smaller than Phase 1's roi_subset.  Any organ
        # that was modelled explicitly in PBPK but is NOT present in this simulation's
        # class_seg is geometrically absorbed into SIMIND's remaining_body mask.
        # Its activity must therefore be folded into the remaining_body TAC so that
        # the total activity assigned to that mask is correct.
        simulation_rois: set = set(class_seg.keys()) - {"remaining_body"}
        if "remaining_body" in tac_values:
            effective_rb = tac_values["remaining_body"].copy().astype(np.float64)
            for _rn, _rt in self.context.pbpk_tac_values.items():
                if _rn != "remaining_body" and _rn not in simulation_rois:
                    effective_rb += np.asarray(_rt, dtype=np.float64)
                    if self.debug:
                        print(
                            f"[SpectPostprocessStage] Folding excluded ROI '{_rn}' TAC "
                            "into remaining_body (not in SIMIND roi_subset)."
                        )
            tac_values["remaining_body"] = effective_rb.astype(np.float32)

        n_frames = len(self.frame_start)
        roi_body_seg_arr = self.context.roi_body_seg_arr                               
        mask_roi_body = self.context.mask_roi_body                                     

        activity_map = np.zeros((n_frames, *roi_body_seg_arr.shape), dtype=np.float32) 
        activity_organ_sum: Dict[str, np.ndarray] = {}
        organ_paths: List[str] = []
        pbpk_projection_paths = build_frame_projection_paths(self.output_dir, self.prefix, self.frame_start)
        frame_projection_sums: Dict[str, Dict[str, Optional[np.ndarray]]] = {
            label: {"w1": None, "w2": None, "w3": None} for label in pbpk_projection_paths
        }

        for roi_name, label_value in class_seg.items():
            # Get TAC for this ROI 
            if roi_name not in tac_values:                                             
                if self.debug:                                                         
                    print(f"[SpectPostprocessStage] No TAC for ROI '{roi_name}', skipping.") 
                continue                                                               

            tac_voi = tac_values[roi_name]                                             
            tac_interp = np.interp(np.asarray(self.frame_start, dtype=float), tac_time, tac_voi) 

            mask = mask_roi_body[int(label_value)]                                     
            n_vox = int(np.sum(mask))                                                  
            if n_vox == 0:                                                             
                continue                                                               

            # Build per-frame activity map for this ROI 
            activity_map_organ = np.zeros((n_frames, *roi_body_seg_arr.shape), dtype=np.float32) 
            activity_map_organ[:, mask] = tac_interp[:, None] / (n_vox * voxel_vol_ml) 
            activity_map[:, mask] = activity_map_organ[:, mask]                        

            organ_sum = np.sum(activity_map_organ, axis=(1, 2, 3)) * voxel_vol_ml     
            activity_organ_sum[roi_name] = organ_sum                                   

            # Save first-frame activity map for provenance 
            organ_map_path = os.path.join(self.work_dir, f"{self.prefix}_{roi_name}_act_av.nii.gz") 
            write_activity_map_nifti(activity_map_organ[0], self.context.arr_px_spacing_cm, organ_map_path)
            organ_paths.append(organ_map_path)                                         

            if roi_name not in self.context.simind_projection_paths:
                raise KeyError(f"Missing phase-2 SIMIND projections for ROI: {roi_name}")

            roi_projection_paths = self.context.simind_projection_paths[roi_name]
            for i in range(n_frames):
                frame_label = self._frame_label(i)
                # frame_scale: counts = (counts/MBq/s) * MBq * s
                frame_scale = float(organ_sum[i])                                      
                if self.apply_frame_duration:                                          
                    frame_scale *= float(self.frame_durations_s[i])                    
                for window_key in ("w1", "w2", "w3"):
                    proj_path = roi_projection_paths[window_key]
                    if not os.path.exists(proj_path):
                        raise FileNotFoundError(f"SIMIND projection not found: {proj_path}")
                    proj_arr = read_projection_bin(
                        proj_path,
                        proj_dim1=self.proj_dim1,
                        proj_dim2=self.proj_dim2,
                        num_proj=self.num_proj,
                    )
                    if self.apply_tac:                                                 
                        weighted = np.asarray(proj_arr * frame_scale, dtype=np.float32) 
                    else:                                                              
                        weighted = proj_arr.copy()                                     
                    if frame_projection_sums[frame_label][window_key] is None:
                        frame_projection_sums[frame_label][window_key] = weighted
                    else:
                        frame_projection_sums[frame_label][window_key] += weighted

        activity_map_sum = np.sum(activity_map, axis=(1, 2, 3)) * voxel_vol_ml

        # Write PBPK-weighted projection NIfTIs
        for frame_label, window_dict in frame_projection_sums.items():
            for window_key, proj_arr in window_dict.items():
                if proj_arr is None:
                    raise ValueError(
                        f"No PBPK-weighted projection generated for {frame_label} {window_key}"
                    )
                write_projection_nifti(
                    proj_arr,
                    self.simind_output_pixel_width_cm,
                    pbpk_projection_paths[frame_label][window_key],
                )

        recon_paths: Dict[str, str] = {}

        if self.apply_reconstruction:                                                  
            if not os.path.exists(self.calibration_file):
                raise FileNotFoundError(f"Calibration file not found: {self.calibration_file}")
            sensitivity = read_sensitivity_from_calibration_file(self.calibration_file)

            for p in (photopeak_h, lower_h, upper_h):
                if not os.path.exists(p):
                    raise FileNotFoundError(f"Missing SIMIND header file: {p}")

            cor_path: Optional[str] = None
            if self.detector_distance < 0:
                cor_path = os.path.join(self.header_dir, f"{self.simind_prefix}_{roi_list[0]}_0.cor") 
                if not os.path.exists(cor_path):
                    raise FileNotFoundError(f"Missing COR file: {cor_path}")
                _ = load_cor_data(cor_path)
                object_meta, proj_meta = simind.get_metadata(photopeak_h, cor_path)
            else:
                object_meta, proj_meta = simind.get_metadata(photopeak_h) 

            _, _, _, ww_peak, ww_lower, ww_upper = read_header_dims_and_windows(
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

            for time_index, time_val in enumerate(self.frame_start):
                frame_label = self._frame_label(time_index)
                recon_output_path = os.path.join(
                    self.phase_output_dir, f"reconstructed_SPECT_{frame_label}.nii.gz"
                )

                if os.path.exists(recon_output_path):
                    recon_paths[frame_label] = recon_output_path
                    continue

                if frame_label not in pbpk_projection_paths:
                    raise KeyError(f"Missing PBPK projection paths for frame: {frame_label}")

                lower = torch.tensor(
                    read_projection_nifti(pbpk_projection_paths[frame_label]["w1"]).copy()
                ).to(pytomography.device)
                photopeak = torch.tensor(
                    read_projection_nifti(pbpk_projection_paths[frame_label]["w2"]).copy()
                ).to(pytomography.device)
                upper = torch.tensor(
                    read_projection_nifti(pbpk_projection_paths[frame_label]["w3"]).copy()
                ).to(pytomography.device)

                # Add Poisson noise to produce one stochastic realization per frame.
                if self.apply_poisson_noise:                                           
                    photopeak_realization = torch.poisson(photopeak)                    
                    lower_realization = torch.poisson(lower)                            
                    upper_realization = torch.poisson(upper)                            
                else:                                                                  
                    photopeak_realization = photopeak                                   
                    lower_realization = lower                                           
                    upper_realization = upper                                           

                scatter_estimate_tew = simind.compute_EW_scatter(
                    lower_realization, upper_realization, ww_lower, ww_upper, ww_peak
                )

                likelihood = PoissonLogLikelihood(
                    system_matrix=system_matrix,
                    projections=photopeak_realization,
                    additive_term=scatter_estimate_tew,
                )

                recon_img = self._get_recon_img(likelihood, sensitivity, self.frame_durations_s[time_index]) 
                sitk.WriteImage(recon_img, recon_output_path, imageIO="NiftiImageIO")
                recon_paths[frame_label] = recon_output_path

            _, recon_atn_path = self._write_recon_atn_img(amap)

        self._save_stage_metadata(
            pbpk_projection_paths=pbpk_projection_paths,
            recon_paths=recon_paths,
        )

        self.context.activity_map_sum = activity_map_sum
        self.context.activity_organ_sum = activity_organ_sum
        self.context.activity_map_paths_by_organ = organ_paths
        self.context.pbpk_frame_start_times_min = np.asarray(self.frame_start, dtype=float)
        self.context.pbpk_frame_durations_s = np.asarray(self.frame_durations_s, dtype=float)
        self.context.pbpk_projection_paths = pbpk_projection_paths
        self.context.reconstruction_output_dir = self.phase_output_dir

        return self.context
