"""
Pipeline Context container for the Theranostic Digital Twin (TDT) pipeline.

`Context` is a lightweight, mutable object used to pass configuration, runtime metadata,
and intermediate outputs between pipeline stages. All stage outputs are written here
so later stages can access them without needing file I/O for in-memory data.

Maintainer / contact: pyazdi@bccrc.ca
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class Context:
    """
    Shared pipeline state passed between stages.

    Attributes are grouped by pipeline phase.  Each stage reads fields it needs
    from this object and writes its outputs back, advancing the state for the
    next stage.
    """

    def __init__(self, logger: Optional[Any] = None) -> None:
        # Internal / compatibility
        self._logger: Optional[Any] = logger
        self._log_enabled: bool = True

        # Free-form storage for debugging, provenance, and stage metadata.
        self.extras: Dict[str, Any] = {}

        # ----------------------------- Initial setup -----------------------------
        self.mode: Optional[str] = None                          # "DEBUG" or "PRODUCTION"
        self.ct_input_path: Optional[str] = None                 # raw CT input (NIfTI or DICOM dir)
        self.ct_input_type: Optional[str] = None                 # "nii" or "dicom"
        self.ct_indx: Optional[int] = None                       # index in the batch (used for output naming)
        self.output_folder_path: Optional[str] = None            # root output folder for this CT
        self.subdir_paths: Optional[Dict[str, str]] = None       # {phase_key: abs path}
        self.subdir_names: Optional[Dict[str, str]] = None       # {phase_key: dir name}
        self.synthetic_lesions_enabled: Optional[bool] = None    # whether the synthetic lesions stage will run
        self.downstream_roi_subset: Optional[list[str]] = None   # ROI names propagated to all downstream stages
        self.run_spect: Optional[bool] = None                    
        self.run_dosimetry: Optional[bool] = None                
        self.run_postprocess: Optional[bool] = None              

        # Full config dict snapshot (deep-copied from parsed JSON at pipeline start).
        self.config: Dict[str, Any] = {}

        # ----------------------------- Phase 1: Digital Twin & Ground Truth ----------------------------- 
        # Stage 1.1: TotalSegmentator + ROI Unification (merged) 
        self.ct_nii_path: Optional[str] = None                          # standardised CT NIfTI
        self.body_ml_path: Optional[str] = None                         # body task multilabel mask
        self.head_glands_cavities_ml_path: Optional[str] = None         # head task multilabel mask
        self.total_ml_path: Optional[str] = None                        # total task multilabel mask
        self.totseg_plan: Optional[Dict[str, Any]] = None               # which tasks ran + ROI subsets
        self.tdt_roi_seg_path: Optional[str] = None                     # unified TDT multilabel NIfTI handoff 

        # --- Removed: Stage 1.2 ROI Unification was separate, now merged into segmentation_stage --- 

        # Stage 1.2: Synthetic Lesions (optional) 
        self.synthetic_lesions_outdir: Optional[str] = None
        self.synthetic_lesions_results: Optional[Dict[str, Any]] = None
        self.synthetic_lesions_backup_seg_path: Optional[str] = None
        self.synthetic_lesions_global_binary_path: Optional[str] = None
        self.synthetic_lesions_global_labels_path: Optional[str] = None

        # Stage 1.3: PBPK TAC Generation 
        self.pbpk_tacs_by_organ: Optional[Dict[str, Any]] = None        # {roi: {tac_time/values/...}}
        self.pbpk_tac_json_path: Optional[str] = None                   
        self.pbpk_tac_npz_path: Optional[str] = None                    
        self.pbpk_tac_time: Optional[Any] = None                        
        self.pbpk_tac_values: Optional[Dict[str, Any]] = None           
        self.pbpk_height_m: Optional[float] = None                      # patient height extracted from DICOM (m)
        self.pbpk_weight_kg: Optional[float] = None                     # patient weight extracted from DICOM (kg)
        self.pbpk_parameters: Optional[Dict[str, Any]] = None           # PyCNO parameter overrides used
        self.pbpk_vois: Optional[list[str]] = None                      

        # ----------------------------- Phase 2: Simulations ----------------------------- 
        # Stage 2.1: SIMIND Preprocessing + Simulation (merged) 
        self.body_seg_arr: Optional[Any] = None                         # float32 body mask in SIMIND grid
        self.roi_body_seg_arr: Optional[Any] = None                     # int16 labels in SIMIND grid (requested ROIs only)
        self.mask_roi_body: Optional[Dict[int, Any]] = None             # {label_id: bool mask}
        self.class_seg: Optional[Dict[str, int]] = None                 # {roi_name: label_id}
        self.atn_av_path: Optional[str] = None                          # SIMIND attenuation binary (.bin)
        self.binary_roi_act_map_paths: Optional[Dict[str, str]] = None  # {roi_name: binary source map path}
        self.arr_px_spacing_cm: Optional[Any] = None                    # (z, y, x) spacing in cm
        self.arr_shape_new: Optional[Any] = None                        # (z, y, x) array shape after optional resize

        self.spect_sim_output_dir: Optional[str] = None
        self.simind_stage_output_dir: Optional[str] = None
        self.simind_work_dir: Optional[str] = None
        self.simind_metadata_path: Optional[str] = None
        self.simind_calibration_path: Optional[str] = None
        self.simind_projection_paths: Optional[Dict[str, Any]] = None   # {roi_name: {w1/w2/w3: path}}
        self.simind_summed_projection_paths: Optional[Dict[str, str]] = None  
        self.simind_num_cores: Optional[int] = None
        self.simind_geometry: Optional[Dict[str, Any]] = None
        self.simind_total_num_voxels: Optional[int] = None
        self.simind_scale_factor: Optional[float] = None
        self.simind_switches_by_organ: Optional[Dict[str, str]] = None
        self.simind_header_dir: Optional[str] = None                    

        # Stage 2.2: OpenGATE Simulation 
        self.dosimetry_output_dir: Optional[str] = None
        self.dosimetry_stage_output_dir: Optional[str] = None
        self.dosimetry_work_dir: Optional[str] = None
        self.dosimetry_metadata_path: Optional[str] = None
        self.dosimetry_mask_paths: Optional[Dict[str, str]] = None          # {roi: source mask path}
        self.dosimetry_raw_dose_paths: Optional[Dict[str, str]] = None      # {roi: per-roi dose NIfTI}
        self.dosimetry_raw_uncertainty_paths: Optional[Dict[str, str]] = None
        self.dosimetry_sum_dose_path: Optional[str] = None                  # summed dose NIfTI across all ROIs
        self.dosimetry_material_label_path: Optional[str] = None            # material label image NIfTI

        # ----------------------------- Phase 3: Post-Processing ----------------------------- 
        # Stage 3.1: SPECT Post-Processing 
        self.pbpk_frame_start_times_min: Optional[Any] = None           # np.ndarray of frame start times (min)
        self.pbpk_frame_durations_s: Optional[Any] = None               # np.ndarray of frame durations (s)
        self.pbpk_projection_paths: Optional[Dict[str, Any]] = None     # {frame_label: {w1/w2/w3: path}}
        self.activity_map_sum: Optional[Any] = None                     # np.ndarray (n_frames,) total activity [MBq]
        self.activity_organ_sum: Optional[Dict[str, Any]] = None        # {roi: np.ndarray (n_frames,)} [MBq]
        self.activity_map_paths_by_organ: Optional[Any] = None          # list[str] paths to per-organ activity maps
        self.reconstruction_output_dir: Optional[str] = None

        # Stage 3.2: Dosimetry Post-Processing 
        self.dosemap_postprocess_output_dir: Optional[str] = None       
        self.dosemap_postprocess_paths: Optional[Dict[str, str]] = None 

        # --- Removed: Phase 4 dosimetry fields moved to Phase 2 Stage 2.2 above --- 

    def require(self, *names: str) -> None:
        """
        Assert that required Context fields are set (non-None).

        Call at the top of each stage's ``__init__`` or ``run`` to fail fast
        with a clear message if an upstream stage did not complete.

        Parameters
        ----------
        *names : str
            Attribute names that must exist on the Context and be non-None.

        Raises
        ------
        AttributeError
            If one or more required fields are missing or None.
        """
        missing = [n for n in names if not hasattr(self, n) or getattr(self, n) is None]
        if missing:
            raise AttributeError(f"Context missing required fields: {missing}")
