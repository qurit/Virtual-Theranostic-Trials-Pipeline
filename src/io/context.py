"""
Shared runtime context for the PyTheraTwin pipeline.

`Context` is the handoff object passed between stages. It stores the parsed
configuration, run metadata, and stage outputs so later phases can reuse
intermediate results without rebuilding state.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional


class Context:
    """
    Shared pipeline state passed between stages.

    Attributes are grouped by pipeline phase.  Each stage reads fields it needs
    from this object and writes its outputs back, advancing the state for the
    next stage.
    """

    def __init__(self, logger: Optional[Any] = None) -> None:
        # Internal state
        self._logger: Optional[Any] = logger
        self._log_enabled: bool = True

        # Free-form storage for debugging, provenance, and stage metadata.
        self.extras: Dict[str, Any] = {}

        # Set to True by assert_stage_rerun_safe when a stage is fully skipped.
        self.stage_skipped: bool = False

        # Runtime metadata
        self.mode: Optional[str] = None                        # "DEBUG" or "PRODUCTION"
        self.ct_input_path: Optional[str] = None               # Raw CT input (NIfTI or DICOM directory)
        self.ct_input_type: Optional[str] = None               # "nii" or "dicom"
        self.ct_input_identity: Optional[Dict[str, Any]] = None # Fingerprinted CT provenance for rerun safety
        self.ct_saved_copy_path: Optional[str] = None          # Saved CT copy inside the patient output folder
        self.ct_index: Optional[int] = None                    # Batch index used in output naming
        self.output_folder_path: Optional[str] = None          # Root output folder for this CT
        self.metadata_dir: Optional[str] = None                # Persistent metadata dir (survives work_dir cleanup)
        self.subdir_paths: Optional[Dict[str, str]] = None     # {phase_key: absolute path}
        self.subdir_names: Optional[Dict[str, str]] = None     # {phase_key: directory name}
        self.synthetic_lesions_enabled: Optional[bool] = None  # Whether the synthetic lesions stage will run
        self.downstream_roi_subset: Optional[list[str]] = None # ROI names propagated to downstream stages
        self.run_spect: Optional[bool] = None
        self.run_dosimetry: Optional[bool] = None
        self.run_postprocess: Optional[bool] = None

        # Full config dict snapshot (deep-copied from parsed JSON at pipeline start).
        self.config: Dict[str, Any] = {}

        # Phase 1: Digital Twin & Ground Truth
        self.ct_nii_path: Optional[str] = None                        # Standardized CT NIfTI
        self.body_ml_path: Optional[str] = None                       # Body task multilabel mask
        self.head_glands_cavities_ml_path: Optional[str] = None       # Head-gland task multilabel mask
        self.total_ml_path: Optional[str] = None                      # Total task multilabel mask
        self.totseg_plan: Optional[Dict[str, Any]] = None             # TotalSegmentator execution plan
        self.pytheratwin_roi_seg_path: Optional[str] = None                   # Unified label map handoff

        # Phase 1.2: Synthetic lesions
        self.synthetic_lesions_outdir: Optional[str] = None
        self.synthetic_lesions_results: Optional[Dict[str, Any]] = None
        self.synthetic_lesions_backup_seg_path: Optional[str] = None
        self.synthetic_lesions_global_binary_path: Optional[str] = None
        self.synthetic_lesions_global_labels_path: Optional[str] = None

        # Phase 1.3: PBPK TAC generation
        self.pbpk_tacs_by_organ: Optional[Dict[str, Any]] = None      # {roi: {tac_time/values/...}}
        self.pbpk_tac_json_path: Optional[str] = None
        self.pbpk_tac_npz_path: Optional[str] = None
        self.pbpk_tac_time: Optional[Any] = None
        self.pbpk_tac_values: Optional[Dict[str, Any]] = None
        self.pbpk_height_m: Optional[float] = None                    # Patient height from DICOM (m)
        self.pbpk_weight_kg: Optional[float] = None                   # Patient weight from DICOM (kg)
        self.pbpk_parameters: Optional[Dict[str, Any]] = None         # PyCNO parameter overrides
        self.pbpk_vois: Optional[list[str]] = None
        self.pbpk_isotope: Optional[str] = None
        self.pbpk_stop_time_min: Optional[float] = None

        # Phase 2: Simulations
        self.body_seg_arr: Optional[Any] = None                       # float32 body mask in SIMIND grid
        self.roi_body_seg_arr: Optional[Any] = None                   # int16 ROI labels in SIMIND grid
        self.mask_roi_body: Optional[Dict[int, Any]] = None           # {label_id: bool mask}
        self.class_seg: Optional[Dict[str, int]] = None               # {roi_name: label_id}
        self.atn_av_path: Optional[str] = None                        # SIMIND attenuation binary (.bin)
        self.binary_roi_act_map_paths: Optional[Dict[str, str]] = None # {roi_name: source map path}
        self.arr_px_spacing_cm: Optional[Any] = None                  # (z, y, x) spacing in cm
        self.arr_shape_new: Optional[Any] = None                      # (z, y, x) shape after resize

        self.spect_sim_output_dir: Optional[str] = None
        self.simind_stage_output_dir: Optional[str] = None
        self.simind_work_dir: Optional[str] = None
        self.simind_metadata_path: Optional[str] = None
        self.simind_calibration_path: Optional[str] = None
        self.simind_projection_paths: Optional[Dict[str, Any]] = None   # {roi_name: {w1/w2/w3: path}}
        self.simind_summed_projection_paths: Optional[Dict[str, str]] = None  
        self.simind_num_cpu: Optional[int] = None
        self.simind_geometry: Optional[Dict[str, Any]] = None
        self.simind_total_num_voxels: Optional[int] = None
        self.simind_scale_factor: Optional[float] = None
        self.simind_switches_by_organ: Optional[Dict[str, str]] = None
        self.simind_header_dir: Optional[str] = None

        # Phase 2.2: OpenGATE dosimetry
        self.dosimetry_output_dir: Optional[str] = None
        self.dosimetry_stage_output_dir: Optional[str] = None
        self.dosimetry_work_dir: Optional[str] = None
        self.dosimetry_metadata_path: Optional[str] = None
        self.dosimetry_mask_paths: Optional[Dict[str, str]] = None        # {roi: source mask path}
        self.dosimetry_raw_dose_paths: Optional[Dict[str, str]] = None    # {roi: per-ROI dose NIfTI}
        self.dosimetry_raw_uncertainty_paths: Optional[Dict[str, str]] = None
        self.dosimetry_sum_dose_path: Optional[str] = None                # Summed dose NIfTI across all ROIs
        self.dosimetry_sum_uncertainty_path: Optional[str] = None
        self.dosimetry_material_label_path: Optional[str] = None          # Material label image NIfTI

        # Phase 3: Post-processing
        self.pbpk_frame_start_times_min: Optional[Any] = None         # np.ndarray of frame start times (min)
        self.pbpk_frame_durations_s: Optional[Any] = None             # np.ndarray of frame durations (s)
        self.pbpk_projection_paths: Optional[Dict[str, Any]] = None   # {frame_label: {w1/w2/w3: path}}
        self.activity_map_sum: Optional[Any] = None                   # np.ndarray (n_frames,) total activity [MBq]
        self.activity_organ_sum: Optional[Dict[str, Any]] = None      # {roi: np.ndarray (n_frames,)} [MBq]
        self.activity_map_paths_by_organ: Optional[Any] = None        # Paths to per-organ activity maps
        self.reconstruction_output_dir: Optional[str] = None

        # Phase 3.2: Dosimetry post-processing
        self.dosemap_postprocess_output_dir: Optional[str] = None
        self.dosemap_postprocess_dose_path: Optional[str] = None

    @classmethod
    def from_pipeline_run(
        cls,
        *,
        logger: Optional[Any],
        config: Dict[str, Any],
        subdir_paths: Dict[str, str],
        subdir_names: Dict[str, str],
        mode: str,
        ct_input_path: str,
        ct_input_type: str,
        ct_input_identity: Dict[str, Any],
        ct_saved_copy_path: str,
        ct_index: int,
        output_folder_path: str,
        metadata_dir: str,
        synthetic_lesions_enabled: bool,
        run_spect: bool,
        run_dosimetry: bool,
        run_postprocess: bool,
    ) -> "Context":
        """Build a fully initialized pipeline context for one CT run."""
        context = cls(logger=logger)

        context.config = copy.deepcopy(config)
        context.subdir_paths = copy.deepcopy(subdir_paths)
        context.subdir_names = copy.deepcopy(subdir_names)

        context.mode = mode
        context.ct_input_path = ct_input_path
        context.ct_input_type = ct_input_type
        context.ct_input_identity = copy.deepcopy(ct_input_identity)
        context.ct_saved_copy_path = ct_saved_copy_path
        context.ct_index = ct_index
        context.output_folder_path = output_folder_path
        context.metadata_dir = metadata_dir
        context.synthetic_lesions_enabled = synthetic_lesions_enabled
        context.run_spect = run_spect
        context.run_dosimetry = run_dosimetry
        context.run_postprocess = run_postprocess

        roi_subset = config["phase_1"]["segmentation_stage"]["roi_subset"]
        if isinstance(roi_subset, str):
            roi_subset = [roi_subset]
        normalized_roi_subset = [str(r).strip() for r in roi_subset if str(r).strip()]
        if "remaining_body" not in normalized_roi_subset:
            normalized_roi_subset.append("remaining_body")
        context.downstream_roi_subset = normalized_roi_subset

        return context

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
