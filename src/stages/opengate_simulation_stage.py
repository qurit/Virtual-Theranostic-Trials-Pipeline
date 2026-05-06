"""
OpenGATE dosimetry for the VTT pipeline.

This stage runs voxel-source Monte Carlo dose calculations on the phase-1 CT grid
(using the native CT grid or an optional downsampled simulation grid) with OpenGATE.
Each requested ROI is simulated independently as a binary voxel source, then the
resulting dose maps are saved and summed on that same simulation grid.

Core responsibilities
---------------------
- Validate required context fields and stage configuration.
- Load the phase-1 CT and unified VTT ROI segmentation.
- Optionally downsample the simulation inputs for faster Monte Carlo execution.
- Convert simulation inputs to OpenGATE's centered identity-direction convention.
- Build one binary voxel-source mask per requested ROI.
- Run adaptive OpenGATE batches per ROI using Lu-177 decay physics.
- Skip simulation for any ROI whose final dose NIfTI output already exists on disk
  (allows resuming a crashed run without re-simulating completed ROIs).
- Save per-ROI dose maps, uncertainty maps, and a summed dose map
  on the selected simulation grid.
- Save stage metadata and optional OpenGATE material-label / MHD outputs.

Dose units
----------
OpenGATE batch dose arrays are accumulated and scaled by the actual total number
of simulated histories, so the saved NIfTI outputs are:

    Gy/decay

To obtain absolute dose in Gy, multiply by the total number of physical decays.

Important coordinate note
-------------------------
OpenGATE centers Image volumes at the world origin regardless of the original NIfTI
origin/direction metadata. To keep the voxel source aligned with the patient image,
this stage converts the CT and segmentation into a centered, identity-direction form
before simulation, then un-flips the dose arrays back to the original voxel ordering
before saving.

Expected Context interface
--------------------------
Incoming `context` is expected to provide:
- context.subdir_paths["phase_2"] : str
- context.config["phase_2"]["opengate_stage"] : dict (including roi_subset)
- context.ct_nii_path : str
- context.vtt_roi_seg_path : str
- context.downstream_roi_subset : list[str] | str
- label map loaded from ``src/data/pipeline_paths.json`` input_paths.label_map_path

On success, this stage sets:
- context.dosimetry_output_dir : str
- context.dosimetry_stage_output_dir : str
- context.dosimetry_work_dir : str
- context.dosimetry_metadata_path : str
- context.dosimetry_mask_paths : dict[str, str]
- context.dosimetry_raw_dose_paths : dict[str, str]
- context.dosimetry_raw_uncertainty_paths : dict[str, str]
- context.dosimetry_sum_dose_path : Optional[str]
- context.dosimetry_material_label_path : Optional[str]
- context.extras["opengate_simulation_stage"] : dict
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk

from src.io.config_paths import get_label_map_path
from src.io.rerun_guard import (
    assert_stage_rerun_safe,
    build_stage_metadata,
    build_opengate_rerun_snapshot,
    fingerprint_optional_file,
    load_json,
    stage_metadata_path,
    write_json,
)
from src.utils.nifti_utils import save_nii_sitk
from src.utils.label_utils import load_vtt_label_map, filter_roi_seg_to_subset, load_isotope_config
from src.utils.opengate_utils import (
    cleanup_mhd,
    find_hu_tables,
    resample_image,
    to_centered_identity,
)
from src.utils.resize_utils import resolve_simulation_grid


# Developer-only adaptive stopping controls. These are intentionally not exposed
# in config/UI because users should choose batch size and target uncertainty only.
OPENGATE_UNCERTAINTY_PERCENTILE = 95.0
OPENGATE_MAX_BATCHES_PER_ROI = 100
OPENGATE_BASE_RANDOM_SEED = 0


class OpenGateSimulationStage:
    """
    Run OpenGATE voxel-source dosimetry on the phase-1 CT grid.

    Notes
    -----
    - Simulation may run on the native CT grid or on an optional downsampled grid.
    - Source masks are binary voxel maps derived from the unified VTT segmentation.
    - ROI dose outputs are accumulated into a summed dose map in simulation space.
    - If a per-ROI dose NIfTI already exists on disk the simulation for that ROI is
      skipped, allowing a crashed run to be resumed from where it left off.
    """

    def __init__(self, context: Any) -> None:
        context.require(
            "subdir_paths",
            "config",
            "ct_nii_path",
            "vtt_roi_seg_path",
            "downstream_roi_subset",
            "output_folder_path",
            "ct_input_identity",
        )
        self.context = context
        self.debug: bool = getattr(context, "mode", "").upper() == "DEBUG"
        self.ct_input_identity: Dict[str, Any] = context.ct_input_identity

        self.phase_output_dir: str = context.subdir_paths["phase_2"]                   
        self.stage_cfg: Dict[str, Any] = context.config["phase_2"]["opengate_stage"]   
        self.stage_output_dir: str = os.path.join(
            self.phase_output_dir,
            self.stage_cfg.get("sub_dir_name", "opengate_simulation"),
        )
        self.output_dir: str = self.stage_output_dir
        self.work_dir: str = os.path.join(self.stage_output_dir, "work_dir")
        self.source_mask_dir: str = os.path.join(self.work_dir, "source_masks")
        self.resample_dir: str = os.path.join(self.work_dir, "resampled_inputs")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(self.source_mask_dir, exist_ok=True)
        os.makedirs(self.resample_dir, exist_ok=True)

        self.prefix: str = str(self.stage_cfg.get("file_prefix", "dosimetry"))
        self.metadata_path: str = stage_metadata_path(context.output_folder_path, "opengate_simulation_stage")

        # Save / provenance options
        self.save_per_roi_dose_maps: bool = bool(self.stage_cfg.get("save_per_roi_dose_maps", True))
        self.save_summed_dose_map: bool = True
        self.save_uncertainty_map: bool = True
        self.save_material_label_image: bool = bool(self.stage_cfg.get("save_material_label_image", True))
        self.write_mhd_outputs: bool = bool(self.stage_cfg.get("write_mhd_outputs", False))

        # Target voxel spacing in mm [x, y, z], or null to use native CT spacing
        self.xyz_spacing_mm = self.stage_cfg.get("xyz_spacing_mm", None)

        # OpenGATE execution controls
        gate_cfg = self.stage_cfg.get("gate", {})
        deprecated_gate_fields = {
            "total_histories": "total_histories_per_batch",
            "num_cpu": "num_threads",
            "random_seed": None,
        }
        for old_field, new_field in deprecated_gate_fields.items():
            if old_field in gate_cfg:
                if new_field is None:
                    raise ValueError(
                        f"gate.{old_field} is no longer user-configurable; "
                        "adaptive OpenGATE batches derive deterministic independent seeds internally."
                    )
                raise ValueError(f"gate.{old_field} has been replaced by gate.{new_field}")
        self.requested_total_histories_per_batch: int = int(
            gate_cfg.get("total_histories_per_batch", 1_000_000)
        )
        # num_threads: 0 = use all available CPUs (same convention as SIMIND).
        requested_num_threads: int = int(gate_cfg.get("num_threads", 1))
        self.configured_num_threads: int = requested_num_threads
        self.target_uncertainty_percent: float = float(gate_cfg.get("target_uncertainty_percent", 5.0))
        self.target_uncertainty_fraction: float = self.target_uncertainty_percent / 100.0
        self.uncertainty_percentile: float = OPENGATE_UNCERTAINTY_PERCENTILE
        self.max_batches_per_roi: int = OPENGATE_MAX_BATCHES_PER_ROI
        self.random_seed_base: int = OPENGATE_BASE_RANDOM_SEED
        self.start_new_process: bool = bool(gate_cfg.get("start_new_process", True))

        if self.requested_total_histories_per_batch <= 0:
            raise ValueError("gate.total_histories_per_batch must be > 0")
        if requested_num_threads < 0:
            raise ValueError("gate.num_threads must be >= 0 (0 = use all available CPUs)")
        if not 0.0 <= self.target_uncertainty_percent <= 100.0:
            raise ValueError("gate.target_uncertainty_percent must be between 0 and 100")
        if requested_num_threads == 0:
            effective_requested_threads = os.cpu_count() or 1
        else:
            effective_requested_threads = requested_num_threads
        self.requested_num_threads: int = effective_requested_threads

        # OpenGATE source.n is set per thread, so convert requested batch histories
        # into a per-thread count to keep each batch total under control.
        self.num_threads: int = min(self.requested_num_threads, self.requested_total_histories_per_batch)
        self.histories_per_thread_per_batch: int = self.requested_total_histories_per_batch // self.num_threads
        self.actual_histories_per_batch: int = self.histories_per_thread_per_batch * self.num_threads
        self.history_rounding_loss_per_batch: int = (
            self.requested_total_histories_per_batch - self.actual_histories_per_batch
        )
        if self.history_rounding_loss_per_batch > 0:
            print(
                "[OpenGateSimulationStage] Batch history rounding: requested "
                f"{self.requested_total_histories_per_batch}, running {self.actual_histories_per_batch} "
                f"({self.histories_per_thread_per_batch} per thread x {self.num_threads} threads). "
                f"Loss per batch: {self.history_rounding_loss_per_batch} histories."
            )

        source_cfg = self.stage_cfg.get("source", {})
        self.isotope: str = str(source_cfg.get("isotope", "lu177")).lower().strip().replace("-", "")

        # Physics / geometry controls
        self.variance_reduction: bool = bool(self.stage_cfg.get("variance_reduction", True))
        physics_cfg = self.stage_cfg.get("physics", {})
        self.density_tolerance_gcm3: float = float(physics_cfg.get("density_tolerance_gcm3", 0.05))
        self.world_margin_scale: float = float(physics_cfg.get("world_margin_scale", 1.4))
        self.world_min_size_mm: float = float(physics_cfg.get("world_min_size_mm", 400.0))
        self.electron_production_cut_mm: float = float(physics_cfg.get("electron_production_cut_mm", 0.1))

        # Input paths
        self.ct_nii_path: Path = Path(context.ct_nii_path)
        self.vtt_roi_seg_path: Path = Path(context.vtt_roi_seg_path)
        self.label_map_path: Path = Path(get_label_map_path())
        self.vtt_name2id: Dict[str, int] = load_vtt_label_map(self.label_map_path)

        _nuclear = load_isotope_config()["nuclear"].get(self.isotope)
        if _nuclear is None:
            raise ValueError(f"Isotope '{self.isotope}' not found in isotope_config.json nuclear data.")
        self._isotope_Z: int = _nuclear["Z"]
        self._isotope_A: int = _nuclear["A"]

        # Build final ROI list from opengate_stage config (independent from phase_1) 
        using_config_roi_subset = self.stage_cfg.get("roi_subset") is not None
        opengate_roi_subset = self.stage_cfg.get("roi_subset")                         
        if opengate_roi_subset is None:                                                
            opengate_roi_subset = getattr(context, "downstream_roi_subset", None)      
        if isinstance(opengate_roi_subset, str):                                       
            opengate_roi_subset = [opengate_roi_subset]                                
        if opengate_roi_subset is None:                                                
            raise ValueError("OpenGATE roi_subset must be provided (in config or context)") 
        opengate_roi_subset = [str(r).strip() for r in opengate_roi_subset if str(r).strip()]

        lesion_specs = (
            self.context.config.get("phase_1", {})
            .get("synthetic_lesions_stage", {})
            .get("specs")
        )
        lesion_hosts = [str(r).strip() for r in lesion_specs] if isinstance(lesion_specs, dict) else []
        picked_lesion_hosts = [r for r in lesion_hosts if r in opengate_roi_subset]
        missing_lesion_hosts = [r for r in lesion_hosts if r not in opengate_roi_subset]
        if using_config_roi_subset and "synthetic_lesion" in opengate_roi_subset:
            raise ValueError(
                "OpenGATE roi_subset contains internal ROI 'synthetic_lesion'. "
                "Select lesion host ROIs instead; "
                "the synthetic_lesion source is added automatically."
            )
        if picked_lesion_hosts and missing_lesion_hosts:
            raise ValueError(
                "OpenGATE roi_subset selects some synthetic-lesion host ROIs "
                f"({picked_lesion_hosts}) but not all ({missing_lesion_hosts}). "
                "Include all lesion host ROIs or none."
            )
        if picked_lesion_hosts and "synthetic_lesion" not in opengate_roi_subset:
            opengate_roi_subset.append("synthetic_lesion")

        # Validate OpenGATE ROI subset against phase_1 segmented ROIs
        phase1_rois = set(getattr(context, "downstream_roi_subset", []) or [])
        _internal = {"remaining_body", "synthetic_lesion"}
        invalid_rois = [r for r in opengate_roi_subset if r not in phase1_rois and r not in _internal]
        if invalid_rois:
            raise ValueError(
                f"OpenGATE roi_subset contains ROIs not segmented in Phase 1: {invalid_rois}. "
                f"Available: {sorted(phase1_rois)}"
            )                                                                          

        roi_list: List[str] = []
        if "remaining_body" in self.vtt_name2id:
            roi_list.append("remaining_body")
        for roi_name in opengate_roi_subset:
            if roi_name not in roi_list:
                roi_list.append(roi_name)
        if not roi_list:
            raise ValueError("No ROI names found for dosimetry simulation")
        self.requested_roi_subset: List[str] = roi_list

        # Populated during image preparation and used later when restoring dose arrays.
        self._original_sim_ct_img: Optional[sitk.Image] = None
        self._centering_flipped_axes: List[int] = []

    def _rerun_config_snapshot(self) -> Dict[str, Any]:
        """Return the OpenGATE settings that must match for cached outputs to remain valid."""
        return build_opengate_rerun_snapshot(
            self.context.config,
            synthetic_enabled=bool(getattr(self.context, "synthetic_lesions_enabled", False)),
        )

    def _current_dependency_fingerprints(self) -> Dict[str, Any]:
        """Return fingerprints for the current stage inputs that affect dosimetry outputs."""
        deps = {
            "ct_nii": fingerprint_optional_file(self.ct_nii_path),
            "vtt_roi_seg": fingerprint_optional_file(self.vtt_roi_seg_path),
            "label_map_json": fingerprint_optional_file(self.label_map_path),
            "segmentation_stage_metadata": fingerprint_optional_file(
                stage_metadata_path(self.context.output_folder_path, "segmentation_stage")
            ),
        }
        if "synthetic_lesion" in self.requested_roi_subset:
            deps["synthetic_lesions_stage_metadata"] = fingerprint_optional_file(
                stage_metadata_path(self.context.output_folder_path, "synthetic_lesions_stage")
            )
        return deps

    def _cache_marker_paths(self) -> List[str]:
        """
        Return stage-completion outputs that signal a full previous run finished.

        Per-ROI dose NIfTIs are intentionally excluded: they serve as resume markers
        within a run (the per-ROI skip logic in run()) but must not trigger the rerun
        guard on partial runs, otherwise changing the config mid-run raises a false
        conflict instead of allowing a clean restart.
        """
        markers: List[str] = []
        if self.save_summed_dose_map:
            markers.append(str(Path(self.phase_output_dir) / f"{self.prefix}_dose_sum.nii.gz"))
        return markers

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        """Print a stage-local debug message only in DEBUG mode."""
        if self.debug:
            print(f"[OpenGateSimulationStage] {msg}")

    def _roi_dose_nii_path(self, roi_name: str) -> Path:
        """Return the expected final NIfTI output path for a given ROI dose map."""
        return Path(self.stage_output_dir) / f"{self.prefix}_dose_{roi_name}.nii.gz"

    def _roi_unc_nii_path(self, roi_name: str) -> Path:
        """Return the expected final NIfTI output path for a given ROI uncertainty map."""
        return Path(self.stage_output_dir) / f"{self.prefix}_unc_{roi_name}.nii.gz"

    def _material_label_nii_path(self) -> Path:
        """Return the persistent material-label output path."""
        return Path(self.stage_output_dir) / f"{self.prefix}_material_labels.nii.gz"

    def _roi_batch_dir(self, roi_name: str, batch_index: int) -> Path:
        """Return the scratch directory for one ROI batch."""
        return Path(self.work_dir) / roi_name / "batches" / f"batch_{batch_index:06d}"

    def _batch_seed(self, roi_name: str, batch_index: int) -> int:
        """Return a deterministic seed unique to each ROI/batch pair."""
        h = self.random_seed_base
        for ch in roi_name:
            h = (h * 31 + ord(ch)) & 0x7FFFFFFF
        return (h + batch_index * 999_983) % 2_147_483_646 + 1

    def _aligned_roi_mask_array(self, mask_path: str) -> np.ndarray:
        """Load a source mask and flip it into the same voxel order as saved dose arrays."""
        mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path))).astype(bool)
        for np_axis in reversed(self._centering_flipped_axes):
            mask_arr = np.flip(mask_arr, axis=np_axis).copy()
        return mask_arr

    @staticmethod
    def _combine_relative_uncertainty(
        dose_sum_arr: np.ndarray,
        uncertainty_numerator_var_arr: np.ndarray,
    ) -> np.ndarray:
        """Combine independent relative uncertainty contributions for an accumulated dose."""
        sigma_num = np.sqrt(uncertainty_numerator_var_arr)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_unc = np.divide(
                sigma_num,
                np.abs(dose_sum_arr),
                out=np.full_like(dose_sum_arr, np.inf, dtype=np.float64),
                where=np.abs(dose_sum_arr) > 0,
            )
        return rel_unc.astype(np.float32)

    @staticmethod
    def _relative_uncertainty_variance_numerator(rel_unc: np.ndarray, dose_arr: np.ndarray) -> np.ndarray:
        """Return ``(relative_uncertainty * dose)^2`` while treating zero-dose voxels safely."""
        dose_abs = np.abs(np.asarray(dose_arr, dtype=np.float64))
        unc = np.asarray(rel_unc, dtype=np.float64)
        clean_unc = np.where(np.isfinite(unc) & (unc >= 0), unc, np.inf)
        sigma_numerator = np.zeros_like(dose_abs, dtype=np.float64)
        nonzero = dose_abs > 0
        sigma_numerator[nonzero] = clean_unc[nonzero] * dose_abs[nonzero]
        return np.square(sigma_numerator)

    def _uncertainty_percentile_in_mask(self, unc_arr: np.ndarray, mask_arr: np.ndarray) -> float:
        """Return the configured uncertainty percentile inside an ROI mask.

        Zero-dose voxels carry inf relative uncertainty and are excluded so that
        unsampled edge voxels in early batches do not permanently block convergence.
        """
        vals = np.asarray(unc_arr[mask_arr], dtype=np.float64)
        vals = vals[np.isfinite(vals) & (vals >= 0)]
        if vals.size == 0:
            return float("inf")
        return float(np.percentile(vals, self.uncertainty_percentile))

    @staticmethod
    def _finite_float_or_none(value: float) -> Optional[float]:
        """Return a JSON-safe float, or None for infinity/NaN."""
        return float(value) if np.isfinite(value) else None

    @staticmethod
    def _same_image_grid(a: sitk.Image, b: sitk.Image, *, atol: float = 1e-5) -> bool:
        """Return True when two images share the same voxel grid geometry."""
        return (
            tuple(a.GetSize()) == tuple(b.GetSize())
            and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=atol)
            and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=atol)
            and np.allclose(a.GetDirection(), b.GetDirection(), atol=atol)
        )

    def _prepare_simulation_images(
        self,
        ct: sitk.Image,
        seg: sitk.Image,
    ) -> Tuple[sitk.Image, sitk.Image, Path, Path, bool]:
        """
        Prepare CT and segmentation images for OpenGATE.

        Steps
        -----
        1) Optionally downsample.
        2) Store the pre-centered simulation CT geometry used for dose outputs.
        3) Convert both images to centered identity-direction form.
        4) Write the prepared images to disk and re-read them.
        """
        sim_ct_path = Path(self.resample_dir) / "ct_sim.nii.gz"
        sim_seg_path = Path(self.resample_dir) / "seg_sim.nii.gz"

        was_resampled = False
        if self.xyz_spacing_mm is not None:
            native_sp = ct.GetSpacing()
            self._log(
                f"Native CT spacing (x,y,z): ({native_sp[0]:.3f}, {native_sp[1]:.3f}, {native_sp[2]:.3f}) mm  →  "
                f"target spacing: {tuple(self.xyz_spacing_mm)} mm"
            )
            _grid = resolve_simulation_grid(self.xyz_spacing_mm, ct.GetSize(), native_sp)
            if _grid is not None:
                (nx, ny, nz), (sx, sy, sz) = _grid
                new_size = (nx, ny, nz)
                new_spacing = (native_sp[0] / sx, native_sp[1] / sy, native_sp[2] / sz)
                self._log(f"Downsampling to {new_size} (spacing {tuple(round(s, 2) for s in new_spacing)} mm)")
                ct = resample_image(ct, new_size, new_spacing, is_label=False)
                seg = resample_image(seg, new_size, new_spacing, is_label=True)
                was_resampled = True

        # Preserve the simulation-space geometry before centering so saved dose maps
        # stay on the selected OpenGATE grid.
        self._original_sim_ct_img = sitk.GetImageFromArray(sitk.GetArrayFromImage(ct))
        self._original_sim_ct_img.CopyInformation(ct)

        ct_centered, self._centering_flipped_axes = to_centered_identity(ct)
        seg_centered, _ = to_centered_identity(seg, is_label=True)

        self._log(
            f"Centered CT: origin={tuple(round(o, 1) for o in ct_centered.GetOrigin())}, "
            f"flipped_axes={self._centering_flipped_axes}"
        )

        sitk.WriteImage(ct_centered, str(sim_ct_path), imageIO="NiftiImageIO")
        sitk.WriteImage(seg_centered, str(sim_seg_path), imageIO="NiftiImageIO")

        return (
            sitk.ReadImage(str(sim_ct_path)),
            sitk.ReadImage(str(sim_seg_path)),
            sim_ct_path,
            sim_seg_path,
            was_resampled,
        )

    # ------------------------------------------------------------------
    # source masks
    # ------------------------------------------------------------------

    def _build_source_masks(
        self,
        sim_ct: sitk.Image,
        sim_seg: sitk.Image,
    ) -> Tuple[Dict[str, str], Dict[str, int], List[str]]:
        """
        Build one binary voxel-source mask per requested ROI on the simulation grid.

        remaining_body is recomputed for this stage's ROI subset before mask extraction,
        ensuring it covers body_outline − union(stage_rois) rather than the phase-1 value.

        Returns
        -------
        mask_paths : dict[str, str]
        counts : dict[str, int]
        names : list[str]
            Only ROIs with non-zero voxels are returned.
        """
        seg_arr = sitk.GetArrayFromImage(sim_seg).astype(np.int32)
        seg_arr = filter_roi_seg_to_subset(seg_arr, self.requested_roi_subset, self.vtt_name2id)

        mask_paths: Dict[str, str] = {}
        counts: Dict[str, int] = {}
        names: List[str] = []

        for roi_name in self.requested_roi_subset:
            label_id = self.vtt_name2id.get(roi_name)
            if label_id is None:
                raise ValueError(f"ROI '{roi_name}' not in VTT label map")

            mask = (seg_arr == int(label_id)).astype(np.uint8)
            n_vox = int(np.count_nonzero(mask))
            if n_vox == 0:
                self._log(f"Skipping '{roi_name}': zero voxels")
                continue

            mask_path = Path(self.source_mask_dir) / f"{self.prefix}_{roi_name}_source_mask.nii.gz"
            save_nii_sitk(sim_ct, mask, mask_path)
            mask_paths[roi_name] = str(mask_path)
            counts[roi_name] = n_vox
            names.append(roi_name)
            self._log(f"Mask '{roi_name}': {n_vox} voxels")

        if not names:
            raise ValueError("No ROIs had voxels in segmentation")
        return mask_paths, counts, names

    # ------------------------------------------------------------------
    # single-batch simulation
    # ------------------------------------------------------------------

    def _run_single_roi(
        self,
        roi_name: str,
        mask_path: str,
        sim_ct_path: Path,
        batch_dir: Path,
        histories_per_thread: int,
        random_seed: int,
        save_labels: bool,
    ) -> Dict[str, Any]:
        """
        Run one OpenGATE batch for a single ROI voxel source.

        Returns a dictionary containing raw arrays, file paths, and provenance fields.
        """
        batch_dir.mkdir(exist_ok=True, parents=True)

        ct = sitk.ReadImage(str(sim_ct_path))
        size_xyz = ct.GetSize()
        spacing_xyz = ct.GetSpacing()
        phys_size_mm = np.asarray(size_xyz, dtype=np.float64) * np.asarray(spacing_xyz, dtype=np.float64)

        self._log(f"ROI: {roi_name} | size={size_xyz} | phys={tuple(round(p, 1) for p in phys_size_mm)} mm")

        mm = gate.g4_units.mm
        gcm3 = gate.g4_units.g_cm3

        sim = gate.Simulation()
        sim.g4_verbose = False
        sim.visu = False
        sim.number_of_threads = self.num_threads
        sim.random_seed = random_seed
        sim.output_dir = batch_dir

        # World volume sized from the CT physical extent with a minimum floor.
        world_size_mm = np.maximum(phys_size_mm * self.world_margin_scale, self.world_min_size_mm)
        sim.world.size = [world_size_mm[0] * mm, world_size_mm[1] * mm, world_size_mm[2] * mm]
        sim.world.material = "G4_AIR"

        # Electromagnetic physics + radioactive decay for Lu-177 ion emission.
        sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option4"
        sim.physics_manager.special_physics_constructors.G4DecayPhysics = True
        sim.physics_manager.special_physics_constructors.G4RadioactiveDecayPhysics = True
        if self.variance_reduction:
            sim.physics_manager.global_production_cuts.electron = self.electron_production_cut_mm * mm
            sim.physics_manager.global_production_cuts.positron = self.electron_production_cut_mm * mm

        # CT image as patient geometry, with material assignment from Schneider HU tables.
        patient = sim.add_volume("Image", "patient")
        patient.image = str(sim_ct_path)
        patient.material = "G4_AIR"

        mat_table, den_table = find_hu_tables(gate)
        voxel_materials, generated_materials = gate.geometry.materials.HounsfieldUnit_to_material(
            sim,
            self.density_tolerance_gcm3 * gcm3,
            str(mat_table),
            str(den_table),
        )
        patient.voxel_materials = voxel_materials

        label_mhd: Optional[Path] = None
        if save_labels and self.save_material_label_image:
            label_mhd = batch_dir / "patient_material_labels.mhd"
            patient.dump_label_image = str(label_mhd)

        # Lu-177 voxel source. `src.n` is per thread.
        src = sim.add_source("VoxelSource", "src_lu177")
        src.particle = f"ion {self._isotope_Z} {self._isotope_A} 0 0"
        src.n = histories_per_thread
        src.image = str(mask_path)
        src.direction.type = "iso"
        src.energy.type = "mono"
        src.energy.mono = 0

        stats = sim.add_actor("SimulationStatisticsActor", "Stats")
        stats.output_filename = "stats.txt"

        dose = sim.add_actor("DoseActor", "dose")
        dose.attached_to = patient
        dose.size = list(size_xyz)
        dose.spacing = [float(s) * mm for s in spacing_xyz]
        dose.dose.active = True
        dose.dose.output_filename = "dose_map.mhd"
        dose.dose_uncertainty.active = True
        dose.dose_uncertainty.output_filename = "dose_uncertainty.mhd"

        sim.run(start_new_process=self.start_new_process)

        dose_mhd = batch_dir / "dose_map.mhd"
        if not dose_mhd.exists():
            raise FileNotFoundError(f"Dose output not found: {dose_mhd}")

        dose_img = sitk.ReadImage(str(dose_mhd))
        dose_arr = sitk.GetArrayFromImage(dose_img).astype(np.float32)

        if self.debug:
            nonzero = dose_arr[dose_arr > 0]
            if nonzero.size > 0:
                centroid_frac = np.argwhere(dose_arr > np.percentile(nonzero, 90)).mean(axis=0) / np.array(dose_arr.shape)
                self._log(
                    f"  Dose centroid (frac): z={centroid_frac[0]:.2f} "
                    f"y={centroid_frac[1]:.2f} x={centroid_frac[2]:.2f}"
                )

        # Undo the centering flips so the dose array matches the original simulation CT voxel order.
        for np_axis in reversed(self._centering_flipped_axes):
            dose_arr = np.flip(dose_arr, axis=np_axis).copy()

        unc_mhd = batch_dir / "dose_uncertainty.mhd"
        if not unc_mhd.exists():
            raise FileNotFoundError(f"Dose uncertainty output not found: {unc_mhd}")
        unc_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(unc_mhd))).astype(np.float32)
        for np_axis in reversed(self._centering_flipped_axes):
            unc_arr = np.flip(unc_arr, axis=np_axis).copy()

        dose_nii_path = save_nii_sitk(
            self._original_sim_ct_img,
            dose_arr.astype(np.float32),
            batch_dir / "dose_raw.nii.gz",
        )
        unc_nii_path = save_nii_sitk(
            self._original_sim_ct_img,
            unc_arr.astype(np.float32),
            batch_dir / "dose_uncertainty.nii.gz",
        )

        if self.debug:
            with open(batch_dir / "coordinate_debug.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "dose_mhd_origin": [round(o, 2) for o in dose_img.GetOrigin()],
                        "centered_ct_origin": [round(o, 2) for o in ct.GetOrigin()],
                        "original_ct_origin": [round(o, 2) for o in self._original_sim_ct_img.GetOrigin()],
                        "flipped_axes": self._centering_flipped_axes,
                    },
                    f,
                    indent=2,
                )

        stats_events = None
        try:
            stats_events = int(stats.counts.events)
        except Exception:
            pass

        if label_mhd is not None and not label_mhd.exists():
            label_mhd = None

        return {
            "roi_name": roi_name,
            "roi_work_dir": str(batch_dir),
            "dose_mhd_path": str(dose_mhd),
            "unc_mhd_path": str(unc_mhd),
            "dose_nii_path": str(dose_nii_path),
            "unc_nii_path": str(unc_nii_path),
            "label_mhd_path": None if label_mhd is None else str(label_mhd),
            "dose_arr": dose_arr,
            "unc_arr": unc_arr,
            "random_seed": random_seed,
            "histories_per_thread": histories_per_thread,
            "size_xyz": tuple(int(v) for v in size_xyz),
            "spacing_xyz": tuple(float(v) for v in spacing_xyz),
            "stats_str": str(stats),
            "stats_events": stats_events,
            "mat_table": str(mat_table),
            "den_table": str(den_table),
            "n_materials": len(generated_materials),
        }

    def _run_roi_batches(
        self,
        roi_name: str,
        mask_path: str,
        sim_ct: sitk.Image,
        sim_ct_path: Path,
        save_labels: bool,
    ) -> Dict[str, Any]:
        """Run adaptive OpenGATE batches for one ROI and return accumulated outputs."""
        mask_arr = self._aligned_roi_mask_array(mask_path)
        dose_sum_raw: Optional[np.ndarray] = None
        uncertainty_numerator_var_arr: Optional[np.ndarray] = None
        total_actual_histories = 0
        batches: List[Dict[str, Any]] = []
        final_unc_arr: Optional[np.ndarray] = None
        final_uncertainty_percentile = float("inf")
        converged = False

        material_label_nii = self._material_label_nii_path()
        mat_label_path: Optional[str] = str(material_label_nii) if material_label_nii.exists() else None
        n_materials: Optional[int] = None
        mat_table: Optional[str] = None
        den_table: Optional[str] = None
        size_xyz: Optional[List[int]] = None
        spacing_xyz: Optional[List[float]] = None

        for batch_index in range(1, self.max_batches_per_roi + 1):
            batch_dir = self._roi_batch_dir(roi_name, batch_index)
            random_seed = self._batch_seed(roi_name, batch_index)
            res = self._run_single_roi(
                roi_name,
                mask_path,
                sim_ct_path,
                batch_dir=batch_dir,
                histories_per_thread=self.histories_per_thread_per_batch,
                random_seed=random_seed,
                save_labels=(save_labels and batch_index == 1),
            )

            batch_dose_raw = np.asarray(res["dose_arr"], dtype=np.float64)
            batch_unc = np.asarray(res["unc_arr"], dtype=np.float64)

            if dose_sum_raw is None:
                dose_sum_raw = np.zeros_like(batch_dose_raw, dtype=np.float64)
                uncertainty_numerator_var_arr = np.zeros_like(batch_dose_raw, dtype=np.float64)

            dose_sum_raw += batch_dose_raw
            total_actual_histories += self.actual_histories_per_batch

            uncertainty_numerator_var_arr += self._relative_uncertainty_variance_numerator(
                batch_unc,
                batch_dose_raw,
            )

            final_unc_arr = self._combine_relative_uncertainty(
                dose_sum_raw,
                uncertainty_numerator_var_arr,
            )
            final_uncertainty_percentile = self._uncertainty_percentile_in_mask(final_unc_arr, mask_arr)
            converged = final_uncertainty_percentile <= self.target_uncertainty_fraction

            batch_meta = {
                "batch_index": batch_index,
                "work_dir": res["roi_work_dir"],
                "random_seed": random_seed,
                "histories_per_thread": int(res["histories_per_thread"]),
                "actual_total_histories": int(self.actual_histories_per_batch),
                "cumulative_actual_histories": int(total_actual_histories),
                "dose_raw_path": res["dose_nii_path"],
                "uncertainty_path": res["unc_nii_path"],
                "uncertainty_percentile": self.uncertainty_percentile,
                "uncertainty_percentile_fraction": self._finite_float_or_none(final_uncertainty_percentile),
                "uncertainty_percentile_percent": self._finite_float_or_none(final_uncertainty_percentile * 100.0),
                "converged_after_batch": bool(converged),
                "stats": res["stats_str"],
                "events": res["stats_events"],
            }
            batches.append(batch_meta)
            print(
                f"[OpenGateSimulationStage] {roi_name} batch {batch_index}: "
                f"p{self.uncertainty_percentile:g} uncertainty={final_uncertainty_percentile * 100.0:.3g}% "
                f"(target {self.target_uncertainty_percent:g}%), "
                f"histories={total_actual_histories:,}"
            )

            if mat_label_path is None and self.save_material_label_image and res["label_mhd_path"] and os.path.exists(res["label_mhd_path"]):
                labels = sitk.GetArrayFromImage(sitk.ReadImage(res["label_mhd_path"])).astype(np.int16)
                mat_label_path = save_nii_sitk(
                    sim_ct,
                    labels,
                    self._material_label_nii_path(),
                )

            n_materials = int(res["n_materials"])
            mat_table = res["mat_table"]
            den_table = res["den_table"]
            size_xyz = list(res["size_xyz"])
            spacing_xyz = list(res["spacing_xyz"])

            if not self.write_mhd_outputs:
                cleanup_mhd(Path(res["dose_mhd_path"]))
                cleanup_mhd(Path(res["unc_mhd_path"]))
                if res["label_mhd_path"]:
                    cleanup_mhd(Path(res["label_mhd_path"]))

            if converged:
                break

        if dose_sum_raw is None or uncertainty_numerator_var_arr is None or final_unc_arr is None:
            raise RuntimeError(f"OpenGATE produced no batches for ROI '{roi_name}'")

        if not converged:
            print(
                f"[OpenGateSimulationStage] WARNING: '{roi_name}' reached the developer cap of "
                f"{self.max_batches_per_roi} batches without meeting target uncertainty "
                f"({final_uncertainty_percentile * 100.0:.3g}% > {self.target_uncertainty_percent:g}%). "
                "Saving the best accumulated result."
            )

        dose_grid = dose_sum_raw / float(total_actual_histories)
        return {
            "roi_name": roi_name,
            "dose_arr": dose_grid,
            "unc_arr": final_unc_arr,
            "total_actual_histories": int(total_actual_histories),
            "num_batches": len(batches),
            "converged": bool(converged),
            "cap_reached": bool(not converged),
            "final_uncertainty_percentile_fraction": self._finite_float_or_none(final_uncertainty_percentile),
            "final_uncertainty_percentile_percent": self._finite_float_or_none(final_uncertainty_percentile * 100.0),
            "batches": batches,
            "material_label_path": mat_label_path,
            "size": size_xyz,
            "spacing": spacing_xyz,
            "n_materials": n_materials,
            "mat_table": mat_table,
            "den_table": den_table,
        }

    # ------------------------------------------------------------------
    # metadata
    # ------------------------------------------------------------------

    def _save_metadata(
        self,
        roi_names: List[str],
        roi_counts: Dict[str, int],
        roi_meta: Dict[str, Any],
        mask_paths: Dict[str, str],
        dose_paths: Dict[str, str],
        unc_paths: Dict[str, str],
        sum_path: Optional[str],
        sum_unc_path: Optional[str],
        mat_label_path: Optional[str],
        was_resampled: bool,
        sim_ct_path: str,
        sim_seg_path: str,
    ) -> None:
        """Save stage metadata for provenance, debugging, and rerun reproducibility."""
        outputs: Dict[str, Any] = {
            "dose_paths": dose_paths,
            "unc_paths": unc_paths,
            "sum_path": sum_path,
            "sum_unc_path": sum_unc_path,
            "mat_label_path": mat_label_path,
        }
        extra: Dict[str, Any] = {
            "stage": "opengate_simulation_stage",
            "dose_units": "Gy/decay",
            "ct_nii_path": str(self.ct_nii_path),
            "vtt_roi_seg_path": str(self.vtt_roi_seg_path),
            "label_map_path": str(self.label_map_path),
            "requested_roi_subset": list(self.requested_roi_subset),
            "simulated_roi_names": list(roi_names),
            "skipped_zero_voxel_rois": [
                roi for roi in self.requested_roi_subset
                if roi not in roi_names
            ],
            "roi_voxel_counts": {k: int(v) for k, v in roi_counts.items()},
            "downsampling": {
                "xyz_spacing_mm": self.xyz_spacing_mm,
                "was_resampled": was_resampled,
                "sim_ct_path": sim_ct_path,
                "sim_seg_path": sim_seg_path,
            },
            "dose_output_grid": {
                "size_xyz": list(self._original_sim_ct_img.GetSize()),
                "spacing_xyz_mm": [float(v) for v in self._original_sim_ct_img.GetSpacing()],
                "origin_xyz_mm": [float(v) for v in self._original_sim_ct_img.GetOrigin()],
            },
            "centering": {
                "flipped_axes": self._centering_flipped_axes,
                "original_origin": [round(o, 2) for o in self._original_sim_ct_img.GetOrigin()],
            },
            "isotope": self.isotope,
            "save_options": {
                "save_per_roi_dose_maps": self.save_per_roi_dose_maps,
                "save_summed_dose_map": self.save_summed_dose_map,
                "save_uncertainty_map": self.save_uncertainty_map,
                "save_material_label_image": self.save_material_label_image,
                "write_mhd_outputs": self.write_mhd_outputs,
            },
            "gate": {
                "requested_total_histories_per_batch": self.requested_total_histories_per_batch,
                "actual_histories_per_batch": self.actual_histories_per_batch,
                "configured_num_threads": self.configured_num_threads,
                "effective_num_threads": self.num_threads,
                "num_threads": self.num_threads,
                "histories_per_thread_per_batch": self.histories_per_thread_per_batch,
                "history_rounding_loss_per_batch": self.history_rounding_loss_per_batch,
                "target_uncertainty_percent": self.target_uncertainty_percent,
                "target_uncertainty_fraction": self.target_uncertainty_fraction,
                "uncertainty_percentile": self.uncertainty_percentile,
                "max_batches_per_roi": self.max_batches_per_roi,
                "random_seed_base": self.random_seed_base,
                "start_new_process": self.start_new_process,
            },
            "physics": {
                "density_tolerance_gcm3": self.density_tolerance_gcm3,
                "world_margin_scale": self.world_margin_scale,
                "world_min_size_mm": self.world_min_size_mm,
                "electron_production_cut_mm": self.electron_production_cut_mm,
                "variance_reduction": self.variance_reduction,
            },
            "paths": {
                "masks": mask_paths,
                "doses": dose_paths,
                "uncertainties": unc_paths,
                "sum_dose": sum_path,
                "sum_uncertainty": sum_unc_path,
                "material_labels": mat_label_path,
            },
            "roi_metadata": roi_meta,
        }
        metadata = build_stage_metadata(
            stage_name="opengate_simulation_stage",
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
        Execute OpenGATE dosimetry for all requested ROIs.

        If the expected per-ROI dose NIfTI already exists on disk the simulation for
        that ROI is skipped and the file is loaded directly. This lets a crashed run
        resume from where it left off without re-simulating completed ROIs.

        Returns
        -------
        context : Context-like
        """
        assert_stage_rerun_safe(
            stage_name="opengate_simulation_stage",
            metadata_path=self.metadata_path,
            required_outputs=self._cache_marker_paths(),
            current_config_snapshot=self._rerun_config_snapshot(),
            current_ct_identity=self.ct_input_identity,
            current_upstream_fingerprints=self._current_dependency_fingerprints(),
            context=self.context,
        )
        if self.context.stage_skipped:
            self.context.dosimetry_output_dir = self.output_dir
            self.context.dosimetry_stage_output_dir = self.stage_output_dir
            self.context.dosimetry_work_dir = self.work_dir
            self.context.dosimetry_metadata_path = self.metadata_path
            meta = load_json(self.metadata_path)
            paths = meta.get("paths", {})
            self.context.dosimetry_raw_dose_paths = paths.get("doses", {})
            self.context.dosimetry_raw_uncertainty_paths = paths.get("uncertainties", {})
            self.context.dosimetry_sum_dose_path = paths.get("sum_dose")
            self.context.dosimetry_sum_uncertainty_path = paths.get("sum_uncertainty")
            self.context.dosimetry_material_label_path = paths.get("material_labels")
            self.context.dosimetry_mask_paths = paths.get("masks", {})
            gate_meta = meta.get("gate", {}) if isinstance(meta.get("gate"), dict) else {}
            simulated_roi_names = list(meta.get("simulated_roi_names", []))
            requested_roi_subset = list(meta.get("requested_roi_subset", self.requested_roi_subset))
            skipped_zero_voxel_rois = meta.get("skipped_zero_voxel_rois")
            if skipped_zero_voxel_rois is None:
                skipped_zero_voxel_rois = [
                    roi for roi in requested_roi_subset
                    if roi not in simulated_roi_names
                ]
                meta["skipped_zero_voxel_rois"] = skipped_zero_voxel_rois
                write_json(self.metadata_path, meta)
            self.context.extras["opengate_simulation_stage"] = {
                "simulated_roi_names": simulated_roi_names,
                "requested_roi_subset": requested_roi_subset,
                "skipped_zero_voxel_rois": list(skipped_zero_voxel_rois),
                "simulated_actual_histories_this_run": 0,
                "actual_histories_per_batch": gate_meta.get("actual_histories_per_batch"),
                "target_uncertainty_percent": gate_meta.get("target_uncertainty_percent"),
                "uncertainty_percentile": gate_meta.get("uncertainty_percentile"),
                "nonconverged_rois": [],
                "effective_num_threads": gate_meta.get("effective_num_threads", gate_meta.get("num_threads")),
                "history_rounding_loss_per_batch": gate_meta.get("history_rounding_loss_per_batch"),
                "dose_units": meta.get("dose_units", "Gy/decay"),
                "metadata_path": self.metadata_path,
            }
            return self.context

        # Lazy import: opengate is only available on compatible platforms.
        # Importing it here (rather than at module level) lets the rest of the
        # pipeline load even when opengate_core cannot be dlopen'd.
        global gate
        import opengate as gate  # noqa: F811

        native_ct = sitk.ReadImage(str(self.ct_nii_path))
        native_seg = sitk.ReadImage(str(self.vtt_roi_seg_path))

        sim_ct, sim_seg, sim_ct_path, sim_seg_path, was_resampled = self._prepare_simulation_images(
            native_ct,
            native_seg,
        )
        dose_ref_img = self._original_sim_ct_img
        if dose_ref_img is None:
            raise RuntimeError("OpenGATE simulation reference grid was not initialized.")

        mask_paths, roi_counts, roi_names = self._build_source_masks(sim_ct, sim_seg)

        dose_paths: Dict[str, str] = {}
        unc_paths: Dict[str, str] = {}
        roi_meta: Dict[str, Any] = {}
        sum_arr: Optional[np.ndarray] = None
        sum_uncertainty_var_arr: Optional[np.ndarray] = None
        mat_label_path: Optional[str] = None
        simulated_actual_histories_this_run = 0
        nonconverged_rois: List[str] = []

        for idx, roi_name in enumerate(roi_names):
            expected_nii = self._roi_dose_nii_path(roi_name)
            expected_unc_nii = self._roi_unc_nii_path(roi_name)

            # ----------------------------------------------------------
            # Resume logic: skip simulation if output already exists.
            # ----------------------------------------------------------
            if expected_nii.exists():
                print(f"[OpenGateSimulationStage] '{roi_name}': output exists, skipping simulation.")
                dose_img = sitk.ReadImage(str(expected_nii))
                if not self._same_image_grid(dose_img, dose_ref_img):
                    raise ValueError(
                        f"Existing dose map for '{roi_name}' does not match the current "
                        "OpenGATE simulation grid. Delete the old OpenGATE outputs or choose "
                        "a fresh output folder before rerunning."
                    )
                dose_grid = sitk.GetArrayFromImage(dose_img).astype(np.float64)
                dose_paths[roi_name] = str(expected_nii)
                if expected_unc_nii.exists():
                    unc_paths[roi_name] = str(expected_unc_nii)
                    unc_grid = sitk.GetArrayFromImage(sitk.ReadImage(str(expected_unc_nii))).astype(np.float64)
                    unc_contrib = self._relative_uncertainty_variance_numerator(unc_grid, dose_grid)
                    sum_uncertainty_var_arr = (
                        unc_contrib.copy()
                        if sum_uncertainty_var_arr is None
                        else sum_uncertainty_var_arr + unc_contrib
                    )
                roi_meta[roi_name] = {"skipped": True, "loaded_from": str(expected_nii)}
                sum_arr = dose_grid.copy() if sum_arr is None else sum_arr + dose_grid
                continue

            # ----------------------------------------------------------
            # Normal path: run adaptive batches for this ROI.
            # ----------------------------------------------------------
            res = self._run_roi_batches(
                roi_name,
                mask_paths[roi_name],
                sim_ct,
                sim_ct_path,
                save_labels=(idx == 0),
            )

            # Preserve the selected simulation grid. If xyz_spacing_mm requested a
            # coarser grid, saving directly here avoids an extra interpolation back
            # to native CT resolution.
            dose_grid = np.asarray(res["dose_arr"], dtype=np.float64)
            unc_grid = np.asarray(res["unc_arr"], dtype=np.float64)
            sum_arr = dose_grid.copy() if sum_arr is None else sum_arr + dose_grid
            unc_contrib = self._relative_uncertainty_variance_numerator(unc_grid, dose_grid)
            sum_uncertainty_var_arr = (
                unc_contrib.copy()
                if sum_uncertainty_var_arr is None
                else sum_uncertainty_var_arr + unc_contrib
            )
            simulated_actual_histories_this_run += int(res["total_actual_histories"])
            if not res["converged"]:
                nonconverged_rois.append(roi_name)

            if self.save_per_roi_dose_maps:
                dose_paths[roi_name] = save_nii_sitk(
                    dose_ref_img,
                    dose_grid.astype(np.float32),
                    expected_nii,
                )

            if self.save_uncertainty_map:
                unc_paths[roi_name] = save_nii_sitk(
                    dose_ref_img,
                    unc_grid.astype(np.float32),
                    expected_unc_nii,
                )

            if mat_label_path is None and res["material_label_path"]:
                mat_label_path = res["material_label_path"]

            roi_meta[roi_name] = {
                "work_dir": str(Path(self.work_dir) / roi_name),
                "num_batches": res["num_batches"],
                "converged": res["converged"],
                "cap_reached": res["cap_reached"],
                "total_actual_histories": res["total_actual_histories"],
                "final_uncertainty_percentile": self.uncertainty_percentile,
                "final_uncertainty_percentile_fraction": res["final_uncertainty_percentile_fraction"],
                "final_uncertainty_percentile_percent": res["final_uncertainty_percentile_percent"],
                "target_uncertainty_percent": self.target_uncertainty_percent,
                "batches": res["batches"],
                "size": res["size"],
                "spacing": res["spacing"],
                "n_materials": res["n_materials"],
                "mat_table": res["mat_table"],
                "den_table": res["den_table"],
            }

        sum_path: Optional[str] = None
        if self.save_summed_dose_map and sum_arr is not None:
            sum_path = save_nii_sitk(
                dose_ref_img,
                sum_arr.astype(np.float32),
                Path(self.phase_output_dir) / f"{self.prefix}_dose_sum.nii.gz",
            )

        sum_unc_path: Optional[str] = None
        if self.save_uncertainty_map and sum_arr is not None and sum_uncertainty_var_arr is not None:
            sum_unc_arr = self._combine_relative_uncertainty(sum_arr, sum_uncertainty_var_arr)
            sum_unc_path = save_nii_sitk(
                dose_ref_img,
                sum_unc_arr.astype(np.float32),
                Path(self.phase_output_dir) / f"{self.prefix}_dose_sum_unc.nii.gz",
            )

        self._save_metadata(
            roi_names=roi_names,
            roi_counts=roi_counts,
            roi_meta=roi_meta,
            mask_paths=mask_paths,
            dose_paths=dose_paths,
            unc_paths=unc_paths,
            sum_path=sum_path,
            sum_unc_path=sum_unc_path,
            mat_label_path=mat_label_path,
            was_resampled=was_resampled,
            sim_ct_path=str(sim_ct_path),
            sim_seg_path=str(sim_seg_path),
        )

        self.context.dosimetry_output_dir = self.output_dir
        self.context.dosimetry_stage_output_dir = self.stage_output_dir
        self.context.dosimetry_work_dir = self.work_dir
        self.context.dosimetry_metadata_path = self.metadata_path
        self.context.dosimetry_mask_paths = mask_paths
        self.context.dosimetry_raw_dose_paths = dose_paths
        self.context.dosimetry_raw_uncertainty_paths = unc_paths
        self.context.dosimetry_sum_dose_path = sum_path
        self.context.dosimetry_sum_uncertainty_path = sum_unc_path
        self.context.dosimetry_material_label_path = mat_label_path
        self.context.extras["opengate_simulation_stage"] = {
            "simulated_roi_names": list(roi_names),
            "requested_roi_subset": list(self.requested_roi_subset),
            "skipped_zero_voxel_rois": [
                roi for roi in self.requested_roi_subset
                if roi not in roi_names
            ],
            "simulated_actual_histories_this_run": simulated_actual_histories_this_run,
            "actual_histories_per_batch": self.actual_histories_per_batch,
            "target_uncertainty_percent": self.target_uncertainty_percent,
            "uncertainty_percentile": self.uncertainty_percentile,
            "nonconverged_rois": nonconverged_rois,
            "effective_num_threads": self.num_threads,
            "history_rounding_loss_per_batch": self.history_rounding_loss_per_batch,
            "dose_units": "Gy/decay",
            "metadata_path": self.metadata_path,
        }
        return self.context
