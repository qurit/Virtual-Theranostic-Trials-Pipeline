"""
OpenGATE dosimetry stage for the Theranostic Digital Twin (TDT) pipeline.

This stage runs voxel-source Monte Carlo dose calculations on the phase-1 CT grid
(using the native CT grid or an optional downsampled simulation grid) with OpenGATE.
Each requested ROI is simulated independently as a binary voxel source, then the
resulting dose maps are optionally resampled back to the native CT space and summed.

Now part of Phase 2 (Simulations) with its own independent ROI subset.

Core responsibilities
---------------------
- Validate required context fields and stage configuration.
- Load the phase-1 CT and unified TDT ROI segmentation.
- Optionally downsample the simulation inputs for faster Monte Carlo execution.
- Convert simulation inputs to OpenGATE's centered identity-direction convention.
- Build one binary voxel-source mask per requested ROI.
- Run one OpenGATE simulation per ROI using Lu-177 decay physics.
- Skip simulation for any ROI whose final dose NIfTI output already exists on disk
  (allows resuming a crashed run without re-simulating completed ROIs).
- Save per-ROI dose maps, optional uncertainty maps, and an optional summed dose map.
- Save stage metadata and optional OpenGATE material-label / MHD outputs.

Dose units
----------
OpenGATE returns dose per simulated primary. In this stage, each per-ROI dose map is
scaled by the actual total number of simulated histories, so the saved NIfTI outputs are:

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
- context.tdt_roi_seg_path : str
- context.downstream_roi_subset : list[str] | str
- context.config["phase_1"]["segmentation_stage"]["label_map_path"] : str

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

Maintainer / contact: pyazdi@bccrc.ca
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import opengate as gate
import SimpleITK as sitk
from json_minify import json_minify

_LU177_Z = 71
_LU177_A = 177


class OpenGateSimulationStage:
    """
    Run OpenGATE voxel-source dosimetry on the phase-1 CT grid.

    Notes
    -----
    - Simulation may run on the native CT grid or on an optional downsampled grid.
    - Source masks are binary voxel maps derived from the unified TDT segmentation.
    - ROI dose outputs are accumulated into a summed dose map in native CT space.
    - If a per-ROI dose NIfTI already exists on disk the simulation for that ROI is
      skipped, allowing a crashed run to be resumed from where it left off.
    """

    def __init__(self, context: Any) -> None:
        context.require(
            "subdir_paths",
            "config",
            "ct_nii_path",
            "tdt_roi_seg_path",
            "downstream_roi_subset",
        )
        self.context = context
        self.debug: bool = getattr(context, "mode", "").upper() == "DEBUG"

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
        self.metadata_path: str = os.path.join(self.work_dir, f"{self.prefix}_metadata.json")

        # Save / provenance options
        self.save_per_roi_dose_maps: bool = bool(self.stage_cfg.get("save_per_roi_dose_maps", True))
        self.save_summed_dose_map: bool = bool(self.stage_cfg.get("save_summed_dose_map", True))
        self.save_uncertainty_map: bool = bool(self.stage_cfg.get("save_uncertainty_map", True))
        self.save_material_label_image: bool = bool(self.stage_cfg.get("save_material_label_image", True))
        self.write_mhd_outputs: bool = bool(self.stage_cfg.get("write_mhd_outputs", False))

        # Optional in-plane downsampling for Monte Carlo acceleration
        self.xy_dim: Optional[int] = self.stage_cfg.get("xy_dim", None)
        if self.xy_dim is not None:
            self.xy_dim = int(self.xy_dim)
            if self.xy_dim <= 0:
                raise ValueError("xy_dim must be a positive integer")

        # Optional dose-actor grid override in mm (x, y, z)
        output_spacing = self.stage_cfg.get("output_dose_spacing_mm", None)
        if output_spacing is not None:
            self.output_dose_spacing_mm: Optional[List[float]] = [float(v) for v in output_spacing]
            if len(self.output_dose_spacing_mm) != 3 or any(v <= 0 for v in self.output_dose_spacing_mm):
                raise ValueError("output_dose_spacing_mm must be 3 positive floats [x, y, z]")
        else:
            self.output_dose_spacing_mm = None

        # OpenGATE execution controls
        gate_cfg = self.stage_cfg.get("gate", {})
        self.requested_total_histories: int = int(gate_cfg.get("total_histories", 100_000))
        self.requested_num_threads: int = int(gate_cfg.get("num_threads", 1))
        self.start_new_process: bool = bool(gate_cfg.get("start_new_process", True))
        self.random_seed: Any = gate_cfg.get("random_seed", "auto")

        if self.requested_total_histories <= 0:
            raise ValueError("gate.total_histories must be > 0")
        if self.requested_num_threads <= 0:
            raise ValueError("gate.num_threads must be > 0")

        # OpenGATE source.n is set per thread, so convert requested total histories
        # into a per-thread count to keep the total under control.
        self.num_threads: int = min(self.requested_num_threads, self.requested_total_histories)
        self.histories_per_thread: int = self.requested_total_histories // self.num_threads
        self.actual_total_histories: int = self.histories_per_thread * self.num_threads
        self.history_rounding_loss: int = self.requested_total_histories - self.actual_total_histories

        # Only Lu-177 is currently supported in this stage.
        source_cfg = self.stage_cfg.get("source", {})
        raw_isotope = str(source_cfg.get("isotope", "")).lower().strip()
        if raw_isotope not in ("lu177", "lu-177"):
            raise ValueError(f"source.isotope must be 'lu177'. Got: '{raw_isotope}'")
        self.isotope_name: str = "lu177"

        # Physics / geometry controls
        self.variance_reduction: bool = bool(self.stage_cfg.get("variance_reduction", True))
        physics_cfg = self.stage_cfg.get("physics", {})
        self.density_tolerance_gcm3: float = float(physics_cfg.get("density_tolerance_gcm3", 0.05))
        self.world_margin_scale: float = float(physics_cfg.get("world_margin_scale", 1.4))
        self.world_min_size_mm: float = float(physics_cfg.get("world_min_size_mm", 400.0))
        self.electron_production_cut_mm: float = float(physics_cfg.get("electron_production_cut_mm", 0.1))

        # Input paths
        self.ct_nii_path: Path = Path(context.ct_nii_path)
        self.tdt_roi_seg_path: Path = Path(context.tdt_roi_seg_path)
        self.label_map_path: Path = Path(context.config["phase_1"]["segmentation_stage"]["label_map_path"]) 
        self.tdt_name2id: Dict[str, int] = self._load_tdt_label_map(self.label_map_path)

        # Build final ROI list from opengate_stage config (independent from phase_1) 
        opengate_roi_subset = self.stage_cfg.get("roi_subset")
        if opengate_roi_subset is None:
            opengate_roi_subset = getattr(context, "downstream_roi_subset", None)
        if isinstance(opengate_roi_subset, str):
            opengate_roi_subset = [opengate_roi_subset]
        if opengate_roi_subset is None:
            raise ValueError("OpenGATE roi_subset must be provided (in config or context)") 

        normalized_opengate_roi_subset: List[str] = []
        for roi_name in opengate_roi_subset:
            roi_name = str(roi_name).strip()
            if roi_name and roi_name not in normalized_opengate_roi_subset:
                normalized_opengate_roi_subset.append(roi_name)

        if "body" not in normalized_opengate_roi_subset:
            normalized_opengate_roi_subset.append("body")
        if getattr(context, "synthetic_lesions_enabled", False) and "synthetic_lesion" not in normalized_opengate_roi_subset:
            normalized_opengate_roi_subset.append("synthetic_lesion")

        # Validate OpenGATE ROI subset against phase_1 segmented ROIs 
        phase1_rois = set(getattr(context, "downstream_roi_subset", []) or [])         
        invalid_rois = [r for r in normalized_opengate_roi_subset if r not in phase1_rois and r != "body"] 
        if invalid_rois:                                                               
            raise ValueError(                                                          
                f"OpenGATE roi_subset contains ROIs not segmented in Phase 1: {invalid_rois}. " 
                f"Available: {sorted(phase1_rois)}"                                    
            )                                                                          

        roi_list: List[str] = []
        if "body" in self.tdt_name2id:
            roi_list.append("body")
        for roi_name in normalized_opengate_roi_subset:
            if roi_name not in roi_list:
                roi_list.append(roi_name)
        if not roi_list:
            raise ValueError("No ROI names found for dosimetry simulation")
        self.requested_roi_subset: List[str] = roi_list

        # Populated during image preparation and used later when restoring dose arrays.
        self._original_sim_ct_img: Optional[sitk.Image] = None
        self._centering_flipped_axes: List[int] = []

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        """Print a stage-local debug message only in DEBUG mode."""
        if self.debug:
            print(f"[OpenGateSimulationStage] {msg}")

    def _roi_dose_nii_path(self, roi_name: str) -> Path:
        """Return the expected final NIfTI output path for a given ROI dose map."""
        return Path(self.work_dir) / f"{self.prefix}_dose_{roi_name}.nii.gz"

    @staticmethod
    def _load_tdt_label_map(path: Path) -> Dict[str, int]:
        """Load the TDT_Pipeline label map as {roi_name: label_id}."""
        if not path.exists():
            raise FileNotFoundError(f"TDT label map not found: {path}")
        with open(path, encoding="utf-8") as f:
            data = json.loads(json_minify(f.read()))
        if "TDT_Pipeline" not in data:
            raise KeyError("Label map JSON missing 'TDT_Pipeline' key")
        return {name: int(label_id) for label_id, name in data["TDT_Pipeline"].items()}

    @staticmethod
    def _save_nii(ref: sitk.Image, arr: np.ndarray, path: Path) -> str:
        """Save a numpy array as NIfTI using `ref` geometry."""
        img = sitk.GetImageFromArray(np.asarray(arr))
        img.CopyInformation(ref)
        sitk.WriteImage(img, str(path), imageIO="NiftiImageIO")
        return str(path)

    @staticmethod
    def _cleanup_mhd(path: Path) -> None:
        """Delete an MHD file and its paired raw/zraw payload if they exist."""
        for candidate in [path, path.with_suffix(".raw"), path.with_suffix(".zraw")]:
            try:
                candidate.unlink(missing_ok=True)
            except Exception:
                pass

    @staticmethod
    def _find_hu_tables() -> Tuple[Path, Path]:
        """Locate OpenGATE Schneider HU-to-material conversion tables."""
        pkg = Path(gate.__file__).resolve().parent
        for root in [pkg, pkg.parent, *list(pkg.parents[:4])]:
            for rel in [Path("tests/data"), Path("opengate/tests/data")]:
                mat_table = root / rel / "Schneider2000MaterialsTable.txt"
                den_table = root / rel / "Schneider2000DensitiesTable.txt"
                if mat_table.exists() and den_table.exists():
                    return mat_table, den_table
        raise FileNotFoundError("Could not locate OpenGATE Schneider HU tables")

    @staticmethod
    def _to_centered_identity(img: sitk.Image, is_label: bool = False) -> Tuple[sitk.Image, List[int]]:
        """
        Convert an image to OpenGATE's centered identity-direction convention.

        Behavior
        --------
        - Axes with negative direction cosines on the diagonal are flipped in numpy space.
        - Output direction is set to identity.
        - Output origin is set so the image is centered at the world origin.

        Returns
        -------
        (converted_image, flipped_numpy_axes)
        """
        size = np.array(img.GetSize(), dtype=np.float64)
        spacing = np.array(img.GetSpacing(), dtype=np.float64)
        direction = np.array(img.GetDirection()).reshape(3, 3)

        arr = sitk.GetArrayFromImage(img)
        if is_label:
            arr = arr.astype(np.int32)

        flipped_axes: List[int] = []
        col_to_npy = {0: 2, 1: 1, 2: 0}  # sitk (x,y,z) -> numpy (z,y,x)
        for col in range(3):
            if direction[col, col] < 0:
                np_axis = col_to_npy[col]
                arr = np.flip(arr, axis=np_axis).copy()
                flipped_axes.append(np_axis)

        out = sitk.GetImageFromArray(arr)
        out.SetOrigin((-(size * spacing) / 2.0 + spacing / 2.0).tolist())
        out.SetSpacing(spacing.tolist())
        out.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
        return out, flipped_axes

    # ------------------------------------------------------------------
    # image preparation
    # ------------------------------------------------------------------

    def _compute_resampled_geometry(self, img: sitk.Image) -> Optional[Tuple[Tuple[int, ...], Tuple[float, ...]]]:
        """
        Compute an isotropically scaled in-plane geometry if xy downsampling is requested.

        Returns None when downsampling is disabled or unnecessary.
        """
        if self.xy_dim is None:
            return None

        sx, sy, sz = img.GetSize()
        spx, spy, spz = img.GetSpacing()

        if sx <= self.xy_dim and sy <= self.xy_dim:
            return None

        scale = min(self.xy_dim / sx, self.xy_dim / sy)
        nx = max(1, round(sx * scale))
        ny = max(1, round(sy * scale))
        nz = max(1, round(sz * scale))
        return (nx, ny, nz), (spx * sx / nx, spy * sy / ny, spz * sz / nz)

    @staticmethod
    def _resample(
        img: sitk.Image,
        size: Tuple[int, ...],
        spacing: Tuple[float, ...],
        is_label: bool = False,
    ) -> sitk.Image:
        """Resample an image onto a new grid, using NN for labels and linear for CT."""
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(list(spacing))
        resampler.SetSize(list(size))
        resampler.SetOutputDirection(img.GetDirection())
        resampler.SetOutputOrigin(img.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(0)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
        return resampler.Execute(img)

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
        2) Store the pre-centered simulation CT geometry for later dose resampling.
        3) Convert both images to centered identity-direction form.
        4) Write the prepared images to disk and re-read them.
        """
        sim_ct_path = Path(self.resample_dir) / "ct_sim.nii.gz"
        sim_seg_path = Path(self.resample_dir) / "seg_sim.nii.gz"

        geometry = self._compute_resampled_geometry(ct)
        was_resampled = False
        if geometry is not None:
            new_size, new_spacing = geometry
            self._log(f"Downsampling to {new_size} (spacing {tuple(round(s, 2) for s in new_spacing)} mm)")
            ct = self._resample(ct, new_size, new_spacing, is_label=False)
            seg = self._resample(seg, new_size, new_spacing, is_label=True)
            was_resampled = True

        # Preserve the simulation-space geometry before centering so dose can later be
        # resampled or written back into a CT-aligned grid.
        self._original_sim_ct_img = sitk.GetImageFromArray(sitk.GetArrayFromImage(ct))
        self._original_sim_ct_img.CopyInformation(ct)

        ct_centered, self._centering_flipped_axes = self._to_centered_identity(ct)
        seg_centered, _ = self._to_centered_identity(seg, is_label=True)

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

    def _upsample_to_native(self, arr: np.ndarray, ref: sitk.Image, native: sitk.Image) -> np.ndarray:
        """Resample a simulation-space dose/uncertainty array back to the native CT grid."""
        img = sitk.GetImageFromArray(arr.astype(np.float32))
        img.CopyInformation(ref)

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(native)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0.0)
        resampler.SetTransform(sitk.Transform())
        return sitk.GetArrayFromImage(resampler.Execute(img)).astype(np.float32)

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

        Returns
        -------
        mask_paths : dict[str, str]
        counts : dict[str, int]
        names : list[str]
            Only ROIs with non-zero voxels are returned.
        """
        seg_arr = sitk.GetArrayFromImage(sim_seg).astype(np.int32)
        mask_paths: Dict[str, str] = {}
        counts: Dict[str, int] = {}
        names: List[str] = []

        for roi_name in self.requested_roi_subset:
            label_id = self.tdt_name2id.get(roi_name)
            if label_id is None:
                raise ValueError(f"ROI '{roi_name}' not in TDT label map")

            mask = (seg_arr == int(label_id)).astype(np.uint8)
            n_vox = int(np.count_nonzero(mask))
            if n_vox == 0:
                self._log(f"Skipping '{roi_name}': zero voxels")
                continue

            mask_path = Path(self.source_mask_dir) / f"{self.prefix}_{roi_name}_source_mask.nii.gz"
            self._save_nii(sim_ct, mask, mask_path)
            mask_paths[roi_name] = str(mask_path)
            counts[roi_name] = n_vox
            names.append(roi_name)
            self._log(f"Mask '{roi_name}': {n_vox} voxels")

        if not names:
            raise ValueError("No ROIs had voxels in segmentation")
        return mask_paths, counts, names

    # ------------------------------------------------------------------
    # single-ROI simulation
    # ------------------------------------------------------------------

    def _run_single_roi(
        self,
        roi_name: str,
        mask_path: str,
        sim_ct_path: Path,
        save_labels: bool,
    ) -> Dict[str, Any]:
        """
        Run one OpenGATE simulation for a single ROI voxel source.

        Returns a dictionary containing raw arrays, file paths, and provenance fields.
        """
        roi_dir = Path(self.work_dir) / roi_name
        roi_dir.mkdir(exist_ok=True, parents=True)

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
        sim.random_seed = self.random_seed
        sim.output_dir = roi_dir

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

        mat_table, den_table = self._find_hu_tables()
        voxel_materials, generated_materials = gate.geometry.materials.HounsfieldUnit_to_material(
            sim,
            self.density_tolerance_gcm3 * gcm3,
            str(mat_table),
            str(den_table),
        )
        patient.voxel_materials = voxel_materials

        label_mhd: Optional[Path] = None
        if save_labels and self.save_material_label_image:
            label_mhd = roi_dir / "patient_material_labels.mhd"
            patient.dump_label_image = str(label_mhd)

        # Lu-177 voxel source. `src.n` is per thread, so histories_per_thread is used.
        src = sim.add_source("VoxelSource", "src_lu177")
        src.particle = f"ion {_LU177_Z} {_LU177_A} 0 0"
        src.n = self.histories_per_thread
        src.image = str(mask_path)
        src.direction.type = "iso"
        src.energy.type = "mono"
        src.energy.mono = 0

        stats = sim.add_actor("SimulationStatisticsActor", "Stats")
        stats.output_filename = "stats.txt"

        dose = sim.add_actor("DoseActor", "dose")
        dose.attached_to = patient
        if self.output_dose_spacing_mm is not None:
            dose.spacing = [float(s) * mm for s in self.output_dose_spacing_mm]
            dose.size = [
                max(1, round(size_xyz[i] * spacing_xyz[i] / self.output_dose_spacing_mm[i]))
                for i in range(3)
            ]
        else:
            dose.size = list(size_xyz)
            dose.spacing = [float(s) * mm for s in spacing_xyz]
        dose.dose.active = True
        dose.dose.output_filename = "dose_map.mhd"
        dose.dose_uncertainty.active = self.save_uncertainty_map
        if self.save_uncertainty_map:
            dose.dose_uncertainty.output_filename = "dose_uncertainty.mhd"

        sim.run(start_new_process=self.start_new_process)

        dose_mhd = roi_dir / "dose_map.mhd"
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

        unc_arr: Optional[np.ndarray] = None
        unc_mhd: Optional[Path] = None
        if self.save_uncertainty_map:
            unc_mhd = roi_dir / "dose_uncertainty.mhd"
            if unc_mhd.exists():
                unc_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(unc_mhd))).astype(np.float32)
                for np_axis in reversed(self._centering_flipped_axes):
                    unc_arr = np.flip(unc_arr, axis=np_axis).copy()

        if self.debug:
            with open(roi_dir / "coordinate_debug.json", "w", encoding="utf-8") as f:
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
            "roi_work_dir": str(roi_dir),
            "dose_mhd_path": str(dose_mhd),
            "unc_mhd_path": None if unc_mhd is None else str(unc_mhd),
            "label_mhd_path": None if label_mhd is None else str(label_mhd),
            "dose_arr": dose_arr,
            "unc_arr": unc_arr,
            "size_xyz": tuple(int(v) for v in size_xyz),
            "spacing_xyz": tuple(float(v) for v in spacing_xyz),
            "stats_str": str(stats),
            "stats_events": stats_events,
            "mat_table": str(mat_table),
            "den_table": str(den_table),
            "n_materials": len(generated_materials),
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
        mat_label_path: Optional[str],
        was_resampled: bool,
        sim_ct_path: str,
        sim_seg_path: str,
    ) -> None:
        """Save stage metadata for provenance, debugging, and rerun reproducibility."""
        metadata: Dict[str, Any] = {
            "stage": "opengate_simulation_stage",
            "dose_units": "Gy/decay",
            "ct_nii_path": str(self.ct_nii_path),
            "tdt_roi_seg_path": str(self.tdt_roi_seg_path),
            "label_map_path": str(self.label_map_path),
            "requested_roi_subset": list(self.requested_roi_subset),
            "simulated_roi_names": list(roi_names),
            "roi_voxel_counts": {k: int(v) for k, v in roi_counts.items()},
            "downsampling": {
                "xy_dim": self.xy_dim,
                "was_resampled": was_resampled,
                "sim_ct_path": sim_ct_path,
                "sim_seg_path": sim_seg_path,
            },
            "centering": {
                "flipped_axes": self._centering_flipped_axes,
                "original_origin": [round(o, 2) for o in self._original_sim_ct_img.GetOrigin()],
            },
            "isotope": self.isotope_name,
            "output_dose_spacing_mm": self.output_dose_spacing_mm,
            "save_options": {
                "save_per_roi_dose_maps": self.save_per_roi_dose_maps,
                "save_summed_dose_map": self.save_summed_dose_map,
                "save_uncertainty_map": self.save_uncertainty_map,
                "save_material_label_image": self.save_material_label_image,
                "write_mhd_outputs": self.write_mhd_outputs,
            },
            "gate": {
                "requested_total_histories": self.requested_total_histories,
                "requested_num_threads": self.requested_num_threads,
                "actual_total_histories": self.actual_total_histories,
                "num_threads": self.num_threads,
                "histories_per_thread": self.histories_per_thread,
                "history_rounding_loss": self.history_rounding_loss,
                "random_seed": self.random_seed,
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
                "material_labels": mat_label_path,
            },
            "roi_metadata": roi_meta,
        }
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

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
        native_ct = sitk.ReadImage(str(self.ct_nii_path))
        native_seg = sitk.ReadImage(str(self.tdt_roi_seg_path))

        sim_ct, sim_seg, sim_ct_path, sim_seg_path, was_resampled = self._prepare_simulation_images(
            native_ct,
            native_seg,
        )

        mask_paths, roi_counts, roi_names = self._build_source_masks(sim_ct, sim_seg)

        dose_paths: Dict[str, str] = {}
        unc_paths: Dict[str, str] = {}
        roi_meta: Dict[str, Any] = {}
        sum_arr: Optional[np.ndarray] = None
        mat_label_path: Optional[str] = None

        needs_upsample = (
            was_resampled
            or self.output_dose_spacing_mm is not None
            or self._original_sim_ct_img.GetSize() != native_ct.GetSize()
        )

        for idx, roi_name in enumerate(roi_names):
            expected_nii = self._roi_dose_nii_path(roi_name)

            # ----------------------------------------------------------
            # Resume logic: skip simulation if output already exists.
            # ----------------------------------------------------------
            if expected_nii.exists():
                print(f"[OpenGateSimulationStage] '{roi_name}': output exists, skipping simulation.")
                dose_native = sitk.GetArrayFromImage(sitk.ReadImage(str(expected_nii))).astype(np.float64)
                dose_paths[roi_name] = str(expected_nii)
                unc_nii = Path(self.work_dir) / f"{self.prefix}_unc_{roi_name}.nii.gz"
                if unc_nii.exists():
                    unc_paths[roi_name] = str(unc_nii)
                roi_meta[roi_name] = {"skipped": True, "loaded_from": str(expected_nii)}
                sum_arr = dose_native.copy() if sum_arr is None else sum_arr + dose_native
                continue

            # ----------------------------------------------------------
            # Normal path: run the simulation for this ROI.
            # ----------------------------------------------------------
            res = self._run_single_roi(
                roi_name,
                mask_paths[roi_name],
                sim_ct_path,
                save_labels=(idx == 0),
            )

            # Convert Gy/primary -> Gy/decay using the actual simulated history count.
            dose_sim = np.asarray(res["dose_arr"], dtype=np.float64) / self.actual_total_histories

            dose_native = (
                self._upsample_to_native(
                    dose_sim.astype(np.float32),
                    self._original_sim_ct_img,
                    native_ct,
                ).astype(np.float64)
                if needs_upsample
                else dose_sim
            )

            sum_arr = dose_native.copy() if sum_arr is None else sum_arr + dose_native

            if self.save_per_roi_dose_maps:
                dose_paths[roi_name] = self._save_nii(
                    native_ct,
                    dose_native.astype(np.float32),
                    expected_nii,
                )

            if self.save_uncertainty_map and res["unc_arr"] is not None:
                unc_out = (
                    self._upsample_to_native(res["unc_arr"], self._original_sim_ct_img, native_ct)
                    if needs_upsample
                    else res["unc_arr"]
                )
                unc_paths[roi_name] = self._save_nii(
                    native_ct,
                    unc_out,
                    Path(self.work_dir) / f"{self.prefix}_unc_{roi_name}.nii.gz",
                )

            # Save the material label image from the first successful ROI only.
            if mat_label_path is None and self.save_material_label_image and res["label_mhd_path"]:
                if os.path.exists(res["label_mhd_path"]):
                    labels = sitk.GetArrayFromImage(sitk.ReadImage(res["label_mhd_path"])).astype(np.int16)
                    mat_label_path = self._save_nii(
                        sim_ct,
                        labels,
                        Path(self.work_dir) / f"{self.prefix}_material_labels.nii.gz",
                    )

            if not self.write_mhd_outputs:
                self._cleanup_mhd(Path(res["dose_mhd_path"]))
                if res["unc_mhd_path"]:
                    self._cleanup_mhd(Path(res["unc_mhd_path"]))
                if res["label_mhd_path"]:
                    self._cleanup_mhd(Path(res["label_mhd_path"]))

            roi_meta[roi_name] = {
                "work_dir": res["roi_work_dir"],
                "stats": res["stats_str"],
                "events": res["stats_events"],
                "size": list(res["size_xyz"]),
                "spacing": list(res["spacing_xyz"]),
                "n_materials": res["n_materials"],
                "mat_table": res["mat_table"],
                "den_table": res["den_table"],
            }

        sum_path: Optional[str] = None
        if self.save_summed_dose_map and sum_arr is not None:
            sum_path = self._save_nii(
                native_ct,
                sum_arr.astype(np.float32),
                Path(self.output_dir) / f"{self.prefix}_dose_sum.nii.gz",
            )

        self._save_metadata(
            roi_names=roi_names,
            roi_counts=roi_counts,
            roi_meta=roi_meta,
            mask_paths=mask_paths,
            dose_paths=dose_paths,
            unc_paths=unc_paths,
            sum_path=sum_path,
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
        self.context.dosimetry_material_label_path = mat_label_path
        self.context.extras["opengate_simulation_stage"] = {
            "simulated_roi_names": list(roi_names),
            "requested_roi_subset": list(self.requested_roi_subset),
            "actual_total_histories": self.actual_total_histories,
            "effective_num_threads": self.num_threads,
            "history_rounding_loss": self.history_rounding_loss,
            "dose_units": "Gy/decay",
            "metadata_path": self.metadata_path,
        }
        return self.context
