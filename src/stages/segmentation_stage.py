"""
TotalSegmentator-based segmentation and ROI unification for the VTT pipeline.

This stage standardizes the CT input to NIfTI, runs the required
TotalSegmentator tasks for the requested ROI set, and combines the resulting
masks into a single multilabel volume in the shared VTT label space.

Expected Context interface
--------------------------
Incoming `context` is expected to provide:
- context.ct_input_path : str
- context.config["phase_1"]["segmentation_stage"]["roi_subset"] : list[str]
- context.config["phase_1"]["segmentation_stage"]["file_prefix"] : str
- context.config["phase_1"]["segmentation_stage"]["unification_prefix"] : str
- label map loaded from ``src/data/pipeline_paths.json`` input_paths.label_map_path
- context.subdir_paths["phase_1"] : str

On success, this stage sets:
- context.ct_nii_path : str
- context.body_ml_path : str
- context.total_ml_path : Optional[str]  (None if not required by requested ROIs)
- context.head_glands_cavities_ml_path : Optional[str]  (None if not required)
- context.totseg_plan : dict  (which tasks ran, which roi_subsets were passed to TotalSegmentator)
- context.vtt_roi_seg_path : str  (path to unified multilabel NIfTI handoff)
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, TypedDict, Union

import dicom2nifti
import torch
import numpy as np                                                                     
import nibabel as nib                                                                  
import SimpleITK as sitk
from torch.cuda.amp import GradScaler as _GradScaler
from totalsegmentator.python_api import totalsegmentator
from json_minify import json_minify

from src.io.config_paths import get_label_map_path
from src.io.rerun_guard import (
    assert_stage_rerun_safe,
    build_stage_metadata,
    build_segmentation_rerun_snapshot,
    fingerprint_optional_file,
    stage_metadata_path,
    write_json,
)
from src.utils.nifti_utils import load_int_seg

# ---------------------------------------------------------------------------
# Compatibility: some TotalSegmentator/torch combinations expect `torch.GradScaler`.
# ---------------------------------------------------------------------------
torch.GradScaler = _GradScaler


CTInputType = Literal["nii", "dicom"]


def _nifti_gz_intact(path: str) -> bool:
    """Return True if a .nii.gz file can be fully decompressed without error."""
    import gzip
    try:
        with gzip.open(path, "rb") as f:
            while f.read(1 << 20):
                pass
        return True
    except (EOFError, OSError):
        return False


class TotSegPlan(TypedDict):
    """Execution plan describing which TotalSegmentator tasks will run."""
    run_body: bool
    run_total: bool
    run_head_glands_cavities: bool
    total_roi_subset: List[str]   # TotalSegmentator ROI names for the total task
    head_roi_subset: List[str]    # TotalSegmentator ROI names for head_glands_cavities task
    vtt_roi_subset: List[str]     # User-facing VTT ROI names (passed in from config)


class SegmentationStage:
    """
    VTT segmentation stage: TotalSegmentator + ROI unification.

    Always runs the `body` task (used downstream as a patient mask for all ROI operations).
    The `total` and `head_glands_cavities` tasks run only if needed by the requested ROIs.
    After segmentation, combines outputs into a single multilabel volume in VTT label space.

    Parameters
    ----------
    context : Context-like
        Pipeline context object. Must provide `ct_input_path`, `config`, and `subdir_paths`.
    """

    def __init__(self, context: Any) -> None:
        context.require("ct_input_path", "config", "subdir_paths", "output_folder_path", "ct_input_identity")
        self.context = context
        self.debug: bool = getattr(context, "mode", "").upper() == "DEBUG"             

        self.ct_input_path: str = context.ct_input_path
        self.ct_input_identity: Dict[str, Any] = context.ct_input_identity
        self.roi_subset: Union[str, Sequence[str]] = context.config["phase_1"]["segmentation_stage"]["roi_subset"]
        self.ml: bool = True  # always use multilabel output

        self.phase_output_dir: str = context.subdir_paths["phase_1"]
        self.output_dir: str = os.path.join(self.phase_output_dir, "segmentation_stage")
        self.work_dir: str = os.path.join(self.output_dir, "work_dir")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.work_dir, exist_ok=True)

        self.prefix: str = context.config["phase_1"]["segmentation_stage"]["file_prefix"]
        self.unification_prefix: str = context.config["phase_1"]["segmentation_stage"]["unification_prefix"] 

        self.ct_nii_path: Optional[str] = None
        self.body_ml_path: Optional[str] = None
        self.head_glands_cavities_ml_path: Optional[str] = None
        self.total_ml_path: Optional[str] = None
        self.metadata_path: str = stage_metadata_path(context.output_folder_path, "segmentation_stage")

        # Phase handoff: overwritten by every run so downstream stages always have a clean path. 
        self.final_output_path: str = os.path.join(self.phase_output_dir, "digital_twin.nii.gz") 
        self.stage_output_path: str = os.path.join(self.output_dir, f"{self.unification_prefix}.nii.gz") 
        self.unification_metadata_path: str = os.path.join(self.output_dir, f"{self.unification_prefix}_metadata.json") 

        # Load label map for ROI unification
        self.ts_map_path: str = get_label_map_path()
        if not os.path.exists(self.ts_map_path):
            raise FileNotFoundError(f"Class map json not found: {self.ts_map_path}") 

        with open(self.ts_map_path, encoding="utf-8") as f: 
            ts_map_json: Dict[str, Dict[str, str]] = json.loads(json_minify(f.read())) 

        # Convert all JSON maps from {str(id): name} to {name: int(id)} for lookup.
        self.total_name2id: Dict[str, int] = {
            name: int(lab) for lab, name in ts_map_json["total"].items()
        }
        self.head_name2id: Dict[str, int] = {
            name: int(lab) for lab, name in ts_map_json["head_glands_cavities"].items()
        }
        self.vtt_name2id: Dict[str, int] = {
            name: int(lab) for lab, name in ts_map_json["VTT_Pipeline"].items()
        }
        # VTT_allowed_rois and VTT_to_totseg live in pipeline_roi_naming_map.json — loaded
        # here so the stage has no hardcoded ROI or TotalSegmentator coupling.
        self.vtt_allowed_rois: set = set(ts_map_json["VTT_allowed_rois"])
        self.vtt_to_totseg: Dict[str, Any] = ts_map_json["VTT_to_totseg"]

    def _rerun_config_snapshot(self, plan: TotSegPlan) -> Dict[str, Any]:
        """Return the config subset that must match for cached outputs to remain valid."""
        return build_segmentation_rerun_snapshot(self.context.config)

    def _current_upstream_fingerprints(self) -> Dict[str, Any]:
        """Return fingerprints for small upstream inputs that affect segmentation outputs."""
        return {
            "label_map_json": fingerprint_optional_file(self.ts_map_path),
        }

    # ------------------------------------------------------------------
    # helpers — segmentation
    # ------------------------------------------------------------------

    def _standardize_ct_to_nifti(self) -> None:
        """
        Convert/copy the CT input into a standardized NIfTI at <phase_output_dir>/ct.nii.gz.

        Behavior
        --------
        - DICOM directory -> converted via dicom2nifti (reoriented).
        - Existing NIfTI (.nii/.nii.gz) -> re-written via SimpleITK for consistent downstream paths.
        - If output already exists, this is a no-op.
        """
        self.ct_nii_path = os.path.join(self.phase_output_dir, "ct.nii.gz")
        if os.path.exists(self.ct_nii_path):
            if _nifti_gz_intact(self.ct_nii_path):
                return
            os.remove(self.ct_nii_path)

        if os.path.isdir(self.ct_input_path):
            dicom2nifti.dicom_series_to_nifti(
                self.ct_input_path,
                self.ct_nii_path,
                reorient_nifti=True,
            )
        else:
            lower_input = self.ct_input_path.lower()
            if lower_input.endswith((".nii", ".nii.gz")):
                sitk.WriteImage(sitk.ReadImage(self.ct_input_path), self.ct_nii_path, True)
            else:
                raise ValueError(
                    "Unsupported CT input. Provide a DICOM folder or a NIfTI file "
                    f"(.nii/.nii.gz). Got: {self.ct_input_path}"
                )

    def _pre_totalsegmentation_checks(self) -> TotSegPlan:
        """
        Validate the user ROI list and compute which TotalSegmentator tasks to run.

        Returns
        -------
        TotSegPlan

        Raises
        ------
        ValueError
            If ROI list is empty or contains unsupported names.
        """
        rois = self.roi_subset
        if isinstance(rois, str):
            rois = [rois]
        rois = [str(r).strip() for r in rois if str(r).strip()]

        if not rois:
            raise ValueError(
                f"roi_subset must contain at least one ROI from: {sorted(self.vtt_allowed_rois)}"
            )

        invalid = [r for r in rois if r not in self.vtt_allowed_rois]
        if invalid:
            raise ValueError(
                f"Invalid ROI(s): {invalid}. Allowed: {sorted(self.vtt_allowed_rois)}"
            )

        total_rois: List[str] = []
        head_rois: List[str] = []
        seen_total: set = set()
        seen_head: set = set()

        for r in rois:
            entry = self.vtt_to_totseg[r]
            task, expanded = entry["task"], entry["totseg_rois"]
            if task == "total":
                for x in expanded:
                    if x not in seen_total:
                        total_rois.append(x)
                        seen_total.add(x)
            elif task == "head_glands_cavities":
                for x in expanded:
                    if x not in seen_head:
                        head_rois.append(x)
                        seen_head.add(x)

        return TotSegPlan(
            run_body=True,  # body task always runs
            run_total=bool(total_rois),
            run_head_glands_cavities=bool(head_rois),
            total_roi_subset=total_rois,
            head_roi_subset=head_rois,
            vtt_roi_subset=rois,
        )

    def _files_exist(self) -> Tuple[bool, bool, bool]:
        """
        Check whether expected output mask files already exist on disk.

        Also populates the instance output path attributes as a side effect.

        Returns
        -------
        tuple[bool, bool, bool]
            (body_ml_done, head_glands_cavities_ml_done, total_ml_done)
        """
        self.body_ml_path = os.path.join(self.output_dir, f"{self.prefix}_body_ml.nii.gz")
        self.head_glands_cavities_ml_path = os.path.join(
            self.output_dir, f"{self.prefix}_head_glands_cavities_ml.nii.gz"
        )
        self.total_ml_path = os.path.join(self.output_dir, f"{self.prefix}_total_ml.nii.gz")

        return (
            os.path.exists(self.body_ml_path),
            os.path.exists(self.head_glands_cavities_ml_path),
            os.path.exists(self.total_ml_path),
        )

    # ------------------------------------------------------------------
    # helpers — ROI unification
    # ------------------------------------------------------------------


    def _assert_unification_inputs_exist(self, plan: TotSegPlan) -> None:              
        """
        Validate that all required input files exist based on the plan.

        Raises
        ------
        FileNotFoundError
        """
        if self.ct_nii_path is None or not os.path.exists(self.ct_nii_path):           
            raise FileNotFoundError(f"CT not found: {self.ct_nii_path}")               
        if self.body_ml_path is None or not os.path.exists(self.body_ml_path):         
            raise FileNotFoundError(f"Body seg not found: {self.body_ml_path}")        
        if plan.get("run_total", False):                                               
            if self.total_ml_path is None or not os.path.exists(self.total_ml_path):   
                raise FileNotFoundError(f"Total seg not found: {self.total_ml_path}")  
        if plan.get("run_head_glands_cavities", False):                                
            if self.head_glands_cavities_ml_path is None or not os.path.exists(self.head_glands_cavities_ml_path): 
                raise FileNotFoundError(f"Head seg not found: {self.head_glands_cavities_ml_path}") 

    def _create_roi_unified(
        self,
        body_seg: np.ndarray,
        total_seg: Optional[np.ndarray],
        head_seg: Optional[np.ndarray],
        plan: TotSegPlan,
    ) -> np.ndarray:
        """
        Build the unified VTT multilabel volume by painting ROIs in priority order.

        Fully data-driven: reads totseg_rois lists from pipeline_roi_naming_map.json
        via self.vtt_to_totseg — no hardcoded per-organ logic.

        Parameters
        ----------
        body_seg : np.ndarray
        total_seg : Optional[np.ndarray]
        head_seg : Optional[np.ndarray]
        plan : TotSegPlan

        Returns
        -------
        np.ndarray  (uint8, VTT label IDs)
        """
        roi_unified = np.zeros(body_seg.shape, dtype=np.uint8)
        roi_unified[body_seg > 0] = self.vtt_name2id["remaining_body"]

        if total_seg is not None and total_seg.shape != body_seg.shape:
            raise ValueError(
                f"Shape mismatch body vs total: {body_seg.shape} vs {total_seg.shape}"
            )
        if head_seg is not None and head_seg.shape != body_seg.shape:
            raise ValueError(
                f"Shape mismatch body vs head: {body_seg.shape} vs {head_seg.shape}"
            )

        assigned_voxels = roi_unified > self.vtt_name2id["remaining_body"]
        for vtt_roi in plan["vtt_roi_subset"]:
            entry = self.vtt_to_totseg.get(vtt_roi)
            if not entry:
                continue
            vtt_label = self.vtt_name2id.get(vtt_roi)
            if vtt_label is None:
                continue
            task = entry["task"]
            totseg_rois = entry["totseg_rois"]

            new_mask = np.zeros(body_seg.shape, dtype=bool)
            if task == "total" and total_seg is not None:
                ids = [self.total_name2id[r] for r in totseg_rois if r in self.total_name2id]
                if ids:
                    new_mask = np.isin(total_seg, ids)
            elif task == "head_glands_cavities" and head_seg is not None:
                ids = [self.head_name2id[r] for r in totseg_rois if r in self.head_name2id]
                if ids:
                    new_mask = np.isin(head_seg, ids)

            overlap = new_mask & assigned_voxels
            if overlap.any():
                raise AssertionError(
                    f"ROI '{vtt_roi}' overlaps with previously assigned voxels in the unified label map. "
                    "Check VTT_to_totseg for duplicate totseg_rois entries across ROIs."
                )
            roi_unified[new_mask] = vtt_label
            assigned_voxels |= new_mask

        return roi_unified

    def _run_unification(self, plan: TotSegPlan) -> None:                              
        """
        Run ROI unification and write the unified segmentation NIfTI.

        Combines TotalSegmentator outputs into a single multilabel VTT volume.
        Skips if the final output already exists.
        """
        # Skip if unified output already exists 
        if os.path.exists(self.final_output_path) and os.path.exists(self.stage_output_path): 
            if self.debug: 
                print("[SegmentationStage] Unified output already exists, skipping unification.")
            return 

        self._assert_unification_inputs_exist(plan)                                    

        ct_nii = nib.load(self.ct_nii_path)                                            

        body_seg = load_int_seg(self.body_ml_path)                               
        total_seg = load_int_seg(self.total_ml_path) if plan.get("run_total", False) else None 
        head_seg = (                                                                   
            load_int_seg(self.head_glands_cavities_ml_path) if plan.get("run_head_glands_cavities", False) else None 
        )                                                                              

        roi_unified = self._create_roi_unified(body_seg, total_seg, head_seg, plan)    

        out_img = nib.Nifti1Image(roi_unified.astype(np.uint8), ct_nii.affine, ct_nii.header) 
        out_img.set_data_dtype(np.uint8)                                               
        nib.save(out_img, self.stage_output_path)                                      
        nib.save(out_img, self.final_output_path)                                      

        # Save unification metadata 
        unification_metadata: Dict[str, Any] = {                                       
            "stage": "unification (merged into segmentation_stage)",                   
            "ct_nii_path": self.ct_nii_path,                                           
            "body_ml_path": self.body_ml_path,                                         
            "total_ml_path": self.total_ml_path if plan.get("run_total", False) else None, 
            "head_ml_path": self.head_glands_cavities_ml_path if plan.get("run_head_glands_cavities", False) else None, 
            "label_map_path": self.ts_map_path,                                        
            "stage_output_path": self.stage_output_path,                               
            "final_output_path": self.final_output_path,                               
            "plan": dict(plan),                                                        
        }                                                                              
        with open(self.unification_metadata_path, "w", encoding="utf-8") as f:
            json.dump(unification_metadata, f, indent=4)

    def _save_stage_metadata(self, plan: TotSegPlan) -> None:
        """Save stage-specific metadata for debugging / provenance."""
        outputs: Dict[str, Any] = {
            "body_ml_path": self.body_ml_path,
            "total_ml_path": self.total_ml_path if plan["run_total"] else None,
            "head_glands_cavities_ml_path": (
                self.head_glands_cavities_ml_path if plan["run_head_glands_cavities"] else None
            ),
            "stage_output_path": self.stage_output_path,
            "final_output_path": self.final_output_path,
        }
        extra: Dict[str, Any] = {
            "stage": "segmentation_stage",
            "ct_input_path": self.ct_input_path,
            "ct_nii_path": self.ct_nii_path,
            "output_dir": self.output_dir,
            "work_dir": self.work_dir,
            "file_prefix": self.prefix,
            "ml": self.ml,
            "plan": plan,
            "body_ml_path": self.body_ml_path,
            "total_ml_path": self.total_ml_path if plan["run_total"] else None,
            "head_glands_cavities_ml_path": (
                self.head_glands_cavities_ml_path if plan["run_head_glands_cavities"] else None
            ),
            "label_map_path": self.ts_map_path,                                        
            "final_output_path": self.final_output_path,                               
            "stage_output_path": self.stage_output_path,                               
        }
        metadata = build_stage_metadata(
            stage_name="segmentation_stage",
            config_snapshot=self._rerun_config_snapshot(plan),
            ct_identity=self.ct_input_identity,
            upstream_fingerprints=self._current_upstream_fingerprints(),
            outputs=outputs,
            extra=extra,
        )
        write_json(self.metadata_path, metadata)

    # ------------------------------------------------------------------
    # main
    # ------------------------------------------------------------------
    def run(self) -> Any:
        """
        Run the TotalSegmentator stage + ROI unification.

        Returns
        -------
        context : Context-like
            The updated context object (same instance as `self.context`).
        """
        self._standardize_ct_to_nifti()
        if self.ct_nii_path is None:
            raise RuntimeError("CT standardisation failed: ct_nii_path is None after _standardize_ct_to_nifti()")

        plan = self._pre_totalsegmentation_checks()
        body_ml_done, head_glands_cavities_ml_done, total_ml_done = self._files_exist()

        assert_stage_rerun_safe(
            stage_name="segmentation_stage",
            metadata_path=self.metadata_path,
            required_outputs=[
                self.body_ml_path,
                self.total_ml_path if plan["run_total"] else None,
                self.head_glands_cavities_ml_path if plan["run_head_glands_cavities"] else None,
                self.stage_output_path,
                self.final_output_path,
            ],
            current_config_snapshot=self._rerun_config_snapshot(plan),
            current_ct_identity=self.ct_input_identity,
            current_upstream_fingerprints=self._current_upstream_fingerprints(),
            context=self.context,
        )
        if self.context.stage_skipped:
            self.context.ct_nii_path = self.ct_nii_path
            self.context.body_ml_path = self.body_ml_path
            self.context.total_ml_path = self.total_ml_path if plan["run_total"] else None
            self.context.head_glands_cavities_ml_path = (
                self.head_glands_cavities_ml_path if plan["run_head_glands_cavities"] else None
            )
            self.context.totseg_plan = plan
            self.context.vtt_roi_seg_path = self.final_output_path
            return self.context

        device = "gpu" if torch.cuda.is_available() else "cpu"

        if plan["run_body"] and not body_ml_done:
            print("Running TotalSegmentator for task: BODY...")
            totalsegmentator(self.ct_nii_path, self.body_ml_path, ml=self.ml, task="body", device=device)

        if plan["run_total"] and not total_ml_done:
            print("Running TotalSegmentator for task: TOTAL...")
            totalsegmentator(
                self.ct_nii_path,
                self.total_ml_path,
                ml=self.ml,
                task="total",
                roi_subset=plan["total_roi_subset"],
                device=device,
            )

        if plan["run_head_glands_cavities"] and not head_glands_cavities_ml_done:
            print("Running TotalSegmentator for task: HEAD_GLANDS_CAVITIES...")
            totalsegmentator(
                self.ct_nii_path,
                self.head_glands_cavities_ml_path,
                ml=self.ml,
                task="head_glands_cavities",
                device=device,
            )

        # Final existence checks (only for what was requested).
        body_done, head_done, total_done = self._files_exist()
        if plan["run_body"] and not body_done:
            raise FileNotFoundError(f"Body seg not found: {self.body_ml_path}")
        if plan["run_total"] and not total_done:
            raise FileNotFoundError(f"Total seg not found: {self.total_ml_path}")
        if plan["run_head_glands_cavities"] and not head_done:
            raise FileNotFoundError(f"Head glands seg not found: {self.head_glands_cavities_ml_path}")

        self._run_unification(plan)

        self._save_stage_metadata(plan)

        self.context.ct_nii_path = self.ct_nii_path
        self.context.body_ml_path = self.body_ml_path
        self.context.total_ml_path = self.total_ml_path if plan["run_total"] else None
        self.context.head_glands_cavities_ml_path = (
            self.head_glands_cavities_ml_path if plan["run_head_glands_cavities"] else None
        )
        self.context.totseg_plan = plan
        self.context.vtt_roi_seg_path = self.final_output_path                         
        self.context.extras["segmentation_stage"] = {
            "output_dir": self.output_dir,
            "work_dir": self.work_dir,
            "metadata_path": self.metadata_path,
            "unification_metadata_path": self.unification_metadata_path,               
        }

        return self.context
