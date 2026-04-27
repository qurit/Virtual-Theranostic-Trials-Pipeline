"""
TotalSegmentator-based segmentation + ROI unification stage for the TDT pipeline.

This stage:
- Standardizes the CT input into a NIfTI file in the phase output directory.
- Runs TotalSegmentator for the required task(s) based on a user-facing ROI list.
- Writes multilabel output masks (NIfTI) for each task and stores paths + plan in `context`.
- Combines the TotalSegmentator outputs (body, total, head_glands_cavities)
  into a single multilabel NIfTI volume in the TDT pipeline label space.

Expected Context interface
--------------------------
Incoming `context` is expected to provide:
- context.ct_input_path : str
- context.config["phase_1"]["segmentation_stage"]["roi_subset"] : list[str]
- context.config["phase_1"]["segmentation_stage"]["file_prefix"] : str
- context.config["phase_1"]["segmentation_stage"]["unification_prefix"] : str
- context.config["phase_1"]["segmentation_stage"]["label_map_path"] : str
- context.subdir_paths["phase_1"] : str

On success, this stage sets:
- context.ct_nii_path : str
- context.body_ml_path : str
- context.total_ml_path : Optional[str]  (None if not required by requested ROIs)
- context.head_glands_cavities_ml_path : Optional[str]  (None if not required)
- context.totseg_plan : dict  (which tasks ran, which roi_subsets were passed to TotalSegmentator)
- context.tdt_roi_seg_path : str  (path to unified multilabel NIfTI handoff)

Maintainer / contact: pyazdi@bccrc.ca
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

# ---------------------------------------------------------------------------
# Compatibility: some TotalSegmentator/torch combinations expect `torch.GradScaler`.
# ---------------------------------------------------------------------------
torch.GradScaler = _GradScaler


# User-facing ROI names supported by this pipeline stage.
TDT_ALLOWED_ROIS = {
    "body",
    "kidney",
    "liver",
    "prostate",
    "spleen",
    "heart",
    "salivary_glands",
}

# Maps each TDT ROI name to the TotalSegmentator (task, roi_subset) it requires.
TDT_TO_TOTSEG = {
    "kidney":         ("total", ["kidney_left", "kidney_right"]),
    "liver":          ("total", ["liver"]),
    "prostate":       ("total", ["prostate"]),
    "spleen":         ("total", ["spleen"]),
    "heart":          ("total", ["heart"]),
    "salivary_glands":("head_glands_cavities", [
        "parotid_gland_left",
        "parotid_gland_right",
        "submandibular_gland_left",
        "submandibular_gland_right",
    ]),
    "body":           ("body", []),
}

CTInputType = Literal["nii", "dicom"]


class TotSegPlan(TypedDict):
    """Execution plan describing which TotalSegmentator tasks will run."""
    run_body: bool
    run_total: bool
    run_head_glands_cavities: bool
    total_roi_subset: List[str]   # TotalSegmentator ROI names for the total task
    head_roi_subset: List[str]    # TotalSegmentator ROI names for head_glands_cavities task
    tdt_roi_subset: List[str]     # User-facing TDT ROI names (passed in from config)


class SegmentationStage:                                                               
    """
    TDT Stage: TotalSegmentator segmentation + ROI unification.

    Always runs the `body` task (used downstream as a patient mask for all ROI operations).
    The `total` and `head_glands_cavities` tasks run only if needed by the requested ROIs.
    After segmentation, combines outputs into a single multilabel volume in TDT label space.

    Parameters
    ----------
    context : Context-like
        Pipeline context object. Must provide `ct_input_path`, `config`, and `subdir_paths`.
    """

    def __init__(self, context: Any) -> None:
        context.require("ct_input_path", "config", "subdir_paths")
        self.context = context
        self.debug: bool = getattr(context, "mode", "").upper() == "DEBUG"             

        self.ct_input_path: str = context.ct_input_path
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
        self.metadata_path: str = os.path.join(self.work_dir, f"{self.prefix}_metadata.json")

        # Phase handoff: overwritten by every run so downstream stages always have a clean path. 
        self.final_output_path: str = os.path.join(self.phase_output_dir, "digital_twin.nii.gz") 
        self.stage_output_path: str = os.path.join(self.output_dir, f"{self.unification_prefix}.nii.gz") 
        self.unification_metadata_path: str = os.path.join(self.work_dir, f"{self.unification_prefix}_metadata.json") 

        # Load label map for ROI unification 
        self.ts_map_path: str = context.config["phase_1"]["segmentation_stage"]["label_map_path"] 
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
        self.tdt_name2id: Dict[str, int] = { 
            name: int(lab) for lab, name in ts_map_json["TDT_Pipeline"].items() 
        } 

    # -----------------------------
    # helpers — segmentation
    # -----------------------------

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
            return

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
                f"roi_subset must contain at least one ROI from: {sorted(TDT_ALLOWED_ROIS)}"
            )

        invalid = [r for r in rois if r not in TDT_ALLOWED_ROIS]
        if invalid:
            raise ValueError(
                f"Invalid ROI(s): {invalid}. Allowed: {sorted(TDT_ALLOWED_ROIS)}"
            )

        total_rois: List[str] = []
        head_rois: List[str] = []
        seen_total: set = set()
        seen_head: set = set()

        for r in rois:
            task, expanded = TDT_TO_TOTSEG[r]
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
            tdt_roi_subset=rois,
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

    # -----------------------------
    # helpers — unification (merged from TdtRoiUnifyStage)
    # -----------------------------

    @staticmethod                                                                      
    def _load_int_seg(path: str) -> np.ndarray:                                        
        """Load a NIfTI segmentation and return it as int16 (sufficient for label IDs).""" 
        return nib.load(path).get_fdata().astype(np.int16)                             

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
        Build the unified TDT multilabel volume by painting ROIs in priority order.

        Parameters
        ----------
        body_seg : np.ndarray
        total_seg : Optional[np.ndarray]
        head_seg : Optional[np.ndarray]
        plan : TotSegPlan

        Returns
        -------
        np.ndarray  (uint8, TDT label IDs)
        """
        roi_unified = np.zeros(body_seg.shape, dtype=np.uint8)
        roi_unified[body_seg > 0] = self.tdt_name2id["body"]

        requested = set(plan["tdt_roi_subset"])                                        

        if total_seg is not None:
            if total_seg.shape != body_seg.shape:
                raise ValueError(
                    f"Shape mismatch body vs total: {body_seg.shape} vs {total_seg.shape}"
                )
            if "kidney" in requested:
                kL = self.total_name2id["kidney_left"]
                kR = self.total_name2id["kidney_right"]
                roi_unified[(total_seg == kL) | (total_seg == kR)] = self.tdt_name2id["kidney"]
            if "liver" in requested:
                roi_unified[total_seg == self.total_name2id["liver"]] = self.tdt_name2id["liver"]
            if "prostate" in requested:
                roi_unified[total_seg == self.total_name2id["prostate"]] = self.tdt_name2id["prostate"]
            if "spleen" in requested:
                roi_unified[total_seg == self.total_name2id["spleen"]] = self.tdt_name2id["spleen"]
            if "heart" in requested:
                roi_unified[total_seg == self.total_name2id["heart"]] = self.tdt_name2id["heart"]

        if head_seg is not None and "salivary_glands" in requested:
            if head_seg.shape != body_seg.shape:
                raise ValueError(
                    f"Shape mismatch body vs head: {body_seg.shape} vs {head_seg.shape}"
                )
            pL = self.head_name2id["parotid_gland_left"]
            pR = self.head_name2id["parotid_gland_right"]
            sL = self.head_name2id["submandibular_gland_left"]
            sR = self.head_name2id["submandibular_gland_right"]
            roi_unified[np.isin(head_seg, [pL, pR, sL, sR])] = self.tdt_name2id["salivary_glands"]

        return roi_unified

    def _run_unification(self, plan: TotSegPlan) -> None:                              
        """
        Run ROI unification and write the unified segmentation NIfTI.

        Combines TotalSegmentator outputs into a single multilabel TDT volume.
        Skips if the final output already exists.
        """
        # Skip if unified output already exists 
        if os.path.exists(self.final_output_path) and os.path.exists(self.stage_output_path): 
            if self.debug: 
                print(f"[SegmentationStage] Unified output already exists, skipping unification.") 
            return 

        self._assert_unification_inputs_exist(plan)                                    

        ct_nii = nib.load(self.ct_nii_path)                                            

        body_seg = self._load_int_seg(self.body_ml_path)                               
        total_seg = self._load_int_seg(self.total_ml_path) if plan.get("run_total", False) else None 
        head_seg = (                                                                   
            self._load_int_seg(self.head_glands_cavities_ml_path) if plan.get("run_head_glands_cavities", False) else None 
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
        metadata: Dict[str, Any] = {
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
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

    # -----------------------------
    # main
    # -----------------------------
    def run(self) -> Any:
        """
        Run the TotalSegmentator stage + ROI unification.

        Returns
        -------
        context : Context-like
            The updated context object (same instance as `self.context`).
        """
        self._standardize_ct_to_nifti()
        assert self.ct_nii_path is not None

        plan = self._pre_totalsegmentation_checks()
        body_ml_done, head_glands_cavities_ml_done, total_ml_done = self._files_exist()

        if plan["run_body"] and not body_ml_done:
            print("Running TotalSegmentator for task: BODY...")
            totalsegmentator(self.ct_nii_path, self.body_ml_path, ml=self.ml, task="body")

        if plan["run_total"] and not total_ml_done:
            print("Running TotalSegmentator for task: TOTAL...")
            totalsegmentator(
                self.ct_nii_path,
                self.total_ml_path,
                ml=self.ml,
                task="total",
                roi_subset=plan["total_roi_subset"],
            )

        if plan["run_head_glands_cavities"] and not head_glands_cavities_ml_done:
            print("Running TotalSegmentator for task: HEAD_GLANDS_CAVITIES...")
            totalsegmentator(
                self.ct_nii_path,
                self.head_glands_cavities_ml_path,
                ml=self.ml,
                task="head_glands_cavities",
            )

        # Final existence checks (only for what was requested).
        body_done, head_done, total_done = self._files_exist()
        if plan["run_body"] and not body_done:
            raise FileNotFoundError(f"Body seg not found: {self.body_ml_path}")
        if plan["run_total"] and not total_done:
            raise FileNotFoundError(f"Total seg not found: {self.total_ml_path}")
        if plan["run_head_glands_cavities"] and not head_done:
            raise FileNotFoundError(f"Head glands seg not found: {self.head_glands_cavities_ml_path}")

        # Run ROI unification (merged from TdtRoiUnifyStage) 
        self._run_unification(plan)                                                    

        self._save_stage_metadata(plan)

        self.context.ct_nii_path = self.ct_nii_path
        self.context.body_ml_path = self.body_ml_path
        self.context.total_ml_path = self.total_ml_path if plan["run_total"] else None
        self.context.head_glands_cavities_ml_path = (
            self.head_glands_cavities_ml_path if plan["run_head_glands_cavities"] else None
        )
        self.context.totseg_plan = plan
        self.context.tdt_roi_seg_path = self.final_output_path                         
        self.context.extras["segmentation_stage"] = {
            "output_dir": self.output_dir,
            "work_dir": self.work_dir,
            "metadata_path": self.metadata_path,
            "unification_metadata_path": self.unification_metadata_path,               
        }

        return self.context
