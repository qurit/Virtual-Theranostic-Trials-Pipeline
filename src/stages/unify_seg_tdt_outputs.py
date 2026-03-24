"""
TDT ROI Unification Stage (TotalSegmentator outputs -> TDT multilabel segmentation).

This stage combines the TotalSegmentator outputs:
- task="body"
- task="total"
- task="head_glands_cavities"

into a single multilabel NIfTI volume in the TDT pipeline label space.

Key behaviors
-------------
- Uses user-provided `label_map_path` to translate TotalSegmentator class IDs -> ROI names -> TDT IDs.
- Paints a single output volume aligned to the CT NIfTI (affine/header).
- Only maps ROIs that were requested in the TotalSegmentator plan (`context.totseg_plan`).
- Writes a stage-local output and the phase handoff file `digital_twin.nii.gz`.

Expected Context interface
--------------------------
Incoming `context` is expected to provide:
- context.subdir_paths["phase_1"] : str
- context.config["phase_1"]["unification_stage"]["file_prefix"] : str
- context.config["phase_1"]["unification_stage"]["label_map_path"] : str
- context.ct_nii_path : str
- context.body_ml_path : str
- context.total_ml_path : Optional[str]  (required if plan.run_total)
- context.head_glands_cavities_ml_path : Optional[str]  (required if plan.run_head_glands_cavities)
- context.totseg_plan : dict

On success, this stage sets:
- context.tdt_roi_seg_path : str  (path to unified multilabel NIfTI handoff file)

Maintainer / contact: pyazdi@bccrc.ca
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np
import nibabel as nib
from json_minify import json_minify


class TotSegPlan(TypedDict, total=False):
    """Minimal typing for the TotalSegmentator execution plan."""
    run_total: bool
    run_head_glands_cavities: bool
    tdt_roi_subset: List[str]


class TdtRoiUnifyStage:
    """
    Combine TotalSegmentator outputs into a single multilabel segmentation in TDT label space.

    Label mapping is defined externally in the label map JSON:
    - "total"                : TotalSegmentator label id -> ROI name
    - "head_glands_cavities" : TotalSegmentator label id -> ROI name
    - "TDT_Pipeline"         : TDT label id -> TDT ROI name

    The unified output uses TDT_Pipeline label IDs (uint8).
    """

    def __init__(self, context: Any) -> None:
        context.require("subdir_paths", "config", "ct_nii_path", "body_ml_path", "totseg_plan")
        self.context = context

        self.phase_output_dir: str = context.subdir_paths["phase_1"]
        self.output_dir: str = os.path.join(self.phase_output_dir, "unification_stage")
        self.work_dir: str = os.path.join(self.output_dir, "work_dir")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.work_dir, exist_ok=True)

        self.prefix: str = context.config["phase_1"]["unification_stage"]["file_prefix"]
        # Phase handoff: overwritten by every run so downstream stages always have a clean path.
        self.final_output_path: str = os.path.join(self.phase_output_dir, "digital_twin.nii.gz")
        self.stage_output_path: str = os.path.join(self.output_dir, f"{self.prefix}.nii.gz")
        self.metadata_path: str = os.path.join(self.work_dir, f"{self.prefix}_metadata.json")

        self.ts_map_path: str = context.config["phase_1"]["unification_stage"]["label_map_path"]
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

        self.ct_nii_path: Optional[str] = context.ct_nii_path
        self.body_ml_path: Optional[str] = context.body_ml_path
        self.total_ml_path: Optional[str] = context.total_ml_path
        self.head_ml_path: Optional[str] = context.head_glands_cavities_ml_path

        self.plan: TotSegPlan = context.totseg_plan
        if self.plan is None:
            raise ValueError("Missing context.totseg_plan; run TotalSegmentationStage first.")

    # -----------------------------
    # helpers
    # -----------------------------

    @staticmethod
    def _load_int_seg(path: str) -> np.ndarray:
        """Load a NIfTI segmentation and return it as int16 (sufficient for label IDs)."""
        return nib.load(path).get_fdata().astype(np.int16)

    def _assert_inputs_exist(self) -> None:
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
        if self.plan.get("run_total", False):
            if self.total_ml_path is None or not os.path.exists(self.total_ml_path):
                raise FileNotFoundError(f"Total seg not found: {self.total_ml_path}")
        if self.plan.get("run_head_glands_cavities", False):
            if self.head_ml_path is None or not os.path.exists(self.head_ml_path):
                raise FileNotFoundError(f"Head seg not found: {self.head_ml_path}")

    def _create_roi_unified(
        self,
        body_seg: np.ndarray,
        total_seg: Optional[np.ndarray],
        head_seg: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Build the unified TDT multilabel volume by painting ROIs in priority order.

        Parameters
        ----------
        body_seg : np.ndarray
        total_seg : Optional[np.ndarray]
        head_seg : Optional[np.ndarray]

        Returns
        -------
        np.ndarray  (uint8, TDT label IDs)
        """
        roi_unified = np.zeros(body_seg.shape, dtype=np.uint8)
        roi_unified[body_seg > 0] = self.tdt_name2id["body"]

        requested = set(self.plan["tdt_roi_subset"])

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

    def _save_stage_metadata(self) -> None:
        """Save stage-specific metadata for debugging / provenance."""
        metadata: Dict[str, Any] = {
            "stage": "unification_stage",
            "ct_nii_path": self.ct_nii_path,
            "body_ml_path": self.body_ml_path,
            "total_ml_path": self.total_ml_path if self.plan.get("run_total", False) else None,
            "head_ml_path": self.head_ml_path if self.plan.get("run_head_glands_cavities", False) else None,
            "label_map_path": self.ts_map_path,
            "output_dir": self.output_dir,
            "work_dir": self.work_dir,
            "stage_output_path": self.stage_output_path,
            "final_output_path": self.final_output_path,
            "plan": dict(self.plan),
        }
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

    # -----------------------------
    # main
    # -----------------------------
    def run(self) -> Any:
        """
        Run ROI unification and write the unified segmentation NIfTI.

        Returns
        -------
        context : Context-like
            Updated context object with `tdt_roi_seg_path` set.
        """
        self._assert_inputs_exist()

        ct_nii = nib.load(self.ct_nii_path)

        body_seg = self._load_int_seg(self.body_ml_path)
        total_seg = self._load_int_seg(self.total_ml_path) if self.plan.get("run_total", False) else None
        head_seg = (
            self._load_int_seg(self.head_ml_path) if self.plan.get("run_head_glands_cavities", False) else None
        )

        roi_unified = self._create_roi_unified(body_seg, total_seg, head_seg)

        out_img = nib.Nifti1Image(roi_unified.astype(np.uint8), ct_nii.affine, ct_nii.header)
        out_img.set_data_dtype(np.uint8)
        nib.save(out_img, self.stage_output_path)
        nib.save(out_img, self.final_output_path)

        self._save_stage_metadata()

        self.context.tdt_roi_seg_path = self.final_output_path
        self.context.extras["unification_stage"] = {
            "output_dir": self.output_dir,
            "work_dir": self.work_dir,
            "stage_output_path": self.stage_output_path,
            "metadata_path": self.metadata_path,
        }

        return self.context