"""
Theranostic Digital Twin (TDT) Pipeline Runner.

This module provides:
- `TdtPipeline`: an orchestrator that runs all pipeline stages for a single CT input.
- A CLI entrypoint that iterates through a directory of CT inputs and runs the pipeline.

Notes
-----
- A CT input may be either a NIfTI file (.nii / .nii.gz) or a DICOM directory.
- The pipeline writes outputs into an output folder derived from the config and CT index.
- Phases run sequentially; all stage-to-stage communication goes through the `Context` object.

For any questions or issues, please contact: pyazdi@bccrc.ca
"""

from __future__ import annotations

import os
import json
from json_minify import json_minify
import logging
import time
import shutil
import argparse
import copy
from typing import Any, Dict, Literal

from src.io.context import Context

from src.stages.segmentation_ts_stage import TotalSegmentationStage
from src.stages.unify_seg_tdt_outputs import TdtRoiUnifyStage
from src.stages.preprocessing_simind_stage import SimindPreprocessStage
from src.stages.synthetic_lesions_stage import SyntheticLesionsStage
from src.stages.simind_simulation_stage import SimindSimulationStage
from src.stages.pbpk_stage import PbpkStage
from src.stages.reconstruction_stage import SpectReconstructionStage
from src.stages.opengate_simulation_stage import OpenGateSimulationStage


CTInputType = Literal["nii", "dicom"]


class TdtPipeline:
    """
    Orchestrates the full TDT pipeline for a single CT input.

    Parameters
    ----------
    config_path : str
        Path to the JSON config file (comments allowed via `json_minify`).
    ct_input : str
        Path to a CT input (either a .nii/.nii.gz file OR a DICOM directory).
    ct_indx : int
        Index used for naming (e.g., output folder suffix "_CT_{ct_indx}").
    logging_on : bool, default=True
        If True, writes a per-CT log file into the CT output folder.
    save_ct_scan : bool, default=False  
        If True, copies the CT input into the CT output folder for provenance/debugging.
    save_config : bool, default=False
        If True, saves a copy of the config JSON into the output folder.
    mode : {"DEBUG", "PRODUCTION"}, default="PRODUCTION"
        Controls verbosity and whether intermediate files are cleaned up.
    synthetic_lesions : bool, default=False
        If True, runs the synthetic lesions stage to add simulated lesions to the CT.
    """

    def __init__(
        self,
        config_path: str,
        ct_input: str,
        ct_indx: int,
        logging_on: bool = True,
        save_ct_scan: bool = False, 
        save_config: bool = False,
        mode: Literal["DEBUG", "PRODUCTION"] = "PRODUCTION",
        synthetic_lesions: bool = False,
    ) -> None:
        self.config_path: str = config_path
        self.ct_input: str = ct_input
        self.ct_indx: int = ct_indx
        self.current_dir_path: str = os.path.abspath(os.path.dirname(__file__))

        self.logging_on: bool = logging_on
        self.save_ct_scan: bool = save_ct_scan  
        self.save_config: bool = save_config
        self.mode: Literal["DEBUG", "PRODUCTION"] = mode
        self.synthetic_lesions: bool = synthetic_lesions

        self.config: Dict[str, Any] = {}
        self.output_folder_path: str = ""
        self.ct_input_type: CTInputType = "dicom"
        self.run_synthetic_lesions: bool = False
        self.synthetic_lesions_disabled_reason: str | None = None
        self.sub_dir_names: Dict[str, str] = {}

        self.logger: logging.Logger = logging.getLogger(f"TDT_CONFIG_LOGGER_CT_{self.ct_indx}")
        self.logger.setLevel(logging.DEBUG if self.mode == "DEBUG" else logging.INFO)
        self.logger.propagate = False

        self._config_setup(config_path)

        if self.logging_on:
            self._log_setup()
        else:
            self.logger.disabled = True

        self.context: Context
        self._context_setup()

        if self.synthetic_lesions_disabled_reason is not None:
            self.logger.info(self.synthetic_lesions_disabled_reason)

        if not self.logging_on:
            self.context._log_enabled = False

    def _save_config_copy(self, config_path: str) -> None:
        """Copy the config JSON into the output folder for provenance."""
        dst = os.path.join(self.output_folder_path, "config.json")
        shutil.copy2(config_path, dst)

    def _save_ct_scan_copy(self) -> None: 
        """Copy the CT input into the output folder for provenance/debugging."""
        dst_name = os.path.basename(self.ct_input)
        dst = os.path.join(self.output_folder_path, dst_name)
        if os.path.isdir(self.ct_input):
            if not os.path.exists(dst):
                shutil.copytree(self.ct_input, dst)
        else:
            shutil.copy2(self.ct_input, dst)

    def _log_setup(self) -> logging.Logger:
        """
        Configure a per-CT log file handler.

        Writes to: <output_folder_path>/logging_file_CT_<ct_indx>.log
        Avoids adding duplicate handlers if the pipeline is constructed multiple times.
        """
        log_path = os.path.join(self.output_folder_path, f"logging_file_CT_{self.ct_indx}.log")
        logger = self.logger

        if not any(
            isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == log_path
            for h in logger.handlers
        ):
            fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
            fh.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            logger.addHandler(fh)

        logger.info("----Log started----")
        logger.info("Output folder: %s", self.output_folder_path)
        return logger

    def _config_setup(self, config_path: str) -> None:
        """
        Load config from disk and prepare output folder + phase subdirectory paths.

        Parameters
        ----------
        config_path : str
            Path to JSON config (may include // comments, stripped via json_minify).

        Raises
        ------
        FileNotFoundError
            If the config file or CT input path does not exist.
        ValueError
            If the CT input file is not a supported NIfTI extension.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, encoding="utf-8") as f:
            self.config = json.loads(json_minify(f.read()))

        output_folder_title = f"{self.config['output_folder_title']}_CT_{self.ct_indx}"
        self.output_folder_path = os.path.join(self.current_dir_path, output_folder_title)
        os.makedirs(self.output_folder_path, exist_ok=True)

        if os.path.isfile(self.ct_input):
            ct_lower = self.ct_input.lower()
            if not (ct_lower.endswith(".nii") or ct_lower.endswith(".nii.gz")):
                raise ValueError(f"CT input file must be .nii or .nii.gz, got: {self.ct_input}")
            self.ct_input_type = "nii"
        elif os.path.isdir(self.ct_input):
            self.ct_input_type = "dicom"
        else:
            raise FileNotFoundError(f"CT input not found: {self.ct_input}")

        if self.save_config:
            self._save_config_copy(config_path)

        if self.save_ct_scan: 
            self._save_ct_scan_copy()

        # Enable synthetic lesions only if the flag is set AND lesion specs are defined in config.
        lesion_specs = self.config["phase_1"]["synthetic_lesions_stage"].get("specs")
        self.run_synthetic_lesions = bool(self.synthetic_lesions and lesion_specs)
        if self.synthetic_lesions and not lesion_specs:
            self.synthetic_lesions_disabled_reason = (
                "Synthetic lesions stage is disabled because "
                "'phase_1.synthetic_lesions_stage.specs' is None or empty."
            )

        # Create phase subdirs
        phases = ["phase_1", "phase_2", "phase_3", "phase_4"]
        self.sub_dir_paths: Dict[str, str] = {}
        self.sub_dir_names: Dict[str, str] = {}
        for phase in phases:
            sub_dir_path = os.path.join(self.output_folder_path, self.config[phase]["sub_dir_name"])
            os.makedirs(sub_dir_path, exist_ok=True)
            self.sub_dir_paths[phase] = sub_dir_path
            self.sub_dir_names[phase] = self.config[phase]["sub_dir_name"]

    def _context_setup(self) -> None:
        """
        Create and populate the shared Context object.

        Sets runtime metadata, config snapshot, subdir paths, and the
        downstream_roi_subset that flows through all phases.
        """
        context = Context(logger=self.logger)
        self.context = context

        context.config = copy.deepcopy(self.config)
        context.subdir_paths = copy.deepcopy(self.sub_dir_paths)
        context.subdir_names = copy.deepcopy(self.sub_dir_names)

        context.mode = self.mode
        context.ct_input_path = self.ct_input
        context.ct_input_type = self.ct_input_type
        context.ct_indx = self.ct_indx
        context.output_folder_path = self.output_folder_path
        context.synthetic_lesions_enabled = self.run_synthetic_lesions

        # downstream_roi_subset flows from phase-1 config into all downstream stages.
        roi_subset = self.config["phase_1"]["segmentation_stage"]["roi_subset"]
        if isinstance(roi_subset, str):
            roi_subset = [roi_subset]
        context.downstream_roi_subset = [str(r).strip() for r in roi_subset if str(r).strip()]

        self.logger.debug("Context initialized for CT_%s", self.ct_indx)

    def run(self) -> Context:
        """
        Execute all pipeline stages sequentially for this CT input.

        Phases
        ------
        1. Digital Twin:
            1.1  TotalSegmentator segmentation
            1.2  ROI unification to TDT label space
            1.3  Synthetic lesion generation (optional)
        2. SPECT Simulation:
            2.1  SIMIND preprocessing (CT + seg -> SIMIND binaries)
            2.2  SIMIND Monte Carlo projection simulation
        3. SPECT Post-Processing:
            3.1  PBPK TAC generation and projection weighting
            3.2  OSEM + TEW reconstruction
        4. Dosimetry:
            4.1  OpenGATE voxel-source Monte Carlo dose calculation

        Returns
        -------
        Context
            The updated context after all stages complete.
        """
        logger = self.logger
        t_pipeline = time.perf_counter()

        print(f"Starting TDT Pipeline (CT_{self.ct_indx}) | mode={self.mode} | input={self.ct_input}")

        context = self.context

        logger.info("Pipeline start | mode=%s", self.mode)
        logger.info("CT input | path=%s | type=%s", self.ct_input, self.ct_input_type)

        # ----------------------------- Phase 1: Digital Twin -----------------------------
        print("-----------------------------Phase 1: Creating Digital Twin-----------------------------")

        logger.info("Stage start: TotalSegmentator")
        t_stage = time.perf_counter()
        print("Running TotalSegmentator Stage...")
        context = TotalSegmentationStage(context).run()
        print("TotalSegmentator Stage completed.")
        logger.info("Stage end: TotalSegmentator | elapsed=%.2fs", time.perf_counter() - t_stage)

        logger.info("Stage start: TDT ROI Unification")
        t_stage = time.perf_counter()
        print("Running TDT ROI Unification Stage...")
        context = TdtRoiUnifyStage(context).run()
        print("TDT ROI Unification Stage completed.")
        logger.info("Stage end: TDT ROI Unification | elapsed=%.2fs", time.perf_counter() - t_stage)

        if self.run_synthetic_lesions:
            logger.info("Stage start: Synthetic Lesions Generation")
            t_stage = time.perf_counter()
            print("Running Synthetic Lesions Generation Stage...")
            context = SyntheticLesionsStage(context).run()
            print("Synthetic Lesions Generation Stage completed.")
            logger.info("Stage end: Synthetic Lesions Generation | elapsed=%.2fs", time.perf_counter() - t_stage)

        # ----------------------------- Phase 2: SPECT Simulation -----------------------------
        print("-----------------------------Phase 2: SPECT Simulation-----------------------------")

        logger.info("Stage start: SIMIND Preprocessing")
        t_stage = time.perf_counter()
        print("Running SIMIND Preprocessing Stage...")
        context = SimindPreprocessStage(context).run()
        print("SIMIND Preprocessing Stage completed.")
        logger.info("Stage end: SIMIND Preprocessing | elapsed=%.2fs", time.perf_counter() - t_stage)

        logger.info("Stage start: SIMIND Simulation")
        t_stage = time.perf_counter()
        print("Running SIMIND Simulation Stage...")
        context = SimindSimulationStage(context).run()
        print("SIMIND Simulation Stage completed.")
        logger.info("Stage end: SIMIND Simulation | elapsed=%.2fs", time.perf_counter() - t_stage)

        # ----------------------------- Phase 3: SPECT Post-Processing -----------------------------
        print("-----------------------------Phase 3: SPECT Post-Processing-----------------------------")

        logger.info("Stage start: PBPK")
        t_stage = time.perf_counter()
        print("Running PBPK Stage...")
        context = PbpkStage(context).run()
        print("PBPK Stage completed.")
        logger.info("Stage end: PBPK | elapsed=%.2fs", time.perf_counter() - t_stage)

        logger.info("Stage start: SPECT Reconstruction")
        t_stage = time.perf_counter()
        print("Running SPECT Reconstruction Stage...")
        context = SpectReconstructionStage(context).run()
        print("SPECT Reconstruction Stage completed.")
        logger.info("Stage end: SPECT Reconstruction | elapsed=%.2fs", time.perf_counter() - t_stage)

        # ----------------------------- Phase 4: Dosimetry -----------------------------
        print("-----------------------------Phase 4: Dosimetry-----------------------------")

        logger.info("Stage start: OpenGATE Simulation")
        t_stage = time.perf_counter()
        print("Running OpenGATE Simulation Stage...")
        context = OpenGateSimulationStage(context).run()
        print("OpenGATE Simulation Stage completed.")
        logger.info("Stage end: OpenGATE Simulation | elapsed=%.2fs", time.perf_counter() - t_stage)

        logger.info("Pipeline end | total_elapsed=%.2fs", time.perf_counter() - t_pipeline)
        print("TDT Pipeline completed successfully.")
        return context


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the TDT pipeline."""
    parser = argparse.ArgumentParser(
        description="Theranostic Digital Twin (TDT) Pipeline Runner"
    )

    parser.add_argument("--config_file", required=True, type=str,
                        help="Path to JSON config file.")
    parser.add_argument("--input_ct_dir", required=True, type=str,
                        help="Directory containing CT inputs (NIfTI files or DICOM folders).")

    parser.add_argument(
        "--mode",
        default="PRODUCTION",
        choices=["DEBUG", "PRODUCTION"],
        help="Pipeline mode. DEBUG keeps more intermediate files. Default: PRODUCTION.",
    )
    parser.add_argument(
        "--logging_on",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable per-CT log file writing. Default: enabled.",
    )
    parser.add_argument(
        "--save_ct_scan",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Copy the CT input into the CT output folder for provenance. Default: disabled.",
    )
    parser.add_argument(
        "--save_config",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Copy the config JSON into each CT output folder. Default: disabled.",
    )
    parser.add_argument(
        "--synthetic_lesions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run synthetic lesion generation. Requires specs in config. Default: disabled.",
    )

    return parser


def main() -> int:
    """
    CLI entrypoint. Iterates through all CT inputs in `input_ct_dir` and runs the pipeline.

    Returns
    -------
    int
        Process exit code (0 = all CTs processed; individual failures are logged but don't stop the batch).
    """
    parser = build_arg_parser()
    args = parser.parse_args()

    ct_inputs_dir = os.path.abspath(args.input_ct_dir)
    if not os.path.isdir(ct_inputs_dir):
        raise NotADirectoryError(f"input_ct_dir must be a directory: {ct_inputs_dir}")

    # Filter hidden files/dirs; sort for deterministic ordering.
    items = [n for n in sorted(os.listdir(ct_inputs_dir)) if not n.startswith(".")]
    print("----------------------------- Starting TDT Pipeline -----------------------------")
    print("")
    print(f"Discovered {len(items)} CT item(s) in: {ct_inputs_dir}")
    print("")

    for idx, name in enumerate(items):
        ct_path = os.path.join(ct_inputs_dir, name)

        try:
            pipeline = TdtPipeline(
                config_path=args.config_file,
                ct_input=ct_path,
                ct_indx=idx,
                logging_on=args.logging_on,
                save_ct_scan=args.save_ct_scan,  
                save_config=args.save_config,
                mode=args.mode,
                synthetic_lesions=args.synthetic_lesions,
            )
            pipeline.run()
        except Exception as e:
            print(f"[ERROR] CT index {idx} failed for input: {ct_path}\n{e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# quick run: python main.py --config_file inputs/config.json --input_ct_dir inputs/ct_testing --mode DEBUG --logging_on --save_config --synthetic_lesions