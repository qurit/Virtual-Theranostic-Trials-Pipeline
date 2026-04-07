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
from src.tests.validate_config import validate_config

from src.stages.segmentation_stage import SegmentationStage
from src.stages.synthetic_lesions_stage import SyntheticLesionsStage
from src.stages.pbpk_tac_stage import PbpkTacStage
from src.stages.simind_simulation_stage import SimindSimulationStage
from src.stages.spect_postprocess_stage import SpectPostprocessStage
#from src.stages.opengate_simulation_stage import OpenGateSimulationStage
from src.stages.dosemap_postprocess_stage import DosemapPostprocessStage


CTInputType = Literal["nii", "dicom"]

# ANSI color codes for DEBUG terminal output 
_GREEN = "\033[92m"  
_CYAN = "\033[96m"   
_YELLOW = "\033[93m" 
_RESET = "\033[0m"   
_BOLD = "\033[1m"    


def _debug_print(msg: str, phase: str = "", logger: logging.Logger = None) -> None: 
    """Print a colored debug message to terminal and optionally log it.""" 
    colored = f"{_GREEN}[DEBUG]{_RESET} {_CYAN}[{phase}]{_RESET} {msg}" if phase else f"{_GREEN}[DEBUG]{_RESET} {msg}" 
    print(colored) 
    if logger: 
        logger.debug(f"[{phase}] {msg}" if phase else msg) 


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
        The log records launched_via, config path, output folder, and timing.
    save_config : bool, default=False
        If True, saves a copy of the config JSON into the output folder.
    mode : {"DEBUG", "PRODUCTION"}, default="PRODUCTION"
        Controls verbosity and whether intermediate files are cleaned up.
    synthetic_lesions : bool, default=False
        If True, runs the synthetic lesions stage to add simulated lesions to the CT.
    run_spect : bool, default=False
        If True, runs SIMIND SPECT simulation in Phase 2.
    run_dosimetry : bool, default=False
        If True, runs OpenGATE dosimetry simulation in Phase 2.
    run_postprocess : bool, default=False
        If True, runs post-processing in Phase 3 for whichever simulations ran.
    """

    def __init__(
        self,
        config_path: str,
        ct_input: str,
        ct_indx: int,
        logging_on: bool = True,
        save_config: bool = False,
        mode: Literal["DEBUG", "PRODUCTION"] = "PRODUCTION",
        synthetic_lesions: bool = False,
        run_spect: bool = False,
        run_dosimetry: bool = False,
        run_postprocess: bool = False,
        launched_via: str = "cli",
    ) -> None:
        self.config_path: str = config_path
        self.ct_input: str = ct_input
        self.ct_indx: int = ct_indx
        self.current_dir_path: str = os.path.abspath(os.path.dirname(__file__))

        self.logging_on: bool = logging_on
        self.save_ct_scan: bool = True       # always enabled; CT copy is always written
        self.save_config: bool = save_config
        self.launched_via: str = launched_via
        self.mode: Literal["DEBUG", "PRODUCTION"] = mode
        self.synthetic_lesions: bool = synthetic_lesions
        self.run_spect: bool = run_spect       
        self.run_dosimetry: bool = run_dosimetry 
        self.run_postprocess: bool = run_postprocess 

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

    def _inject_computed_paths(self) -> None:
        """
        Inject developer-controlled paths and naming fields from
        ``src/data/pipeline_paths.json`` into the live config so that
        per-run config files do not need to carry them.

        Fields injected
        ---------------
        - ``input_paths.label_map_path``   → phase_1.segmentation_stage.label_map_path
        - ``input_paths.SIMINDDirectory``  → phase_2.simind_stage.SIMINDDirectory
        - ``phase_*/sub_dir_name``         → each phase's sub_dir_name
        - ``phase_*/*/file_prefix``        → each stage's file_prefix
        - ``phase_1/segmentation_stage/unification_prefix``
        """
        repo_root = self.current_dir_path
        paths_file = os.path.join(repo_root, "src", "data", "pipeline_paths.json")

        pipeline_paths: Dict[str, Any] = {}
        try:
            with open(paths_file, encoding="utf-8") as fh:
                pipeline_paths = json.loads(json_minify(fh.read()))
        except Exception:
            pass

        input_paths = pipeline_paths.get("input_paths", {})

        # ── label_map_path ────────────────────────────────────────────────────
        seg_stage = self.config["phase_1"]["segmentation_stage"]
        lmp = input_paths.get("label_map_path", "").strip()
        if not lmp:
            lmp = os.path.join(repo_root, "src", "data", "tdt_map.json")
        seg_stage["label_map_path"] = lmp

        # ── SIMINDDirectory ───────────────────────────────────────────────────
        simind_dir = input_paths.get("SIMINDDirectory", "").strip()
        simind_stage = self.config["phase_2"]["simind_stage"]
        if simind_dir:
            simind_stage["SIMINDDirectory"] = simind_dir

        # ── Sub-directory names and file prefixes from pipeline_paths.json ────
        # These are developer-controlled naming conventions that should not
        # appear in user-facing config files.  Inject them here so all stages
        # find them in context.config regardless of what the user's config contains.
        _STAGE_FIELDS = ("file_prefix", "unification_prefix", "sub_dir_name")
        for phase_key in ("phase_1", "phase_2", "phase_3"):
            pp_phase = pipeline_paths.get(phase_key, {})
            if not isinstance(pp_phase, dict):
                continue
            cfg_phase = self.config.setdefault(phase_key, {})
            # Top-level sub_dir_name for the phase
            if "sub_dir_name" in pp_phase:
                cfg_phase["sub_dir_name"] = pp_phase["sub_dir_name"]
            # Per-stage fields
            for stage_key, stage_data in pp_phase.items():
                if stage_key == "sub_dir_name" or not isinstance(stage_data, dict):
                    continue
                cfg_stage = cfg_phase.setdefault(stage_key, {})
                for field in _STAGE_FIELDS:
                    if field in stage_data:
                        cfg_stage[field] = stage_data[field]

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
        logger.info("Launched via: %s", self.launched_via)
        logger.info("Config file:  %s", self.config_path)
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

        # Inject paths that are no longer stored in the config file.
        # label_map_path: always the repo-relative tdt_map.json.
        # SIMINDDirectory: read from pipeline_paths.json (set once by the user there).
        self._inject_computed_paths()

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
        phases = ["phase_1", "phase_2", "phase_3"]                                     
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
        context.run_spect = self.run_spect           
        context.run_dosimetry = self.run_dosimetry   
        context.run_postprocess = self.run_postprocess 

        # downstream_roi_subset flows from phase-1 config into all downstream stages.
        # 'body' is always segmented and auto-added here so downstream stages never
        # require the user to list it explicitly.
        roi_subset = self.config["phase_1"]["segmentation_stage"]["roi_subset"]
        if isinstance(roi_subset, str):
            roi_subset = [roi_subset]
        normalized_roi_subset = [str(r).strip() for r in roi_subset if str(r).strip()]
        if "body" not in normalized_roi_subset:
            normalized_roi_subset.append("body")
        context.downstream_roi_subset = normalized_roi_subset

        self.logger.debug("Context initialized for CT_%s", self.ct_indx)

    def _phase_banner(self, phase_num: int, phase_name: str) -> None: 
        """Print a phase banner with optional color in DEBUG mode.""" 
        banner = f"-----------------------------Phase {phase_num}: {phase_name}-----------------------------" 
        if self.mode == "DEBUG": 
            print(f"{_BOLD}{_YELLOW}{banner}{_RESET}") 
        else: 
            print(banner) 

    def _cleanup_work_dir(self, work_dir: str) -> None: 
        """Remove a work directory in PRODUCTION mode to save disk space.""" 
        if self.mode == "PRODUCTION" and work_dir and os.path.exists(work_dir): 
            work_dir_abs = os.path.abspath(work_dir) 
            shutil.rmtree(work_dir_abs, ignore_errors=True) 
            if self.mode == "DEBUG": 
                _debug_print(f"Cleaned up work_dir: {work_dir_abs}", "CLEANUP", self.logger) 

    def run(self) -> Context:
        """
        Execute all pipeline stages sequentially for this CT input.

        Phases
        ------
        1. Digital Twin & Ground Truth:
            1.1  TotalSegmentator segmentation + ROI unification to TDT label space
            1.2  Synthetic lesion generation (optional)
            1.3  PBPK TAC generation
        2. Simulations:
            2.1  SIMIND preprocessing + Monte Carlo projection simulation (optional, --spect)
            2.2  OpenGATE Monte Carlo dosimetry simulation (optional, --dosimetry)
        3. Post-Processing:
            3.1  SPECT post-processing (optional, --postprocess with --spect)
            3.2  Dosimetry post-processing (optional, --postprocess with --dosimetry)

        Returns
        -------
        Context
            The updated context after all stages complete.
        """
        logger = self.logger
        t_pipeline = time.perf_counter()

        print(f"Starting TDT Pipeline (CT_{self.ct_indx}) | mode={self.mode} | input={self.ct_input}")
        if self.mode == "DEBUG": 
            _debug_print(f"run_spect={self.run_spect} | run_dosimetry={self.run_dosimetry} | run_postprocess={self.run_postprocess}", "INIT", logger) 
            _debug_print(f"synthetic_lesions={self.run_synthetic_lesions}", "INIT", logger) 

        context = self.context

        logger.info("Pipeline start | mode=%s", self.mode)
        logger.info("CT input | path=%s | type=%s", self.ct_input, self.ct_input_type)

        # ----------------------------- Phase 1: Digital Twin & Ground Truth ----------------------------- 
        self._phase_banner(1, "Digital Twin & Ground Truth") 

        logger.info("Stage start: Segmentation + ROI Unification") 
        t_stage = time.perf_counter()
        print("Running Segmentation + ROI Unification Stage...") 
        if self.mode == "DEBUG": 
            _debug_print("Segmentation will run TotalSegmentator tasks then unify to TDT label space", "Phase 1", logger) 
        context = SegmentationStage(context).run() 
        print("Segmentation + ROI Unification Stage completed.") 
        logger.info("Stage end: Segmentation + ROI Unification | elapsed=%.2fs", time.perf_counter() - t_stage) 

        if self.run_synthetic_lesions:
            logger.info("Stage start: Synthetic Lesions Generation")
            t_stage = time.perf_counter()
            print("Running Synthetic Lesions Generation Stage...")
            if self.mode == "DEBUG": 
                _debug_print("Inserting synthetic lesions into unified segmentation", "Phase 1", logger) 
            context = SyntheticLesionsStage(context).run()
            print("Synthetic Lesions Generation Stage completed.")
            logger.info("Stage end: Synthetic Lesions Generation | elapsed=%.2fs", time.perf_counter() - t_stage)

        logger.info("Stage start: PBPK TAC Generation") 
        t_stage = time.perf_counter() 
        print("Running PBPK TAC Generation Stage...") 
        if self.mode == "DEBUG": 
            _debug_print("Generating TACs for all segmented ROIs via PyCNO PSMA model", "Phase 1", logger) 
        context = PbpkTacStage(context).run() 
        print("PBPK TAC Generation Stage completed.") 
        logger.info("Stage end: PBPK TAC Generation | elapsed=%.2fs", time.perf_counter() - t_stage) 

        # ----------------------------- Phase 2: Simulations ----------------------------- 
        if self.run_spect or self.run_dosimetry: 
            self._phase_banner(2, "Simulations") 
        else: 
            print("-----------------------------Phase 2: Simulations (skipped - no --spect or --dosimetry)-----------------------------") 
            logger.info("Phase 2 skipped: no --spect or --dosimetry flags") 

        if self.run_spect: 
            logger.info("Stage start: SIMIND Simulation") 
            t_stage = time.perf_counter()
            print("Running SIMIND Simulation Stage (includes preprocessing)...") 
            if self.mode == "DEBUG": 
                _debug_print("SIMIND: preprocessing CT/seg -> binaries, then running Monte Carlo projections", "Phase 2", logger) 
            context = SimindSimulationStage(context).run() 
            print("SIMIND Simulation Stage completed.") 
            logger.info("Stage end: SIMIND Simulation | elapsed=%.2fs", time.perf_counter() - t_stage) 

        if self.run_dosimetry: 
            logger.info("Stage start: OpenGATE Simulation") 
            t_stage = time.perf_counter()
            print("Running OpenGATE Simulation Stage...") 
            if self.mode == "DEBUG": 
                _debug_print("OpenGATE: running voxel-source Monte Carlo dosimetry per ROI", "Phase 2", logger) 
            context = OpenGateSimulationStage(context).run()
            print("OpenGATE Simulation Stage completed.") 
            logger.info("Stage end: OpenGATE Simulation | elapsed=%.2fs", time.perf_counter() - t_stage)

        # ----------------------------- Phase 3: Post-Processing ----------------------------- 
        if self.run_postprocess and (self.run_spect or self.run_dosimetry): 
            self._phase_banner(3, "Post-Processing") 
        elif self.run_postprocess: 
            print("-----------------------------Phase 3: Post-Processing (skipped - no simulations ran)-----------------------------") 
            logger.info("Phase 3 skipped: --postprocess set but no simulations ran") 
        else: 
            print("-----------------------------Phase 3: Post-Processing (skipped - no --postprocess)-----------------------------") 
            logger.info("Phase 3 skipped: no --postprocess flag") 

        if self.run_postprocess and self.run_spect: 
            logger.info("Stage start: SPECT Post-Processing") 
            t_stage = time.perf_counter() 
            print("Running SPECT Post-Processing Stage...") 
            if self.mode == "DEBUG": 
                _debug_print("Applying TAC weighting, Poisson noise, and OSEM+TEW reconstruction", "Phase 3", logger) 
            context = SpectPostprocessStage(context).run() 
            print("SPECT Post-Processing Stage completed.") 
            logger.info("Stage end: SPECT Post-Processing | elapsed=%.2fs", time.perf_counter() - t_stage) 

        if self.run_postprocess and self.run_dosimetry: 
            logger.info("Stage start: Dosimetry Post-Processing") 
            t_stage = time.perf_counter() 
            print("Running Dosimetry Post-Processing Stage...") 
            if self.mode == "DEBUG": 
                _debug_print("Applying TAC-weighted cumulated activity to dose maps", "Phase 3", logger) 
            context = DosemapPostprocessStage(context).run() 
            print("Dosimetry Post-Processing Stage completed.") 
            logger.info("Stage end: Dosimetry Post-Processing | elapsed=%.2fs", time.perf_counter() - t_stage) 

        # ----------------------------- PRODUCTION cleanup ----------------------------- 
        if self.mode == "PRODUCTION": 
            # Clean up SIMIND work_dir after post-processing is done 
            simind_work = getattr(context, "simind_work_dir", None) 
            if simind_work and os.path.exists(simind_work): 
                # Only clean if postprocess is done or not requested 
                if not self.run_postprocess or not self.run_spect: 
                    self._cleanup_work_dir(simind_work) 
                elif self.run_postprocess and self.run_spect: 
                    # Post-process already ran, safe to clean 
                    self._cleanup_work_dir(simind_work) 

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
        "--ct_index_start",
        type=int,
        default=0,
        help="Starting CT index for output folder naming (e.g. 1 → _CT_1, _CT_2, …). Default: 0.",
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
    parser.add_argument( 
        "--spect", 
        action=argparse.BooleanOptionalAction, 
        default=False, 
        help="Run SIMIND SPECT projection simulation in Phase 2. Default: disabled.", 
    ) 
    parser.add_argument( 
        "--dosimetry", 
        action=argparse.BooleanOptionalAction, 
        default=False, 
        help="Run OpenGATE dosimetry simulation in Phase 2. Default: disabled.", 
    ) 
    parser.add_argument( 
        "--postprocess", 
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run post-processing in Phase 3 for whichever simulations ran. Default: disabled.",
    )
    parser.add_argument(
        "--launched_via",
        default="cli",
        choices=["cli", "web_ui"],
        help="How the pipeline was invoked — recorded in the log file. Default: cli.",
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
    print(f"Flags: --spect={args.spect} --dosimetry={args.dosimetry} --postprocess={args.postprocess}") 
    print("")

    # Pre-flight: validate config before touching any CT.
    # Inject computed paths first so validation sees the real values.
    try:
        with open(args.config_file, encoding="utf-8") as _f:
            _cfg_raw = json.loads(json_minify(_f.read()))

        # Inject paths from pipeline_paths.json (same logic as TdtPipeline._inject_computed_paths)
        _repo = os.path.abspath(os.path.dirname(__file__))
        _pp_path = os.path.join(_repo, "src", "data", "pipeline_paths.json")
        _pp: Dict[str, Any] = {}
        try:
            with open(_pp_path, encoding="utf-8") as _ppf:
                _pp = json.loads(json_minify(_ppf.read()))
        except Exception:
            pass
        _input_paths = _pp.get("input_paths", {})

        _lmp = _input_paths.get("label_map_path", "").strip()
        if not _lmp:
            _lmp = os.path.join(_repo, "src", "data", "tdt_map.json")
        _cfg_raw["phase_1"]["segmentation_stage"]["label_map_path"] = _lmp

        _sdir = _input_paths.get("SIMINDDirectory", "").strip()
        if _sdir:
            _cfg_raw["phase_2"]["simind_stage"]["SIMINDDirectory"] = _sdir

        validate_config(
            _cfg_raw,
            run_spect=args.spect,
            run_dosimetry=args.dosimetry,
            run_postprocess=args.postprocess,
            synthetic_lesions=args.synthetic_lesions,
        )
    except ValueError as _ve:
        print(f"\n[ERROR] {_ve}\n")
        return 1
    except FileNotFoundError as _fnf:
        print(f"\n[ERROR] Config file not found: {_fnf}\n")
        return 1

    any_failed = False
    for idx, name in enumerate(items, start=args.ct_index_start):
        ct_path = os.path.join(ct_inputs_dir, name)

        try:
            pipeline = TdtPipeline(
                config_path=args.config_file,
                ct_input=ct_path,
                ct_indx=idx,
                logging_on=args.logging_on,
                save_config=args.save_config,
                mode=args.mode,
                synthetic_lesions=args.synthetic_lesions,
                run_spect=args.spect,
                run_dosimetry=args.dosimetry,
                run_postprocess=args.postprocess,
                launched_via=args.launched_via,
            )
            pipeline.run()
        except Exception:
            import traceback
            print(f"[ERROR] CT index {idx} failed for input: {ct_path}")
            traceback.print_exc()
            any_failed = True

    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())


# quick run: python main.py --config_file inputs/config.json --input_ct_dir inputs/ct_testing --mode DEBUG --logging_on --save_config --synthetic_lesions --spect --dosimetry --postprocess 
