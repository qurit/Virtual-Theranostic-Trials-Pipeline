"""
Command-line entry point for the PyTheraTwin pipeline.

This module defines the per-patient pipeline runner, injects developer-controlled
paths into the live config, and provides the batch CLI used by both local runs
and the web UI subprocess launcher.
"""

from __future__ import annotations

import os
import re
import logging
import time
import shutil
import argparse
import traceback
from typing import Any, Dict, List, Literal, Optional

from src.io.context import Context
from src.io.pipeline_logging import PipelineReporter
from src.io.profiler import StageProfiler, parse_profile_interval_arg
from src.io.rerun_guard import ensure_ct_matches_saved_copy, ensure_metadata_dir
from src.io.runtime_config import load_and_validate_runtime_config, load_runtime_config
from src.tests.validate_ct import CTInputType, discover_ct_inputs, validate_ct_input_path


class PyTheraTwinPipeline:
    """
    Orchestrates the full PyTheraTwin pipeline for a single CT input.

    Parameters
    ----------
    config_path : str
        Path to the JSON config file (comments allowed via `json_minify`).
    ct_input : str
        Path to a CT input (either a .nii/.nii.gz file OR a DICOM directory).
    ct_index : int
        Index used for naming (e.g., output folder suffix "_CT_{ct_index}").
    mode : {"DEBUG", "PRODUCTION"}, default="PRODUCTION"
        Controls verbosity and whether intermediate files are cleaned up.
    web_ui : bool, default=False
        Set to True when launched from the web UI; recorded in the log file.
    run_segmentation : bool, default=False
        If True, runs TotalSegmentator segmentation + ROI unification in Phase 1.
    run_pbpk : bool, default=False
        If True, runs PBPK TAC generation in Phase 1.
    run_synthetic_lesions : bool, default=False
        If True, runs synthetic lesion generation in Phase 1 (requires specs in config).
    run_spect : bool, default=False
        If True, runs SIMIND SPECT simulation in Phase 2.
    run_dosimetry : bool, default=False
        If True, runs OpenGATE dosimetry simulation in Phase 2.
    run_postprocess : bool, default=False
        If True, runs post-processing in Phase 3 for whichever simulations ran.
    startup_banner_lines : list of str, optional
        Pre-rendered banner lines from the batch startup banner, written into
        the per-CT log file header so the log is self-contained.
    profile_interval_s : float or None, default=None
        When set to a float (0.1–3.0 s), enables CPU/RAM profiling per stage
        and writes ``profiling_CT_<ct_index>.json`` into the output folder.
        None disables profiling.
    """

    def __init__(
        self,
        config_path: str,
        ct_input: str,
        ct_index: int,
        mode: Literal["DEBUG", "PRODUCTION"] = "PRODUCTION",
        web_ui: bool = False,
        run_segmentation: bool = False,
        run_pbpk: bool = False,
        run_synthetic_lesions: bool = False,
        run_spect: bool = False,
        run_dosimetry: bool = False,
        run_postprocess: bool = False,
        startup_banner_lines: Optional[List[str]] = None,
        profile_interval_s: Optional[float] = None,
        output_dir: Optional[str] = None,
    ) -> None:
        self.config_path: str = config_path
        self.ct_input: str = ct_input
        self.ct_index: int = ct_index
        self.current_dir_path: str = os.path.abspath(os.path.dirname(__file__))

        self.launched_via: str = "web_ui" if web_ui else "cli"
        self.mode: Literal["DEBUG", "PRODUCTION"] = mode
        self.run_segmentation: bool = run_segmentation
        self.run_pbpk: bool = run_pbpk
        self.run_synthetic_lesions: bool = run_synthetic_lesions
        self.run_spect: bool = run_spect
        self.run_dosimetry: bool = run_dosimetry
        self.run_postprocess: bool = run_postprocess
        self.startup_banner_lines: List[str] = startup_banner_lines or []
        self.profile: bool = profile_interval_s is not None
        self.profile_interval_s: float = profile_interval_s if profile_interval_s is not None else StageProfiler.DEFAULT_INTERVAL_S

        self.config: Dict[str, Any] = {}
        self.output_folder_path: str = ""
        self.metadata_dir_path: str = ""
        self.ct_saved_copy_path: str = ""
        self.ct_input_identity: Dict[str, Any] = {}
        self.sub_dir_names: Dict[str, str] = {}
        self._override_output_dir: Optional[str] = os.path.abspath(output_dir) if output_dir else None

        # Validate CT input before creating any output directories
        self.ct_input_type: CTInputType = validate_ct_input_path(self.ct_input)

        self.logger: logging.Logger = logging.getLogger(f"PyTheraTwin_PIPELINE_CT_{self.ct_index}")
        self.logger.setLevel(logging.DEBUG if self.mode == "DEBUG" else logging.INFO)
        self.logger.propagate = False
        self.reporter = PipelineReporter(mode=self.mode, logger=self.logger)

        self._config_setup(config_path)

        PipelineReporter.configure_file_logger(
            self.logger,
            log_path=os.path.join(
                self.output_folder_path,
                f"logging_file_CT_{self.ct_index}.log",
            ),
            startup_banner_lines=self.startup_banner_lines,
            launched_via=self.launched_via,
            config_path=self.config_path,
            output_folder_path=self.output_folder_path,
        )

        self.context = Context.from_pipeline_run(
            logger=self.logger,
            config=self.config,
            subdir_paths=self.sub_dir_paths,
            subdir_names=self.sub_dir_names,
            mode=self.mode,
            ct_input_path=self.ct_input,
            ct_input_type=self.ct_input_type,
            ct_input_identity=self.ct_input_identity,
            ct_saved_copy_path=self.ct_saved_copy_path,
            ct_index=self.ct_index,
            output_folder_path=self.output_folder_path,
            metadata_dir=self.metadata_dir_path,
            synthetic_lesions_enabled=self.run_synthetic_lesions,
            run_spect=self.run_spect,
            run_dosimetry=self.run_dosimetry,
            run_postprocess=self.run_postprocess,
        )
        self.logger.debug("Context initialized for CT_%s", self.ct_index)

    def _save_config_copy(self, config_path: str) -> None:
        """Copy the config JSON into the output folder for provenance."""
        dst = os.path.join(self.output_folder_path, "config.json")
        if os.path.abspath(config_path) != os.path.abspath(dst):
            shutil.copy2(config_path, dst)

    def _save_ct_scan_copy(self) -> None: 
        """Copy the CT input into the output folder for provenance/debugging."""
        dst_name = os.path.basename(self.ct_input)
        dst = os.path.join(self.output_folder_path, dst_name)
        if os.path.isdir(self.ct_input):
            if not os.path.exists(dst):
                shutil.copytree(self.ct_input, dst)
        else:
            if not os.path.exists(dst):
                shutil.copy2(self.ct_input, dst)

        self.ct_saved_copy_path = dst
        self.ct_input_identity = ensure_ct_matches_saved_copy(
            current_input_path=self.ct_input,
            saved_copy_path=dst,
            output_folder_path=self.output_folder_path,
            input_type=self.ct_input_type,
        )

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
            If the config file does not exist.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        self.config = load_runtime_config(
            config_path,
            repo_root=self.current_dir_path,
            include_input_paths=True,
        )

        if self._override_output_dir:
            self.output_folder_path = self._override_output_dir
        else:
            project_dir = os.path.join(self.current_dir_path, self.config['output_folder_title'])
            os.makedirs(project_dir, exist_ok=True)
            self.output_folder_path = os.path.join(project_dir, f"CT_{self.ct_index}")
        os.makedirs(self.output_folder_path, exist_ok=True)
        self.metadata_dir_path = str(ensure_metadata_dir(self.output_folder_path))

        self._save_config_copy(config_path)
        self._save_ct_scan_copy()

        # Create phase subdirs
        phases = ["phase_1", "phase_2", "phase_3"]                                     
        self.sub_dir_paths: Dict[str, str] = {}
        self.sub_dir_names: Dict[str, str] = {}
        for phase in phases:
            sub_dir_path = os.path.join(self.output_folder_path, self.config[phase]["sub_dir_name"])
            os.makedirs(sub_dir_path, exist_ok=True)
            self.sub_dir_paths[phase] = sub_dir_path
            self.sub_dir_names[phase] = self.config[phase]["sub_dir_name"]

    def _cleanup_work_dir(self, work_dir: str) -> None:
        """Remove a work directory in PRODUCTION mode to save disk space."""
        if self.mode == "PRODUCTION" and work_dir and os.path.exists(work_dir):
            work_dir_abs = os.path.abspath(work_dir)
            shutil.rmtree(work_dir_abs, ignore_errors=True)
        elif self.mode == "DEBUG":
            self.reporter.debug(
                f"Skipping cleanup of work_dir (DEBUG mode): {work_dir}",
                "CLEANUP",
            )

    def _run_stage(
        self,
        context: Context,
        *,
        stage_label: str,
        stage_key: str,
        stage_cls: type,
        profiler: Optional[StageProfiler],
    ) -> Context:
        """Run one pipeline stage with consistent terminal/log/profiler handling."""
        t_stage = time.perf_counter()
        print(f"Running {stage_label} Stage...")
        self.logger.info("Stage start: %s", stage_label)

        context.stage_skipped = False
        if profiler:
            profiler.start_stage(stage_key)
        context = stage_cls(context).run()
        if profiler:
            if context.stage_skipped:
                profiler.cancel_stage()
                profile_stage = None
            else:
                profile_stage = profiler.end_stage()
        else:
            profile_stage = None

        if context.stage_skipped:
            print(f"{stage_label} Stage skipped (outputs up to date).")
        else:
            print(f"{stage_label} Stage completed.")
        self.reporter.log_stage_completion(
            stage_label=stage_label,
            stage_key=stage_key,
            context=context,
            stage_start=t_stage,
            profile_stage=profile_stage,
            profile_interval_s=self.profile_interval_s,
        )
        return context

    def run(self) -> Context:
        """
        Run all pipeline stages sequentially for this CT input.

        Phase 1 — Digital Twin & Ground Truth:
            1.1  TotalSegmentator segmentation + ROI unification
            1.2  Synthetic lesion generation (optional)
            1.3  PBPK TAC generation

        Phase 2 — Simulations:
            2.1  SIMIND Monte Carlo SPECT projection simulation (optional, --spect)
            2.2  OpenGATE Monte Carlo dosimetry simulation (optional, --dosimetry)

        Phase 3 — Post-Processing:
            3.1  SPECT post-processing (optional, --postprocess with --spect)
            3.2  Dosimetry post-processing (optional, --postprocess with --dosimetry)

        Returns
        -------
        Context
            Updated context after all stages complete.
        """
        logger = self.logger
        t_pipeline = time.perf_counter()
        profiler = StageProfiler(interval_s=self.profile_interval_s) if self.profile else None
        if profiler:
            profile_path = os.path.join(
                self.output_folder_path, f"profiling_CT_{self.ct_index}.json"
            )
            profiler.configure_autosave(profile_path, self.config, t_pipeline)

        self.reporter.emit_patient_banner(
            config=self.config,
            ct_input=self.ct_input,
            ct_index=self.ct_index,
            ct_input_type=self.ct_input_type,
            output_folder_path=self.output_folder_path,
            run_segmentation=self.run_segmentation,
            run_pbpk=self.run_pbpk,
            run_synthetic_lesions=self.run_synthetic_lesions,
            run_spect=self.run_spect,
            run_dosimetry=self.run_dosimetry,
            run_postprocess=self.run_postprocess,
        )

        if self.mode == "DEBUG":
            self.reporter.debug(
                f"run_segmentation={self.run_segmentation} | run_pbpk={self.run_pbpk} | run_synthetic_lesions={self.run_synthetic_lesions}",
                "INIT",
            )
            self.reporter.debug(
                f"run_spect={self.run_spect} | run_dosimetry={self.run_dosimetry} | run_postprocess={self.run_postprocess}",
                "INIT",
            )

        context = self.context

        logger.info("Pipeline start | mode=%s", self.mode)
        logger.info("CT input | path=%s | type=%s", self.ct_input, self.ct_input_type)

        # ── Phase 1: Digital Twin & Ground Truth ──────────────────────────────────
        if self.run_segmentation or self.run_pbpk or self.run_synthetic_lesions:
            self.reporter.print_phase_banner(1, "Digital Twin & Ground Truth")
        else:
            print("Phase 1: Digital Twin & Ground Truth (skipped — no Phase 1 flags set)")
            logger.info("Phase 1 skipped: no Phase 1 flags set")

        if self.run_segmentation:
            from src.stages.segmentation_stage import SegmentationStage

            context = self._run_stage(
                context,
                stage_label="Segmentation + ROI Unification",
                stage_key="segmentation",
                stage_cls=SegmentationStage,
                profiler=profiler,
            )

        if self.run_synthetic_lesions:
            from src.stages.synthetic_lesions_stage import SyntheticLesionsStage

            context = self._run_stage(
                context,
                stage_label="Synthetic Lesions Generation",
                stage_key="synthetic_lesions",
                stage_cls=SyntheticLesionsStage,
                profiler=profiler,
            )

        if self.run_pbpk:
            from src.stages.pbpk_tac_stage import PbpkTacStage

            context = self._run_stage(
                context,
                stage_label="PBPK TAC Generation",
                stage_key="pbpk_tac",
                stage_cls=PbpkTacStage,
                profiler=profiler,
            )

        # ── Phase 1 → Phase 2 handoff ─────────────────────────────────────────
        # If segmentation was skipped, resolve the Phase 1 outputs from disk so
        # Phase 2 stages can find them (they were produced by a prior run).
        if (self.run_spect or self.run_dosimetry) and not self.run_segmentation:
            phase1_dir = context.subdir_paths["phase_1"]
            if context.ct_nii_path is None:
                context.ct_nii_path = os.path.join(phase1_dir, "ct.nii.gz")
            if context.pytheratwin_roi_seg_path is None:
                context.pytheratwin_roi_seg_path = os.path.join(phase1_dir, "digital_twin.nii.gz")

        # ── Phase 2: Simulations ───────────────────────────────────────────────
        if self.run_spect or self.run_dosimetry:
            self.reporter.print_phase_banner(2, "Simulations")
        else:
            print("Phase 2: Simulations (skipped — no --spect or --dosimetry)")
            logger.info("Phase 2 skipped: no --spect or --dosimetry flags")

        if self.run_spect:
            from src.stages.simind_simulation_stage import SimindSimulationStage

            context = self._run_stage(
                context,
                stage_label="SIMIND Simulation",
                stage_key="simind",
                stage_cls=SimindSimulationStage,
                profiler=profiler,
            )

        if self.run_dosimetry:
            from src.stages.opengate_simulation_stage import OpenGateSimulationStage

            context = self._run_stage(
                context,
                stage_label="OpenGATE Simulation",
                stage_key="opengate",
                stage_cls=OpenGateSimulationStage,
                profiler=profiler,
            )

        # ── Phase 3: Post-Processing ───────────────────────────────────────────
        if self.run_postprocess and (self.run_spect or self.run_dosimetry):
            self.reporter.print_phase_banner(3, "Post-Processing")
        elif self.run_postprocess:
            print("Phase 3: Post-Processing (skipped — no simulations ran)")
            logger.info("Phase 3 skipped: --postprocess set but no simulations ran")
        else:
            print("Phase 3: Post-Processing (skipped — no --postprocess)")
            logger.info("Phase 3 skipped: no --postprocess flag")

        if self.run_postprocess and self.run_spect:
            from src.stages.spect_postprocess_stage import SpectPostprocessStage

            context = self._run_stage(
                context,
                stage_label="SPECT Post-Processing",
                stage_key="spect_postprocess",
                stage_cls=SpectPostprocessStage,
                profiler=profiler,
            )

        if self.run_postprocess and self.run_dosimetry:
            from src.stages.dosemap_postprocess_stage import DosemapPostprocessStage

            context = self._run_stage(
                context,
                stage_label="Dosimetry Post-Processing",
                stage_key="dosemap_postprocess",
                stage_cls=DosemapPostprocessStage,
                profiler=profiler,
            )

        # ── PRODUCTION cleanup ─────────────────────────────────────────────────
        if self.mode == "PRODUCTION":
            simind_work = getattr(context, "simind_work_dir", None)
            if simind_work and os.path.exists(simind_work):
                self._cleanup_work_dir(simind_work)
            dosimetry_work = getattr(context, "dosimetry_work_dir", None)
            if dosimetry_work and os.path.exists(dosimetry_work):
                self._cleanup_work_dir(dosimetry_work)

        pipeline_elapsed = time.perf_counter() - t_pipeline
        logger.info("Pipeline end | total_elapsed=%.2fs", pipeline_elapsed)

        if profiler:
            profiler.save(profile_path, pipeline_elapsed, self.config)
            print(f"Profiling data saved to: {profile_path}")

        print("PyTheraTwin pipeline completed successfully.")
        return context


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the PyTheraTwin pipeline."""
    parser = argparse.ArgumentParser(
        description="PyTheraTwin pipeline runner"
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
        "--segmentation",
        action="store_true",
        default=False,
        help="Run TotalSegmentator segmentation + ROI unification in Phase 1. Default: disabled.",
    )
    parser.add_argument(
        "--pbpk",
        action="store_true",
        default=False,
        help="Run PBPK TAC generation in Phase 1. Default: disabled.",
    )
    parser.add_argument(
        "--synthetic_lesions",
        action="store_true",
        default=False,
        help="Run synthetic lesion generation in Phase 1 (requires specs in config). Default: disabled.",
    )
    parser.add_argument(
        "--spect",
        action="store_true",
        default=False,
        help="Run SIMIND SPECT projection simulation in Phase 2. Default: disabled.",
    )
    parser.add_argument(
        "--dosimetry",
        action="store_true",
        default=False,
        help="Run OpenGATE dosimetry simulation in Phase 2. Default: disabled.",
    )
    parser.add_argument(
        "--postprocess",
        action="store_true",
        default=False,
        help="Run post-processing in Phase 3 for whichever simulations ran. Default: disabled.",
    )
    parser.add_argument(
        "--profile",
        type=parse_profile_interval_arg,
        default=None,
        metavar="INTERVAL_S",
        help=(
            "Enable CPU/RAM profiling and write profiling_CT_<index>.json into each CT output folder. "
            f"Pass the sampling interval in seconds "
            f"({StageProfiler.MIN_INTERVAL_S}–{StageProfiler.MAX_INTERVAL_S}). "
            "Requires psutil."
        ),
    )
    parser.add_argument(
        "--web_ui",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help=argparse.SUPPRESS,
    )

    return parser
def main() -> int:
    """
    CLI entrypoint. Iterates through all CT inputs and runs the pipeline.

    Returns
    -------
    int
        Process exit code (0 = all CTs processed; individual failures logged but don't stop batch).
    """
    parser = build_arg_parser()
    args = parser.parse_args()

    ct_inputs_dir = os.path.abspath(args.input_ct_dir)
    if not os.path.isdir(ct_inputs_dir):
        raise NotADirectoryError(f"input_ct_dir must be a directory: {ct_inputs_dir}")
    items, skipped_items = discover_ct_inputs(ct_inputs_dir)
    if not items:
        msg = f"\n[ERROR] No supported CT inputs found in: {ct_inputs_dir}\n"
        if skipped_items:
            msg += (
                "Ignored unsupported entries: "
                + ", ".join(skipped_items[:10])
                + (" ..." if len(skipped_items) > 10 else "")
                + "\n"
            )
        print(msg)
        return 1

    # Pre-flight: validate config before touching any CT.
    # Inject computed paths first so validation sees the real values.
    try:
        _cfg_raw = load_and_validate_runtime_config(
            args.config_file,
            repo_root=os.path.abspath(os.path.dirname(__file__)),
            run_spect=args.spect,
            run_dosimetry=args.dosimetry,
            run_postprocess=args.postprocess,
        )
    except ValueError as _ve:
        print(f"\n[ERROR] {_ve}\n")
        return 1
    except FileNotFoundError as _fnf:
        print(f"\n[ERROR] Config file not found: {_fnf}\n")
        return 1

    startup_reporter = PipelineReporter(mode=args.mode)
    _banner_lines = startup_reporter.emit_startup_banner(
        args,
        items,
        _cfg_raw,
        skipped_items=skipped_items,
    )

    any_failed = False
    for idx, name in enumerate(items, start=1):
        ct_path = os.path.join(ct_inputs_dir, name)
        pipeline: Optional[PyTheraTwinPipeline] = None

        # In web UI mode each job runs with one CT in a temp dir (idx is always 1).
        # Derive the actual index from the output directory name (e.g. CT_11 → 11).
        if args.output_dir:
            _folder = os.path.basename(os.path.abspath(args.output_dir))
            _m = re.match(r"^CT_(\d+)$", _folder)
            ct_index = int(_m.group(1)) if _m else idx
        else:
            ct_index = idx

        try:
            pipeline = PyTheraTwinPipeline(
                config_path=args.config_file,
                ct_input=ct_path,
                ct_index=ct_index,
                mode=args.mode,
                web_ui=args.web_ui,
                run_segmentation=args.segmentation,
                run_pbpk=args.pbpk,
                run_synthetic_lesions=args.synthetic_lesions,
                run_spect=args.spect,
                run_dosimetry=args.dosimetry,
                run_postprocess=args.postprocess,
                startup_banner_lines=_banner_lines,
                profile_interval_s=args.profile,
                output_dir=args.output_dir,
            )
            pipeline.run()
        except Exception:
            if pipeline is not None:
                pipeline.logger.exception("Pipeline failed for CT_%s (%s)", ct_index, ct_path)
            print(f"[ERROR] CT index {ct_index} failed for input: {ct_path}")
            traceback.print_exc()
            any_failed = True

    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
