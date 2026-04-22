"""
Command-line entry point for the Virtual Theranostic Trials pipeline.

This module defines the per-patient pipeline runner, injects developer-controlled
paths into the live config, and provides the batch CLI used by both local runs
and the web UI subprocess launcher.
"""

from __future__ import annotations

import os
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

from src.stages.segmentation_stage import SegmentationStage
from src.stages.synthetic_lesions_stage import SyntheticLesionsStage
from src.stages.pbpk_tac_stage import PbpkTacStage
from src.stages.simind_simulation_stage import SimindSimulationStage
from src.stages.spect_postprocess_stage import SpectPostprocessStage
from src.stages.opengate_simulation_stage import OpenGateSimulationStage
from src.stages.dosemap_postprocess_stage import DosemapPostprocessStage
class TdtPipeline:
    """
    Orchestrates the full VTT pipeline for a single CT input.

    Parameters
    ----------
    config_path : str
        Path to the JSON config file (comments allowed via `json_minify`).
    ct_input : str
        Path to a CT input (either a .nii/.nii.gz file OR a DICOM directory).
    ct_index : int
        Index used for naming (e.g., output folder suffix "_CT_{ct_index}").
    logging_on : bool, default=True
        If True, writes a per-CT log file into the CT output folder.
        The log records launched_via, config path, output folder, and timing.
    save_config : bool, default=False
        If True, saves a copy of the config JSON into the output folder.
    mode : {"DEBUG", "PRODUCTION"}, default="PRODUCTION"
        Controls verbosity and whether intermediate files are cleaned up.
    run_spect : bool, default=False
        If True, runs SIMIND SPECT simulation in Phase 2.
    run_dosimetry : bool, default=False
        If True, runs OpenGATE dosimetry simulation in Phase 2.
    run_postprocess : bool, default=False
        If True, runs post-processing in Phase 3 for whichever simulations ran.
    profile : bool, default=False
        If True, samples pipeline process-tree CPU/RAM per stage
        and writes ``profiling_CT_<ct_index>.json`` into the output folder.
    profile_interval_s : float, default=2.0
        Sampling interval in seconds used when profiling is enabled.
    """

    def __init__(
        self,
        config_path: str,
        ct_input: str,
        ct_index: int,
        logging_on: bool = True,
        save_config: bool = False,
        mode: Literal["DEBUG", "PRODUCTION"] = "PRODUCTION",
        run_spect: bool = False,
        run_dosimetry: bool = False,
        run_postprocess: bool = False,
        launched_via: str = "cli",
        startup_banner_lines: Optional[List[str]] = None,
        profile: bool = False,
        profile_interval_s: float = StageProfiler.DEFAULT_INTERVAL_S,
    ) -> None:
        self.config_path: str = config_path
        self.ct_input: str = ct_input
        self.ct_index: int = ct_index
        self.current_dir_path: str = os.path.abspath(os.path.dirname(__file__))

        self.logging_on: bool = logging_on
        self.save_config: bool = save_config
        self.launched_via: str = launched_via
        self.mode: Literal["DEBUG", "PRODUCTION"] = mode
        self.run_spect: bool = run_spect
        self.run_dosimetry: bool = run_dosimetry
        self.run_postprocess: bool = run_postprocess
        self.startup_banner_lines: List[str] = startup_banner_lines or []
        self.profile: bool = profile
        self.profile_interval_s: float = profile_interval_s

        self.config: Dict[str, Any] = {}
        self.output_folder_path: str = ""
        self.metadata_dir_path: str = ""
        self.ct_input_type: CTInputType = "dicom"
        self.ct_saved_copy_path: str = ""
        self.ct_input_identity: Dict[str, Any] = {}
        self.run_synthetic_lesions: bool = False
        self.sub_dir_names: Dict[str, str] = {}

        self.logger: logging.Logger = logging.getLogger(f"VTT_PIPELINE_CT_{self.ct_index}")
        self.logger.setLevel(logging.DEBUG if self.mode == "DEBUG" else logging.INFO)
        self.logger.propagate = False
        self.reporter = PipelineReporter(mode=self.mode, logger=self.logger)

        self._config_setup(config_path)

        if self.logging_on:
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
        else:
            self.logger.disabled = True

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
            If the config file or CT input path does not exist.
        ValueError
            If the CT input file is not a supported NIfTI extension.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        self.config = load_runtime_config(
            config_path,
            repo_root=self.current_dir_path,
            include_input_paths=True,
        )

        output_folder_title = f"{self.config['output_folder_title']}_CT_{self.ct_index}"
        self.output_folder_path = os.path.join(self.current_dir_path, output_folder_title)
        os.makedirs(self.output_folder_path, exist_ok=True)
        self.metadata_dir_path = str(ensure_metadata_dir(self.output_folder_path))

        self.ct_input_type = validate_ct_input_path(self.ct_input)

        if self.save_config:
            self._save_config_copy(config_path)

        self._save_ct_scan_copy()

        # Run synthetic lesions automatically when specs are defined in config.
        lesion_specs = self.config["phase_1"]["synthetic_lesions_stage"].get("specs")
        self.run_synthetic_lesions = bool(lesion_specs)

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
        Execute all pipeline stages sequentially for this CT input.

        Phases
        ------
        1. Digital Twin & Ground Truth:
            1.1  TotalSegmentator segmentation + ROI unification to the shared label space
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
            run_synthetic_lesions=self.run_synthetic_lesions,
            run_spect=self.run_spect,
            run_dosimetry=self.run_dosimetry,
            run_postprocess=self.run_postprocess,
        )

        if self.mode == "DEBUG":
            self.reporter.debug(
                f"run_spect={self.run_spect} | run_dosimetry={self.run_dosimetry} | run_postprocess={self.run_postprocess}",
                "INIT",
            )
            self.reporter.debug(
                f"synthetic_lesions={self.run_synthetic_lesions}",
                "INIT",
            )

        context = self.context

        logger.info("Pipeline start | mode=%s", self.mode)
        logger.info("CT input | path=%s | type=%s", self.ct_input, self.ct_input_type)

        # ── Phase 1: Digital Twin & Ground Truth ──────────────────────────────────
        self.reporter.print_phase_banner(1, "Digital Twin & Ground Truth")

        context = self._run_stage(
            context,
            stage_label="Segmentation + ROI Unification",
            stage_key="segmentation",
            stage_cls=SegmentationStage,
            profiler=profiler,
        )

        if self.run_synthetic_lesions:
            context = self._run_stage(
                context,
                stage_label="Synthetic Lesions Generation",
                stage_key="synthetic_lesions",
                stage_cls=SyntheticLesionsStage,
                profiler=profiler,
            )

        context = self._run_stage(
            context,
            stage_label="PBPK TAC Generation",
            stage_key="pbpk_tac",
            stage_cls=PbpkTacStage,
            profiler=profiler,
        )

        # ── Phase 2: Simulations ───────────────────────────────────────────────
        if self.run_spect or self.run_dosimetry:
            self.reporter.print_phase_banner(2, "Simulations")
        else:
            print("Phase 2: Simulations (skipped — no --spect or --dosimetry)")
            logger.info("Phase 2 skipped: no --spect or --dosimetry flags")

        if self.run_spect:
            context = self._run_stage(
                context,
                stage_label="SIMIND Simulation",
                stage_key="simind",
                stage_cls=SimindSimulationStage,
                profiler=profiler,
            )

        if self.run_dosimetry:
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
            context = self._run_stage(
                context,
                stage_label="SPECT Post-Processing",
                stage_key="spect_postprocess",
                stage_cls=SpectPostprocessStage,
                profiler=profiler,
            )

        if self.run_postprocess and self.run_dosimetry:
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

        pipeline_elapsed = time.perf_counter() - t_pipeline
        logger.info("Pipeline end | total_elapsed=%.2fs", pipeline_elapsed)

        if profiler:
            profiler.save(profile_path, pipeline_elapsed, self.config)
            print(f"Profiling data saved to: {profile_path}")

        print("VTT pipeline completed successfully.")
        return context


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the VTT pipeline."""
    parser = argparse.ArgumentParser(
        description="Virtual Theranostic Trials pipeline runner"
    )

    parser.add_argument("--config_file", required=True, type=str,
                        help="Path to JSON config file.")

    ct_group = parser.add_mutually_exclusive_group(required=True)
    ct_group.add_argument("--input_ct_dir", type=str,
                          help="Directory containing CT inputs (NIfTI files or DICOM folders).")
    ct_group.add_argument("--input_ct", type=str,
                          help="Direct path to a single CT input (NIfTI file or DICOM directory).")

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
        default=1,
        help="Starting CT index for output folder naming (e.g. 1 → _CT_1, _CT_2, …). Default: 1.",
    )
    parser.add_argument(
        "--save_config",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Copy the config JSON into each CT output folder. Default: disabled.",
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
    parser.add_argument(
        "--profile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Sample pipeline CPU/RAM per stage and write "
            "profiling_CT_<index>.json into each CT output folder. "
            "Requires psutil. Default: disabled."
        ),
    )
    parser.add_argument(
        "--profile_interval_s",
        type=parse_profile_interval_arg,
        default=StageProfiler.DEFAULT_INTERVAL_S,
        help=(
            "Profiler sampling interval in seconds when --profile is enabled. "
            f"Range: {StageProfiler.MIN_INTERVAL_S}-{StageProfiler.MAX_INTERVAL_S}. "
            f"Default: {StageProfiler.DEFAULT_INTERVAL_S}."
        ),
    )

    return parser
def main() -> int:
    """
    CLI entrypoint. Iterates through all CT inputs and runs the pipeline.

    Accepts either --input_ct_dir (directory of CT items) or --input_ct (single CT path).

    Returns
    -------
    int
        Process exit code (0 = all CTs processed; individual failures logged but don't stop batch).
    """
    parser = build_arg_parser()
    args = parser.parse_args()

    # Resolve CT input items from either --input_ct or --input_ct_dir.
    if args.input_ct:
        ct_path = os.path.abspath(args.input_ct)
        validate_ct_input_path(ct_path)
        ct_inputs_dir = os.path.dirname(ct_path)
        items = [os.path.basename(ct_path)]
        skipped_items: List[str] = []
    else:
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
    for idx, name in enumerate(items, start=args.ct_index_start):
        ct_path = os.path.join(ct_inputs_dir, name)
        pipeline: Optional[TdtPipeline] = None

        try:
            pipeline = TdtPipeline(
                config_path=args.config_file,
                ct_input=ct_path,
                ct_index=idx,
                logging_on=args.logging_on,
                save_config=args.save_config,
                mode=args.mode,
                run_spect=args.spect,
                run_dosimetry=args.dosimetry,
                run_postprocess=args.postprocess,
                launched_via=args.launched_via,
                startup_banner_lines=_banner_lines,
                profile=args.profile,
                profile_interval_s=args.profile_interval_s,
            )
            pipeline.run()
        except Exception:
            if pipeline is not None:
                pipeline.logger.exception("Pipeline failed for CT_%s (%s)", idx, ct_path)
            print(f"[ERROR] CT index {idx} failed for input: {ct_path}")
            traceback.print_exc()
            any_failed = True

    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())


# quick run: python main.py --config_file inputs/my_config.json --input_ct_dir inputs/ct_testing --mode DEBUG --logging_on --save_config --spect --dosimetry --postprocess
