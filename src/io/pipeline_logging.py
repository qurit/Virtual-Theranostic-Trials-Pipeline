"""
Shared terminal and log-file reporting helpers for pipeline runs.

The goal of this module is to keep ``main.py`` focused on orchestration while
centralizing banners, debug printing, and concise post-stage summaries.
"""

from __future__ import annotations

import datetime
import os
import time
import logging
from typing import Any, Dict, List, Optional

from src.utils.dicom_utils import extract_height_weight


class PipelineReporter:
    """Terminal and logging helper for pipeline startup and per-patient runs."""

    _GREEN = "\033[92m"
    _CYAN = "\033[96m"
    _YELLOW = "\033[93m"
    _RESET = "\033[0m"
    _BOLD = "\033[1m"

    def __init__(self, mode: str = "PRODUCTION", logger: Optional[Any] = None) -> None:
        self.mode = mode
        self.logger = logger

    @staticmethod
    def configure_file_logger(
        logger: logging.Logger,
        *,
        log_path: str,
        startup_banner_lines: Optional[List[str]],
        launched_via: str,
        config_path: str,
        output_folder_path: str,
    ) -> logging.Logger:
        """Attach a per-CT file handler to *logger* if needed and write run-header lines."""
        if not any(
            isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == log_path
            for h in logger.handlers
        ):
            handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            logger.addHandler(handler)

        if startup_banner_lines:
            for line in startup_banner_lines:
                logger.info(line)

        logger.info("----Log started----")
        logger.info("Launched via: %s", launched_via)
        logger.info("Config file:  %s", config_path)
        logger.info("Output folder: %s", output_folder_path)
        return logger

    def debug(self, msg: str, phase: str = "") -> None:
        """Print a colored debug message to terminal and optionally mirror it to the logger."""
        colored = (
            f"{self._GREEN}[DEBUG]{self._RESET} {self._CYAN}[{phase}]{self._RESET} {msg}"
            if phase
            else f"{self._GREEN}[DEBUG]{self._RESET} {msg}"
        )
        print(colored)
        if self.logger:
            self.logger.debug(f"[{phase}] {msg}" if phase else msg)

    def print_phase_banner(self, phase_num: int, phase_name: str) -> None:
        """Print a phase banner with optional DEBUG coloring."""
        banner = (
            f"-----------------------------Phase {phase_num}: {phase_name}"
            "-----------------------------"
        )
        if self.mode == "DEBUG":
            print(f"{self._BOLD}{self._YELLOW}{banner}{self._RESET}")
        else:
            print(banner)

    def emit_startup_banner(
        self,
        args: Any,
        items: List[str],
        cfg: Dict[str, Any],
        skipped_items: Optional[List[str]] = None,
    ) -> List[str]:
        """Print the batch startup banner and return the plain-text lines."""
        avail = os.cpu_count() or 1
        width = 68
        border = "=" * width
        lines: List[str] = []

        def add(text: str = "") -> None:
            lines.append(text)

        add(f"+{border}+")
        add(f"|  {'Virtual Theranostic Trials  -  VTT Pipeline'.center(width - 4)}  |")
        add(f"|  {'BC Cancer  -  QURIT Lab'.center(width - 4)}  |")
        add(f"+{border}+")
        add()
        add(f"  Started   : {datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
        add(f"  Mode      : {args.mode}")
        add(f"  CT inputs : {len(items)} item(s)")
        if skipped_items:
            suffix = "y" if len(skipped_items) == 1 else "ies"
            add(f"  Ignored   : {len(skipped_items)} unsupported entr{suffix}")
        add()

        phase1_tasks = ["segmentation", "PBPK"]
        if cfg.get("phase_1", {}).get("synthetic_lesions_stage", {}).get("specs"):
            phase1_tasks.append("synthetic lesions")
        add("  Phases to run:")
        add(f"    OK  Phase 1  -  {', '.join(phase1_tasks)}")
        if args.spect:
            nc = cfg.get("phase_2", {}).get("simind_stage", {}).get("num_cpu", 1)
            eff = avail if nc == 0 else min(nc, avail)
            using = f"using all {avail}" if nc == 0 else str(eff)
            add(f"    OK  Phase 2  -  SIMIND SPECT  ({using} core(s))")
        if args.dosimetry:
            gate_cfg = cfg.get("phase_2", {}).get("opengate_stage", {}).get("gate", {})
            nc2 = gate_cfg.get("num_threads", 1)
            eff2 = avail if nc2 == 0 else min(nc2, avail)
            using2 = f"using all {avail}" if nc2 == 0 else str(eff2)
            add(f"    OK  Phase 2  -  OpenGATE dosimetry  ({using2} thread(s))")
        if args.postprocess:
            sims: List[str] = []
            if args.spect:
                sims.append("SPECT")
            if args.dosimetry:
                sims.append("dose map")
            add(f"    OK  Phase 3  -  Post-processing  ({', '.join(sims) if sims else 'no simulations'})")
        add()

        add(f"  CPU available : {avail} core(s)")
        if args.spect:
            nc = cfg.get("phase_2", {}).get("simind_stage", {}).get("num_cpu", 1)
            eff_nc = avail if nc == 0 else min(nc, avail)
            note = f"num_cpu={nc} -> using all {avail}" if nc == 0 else f"num_cpu={nc}"
            add(f"  SIMIND        : {eff_nc} core(s)  ({note})")
        if args.dosimetry:
            gate_cfg2 = cfg.get("phase_2", {}).get("opengate_stage", {}).get("gate", {})
            nc2 = gate_cfg2.get("num_threads", 1)
            eff_nc2 = avail if nc2 == 0 else min(nc2, avail)
            note2 = f"num_threads={nc2} -> using all {avail}" if nc2 == 0 else f"num_threads={nc2}"
            add(f"  OpenGATE      : {eff_nc2} thread(s)  ({note2})")
        if args.profile:
            add(f"  Profiling     : ON  (interval={args.profile_interval_s:.3g} s)")
        add()
        add("  Existing stage outputs are reused only when CT/config metadata still match.")
        add()
        add("-" * (width + 2))
        add()

        for line in lines:
            print(line)

        return lines

    def emit_patient_banner(
        self,
        *,
        config: Dict[str, Any],
        ct_input: str,
        ct_index: int,
        ct_input_type: str,
        output_folder_path: str,
        run_synthetic_lesions: bool,
        run_spect: bool,
        run_dosimetry: bool,
        run_postprocess: bool,
    ) -> None:
        """Print and log a per-patient summary banner."""
        avail = os.cpu_count() or 1
        width = 68
        lines: List[str] = []

        def add(text: str = "") -> None:
            lines.append(text)

        ct_name = os.path.basename(ct_input.rstrip("/\\"))
        ct_type_str = "DICOM directory" if ct_input_type == "dicom" else "NIfTI"
        border = "=" * width
        add(f"+{border}+")
        add(f"|  {'Patient  CT_' + str(ct_index) + '  -  ' + ct_name:<{width - 4}}  |")
        add(f"+{border}+")
        add(f"  Input      : {ct_input}  ({ct_type_str})")
        add(f"  Output     : {output_folder_path}")

        hw = self._read_patient_hw(ct_input, ct_input_type)
        if hw["source"] == "not_dicom":
            add("  Height/Wt  : not available  (NIfTI input - no DICOM tags)")
        elif hw["source"] in ("no_files", "pydicom_missing"):
            add("  Height/Wt  : not available  (could not read DICOM files)")
        else:
            h_str = f"{hw['height_m']:.2f} m" if hw["height_m"] is not None else "not found"
            w_str = f"{hw['weight_kg']:.1f} kg" if hw["weight_kg"] is not None else "not found"
            add(f"  Height     : {h_str:<14}  Weight : {w_str}")
            if hw["missing"]:
                add(f"  Missing    : {', '.join(hw['missing'])}  (tag absent in DICOM headers)")

        add("  " + "-" * (width - 2))

        seg_cfg = config.get("phase_1", {}).get("segmentation_stage", {})
        pbpk_cfg = config.get("phase_1", {}).get("pbpk_tac_stage", {})
        les_cfg = config.get("phase_1", {}).get("synthetic_lesions_stage", {})

        roi_list = seg_cfg.get("roi_subset", [])
        roi_str = ", ".join(roi_list) if roi_list else "-"
        add("  Phase 1  -  Digital Twin & Ground Truth")
        add(f"    ROI subset   : {roi_str}")
        add(
            "    PBPK model   : "
            f"{pbpk_cfg.get('model_type', '?')}  -  isotope : {pbpk_cfg.get('isotope', '?')}"
        )

        if run_synthetic_lesions:
            specs = les_cfg.get("specs") or {}
            organs = ", ".join(specs.keys()) if specs else "none"
            add(f"    Syn. lesions : ON  ({organs})")
        else:
            add("    Syn. lesions : OFF")

        add("  Phase 2  -  Simulations")
        if run_spect:
            simind_cfg = config.get("phase_2", {}).get("simind_stage", {})
            nc = simind_cfg.get("num_cpu", 0)
            eff = avail if nc == 0 else min(nc, avail)
            cpu_note = f"all {avail}" if nc == 0 else str(eff)
            add(
                "    SPECT        : "
                f"collimator={simind_cfg.get('Collimator', '?')}  "
                f"photons={simind_cfg.get('NumPhotons', '?')}  "
                f"proj={simind_cfg.get('NumProjections', '?')}  "
                f"CPUs={cpu_note}"
            )
            add(f"                   ROIs : {', '.join(simind_cfg.get('roi_subset', roi_list))}")
        else:
            add("    SPECT        : skipped")

        if run_dosimetry:
            og_cfg = config.get("phase_2", {}).get("opengate_stage", {})
            gate_cfg = og_cfg.get("gate", {})
            nc2 = gate_cfg.get("num_threads", 0)
            eff2 = avail if nc2 == 0 else min(nc2, avail)
            cpu_note2 = f"all {avail}" if nc2 == 0 else str(eff2)
            add(
                "    Dosimetry    : "
                f"batch_histories={gate_cfg.get('total_histories_per_batch', '?')}  "
                f"target_unc={gate_cfg.get('target_uncertainty_percent', '?')}%  "
                f"threads={cpu_note2}  "
            )
            add(f"                   ROIs : {', '.join(og_cfg.get('roi_subset', roi_list))}")
        else:
            add("    Dosimetry    : skipped")

        add("  Phase 3  -  Post-Processing")
        if run_postprocess and run_spect:
            sp_cfg = config.get("phase_3", {}).get("spect_postprocess_stage", {})
            add(
                "    SPECT recon  : "
                f"{sp_cfg.get('ReconstructionAlgorithm', 'OSEM')}  "
                f"iter={sp_cfg.get('Iterations', '?')}  "
                f"subsets={sp_cfg.get('Subsets', '?')}  "
                f"noise={'ON' if sp_cfg.get('apply_poisson_noise') else 'OFF'}"
            )
        elif run_postprocess:
            add("    SPECT recon  : skipped  (no SPECT simulation)")
        else:
            add("    SPECT recon  : skipped")

        if run_postprocess and run_dosimetry:
            dp_cfg = config.get("phase_3", {}).get("dosemap_postprocess_stage", {})
            add(
                "    Dose map     : TAC-weighted  "
                f"apply_tac={'ON' if dp_cfg.get('apply_tac', True) else 'OFF'}"
            )
        elif run_postprocess:
            add("    Dose map     : skipped  (no dosimetry simulation)")
        else:
            add("    Dose map     : skipped")

        add("  " + "-" * (width - 2))

        if self.mode == "DEBUG":
            print(f"{self._BOLD}{self._CYAN}", end="")
        for line in lines:
            print(line)
        if self.mode == "DEBUG":
            print(self._RESET, end="")

        if self.logger:
            for line in lines:
                self.logger.info(line)

    def log_stage_completion(
        self,
        *,
        stage_label: str,
        stage_key: str,
        context: Any,
        stage_start: float,
        profile_stage: Optional[Dict[str, Any]] = None,
        profile_interval_s: float = 2.0,
    ) -> None:
        """Write stage timing, high-level stage outputs, and optional profiler summaries."""
        if self.logger is None:
            return

        elapsed = time.perf_counter() - stage_start
        self.logger.info("Stage end: %s | elapsed=%.2fs", stage_label, elapsed)

        for line in self._build_stage_summary_entries(stage_key, context):
            self.logger.info("Stage summary [%s] | %s", stage_key, line)

        if profile_stage:
            samples = len(profile_stage.get("sample_times_s", []) or [])
            self.logger.info(
                "Stage profile [%s] | interval=%.3fs | samples=%d | raw profiler samples captured",
                stage_key,
                profile_interval_s,
                samples,
            )

    @staticmethod
    def _read_patient_hw(ct_input: str, ct_input_type: str) -> Dict[str, Any]:
        """Extract patient height/weight from DICOM headers when available."""
        if ct_input_type != "dicom":
            return {
                "height_m": None,
                "weight_kg": None,
                "missing": ["height", "weight"],
                "source": "not_dicom",
            }

        try:
            height, weight = extract_height_weight(ct_input)
        except Exception:
            return {
                "height_m": None,
                "weight_kg": None,
                "missing": ["height", "weight"],
                "source": "no_files",
            }

        missing = (["height"] if height is None else []) + (["weight"] if weight is None else [])
        return {
            "height_m": height,
            "weight_kg": weight,
            "missing": missing,
            "source": "dicom",
        }

    @staticmethod
    def _format_seq(values: Optional[List[Any] | tuple[Any, ...]], max_items: int = 6) -> str:
        """Format a short one-line preview of a sequence for log messages."""
        if not values:
            return "none"
        txt = [str(v) for v in values]
        if len(txt) <= max_items:
            return ", ".join(txt)
        return f"{', '.join(txt[:max_items])}, ... ({len(txt)} total)"

    def _build_stage_summary_entries(self, stage_key: str, context: Any) -> List[str]:
        """Select a few useful stage-specific summary lines for the per-patient log."""
        extras = getattr(context, "extras", {}) or {}
        lines: List[str] = []

        if stage_key == "segmentation":
            seg_extra = extras.get("segmentation_stage", {})
            if context.vtt_roi_seg_path:
                lines.append(f"roi_handoff={context.vtt_roi_seg_path}")
            if seg_extra.get("metadata_path"):
                lines.append(f"metadata={seg_extra['metadata_path']}")
        elif stage_key == "synthetic_lesions":
            results = getattr(context, "synthetic_lesions_results", {}) or {}
            lesion_organs = sorted(results.keys()) if isinstance(results, dict) else []
            total_lesions = sum(
                len(meta.get("centers_zyx", []))
                for meta in results.values()
                if isinstance(meta, dict)
            )
            lines.append(f"lesion_organs={self._format_seq(lesion_organs)}")
            lines.append(f"total_lesions={total_lesions}")
            synth_extra = extras.get("synthetic_lesions_stage", {})
            if synth_extra.get("metadata_path"):
                lines.append(f"metadata={synth_extra['metadata_path']}")
        elif stage_key == "pbpk_tac":
            if context.pbpk_tac_json_path:
                lines.append(f"tac_json={context.pbpk_tac_json_path}")
            if context.pbpk_stop_time_min is not None:
                lines.append(f"stop_time_min={float(context.pbpk_stop_time_min):.2f}")
            pbpk_extra = extras.get("pbpk_tac_stage", {})
            if pbpk_extra.get("metadata_path"):
                lines.append(f"metadata={pbpk_extra['metadata_path']}")
        elif stage_key == "simind":
            simind_extra = extras.get("simind_simulation_stage", {})
            roi_names = sorted((context.simind_projection_paths or {}).keys())
            lines.append(f"simulated_rois={self._format_seq(roi_names)}")
            if simind_extra.get("skipped_zero_voxel_rois"):
                lines.append(
                    "skipped_zero_voxel_rois="
                    f"{self._format_seq(simind_extra.get('skipped_zero_voxel_rois', []))}"
                )
            if context.arr_shape_new is not None:
                lines.append(f"grid_zyx={tuple(context.arr_shape_new)}")
            if context.simind_total_num_voxels is not None:
                lines.append(f"total_source_voxels={int(context.simind_total_num_voxels)}")
            if context.simind_scale_factor is not None:
                lines.append(f"scale_factor={float(context.simind_scale_factor):.6g}")
            if context.simind_metadata_path:
                lines.append(f"metadata={context.simind_metadata_path}")
        elif stage_key == "opengate":
            og_extra = extras.get("opengate_simulation_stage", {})
            simulated_rois = og_extra.get("simulated_roi_names")
            if not simulated_rois and isinstance(context.dosimetry_raw_dose_paths, dict):
                simulated_rois = sorted(context.dosimetry_raw_dose_paths.keys())
            lines.append(
                f"simulated_rois={self._format_seq(simulated_rois or [])}"
            )
            if og_extra.get("skipped_zero_voxel_rois"):
                lines.append(
                    "skipped_zero_voxel_rois="
                    f"{self._format_seq(og_extra.get('skipped_zero_voxel_rois', []))}"
                )
            if og_extra.get("simulated_actual_histories_this_run") is not None:
                lines.append(f"simulated_actual_histories_this_run={int(og_extra['simulated_actual_histories_this_run'])}")
            if og_extra.get("actual_histories_per_batch") is not None:
                lines.append(f"actual_histories_per_batch={int(og_extra['actual_histories_per_batch'])}")
            if og_extra.get("target_uncertainty_percent") is not None:
                lines.append(f"target_uncertainty={float(og_extra['target_uncertainty_percent']):.6g}%")
            if og_extra.get("uncertainty_percentile") is not None:
                lines.append(f"uncertainty_percentile={float(og_extra['uncertainty_percentile']):.6g}")
            if og_extra.get("effective_num_threads") is not None:
                lines.append(f"threads={int(og_extra['effective_num_threads'])}")
            if og_extra.get("history_rounding_loss_per_batch") is not None:
                lines.append(f"history_rounding_loss_per_batch={int(og_extra['history_rounding_loss_per_batch'])}")
            if og_extra.get("nonconverged_rois"):
                lines.append(f"nonconverged_rois={self._format_seq(og_extra.get('nonconverged_rois', []))}")
            if context.dosimetry_sum_dose_path:
                lines.append(f"sum_dose={context.dosimetry_sum_dose_path}")
            if getattr(context, "dosimetry_sum_uncertainty_path", None):
                lines.append(f"sum_uncertainty={context.dosimetry_sum_uncertainty_path}")
            if context.dosimetry_metadata_path:
                lines.append(f"metadata={context.dosimetry_metadata_path}")
        elif stage_key == "spect_postprocess":
            spect_extra = extras.get("spect_postprocess_stage", {})
            frame_paths = context.pbpk_projection_paths or {}
            if isinstance(frame_paths, dict):
                lines.append(f"frames={len(frame_paths)}")
            if context.reconstruction_output_dir:
                lines.append(f"recon_output_dir={context.reconstruction_output_dir}")
            folded_rois = spect_extra.get("folded_remaining_body_rois") or spect_extra.get("remaining_body_folded_rois")
            if folded_rois:
                lines.append(f"remaining_body_folded_rois={self._format_seq(folded_rois)}")
            if spect_extra.get("rest_owned_skipped_rois"):
                lines.append(
                    "rest_owned_skipped_rois="
                    f"{self._format_seq(spect_extra.get('rest_owned_skipped_rois', []))}"
                )
            if spect_extra.get("metadata_path"):
                lines.append(f"metadata={spect_extra['metadata_path']}")
        elif stage_key == "dosemap_postprocess":
            dose_extra = extras.get("dosemap_postprocess_stage", {})
            if context.dosemap_postprocess_dose_path:
                lines.append(f"dose_path={context.dosemap_postprocess_dose_path}")
            roi_inputs = context.dosimetry_raw_dose_paths or {}
            if isinstance(roi_inputs, dict):
                lines.append(f"dose_source_rois={len(roi_inputs)}")
            folded_rois = dose_extra.get("folded_remaining_body_rois") or dose_extra.get("remaining_body_folded_rois")
            if folded_rois:
                lines.append(f"remaining_body_folded_rois={self._format_seq(folded_rois)}")
            if dose_extra.get("rest_owned_skipped_rois"):
                lines.append(
                    "rest_owned_skipped_rois="
                    f"{self._format_seq(dose_extra.get('rest_owned_skipped_rois', []))}"
                )
            if dose_extra.get("metadata_path"):
                lines.append(f"metadata={dose_extra['metadata_path']}")

        return lines
