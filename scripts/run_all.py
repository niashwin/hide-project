#!/usr/bin/env python3
"""
HIDE Project -- Master Reproduction Script
============================================
Runs all experiments across 5 seeds and generates figures.

Usage:
    python scripts/run_all.py                  # Full pipeline
    python scripts/run_all.py --phase 1        # Single phase
    python scripts/run_all.py --figures-only    # Regenerate figures from results
    python scripts/run_all.py --skip-download   # Skip dataset pre-caching
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

SEEDS = [42, 123, 456, 789, 1024]


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("HIDE")


def run_script(script_path, logger, description=""):
    """Run a Python script as a subprocess."""
    logger.info(f"Running: {description or script_path}")
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=False,
    )
    if result.returncode != 0:
        logger.error(f"FAILED: {script_path} (exit code {result.returncode})")
        return False
    logger.info(f"  OK: {description or script_path}")
    return True


def download_data(logger):
    """Pre-download datasets."""
    logger.info("=" * 60)
    logger.info("DOWNLOADING DATASETS")
    logger.info("=" * 60)
    download_script = PROJECT_ROOT / "data" / "download_data.sh"
    if download_script.exists():
        subprocess.run(["bash", str(download_script)], cwd=str(PROJECT_ROOT))


def run_experiments(phases, logger):
    """Run experiment scripts for specified phases."""
    phase_scripts = {
        1: PROJECT_ROOT / "experiments" / "phase1" / "run_phase1.py",
        2: PROJECT_ROOT / "experiments" / "phase2" / "run_phase2.py",
        3: PROJECT_ROOT / "experiments" / "phase3" / "run_phase3.py",
        4: PROJECT_ROOT / "experiments" / "phase4" / "run_phase4.py",
        5: PROJECT_ROOT / "experiments" / "phase5" / "run_phase5.py",
    }

    spectral_scripts = [
        (PROJECT_ROOT / "experiments" / "spectral" / "run_spectral.py", "Effective dimensionality analysis"),
        (PROJECT_ROOT / "experiments" / "spectral" / "run_minilm_interference.py", "MiniLM interference experiment"),
    ]

    for p in phases:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"PHASE {p}")
        logger.info("=" * 60)
        script = phase_scripts.get(p)
        if script and script.exists():
            run_script(script, logger, f"Phase {p} experiments")
        else:
            logger.warning(f"Phase {p} script not found: {script}")

    # Spectral analysis (after main phases)
    if not phases or 5 in phases:
        logger.info(f"\n{'=' * 60}")
        logger.info("SPECTRAL ANALYSIS")
        logger.info("=" * 60)
        for script, desc in spectral_scripts:
            if script.exists():
                run_script(script, logger, desc)


def generate_figures(logger):
    """Generate all figures from results."""
    logger.info(f"\n{'=' * 60}")
    logger.info("GENERATING FIGURES")
    logger.info("=" * 60)

    fig_script = SCRIPT_DIR / "generate_figures.py"
    if fig_script.exists():
        run_script(fig_script, logger, "All figures")
    else:
        # Fall back to individual scripts
        fig_scripts = sorted(SCRIPT_DIR.glob("gen_fig*.py")) + sorted(SCRIPT_DIR.glob("gen_extended_data*.py"))
        for script in fig_scripts:
            run_script(script, logger, script.stem)


def main():
    parser = argparse.ArgumentParser(description="HIDE Project Reproduction Script")
    parser.add_argument("--phase", type=int, default=None, help="Run single phase (1-5)")
    parser.add_argument("--figures-only", action="store_true", help="Only regenerate figures")
    parser.add_argument("--skip-download", action="store_true", help="Skip dataset downloads")
    args = parser.parse_args()

    logger = setup_logging()
    logger.info(f"HIDE Project -- {datetime.now().isoformat()}")

    if args.figures_only:
        generate_figures(logger)
        logger.info("Done (figures only).")
        return

    if not args.skip_download:
        download_data(logger)

    phases = [args.phase] if args.phase else [1, 2, 3, 4, 5]
    run_experiments(phases, logger)
    generate_figures(logger)

    logger.info(f"\n{'=' * 60}")
    logger.info("ALL COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
