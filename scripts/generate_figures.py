#!/usr/bin/env python3
"""
Generate all HIDE figures from cached results.
Requires only CPU -- no GPU needed.

Usage:
    python scripts/generate_figures.py
"""

import sys
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent


def main():
    # Main figures (Fig 1-6)
    main_figs = [
        "gen_fig1_interference.py",
        "gen_fig2_drm.py",
        "gen_fig3_spacing.py",
        "gen_fig4_topology.py",
        "gen_fig5_crossmodal.py",
        "gen_fig6_summary.py",
    ]

    # Extended data figures
    ed_figs = [
        "gen_extended_data_1to5.py",
        "gen_extended_data_6to10.py",
        "gen_extended_data_dimensionality.py",
    ]

    all_scripts = main_figs + ed_figs
    success = 0
    failed = 0

    for script_name in all_scripts:
        script_path = SCRIPT_DIR / script_name
        if not script_path.exists():
            print(f"  SKIP (not found): {script_name}")
            continue

        print(f"  Generating: {script_name} ...", end=" ", flush=True)
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("OK")
            success += 1
        else:
            print(f"FAILED (exit {result.returncode})")
            if result.stderr:
                print(f"    {result.stderr[:200]}")
            failed += 1

    print(f"\nDone: {success} succeeded, {failed} failed out of {len(all_scripts)} scripts.")


if __name__ == "__main__":
    main()
