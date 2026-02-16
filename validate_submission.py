#!/usr/bin/env python3
"""
Validate HIDE Submission Integrity
====================================
Checks that all results, figures, imports, and paper are intact.
Saves report to validation_report/ without modifying existing results.
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "results"
PAPER_DIR = PROJECT_ROOT / "paper"
FIGURES_DIR = PAPER_DIR / "figures"
REPORT_DIR = PROJECT_ROOT / "validation_report"
REPORT_DIR.mkdir(exist_ok=True)

# Add paths for imports
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "hide" / "core"))
sys.path.insert(0, str(PROJECT_ROOT / "hide" / "models"))
sys.path.insert(0, str(PROJECT_ROOT / "hide" / "utils"))

checks = []
errors = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    msg = f"[{status}] {name}"
    if detail:
        msg += f" — {detail}"
    checks.append(msg)
    if not condition:
        errors.append(name)
    print(msg)
    return condition


# ═══════════════════════════════════════════════════════════════
# 1. IMPORTS
# ═══════════════════════════════════════════════════════════════
print("\n=== 1. IMPORT CHECKS ===")
try:
    from hide_space import HIDESpace
    check("Import HIDESpace", True)
except Exception as e:
    check("Import HIDESpace", False, str(e))

try:
    from embedding_models import EmbeddingManager
    check("Import EmbeddingManager", True)
except Exception as e:
    check("Import EmbeddingManager", False, str(e))

try:
    from qwen_adapter import QwenGenerator
    check("Import QwenGenerator", True)
except Exception as e:
    check("Import QwenGenerator", False, str(e))

try:
    from metrics import accuracy, bootstrap_ci, fit_power_law, r_squared, cohens_d, aggregate_seeds, participation_ratio
    check("Import all metrics functions", True)
except Exception as e:
    check("Import all metrics functions", False, str(e))

# ═══════════════════════════════════════════════════════════════
# 2. HIDESPACE FUNCTIONAL TEST
# ═══════════════════════════════════════════════════════════════
print("\n=== 2. HIDESPACE FUNCTIONAL TEST ===")
try:
    space = HIDESpace(dim=64, max_memories=100)
    for i in range(10):
        emb = np.random.randn(64).astype(np.float32)
        space.store(emb, {"id": i, "text": f"fact {i}"})

    check("HIDESpace store", space.count == 10, f"count={space.count}")

    query = np.random.randn(64).astype(np.float32)
    results = space.retrieve(query, k=3)
    check("HIDESpace retrieve returns 3-tuples",
          len(results) == 3 and all(len(r) == 3 for r in results),
          f"got {len(results)} results, tuple lengths: {[len(r) for r in results]}")

    check("HIDESpace retrieve has metadata",
          all(isinstance(r[2], dict) and "text" in r[2] for r in results))

    # Test replace + remove_indices
    space.replace(0, np.random.randn(64).astype(np.float32))
    check("HIDESpace replace works", True)

    space.remove_indices([8, 9])
    check("HIDESpace remove_indices", space.count == 8, f"count={space.count}")

except Exception as e:
    check("HIDESpace functional test", False, str(e))

# ═══════════════════════════════════════════════════════════════
# 3. RESULTS FILES
# ═══════════════════════════════════════════════════════════════
print("\n=== 3. RESULTS FILE CHECKS ===")
SEEDS = [42, 123, 456, 789, 1024]

for phase_num in range(1, 6):
    phase_dir = RESULTS_DIR / f"phase{phase_num}"
    check(f"Phase {phase_num} results dir exists", phase_dir.exists())

    if phase_dir.exists():
        # Check per-seed results
        for seed in SEEDS:
            json_file = phase_dir / f"results_seed{seed}.json"
            check(f"Phase {phase_num} seed {seed} JSON", json_file.exists())

        # Check summary
        summary_file = phase_dir / "summary.json"
        check(f"Phase {phase_num} summary.json", summary_file.exists())

# Check special results directories
for special in ["interference", "spacing_sweep", "topology", "spectral"]:
    d = RESULTS_DIR / special
    check(f"Results/{special}/ exists", d.exists())

# ═══════════════════════════════════════════════════════════════
# 4. KEY RESULTS VALIDATION
# ═══════════════════════════════════════════════════════════════
print("\n=== 4. KEY RESULTS VALIDATION ===")

# Phase 1: HIDE should beat baselines
p1_summary = RESULTS_DIR / "phase1" / "summary.json"
if p1_summary.exists():
    with open(p1_summary) as f:
        p1 = json.load(f)

    # Check validation fields
    validation = p1.get("validation", {})
    hide_beats_no_mem = validation.get("hide_beats_no_memory", 0)
    hide_beats_random = validation.get("hide_beats_random", 0)
    check("Phase 1: HIDE beats no-memory", hide_beats_no_mem >= 4,
          f"HIDE > no_memory on {hide_beats_no_mem}/5 tasks")
    check("Phase 1: HIDE beats random", hide_beats_random >= 4,
          f"HIDE > random on {hide_beats_random}/5 tasks")

# Phase 5: DRM false memory
p5_summary = RESULTS_DIR / "phase5" / "summary.json"
if p5_summary.exists():
    with open(p5_summary) as f:
        p5 = json.load(f)

    drm_fa = p5.get("drm", {}).get("fa_critical_mean", None)
    check("Phase 5: DRM false alarm rate present", drm_fa is not None, f"fa={drm_fa}")
    if drm_fa:
        check("Phase 5: DRM FA near human (~0.55)", 0.3 < drm_fa < 0.9, f"fa={drm_fa}")

# ═══════════════════════════════════════════════════════════════
# 5. FIGURES
# ═══════════════════════════════════════════════════════════════
print("\n=== 5. FIGURE CHECKS ===")

main_figs = [
    "fig1_interference.pdf", "fig2_drm.pdf", "fig3_spacing.pdf",
    "fig4_topology.pdf", "fig5_crossmodal.pdf", "fig6_summary.pdf",
]
for fig in main_figs:
    path = FIGURES_DIR / fig
    check(f"Main figure: {fig}", path.exists(),
          f"size={path.stat().st_size}B" if path.exists() else "MISSING")

ed_figs = [f"ed_fig{i}" for i in range(1, 12)]
for ed in ed_figs:
    matches = list(FIGURES_DIR.glob(f"{ed}*.pdf"))
    check(f"Extended data: {ed}", len(matches) > 0,
          matches[0].name if matches else "MISSING")

# ═══════════════════════════════════════════════════════════════
# 6. PAPER
# ═══════════════════════════════════════════════════════════════
print("\n=== 6. PAPER CHECKS ===")

tex_file = PAPER_DIR / "hide_paper.tex"
pdf_file = PAPER_DIR / "hide_paper.pdf"

check("LaTeX source exists", tex_file.exists())
check("Compiled PDF exists", pdf_file.exists(),
      f"size={pdf_file.stat().st_size/1024:.0f}KB" if pdf_file.exists() else "MISSING")

if tex_file.exists():
    tex = tex_file.read_text()
    check("LaTeX has \\begin{document}", "\\begin{document}" in tex)
    check("LaTeX has \\end{document}", "\\end{document}" in tex)
    check("LaTeX has abstract", "\\begin{abstract}" in tex)
    check("LaTeX has bibliography", "\\bibliography" in tex or "\\begin{thebibliography}" in tex)

    # Check all figure references resolve
    import re
    includes = re.findall(r'\\includegraphics.*?\{(.+?)\}', tex)
    for inc in includes:
        fig_path = PAPER_DIR / inc
        check(f"Figure reference: {inc}", fig_path.exists())

# ═══════════════════════════════════════════════════════════════
# 7. CONFIGS
# ═══════════════════════════════════════════════════════════════
print("\n=== 7. CONFIG CHECKS ===")
config_dir = PROJECT_ROOT / "configs"
for phase in range(1, 6):
    cfg = config_dir / f"phase{phase}.yaml"
    check(f"Config phase{phase}.yaml", cfg.exists())
check("Config spectral.yaml", (config_dir / "spectral.yaml").exists())

# ═══════════════════════════════════════════════════════════════
# 8. DOCUMENTATION
# ═══════════════════════════════════════════════════════════════
print("\n=== 8. DOCUMENTATION CHECKS ===")
check("README.md exists", (PROJECT_ROOT / "README.md").exists())
check("LICENSE exists", (PROJECT_ROOT / "LICENSE").exists())
check("requirements.txt exists", (PROJECT_ROOT / "requirements.txt").exists())

# ═══════════════════════════════════════════════════════════════
# 9. CROSS-FILE CONSISTENCY
# ═══════════════════════════════════════════════════════════════
print("\n=== 9. CROSS-FILE CONSISTENCY ===")

# Check all 5 phase experiment scripts have the fixed imports
for phase in range(1, 6):
    phase_script = PROJECT_ROOT / "experiments" / f"phase{phase}" / f"run_phase{phase}.py"
    if phase_script.exists():
        content = phase_script.read_text()
        has_fix = 'hide" / "core"' in content or "hide/core" in content
        check(f"Phase {phase} script has fixed imports", has_fix)

# ═══════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
n_pass = sum(1 for c in checks if "[PASS]" in c)
n_fail = sum(1 for c in checks if "[FAIL]" in c)
print(f"TOTAL: {n_pass} passed, {n_fail} failed out of {len(checks)} checks")

if errors:
    print(f"\nFAILED CHECKS:")
    for e in errors:
        print(f"  - {e}")

# Save report
report = {
    "total_checks": len(checks),
    "passed": n_pass,
    "failed": n_fail,
    "checks": checks,
    "errors": errors,
}

report_path = REPORT_DIR / "validation_report.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

print(f"\nReport saved to {report_path}")

sys.exit(0 if n_fail == 0 else 1)
