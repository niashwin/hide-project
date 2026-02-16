"""Basic import and sanity tests for the HIDE package."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_core_imports():
    from hide.core.hide_space import HIDESpace
    from hide.core.temporal import TemporalEncoding, power_law_decay
    from hide.core.interference import age_proportional_noise, fit_forgetting_curve, bootstrap_ci
    from hide.core.consolidation import consolidate_memories
    from hide.core.emergent import DRM_LISTS, drm_experiment
    from hide.core.multimodal import CrossModalSpace
    from hide.core.gpu_manager import get_device
    print("All core imports OK")


def test_model_imports():
    from hide.models.embedding_models import EmbeddingManager
    from hide.models.qwen_adapter import QwenAdapter
    print("All model imports OK")


def test_utils_imports():
    from hide.utils.metrics import bootstrap_ci, cohens_d, participation_ratio
    from hide.utils.data_loader import load_drm_word_lists
    from hide.utils.visualization import set_nature_style, COLORS
    print("All utils imports OK")


def test_hide_space():
    import numpy as np
    from hide.core.hide_space import HIDESpace

    space = HIDESpace(dim=64)
    for i in range(10):
        emb = np.random.randn(64)
        space.store(emb, {"id": i, "time": float(i)})

    query = np.random.randn(64)
    results = space.retrieve(query, k=3)
    assert len(results) == 3
    assert all(isinstance(r, tuple) and len(r) == 3 for r in results)
    print("HIDESpace basic test OK")


def test_drm_lists():
    from hide.core.emergent import DRM_LISTS
    assert len(DRM_LISTS) == 24
    for name, data in DRM_LISTS.items():
        assert len(data["studied"]) == 15
        assert isinstance(data["lure"], str)
    print("DRM lists test OK (24 lists, 15 words + 1 lure each)")


def test_metrics():
    import numpy as np
    from hide.utils.metrics import participation_ratio, bootstrap_ci

    eigenvalues = np.array([10.0, 5.0, 1.0, 0.1, 0.01])
    d_eff = participation_ratio(eigenvalues)
    assert 1.0 < d_eff < 5.0
    print(f"Participation ratio test OK (d_eff={d_eff:.2f})")

    ci = bootstrap_ci(np.array([0.4, 0.5, 0.6, 0.45, 0.55]))
    # Returns (point_estimate, ci_lower, ci_upper)
    assert ci[1] < 0.5 < ci[2]
    print(f"Bootstrap CI test OK (mean={ci[0]:.3f}, CI=[{ci[1]:.3f}, {ci[2]:.3f}])")


if __name__ == "__main__":
    test_core_imports()
    test_model_imports()
    test_utils_imports()
    test_hide_space()
    test_drm_lists()
    test_metrics()
    print("\nAll tests passed!")
