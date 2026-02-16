# The Geometry of Forgetting

Code and data for: *The Geometry of Forgetting* — showing that high-dimensional embedding spaces, subjected to noise, interference, and temporal degradation, reproduce quantitative signatures of human memory with no phenomenon-specific engineering.

**Authors:** Sambartha Ray Barman, Andrey Starenky, Sophia Bodnar, Nikhil Narasimhan, Ashwin Gopinath

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all experiments (requires 4x A100 GPUs)
python scripts/run_all.py

# Generate all figures from cached results
python scripts/generate_figures.py

# Compile paper
cd paper && pdflatex hide_paper.tex && cd ..
```

## Repository Structure

```
submission/
  paper/                    # Manuscript and figures
    hide_paper.tex          # Main LaTeX source
    hide_paper.pdf          # Compiled PDF
    figures/                # All figures (6 main + 11 extended data)
  hide/                     # Core library
    core/                   # HIDESpace, temporal encoding, interference, consolidation
    models/                 # Embedding models, Qwen adapter
    utils/                  # Metrics, data loading, visualization
  experiments/              # Experiment scripts (one directory per phase)
    phase1/                 # bAbI reasoning (memory retrieval)
    phase2/                 # Temporal memory (Ebbinghaus forgetting)
    phase3/                 # Consolidation and interference
    phase4/                 # Cross-modal binding
    phase5/                 # Emergent phenomena (DRM, spacing, topology)
    spectral/               # Effective dimensionality analysis
  results/                  # All experimental results (JSON + CSV)
    phase1/ ... phase5/     # Per-phase results with 5-seed replication
    spectral/               # Dimensionality and MiniLM interference results
    interference/           # Interference theory experiment results
    spacing_sweep/          # Spacing effect sweep results
    topology/               # Persistent homology results
  configs/                  # YAML hyperparameter configs per phase
  scripts/                  # Reproduction scripts
    run_all.py              # Master experiment runner
    generate_figures.py     # Figure generation from results
    figure_style.py         # Nature-quality matplotlib settings
    gen_fig*.py             # Individual figure generators
  data/                     # DRM word lists + download script
  tests/                    # Sanity tests
  LICENSE                   # Apache 2.0
  requirements.txt          # Pinned dependencies
```

## Hardware Requirements

- **Full reproduction**: 4x NVIDIA A100 (80GB) GPUs
  - GPU 0: Qwen2.5-7B (answer generation)
  - GPU 1: Embedding models (MiniLM / BGE-base / BGE-large)
  - GPU 2: CLIP + parallel experiments
  - GPU 3: FAISS-GPU + batch compute
- **Figures only**: CPU (regenerate from cached results in `results/`)
- **Tests only**: CPU

## Models (All Open-Weight)

| Model | HuggingFace ID | Use | License |
|-------|----------------|-----|---------|
| Qwen2.5-7B | `Qwen/Qwen2.5-7B` | Answer generation | Apache 2.0 |
| MiniLM-L6-v2 | `sentence-transformers/all-MiniLM-L6-v2` | Text embedding | Apache 2.0 |
| BGE-base | `BAAI/bge-base-en-v1.5` | Text embedding | MIT |
| BGE-large | `BAAI/bge-large-en-v1.5` | Text embedding (scale) | MIT |
| CLIP ViT-B/32 | `openai/clip-vit-base-patch32` | Image embedding | MIT |

## Datasets (All Public)

| Dataset | Source | License |
|---------|--------|---------|
| bAbI QA (en-10k) | `Muennighoff/babi` | BSD |
| TempLAMA | `Yova/templama` | MIT |
| CIFAR-100 | `torchvision.datasets.CIFAR100` | BSD |
| COCO Captions | `jxie/coco_captions` | CC BY 4.0 |
| Flickr30k | `lmms-lab/flickr30k` | Research |
| Wikipedia (en) | `wikipedia` (20220301.en, streaming) | CC BY-SA |
| DRM word lists | Roediger & McDermott (1995) | Public domain |

## Reproduction

### Full pipeline (4x A100)

```bash
# Download datasets
bash data/download_data.sh

# Run all experiments across 5 seeds [42, 123, 456, 789, 1024]
python scripts/run_all.py

# Generate figures
python scripts/generate_figures.py
```

### Figures only (CPU)

```bash
# Results are pre-cached in results/
python scripts/generate_figures.py
```

### Run tests

```bash
python -m pytest tests/ -v
```

## Key Results

| Phenomenon | Observed | Human | Notes |
|-----------|----------|-------|-------|
| Forgetting exponent | 0.460 +/- 0.183 | ~0.5 | Interference-driven, not decay |
| DRM false alarm rate | 0.583 | ~0.55 | Unbaked — zero parameter tuning |
| Spacing effect | massed < short < med < long | Same ordering | Boundary-condition-dependent |
| Effective dimensionality | d_eff ~ 16 | d = 100-500 (cortex) | All models concentrate in ~16 dims |
| Tip-of-tongue rate | 3.66% | ~1.5% | Qualitative emergence |

## Seeds and Reproducibility

All experiments use seeds `[42, 123, 456, 789, 1024]` with bootstrap 95% CIs (10,000 resamples). Results are deterministic given the same seed, model weights, and hardware.

## Citation

```bibtex
@article{geometryofforgetting2025,
  title={The Geometry of Forgetting},
  author={Ray Barman, Sambartha and Starenky, Andrey and Bodnar, Sophia and Narasimhan, Nikhil and Gopinath, Ashwin},
  year={2025},
  note={Under review}
}
```

## License

Apache 2.0. See [LICENSE](LICENSE).

## Acknowledgements

Code generation assisted by Claude (Anthropic).
