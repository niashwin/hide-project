#!/usr/bin/env python3
"""
HIDE Project — Effective Dimensionality Analysis (Iteration 5)

Measures effective dimensionality of 3 embedding models:
  1. MiniLM-L6-v2 (384-dim nominal)
  2. BGE-base-en-v1.5 (768-dim nominal)
  3. BGE-large-en-v1.5 (1024-dim nominal)

Computes:
  - Participation ratio: d_eff = (sum(eigenvalues))^2 / sum(eigenvalues^2)
  - d_95: PCA components for 95% variance
  - d_99: PCA components for 99% variance
  - Full eigenvalue spectrum and cumulative variance curve

Uses 10,000 Wikipedia sentences from cached data, across 5 seeds.
Models loaded on separate GPUs for parallel encoding.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import json
import pickle
import time
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ── Configuration ──────────────────────────────────────────────────────────────

SEEDS = [42, 123, 456, 789, 1024]
N_SENTENCES = 10000
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_CACHE_DIR = PROJECT_ROOT / "data_cache"
OUTPUT_DIR = PROJECT_ROOT / "results" / "spectral"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    {
        "name": "MiniLM-L6-v2",
        "hf_id": "sentence-transformers/all-MiniLM-L6-v2",
        "nominal_dim": 384,
        "device": "cuda:1",
    },
    {
        "name": "BGE-base-en-v1.5",
        "hf_id": "BAAI/bge-base-en-v1.5",
        "nominal_dim": 768,
        "device": "cuda:2",
    },
    {
        "name": "BGE-large-en-v1.5",
        "hf_id": "BAAI/bge-large-en-v1.5",
        "nominal_dim": 1024,
        "device": "cuda:3",
    },
]

BATCH_SIZE = 512


# ── Helper functions ───────────────────────────────────────────────────────────

def load_sentences(seed):
    """Load cached Wikipedia sentences for a given seed."""
    path = DATA_CACHE_DIR / f"wiki_sentences_100000_s{seed}.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)
    # Take first N_SENTENCES, extract text
    sentences = [item["text"] for item in data[:N_SENTENCES]]
    print(f"  Loaded {len(sentences)} sentences from seed {seed}")
    return sentences


def compute_dimensionality(embeddings):
    """
    Compute effective dimensionality metrics from an embedding matrix.

    Args:
        embeddings: (N, D) array of embeddings

    Returns:
        dict with d_eff, d_95, d_99, eigenvalues, cumulative_variance
    """
    # Center the data
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)

    # Compute covariance matrix
    cov = np.cov(centered, rowvar=False)  # (D, D)

    # Eigenvalue decomposition (symmetric matrix)
    eigenvalues = np.linalg.eigvalsh(cov)  # ascending order
    eigenvalues = eigenvalues[::-1]  # descending order

    # Clip small negative eigenvalues from numerical noise
    eigenvalues = np.maximum(eigenvalues, 0.0)

    # Participation ratio
    sum_eig = eigenvalues.sum()
    sum_eig_sq = (eigenvalues ** 2).sum()
    d_eff = float((sum_eig ** 2) / sum_eig_sq) if sum_eig_sq > 0 else 0.0

    # Cumulative variance explained
    if sum_eig > 0:
        cumulative_variance = np.cumsum(eigenvalues) / sum_eig
    else:
        cumulative_variance = np.zeros_like(eigenvalues)

    # d_95 and d_99
    d_95 = int(np.searchsorted(cumulative_variance, 0.95) + 1)
    d_99 = int(np.searchsorted(cumulative_variance, 0.99) + 1)

    return {
        "d_eff": d_eff,
        "d_95": d_95,
        "d_99": d_99,
        "eigenvalues": eigenvalues.tolist(),
        "cumulative_variance": cumulative_variance.tolist(),
    }


# ── Main execution ─────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("HIDE Dimensionality Analysis — Iteration 5")
    print("=" * 70)

    all_results = {}

    for model_cfg in MODELS:
        model_name = model_cfg["name"]
        hf_id = model_cfg["hf_id"]
        device = model_cfg["device"]
        nominal_dim = model_cfg["nominal_dim"]

        print(f"\n{'─' * 60}")
        print(f"Model: {model_name} ({hf_id})")
        print(f"Device: {device} | Nominal dim: {nominal_dim}")
        print(f"{'─' * 60}")

        # Load model
        t0 = time.time()
        model = SentenceTransformer(hf_id, device=device)
        print(f"  Model loaded in {time.time() - t0:.1f}s")

        seed_results = []

        for seed in SEEDS:
            print(f"\n  Seed {seed}:")
            sentences = load_sentences(seed)

            # Encode
            t0 = time.time()
            embeddings = model.encode(
                sentences,
                batch_size=BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,  # raw embeddings for PCA
            )
            encode_time = time.time() - t0
            print(f"    Encoded {embeddings.shape[0]} sentences -> {embeddings.shape[1]}-dim in {encode_time:.1f}s")

            # Compute dimensionality
            t0 = time.time()
            result = compute_dimensionality(embeddings)
            analysis_time = time.time() - t0
            print(f"    d_eff={result['d_eff']:.1f}, d_95={result['d_95']}, d_99={result['d_99']} (computed in {analysis_time:.1f}s)")

            seed_results.append({
                "seed": seed,
                "d_eff": result["d_eff"],
                "d_95": result["d_95"],
                "d_99": result["d_99"],
                "eigenvalues": result["eigenvalues"],
                "cumulative_variance": result["cumulative_variance"],
                "encode_time_s": encode_time,
            })

        # Aggregate across seeds
        d_effs = [r["d_eff"] for r in seed_results]
        d_95s = [r["d_95"] for r in seed_results]
        d_99s = [r["d_99"] for r in seed_results]

        all_results[model_name] = {
            "hf_id": hf_id,
            "nominal_dim": nominal_dim,
            "n_sentences": N_SENTENCES,
            "seeds": SEEDS,
            "per_seed": seed_results,
            "summary": {
                "d_eff_mean": float(np.mean(d_effs)),
                "d_eff_std": float(np.std(d_effs)),
                "d_95_mean": float(np.mean(d_95s)),
                "d_95_std": float(np.std(d_95s)),
                "d_99_mean": float(np.mean(d_99s)),
                "d_99_std": float(np.std(d_99s)),
            },
        }

        # Free GPU memory
        del model
        import torch
        torch.cuda.empty_cache()

    # ── Save results ──────────────────────────────────────────────────────────

    output_path = OUTPUT_DIR / "dimensionality_analysis.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # ── Summary table ─────────────────────────────────────────────────────────

    print("\n" + "=" * 90)
    print("SUMMARY: Effective Dimensionality Analysis")
    print("=" * 90)
    print(f"{'Model':<22} {'Nominal':>8} {'d_eff':>14} {'d_95':>14} {'d_99':>14} {'Ratio':>10}")
    print(f"{'':22} {'dim':>8} {'(mean+/-std)':>14} {'(mean+/-std)':>14} {'(mean+/-std)':>14} {'d_eff/nom':>10}")
    print("-" * 90)

    for model_cfg in MODELS:
        name = model_cfg["name"]
        nom = model_cfg["nominal_dim"]
        s = all_results[name]["summary"]
        ratio = s["d_eff_mean"] / nom
        print(
            f"{name:<22} {nom:>8d} "
            f"{s['d_eff_mean']:>7.1f}+/-{s['d_eff_std']:<5.1f} "
            f"{s['d_95_mean']:>7.1f}+/-{s['d_95_std']:<5.1f} "
            f"{s['d_99_mean']:>7.1f}+/-{s['d_99_std']:<5.1f} "
            f"{ratio:>9.3f}"
        )

    print("-" * 90)
    print(f"N = {N_SENTENCES} sentences | Seeds: {SEEDS}")
    print(f"Participation ratio: d_eff = (sum lambda)^2 / sum(lambda^2)")
    print("=" * 90)


if __name__ == "__main__":
    main()
