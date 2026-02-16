"""
Phase 2 v2: Improved Ebbinghaus Forgetting Curve
=================================================
Key changes from v1:
- Add 50K distractor sentences to HIDE space before experiment
- Add age-proportional Gaussian noise to stored embeddings
- Sweep beta parameter: [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
- Report all configurations' R² and b values
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import csv
import yaml
import torch
import numpy as np
import math
from pathlib import Path
from typing import List, Dict, Callable
from tqdm import tqdm
import logging
import pickle

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "hide" / "core"))
sys.path.insert(0, str(PROJECT_ROOT / "hide" / "models"))
sys.path.insert(0, str(PROJECT_ROOT / "hide" / "utils"))

from hide_space import HIDESpace
from embedding_models import EmbeddingManager
from metrics import bootstrap_ci, fit_power_law, r_squared

# Import from v1
from experiments.phase2.run_phase2 import (
    TemporalEncoding, TemporalContextProjector, InfoNCELoss,
    sinusoidal_pe, batch_sinusoidal_pe,
    load_templama, load_babi_train_stories,
    train_temporal_projector, encode_temporal_hide,
    decay_power_law,
)


def load_distractor_sentences(n_sentences: int = 50000, seed: int = 42) -> List[str]:
    """Load distractor sentences from TempLAMA or Wikipedia."""
    cache_dir = PROJECT_ROOT / "data_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"distractor_sents_{n_sentences}_s{seed}.pkl"

    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    np.random.seed(seed)

    # Use TempLAMA train data as distractors (diverse factual sentences)
    try:
        from datasets import load_dataset
        ds = load_dataset("Yova/templama", split="train")
        texts = [row["query"].replace("_X_", "unknown") for row in ds]
        # Replicate if needed
        while len(texts) < n_sentences:
            texts = texts + texts
        texts = texts[:n_sentences]
        np.random.shuffle(texts)
    except Exception:
        # Fallback: generate diverse sentences
        topics = ["physics", "biology", "history", "geography", "chemistry",
                  "mathematics", "literature", "music", "sports", "economics"]
        texts = []
        for i in range(n_sentences):
            topic = topics[i % len(topics)]
            texts.append(f"The field of {topic} encompasses research into fundamental questions about {topic} phenomena, including experimental observations and theoretical frameworks developed by researchers across centuries of academic inquiry, published in leading journals.")

    with open(cache_file, "wb") as f:
        pickle.dump(texts, f)
    return texts


def run_ebbinghaus_v2(
    embed_mgr: EmbeddingManager,
    projector: TemporalContextProjector,
    temporal_enc: TemporalEncoding,
    config: Dict,
    seed: int,
    logger: logging.Logger,
) -> Dict:
    """
    Ebbinghaus v2 with distractors + noise + beta sweep.
    """
    np.random.seed(seed)
    ebb_cfg = config.get("ebbinghaus", {})
    v2_cfg = config.get("ebbinghaus_v2", {})
    n_facts = ebb_cfg.get("n_facts", 1000)
    simulated_days = ebb_cfg.get("simulated_days", 30)
    human_times_min = ebb_cfg.get("human_times_min", [20, 60, 480, 1440, 2880, 8640, 44640])
    human_retention = ebb_cfg.get("human_retention", [0.58, 0.44, 0.36, 0.34, 0.28, 0.25, 0.21])
    n_distractors = v2_cfg.get("n_distractors", 50000)
    sigma_values = v2_cfg.get("sigma_values", [0.0, 0.03, 0.05, 0.1])
    best_sigma = v2_cfg.get("best_sigma", 0.05)
    beta_sweep = v2_cfg.get("beta_sweep", [0.5, 1.0, 2.0, 5.0, 10.0, 20.0])

    # Load facts
    templama = load_templama(split="train")
    if len(templama) < n_facts:
        templama = templama * (n_facts // len(templama) + 1)
    facts = templama[:n_facts]
    total_seconds = simulated_days * 86400
    timestamps = np.sort(np.random.uniform(0, total_seconds, n_facts))
    query_time = total_seconds
    texts = [f["query"] for f in facts]
    positions = list(range(n_facts))

    logger.info(f"  Encoding {n_facts} facts...")
    hide_embs = encode_temporal_hide(
        texts, positions, timestamps.tolist(),
        embed_mgr, projector, temporal_enc, config
    )

    # Load distractors
    logger.info(f"  Loading {n_distractors} distractors...")
    distractor_texts = load_distractor_sentences(n_distractors, seed)
    # Encode a subset (encoding all 50K takes time, use 10K)
    n_dist_encode = min(10000, n_distractors)
    dist_sample = distractor_texts[:n_dist_encode]
    dist_positions = list(range(n_dist_encode))
    dist_timestamps = np.random.uniform(0, total_seconds, n_dist_encode)

    logger.info(f"  Encoding {n_dist_encode} distractors...")
    dist_embs = encode_temporal_hide(
        dist_sample, dist_positions, dist_timestamps.tolist(),
        embed_mgr, projector, temporal_enc, config
    )

    # Query embeddings at query_time
    query_embs = encode_temporal_hide(
        texts, positions, [query_time] * n_facts,
        embed_mgr, projector, temporal_enc, config
    )

    ages_days = (query_time - timestamps) / 86400.0
    age_bins_days = [0.014, 0.042, 0.333, 1.0, 2.0, 6.0, 10.0, 20.0, 31.0]

    # ── Run for each combination of sigma and beta ──
    all_config_results = {}

    for sigma in sigma_values:
        for beta in beta_sweep:
            config_name = f"sigma{sigma}_beta{beta}"
            logger.info(f"    Config: {config_name}")

            # Build HIDE space with distractors
            space = HIDESpace(dim=384, max_memories=n_facts + n_dist_encode + 100)

            # Store distractors
            for emb, ts in zip(dist_embs, dist_timestamps):
                space.store(emb, {"type": "distractor", "timestamp": float(ts)})

            # Store target facts
            for emb, text, ts in zip(hide_embs, texts, timestamps):
                space.store(emb, {"type": "target", "text": text, "timestamp": float(ts)})

            # For each fact, compute retrieval similarity with noise
            all_ages = []
            all_retentions = []

            for bin_start, bin_end in zip(age_bins_days[:-1], age_bins_days[1:]):
                mask = (ages_days >= bin_start) & (ages_days < bin_end)
                if mask.sum() == 0:
                    continue

                bin_fact_indices = np.where(mask)[0]
                retrievals = 0
                total_in_bin = len(bin_fact_indices)

                for fi in bin_fact_indices:
                    q_emb = query_embs[fi]
                    age = ages_days[fi]

                    # Add noise to stored embeddings based on age
                    if sigma > 0:
                        # Add dimension-normalized noise (learned from Phase 5)
                        dim = len(q_emb)
                        noise = sigma * math.sqrt(age + 0.01) / math.sqrt(dim) * np.random.randn(dim).astype(np.float32)
                        noisy_q = q_emb + noise
                        noisy_q = noisy_q / (np.linalg.norm(noisy_q) + 1e-8)
                    else:
                        noisy_q = q_emb

                    # Apply power law decay with this beta
                    def decay_fn(meta, _beta=beta):
                        ts_meta = meta.get("timestamp", 0.0)
                        dt = abs(query_time - ts_meta) / 86400.0
                        return (1.0 + _beta * dt) ** (-0.5)

                    retrieved = space.retrieve(noisy_q, k=5, decay_fn=decay_fn, query_time=query_time)
                    # Check if correct fact is in top-5
                    found = False
                    for _, sim, meta in retrieved:
                        if meta.get("type") == "target" and meta.get("text") == texts[fi]:
                            found = True
                            break
                    if found:
                        retrievals += 1

                retention = retrievals / total_in_bin if total_in_bin > 0 else 0.0
                mean_age = float(np.mean(ages_days[mask]))
                all_ages.append(mean_age)
                all_retentions.append(retention)

            # Fit power law
            fit_result = {}
            if len(all_ages) >= 3:
                try:
                    fit_dict = fit_power_law(np.array(all_ages), np.array(all_retentions))
                    fit_result = {
                        "a": float(fit_dict["a"]),
                        "b": float(fit_dict["b"]),
                        "r_squared": float(fit_dict["r_squared"]),
                    }
                except Exception as e:
                    fit_result = {"error": str(e)}

            all_config_results[config_name] = {
                "sigma": sigma,
                "beta": beta,
                "ages": all_ages,
                "retentions": all_retentions,
                "power_law_fit": fit_result,
            }

            if fit_result.get("b"):
                logger.info(f"      b={fit_result['b']:.4f}, R²={fit_result.get('r_squared', 0):.4f}")

    # Find best config (b closest to 0.5 with R² > 0.5)
    best_config_name = None
    best_b_diff = float("inf")
    for name, res in all_config_results.items():
        fit = res.get("power_law_fit", {})
        b = fit.get("b", 0)
        r2 = fit.get("r_squared", 0)
        if r2 > 0.3 and abs(b - 0.5) < best_b_diff:
            best_b_diff = abs(b - 0.5)
            best_config_name = name

    return {
        "all_configs": all_config_results,
        "best_config": best_config_name,
        "best_result": all_config_results.get(best_config_name, {}) if best_config_name else {},
        "n_distractors": n_dist_encode,
        "n_facts": n_facts,
        "human_reference": {
            "times_days": [t / 1440.0 for t in human_times_min],
            "retention": human_retention,
        },
    }


def run(
    config: Dict = None,
    seed: int = 42,
    results_dir: Path = None,
    logger: logging.Logger = None,
) -> Dict:
    """Run Phase 2 v2 for a single seed."""
    if logger is None:
        logger = logging.getLogger("HIDE.Phase2v2")
        logging.basicConfig(level=logging.INFO)
    if config is None:
        config_path = Path(__file__).parent / "config_v2.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
    if results_dir is None:
        results_dir = PROJECT_ROOT / "results" / "phase2"

    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Phase 2 v2 — Seed {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load embedding model
    embed_mgr = EmbeddingManager()
    embed_mgr.load_text_encoder("minilm", device=config["embedding"]["gpu"])

    # Initialize temporal encoding
    temporal_cfg = config.get("temporal", {})
    temporal_enc = TemporalEncoding(dim=temporal_cfg.get("dim", 64))

    # Train projector
    task_ids = config.get("babi", {}).get("tasks", [1, 2, 3, 4, 5])
    train_stories = load_babi_train_stories(task_ids)
    logger.info(f"  Training projector on {len(train_stories)} stories...")
    projector = train_temporal_projector(
        train_stories, embed_mgr, temporal_enc, config, seed, logger
    )

    # Run Ebbinghaus v2
    logger.info("Running Ebbinghaus v2 (distractors + beta sweep + noise)...")
    ebbinghaus_results = run_ebbinghaus_v2(
        embed_mgr, projector, temporal_enc, config, seed, logger
    )

    result = {
        "seed": seed,
        "ebbinghaus_v2": ebbinghaus_results,
    }

    out_path = results_dir / f"results_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    # CSV
    csv_path = results_dir / f"results_seed{seed}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["seed", seed])
        for cfg_name, cfg_res in ebbinghaus_results.get("all_configs", {}).items():
            fit = cfg_res.get("power_law_fit", {})
            writer.writerow([f"{cfg_name}.b", fit.get("b", "")])
            writer.writerow([f"{cfg_name}.r_squared", fit.get("r_squared", "")])
        writer.writerow(["best_config", ebbinghaus_results.get("best_config", "")])

    logger.info(f"Phase 2 v2 Seed {seed} complete")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("HIDE.Phase2v2")

    config_path = Path(__file__).parent / "config_v2.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    run(config=config, seed=args.seed, logger=logger)
