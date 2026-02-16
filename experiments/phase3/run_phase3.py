"""
Phase 3 v2: Gentle Consolidation + Selective (Age-Based) Consolidation
======================================================================
Key changes from v1:
- Conservative consolidation: min_cluster_size=10, merge_threshold=0.05, sigma=2.0
- Selective consolidation: only consolidate memories older than N steps
- New conditions: gentle_consolidation, full_hide_v2
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
from pathlib import Path
from typing import Dict, List
from collections import Counter
from tqdm import tqdm
import logging

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "hide" / "core"))
sys.path.insert(0, str(PROJECT_ROOT / "hide" / "models"))
sys.path.insert(0, str(PROJECT_ROOT / "hide" / "utils"))

from hide_space import HIDESpace

# Import shared functions from v1
from experiments.phase3.run_phase3 import (
    load_clip, encode_images_clip, load_cifar100,
    get_task_classes, classify_topk, replay, compute_interference,
)


def consolidate_gentle(space: HIDESpace, config: Dict, logger: logging.Logger,
                       total_stored: int = 0) -> Dict:
    """
    Gentle consolidation with:
    - Higher min_cluster_size (10 vs 5)
    - Lower merge threshold (0.05 vs 0.15)
    - Higher outlier sigma (2.0 vs 1.0)
    - Selective: only consolidate memories older than min_age steps
    """
    cons_cfg = config.get("consolidation", {})
    min_cluster = cons_cfg.get("hdbscan_min_cluster_size", 10)
    merge_thresh = cons_cfg.get("merge_centroid_threshold", 0.05)
    sigma = cons_cfg.get("outlier_sigma", 2.0)
    min_age = cons_cfg.get("consolidation_min_age", 500)

    if space.count < min_cluster * 2:
        return {"memories_before": space.count, "memories_after": space.count, "compression_ratio": 1.0}

    memories_before = space.count
    embeddings = space.get_all_embeddings()

    # Selective: only consider memories older than min_age
    old_mask = np.zeros(space.count, dtype=bool)
    for i in range(space.count):
        store_step = space.metadata[i].get("store_step", 0)
        if total_stored - store_step > min_age:
            old_mask[i] = True

    old_indices = np.where(old_mask)[0]
    if len(old_indices) < min_cluster * 2:
        return {"memories_before": memories_before, "memories_after": space.count,
                "compression_ratio": 1.0, "skipped": "not enough old memories"}

    old_embeddings = embeddings[old_indices]

    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster, metric="euclidean")
        labels = clusterer.fit_predict(old_embeddings)
    except ImportError:
        from sklearn.cluster import KMeans
        n_clusters = max(2, len(old_indices) // 20)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
        labels = kmeans.fit_predict(old_embeddings)

    unique_labels = set(labels)
    unique_labels.discard(-1)

    indices_to_remove = []

    for label in unique_labels:
        cluster_mask = labels == label
        cluster_local_indices = np.where(cluster_mask)[0]
        cluster_global_indices = old_indices[cluster_local_indices]

        if len(cluster_global_indices) < 2:
            continue

        cluster_embs = embeddings[cluster_global_indices]
        centroid = cluster_embs.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

        dists = np.linalg.norm(cluster_embs - centroid, axis=1)
        mean_dist = dists.mean()
        std_dist = dists.std() + 1e-8

        close_mask = dists <= (mean_dist + sigma * std_dist)
        close_global = cluster_global_indices[close_mask]

        if len(close_global) > 1:
            close_embs = embeddings[close_global]
            merged = close_embs.mean(axis=0)
            merged = merged / (np.linalg.norm(merged) + 1e-8)
            space.replace(close_global[0], merged)
            merged_meta = space.metadata[close_global[0]].copy()
            merged_meta["merged_count"] = len(close_global)
            space.metadata[close_global[0]] = merged_meta
            indices_to_remove.extend(close_global[1:].tolist())

    # Between-cluster merge (conservative threshold)
    if len(unique_labels) > 1:
        centroids = {}
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_local_indices = np.where(cluster_mask)[0]
            cluster_global_indices = old_indices[cluster_local_indices]
            c = embeddings[cluster_global_indices].mean(axis=0)
            centroids[label] = c / (np.linalg.norm(c) + 1e-8)

        merged_pairs = set()
        label_list = list(centroids.keys())
        for i, l1 in enumerate(label_list):
            for l2 in label_list[i+1:]:
                cos_dist = 1.0 - np.dot(centroids[l1], centroids[l2])
                if cos_dist < merge_thresh and l1 not in merged_pairs and l2 not in merged_pairs:
                    l2_local = np.where(labels == l2)[0]
                    l2_global = old_indices[l2_local]
                    indices_to_remove.extend(l2_global.tolist())
                    merged_pairs.add(l2)

    indices_to_remove = list(set(indices_to_remove))
    if indices_to_remove:
        space.remove_indices(sorted(indices_to_remove))

    memories_after = space.count
    compression = memories_after / memories_before if memories_before > 0 else 1.0

    return {
        "memories_before": memories_before,
        "memories_after": memories_after,
        "compression_ratio": float(compression),
        "removed": len(indices_to_remove),
        "old_candidates": len(old_indices),
    }


def run_continual_learning_v2(
    train_ds, test_ds, class_names, model, processor,
    config: Dict, seed: int, logger: logging.Logger,
    condition: str = "gentle_consolidation",
) -> Dict:
    """Run CL with v2 gentle consolidation or v2 full_hide."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    cl_cfg = config.get("continual_learning", {})
    n_tasks = cl_cfg.get("n_tasks", 10)
    classes_per_task = cl_cfg.get("classes_per_task", 10)
    top_k = cl_cfg.get("top_k", 5)
    cons_cfg = config.get("consolidation", {})
    trigger_interval = cons_cfg.get("trigger_interval", 1000)
    replay_count = cons_cfg.get("replay_count", 100)

    device = config.get("clip", {}).get("gpu", "cuda:2")
    task_classes = get_task_classes(n_tasks, classes_per_task)

    space = HIDESpace(dim=512, max_memories=60000)
    max_capacity = 20000

    alpha_classes = sorted(train_ds.classes)
    orig_to_alpha = {}
    for orig_idx, name in enumerate(train_ds.classes):
        orig_to_alpha[orig_idx] = alpha_classes.index(name)

    train_by_class = {c: [] for c in range(100)}
    for idx in range(len(train_ds)):
        _, orig_label = train_ds[idx]
        alpha_label = orig_to_alpha[orig_label]
        train_by_class[alpha_label].append(idx)

    test_by_class = {c: [] for c in range(100)}
    for idx in range(len(test_ds)):
        _, orig_label = test_ds[idx]
        alpha_label = orig_to_alpha[orig_label]
        test_by_class[alpha_label].append(idx)

    task_accuracies = {}
    consolidation_log = []
    memory_sizes = []
    total_stored = 0

    for task_id in range(n_tasks):
        task_class_ids = task_classes[task_id]
        logger.info(f"    [{condition}] Task {task_id}: classes {task_class_ids[0]}-{task_class_ids[-1]}")

        task_train_indices = []
        for c in task_class_ids:
            task_train_indices.extend(train_by_class[c])

        if len(task_train_indices) > 1000:
            task_train_indices = list(np.random.choice(task_train_indices, 1000, replace=False))

        from PIL import Image
        task_images = []
        task_labels = []
        for idx in task_train_indices:
            img, orig_label = train_ds[idx]
            if isinstance(img, torch.Tensor):
                img_pil = Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
            else:
                img_pil = img
            task_images.append(img_pil)
            task_labels.append(orig_to_alpha[orig_label])

        task_embs = encode_images_clip(task_images, model, processor, device, batch_size=256)

        for emb, label in zip(task_embs, task_labels):
            space.store(emb, {"label": label, "task_id": task_id, "store_step": total_stored})
            total_stored += 1

            if total_stored % trigger_interval == 0:
                if condition in ("gentle_consolidation", "full_hide_v2"):
                    cons_result = consolidate_gentle(space, config, logger, total_stored)
                    consolidation_log.append(cons_result)

                if condition in ("full_hide_v2",):
                    replay(space, model, processor, device, n_replay=replay_count, logger=logger)

                # Fallback conditions from v1
                if condition == "consolidation_only":
                    from experiments.phase3.run_phase3 import consolidate
                    cons_result = consolidate(space, config, logger)
                    consolidation_log.append(cons_result)

                if condition in ("replay_only", "full_hide"):
                    replay(space, model, processor, device, n_replay=replay_count, logger=logger)

                if condition == "full_hide":
                    from experiments.phase3.run_phase3 import consolidate
                    consolidate(space, config, logger)

                if condition == "naive_pruning" and space.count > max_capacity:
                    n_remove = space.count - max_capacity
                    space.remove_indices(list(range(n_remove)))

                if condition == "experience_replay" and space.count > max_capacity:
                    keep_idx = np.random.choice(space.count, max_capacity, replace=False)
                    remove_idx = [i for i in range(space.count) if i not in set(keep_idx)]
                    if remove_idx:
                        space.remove_indices(sorted(remove_idx))

        memory_sizes.append(space.count)

        # Test on all tasks seen so far
        task_accuracies[task_id] = {}
        for test_task_id in range(task_id + 1):
            test_class_ids = task_classes[test_task_id]
            test_indices = []
            for c in test_class_ids:
                test_indices.extend(test_by_class[c][:50])

            correct = 0
            total = 0
            from PIL import Image as PILImage
            for idx in test_indices:
                img, orig_label = test_ds[idx]
                alpha_label = orig_to_alpha[orig_label]
                if isinstance(img, torch.Tensor):
                    img_pil = PILImage.fromarray((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
                else:
                    img_pil = img
                q_emb = encode_images_clip([img_pil], model, processor, device, batch_size=1)[0]
                pred = classify_topk(space, q_emb, k=top_k)
                if pred == alpha_label:
                    correct += 1
                total += 1

            acc = correct / total if total > 0 else 0.0
            task_accuracies[task_id][test_task_id] = acc

        seen_accs = [task_accuracies[task_id][t] for t in range(task_id + 1)]
        logger.info(f"      Mean acc: {np.mean(seen_accs):.4f}, memory: {space.count}")

    backward_transfer = {}
    for test_task_id in range(n_tasks):
        if test_task_id in task_accuracies.get(test_task_id, {}):
            initial_acc = task_accuracies[test_task_id][test_task_id]
            if n_tasks - 1 in task_accuracies and test_task_id in task_accuracies[n_tasks - 1]:
                final_acc = task_accuracies[n_tasks - 1][test_task_id]
                backward_transfer[test_task_id] = float(final_acc - initial_acc)

    compression = space.count / total_stored if total_stored > 0 else 1.0

    return {
        "condition": condition,
        "task_accuracies": {str(k): {str(k2): v2 for k2, v2 in v.items()} for k, v in task_accuracies.items()},
        "backward_transfer": {str(k): v for k, v in backward_transfer.items()},
        "mean_backward_transfer": float(np.mean(list(backward_transfer.values()))) if backward_transfer else 0.0,
        "memory_sizes": memory_sizes,
        "final_memory_count": space.count,
        "total_stored": total_stored,
        "compression_ratio": float(compression),
        "consolidation_log": consolidation_log[:10],
    }


def run(
    config: Dict = None,
    seed: int = 42,
    results_dir: Path = None,
    logger: logging.Logger = None,
) -> Dict:
    """Run Phase 3 v2 for a single seed."""
    if logger is None:
        logger = logging.getLogger("HIDE.Phase3v2")
        logging.basicConfig(level=logging.INFO)
    if config is None:
        config_path = Path(__file__).parent / "config_v2.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
    if results_dir is None:
        results_dir = PROJECT_ROOT / "results" / "phase3"

    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Phase 3 v2 — Seed {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)

    clip_device = config.get("clip", {}).get("gpu", "cuda:2")
    logger.info(f"Loading CLIP on {clip_device}...")
    clip_model, clip_processor = load_clip(clip_device)

    logger.info("Loading CIFAR-100...")
    train_ds, test_ds, class_names, name_to_alpha = load_cifar100()

    # Run v2 conditions: gentle_consolidation and full_hide_v2
    # Also re-run no_consolidation as reference
    conditions = ["no_consolidation", "gentle_consolidation", "full_hide_v2"]
    condition_results = {}

    for condition in conditions:
        logger.info(f"  Running condition: {condition}")
        result = run_continual_learning_v2(
            train_ds, test_ds, class_names,
            clip_model, clip_processor,
            config, seed, logger,
            condition=condition,
        )
        condition_results[condition] = result
        logger.info(f"    {condition}: compression={result['compression_ratio']:.3f}, "
                     f"mean_BT={result['mean_backward_transfer']:.4f}, "
                     f"final_mem={result['final_memory_count']}")

    interference = compute_interference(condition_results, logger)

    del clip_model, clip_processor
    torch.cuda.empty_cache()

    result = {
        "seed": seed,
        "conditions": condition_results,
        "interference": interference,
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
        for cond, r in condition_results.items():
            writer.writerow([f"{cond}.compression_ratio", r["compression_ratio"]])
            writer.writerow([f"{cond}.mean_backward_transfer", r["mean_backward_transfer"]])
            writer.writerow([f"{cond}.final_memory_count", r["final_memory_count"]])

    logger.info(f"Phase 3 v2 Seed {seed} complete")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("HIDE.Phase3v2")

    config_path = Path(__file__).parent / "config_v2.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    run(config=config, seed=args.seed, logger=logger)
