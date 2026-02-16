"""
Phase 1: "Memory in a Box" — bAbI Reasoning with HIDE
======================================================
HIDE contextual embeddings outperform naive retrieval on memory-dependent QA.
Uses Muennighoff/babi (real bAbI en-10k data, public benchmark).
MiniLM-L6-v2 embeddings on cuda:1, Qwen2.5-7B generation on cuda:0.

GPU allocation:
  cuda:0 — Qwen2.5-7B (generation)
  cuda:1 — MiniLM-L6-v2 (embeddings) + ContextProjector training
  cuda:2 — available for parallel seeds
  cuda:3 — available for parallel seeds
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import time
import yaml
import math
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import logging

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "hide" / "core"))
sys.path.insert(0, str(PROJECT_ROOT / "hide" / "models"))
sys.path.insert(0, str(PROJECT_ROOT / "hide" / "utils"))

from hide_space import HIDESpace
from embedding_models import EmbeddingManager
from qwen_adapter import QwenGenerator
from metrics import accuracy, bootstrap_ci


# ─────────────────────────────────────────────────────
# Sinusoidal Positional Encoding
# ─────────────────────────────────────────────────────

def sinusoidal_pe(position: int, dim: int = 384, max_len: int = 500) -> np.ndarray:
    """Generate sinusoidal positional encoding for a single position."""
    pe = np.zeros(dim, dtype=np.float32)
    for i in range(0, dim, 2):
        angle = position / (max_len ** (i / dim))
        pe[i] = math.sin(angle)
        if i + 1 < dim:
            pe[i + 1] = math.cos(angle)
    return pe


def batch_sinusoidal_pe(positions: List[int], dim: int = 384, max_len: int = 500) -> np.ndarray:
    """Generate sinusoidal positional encodings for a batch."""
    pe = np.zeros((len(positions), dim), dtype=np.float32)
    for idx, pos in enumerate(positions):
        pe[idx] = sinusoidal_pe(pos, dim, max_len)
    return pe


# ─────────────────────────────────────────────────────
# Context Projector (Linear 768 → 384 + LayerNorm)
# ─────────────────────────────────────────────────────

class ContextProjector(nn.Module):
    """Projects concatenated [content; context] embeddings into HIDE space."""

    def __init__(self, input_dim: int = 768, output_dim: int = 384):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


# ─────────────────────────────────────────────────────
# InfoNCE Loss
# ─────────────────────────────────────────────────────

class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchors: torch.Tensor, positives: torch.Tensor) -> torch.Tensor:
        anchors = nn.functional.normalize(anchors, dim=-1)
        positives = nn.functional.normalize(positives, dim=-1)
        logits = anchors @ positives.T / self.temperature
        labels = torch.arange(len(anchors), device=anchors.device)
        return nn.functional.cross_entropy(logits, labels)


# ─────────────────────────────────────────────────────
# bAbI Data Loading (from Muennighoff/babi)
# ─────────────────────────────────────────────────────

def load_babi_task(task_id: int, split: str = "train") -> List[Dict]:
    """
    Load bAbI task from Muennighoff/babi dataset.
    Returns list of question dicts with:
      - passage: the story text
      - sentences: list of story sentences
      - question: question text
      - answer: gold answer
    """
    # HF_TOKEN should be set via environment variable if needed
    from datasets import load_dataset

    ds = load_dataset("Muennighoff/babi", split=split)
    # Filter to task
    task_data = [item for item in ds if item["task"] == task_id]

    questions = []
    for item in task_data:
        passage = item["passage"].strip()
        sentences = [s.strip() for s in passage.split("\n") if s.strip()]
        questions.append({
            "passage": passage,
            "sentences": sentences,
            "question": item["question"],
            "answer": item["answer"],
        })

    return questions


def load_babi_train_stories(task_ids: List[int]) -> List[Dict]:
    """Load training data, group by unique passages to create 'stories'."""
    all_stories = {}
    for task_id in task_ids:
        items = load_babi_task(task_id, split="train")
        for item in items:
            key = item["passage"]
            if key not in all_stories:
                all_stories[key] = {
                    "sentences": item["sentences"],
                    "passage": item["passage"],
                    "questions": [],
                }
            all_stories[key]["questions"].append({
                "question": item["question"],
                "answer": item["answer"],
            })
    return list(all_stories.values())


# ─────────────────────────────────────────────────────
# Training the ContextProjector
# ─────────────────────────────────────────────────────

def train_context_projector(
    stories: List[Dict],
    embed_mgr: EmbeddingManager,
    config: Dict,
    seed: int,
    logger: logging.Logger,
) -> ContextProjector:
    """Train ContextProjector on bAbI train stories using InfoNCE."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = config.get("context_projector", {})
    device = cfg.get("gpu", "cuda:1")
    lr = cfg.get("lr", 1e-3)
    weight_decay = cfg.get("weight_decay", 1e-4)
    epochs = cfg.get("epochs", 10)
    batch_size = cfg.get("batch_size", 256)
    temperature = cfg.get("temperature", 0.07)
    val_split = cfg.get("validation_split", 0.1)
    pe_dim = config.get("positional_encoding", {}).get("dim", 384)

    # Collect all (sentence, position, story_id)
    all_items = []
    for story_id, story in enumerate(stories):
        for pos, sent in enumerate(story["sentences"]):
            all_items.append((sent, pos, story_id))

    texts = [item[0] for item in all_items]
    logger.info(f"  Encoding {len(texts)} sentences for projector training...")
    content_embs = embed_mgr.encode_text(texts, batch_size=512, show_progress=True)

    positions = [item[1] for item in all_items]
    pos_embs = batch_sinusoidal_pe(positions, dim=pe_dim)

    combined = np.concatenate([content_embs, pos_embs], axis=1)
    story_ids = np.array([item[2] for item in all_items])

    n_val = max(1, int(len(combined) * val_split))
    indices = np.random.permutation(len(combined))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_combined = torch.tensor(combined[train_idx], device=device)
    train_stories = story_ids[train_idx]
    val_combined = torch.tensor(combined[val_idx], device=device)
    val_stories = story_ids[val_idx]

    projector = ContextProjector(
        input_dim=cfg.get("input_dim", 768),
        output_dim=cfg.get("output_dim", 384),
    ).to(device)

    optimizer = torch.optim.AdamW(
        projector.parameters(), lr=lr, weight_decay=weight_decay
    )
    criterion = InfoNCELoss(temperature=temperature)

    best_val_loss = float("inf")
    best_state = None

    # Build story-to-indices mapping for fast positive sampling
    story_to_train_idx = {}
    for i, sid in enumerate(train_stories):
        story_to_train_idx.setdefault(int(sid), []).append(i)

    for epoch in range(epochs):
        projector.train()
        epoch_losses = []

        perm = np.random.permutation(len(train_idx))
        for start in range(0, len(perm) - batch_size + 1, batch_size):
            batch_idx = perm[start:start + batch_size]
            anchor_combined = train_combined[batch_idx]

            # Positive: same story, different sentence
            pos_indices = []
            for idx in batch_idx:
                sid = int(train_stories[idx])
                candidates = story_to_train_idx.get(sid, [idx])
                candidates = [c for c in candidates if c != idx]
                if candidates:
                    pos_indices.append(np.random.choice(candidates))
                else:
                    pos_indices.append(idx)

            pos_combined = train_combined[pos_indices]

            anchor_proj = projector(anchor_combined)
            pos_proj = projector(pos_combined)

            loss = criterion(anchor_proj, pos_proj)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        # Validation loss
        projector.eval()
        with torch.no_grad():
            # Compute InfoNCE on validation set (sample a batch)
            n_val_batch = min(batch_size, len(val_idx))
            val_perm = np.random.permutation(len(val_idx))[:n_val_batch]
            val_batch = val_combined[val_perm]

            val_pos_indices = []
            story_to_val_idx = {}
            for i, sid in enumerate(val_stories):
                story_to_val_idx.setdefault(int(sid), []).append(i)

            for idx in val_perm:
                sid = int(val_stories[idx])
                candidates = story_to_val_idx.get(sid, [idx])
                candidates = [c for c in candidates if c != idx]
                if candidates:
                    val_pos_indices.append(np.random.choice(candidates))
                else:
                    val_pos_indices.append(idx)

            val_pos = val_combined[val_pos_indices]
            val_anchor_proj = projector(val_batch)
            val_pos_proj = projector(val_pos)
            val_loss = criterion(val_anchor_proj, val_pos_proj).item()

        avg_train = np.mean(epoch_losses) if epoch_losses else 0
        logger.info(f"    Epoch {epoch+1}/{epochs}: train_loss={avg_train:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in projector.state_dict().items()}

    if best_state:
        projector.load_state_dict(best_state)
    projector.to(device)
    projector.eval()
    return projector


# ─────────────────────────────────────────────────────
# HIDE Encoding
# ─────────────────────────────────────────────────────

def encode_hide(
    texts: List[str],
    positions: List[int],
    embed_mgr: EmbeddingManager,
    projector: ContextProjector,
    config: Dict,
) -> np.ndarray:
    """Encode texts into HIDE space."""
    pe_dim = config.get("positional_encoding", {}).get("dim", 384)
    device = config.get("context_projector", {}).get("gpu", "cuda:1")

    content_embs = embed_mgr.encode_text(texts, batch_size=512)
    pos_embs = batch_sinusoidal_pe(positions, dim=pe_dim)
    combined = np.concatenate([content_embs, pos_embs], axis=1)

    with torch.no_grad():
        combined_t = torch.tensor(combined, device=device)
        projected = projector(combined_t)
        return projected.cpu().numpy()


# ─────────────────────────────────────────────────────
# Evaluation on a single bAbI task
# ─────────────────────────────────────────────────────

def evaluate_task(
    task_id: int,
    test_questions: List[Dict],
    embed_mgr: EmbeddingManager,
    projector: ContextProjector,
    qwen: QwenGenerator,
    config: Dict,
    seed: int,
    logger: logging.Logger,
    max_samples: int = 1000,
) -> Dict:
    """Evaluate HIDE + baselines on a single bAbI task."""
    np.random.seed(seed)
    retrieval_k_map = config.get("babi", {}).get("retrieval_k", {})
    retrieval_k = retrieval_k_map.get(task_id, retrieval_k_map.get(str(task_id), 5))

    questions = test_questions[:max_samples]
    logger.info(f"  Task {task_id}: {len(questions)} questions, k={retrieval_k}")

    results = {
        "task_id": task_id,
        "n_questions": len(questions),
        "retrieval_k": retrieval_k,
    }

    # ── HIDE retrieval ──
    hide_preds, hide_golds = [], []
    hide_latencies = {"encode": [], "retrieve": [], "generate": []}

    for q in tqdm(questions, desc=f"  HIDE T{task_id}", leave=False):
        sents = q["sentences"]
        question = q["question"]
        gold = q["answer"]

        space = HIDESpace(dim=384, max_memories=len(sents) + 10)
        positions = list(range(len(sents)))

        t0 = time.time()
        hide_embs = encode_hide(sents, positions, embed_mgr, projector, config)
        t_encode = time.time() - t0

        for i, (emb, sent) in enumerate(zip(hide_embs, sents)):
            space.store(emb, {"text": sent, "position": i})

        q_emb = encode_hide([question], [len(sents)], embed_mgr, projector, config)

        t0 = time.time()
        retrieved = space.retrieve(q_emb[0], k=retrieval_k)
        t_retrieve = time.time() - t0

        retrieved_texts = [r[2]["text"] for r in retrieved]

        t0 = time.time()
        pred = qwen.generate_answer(question, retrieved_texts)
        t_generate = time.time() - t0

        hide_preds.append(pred)
        hide_golds.append(gold)
        hide_latencies["encode"].append(t_encode)
        hide_latencies["retrieve"].append(t_retrieve)
        hide_latencies["generate"].append(t_generate)

    hide_acc = accuracy(hide_preds, hide_golds)
    results["hide_accuracy"] = hide_acc
    results["hide_latency_encode_ms"] = float(np.mean(hide_latencies["encode"]) * 1000)
    results["hide_latency_retrieve_ms"] = float(np.mean(hide_latencies["retrieve"]) * 1000)
    results["hide_latency_generate_ms"] = float(np.mean(hide_latencies["generate"]) * 1000)
    logger.info(f"    HIDE accuracy: {hide_acc:.4f}")

    # ── Baseline 1: No memory ──
    no_mem_preds = []
    for q in tqdm(questions, desc=f"  NoMem T{task_id}", leave=False):
        pred = qwen.generate_answer_no_context(q["question"])
        no_mem_preds.append(pred)
    no_mem_acc = accuracy(no_mem_preds, hide_golds)
    results["no_memory_accuracy"] = no_mem_acc
    logger.info(f"    No-memory accuracy: {no_mem_acc:.4f}")

    # ── Baseline 2: Full context ──
    full_ctx_preds = []
    for q in tqdm(questions, desc=f"  FullCtx T{task_id}", leave=False):
        pred = qwen.generate_answer_full_context(q["question"], q["passage"])
        full_ctx_preds.append(pred)
    full_ctx_acc = accuracy(full_ctx_preds, hide_golds)
    results["full_context_accuracy"] = full_ctx_acc
    logger.info(f"    Full-context accuracy: {full_ctx_acc:.4f}")

    # ── Baseline 3: Random retrieval ──
    random_preds = []
    for q in tqdm(questions, desc=f"  Rand T{task_id}", leave=False):
        sents = q["sentences"]
        k = min(retrieval_k, len(sents))
        idx = np.random.choice(len(sents), k, replace=False)
        random_texts = [sents[i] for i in idx]
        pred = qwen.generate_answer(q["question"], random_texts)
        random_preds.append(pred)
    random_acc = accuracy(random_preds, hide_golds)
    results["random_retrieval_accuracy"] = random_acc
    logger.info(f"    Random-retrieval accuracy: {random_acc:.4f}")

    # ── Baseline 4: Vanilla RAG (flat cosine, no context encoding) ──
    vanilla_preds = []
    for q in tqdm(questions, desc=f"  Vanilla T{task_id}", leave=False):
        sents = q["sentences"]
        content_embs = embed_mgr.encode_text(sents, batch_size=512)
        q_emb = embed_mgr.encode_text([q["question"]], batch_size=1)
        content_norm = content_embs / (np.linalg.norm(content_embs, axis=1, keepdims=True) + 1e-8)
        q_norm = q_emb[0] / (np.linalg.norm(q_emb[0]) + 1e-8)
        sims = content_norm @ q_norm
        k = min(retrieval_k, len(sents))
        top_idx = np.argsort(sims)[-k:][::-1]
        retrieved_texts = [sents[i] for i in top_idx]
        pred = qwen.generate_answer(q["question"], retrieved_texts)
        vanilla_preds.append(pred)
    vanilla_acc = accuracy(vanilla_preds, hide_golds)
    results["vanilla_rag_accuracy"] = vanilla_acc
    logger.info(f"    Vanilla-RAG accuracy: {vanilla_acc:.4f}")

    # Per-question correctness for bootstrap CI
    results["hide_correct"] = [
        1 if p.strip().lower() == g.strip().lower() else 0
        for p, g in zip(hide_preds, hide_golds)
    ]

    return results


# ─────────────────────────────────────────────────────
# Memory Scaling Experiment
# ─────────────────────────────────────────────────────

def run_memory_scaling(
    train_questions: List[Dict],
    embed_mgr: EmbeddingManager,
    projector: ContextProjector,
    config: Dict,
    seed: int,
    logger: logging.Logger,
) -> Dict:
    """Test retrieval precision as stored memory count increases."""
    np.random.seed(seed)
    scaling_cfg = config.get("memory_scaling", {})
    sizes = scaling_cfg.get("sizes", [10, 50, 100, 200, 500, 1000, 2000, 5000])
    n_queries = scaling_cfg.get("n_queries", 100)

    # Collect all unique sentences
    all_sents = []
    for q in train_questions:
        for s in q["sentences"]:
            if s not in all_sents:
                all_sents.append(s)

    # Use a subset of questions as test queries
    test_qs = train_questions[:n_queries]

    results = {}
    for N in sizes:
        if N > len(all_sents):
            logger.info(f"    Scaling N={N}: skipped (only {len(all_sents)} unique sents)")
            continue

        selected_idx = np.random.choice(len(all_sents), N, replace=False)
        memory_sents = [all_sents[i] for i in selected_idx]
        memory_set = set(memory_sents)
        positions = list(range(len(memory_sents)))

        space = HIDESpace(dim=384, max_memories=N + 100)
        hide_embs = encode_hide(memory_sents, positions, embed_mgr, projector, config)
        for i, (emb, sent) in enumerate(zip(hide_embs, memory_sents)):
            space.store(emb, {"text": sent, "position": i})

        p_at_5_scores = []
        for q in test_qs:
            q_emb = encode_hide([q["question"]], [N], embed_mgr, projector, config)
            retrieved = space.retrieve(q_emb[0], k=5)
            retrieved_texts = {r[2]["text"] for r in retrieved}

            # A retrieved sentence is relevant if it's from this question's story
            story_sents_in_memory = set(q["sentences"]) & memory_set
            if story_sents_in_memory:
                hits = len(retrieved_texts & story_sents_in_memory)
                total_relevant = len(story_sents_in_memory)
                p_at_5_scores.append(min(hits / min(5, total_relevant), 1.0))
            else:
                p_at_5_scores.append(0.0)

        results[N] = {
            "mean_p_at_5": float(np.mean(p_at_5_scores)),
            "std_p_at_5": float(np.std(p_at_5_scores)),
        }
        logger.info(f"    Scaling N={N}: P@5={results[N]['mean_p_at_5']:.4f}")

    return results


# ─────────────────────────────────────────────────────
# Main Run Function
# ─────────────────────────────────────────────────────

def run(
    config: Dict = None,
    seed: int = 42,
    data_dir: Path = None,
    results_dir: Path = None,
    logger: logging.Logger = None,
) -> Dict:
    """Run Phase 1 experiment for a single seed."""
    if logger is None:
        logger = logging.getLogger("HIDE.Phase1")
        logging.basicConfig(level=logging.INFO)
    if config is None:
        config_path = Path(__file__).parent / "config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
    if results_dir is None:
        results_dir = PROJECT_ROOT / "results" / "phase1"

    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Phase 1 — Seed {seed}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Load embedding model on GPU1
    logger.info("Loading embedding model on cuda:1...")
    embed_mgr = EmbeddingManager()
    embed_mgr.load_text_encoder("minilm", device=config["embedding"]["gpu"])

    # 2. Load bAbI training data
    logger.info("Loading bAbI training data...")
    task_ids = config["babi"]["tasks"]
    train_stories = load_babi_train_stories(task_ids)
    logger.info(f"  Loaded {len(train_stories)} unique stories for training")

    # 3. Train ContextProjector
    logger.info("Training ContextProjector...")
    projector = train_context_projector(
        train_stories, embed_mgr, config, seed, logger
    )

    proj_path = results_dir / f"projector_seed{seed}.pt"
    torch.save(projector.state_dict(), proj_path)
    logger.info(f"  Projector saved to {proj_path}")

    # 4. Load Qwen on GPU0
    logger.info("Loading Qwen model on cuda:0...")
    qwen = QwenGenerator(device=config["generation"]["gpu"])
    qwen.load(config["generation"]["model"])

    # 5. Evaluate on each bAbI task
    all_task_results = {}
    max_samples = config["babi"].get("max_test_samples", 1000)

    for task_id in task_ids:
        logger.info(f"Evaluating Task {task_id}...")
        test_questions = load_babi_task(task_id, split="test")
        logger.info(f"  Task {task_id} test: {len(test_questions)} questions")

        task_result = evaluate_task(
            task_id, test_questions, embed_mgr, projector, qwen,
            config, seed, logger, max_samples=max_samples,
        )
        all_task_results[task_id] = task_result

    # 6. Memory scaling experiment (use ALL tasks' training data for more unique sents)
    logger.info("Running memory scaling experiment...")
    all_train_qs = []
    for tid in task_ids:
        all_train_qs.extend(load_babi_task(tid, split="train"))
    scaling_results = run_memory_scaling(
        all_train_qs, embed_mgr, projector, config, seed, logger
    )

    # 7. Compile results
    result = {
        "seed": seed,
        "tasks": {},
        "memory_scaling": scaling_results,
        "config": {
            "embedding_model": config["embedding"]["model"],
            "embedding_dim": config["embedding"]["dim"],
            "generation_model": qwen.model_name,
        },
    }

    for task_id, task_result in all_task_results.items():
        result["tasks"][str(task_id)] = task_result

    hide_accs = [all_task_results[t]["hide_accuracy"] for t in task_ids]
    no_mem_accs = [all_task_results[t]["no_memory_accuracy"] for t in task_ids]
    random_accs = [all_task_results[t]["random_retrieval_accuracy"] for t in task_ids]

    result["summary"] = {
        "mean_hide_accuracy": float(np.mean(hide_accs)),
        "mean_no_memory_accuracy": float(np.mean(no_mem_accs)),
        "mean_random_accuracy": float(np.mean(random_accs)),
        "hide_beats_no_memory": sum(1 for h, n in zip(hide_accs, no_mem_accs) if h > n),
        "hide_beats_random": sum(1 for h, r in zip(hide_accs, random_accs) if h > r),
    }

    # Unload Qwen to free GPU memory
    qwen.unload()

    # Save results
    out_path = results_dir / f"results_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"Phase 1 Seed {seed} complete: HIDE mean={result['summary']['mean_hide_accuracy']:.4f}")

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("HIDE.Phase1")

    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    result = run(config=config, seed=args.seed, logger=logger)
    print(json.dumps(result["summary"], indent=2))
