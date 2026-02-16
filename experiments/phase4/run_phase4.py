"""
Phase 4: "Memory That Transfers" — Multi-Modal
================================================
Shared HIDE space across text/image with cross-modal retrieval.
COCO Captions for training, Flickr30k for evaluation.
Text → MiniLM (GPU1), Image → CLIP (GPU2), projections to shared 512-dim.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import logging

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "hide" / "core"))
sys.path.insert(0, str(PROJECT_ROOT / "hide" / "models"))
sys.path.insert(0, str(PROJECT_ROOT / "hide" / "utils"))

from hide_space import HIDESpace
from embedding_models import EmbeddingManager
from metrics import bootstrap_ci


# ─────────────────────────────────────────────────────
# Projection Layers (frozen encoders → shared space)
# ─────────────────────────────────────────────────────

class ModalityProjector(nn.Module):
    """Projects from modality-specific dim to shared HIDE space."""
    def __init__(self, in_dim: int, out_dim: int = 512):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class SymmetricInfoNCE(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, text_embs: torch.Tensor, image_embs: torch.Tensor) -> torch.Tensor:
        text_embs = nn.functional.normalize(text_embs, dim=-1)
        image_embs = nn.functional.normalize(image_embs, dim=-1)
        logits = text_embs @ image_embs.T / self.temperature.exp()
        labels = torch.arange(len(text_embs), device=text_embs.device)
        loss_t2i = nn.functional.cross_entropy(logits, labels)
        loss_i2t = nn.functional.cross_entropy(logits.T, labels)
        return (loss_t2i + loss_i2t) / 2


# ─────────────────────────────────────────────────────
# CLIP encoding
# ─────────────────────────────────────────────────────

def load_clip(device: str = "cuda:2", revision: str = "refs/pr/66"):
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", revision=revision).to(device)
    model.eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", revision=revision)
    return model, processor


def encode_images_clip(images, model, processor, device="cuda:2", batch_size=128):
    all_embs = []
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            vision_out = model.vision_model(pixel_values=inputs["pixel_values"])
            pooled = vision_out.pooler_output
            embs = model.visual_projection(pooled)
            embs = embs / (embs.norm(dim=-1, keepdim=True) + 1e-8)
            all_embs.append(embs.cpu().numpy())
    return np.concatenate(all_embs, axis=0)


# ─────────────────────────────────────────────────────
# Dataset Loading
# ─────────────────────────────────────────────────────

def load_coco_captions(max_samples=5000, cache_dir=None):
    """Load COCO captions (image-text pairs) from jxie/coco_captions.
    Caches text+images to disk to avoid re-downloading for each seed."""
    import pickle
    if cache_dir is None:
        cache_dir = PROJECT_ROOT / "data_cache"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"coco_captions_{max_samples}.pkl"

    if cache_file.exists():
        logging.getLogger("HIDE.Phase4").info(f"  Loading cached COCO from {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    from datasets import load_dataset
    pairs = []
    # jxie/coco_captions has: image, filename, cocoid, caption
    for split_name in ["train", "validation"]:
        try:
            ds = load_dataset("jxie/coco_captions", split=split_name, streaming=True)
            seen_ids = set()
            for item in ds:
                if len(pairs) >= max_samples:
                    break
                img = item.get("image", None)
                caption = item.get("caption", "")
                cocoid = item.get("cocoid", len(pairs))
                if img is None or not caption:
                    continue
                # Deduplicate by cocoid (multiple captions per image)
                if cocoid in seen_ids:
                    continue
                seen_ids.add(cocoid)
                # Convert PIL image to RGB to ensure consistency
                if hasattr(img, 'convert'):
                    img = img.convert("RGB")
                pairs.append({"image": img, "caption": str(caption), "id": len(pairs)})
            if len(pairs) >= max_samples:
                break
        except Exception as e:
            logging.getLogger("HIDE.Phase4").warning(f"Failed to load split {split_name}: {e}")
            continue

    # Cache to disk
    if pairs:
        with open(cache_file, "wb") as f:
            pickle.dump(pairs, f)
        logging.getLogger("HIDE.Phase4").info(f"  Cached {len(pairs)} COCO pairs to {cache_file}")
    return pairs


def load_flickr30k(max_samples=1000, cache_dir=None):
    """Load Flickr30k for cross-modal evaluation from lmms-lab/flickr30k."""
    import pickle
    if cache_dir is None:
        cache_dir = PROJECT_ROOT / "data_cache"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"flickr30k_{max_samples}.pkl"

    if cache_file.exists():
        logging.getLogger("HIDE.Phase4").info(f"  Loading cached Flickr30k from {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    from datasets import load_dataset
    pairs = []
    for split_name in ["test", "validation", "train"]:
        try:
            ds = load_dataset("lmms-lab/flickr30k", split=split_name, streaming=True)
            for item in ds:
                if len(pairs) >= max_samples:
                    break
                img = item.get("image", None)
                if img is None:
                    continue
                captions = item.get("caption", item.get("captions", []))
                if isinstance(captions, str):
                    captions = [captions]
                elif isinstance(captions, list) and len(captions) > 0:
                    if isinstance(captions[0], dict):
                        captions = [c.get("raw", str(c)) for c in captions]
                if not captions:
                    continue
                if hasattr(img, 'convert'):
                    img = img.convert("RGB")
                pairs.append({"image": img, "caption": str(captions[0]), "id": len(pairs)})
            if len(pairs) >= max_samples:
                break
        except Exception as e:
            logging.getLogger("HIDE.Phase4").warning(f"Flickr30k split {split_name} failed: {e}")
            continue

    if pairs:
        with open(cache_file, "wb") as f:
            pickle.dump(pairs, f)
        logging.getLogger("HIDE.Phase4").info(f"  Cached {len(pairs)} Flickr30k pairs to {cache_file}")
    return pairs


# ─────────────────────────────────────────────────────
# Training Projection Layers
# ─────────────────────────────────────────────────────

def train_projections(
    text_embs: np.ndarray,
    image_embs: np.ndarray,
    config: Dict,
    seed: int,
    logger: logging.Logger,
) -> tuple:
    """Train text and image projection layers using symmetric InfoNCE."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    proj_cfg = config.get("projection", {})
    device = proj_cfg.get("gpu", "cuda:1")
    lr = proj_cfg.get("lr", 1e-3)
    batch_size = proj_cfg.get("batch_size", 128)
    epochs = proj_cfg.get("epochs", 20)
    patience = proj_cfg.get("patience", 5)

    text_proj = ModalityProjector(proj_cfg.get("text_in_dim", 384), proj_cfg.get("output_dim", 512)).to(device)
    image_proj = ModalityProjector(proj_cfg.get("image_in_dim", 512), proj_cfg.get("output_dim", 512)).to(device)
    criterion = SymmetricInfoNCE(proj_cfg.get("temperature", 0.07)).to(device)

    params = list(text_proj.parameters()) + list(image_proj.parameters()) + list(criterion.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr)

    # Split into train/val
    n = len(text_embs)
    n_val = max(1, n // 10)
    perm = np.random.permutation(n)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    text_train = torch.tensor(text_embs[train_idx], device=device)
    image_train = torch.tensor(image_embs[train_idx], device=device)
    text_val = torch.tensor(text_embs[val_idx], device=device)
    image_val = torch.tensor(image_embs[val_idx], device=device)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        text_proj.train()
        image_proj.train()
        epoch_losses = []

        perm_train = np.random.permutation(len(train_idx))
        for start in range(0, len(perm_train) - batch_size + 1, batch_size):
            idx = perm_train[start:start + batch_size]
            t_proj = text_proj(text_train[idx])
            i_proj = image_proj(image_train[idx])
            loss = criterion(t_proj, i_proj)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        # Validation
        text_proj.eval()
        image_proj.eval()
        with torch.no_grad():
            val_batch = min(batch_size, len(val_idx))
            t_val = text_proj(text_val[:val_batch])
            i_val = image_proj(image_val[:val_batch])
            val_loss = criterion(t_val, i_val).item()

        avg_train = np.mean(epoch_losses) if epoch_losses else 0
        logger.info(f"    Epoch {epoch+1}/{epochs}: train={avg_train:.4f}, val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                "text_proj": {k: v.cpu().clone() for k, v in text_proj.state_dict().items()},
                "image_proj": {k: v.cpu().clone() for k, v in image_proj.state_dict().items()},
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"    Early stopping at epoch {epoch+1}")
                break

    if best_state:
        text_proj.load_state_dict(best_state["text_proj"])
        image_proj.load_state_dict(best_state["image_proj"])
    text_proj.to(device).eval()
    image_proj.to(device).eval()
    return text_proj, image_proj


# ─────────────────────────────────────────────────────
# Cross-Modal Retrieval Evaluation
# ─────────────────────────────────────────────────────

def evaluate_cross_modal(
    text_embs: np.ndarray,
    image_embs: np.ndarray,
    text_proj: ModalityProjector,
    image_proj: ModalityProjector,
    config: Dict,
    logger: logging.Logger,
) -> Dict:
    """Evaluate image→text and text→image retrieval."""
    device = config.get("projection", {}).get("gpu", "cuda:1")

    with torch.no_grad():
        t_proj = text_proj(torch.tensor(text_embs, device=device))
        i_proj = image_proj(torch.tensor(image_embs, device=device))

        t_proj = nn.functional.normalize(t_proj, dim=-1).cpu().numpy()
        i_proj = nn.functional.normalize(i_proj, dim=-1).cpu().numpy()

    n = len(text_embs)
    # Similarity matrix (image × text)
    sim_matrix = i_proj @ t_proj.T

    # Image → Text retrieval
    i2t_ranks = []
    for i in range(n):
        sorted_idx = np.argsort(sim_matrix[i])[::-1]
        rank = np.where(sorted_idx == i)[0][0] + 1
        i2t_ranks.append(rank)

    # Text → Image retrieval
    t2i_ranks = []
    for i in range(n):
        sorted_idx = np.argsort(sim_matrix[:, i])[::-1]
        rank = np.where(sorted_idx == i)[0][0] + 1
        t2i_ranks.append(rank)

    i2t_ranks = np.array(i2t_ranks)
    t2i_ranks = np.array(t2i_ranks)

    results = {
        "i2t_r1": float(np.mean(i2t_ranks <= 1)),
        "i2t_r5": float(np.mean(i2t_ranks <= 5)),
        "i2t_r10": float(np.mean(i2t_ranks <= 10)),
        "i2t_median_rank": float(np.median(i2t_ranks)),
        "t2i_r1": float(np.mean(t2i_ranks <= 1)),
        "t2i_r5": float(np.mean(t2i_ranks <= 5)),
        "t2i_r10": float(np.mean(t2i_ranks <= 10)),
        "t2i_median_rank": float(np.median(t2i_ranks)),
    }

    logger.info(f"  I→T: R@1={results['i2t_r1']:.4f}, R@5={results['i2t_r5']:.4f}, R@10={results['i2t_r10']:.4f}")
    logger.info(f"  T→I: R@1={results['t2i_r1']:.4f}, R@5={results['t2i_r5']:.4f}, R@10={results['t2i_r10']:.4f}")

    return results


def evaluate_random_projection(text_embs, image_embs, seed, logger):
    """Baseline: random projection instead of trained."""
    np.random.seed(seed)
    n = len(text_embs)
    # Random projection
    text_proj_mat = np.random.randn(text_embs.shape[1], 512).astype(np.float32) * 0.01
    image_proj_mat = np.random.randn(image_embs.shape[1], 512).astype(np.float32) * 0.01

    t_proj = text_embs @ text_proj_mat
    i_proj = image_embs @ image_proj_mat
    t_proj = t_proj / (np.linalg.norm(t_proj, axis=1, keepdims=True) + 1e-8)
    i_proj = i_proj / (np.linalg.norm(i_proj, axis=1, keepdims=True) + 1e-8)

    sim_matrix = i_proj @ t_proj.T

    i2t_ranks = []
    for i in range(n):
        sorted_idx = np.argsort(sim_matrix[i])[::-1]
        rank = np.where(sorted_idx == i)[0][0] + 1
        i2t_ranks.append(rank)

    t2i_ranks = []
    for i in range(n):
        sorted_idx = np.argsort(sim_matrix[:, i])[::-1]
        rank = np.where(sorted_idx == i)[0][0] + 1
        t2i_ranks.append(rank)

    i2t_ranks = np.array(i2t_ranks)
    t2i_ranks = np.array(t2i_ranks)

    return {
        "i2t_r1": float(np.mean(i2t_ranks <= 1)),
        "i2t_r5": float(np.mean(i2t_ranks <= 5)),
        "i2t_r10": float(np.mean(i2t_ranks <= 10)),
        "t2i_r1": float(np.mean(t2i_ranks <= 1)),
        "t2i_r5": float(np.mean(t2i_ranks <= 5)),
        "t2i_r10": float(np.mean(t2i_ranks <= 10)),
    }


# ─────────────────────────────────────────────────────
# Main Run
# ─────────────────────────────────────────────────────

def run(
    config: Dict = None,
    seed: int = 42,
    results_dir: Path = None,
    logger: logging.Logger = None,
) -> Dict:
    """Run Phase 4 for a single seed."""
    if logger is None:
        logger = logging.getLogger("HIDE.Phase4")
        logging.basicConfig(level=logging.INFO)
    if config is None:
        config_path = Path(__file__).parent / "config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
    if results_dir is None:
        results_dir = PROJECT_ROOT / "results" / "phase4"

    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Phase 4 — Seed {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Load encoders
    logger.info("Loading text encoder on cuda:1...")
    embed_mgr = EmbeddingManager()
    embed_mgr.load_text_encoder("minilm", device=config["text_encoder"]["gpu"])

    clip_device = config.get("clip", {}).get("gpu", "cuda:2")
    logger.info(f"Loading CLIP on {clip_device}...")
    clip_model, clip_processor = load_clip(clip_device, config.get("clip", {}).get("revision", "refs/pr/66"))

    # 2. Load COCO training data
    logger.info("Loading COCO captions...")
    max_train = config.get("dataset", {}).get("max_train", 5000)
    coco_pairs = load_coco_captions(max_samples=max_train)
    logger.info(f"  COCO: {len(coco_pairs)} pairs loaded")

    # 3. Encode text and images
    logger.info("Encoding COCO text...")
    coco_texts = [p["caption"] for p in coco_pairs]
    coco_text_embs = embed_mgr.encode_text(coco_texts, batch_size=512)

    logger.info("Encoding COCO images...")
    coco_images = [p["image"] for p in coco_pairs]
    coco_image_embs = encode_images_clip(coco_images, clip_model, clip_processor, clip_device)

    # 4. Train projections
    logger.info("Training projection layers...")
    text_proj, image_proj = train_projections(
        coco_text_embs, coco_image_embs, config, seed, logger
    )

    # 5. Save projections
    torch.save({
        "text_proj": text_proj.state_dict(),
        "image_proj": image_proj.state_dict(),
    }, results_dir / f"projections_seed{seed}.pt")

    # 6. Load Flickr30k for evaluation
    logger.info("Loading Flickr30k for evaluation...")
    max_test = config.get("dataset", {}).get("max_test", 1000)
    flickr_pairs = load_flickr30k(max_samples=max_test)
    logger.info(f"  Flickr30k: {len(flickr_pairs)} pairs loaded")

    # 7. Encode Flickr30k
    logger.info("Encoding Flickr30k...")
    flickr_texts = [p["caption"] for p in flickr_pairs]
    flickr_text_embs = embed_mgr.encode_text(flickr_texts, batch_size=512)
    flickr_images = [p["image"] for p in flickr_pairs]
    flickr_image_embs = encode_images_clip(flickr_images, clip_model, clip_processor, clip_device)

    # 8. Evaluate cross-modal retrieval (HIDE)
    logger.info("Evaluating cross-modal retrieval (HIDE projections)...")
    hide_results = evaluate_cross_modal(
        flickr_text_embs, flickr_image_embs, text_proj, image_proj, config, logger
    )

    # 9. Baseline: random projection
    logger.info("Evaluating random projection baseline...")
    random_results = evaluate_random_projection(flickr_text_embs, flickr_image_embs, seed, logger)
    logger.info(f"  Random I→T R@1={random_results['i2t_r1']:.4f}, T→I R@1={random_results['t2i_r1']:.4f}")

    # 10. Transfer: store COCO text memories, query with Flickr30k images
    logger.info("Evaluating cross-dataset transfer...")
    proj_device = config.get("projection", {}).get("gpu", "cuda:1")
    with torch.no_grad():
        coco_text_projected = text_proj(torch.tensor(coco_text_embs, device=proj_device))
        coco_text_projected = nn.functional.normalize(coco_text_projected, dim=-1).cpu().numpy()
        flickr_image_projected = image_proj(torch.tensor(flickr_image_embs, device=proj_device))
        flickr_image_projected = nn.functional.normalize(flickr_image_projected, dim=-1).cpu().numpy()

    # Store COCO text in HIDE space
    space = HIDESpace(dim=512, max_memories=len(coco_pairs) + 100)
    for i, (emb, text) in enumerate(zip(coco_text_projected, coco_texts)):
        space.store(emb, {"text": text, "source": "coco", "id": i})

    # Query with Flickr images
    transfer_hits = {"r1": 0, "r5": 0, "r10": 0}
    for i in range(min(100, len(flickr_pairs))):
        retrieved = space.retrieve(flickr_image_projected[i], k=10)
        retrieved_texts = [r[2]["text"] for r in retrieved]
        # Simple keyword overlap as relevance measure
        flickr_words = set(flickr_texts[i].lower().split())
        for k_val, key in [(1, "r1"), (5, "r5"), (10, "r10")]:
            for r_text in retrieved_texts[:k_val]:
                r_words = set(r_text.lower().split())
                if len(flickr_words & r_words) > 3:
                    transfer_hits[key] += 1
                    break

    n_transfer = min(100, len(flickr_pairs))
    transfer_results = {k: v / n_transfer for k, v in transfer_hits.items()}
    logger.info(f"  Transfer: R@1={transfer_results['r1']:.4f}, R@5={transfer_results['r5']:.4f}")

    # Cleanup
    del clip_model, clip_processor
    torch.cuda.empty_cache()

    # Compile
    result = {
        "seed": seed,
        "hide_retrieval": hide_results,
        "random_baseline": random_results,
        "transfer": transfer_results,
        "n_coco_train": len(coco_pairs),
        "n_flickr_test": len(flickr_pairs),
    }

    out_path = results_dir / f"results_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"Phase 4 Seed {seed} complete")

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("HIDE.Phase4")

    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    result = run(config=config, seed=args.seed, logger=logger)
    print(json.dumps({
        "hide_i2t_r1": result["hide_retrieval"]["i2t_r1"],
        "hide_t2i_r1": result["hide_retrieval"]["t2i_r1"],
        "random_i2t_r1": result["random_baseline"]["i2t_r1"],
    }, indent=2))
