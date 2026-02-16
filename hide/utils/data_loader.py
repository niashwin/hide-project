"""Dataset loading utilities for all phases."""

from typing import Dict, List


def load_babi(split: str = "train") -> Dict:
    """Load bAbI QA dataset (en-10k)."""
    from datasets import load_dataset
    ds = load_dataset("Muennighoff/babi", "en-10k", split=split)
    return ds


def load_templama() -> Dict:
    """Load TempLAMA temporal QA dataset."""
    from datasets import load_dataset
    ds = load_dataset("Yova/templama")
    return ds


def load_wikipedia_sentences(n_articles: int = 5000, n_sentences_per: int = 20,
                              seed: int = 42) -> List[Dict]:
    """Load Wikipedia sentences for interference/topology experiments."""
    from datasets import load_dataset
    import random
    random.seed(seed)
    ds = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
    sentences = []
    for i, article in enumerate(ds):
        if i >= n_articles:
            break
        text = article["text"]
        sents = [s.strip() for s in text.split(".") if len(s.strip()) > 20][:n_sentences_per]
        for s in sents:
            sentences.append({
                "text": s + ".",
                "article": article["title"],
                "article_id": i,
            })
    return sentences


def load_drm_word_lists() -> Dict:
    """Load DRM word lists (public domain, embedded in code)."""
    from hide.core.emergent import DRM_LISTS
    return DRM_LISTS
