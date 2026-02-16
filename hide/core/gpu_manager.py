"""
GPU allocation manager for 4xA100 cluster.

GPU 0: Qwen2.5-7B (answer generation)
GPU 1: Embedding models (MiniLM/BGE)
GPU 2: CLIP / parallel experiments
GPU 3: FAISS-GPU / batch compute / overflow
"""

import os
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")

GPU_ASSIGNMENTS = {
    "qwen": "cuda:0",
    "embedding": "cuda:1",
    "clip": "cuda:2",
    "compute": "cuda:3",
}


def get_device(role: str = "embedding") -> torch.device:
    """Get the appropriate GPU device for a given role."""
    device_str = GPU_ASSIGNMENTS.get(role, "cuda:1")
    if not torch.cuda.is_available():
        return torch.device("cpu")
    gpu_idx = int(device_str.split(":")[-1])
    if gpu_idx >= torch.cuda.device_count():
        return torch.device("cuda:0")
    return torch.device(device_str)
