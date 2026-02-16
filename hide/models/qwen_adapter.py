"""
hide/models/qwen_adapter.py — Qwen2.5-7B Generation on GPU 0
==============================================================
Handles model loading, prompt formatting, and answer generation.
Always on cuda:0. Falls back to 3B or 4-bit if OOM.
"""

import torch
import logging
from typing import List, Optional

logger = logging.getLogger("HIDE.Qwen")


class QwenGenerator:
    """
    Wraps Qwen2.5-7B for answer generation.
    Pinned to cuda:0 on the 4xA100 cluster.
    """

    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.model = None
        self.tokenizer = None
        self.model_name = None

    def load(self, model_name: str = "Qwen/Qwen2.5-7B", quantize_4bit: bool = False):
        """Load model. Tries 7B first, falls back to 3B, then 4-bit."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        attempts = [
            (model_name, quantize_4bit),
            ("Qwen/Qwen2.5-7B", True),       # 4-bit 7B
            ("Qwen/Qwen2.5-3B", False),       # 3B full precision
            ("Qwen/Qwen2.5-3B", True),        # 3B 4-bit (last resort)
        ]

        for name, quant in attempts:
            try:
                logger.info(f"Loading {name} (4-bit={quant}) on {self.device}")

                load_kwargs = {
                    "trust_remote_code": True,
                    "device_map": {"": self.device},
                }

                if quant:
                    from transformers import BitsAndBytesConfig
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                    )
                else:
                    load_kwargs["torch_dtype"] = torch.bfloat16

                self.tokenizer = AutoTokenizer.from_pretrained(
                    name, trust_remote_code=True
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    name, **load_kwargs
                )
                self.model.eval()
                self.model_name = name
                logger.info(f"Successfully loaded {name}")
                return

            except Exception as e:
                logger.warning(f"Failed to load {name} (4-bit={quant}): {e}")
                self.model = None
                self.tokenizer = None
                torch.cuda.empty_cache()

        raise RuntimeError("Could not load any Qwen model variant")

    @torch.no_grad()
    def generate_answer(
        self,
        question: str,
        retrieved_memories: List[str],
        max_new_tokens: int = 10,
        temperature: float = 0.0,
    ) -> str:
        """
        Generate an answer using retrieved memories as context.
        Returns the generated text.
        """
        context_lines = []
        for i, mem in enumerate(retrieved_memories, 1):
            context_lines.append(f"[{i}]: {mem}")
        context_str = "\n".join(context_lines)

        prompt = (
            f"Context (retrieved from memory):\n{context_str}\n\n"
            f"Question: {question}\n\n"
            f"Answer in one or two words:"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        outputs = self.model.generate(**inputs, **gen_kwargs)

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return answer.strip().split("\n")[0].strip()

    @torch.no_grad()
    def generate_answer_no_context(
        self, question: str, max_new_tokens: int = 10
    ) -> str:
        """Generate answer without any memory context (baseline)."""
        prompt = f"Answer this question: {question}\nAnswer in one or two words:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return answer.strip().split("\n")[0].strip()

    @torch.no_grad()
    def generate_answer_full_context(
        self, question: str, full_story: str, max_new_tokens: int = 10
    ) -> str:
        """Generate answer with full story context (upper bound baseline)."""
        prompt = (
            f"Story:\n{full_story}\n\n"
            f"Question: {question}\n\n"
            f"Answer in one or two words:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return answer.strip().split("\n")[0].strip()

    def unload(self):
        """Free GPU memory."""
        import gc
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Qwen unloaded")


# Alias for backward compatibility
QwenAdapter = QwenGenerator
