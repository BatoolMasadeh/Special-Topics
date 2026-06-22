"""D4 generation utilities for quantized loading and simple context caching.

These helpers are intentionally lightweight so they can be imported by the D4
notebook, FastAPI generation stage, or smoke tests. The cache stores repeated
GraphRAG prompts by a stable hash of the question and retrieved context. The
model loader supports 4-bit BitsAndBytes quantization when a CUDA environment is
available.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class RetrievedChunk:
    """Minimal chunk object passed from GraphRAG to the generator."""

    text: str
    citation: str = "unknown source"
    page_range: str = "unknown pages"


def build_graphrag_prompt(question: str, chunks: Iterable[RetrievedChunk]) -> str:
    """Build a grounded generation prompt from GraphRAG supporting chunks."""
    context_blocks = []
    for idx, chunk in enumerate(chunks, start=1):
        context_blocks.append(
            f"[{idx}] Source: {chunk.citation}; Pages: {chunk.page_range}\n{chunk.text}"
        )

    context = "\n\n".join(context_blocks).strip() or "No retrieved context was supplied."
    return (
        "You are a citation-aware GraphRAG assistant. Answer only from the retrieved context. "
        "If the context is insufficient, say that clearly.\n\n"
        f"Question: {question}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Answer with concise reasoning and cite the source numbers/pages where possible."
    )


class GraphRAGAnswerCache:
    """Small file-based cache for repeated D4 GraphRAG generation prompts."""

    def __init__(self, cache_dir: str | Path = "outputs/cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(question: str, context: str, model_name: str) -> str:
        payload = json.dumps(
            {"question": question, "context": context, "model": model_name},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[str]:
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))["answer"]

    def set(self, key: str, answer: str) -> None:
        path = self.cache_dir / f"{key}.json"
        path.write_text(json.dumps({"answer": answer}, ensure_ascii=False, indent=2), encoding="utf-8")


def load_quantized_causal_lm(model_name: str, use_4bit: bool = True):
    """Load a causal LM with optional 4-bit quantization.

    This function is safe to keep in the repo even when the grading environment
    does not have a GPU. It only imports heavy dependencies when called.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs = {"device_map": "auto"}

    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            import torch

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"4-bit quantization unavailable, loading without it: {exc}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return tokenizer, model


def generate_with_cache(question: str, chunks: Iterable[RetrievedChunk], model_name: str, generator_fn, cache_dir: str | Path = "outputs/cache") -> str:
    """Generate an answer using a supplied generator function and cache repeats."""
    chunks = list(chunks)
    prompt = build_graphrag_prompt(question, chunks)
    cache = GraphRAGAnswerCache(cache_dir)
    cache_key = cache.key(question, prompt, model_name)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    answer = generator_fn(prompt)
    cache.set(cache_key, answer)
    return answer
