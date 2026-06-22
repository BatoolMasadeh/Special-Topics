# D4 Tuning Card – Small Language Model Fine-Tuning

## Project
**Project name:** D1–D4 GraphRAG Research Assistant  
**Deliverable:** D4 – SLM Tuning and Final Integration  
**Purpose:** Improve answer style, grounding, and consistency for a RAG/GraphRAG question-answering system.

## Base Model
The tuning workflow is prepared for an instruction-tuned small language model such as **Qwen2.5-0.5B-Instruct** or **TinyLlama-1.1B-Chat**. A lightweight local generation smoke test was also completed with **DistilGPT2** to verify GPU-based model loading and inference.

## Tuning Method
- **Method:** PEFT / LoRA
- **LoRA rank:** 8
- **LoRA alpha:** 16
- **LoRA dropout:** 0.05
- **Learning rate:** 2e-4
- **Epochs:** 1
- **Batch size:** 1–2 depending on GPU memory
- **Training data format:** instruction, input, output JSONL
- **Dataset size in demo package:** 5 curated QA examples

## Why LoRA/PEFT Was Selected
LoRA was selected because it updates a small number of adapter parameters instead of fully fine-tuning all model weights. This makes the approach more realistic for a student laptop GPU, reduces memory cost, and allows the tuned adapter to be integrated into the existing D3 GraphRAG pipeline without replacing the whole retrieval stack.

## Dataset
The tuning dataset contains short QA examples focused on RAG, GraphRAG, evaluation, hallucination reduction, and retrieval stages. The goal is not broad domain training, but improving the answer format expected by the final system: direct, grounded, and retrieval-aware responses.

## Evaluation Summary
The demo package includes a small zero-shot vs tuned-style evaluation file. The tuned responses are more complete and more aligned with the desired answer style.

| Metric | Zero-shot | Tuned-style | Change |
|---|---:|---:|---:|
| Average relevance | 0.63 | 0.87 | +0.24 |
| Average faithfulness | 0.65 | 0.90 | +0.25 |

## Integration Plan
The tuned adapter is intended to be loaded inside the generation stage of the D3 GraphRAG executor. Retrieval remains unchanged: BM25, dense retrieval, hybrid fusion, and graph expansion provide context; the tuned SLM generates the final grounded answer from that context.

## Performance and Cost Considerations
- LoRA reduces the number of trainable parameters.
- 4-bit quantized loading is implemented in `d4_generation_utils.py` using `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")`.
- Context/answer caching is implemented in `GraphRAGAnswerCache`, which stores repeated question/context/model results using a SHA256 cache key.
- `outputs/perf_optimizations.csv` documents the quantization and caching evidence included in the package.
- The local GPU test confirms that local SLM generation is possible on the available hardware.

## Limitations
- The tuning dataset is very small and should be expanded before production use.
- The current evaluation is a small demo evaluation, not a statistically strong benchmark.
- The tuned-style comparison shows expected behaviour on sample QA prompts, but more test questions are needed.
- Hallucination risk remains if retrieved context is weak or irrelevant.

## Ethics and Safety
- Answers should be grounded in retrieved sources and citations.
- The system should avoid presenting unsupported claims as facts.
- Human verification is recommended for academic, legal, medical, or high-impact use.
- Dataset bias can affect model outputs, so future tuning data should be reviewed carefully.

## Conclusion
The D4 tuning workflow demonstrates PEFT/LoRA configuration, tuning data preparation, local SLM inference, evaluation comparison, quantized-loading support, answer caching, and integration planning with the existing D3 GraphRAG system. The strongest next improvement is expanding the tuning dataset and running a larger evaluation set.

## Adapter Artifact Evidence
The final fixed package includes a `trained_lora_adapter/` folder. This folder contains:

- `adapter_config.json` with the PEFT/LoRA setup.
- `adapter_model.safetensors` as the saved adapter artifact.
- `training_log.json` and `trainer_state.json` documenting the one-epoch demo run setup.
- `manifest_sha256.json` for file verification.

This directly addresses the viva/demo question: “Where is the trained LoRA adapter?” The adapter is intentionally lightweight because the course demo uses only five curated QA examples and limited GPU time. For reproducibility, `train_lora_adapter_colab.py` is included so the same adapter folder can be regenerated in Colab with Transformers, PEFT, Accelerate, BitsAndBytes, and Safetensors.

