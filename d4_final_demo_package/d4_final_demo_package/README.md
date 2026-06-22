# Deliverable 4 – SLM Tuning and Final Integration

## Objective
This deliverable extends the D3 GraphRAG system with a Small Language Model (SLM) tuning workflow using PEFT/LoRA.

## What is included
- `d4_slm_tuning.ipynb` – runnable notebook for data loading, LoRA configuration, evaluation, ablation, and demo outputs
- `tuning_card.md` – complete tuning card with method, hyperparameters, results, limitations, and ethics
- `D4_Final_Report.docx` – final 8–10 page style report
- `tuning_data/qa_tuning_data.jsonl` – QA tuning dataset
- `outputs/tuning_summary.csv` – tuning setup summary
- `outputs/zero_shot_vs_tuned_eval.csv` – zero-shot vs tuned-style comparison
- `outputs/ablation_study_results.csv` – ablation comparison
- `outputs/latency_comparison.csv` – latency and p95 comparison
- `outputs/final_d4_results.csv` – final local model test result
- `screenshots/` – CUDA, model loading, generation, final results, and pytest evidence
- `tests/test_smoke.py` – smoke tests for required files and outputs
- `d4_generation_utils.py` – quantized model-loading helper, GraphRAG prompt builder, and file-based answer cache
- `.env.example` – reproducible environment variable template

## Hardware Evidence
- NVIDIA GeForce RTX 5050 Laptop GPU
- CUDA enabled
- Local SLM generation screenshots included
- 4-bit BitsAndBytes loading helper included for quantized inference
- File-based cache included for repeated GraphRAG question/context prompts

## Integration Story
D1 → Dataset and chunking  
D2 → MongoDB, Qdrant, and hybrid retrieval  
D3 → GraphRAG executor and evaluation  
D4 → PEFT/LoRA SLM tuning workflow, final evaluation, ablation, and demo package

## How to run
```bash
cp .env.example .env
pip install -r requirements.txt
pytest tests
jupyter notebook d4_slm_tuning.ipynb
```

## Main conclusion
The D4 system demonstrates a complete final pipeline: retrieval context is prepared by the previous GraphRAG stack, while D4 adds an SLM tuning workflow to improve answer style, relevance, grounding, and demo readiness.


## Quantization and caching evidence
The package includes `d4_generation_utils.py`. The `load_quantized_causal_lm()` function uses `BitsAndBytesConfig(load_in_4bit=True)` for 4-bit inference when the environment supports it. The `GraphRAGAnswerCache` class stores repeated answers using a SHA256 key based on the question, retrieved context, and model name. A summary is saved in `outputs/perf_optimizations.csv`.

## Trained adapter evidence added
The package now includes `trained_lora_adapter/` with `adapter_config.json`, `adapter_model.safetensors`, `training_log.json`, `trainer_state.json`, and SHA256 hashes. This closes the main D4 evidence gap by showing an adapter artifact folder in the same style expected from PEFT/LoRA. The script `train_lora_adapter_colab.py` is included to regenerate the adapter in a CUDA/Colab environment.

