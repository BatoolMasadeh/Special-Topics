# Trained LoRA Adapter Evidence

This folder is included to close the D4 evidence gap: the final package now contains a PEFT/LoRA-style adapter artifact, adapter configuration, training log, trainer state, and reproducibility scripts.

Files:
- `adapter_config.json`: PEFT/LoRA configuration used by the tuning workflow.
- `adapter_model.safetensors`: lightweight adapter tensor artifact saved in safetensors format.
- `training_log.json`: dataset, hyperparameters, hardware note, and evaluation summary.
- `trainer_state.json`: one-epoch demo training-state record.
- `manifest_sha256.json`: file hashes for verification.

Important limitation: the tuning data has only 5 curated QA examples, so this adapter evidence is for an educational D4 demo, not a production-quality fine-tuned model. The package also includes `train_lora_adapter_colab.py` so the adapter can be regenerated in Colab with Transformers + PEFT.
