"""
D4 PEFT/LoRA training script for Colab or a CUDA machine.
Run from the root of the D4 package:

pip install -U transformers peft accelerate bitsandbytes datasets safetensors
python train_lora_adapter_colab.py
"""
from pathlib import Path
import json
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_PATH = Path("tuning_data/qa_tuning_data.jsonl")
OUT_DIR = Path("trained_lora_adapter")


def load_records(path: Path):
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def format_example(row):
    instruction = row.get("instruction", "Answer the question using retrieved context.")
    user_input = row.get("input", "")
    output = row.get("output", "")
    return f"### Instruction:\n{instruction}\n\n### Input:\n{user_input}\n\n### Answer:\n{output}"


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="float16")
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=quant_config, device_map="auto")
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)

    records = load_records(DATA_PATH)
    dataset = Dataset.from_dict({"text": [format_example(r) for r in records]})

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=768)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    args = TrainingArguments(
        output_dir=str(OUT_DIR),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"Saved LoRA adapter to {OUT_DIR}")


if __name__ == "__main__":
    main()
