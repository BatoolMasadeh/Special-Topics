from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]


def test_tuning_dataset_exists():
    path = BASE_DIR / "tuning_data" / "qa_tuning_data.jsonl"
    assert path.exists()
    assert path.stat().st_size > 0


def test_outputs_exist():
    expected_files = [
        "tuning_summary.csv",
        "zero_shot_vs_tuned_eval.csv",
        "final_d4_results.csv",
    ]

    for file_name in expected_files:
        path = BASE_DIR / "outputs" / file_name
        assert path.exists()
        assert path.stat().st_size > 0


def test_final_results_readable():
    path = BASE_DIR / "outputs" / "final_d4_results.csv"
    df = pd.read_csv(path)

    assert "model" in df.columns
    assert "status" in df.columns
    assert len(df) >= 1

def test_env_example_exists():
    path = BASE_DIR / ".env.example"
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "MONGO_URI" in content
    assert "ENABLE_4BIT" in content
    assert "ENABLE_CACHE" in content


def test_quantization_and_cache_helpers_exist():
    helper = BASE_DIR / "d4_generation_utils.py"
    assert helper.exists()
    text = helper.read_text(encoding="utf-8")
    assert "BitsAndBytesConfig" in text
    assert "load_in_4bit=True" in text
    assert "GraphRAGAnswerCache" in text


def test_perf_optimization_output_exists():
    path = BASE_DIR / "outputs" / "perf_optimizations.csv"
    assert path.exists()
    df = pd.read_csv(path)
    assert set(["optimization", "implemented_file", "evidence"]).issubset(df.columns)
    assert len(df) >= 2


def test_trained_lora_adapter_artifacts_exist():
    adapter_dir = BASE_DIR / "trained_lora_adapter"
    assert adapter_dir.exists()
    expected = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "training_log.json",
        "trainer_state.json",
        "manifest_sha256.json",
        "README.md",
    ]
    for file_name in expected:
        path = adapter_dir / file_name
        assert path.exists(), f"Missing {file_name}"
        assert path.stat().st_size > 0, f"Empty {file_name}"


def test_adapter_config_is_peft_lora():
    import json
    config_path = BASE_DIR / "trained_lora_adapter" / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["peft_type"] == "LORA"
    assert config["task_type"] == "CAUSAL_LM"
    assert config["r"] == 8
    assert "q_proj" in config["target_modules"]


def test_colab_training_script_exists():
    path = BASE_DIR / "train_lora_adapter_colab.py"
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "LoraConfig" in text
    assert "save_pretrained" in text
    assert "BitsAndBytesConfig" in text

