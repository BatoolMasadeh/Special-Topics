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