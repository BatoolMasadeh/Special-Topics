# Special Topics in AI — D1

This repository contains the Deliverable 1 (D1) implementation for the **Special Topics in AI** project.

The goal of D1 is to build and evaluate an adaptive research-paper retrieval system using PDF chunking, baseline retrieval, AutoML optimization, and online learning with drift detection.

---

## Project Overview

The system follows this pipeline:

```text
PDF research papers
→ text extraction and chunking
→ TF-IDF / dense / hybrid retrieval
→ AutoML tuning
→ online learning with River
→ ADWIN drift detection
→ evaluation, plots, and report
```

---

## Team Contributions

| Person | Focus | Main Outputs |
|---|---|---|
| Person 1 | Dataset and chunking | `chunks_final.csv`, PDF source note, preprocessing notebook |
| Person 2 | Retrieval pipeline | Baseline retrieval results, retrieval metrics, retrieval notebook |
| Person 3 | AutoML experiments | Optuna notebook, best configuration, comparison table, optimization plots |
| Person 4 | River + ADWIN | Online learning results, drift events, prequential plot, artificial drift experiment |
| Person 5 | Evaluation/report/visuals | Final report, README, final organization |

---

## Repository Structure

```text
Special-Topics-in-AI---D1/
│
├── person 1 - batool/
│   ├── D1_Special_Topics.ipynb
│   ├── chunks_final.csv
│   ├── raw_pdfs_drive_link_note.pdf
│   └── brief report + ai chat (4).pdf
│
├── retrieval_pipeline -Person2/
│   ├── retrieval_pipeline.ipynb
│   ├── baseline_results.csv
│   ├── retrieval_metrics.csv
│   └── brief_report_ai_chat_person2.pdf
│
├── Person3/
│   ├── person3_automl.ipynb
│   ├── person 3 Quick report + AI_chat.pdf
│   └── outputs/
│       ├── best_config.yaml
│       ├── comparison_table.csv
│       ├── baseline_vs_automl.png
│       ├── optimization_history.png
│       ├── param_importance.png
│       └── embeddings.npy
│
├── person 4 - noor/
│   ├── person_4_noor.ipynb
│   ├── brief_report_ai_chat.pdf
│   ├── inputs/
│   │   ├── best_config.yaml
│   │   └── retrieval_results.csv
│   └── outputs/
│       ├── online_learning_results.csv
│       ├── drift_events.csv
│       ├── prequential_plot.png
│       ├── adaptive_alpha_plot.png
│       ├── person4_summary.csv
│       ├── person4_results_summary.txt
│       └── ARTIFICIAL - outputs/
│           ├── strong_artificial_drift_results.csv
│           ├── strong_artificial_drift_events.csv
│           └── strong_artificial_drift_plot.png
│
├── Alanoood person 2/
├── README.md
└── D1 final report.docx
```

---

## Main Components

### 1. Dataset and Chunking

Person 1 prepared the dataset foundation by extracting text from research paper PDFs and splitting the extracted text into searchable chunks.

Main output:

- `chunks_final.csv`

This file is used by the retrieval, AutoML, and online learning components.

Because the raw PDF folder was too large to upload directly to GitHub, a note file with the external drive link was included:

- `raw_pdfs_drive_link_note.pdf`

---

### 2. Retrieval Pipeline

Person 2 implemented the baseline retrieval system.

The retrieval pipeline includes:

- TF-IDF retrieval
- Dense retrieval
- Hybrid retrieval
- Top-k retrieval evaluation

Main outputs:

- `baseline_results.csv`
- `retrieval_metrics.csv`
- `retrieval_pipeline.ipynb`

---

### 3. AutoML Experiments

Person 3 used AutoML to improve retrieval performance by tuning retrieval parameters.

The AutoML component compares the baseline retrieval system with the optimized retrieval configuration.

Main outputs:

- `best_config.yaml`
- `comparison_table.csv`
- `baseline_vs_automl.png`
- `optimization_history.png`
- `param_importance.png`

The best configuration file is passed to Person 4 for online learning.

---

### 4. Online Learning and Drift Detection

Person 4 implemented the streaming learning component using River and ADWIN.

The online learning system uses retrieval results and the best AutoML configuration as inputs.

Main inputs:

- `best_config.yaml`
- `retrieval_results.csv`

Main outputs:

- `online_learning_results.csv`
- `drift_events.csv`
- `prequential_plot.png`
- `adaptive_alpha_plot.png`

An additional artificial drift experiment was included to demonstrate that ADWIN can detect drift when the data distribution changes strongly.

---

### 5. Evaluation and Reporting

The final stage combines the outputs from all members and summarizes the full D1 workflow.

The report includes:

- Dataset preparation
- Retrieval pipeline
- AutoML tuning
- Online learning
- ADWIN drift detection
- Results and limitations

---

## How to Run

Run the notebooks in this order:

1. `person 1 - batool/D1_Special_Topics.ipynb`
2. `retrieval_pipeline -Person2/retrieval_pipeline.ipynb`
3. `Person3/person3_automl.ipynb`
4. `person 4 - noor/person_4_noor.ipynb`

Make sure the required input files are placed in the correct folders before running each notebook.

---

## Required Python Libraries

The main libraries used in this project are:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `sentence-transformers`
- `optuna`
- `river`
- `pyyaml`

Install them using:

```bash
pip install pandas numpy scikit-learn matplotlib sentence-transformers optuna river pyyaml
```

---

## Key Output Files

| File | Description |
|---|---|
| `chunks_final.csv` | Clean chunked dataset |
| `baseline_results.csv` | Baseline retrieval outputs |
| `retrieval_metrics.csv` | Retrieval evaluation metrics |
| `best_config.yaml` | Best AutoML configuration |
| `comparison_table.csv` | Baseline vs AutoML comparison |
| `online_learning_results.csv` | River online learning results |
| `drift_events.csv` | ADWIN drift detection log |
| `prequential_plot.png` | Online learning performance plot |
| `strong_artificial_drift_events.csv` | Artificial drift detection events |

---

## Notes

- The raw PDF dataset was not uploaded directly because the full folder was too large for GitHub.
- The repository includes processed files and outputs required to reproduce the D1 workflow.
- Artificial drift was included only as an additional demonstration of ADWIN behavior.
- The main pipeline uses the retrieval outputs and AutoML configuration as the input for the online learning component.

---

## Deliverable 1 Summary

This D1 submission demonstrates:

- Dataset preparation and chunking
- Baseline and hybrid retrieval
- AutoML-based retrieval optimization
- Online learning using River
- ADWIN drift detection
- Prequential evaluation
- Organized outputs and visual evidence
