# Special Topics in AI — GraphRAG Research Assistant

## Project Overview

This project was developed across three deliverables and implements a complete GraphRAG pipeline for research paper retrieval and question answering.

The system starts with document processing and chunk generation, builds a retrieval stack using databases and vector search, and finally implements a GraphRAG executor capable of graph-guided retrieval, evaluation, safety mitigation, and answer generation with citations.

---

## Project Pipeline

```text
Research Papers (PDFs)
          ↓
D1: Data Preparation & Retrieval Experiments
          ↓
chunks_final.csv
          ↓
D2: Retrieval Stack
(MongoDB + Qdrant + Neo4j + FastAPI)
          ↓
D3: GraphRAG Executor
          ↓
Answer Generation + Evaluation + Safety
```

---

## Repository Structure

```text
Special-Topics/
│
├── d1_preprocessing/
├── d2_retrieval_stack/
├── d3_graphrag/
│
├── reports/
│   ├── D1_Report.docx
│   ├── D2_Report.docx
│   └── D3_Report.docx
│
├── README.md
└── requirements.txt
```

---

# Deliverable 1 — Data Preparation & Retrieval Experiments

Main Components:

- PDF text extraction
- Text cleaning
- Chunk generation
- TF-IDF retrieval
- Dense retrieval
- Hybrid retrieval
- AutoML optimization using Optuna
- Online learning using River and ADWIN

Output:

- chunks_final.csv

Dataset Summary:

- 99 research papers
- 1,958 chunks

---

# Deliverable 2 — Retrieval Stack

Main Components:

- MongoDB document storage
- Qdrant vector database
- Hybrid retrieval
- FastAPI backend
- Docker deployment
- Neo4j graph preparation

Outputs:

- MongoDB collections
- Qdrant vector collection
- Retrieval API

---

# Deliverable 3 — GraphRAG Executor

Main Components:

- Cypher-based graph retrieval
- Supporting chunk expansion
- Hybrid graph + vector retrieval
- Answer generation with citations and page ranges
- Evaluation
- Safety mitigation
- Ablation study

Evaluation Metrics:

- Answer Relevance
- Faithfulness
- p95 Latency

Safety:

- Provenance filtering

Ablation Comparison:

- BM25 Only
- Dense Only
- Hybrid Retrieval
- GraphRAG

---

# Technologies Used

- Python
- Pandas
- Scikit-Learn
- Sentence Transformers
- Optuna
- River
- MongoDB
- Qdrant
- Neo4j Aura
- FastAPI
- Docker

---

# Outputs

The repository includes:

- Processed dataset
- Retrieval evaluation results
- AutoML optimization results
- Online learning results
- GraphRAG evaluation results
- Safety evaluation results
- Ablation study results
- Screenshots and implementation evidence

---

# Authors

Special Topics in AI Project Team

This repository combines Deliverables 1, 2, and 3 into a single end-to-end GraphRAG system.