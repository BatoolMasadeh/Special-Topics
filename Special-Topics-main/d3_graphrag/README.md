# Deliverable 3 — GraphRAG Executor

## Overview

Deliverable 3 implements a GraphRAG retrieval system using the infrastructure developed in Deliverable 2.

The system combines graph-based retrieval using Neo4j with vector and keyword retrieval to generate answers supported by citations and page ranges.

## Connection to Previous Deliverables

### D1

Produced:

* chunks_final.csv

### D2

Implemented:

* MongoDB document storage
* Qdrant vector database
* Neo4j graph database
* Hybrid retrieval
* FastAPI backend

### D3

Uses the D1 dataset and D2 infrastructure to implement:

* GraphRAG retrieval
* Evaluation
* Safety mitigation
* Ablation studies

## Main Notebook

* Special Topics D3.ipynb

## Outputs

* graph_counts.csv
* answer_relevance.csv
* faithfulness.csv
* final_evaluation.csv
* safety_evaluation.csv
* ablation_study.csv

## Evaluation Metrics

* Answer Relevance
* Faithfulness
* p95 Latency

## Safety Mitigation

* Provenance filtering

## Ablation Study

Comparison of:

* BM25 Only
* Dense Only
* Hybrid Retrieval
* GraphRAG
