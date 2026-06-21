# Special Topics in AI — Deliverable 2: PDF-Papers AI Agent

## Project Overview

This repository contains **Deliverable 2 (D2)** for the **Special Topics in AI / CSAI415** course project.

The project builds a retrieval stack for a scientific PDF question-answering system. It processes research paper chunks, stores metadata in **MongoDB**, stores dense embeddings in **Qdrant**, performs **hybrid retrieval** using BM25 and dense vector search, exposes the system through a **FastAPI backend**, and builds a basic **Neo4j knowledge graph** for future GraphRAG integration.

This deliverable extends the previous D1 work and prepares the system for D3, where graph-guided retrieval, answer generation, safety mitigation, and faithfulness evaluation can be added.

---

## Repository Description

**PDF-Papers AI Agent retrieval stack using MongoDB, Qdrant, BM25+dense hybrid search, FastAPI, Docker Compose, and Neo4j graph build.**

---

## Main Features

- PDF/document chunk ingestion pipeline
- MongoDB storage for paper and chunk metadata
- Qdrant vector database for dense embedding search
- Hybrid retrieval using BM25 lexical search and dense vector retrieval
- FastAPI backend with retrieval endpoints
- Neo4j graph containing paper, author, and topic relationships
- Retrieval evaluation using Recall@5 and latency
- Top-k retrieved citation examples with page and chunk metadata
- Docker/Docker Compose support for reproducibility

---

## System Architecture

```text
PDF Research Papers
        ↓
Text Extraction
        ↓
Document Chunking
        ↓
MongoDB Storage
(Papers + Chunks + Metadata)
        ↓
Embedding Generation
        ↓
Qdrant Vector Database
        ↓
Hybrid Retrieval
(BM25 + Dense Vector Search)
        ↓
FastAPI API Layer
        ↓
Neo4j Knowledge Graph
        ↓
Retrieved Results with Citations
```

---

## Dataset Summary

The final cleaned corpus contains:

| Item | Count |
|---|---:|
| Research papers | 99 |
| Text chunks | 1,958 |
| MongoDB database | `pdf_papers_ai_agent` |
| MongoDB collections | `papers`, `chunks` |
| Qdrant collection | `research_papers` |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector size | 384 |
| Distance metric | Cosine |

The dataset contains successfully processed research papers and their chunked text content. Each chunk includes metadata such as paper ID, title, page number, chunk number, and chunk text.

---

## Main Components

### 1. MongoDB Storage

MongoDB is used to store structured metadata and chunk content.

Main collections:

```text
papers
chunks
```

Example paper document:

```json
{
  "paper_id": "paper_001",
  "title": "...",
  "pdf_file": "..."
}
```

Example chunk document:

```json
{
  "chunk_id": 1,
  "paper_id": "paper_001",
  "title": "...",
  "pdf_file": "...",
  "page": 1,
  "chunk_number": 1,
  "chunk_text": "..."
}
```

Indexes were created on important fields such as `paper_id`, `chunk_id`, and `page` to support efficient lookup and retrieval.

---

### 2. Qdrant Vector Database

Qdrant stores dense vector embeddings for document chunks.

Configuration:

```text
Collection: research_papers
Vector size: 384
Distance: Cosine
Embedding model: sentence-transformers/all-MiniLM-L6-v2
Vectors stored: 1,958
```

Qdrant enables semantic similarity search and supports dense retrieval in the hybrid search pipeline.

---

### 3. Hybrid Retrieval

The retrieval system combines:

- **BM25 lexical retrieval** for keyword-based matching
- **Dense vector retrieval** using Qdrant for semantic matching

The final score is calculated using a weighted fusion method:

```text
final_score = alpha × BM25_score + (1 - alpha) × dense_score
```

This improves retrieval quality by combining exact keyword matching with semantic similarity.

---

### 4. FastAPI Backend

The FastAPI backend provides API access to the retrieval system.

Implemented endpoints:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Confirms the API is running |
| GET | `/stats` | Returns database and retrieval system statistics |
| POST | `/search` | Performs hybrid search and returns relevant chunks |
| POST | `/ingest` | Inserts a new chunk into the storage layer |

Example `/search` request:

```json
{
  "query": "What is retrieval augmented generation?",
  "top_k": 5,
  "alpha": 0.5
}
```

Example `/search` response fields:

```json
{
  "query": "What is retrieval augmented generation?",
  "top_k": 5,
  "alpha": 0.5,
  "latency_ms": 195.51,
  "results": [
    {
      "chunk_id": "chunk_001",
      "paper_id": "paper_001",
      "title": "Example Paper Title",
      "page": 4,
      "chunk_number": 2,
      "score": 0.91,
      "citation": "paper_001, page 4, chunk 2",
      "text": "Retrieved chunk text..."
    }
  ]
}
```

Swagger documentation is available after running the API:

```text
http://127.0.0.1:8000/docs
```

---

### 5. Neo4j Knowledge Graph

Neo4j is used to represent relationships between papers, authors, and topics.

Main node types:

```text
Author
Paper
Topic
```

Main relationships:

```text
(:Author)-[:WROTE]->(:Paper)
(:Paper)-[:ABOUT]->(:Topic)
```

Example Cypher queries:

```cypher
MATCH (a:Author)-[:WROTE]->(p:Paper)-[:ABOUT]->(t:Topic)
RETURN a, p, t
```

```cypher
MATCH (a:Author)
RETURN a
```

```cypher
MATCH (p:Paper)
RETURN p
```

```cypher
MATCH (t:Topic)
RETURN t
```

```cypher
MATCH (n)
RETURN count(n)
```

---

## Evaluation Results

The retrieval system was evaluated using sample queries and measured using Recall@5 and latency.

| Metric | Value |
|---|---:|
| Average Recall@5 | 0.866 |
| Hybrid average latency | 195.51 ms |
| Qdrant Recall@5 | 0.70 |
| Qdrant average latency | 0.1578 s |
| Qdrant p95 latency | 0.1615 s |

Evaluation files are included in the repository:

```text
evaluation/person3_recall_results.csv
evaluation/person3_latency_results.csv
evaluation/top_k_citations.csv
evaluation/qdrant_evaluation.csv
```

The `top_k_citations.csv` file contains retrieved results with query, rank, score, paper ID, chunk ID, title, page number, chunk number, citation, and snippet.

---

## Recommended Repository Structure

```text
Special-Topics-AI-D2-PDF-Papers-Agent/
│
├── README.md
├── Deliverable_2_Final_Report.docx
├── D2_Final_Submission_ingestion_fixed.ipynb
│
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── database.py
│   │   ├── schemas.py
│   │   └── retrieval.py
│   │
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── requirements.txt
│   ├── .env.example
│   └── README.md
│
├── data/
│   ├── chunks_cleaned.csv
│   └── sample_queries.csv
│
├── evaluation/
│   ├── person3_recall_results.csv
│   ├── person3_latency_results.csv
│   ├── top_k_citations.csv
│   └── qdrant_evaluation.csv
│
├── neo4j/
│   ├── cypher_queries.md
│   └── graph_screenshots/
│
├── screenshots/
│   ├── mongodb/
│   ├── qdrant/
│   ├── fastapi/
│   ├── neo4j/
│   └── evaluation/
│
└── ai_chat_logs/
    ├── person1_ai_chat_link.txt
    ├── person2_ai_chat_link.txt
    ├── person3_ai_chat_link.txt
    ├── person4_ai_chat_link.txt
    └── person5_ai_chat_link.txt
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Special-Topics-AI-D2-PDF-Papers-Agent
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r backend/requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file based on `.env.example`.

Example:

```env
MONGO_URI=mongodb://localhost:27017/
MONGO_DB_NAME=pdf_papers_ai_agent
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=research_papers
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

Do not commit real passwords or API keys.

---

## Running with Docker Compose

From the backend folder:

```bash
cd backend
docker compose up --build
```

Then open:

```text
http://127.0.0.1:8000/docs
```

---

## Running Locally Without Docker

From the backend folder:

```bash
cd backend
python -m uvicorn app.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

---

## Example API Usage

### Root Endpoint

```bash
curl http://127.0.0.1:8000/
```

### Stats Endpoint

```bash
curl http://127.0.0.1:8000/stats
```

### Search Endpoint

```bash
curl -X POST "http://127.0.0.1:8000/search" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"What is retrieval augmented generation?\", \"top_k\":5, \"alpha\":0.5}"
```

---

## Team Contributions

| Member | Responsibility |
|---|---|
| Person 1 | MongoDB ingestion, dataset cleaning, paper/chunk storage |
| Person 2 | Qdrant vector database, embeddings, vector search evaluation |
| Person 3 | Hybrid retrieval, BM25 + dense fusion, Recall@5 and latency evaluation |
| Person 4 | FastAPI backend, API endpoints, Docker setup, integration |
| Person 5 | Neo4j graph build, report writing, screenshots, final documentation |

Update this table with the final team member names before submission.

---

## Deliverable 2 Requirements Covered

| D2 Requirement | Status |
|---|---|
| PDF/text/chunk ingestion pipeline | Completed |
| MongoDB storage | Completed |
| Qdrant vector database | Completed |
| Hybrid BM25 + dense retrieval | Completed |
| FastAPI backend | Completed |
| Neo4j graph build | Completed |
| 3–5 Cypher queries | Completed |
| Dataflow diagram | Completed |
| Recall@k and latency metrics | Completed |
| Top-k examples with citations | Completed |
| Docker Compose and reproducibility files | Included |

---

## Limitations and Future Work

This deliverable focuses on building the retrieval stack and graph database foundation required for D2. The current system supports MongoDB storage, Qdrant dense retrieval, BM25+dense hybrid search, FastAPI access, and a basic Neo4j knowledge graph.

Future work for D3 includes:

- Full GraphRAG executor
- Graph-guided retrieval and reranking
- Answer generation with grounded citations
- Faithfulness and answer-relevance evaluation
- Safety mitigation such as provenance filtering and source pinning
- Vector-only vs graph-guided vs hybrid ablation study

---

## Notes

This repository is submitted for **Special Topics in AI — Deliverable 2** and includes the final report, notebook, backend code, evaluation CSV files, screenshots, graph queries, and AI chat log evidence.
