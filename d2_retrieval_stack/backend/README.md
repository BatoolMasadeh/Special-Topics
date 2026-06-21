
# D2 Person 4 - FastAPI & Docker

## Objective
This component implements the backend API for the PDF Papers AI Agent project.

## Completed Features
- FastAPI backend application
- MongoDB integration
- GET / endpoint
- GET /stats endpoint
- POST /search endpoint
- POST /ingest endpoint
- Dockerfile
- docker-compose.yml
- Swagger UI testing screenshots

## Database
Database name: pdf_papers_ai_agent

Collections:
- papers: 99 documents
- chunks: 1958 documents

## API Endpoints
### GET /
Checks whether the API is running.

### GET /stats
Returns the number of papers and chunks stored in MongoDB.

### POST /search
Searches chunk text using a MongoDB text-search fallback.

This endpoint is designed to integrate with the Hybrid Retrieval module
(BM25 + Dense Search) developed as part of D2.

Example request:
{
  "query": "retrieval augmented generation",
  "top_k": 3
}

### POST /ingest
Returns ingestion status. The data was already processed in D1 and loaded into MongoDB.

## Docker Services
The Docker Compose file defines:
- fastapi
- mongodb
- qdrant
- neo4j

## Run Locally
docker compose up --build

Then open:
http://localhost:8000/docs

## Notes
The FastAPI app was tested in Google Colab using Swagger UI. Screenshots are included as evidence.
