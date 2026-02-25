# 🧠 Enterprise Multimodal RAG System

![Docker](https://img.shields.io/badge/Docker-Ready-111111?style=for-the-badge&logo=docker)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-111111?style=for-the-badge&logo=fastapi)
![Redis Stack](https://img.shields.io/badge/Redis_Stack-Vector_DB-111111?style=for-the-badge&logo=redis)
![Python](https://img.shields.io/badge/Python-Production-111111?style=for-the-badge&logo=python)

---

## 🚀 Overview

Production-grade **Multimodal Retrieval-Augmented Generation (RAG)** system built using:

- FastAPI backend  
- Redis Stack vector search  
- Embedding-based semantic retrieval  
- Dockerized deployment  
- Static frontend interface  

Designed for enterprise-scale intelligent document querying.

---

# 🏗 Architecture

## 🔄 System Flow

```
User Query
   │
   ▼
Generate Query Embedding
   │
   ▼
Redis Stack Vector Similarity Search
   │
   ▼
Retrieve Top-K Relevant Chunks
   │
   ▼
LLM Context Augmentation
   │
   ▼
Generated Response
```

---

## 📥 Ingestion Flow

```
PDF / Text Document
      │
      ▼
Text Extraction
      │
      ▼
Chunking
      │
      ▼
Embedding Generation
      │
      ▼
Store in Redis Vector Index
```

---

# 📦 Project Structure

```
enterprise-multimodal-rag/
├── app.py
├── ingest.py
├── static/
│   └── index.html
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env
```

---

# 🧠 Redis Vector Index Design

The system uses **Redis Stack vector search**.

### Example Index Fields

- `content` → TEXT  
- `embedding` → VECTOR (FLOAT32)  
- `metadata` → JSON  

### Similarity Search

- Distance metric: COSINE  
- Top-K retrieval  
- Approximate Nearest Neighbor (ANN) indexing  

Ensures:

- ⚡ Low latency retrieval  
- 📈 Scalable vector storage  
- 🔍 Accurate semantic search  

---

# 🔌 API Example

## ➜ Query Endpoint

### Request

```
POST /query
Content-Type: application/json
```

```json
{
  "query": "Explain DGX-1 system architecture"
}
```

### Response

```json
{
  "answer": "The NVIDIA DGX-1 architecture is designed with multiple V100 GPUs connected via NVLink..."
}
```

---

# 🚀 Run Locally (One Command)

```bash
docker-compose up --build
```

Open in browser:

```
http://localhost:8000
```

---

# 🔐 Environment Variables

Create `.env` file:

```
OPENAI_API_KEY=
REDIS_HOST=redis
REDIS_PORT=6379
```

---

# 🐳 Deployment

This system is fully containerized and supports:

- Local Docker deployment  
- Cloud VM deployment  
- Kubernetes-ready architecture  
- Horizontal scaling  

---

# 🎯 Production Features

- Multimodal ingestion support  
- Redis Stack vector indexing  
- FastAPI async backend  
- Dockerized microservice architecture  
- Enterprise-ready REST API  
- Static frontend interface  

---

# 📊 Technical Highlights

- Vector similarity search with Redis  
- Context-aware LLM augmentation  
- Modular ingestion pipeline  
- Clean REST API interface  
- Container orchestration ready  

---

# 👨‍💻 Author

**Prathamesh Patil**  
Data Scientist | GenAI Engineer | Production ML Architect  

LinkedIn:  
https://www.linkedin.com/in/prathamesh-m-patil-810024229