# AI Semantic Search Web App

A full-stack NLP application that allows users to store text documents and search them using **semantic meaning** rather than keyword matching.

Built with **Cohere embeddings**, this project demonstrates how modern AI systems power features like search, retrieval, and recommendation engines.

---

## Live Demo

🔗 Frontend: https://cohere-search-frontend.vercel.app  
🔗 Backend API: https://cohere-semantic-search-api-p96p.vercel.app  

---

## Features

-  Semantic search using **machine learning embeddings**
-  Add and index documents dynamically
-  Real-time query processing via API
-  Top-k similarity retrieval (returns most relevant results)
-  Full-stack deployment (Next.js + FastAPI)

---

## Future Improvements
- PDF and file upload support
- Persistent storage

---

## How It Works

```text
User Input (Query)
        ↓
Cohere Embedding Model
        ↓
Vector Representation
        ↓
Similarity Comparison (NumPy dot product)
        ↓
Top Matching Documents Returned

