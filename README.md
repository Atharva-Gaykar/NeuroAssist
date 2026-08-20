---
license: apache-2.0
title: Neuroassist
sdk: docker
emoji: 📚
colorFrom: yellow
colorTo: red
pinned: true
short_description: Brain Tumor realted stuff.
---

title: Neuroassist
emoji: 🏃
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
license: apache-2.0
short_description: Project related to brain tumor.

# 🧠 NeuroAssist
### *AI-Powered Brain Tumor Detection & Medical Assistant API*

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.118-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-CNN-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Deployed-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Neon-336791?style=for-the-badge&logo=postgresql&logoColor=white)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.0-D71F00?style=for-the-badge&logo=sqlalchemy&logoColor=white)
![Pinecone](https://img.shields.io/badge/Pinecone-VectorDB-00B383?style=for-the-badge)
![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq-LLM-F55036?style=for-the-badge)
![Cohere](https://img.shields.io/badge/Cohere-Reranker-39594D?style=for-the-badge)
![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-HF-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)


*A production-ready REST API for brain tumor detection & medical Q&A*

</div>

---

## 🔬 What is NeuroAssist?

**NeuroAssist** is a production-grade medical AI backend that:
- 🧬 **Detects brain tumors** from MRI scans using a CNN deep learning model built with TensorFlow
- 💬 **Answers medical questions** through a RAG-powered conversational assistant
- 🌐 **Searches the web** in real-time for treatment options and hospital information
- 📍 **Personalizes responses** based on the patient's city, state, and country
- 🔐 **Manages patients** with secure JWT-based authentication

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🖼️ **MRI Classification** | CNN classifies Glioma, Meningioma, Pituitary Tumor, or No Tumor |
| 🤖 **Medical Chatbot** | Groq LLM powered assistant with conversation memory |
| 🔍 **Hybrid RAG** | Dense + sparse (BM25) retrieval over scraped medical knowledge base |
| 🏆 **Reranking** | Cohere reranker selects the most relevant context chunks |
| 🌍 **Web Search** | SerpAPI fetches real-time hospital & treatment info near the patient |
| 📡 **REST API** | Full FastAPI backend with JWT auth, chat, and MRI endpoints |
| ☁️ **Cloud Native** | Neon PostgreSQL, Cloudinary image storage, HF Spaces hosting |

---

## 🏗️ Architecture

### Basic Flow
<img width="1033" height="677" alt="image" src="https://github.com/user-attachments/assets/e66d628e-8bd1-4571-9305-60b1cc4fde31" />

### Detailed System Design
[View on Eraser![](https://app.eraser.io/workspace/V1TZq958ULPGVF7A77Vt/preview?diagram=0JrgQvLYsg37lLQr5ARG&type=embed)](https://app.eraser.io/workspace/V1TZq958ULPGVF7A77Vt?diagram=0JrgQvLYsg37lLQr5ARG)

> 📚 The medical knowledge base was built by scraping medical websites using **BeautifulSoup**, then chunked and indexed into Pinecone for hybrid retrieval.

---

## 🛠️ Tech Stack

### 🖥️ Backend
- **[FastAPI](https://fastapi.tiangolo.com)** — REST API framework
- **Uvicorn** — ASGI server
- **Python 3.12**

### 🤖 AI / ML
- **[TensorFlow](https://www.tensorflow.org)** — CNN for MRI brain tumor classification
- **[Groq LLM](https://groq.com)** — LLM for medical Q&A responses and query reformulation
- **[LangChain](https://www.langchain.com)** — RAG pipeline orchestration
- **[HuggingFace Sentence Transformers](https://huggingface.co)** — Dense embeddings hosted on HF Hub
- **[Pinecone](https://www.pinecone.io)** — Vector database with hybrid search
- **BM25** — Sparse retrieval
- **[Cohere](https://cohere.com)** — Context reranking
- **NLTK** — Text preprocessing

### 🌐 Web Intelligence
- **[SerpAPI](https://serpapi.com)** — Real-time web search during chat
- **BeautifulSoup + Requests** — Web scraping to build the medical knowledge base

### 🗄️ Database & Storage
- **[PostgreSQL](https://www.postgresql.org)** via **[Neon.tech](https://neon.tech)** — Serverless cloud database
- **[SQLAlchemy](https://www.sqlalchemy.org)** — ORM
- **[Cloudinary](https://cloudinary.com)** — MRI image cloud storage

### 🔐 Authentication
- **JWT** (python-jose) — Token-based authentication
- **Bcrypt / Passlib** — Password hashing

### 🚀 Deployment
- **[Docker](https://www.docker.com)** — Containerization
- **[Hugging Face Spaces](https://huggingface.co/spaces)** — Cloud hosting
- **[Hugging Face Hub](https://huggingface.co)** — Remote embeddings endpoint

---

## 🚀 API Endpoints

### Auth
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/register` | Register new patient with MRI upload |
| `POST` | `/api/sign_in` | Sign in with existing or new MRI |

### Patient
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/patient/info` | Get logged-in patient profile |

### Chat
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/chat/message` | Send message to AI medical assistant |
| `GET` | `/api/chat/history` | Retrieve full chat history |

---

## ⚙️ Environment Variables

```env
# AI Keys
GROQ_API_KEY=
COHERE_API_KEY=
SERP_API_KEY=

# Vector DB
PINECONE_API_KEY=
PINECONE_ENVIRONMENT=

# Database
DATABASE_URL=postgresql://...neon.tech/neondb?sslmode=require

# Storage
CLOUDINARY_CLOUD_NAME=
CLOUDINARY_API_KEY=
CLOUDINARY_API_SECRET=

# Auth
SECRET_KEY=
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# Google
GOOGLE_CREDENTIALS_JSON=
```

---

## 📦 Knowledge Base Construction

The medical knowledge base was built in a multi-step offline pipeline:

### 1. 🕷️ Web Scraping
Medical content was scraped from trusted health websites using **BeautifulSoup + Requests**, targeting tumor-specific pages covering symptoms, diagnosis, treatments, and prognosis.

### 2. 🗂️ Data Structuring
Scraped content was cleaned, segmented into meaningful chunks, and organized by tumor type — Glioma, Meningioma, Pituitary, and General.

### 3. 🏷️ Metadata Enrichment
Each chunk was tagged with structured metadata:
```json
{
  "tumor_type": "glioma",
  "section": "treatment",
  "subsection": "chemotherapy",
  "subsubsection": "temozolomide"
}
```

### 4. 📥 Indexing into Pinecone
Final chunks were embedded using a fine-tuned **MiniLM model** and indexed into Pinecone with both dense vectors and BM25 sparse encodings for hybrid search.

> 🔬 Full scraping and preprocessing pipeline available in the project notebooks.

---

## 🐳 Run with Docker

```bash
# Clone the repo
git clone https://github.com/Gaykar/NeuroAssist.git
cd NeuroAssist

# Set up your .env file
cp .env.example .env

# Build and run
docker build -t neuroassist .
docker run -p 7860:7860 --env-file .env neuroassist
```

API will be live at `http://localhost:7860`

---

## 📁 Project Structure

```
NeuroAssist/
├── app/
│   ├── ai/
│   │   ├── pipeline.py          # RAG orchestration
│   │   ├── vectordatabase.py    # Pinecone hybrid search setup
│   │   ├── agents.py            # Query router agent
│   │   ├── prompt.py            # LLM prompt templates
│   │   ├── cnnmodelapi.py       # Tumor classification inference
│   │   ├── serpretrievertool.py # SerpAPI web search retriever
│   │   └── tumor_chunks/        # Scraped medical knowledge base (JSON)
│   ├── core/
│   │   ├── config.py            # Settings & environment variables
│   │   └── auth.py              # JWT authentication utilities
│   └── database/
│       ├── models.py            # SQLAlchemy models (Patient, ChatMessage)
│       ├── connection.py        # Neon DB connection
│       └── crud.py              # Database operations
├── main.py                      # FastAPI app entry point
├── Dockerfile
└── requirements.txt
```


---

## 📊 Evaluation Strategy

NeuroAssist was evaluated at two independent levels:

1. **End-to-End RAG Pipeline Evaluation**
2. **Retriever-Only Evaluation**

This separation helps analyze:

* retrieval quality independently
* generation quality independently
* overall system effectiveness

---

# 1. 🧠 End-to-End RAG Pipeline Evaluation

The complete RAG pipeline was evaluated using **ROUGE metrics** to measure how closely the generated medical responses matched reference answers.

---

## ✅ Evaluation Flow

For each tumor type:

* Glioma
* Meningioma
* Pituitary Tumor

a dedicated evaluation dataset was created:

```text
glioma_test_questions.json
meningioma_test_questions.json
pituitary_test_questions.json
```

Each sample contains:

```json
{
  "question": "What are common symptoms of glioma?",
  "answer": "Common symptoms include headaches, seizures, nausea, and cognitive issues.",
  "chunk_id": 12
}
```

---

## 🔄 Pipeline Evaluation Procedure

For every evaluation sample:

### Step 1 — Query Execution

The question is passed into the same production RAG pipeline used by the FastAPI backend:

```python
final_context, retrieved_results, num_docs = pipeline.run(
    query=question,
    chat_history=[],
    tumor_type=tumor_type,
    city=city,
    state=state,
    country=country,
)
```

---

### Step 2 — Response Generation

The retrieved context is passed into the medical assistant prompt:

```python
chain = medical_assistant_prompt | llm
```

The LLM then generates the final medical response.

---

### Step 3 — Metric Calculation

Generated responses are compared against reference answers using:

* **ROUGE-1**
* **ROUGE-L**

Metrics computed:

| Metric    | Description                          |
| --------- | ------------------------------------ |
| Precision | Relevance of generated tokens        |
| Recall    | Coverage of reference answer         |
| F1 Score  | Balance between precision and recall |

---

## 📈 Pipeline Metrics

### ROUGE-1

Measures unigram overlap between generated and expected answers.

### ROUGE-L

Measures longest common subsequence similarity, capturing fluency and structural similarity.

---

## 📁 Evaluation Output

Each tumor type produces its own evaluation report:

```text
glioma_evaluation_results.json
meningioma_evaluation_results.json
pituitary_evaluation_results.json
```

This allows independent benchmarking across tumor categories.

---

# 2. 🔍 Retriever-Only Evaluation

The retriever component was evaluated independently to measure how accurately the system retrieves the correct medical chunk before generation.

This evaluation excludes the LLM entirely.

---

## ✅ Goal

Measure:

> “Does the retriever fetch the correct knowledge chunk for the question?”

---

## 📂 Retriever Evaluation Dataset

Dedicated retriever datasets were created:

```text
retriever_glioma_test.json
retriever_meningioma_test.json
retriever_pituitary_test.json
```

Each sample contains:

```json
{
  "question": "Which gender is more frequently diagnosed with meningiomas?",
  "chunk_content": "Meningiomas occur more often in women.",
  "target_chunk_id": 1
}
```

---

## 🔄 Retriever Evaluation Procedure

### Step 1 — Query Retrieval

The question is passed into the retriever pipeline:

```python
final_context, retrieved_results, num_docs = pipeline.run(
    query=question,
    chat_history=[],
    tumor_type=tumor_type,
    city=city,
    state=state,
    country=country,
)
```

---

### Step 2 — Extract Top Retrieved Chunk

The highest-ranked retrieved document is selected:

```python
top_doc = retrieved_results[0]
retrieved_chunk_id = top_doc.metadata["chunk_id"]
```

---

### Step 3 — Compare with Ground Truth

The predicted chunk ID is compared against:

```python
target_chunk_id
```

stored in the evaluation dataset.

---

## 📊 Retriever Accuracy Formula

```text
Accuracy = Correct Retrievals / Total Queries
```

Example:

| Query | Retrieved Chunk | Target Chunk | Correct |
| ----- | --------------- | ------------ | ------- |
| Q1    | 4               | 4            | ✅       |
| Q2    | 7               | 3            | ❌       |
| Q3    | 12              | 12           | ✅       |

Accuracy:

```text
2 / 3 = 66.7%
```

---

## 📁 Retriever Evaluation Output

Separate retriever reports are generated for each tumor type:

```text
glioma_retriever_results.json
meningioma_retriever_results.json
pituitary_retriever_results.json
```

---

# 🎯 Why Separate Evaluations?

Separating retrieval evaluation from generation evaluation helps identify failure sources more precisely.

| Component            | Measures                                 |
| -------------------- | ---------------------------------------- |
| Retriever Evaluation | Knowledge retrieval quality              |
| Pipeline Evaluation  | Final answer quality                     |
| Combined Analysis    | End-to-end medical assistant performance |

This methodology enables targeted optimization of:

* Pinecone retrieval
* BM25 sparse search
* Cohere reranking
* Prompt engineering
* LLM response quality

---

# 🧪 Evaluation Design Philosophy

The evaluation framework was designed to mimic real production usage:

* tumor-specific retrieval
* location-aware context
* hybrid search
* reranking
* conversational medical queries

This ensures that offline benchmark scores closely reflect real-world deployment behavior.

---

## 🌐 Live Demo

🔗 **[huggingface.co/spaces/Gaykar/Neuroassist](https://huggingface.co/spaces/Gaykar/Neuroassist)**

---

<div align="center">

Built with ❤️ by **Atharva Gaykar**

</div>