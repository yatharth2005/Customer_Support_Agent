# 🛡️ Insurance Claims Copilot

> An intelligent AI-powered customer support copilot for insurance claims — combining **LLM reasoning, tool calling, persistent memory, and Retrieval-Augmented Generation (RAG)** to deliver personalized and context-aware support.

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-Orchestration-1C3C3C)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-FF6B6B)](https://www.trychroma.com/)
[![Gemini](https://img.shields.io/badge/Gemini-Embeddings-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)
[![Groq](https://img.shields.io/badge/Groq-LLM-F55036)](https://groq.com/)

---


## ✨ Features

- 🤖 **AI Insurance Copilot**: Intelligent assistant for insurance customer-support and claims workflows.
- 🧠 **Conversational Memory**: Remembers relevant customer interactions and retrieves useful historical context.
- 🔎 **RAG Knowledge Base**: Retrieves relevant information from insurance support documents using ChromaDB and Gemini embeddings.
- 🛠️ **Tool Calling**: Allows the AI agent to interact with application services and retrieve real-time ticket/customer information.
- 🎫 **Ticket Management**: Create, retrieve, and manage customer support tickets.
- 📝 **AI Draft Generation**: Generate customer-support response drafts based on ticket context, knowledge, and memory.
- 👤 **Customer Memory Search**: Retrieve relevant memories associated with a customer.
- 📚 **Knowledge Ingestion**: Ingest `.md` and `.txt` knowledge-base documents into ChromaDB.
- 🧩 **FastAPI Backend**: Clean REST API architecture with automatic Swagger documentation.
- 🎨 **Streamlit Dashboard**: Interactive UI for working with the Insurance Claims Copilot.
- 🔭 **Test Coverage**: API endpoint tests and embedding/integration tests are included.
- ⚡ **Gemini Embeddings**: Uses Google's `gemini-embedding-001` embedding model for semantic knowledge retrieval.

---


# 🛠️ Technology Stack

## Backend

- **FastAPI** — REST API framework
- **Uvicorn** — ASGI server
- **Pydantic / Pydantic Settings** — configuration and validation
- **SQLite** — application persistence
- **ChromaDB** — vector storage and semantic retrieval

## AI / LLM

- **Groq** — LLM inference
- **LangChain** — LLM and application integration
- **LangMem** — conversational memory
- **Google Gemini** — knowledge-base embeddings
- **Gemini Embedding 001** — semantic embeddings

## Frontend

- **Streamlit** — interactive Insurance Claims Copilot dashboard

## Development

- **Python 3.11+**
- **uv**
- **unittest**
- **FastAPI Swagger / OpenAPI**

---

# 🏗️ Project Structure

```text
Customer_Support_Agent/
│
├── customer_support_agent/
│   ├── api/
│   │   ├── app_factory.py
│   │   ├── dependencies.py
│   │   └── routers/
│   ├── core/
│   │   └── settings.py
│   ├── integrations/
│   │   ├── memory/
│   │   ├── rag/
│   │   └── tools/
│   ├── repositories/
│   │   └── sqlite/
│   ├── schemas/
│   │   └── api.py
│   └── services/
│       ├── copilot_service.py
│       ├── draft_service.py
│       └── knowledge_service.py
│
├── knowledge_base/
├── app.py
├── main.py
├── pyproject.toml
├── requirements.txt
├── uv.lock
└── README.md
```

---

# 📚 Knowledge Base

The project includes a local `knowledge_base/` directory containing insurance-support documentation.

Supported formats:

```text
.md
.txt
```

Documents are split into chunks and indexed into **ChromaDB**.

The current embedding configuration uses:

```text
Gemini Embedding Model
        │
        ▼
gemini-embedding-001
        │
        ▼
     ChromaDB
        │
        ▼
Semantic Search
```

After changing the embedding model or rebuilding the knowledge index, re-ingestion is recommended.

---

# 🧠 Memory

The application maintains customer-related memory so the copilot can retrieve relevant information from previous interactions.

```text
Customer
   │
   ▼
Previous Interaction
   │
   ▼
Memory Store
   │
   ▼
Relevant Memories
   │
   ▼
Copilot
   │
   ▼
Personalized Response
```

---

# 🛠️ Tool Calling

The copilot is connected to application tools that allow it to work with real application data.

```text
User:
"What is the status of ticket 123?"

          │
          ▼

      AI Copilot
          │
          ▼
     Ticket Tool
          │
          ▼
      SQLite DB
          │
          ▼
   Ticket Information
          │
          ▼
    AI-generated answer
```

---

# 🎫 Ticket Management

The application provides APIs for customer-support tickets.

Supported workflows include:

- Create tickets
- List tickets
- Retrieve individual tickets
- Validate ticket requests
- Generate AI response drafts

---

# 📝 AI Draft Generation

The copilot can generate a support-response draft for a ticket using available context.

```text
Ticket
  │
  ├──► Customer Context
  ├──► Relevant Memories
  ├──► Knowledge Base
  └──► AI Reasoning
          │
          ▼
     Support Draft
```

---

# 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Check application health |
| `GET` | `/api/tickets` | List support tickets |
| `POST` | `/api/tickets` | Create a ticket |
| `GET` | `/api/tickets/{ticket_id}` | Retrieve a ticket |
| `POST` | `/api/tickets/{ticket_id}/generate-draft` | Generate an AI response draft |
| `GET` | `/api/drafts/{ticket_id}` | Retrieve a ticket draft |
| `PATCH` | `/api/drafts/{draft_id}` | Update a draft |
| `POST` | `/api/knowledge/ingest` | Ingest knowledge-base documents |
| `GET` | `/api/customers/{customer_id}/memories` | Retrieve customer memories |
| `GET` | `/api/customers/{customer_id}/memory-search` | Search customer memories |

---

# 📖 API Documentation

After starting the application, open:

```text
http://localhost:8000/docs
```

Swagger UI allows you to inspect and test the API interactively.

---

# 🚀 Quick Start

## 1. Clone the repository

```bash
git clone https://github.com/yatharth2005/Customer_Support_Agent.git
cd Customer_Support_Agent
```

## 2. Create a virtual environment

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

Using `pip`:

```bash
pip install -r requirements.txt
```

Or using `uv`:

```bash
uv sync
```

---

# 🔐 Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.1-8b-instant
LLM_TEMPERATURE=0.2

GOOGLE_API_KEY=your_google_api_key
GOOGLE_EMBEDDING_MODEL=gemini-embedding-001

API_HOST=0.0.0.0
API_PORT=8000
DASHBOARD_API_URL=http://localhost:8000
```

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | LLM inference |
| `GOOGLE_API_KEY` | Gemini embeddings |

> **Never commit your `.env` file or API keys to GitHub.**

---

# ▶️ Run the Application

## Start FastAPI

```bash
python main.py
```

API:

```text
http://localhost:8000
```

Swagger:

```text
http://localhost:8000/docs
```

## Start Streamlit

```bash
streamlit run app.py
```

---

# 📥 Ingest the Knowledge Base

Trigger:

```http
POST /api/knowledge/ingest
```

The ingestion pipeline is:

```text
knowledge_base/
      │
      ▼
Text Chunking
      │
      ▼
Gemini Embeddings
      │
      ▼
ChromaDB
      │
      ▼
Semantic Search
```

---

# 📊 Technology Stack

| Category | Technology |
|---|---|
| Language | Python 3.11+ |
| API | FastAPI |
| Server | Uvicorn |
| LLM | Groq |
| LLM Framework | LangChain |
| Embeddings | Google Gemini |
| Embedding Model | `gemini-embedding-001` |
| Vector Database | ChromaDB |
| Memory | LangMem |
| Database | SQLite |
| Frontend | Streamlit |
| Configuration | Pydantic Settings |
| Testing | unittest |
| Package Manager | uv / pip |

---

# 💡 Project Goal

Traditional customer-support systems require agents to manually search through customer records, previous conversations, internal documentation, ticket information, and policy information.

The Insurance Claims Copilot brings these sources together:

```text
                 ┌────────────────────┐
                 │  Insurance Support │
                 │     Copilot        │
                 └─────────┬──────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
     Customer          Knowledge        Ticket
      Memory             Base           System
          │                │                │
          └────────────────┼────────────────┘
                           ▼
                      AI Reasoning
                           │
                           ▼
                  Context-Aware Support
```

The goal is to build an **AI copilot that can access relevant information, use application tools, remember customer context, and assist support agents with real workflows.**

---

# 👨‍💻 Author

**Yatharth Shukla**

GitHub:  
https://github.com/yatharth2005

Repository:  
https://github.com/yatharth2005/Customer_Support_Agent

---

# 📄 License

This project is distributed under the **MIT License**.

---

<p align="center">

### 🛡️ Insurance Claims Copilot

**AI-powered customer support with memory, tools, and RAG**

Built with ❤️ using Python, FastAPI, Groq, Gemini, ChromaDB, LangMem & Streamlit.

</p>
