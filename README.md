# AEIRS – (Autonomous Enterprise Intelligence and Reporting System)

Unified AI analysis platform for **Videos** and **Tabular Datasets**, built with **FastAPI + Groq + Pinecone** and a custom modern frontend.

- 🎥 **SnapSum** – upload video/audio , get transcript, summary, sentiment, key points, and chat with the Video.
- 📊 **AEIDS** – upload CSV/Excel datasets, run automated EDA, anomaly & correlation analysis, visualize results, and chat with the analysis.
- 🤖 **NEXUS AI Bot** – intelligent assistant embedded in the Dataset Analysis page and Image Analysis(RAG over your analysis results + vision OCR for image analysis).
---

## Features

### 🎥 SnapSum – Video / Audio Analysis
Available at [`/video-analysis`](http://localhost:8000/video-analysis).

- Upload **MP4 / MOV / MP3 / WAV**.
- Backend sends content to Groq:
  - Transcription
  - Summary
  - Sentiment (with visual bar and tags)
  - Key points list
- Results are stored in **SQLite** and indexed in **Pinecone**.
- Ask free-form questions at:
  - `POST /interviews/{interview_id}/ask` 
- Frontend chat shows a conversational **SnapSum bot** bound to that Video / Audio.

### 📊 AEIDS – Dataset Analysis
Available at [`/dataset-analysis`](http://localhost:8000/dataset-analysis).

- Upload CSV / Excel files.
- Backend AEIDS router under `/dataset-analysis/api` 
  - `POST /upload` – upload dataset  
  - `GET /files` – list uploaded datasets  
  - `POST /analyze/{file_id}` – full analysis (stats, anomalies, correlations)  
  - `GET /analysis/{file_id}` – retrieve analysis  
  - `GET /stats` – platform summary  
- Frontend (`frontend/new.html`) supports:
  - Dataset metadata view (columns, types, row counts)
  - Interactive charts (distributions, anomalies, correlations)
  - Insights feed & export to JSON / HTML report
  - Persistent upload & analysis history (backed by SQLite in the integrated version)

### 🤖 NEXUS AI – Dataset Copilot
- Integrated on the Dataset Analysis page as a floating chat widget.
- Backend endpoint:
  - `POST /dataset-analysis/api/nexus/chat` 
- Capabilities:
  - **Text Q&A** over your dataset analysis.
  - **RAG** over stored analysis results (AEIDS pipeline feeds results into `nexus_core.store_analysis`).
  - Optional **vision/OCR** using Groq vision model via `nexus_core.analyze_image(...)`.

### 🧩 App Shell & Navigation

- **Login page** served at `/` → `login.html`.
- **Welcome dashboard** at `/welcome` lets you choose:
  - Video Analysis
  - Dataset Analysis
- Each analysis module has its own rich, theme-aware UI:
  - `video_analysis.html` for SnapSum
  - `frontend/new.html` for AEIDS/NEXUS.

---

## Tech Stack

- **Backend**
  - FastAPI
  - Uvicorn - ASGI server
  - Groq – LLMs & vision (LLM + OCR)
  - Pinecone, Faiss – vector DB for interview transcripts
  - `pandas`, `numpy` – data analysis
  - `sqlite3` – local persistence for interviews and (optionally) dataset analyses

- **Frontend**
  - Plain HTML + CSS + vanilla JS
  - Font Awesome icons
  - Responsive layouts for desktop and tablet
  - Custom dark / light theme with a sky-style toggle
---

## Do This Befor Run

- GROQ API KEY INSIDE `app.py` --> `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`
- PINECONE API KEY INSIDE `app.py` --> `pcsk_32vRuT_EcaMAmX5NigWy1iQQX2PuJNQTVvBUuruPPCnmFviXb85UULqp2mrXEnesSpRJqF`
- GROQ API KEY INSIDE `nexus_ai.py` --> `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`

- **GO to AEIRS Folder inside the project** --> `cd AEIRS`
- **Install Requirements** --> `pip install -r requirements,txt` (near app.py)

## Run:
`uvicorn app:app --reload`  
(or)  
`py ap.py`
