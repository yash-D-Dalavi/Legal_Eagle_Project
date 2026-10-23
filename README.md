# ⚖️ The Legal Eagle – AI-Powered Legal-Tech Suite

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-green.svg)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple.svg)](https://www.trychroma.com/)
[![Google Gemini](https://img.shields.io/badge/AI%20Engine-Google%20Gemini-orange.svg)](https://deepmind.google/technologies/gemini/)

> **An enterprise-grade AI legal document analyzer and public awareness platform designed specifically for Indian legal contexts.**

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Modules & Features](#-key-modules--features)
- [Tech Stack](#-tech-stack)
- [Architecture & RAG Pipeline](#-architecture--rag-pipeline)
- [Local Setup & Installation](#-local-setup--installation)
- [Environment Variables](#-environment-variables)
- [Usage Guide](#-usage-guide)
- [Disclaimer](#-disclaimer)

---

## 🚀 Overview

**The Legal Eagle** is designed to bridge the gap between complex legal jargon and everyday citizens in India. By leveraging Retrieval-Augmented Generation (RAG) and Google Gemini AI, the platform simplifies contracts, highlights risky clauses, provides domain-specific legal guidance, and generates formal legal notice drafts with multilingual support.

---

## ✨ Key Modules & Features

### 📄 Module 1: Personal Legal Document Analyzer (RAG Engine)

- **Document Upload & Parsing:** Reads and processes multi-page text-based legal PDFs (Rental Agreements, Employment Contracts, NDAs).
- **Vector Indexing:** Splits text using `RecursiveCharacterTextSplitter` and creates vector embeddings stored in **ChromaDB**.
- **Executive Summary:** Generates structured summaries detailing parties, duration, financial terms, and key obligations.
- **Heuristic Risk Meter:** Calculates a numeric Risk Score (0–100) and classifies contract risk into **Low**, **Medium**, or **High**.
- **Red Flag Detector:** Automatically identifies unfair/one-sided clauses (hidden penalties, strict lock-in periods, unilateral termination, non-compete rules) with exact verbatim quotes and practical suggestions.
- **Grounded Document Q&A:** Allows interactive Q&A over the PDF with similarity confidence scores for retrieved source passages.

### 📈 Module 2: Analytics & Quality Dashboard

- **Document Metrics:** Tracks word count, token estimates, clause count, legal term density, and readability complexity index.
- **Interactive Data Visualizations (Plotly):**
  - Risk distribution pie chart.
  - Red flag category breakdown bar chart.
  - Retrieval confidence score gauges and passage similarity match graphs.

### 🌐 Module 3: Public Legal Rights & Awareness

- **Multi-Topic Guidance:** Provides legal assistance across major Indian legal domains:
  - Consumer Protection Act, 2019
  - Information Technology (IT) Act, 2000
  - Motor Vehicles Act & Traffic Rules
  - Tenancy & Rent Laws
  - Fundamental Rights & RTI
  - Employment & Labour Laws
- **Multilingual Support:** Delivers responses across **8 Indian languages**: English, Hinglish, Hindi, Marathi, Tamil, Telugu, Bengali, and Gujarati.
- **Actionable Plans:** Generates short answers, statute citations, step-by-step action plans, and official helpline/portal directories (e.g., NCH 1915, Cybercrime 1930).

### ✍️ Module 4: Legal Notice Generator & Draft Editor

- **Automated Notice Drafting:** Generates formal legal notices in standard legal English (Consumer disputes, deposit refunds, unpaid dues/salary).
- **In-App Editing & Export:** Live text editor with immediate export capabilities to **PDF** and **.TXT** formats.

---

## 🧰 Tech Stack

- **Frontend & Framework:** Streamlit
- **AI & RAG Framework:** LangChain, LangChain Community, LangChain Google GenAI
- **Vector Database:** ChromaDB
- **LLM & Embeddings Engine:** Google Gemini API (`gemini-2.5-flash`)
- **Data Visualization:** Plotly Graph Objects
- **Document Processing & PDF Export:** PyPDFLoader, FPDF

---

## 🧱 Architecture & RAG Pipeline

```text
[Uploaded PDF] ➔ [PyPDFLoader] ➔ [Recursive Text Splitter]
                                          │
                                          ▼
        [ChromaDB Vector Store] ◄── [Google Gemini Embeddings]
                    │
                    ▼
[User Query Retrieval] ➔ [Context + Prompt] ➔ [Google Gemini LLM] ➔ [Structured Output / UI]
```

---

## 💻 Local Setup & Installation

### Prerequisites

- Python 3.10 or higher
- Git installed on your system
- Google Gemini API Key

### Steps to Run Locally

**1. Clone the Repository**

```bash
git clone https://github.com/yash-D-Dalavi/Legal_Eagle_Project.git
cd Legal_Eagle_Project
```

**2. Create and Activate a Virtual Environment**

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Linux / macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install Dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure Environment Variables**

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

**5. Run the Streamlit Application**

```bash
streamlit run app.py
```

---

## 🔐 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_API_KEY` | Your Google Gemini API Key for LLM and Embeddings | Yes |

---

## 📝 Usage Guide

1. Open the application in your browser (default: <http://localhost:8501>).
2. Navigate to **"Personal Legal Document Analyzer"** and click **"1-Click Demo"** or upload your own legal agreement PDF.
3. Review the Risk Index, Analytics Dashboard, and Red Flags.
4. Use the Q&A section to ask specific questions about the document.
5. Switch to **"Public Legal Rights & Awareness"** for instant Indian law advice in your preferred regional language.
6. Use **"Legal Notice Generator"** to draft and download formal legal notices as PDF or TXT.

---

## 🚨 Disclaimer

**The Legal Eagle** provides general legal information and document analysis powered by AI. It does **NOT** offer formal legal advice. Users should consult a qualified legal professional before acting on any information generated by this platform.

---

Developed by **Yash Dhananjay Dalavi**
