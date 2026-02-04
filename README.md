# Root Cause Analysis for Cloud Security & Forensics using Knowledge Graphs and LLMs

**Repository:** [https://github.com/Binduvelpula04/Capstone-Project](https://github.com/Binduvelpula04/Capstone-Project)

## 📌 Project Overview
Modern observability and cloud forensics suffer from fragmented logs and metrics with no semantic interconnectivity. Traditional monitoring tools rely on static threshold alerts that identify symptoms but often miss hardware-level faults or the true root cause.

This project introduces a **4-Phase Causal Architecture** that unifies Windows Event Logs and performance metrics into a **single Knowledge Graph**. By transitioning from correlation → causation, we use graph algorithms and LLM-driven RAG explanations to automatically surface the most likely root cause behind system failures.

**🎯 Goal:** Provide Site Reliability Engineers (SREs) with automated, evidence-backed intelligence to reduce Mean Time to Resolution (MTTR).

---

## 👥 Team Members
- Bharath Cherukuru
- Venkata Mahesh Kundurthi
- Ushabindu Velpula

**Faculty Advisor:** Dr. Dalal Alharthi
**Date:** December 6th, 2025

---

## 📂 Dataset
The project relies on two primary CSV files located in the `Datasets/` directory:

### **1. Event Logs (`event.csv`)**
- 733 rows, 21 columns
- Sample fields: `TenantId`, `SourceSystem`, `TimeGenerated`, `EventID`, `RenderedDescription`, `UserName`, `EventLevel`

### **2. Performance Metrics (`perf.csv`)**
- 1692 rows, 18 columns
- Sample fields: `TenantId`, `Computer`, `ObjectName`, `CounterName`, `CounterValue`, `TimeGenerated`

---

## 🧭 Methodology

This pipeline converts raw telemetry into actionable causal intelligence across **four phases**:

### 🔹 Phase 1: Data Engineering & Temporal Alignment
- **Time Synchronization:** Convert timestamps to UTC
- **Fuzzy Joining:** Align logs + metrics into **1-minute windows**
- **Feature Engineering:**
  - Z-score normalization (Z > 3 → anomaly)
  - Burst detection for log storms

### 🔹 Phase 2: Knowledge Graph Construction
We use **NetworkX DiGraph** to model events, metrics, and systems.
- **Nodes:** Systems, Components, Events, Metrics
- **Edges:** `OCCURS_IN`, `AFFECTS`, `PRECEDES`, `CORRELATES_WITH`
- **Temporal Logic:** Edges represent relationships occurring within **60 seconds**.

### 🔹 Phase 3: Algorithmic Causal Inference
A weighted score ranks potential root causes:
**Score = 0.4(OutDegree) + 0.3(PageRank) + 0.3(Betweenness)**

### 🔹 Phase 4: Hybrid LLM Integration (RAG)
- Extract graph triples
- Retrieve semantically related triples using **FLAN-T5**
- Generate a human-readable summary using **OpenAI GPT-4o-mini**

---

## 🚀 Installation & Execution

### Option 1: Google Colab
1. Open the notebook `Source Code/root_cause_analysis_kg_main.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Upload `event.csv` and `perf.csv` to the session storage.
3. Set your OpenAI API Key in the Secrets Manager (Name: `OPENAI_API_KEY`).
4. Run all cells.

### Option 2: Local Execution
To run the code locally, ensure you have Python 3.8+ installed.

#### 1. Clone the Repository
```bash
git clone https://github.com/Binduvelpula04/Capstone-Project.git
cd Capstone-Project
```

#### 2. Install Dependencies
```bash
pip install -r requirements.txt
# OR manually install:
pip install pandas numpy networkx matplotlib seaborn pyvis scipy scikit-learn torch sentence-transformers plotly kaleido openai transformers
```

#### 3. Set OpenAI API Key
Set your OpenAI API key as an environment variable:
```bash
# macOS/Linux
export OPENAI_API_KEY="your-sk-key-here"

# Windows (Command Prompt)
set OPENAI_API_KEY=your-sk-key-here
```

#### 4. Run the Script
Make sure the `Datasets/` folder contains `event.csv` and `perf.csv`.
```bash
python "Source Code/root_cause_analysis_kg_main.py"
```

---

## 📜 License
This project is for academic purposes as part of the Capstone Project requirement.
