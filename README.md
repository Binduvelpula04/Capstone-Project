# Root Cause Analysis for Cloud Security & Forensics using Knowledge Graphs and LLMs

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

# 📂 Dataset

### **Event Logs**
- 733 rows, 21 columns  
- Sample fields: `TenantId`, `SourceSystem`, `TimeGenerated`, `EventID`, `RenderedDescription`, `UserName`, `EventLevel`

### **Performance Metrics**
- 1692 rows, 18 columns  
- Sample fields: `TenantId`, `Computer`, `ObjectName`, `CounterName`, `CounterValue`, `TimeGenerated`

---

# 🧭 Methodology

This pipeline converts raw telemetry into actionable causal intelligence across **four phases**:

---

## 🔹 Phase 1: Data Engineering & Temporal Alignment
- **Time Synchronization:** Convert timestamps to UTC  
- **Fuzzy Joining:** Align logs + metrics into **1-minute windows**  
- **Feature Engineering:**  
  - Z-score normalization (Z > 3 → anomaly)  
  - Burst detection for log storms  

---

## 🔹 Phase 2: Knowledge Graph Construction
We use **NetworkX DiGraph** to model events, metrics, and systems.

### **Nodes**
- Systems  
- Components (e.g., LogicalDisk)  
- Events  
- Performance Metrics  

### **Edges**
- `OCCURS_IN`  
- `AFFECTS`  
- `PRECEDES`  
- `CORRELATES_WITH`  

**Temporal Logic:** Edges represent relationships occurring within **60 seconds**, capturing causal cascades.

---

## 🔹 Phase 3: Algorithmic Causal Inference
A weighted score ranks potential root causes:

**Score = 0.4(OutDegree) + 0.3(PageRank) + 0.3(Betweenness)**

- **Out-Degree:** Detects originators  
- **PageRank:** Measures influence  
- **Betweenness:** Identifies bridges of failure  

---

## 🔹 Phase 4: Hybrid LLM Integration (RAG)
- Extract graph triples  
- Retrieve semantically related triples using **FLAN-T5**  
- Generate a human-readable summary using **OpenAI GPT-5 Mini**

Outputs include:
- Root cause reports  
- Causal chains  
- Analyst-friendly explanations  

---

# 🚀 Installation & Execution (Google Colab)

This project is built specifically for **Google Colab**.

---

## 🔑 Setting Up OpenAI API Key

### 1. Open Secrets Manager  
Left sidebar → **Key Icon**

### 2. Add Secret  
- **Name:** `OPENAI_API_KEY`  
- **Value:** your OpenAI key (`sk-...`)

### 3. Enable Access  
Toggle **Notebook Access** ON.

### 4. Load Key in Notebook
```python
from google.colab import userdata
import os

api_key = userdata.get('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = api_key
