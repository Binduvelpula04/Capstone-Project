## Graph-Based Dual-Mode Analysis for Cloud Security and Forensics

## Project Overview
This project aims to identify root causes of system issues by analyzing event logs and performance metrics. The approach integrates data preprocessing, feature engineering, knowledge graph construction, and natural language querying using a large language model (FLAN-T5). The ultimate goal is to provide comprehensive root cause analysis (RCA) and identify critical system components and events affecting performance.

---

## Team Members
- Bharath Cherukuru  
- Venkata Mahesh Kundurthi  
- Ushabindu Velpula  

**Faculty Advisor:** Dr. Dalal Alharthi  
**Date:** November 13, 2025

---

## Dataset
**Event Data:**  
- 733 rows, 21 columns  
- Sample columns: `TenantId`, `SourceSystem`, `TimeGenerated [UTC]`, `EventID`, `RenderedDescription`, `UserName`, `EventLevel`  

**Performance Data:**  
- 1692 rows, 18 columns  
- Sample columns: `TenantId`, `Computer`, `ObjectName`, `CounterName`, `CounterValue`, `TimeGenerated [UTC]`  

**Cleaned datasets:**  
- `events_cleaned.csv` → 658 events  
- `perf_cleaned.csv` → 1692 performance records  
- `unified_data.csv` → 43 merged time-series records  

---

## Methodology

### 1. Data Preprocessing
- Removed duplicates and missing/invalid values  
- Created 186 performance features including disk, memory, processor, and network metrics  
- Unified events and metrics into a consolidated time-series dataset  

### 2. Knowledge Graph Construction
- **Nodes:** System, Component, Event, Metric  
- **Relationships:** `OCCURS_IN`, `AFFECTS`, `CORRELATES_WITH`, `PRECEDES`  
- Graph stats: 87 nodes, 828 edges  
- Visualization: `knowledge_graph.html`  
- Graph saved for analysis: `knowledge_graph.gexf`, `knowledge_graph.graphml`  

### 3. Root Cause Analysis
- Computed correlations between events and metrics  
- Added causal edges based on significant correlations  
- Identified top root causes using PageRank, In-Degree/Out-Degree, and Betweenness Centrality  
- Example: `Event_16` (iommu fault reporting initialized) with Out-degree 31 and PageRank 0.1389  

### 4. LLM Integration
- Used `google/flan-t5-base` for natural language querying  
- Extracted 500 knowledge triples from the graph  
- Example queries:
  - “What events are related to the system?”  
  - “Which metrics are affected by events?”  
  - “What are the main root causes of system issues?”

---

## Key Findings
- Most influential system: `forensicsacl2`  
- Critical root events: `Event_16`, `Event_4672`, `Event_4624`  
- Highly affected metrics: LogicalDisk `% Idle Time`, Memory Pool Bytes, System Processor Queue Length  
- Event-metric correlations highlight performance bottlenecks for further investigation

---

## Deliverables
- Cleaned datasets: `events_cleaned.csv`, `perf_cleaned.csv`, `unified_data.csv`  
- Knowledge graph files: `knowledge_graph.html`, `knowledge_graph.gexf`, `knowledge_graph.graphml`  
- Root cause report: `root_cause_report.txt`  
- Correlation analysis plots: `top_correlations.png`  
- Code for preprocessing, feature engineering, knowledge graph construction, RCA, and LLM integration  

---

## Tools & Libraries
- Python: `pandas`, `numpy`, `matplotlib`, `networkx`, `pyvis`, `plotly`, `sklearn`  
- Hugging Face Transformers: `google/flan-t5-base`  
- Graph Visualization: PyVis, Gephi-compatible `.gexf` and `.graphml`  
- Platform: Google Colab (CUDA GPU enabled for LLM inference)  

---

## How to Run
1. Load the cleaned datasets: `events_cleaned.csv`, `perf_cleaned.csv`  
2. Run the preprocessing and feature engineering scripts  
3. Execute the knowledge graph construction code to generate nodes and edges  
4. Perform correlation and root cause analysis  
5. Load `FLAN-T5-base` for natural language querying on the graph  
6. Visualize interactive graph using `knowledge_graph.html`  
7. Open `root_cause_report.txt` for detailed findings  

---

## References
- [NetworkX Documentation](https://networkx.org/)  
- [PyVis Documentation](https://pyvis.readthedocs.io/)  
- [Hugging Face Transformers](https://huggingface.co/)  

---

## License
MIT – Free to use and modify