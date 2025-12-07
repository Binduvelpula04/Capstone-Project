## Root Cause Analysis for Cloud Security and Forensics using Knowledge Graph and LLMs

## Project Overview
The biggest challenge in observability and cloud forensics today is the lack of "semantic interconnectivity" between fragmented logs and metrics. Traditional tools rely on threshold alerts that flag symptoms but often miss the underlying hardware faults or root causes.

This project implements a 4-Phase Causal Architecture that unifies Windows event logs and performance metrics into a single Knowledge Graph. By moving from simple correlation to causation, we utilize graph algorithms to rank failure modes and a Retrieval Augmented Generation (RAG) framework to explain findings in plain English.

**Goal:** To provide Site Reliability Engineers (SREs) with automated, evidence-backed intelligence to reduce Mean Time to Resolution (MTTR).

---

## Team Members
- Bharath Cherukuru  
- Venkata Mahesh Kundurthi  
- Ushabindu Velpula  

**Faculty Advisor:** Dr. Dalal Alharthi  
**Date:** December 6th, 2025

---

## Dataset
**Event Data:**  
- 733 rows, 21 columns  
- Sample columns: `TenantId`, `SourceSystem`, `TimeGenerated [UTC]`, `EventID`, `RenderedDescription`, `UserName`, `EventLevel`  

**Performance Data:**  
- 1692 rows, 18 columns  
- Sample columns: `TenantId`, `Computer`, `ObjectName`, `CounterName`, `CounterValue`, `TimeGenerated [UTC]`

## Methodology

Our solution transforms raw telemetry into actionable intelligence through four distinct phases:

**Phase 1: Data Engineering & Temporal Alignment**

We apply rigorous preprocessing to align asynchronous data sources:

- **Time Synchronization:** All timestamps are converted to Coordinated Universal Time (UTC) to establish causality across distributed systems.
- **Fuzzy Joining:** We utilize a "Fuzzy Join" technique to aggregate discrete event logs and continuous metrics into 1-minute windows, allowing us to assert that events and metric spikes occurred in the same temporal bucket.
- **Feature Engineering:**  
    - **Z-Score Normalization:** Metrics are normalized, with a threshold of Z > 3.0 used to identify statistical anomalies.
    - **Burst Detection:** We track the rate of events per minute to identify "Log Storms" that signal system intensity. 

**Phase 2: Knowledge Graph Construction**

We utilize **NetworkX** to build a directed graph (DiGraph) representing the system state.
- **Ontology:** Nodes include Systems, Components (e.g., LogicalDisk), Events, and Metrics. 
- **Edges:** Relationships include OCCURS_IN, AFFECTS, PRECEDES, and CORRELATES_WITH  
- **Temporal Logic:** An edge is created if an event occurs within 60 seconds of another, capturing immediate causal       cascades while filtering long-term drift.  

**Phase 3: Algorithmic Causal Inference**

We move beyond simple search to "scoring" potential root causes using a weighted algorithm:
         Score** = 0.4(OutDegree) + 0.3(PageRank) + 0.3(Betweenness)
- **Out-Degree (0.4):** Identifies "Originators" that trigger many downstream effects.  
- **PageRank (0.3):** Measures influence based on connections to critical components.  
- **Betweenness (0.3):** Identifies "Bottlenecks" that act as bridges in the failure path.

**Phase 4: Hybrid LLM Integration (RAG)**

- **Triple Extraction:** The graph is serialized into natural language triples (Subject-Predicate-Object).
- **Local Retrieval:** We use Google/FLAN-T5-base for local embedding and retrieval to ensure sensitive log data remains secure.
- **Narrative Synthesis:** Topologically relevant triples are sent to OpenAI GPT-5 Mini to generate human-readable executive summaries.

## Installation & Workflow

**Prerequisites**
The project requires libraries for data handling, graph construction, and machine learning, including **NetworkX**, **PyTorch** (with CUDA support checked), and **matplotlib**.

  **Steps to Run**
1. **Data Preprocessing:** Load event and performance CSVs. The system cleans timestamps, handles missing values, and merges data into a time-aligned dataset.
2. **Build Graph:** Construct the graph with nodes for systems, events, and metrics. Visualize using **matplotlib**(static) or **PyVis** (interactive).
3. **Run Analysis:** Calculate correlations (r > 0.5) to add edges and run graph algorithms (PageRank, Betweenness) to score root causes.
4. **Query LLM:** Use the query function to retrieve relevant graph triples and generate a Root Cause Analysis Report.

## Results: Case Study (forensicsacl2)
The pipeline was tested on a real dataset from the **forensicsacl2** server.
- **Graph Topology:** 87 Nodes and 828 Directed Edges
- **Identified Root Cause:** The system successfully filtered noise to identify Event 16 (IOMMU Fault) as "Patient Zero".
    - **Score:** 0.7064.
    - **Impact:** Directly impacted 31 downstream components, including the Filter Manager.
- **Symptom Identification:** Event 4672 (Special Privileges) was correctly identified as a symptom rather than a cause, as the system attempted to recover crashed services.

## Future Scope
1. **Semantic Retrieval:** Implement Transformer-based models like LogBERT to match logs by semantic intent rather than just keywords.
2. **Formal Ontology:** Evolve the schema into an OWL/RDF Ontology with SWRL rules to infer implicit dependencies.
3. **Dynamic Simulation:** Implement Graph Neural Networks (GNNs) to enable "Digital Twin" capabilities, allowing for "What-If" failure simulations.

## References
1. **Cloud forensic analysis:** Ackcent. https://ackcent.com/cloud-forensic-analysis-all-you-need-to-know/
2. **Cybersecurity Knowledge Graph:** PuppyGraph. https://www.puppygraph.com/blog/cybersecurity-knowledge-graphs
3. **Vulnerable network generator:** ScienceDirect. https://www.sciencedirect.com/science/article/pii/S0167404825002652
4. **Identifying Evidence for Cloud Forensic Analysis:** Academia.edu. https://www.academia.edu/109414970/Identifying_Evidence_for_Cloud_Forensic_Analysis
5. **Cybersecurity Knowledge Graphs:** Springer. https://link.springer.com/article/10.1007/s10115-023-01860-3
6. **PyRCA:** Arxiv. https://arxiv.org/abs/2306.11417
7. **KGroot:** Arxiv. https://arxiv.org/abs/2402.13264
8. **LLMs for Anomaly Detection:** ACL Anthology. https://aclanthology.org/2025.findings-naacl.333.pdf
9. **AIOps for Log Anomaly Detection:** ScienceDirect. https://www.sciencedirect.com/science/article/pii/S2667305325001346
10. **Root-KGD:** Arxiv. https://arxiv.org/abs/2406.13664