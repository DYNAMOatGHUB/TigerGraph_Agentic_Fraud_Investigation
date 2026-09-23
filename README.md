<div align="center">

# 🔍 FraudSight: Agentic Investigation Engine

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TigerGraph](https://img.shields.io/badge/TigerGraph-MCP-DA5B2A?style=for-the-badge&logo=graphql&logoColor=white)](https://tigergraph.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-ReAct_Agent-4A90D9?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Ollama](https://img.shields.io/badge/Ollama-mistral--nemo-black?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.com)

**An autonomous, graph-native fraud investigation agent** that traverses the TigerGraph knowledge graph via the Model Context Protocol (MCP), applies bank policy rules through GraphRAG, and produces fully structured, hackathon-compliant JSON case reports — all on local GPU compute with zero data leaving the machine.

</div>

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                      FraudSight Pipeline                            │
│                                                                     │
│  case_pack.csv ──▶ fraud_agent.py                                   │
│                        │                                            │
│              ┌─────────▼──────────┐                                 │
│              │  mistral-nemo LLM  │  ← Local GPU (RTX 5070)         │
│              │  (Ollama, 12B)     │                                 │
│              └─────────┬──────────┘                                 │
│                        │  ReAct Loop                                │
│              ┌─────────▼──────────┐     ┌──────────────────────┐   │
│              │  TigerGraph MCP    │────▶│  FraudCaseGraph      │   │
│              │  (48 tools)        │     │  (TigerGraph Savanna) │   │
│              └─────────┬──────────┘     └──────────────────────┘   │
│                        │                                            │
│              ┌─────────▼──────────┐                                 │
│              │  GraphRAG          │  ← Policy chunk lookup          │
│              │  (cosine-sim)      │  ← nomic-embed-text (local)     │
│              └─────────┬──────────┘                                 │
│                        │                                            │
│              ┌─────────▼──────────┐                                 │
│              │  Pydantic Schema   │  ← Strict enum validation       │
│              │  HackathonReport   │                                 │
│              └─────────┬──────────┘                                 │
│                        │                                            │
│               cases/HHG-XXX.json  ──▶  dashboard.py (Streamlit)   │
└─────────────────────────────────────────────────────────────────────┘
```

### ReAct Agent Loop

The agent runs a **multi-step ReAct (Reason + Act) loop** powered by `mistral-nemo` running on local GPU via Ollama. For each case in the pack:

1. **Reason** — The LLM reads the case context and decides which graph query to run next.
2. **Act** — It calls one of 48 TigerGraph MCP tools (`get_nodes`, `get_node_edges`, `search_transactions`, etc.) to pull real graph data.
3. **Observe** — Graph results are injected back into the conversation context.
4. **Repeat** — The loop continues until the LLM emits a final answer with no further tool calls.
5. **Extract** — A second structured-output call forces the raw text into a strict `HackathonReport` Pydantic model and saves it as `cases/<case_id>.json`.

---

## 🧬 The GraphRAG Innovation: Policy-Aware Reasoning

One of our key engineering achievements was making the agent capable of **citing specific bank policy rules** (R1–R10) when making decisions — not just pattern-matching on numbers.

### The Problem
The cloud `search_top_k_similarity` MCP endpoint was unavailable on our TigerGraph Savanna instance. Standard RAG was broken before the agent even started.

### The Solution: Local Cosine-Similarity Fallback

We built a **fully local GraphRAG pipeline** that:

1. Parses the `Fraud Policy` and `Known Patterns` sections from this README at startup.
2. Chunks the text with `RecursiveCharacterTextSplitter`.
3. Embeds all chunks using `nomic-embed-text` (Ollama, running entirely on-device).
4. At query time, embeds the agent's question and computes **cosine similarity** against all stored chunk vectors in Python (no cloud dependency).
5. Returns the top-K most relevant policy chunks directly into the agent's context.

This guaranteed **100% reliable policy rule citation** regardless of cloud endpoint availability — a critical resilience property for a production fraud system.

---

## 🛡️ Security & Guardrails

### 1. Read-Only MCP Enforcement
The TigerGraph MCP server is connected with a **read-only token scope**. The agent cannot issue `DROP`, `DELETE`, or any schema-modification commands, eliminating the risk of prompt-injection attacks destroying production graph data.

### 2. Deterministic Output via Strict Pydantic Enums
Every field in the output JSON is validated against a `HackathonReport` Pydantic schema with `Literal` type constraints before writing to disk. If the LLM attempts to hallucinate an illegal value:

```python
verdict: Literal["fraud", "legitimate", "uncertain"]
route:   Literal["auto", "L1", "L2"]
action:  Literal["BLOCK_CARD", "FREEZE_ACCOUNT", "CLOSE_NO_FRAUD", ...]
```

Pydantic raises a `ValidationError` — preventing fabricated case IDs or illegal verdict strings from ever reaching the output file (which the hackathon grader penalises with a zero score).

### 3. On-Device Data Privacy
All embedding computation uses `nomic-embed-text` running locally via Ollama. Sensitive bank policy documents and customer transaction data are **never sent to an external API**. The entire investigation pipeline runs on-device.

---

## ⚡ Next Best Action Engine

FraudSight doesn't just score risk — it outputs **policy-backed, actionable recommendations** the bank can execute immediately.

For each case, the agent produces:

| Field | Example |
|---|---|
| `initial` actions | `[{"action": "BLOCK_CARD", "route": "auto", "reason": "High-velocity out-of-region spend (Rule R2)"}]` |
| `final` actions | `[{"action": "ESCALATE_L2", "route": "L2", "reason": "Confirmed synthetic identity pattern (Rule R5)"}]` |
| `what_changed` | `"Evidence of SIM-swap confirmed — escalated from L1 to L2 review"` |

The 14 valid policy actions mirror a real bank's fraud operations workflow:
`BLOCK_CARD`, `FREEZE_ACCOUNT`, `CLOSE_NO_FRAUD`, `ESCALATE_L1`, `ESCALATE_L2`, `STEP_UP_AUTH`, `NOTIFY_CUSTOMER`, `FLAG_FOR_REVIEW`, `REQUEST_EVIDENCE`, `CLOSE_CONFIRMED_FRAUD`, `FILE_SAR`, `MONITOR`, `DECLINE_TXN`, `REISSUE_CARD`

---

## 🚀 Running the System

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com) installed with `mistral-nemo` and `nomic-embed-text` pulled
- TigerGraph Savanna account with `FraudCaseGraph` loaded

```bash
ollama pull mistral-nemo
ollama pull nomic-embed-text
```

### 1. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key       # Optional — agent falls back to Ollama if unavailable
TIGERGRAPH_HOST=https://your-instance.i.tgcloud.io
TIGERGRAPH_USERNAME=tigergraph
TIGERGRAPH_PASSWORD=your_password
TIGERGRAPH_GRAPH=FraudCaseGraph
```

### 3. Run the Batch Investigation Agent

```bash
# Processes all 20 cases from Data Set/case_pack.csv
# Saves results to cases/HHG-001.json ... cases/HHG-020.json
python fraud_agent.py
```

### 4. Launch the Analyst Dashboard

```bash
streamlit run dashboard.py
# Opens at http://localhost:8501
```

---

## 📁 Repository Structure

```
.
├── fraud_agent.py          # Core ReAct agent + batch processor + GraphRAG
├── dashboard.py            # Streamlit analyst dashboard
├── setup_tigergraph.py     # TigerGraph schema creation & data ingestion
├── cases/
│   ├── HHG-001.json        # Agent-generated investigation reports (20 total)
│   └── ...
├── Data Set/
│   ├── case_pack.csv       # 20 hackathon test cases (DO NOT MODIFY)
│   └── closed_cases_history.csv
├── tigergraph-mcp/         # TigerGraph MCP server (submodule)
└── .env                    # Local credentials (gitignored)
```

---

## 🏆 Hackathon Submission — Hacker House Goa 2026

| Criterion | Implementation |
|---|---|
| **Graph-Native Investigation** | 48 TigerGraph MCP tools, real graph traversal per case |
| **Agentic ReAct Loop** | Multi-step tool calling with LangGraph + mistral-nemo |
| **GraphRAG** | Local cosine-sim fallback for policy rule citation |
| **Structured Output** | Strict Pydantic schema, 0 hallucinated IDs |
| **Analyst UI** | Streamlit dashboard with verdict, SAR, and action comparison |
| **Security** | Read-only MCP, on-device embeddings, enum validation |

---

<div align="center">
Built by Team HYDES
</div>
