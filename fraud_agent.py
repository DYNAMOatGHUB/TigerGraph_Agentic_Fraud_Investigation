import asyncio
import os
import sys
import json
import time
import traceback
import warnings
import pandas as pd
from typing import List, Literal, Optional, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore", message=".*create_react_agent has been moved.*")

from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool

load_dotenv()

# ---------------------------------------------------------------------------
# Pydantic schema for the hackathon deliverable
# ---------------------------------------------------------------------------

class CaseDetails(BaseModel):
    status: Literal["open", "closed_fraud", "closed_legitimate", "escalated"] = Field(description="Where the case stands when your agent stops.")
    verdict: Literal["fraud", "legitimate", "uncertain"] = Field(description="Your conclusion")
    fraud_probability: float = Field(description="Probability between 0.0 and 1.0")
    pattern: Literal["card_testing", "card_not_present_fraud", "card_not_present_new_device", "out_of_region_use", "account_takeover", "undocumented", "none"]
    affected_txn_ids: List[str]
    exposure_usd: float
    evidence: List[str]
    similar_prior_cases: List[str]

class EvidenceRequest(BaseModel):
    type: Literal["customer_validation", "step_up_auth", "analyst_info"]
    asked_after_step: int
    assumed_response: str

class NextBestActionItem(BaseModel):
    action: Literal["ALLOW_TRANSACTION", "DECLINE_TRANSACTION", "MONITOR_CARD", "MONITOR_CONNECTED_CARDS", "WARN_CUSTOMER", "VERIFY_WITH_CUSTOMER", "STEP_UP_AUTH", "BLOCK_CARD", "BLOCK_ALL_CARDS", "GENERATE_REPORT", "CREATE_CASE", "FILE_REPORT", "ESCALATE_TO_ANALYST", "CLOSE_NO_FRAUD"]
    route: Literal["auto", "L1", "L2"]
    reason: str = Field(description="Cite the rule, e.g. 'R2'")

class NextBestActions(BaseModel):
    initial: List[NextBestActionItem]
    final: List[NextBestActionItem]
    what_changed: str

class SARDetails(BaseModel):
    file: bool
    reason: str
    narrative: str
    subjects: List[str]
    total_amount_usd: float

class HackathonReport(BaseModel):
    case_id: str
    case: CaseDetails
    evidence_requests: List[EvidenceRequest]
    next_best_actions: NextBestActions
    sar: SARDetails
    stop_reason: str
    tool_calls: int
    tokens: int
    latency_s: float

# ---------------------------------------------------------------------------
# Custom agentic tool-call loop
# ---------------------------------------------------------------------------

GRAPH_NAME = "FraudCaseGraph"

SYSTEM_PROMPT = f"""You are an expert fraud investigator with direct access to a TigerGraph database via tools.

ABSOLUTE RULES — VIOLATION = DISQUALIFICATION:
1. You MUST call tools to retrieve EVERY piece of data. Never guess or invent values.
2. Do NOT write Python code. Do NOT describe queries. CALL THE TOOLS NOW.
3. Only facts returned by tools may appear in your final answer.
4. You must format probabilities as decimals between 0.0 and 1.0. Next Best Actions must use exact policy action names (e.g., 'BLOCK_CARD', not 'Freeze account') and cite rules (e.g., 'R2').

GRAPH NAME: {GRAPH_NAME}

REAL VERTEX TYPES IN THIS GRAPH:
- FraudCase    (fields: case_id, opened_at, trigger_type, risk_score, outcome, pattern, txn_ids, exposure_usd, connected_card_ids, report_filed)
- Customer     (primary key: customer_id STRING)
- Card         (primary key: card_id STRING)
- Transaction  (primary key: flagged_txn_id INT — this is an INTEGER, not a string)
- FraudTxn     (primary key: first_fraud_txn_id INT)

REAL EDGE TYPES:
- fraud_case_flagged_transaction  (FraudCase <-> Transaction)
- fraud_case_first_fraud_transaction (FraudCase <-> FraudTxn)
- fraud_case_has_card             (FraudCase <-> Card)
- fraud_case_has_customer         (FraudCase <-> Customer)

EXACT TOOL SIGNATURES — use these parameter names exactly:

TOOL: tigergraph__get_nodes
  Required: vertex_type (must exactly match real vertex types above)
  Optional: where (filter string), graph_name="{GRAPH_NAME}", limit=100
  WARNING: Do NOT pass vertex_id. Use 'where' to filter. Do NOT use spaces around == (e.g. use "flagged_txn_id==3514030" not "flagged_txn_id == 3514030").
  EXAMPLE: vertex_type="Transaction", where="flagged_txn_id==3514030", graph_name="{GRAPH_NAME}"

TOOL: tigergraph__get_node_edges
  Required: vertex_type, vertex_id (the primary key VALUE as a string)
  Optional: graph_name="{GRAPH_NAME}", edge_type
  EXAMPLE: vertex_type="Customer", vertex_id="C12382", graph_name="{GRAPH_NAME}"

TOOL: tigergraph__run_query
  Required: query_text (full GSQL)
  EXAMPLE: query_text="INTERPRET QUERY () FOR GRAPH {GRAPH_NAME} {{ SELECT v FROM Transaction:v WHERE v.flagged_txn_id == 3514030 ACCUM PRINT v }}"

INVESTIGATION STEPS — Execute ALL in order:
Step 1: tigergraph__get_nodes(vertex_type="Transaction", where="flagged_txn_id==<FLAGGED_TXN_ID>", graph_name="{GRAPH_NAME}")
Step 2: tigergraph__get_node_edges(vertex_type="Customer", vertex_id="<CUSTOMER_ID>", graph_name="{GRAPH_NAME}")
Step 3: tigergraph__get_nodes(vertex_type="FraudCase", graph_name="{GRAPH_NAME}", limit=50)
Step 4: For any FraudCase with outcome containing "FRAUD" or "CLOSED", call tigergraph__get_node_edges(vertex_type="FraudCase", vertex_id="<case_id>", graph_name="{GRAPH_NAME}")
Step 5: Call search_fraud_policy(query="what is the policy for this situation?") to find applicable fraud policies and known fraud patterns. Use this policy evidence in your final JSON report.
"""

async def run_tool_call_loop(llm_with_tools, tools_by_name: dict, messages: list, max_steps: int = 20) -> tuple[list, int]:
    tool_call_count = 0

    for step in range(max_steps):
        print(f"\n[Step {step + 1}] Invoking LLM...", flush=True)
        response: AIMessage = await llm_with_tools.ainvoke(messages)
        messages.append(response)

        if not response.tool_calls:
            print("[Agent] No more tool calls — final response reached.")
            break

        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_call_id = tc["id"]
            tool_call_count += 1

            print(f"\n[🛠️  Tool Call {tool_call_count}]: {tool_name}")
            print(f"   Args: {json.dumps(tool_args, indent=2)[:300]}")

            if tool_name not in tools_by_name:
                tool_result = f"ERROR: Tool '{tool_name}' not found."
            else:
                try:
                    tool_result = await tools_by_name[tool_name].ainvoke(tool_args)
                except Exception as te:
                    tool_result = f"ERROR executing {tool_name}: {te}"

            result_str = str(tool_result)[:4000]
            print(f"   Result preview: {result_str[:200]}...")
            messages.append(ToolMessage(content=result_str, tool_call_id=tool_call_id))
    else:
        print(f"[Agent] Reached max steps ({max_steps}) — stopping loop.")

    return messages, tool_call_count


# ---------------------------------------------------------------------------
# Main agent entrypoint
# ---------------------------------------------------------------------------

async def run_fraud_agent():
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "tigergraph_mcp.main"],
        env={**os.environ}
    )

    print("🚀 Starting TigerGraph MCP Server...")

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                print("✅ MCP Session Initialized.")

                print("🛠️  Loading TigerGraph tools...")
                tools = await load_mcp_tools(session)
                valid_keywords = ['transaction', 'customer', 'card', 'device', 'case', 'query', 'search', 'gsql', 'nodes', 'edges']
                tools = [t for t in tools if any(kw in t.name.lower() or kw in t.description.lower() for kw in valid_keywords)]
                print(f"Filtered to {len(tools)} relevant tools.")

                tools_by_name = {t.name: t for t in tools}
                
                @tool
                async def search_fraud_policy(query: str) -> str:
                    """
                    Search the Fraud Policy and Known Fraud Patterns knowledge base.
                    Use this tool to find relevant rules, policies, and known patterns 
                    that justify a fraud verdict or explain a specific fraud pattern.
                    Provide a specific query string (e.g. "What is the policy for device age?").
                    """
                    try:
                        from langchain_ollama import OllamaEmbeddings
                        import warnings
                        import math
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            embedder = OllamaEmbeddings(model="nomic-embed-text")
                            
                        print(f"[GraphRAG] Embedding query: '{query}'...")
                        query_vector = await asyncio.to_thread(embedder.embed_query, query)
                        
                        get_nodes_tool = tools_by_name.get("tigergraph__get_nodes")
                        if not get_nodes_tool:
                            return "Error: get_nodes tool not available in MCP server."
                            
                        print(f"[GraphRAG] Fetching PolicyChunks from TigerGraph...")
                        res = await get_nodes_tool.ainvoke({
                            "graph_name": "FraudCaseGraph",
                            "vertex_type": "PolicyChunk",
                            "limit": 100
                        })
                        
                        import json
                        res_str = str(res)
                        if isinstance(res, list) and len(res) > 0 and hasattr(res[0], 'text'):
                            res_str = res[0].text
                        elif isinstance(res, list) and len(res) > 0 and isinstance(res[0], dict) and 'text' in res[0]:
                            res_str = res[0]['text']
                            
                        if "```json" in res_str:
                            res_str = res_str.split("```json")[1].split("```")[0].strip()
                        elif "```" in res_str:
                            res_str = res_str.split("```")[1].strip()
                            
                        res_data = json.loads(res_str)
                        vertices = res_data.get("data", {}).get("vertices", [])
                        
                        if not vertices:
                            return "No policies found."
                            
                        def cosine_sim(v1, v2):
                            dot = sum(a * b for a, b in zip(v1, v2))
                            mag1 = math.sqrt(sum(a * a for a in v1))
                            mag2 = math.sqrt(sum(b * b for b in v2))
                            return dot / (mag1 * mag2) if mag1 * mag2 > 0 else 0
                            
                        scored = []
                        for v in vertices:
                            emb = v.get("attributes", {}).get("embedding", [])
                            content = v.get("attributes", {}).get("content", "")
                            if emb and content:
                                sim = cosine_sim(query_vector, emb)
                                scored.append((sim, content))
                                
                        scored.sort(key=lambda x: x[0], reverse=True)
                        top_3 = [c for _, c in scored[:3]]
                        
                        return "\\n\\n---\\n\\n".join(top_3)
                        
                    except Exception as e:
                        return f"Error during GraphRAG search: {e}"

                tools.append(search_fraud_policy)
                tools_by_name[search_fraud_policy.name] = search_fraud_policy

                google_key = os.environ.get("GOOGLE_API_KEY")
                llm = None
                if google_key:
                    try:
                        print("🧠 Probing Gemini (gemini-3.6-flash)...")
                        test_llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)
                        await test_llm.ainvoke("ping")
                        llm = test_llm
                        print("✅ Gemini quota OK — using gemini-3.6-flash.")
                    except Exception as probe_err:
                        print(f"⚠️  Gemini unavailable: {str(probe_err)[:120]}")

                if llm is None:
                    print("🧠 Using local Ollama model: mistral-nemo")
                    llm = ChatOllama(
                        model="mistral-nemo",
                        temperature=0,
                        num_ctx=16384,
                        num_predict=2000, # Hard limit to prevent infinite generation hangs
                    )

                llm_with_tools = llm.bind_tools(tools)
                extraction_llm = llm.with_structured_output(HackathonReport)

                # ── Batch Processor ──────────────────────────────────────────
                os.makedirs("cases", exist_ok=True)
                df = pd.read_csv("Data Set/case_pack.csv")
                
                print(f"📦 Found {len(df)} cases to process in case_pack.csv")
                
                for idx, row in df.iterrows():
                    case_id = str(row['case_id'])
                    customer_id = str(row['customer_id'])
                    flagged_txn_id = str(row['flagged_txn_id'])
                    
                    file_path = f"cases/{case_id}.json"
                    if os.path.exists(file_path):
                        print(f"⏩ Skipping {case_id} — file already exists.")
                        continue
                        
                    print(f"\n🎯 Triggering Agent — Case {case_id}")
                    print("-" * 60)
                    
                    start_time = time.time()
                    
                    user_task = (
                        f"Investigate transaction {flagged_txn_id} for customer {customer_id}.\n"
                        "Use your tools to:\n"
                        f"1. Find the transaction details for flagged_txn_id={flagged_txn_id}.\n"
                        f"2. Identify all other cards owned by customer {customer_id}.\n"
                        "3. Find any device profiles connected to this customer.\n"
                        "4. Check if those device profiles appear in any prior CLOSED fraud cases.\n"
                        "5. Search the fraud policy (using the search tool) to find the specific known fraud pattern that applies here.\n"
                        "Do NOT fabricate any data. Only report what the tools return."
                    )

                    messages = [
                        SystemMessage(content=SYSTEM_PROMPT.replace("<FLAGGED_TXN_ID>", flagged_txn_id).replace("<CUSTOMER_ID>", customer_id)),
                        HumanMessage(content=user_task),
                    ]

                    messages, tool_call_count = await run_tool_call_loop(llm_with_tools, tools_by_name, messages)

                    if tool_call_count == 0:
                        print(f"\n❌ ABORT: Agent completed {case_id} without calling any tools.")
                        continue

                    print(f"\n✅ Tool calls completed: {tool_call_count}")
                    print("-" * 60)
                    print(f"🏁 Generating structured JSON report for {case_id}...")

                    investigation_summary = "\n".join(
                        f"[{m.__class__.__name__}]: {m.content[:500]}"
                        for m in messages
                        if hasattr(m, "content") and m.content
                    )

                    extraction_prompt = (
                        f"You are generating the final fraud investigation report.\n"
                        f"Case ID: {case_id}\n"
                        f"Tool calls made: {tool_call_count}\n\n"
                        f"Based ONLY on the following tool-returned data (do not invent anything), "
                        f"produce the HackathonReport JSON:\n\n"
                        f"{investigation_summary[:8000]}"
                    )

                    report: HackathonReport = await extraction_llm.ainvoke(extraction_prompt)

                    report.latency_s = round(time.time() - start_time, 2)
                    report.tool_calls = tool_call_count
                    report.tokens = 0

                    with open(file_path, "w") as f:
                        f.write(report.model_dump_json(indent=2))

                    print(f"\n✅ Saved report → {file_path}")
                    
                    # Prevent overloading the local LLM
                    await asyncio.sleep(2)

    except Exception as e:
        print("\n[ERROR] Fatal exception:")
        print(f"  {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_fraud_agent())
