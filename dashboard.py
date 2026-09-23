"""
TigerGraph Agentic Fraud Investigation — Analyst Dashboard
Run: streamlit run dashboard.py
"""

import json
import os
import glob
import streamlit as st

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudSight | TigerGraph",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1526 50%, #0a0e1a 100%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1526 0%, #111827 100%);
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.1em; }
[data-testid="stMetricValue"] { color: #f1f5f9 !important; font-size: 1.6rem !important; font-weight: 700 !important; }
[data-testid="stTabs"] [role="tab"] { color: #64748b !important; font-weight: 600; font-size: 0.85rem; letter-spacing: 0.05em; text-transform: uppercase; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { color: #38bdf8 !important; border-bottom: 2px solid #38bdf8 !important; }
h1, h2, h3, h4 { color: #f1f5f9 !important; }
p, li, span { color: #cbd5e1; }
hr { border-color: #1e3a5f; }
.stMarkdown code { background: #1e293b; color: #7dd3fc; border-radius: 4px; padding: 2px 6px; }
.action-badge {
    display: inline-block; background: rgba(56,189,248,0.12); border: 1px solid rgba(56,189,248,0.3);
    border-radius: 6px; padding: 4px 12px; font-size: 0.78rem; font-weight: 700; color: #38bdf8;
    letter-spacing: 0.08em; margin: 3px 3px 3px 0;
}
.route-badge {
    display: inline-block; background: rgba(168,85,247,0.12); border: 1px solid rgba(168,85,247,0.3);
    border-radius: 6px; padding: 2px 10px; font-size: 0.72rem; font-weight: 600; color: #c084fc;
    letter-spacing: 0.05em; margin-left: 6px;
}
.evidence-item {
    background: rgba(255,255,255,0.03); border-left: 3px solid #38bdf8; border-radius: 0 8px 8px 0;
    padding: 10px 14px; margin: 6px 0; font-size: 0.88rem; color: #cbd5e1;
}
.prior-case-chip {
    display: inline-block; background: rgba(251,191,36,0.1); border: 1px solid rgba(251,191,36,0.3);
    border-radius: 20px; padding: 4px 14px; font-size: 0.78rem; color: #fbbf24; margin: 3px;
}
.what-changed-box {
    background: rgba(251,191,36,0.07); border: 1px solid rgba(251,191,36,0.25);
    border-radius: 10px; padding: 14px 18px; margin: 12px 0 0 0;
}
.tigergraph-logo {
    font-size: 1.1rem; font-weight: 800; letter-spacing: 0.05em;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_cases():
    cases = {}
    for path in sorted(glob.glob("cases/*.json")):
        with open(path) as f:
            try:
                data = json.load(f)
                cid = data.get("case_id", os.path.basename(path).replace(".json", ""))
                cases[cid] = data
            except json.JSONDecodeError:
                pass
    return cases


def normalise_probability(p):
    if p is None:
        return 0.0
    return float(p) / 100.0 if float(p) > 1.0 else float(p)


def verdict_badge(verdict):
    v = (verdict or "unknown").lower()
    if v == "fraud":
        return '<span style="background:rgba(239,68,68,0.15);border:1px solid rgba(239,68,68,0.5);border-radius:8px;padding:4px 14px;font-weight:700;color:#f87171;font-size:1rem;">🔴 FRAUD</span>'
    elif v == "legitimate":
        return '<span style="background:rgba(34,197,94,0.15);border:1px solid rgba(34,197,94,0.5);border-radius:8px;padding:4px 14px;font-weight:700;color:#4ade80;font-size:1rem;">🟢 LEGITIMATE</span>'
    else:
        return f'<span style="background:rgba(251,191,36,0.15);border:1px solid rgba(251,191,36,0.5);border-radius:8px;padding:4px 14px;font-weight:700;color:#fbbf24;font-size:1rem;">🟡 {v.upper()}</span>'


def render_actions(actions):
    if not actions:
        st.caption("No actions recorded.")
        return
    if isinstance(actions, list):
        for act in actions:
            if isinstance(act, dict):
                st.markdown(
                    f'<div style="margin:8px 0;">'
                    f'<span class="action-badge">{act.get("action","?")}</span>'
                    f'<span class="route-badge">⚡ {act.get("route","?").upper()}</span>'
                    f'<span style="color:#94a3b8;font-size:0.82rem;margin-left:8px;">{act.get("reason","")}</span>'
                    f'</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="evidence-item">• {act}</div>', unsafe_allow_html=True)
    else:
        for line in str(actions).split(","):
            st.markdown(f'<div class="evidence-item">• {line.strip()}</div>', unsafe_allow_html=True)


def render_evidence_requests(reqs):
    if not reqs:
        st.caption("No evidence requests made.")
        return
    for i, req in enumerate(reqs, 1):
        if isinstance(req, dict):
            req_type = req.get("type", "unknown").replace("_", " ").title()
            step = req.get("asked_after_step", "?")
            assumed = req.get("assumed_response", "?")
            st.markdown(
                f'<div class="evidence-item">'
                f'<b style="color:#7dd3fc;">#{i}</b> &nbsp;'
                f'<b>{req_type}</b> — asked after step <code>{step}</code>'
                f' · assumed: <code>{assumed}</code>'
                f'</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="evidence-item">📋 {req}</div>', unsafe_allow_html=True)


# ── Load ──────────────────────────────────────────────────────────────────────
cases = load_cases()
if not cases:
    st.error("⚠️ No case files found in `cases/`. Run `fraud_agent.py` first.")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="tigergraph-logo">🔍 FraudSight</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#64748b;font-size:0.75rem;margin-bottom:16px;">TigerGraph · Agentic Investigation</div>', unsafe_allow_html=True)
    st.divider()

    st.markdown("**📂 Case Explorer**")
    selected_id = st.selectbox("Select Case ID", list(cases.keys()), label_visibility="collapsed")

    st.divider()
    total = len(cases)
    frauds  = sum(1 for c in cases.values() if c.get("case",{}).get("verdict","").lower() == "fraud")
    legit   = sum(1 for c in cases.values() if c.get("case",{}).get("verdict","").lower() == "legitimate")
    sars    = sum(1 for c in cases.values() if c.get("sar",{}).get("file") is True)
    uncertain = total - frauds - legit

    st.markdown("**📊 Portfolio Overview**")
    st.metric("Total Cases", total)
    ca, cb = st.columns(2)
    ca.metric("🔴 Fraud", frauds)
    cb.metric("🟢 Legit", legit)
    cc, cd = st.columns(2)
    cc.metric("🟡 Uncertain", uncertain)
    cd.metric("📋 SAR Filed", sars)
    st.divider()
    st.caption("mistral-nemo · TigerGraph MCP · LangGraph")

# ── Main ──────────────────────────────────────────────────────────────────────
data = cases[selected_id]
case = data.get("case", {})
prob_raw = normalise_probability(case.get("fraud_probability", 0))
prob_pct = prob_raw * 100

# Header
col_title, col_prob = st.columns([3, 1])
with col_title:
    st.markdown(f"## 🗂️ Case `{selected_id}`")
    st.markdown(f'<div style="margin-top:-8px;">{verdict_badge(case.get("verdict","unknown"))}</div>', unsafe_allow_html=True)
with col_prob:
    st.markdown("<br>", unsafe_allow_html=True)
    prob_color = "#f87171" if prob_pct >= 70 else "#fbbf24" if prob_pct >= 40 else "#4ade80"
    st.markdown(
        f'<div style="text-align:right;">'
        f'<div style="color:#94a3b8;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;">Fraud Probability</div>'
        f'<div style="font-size:2.2rem;font-weight:800;color:{prob_color};">{prob_pct:.0f}%</div>'
        f'</div>', unsafe_allow_html=True)

st.divider()

# Metrics row
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("💰 Exposure (USD)", f"${case.get('exposure_usd', 0):,.2f}")
m2.metric("📌 Status", case.get("status", "—").upper())
m3.metric("🔎 Pattern", case.get("pattern", "—").replace("_", " ").title())
m4.metric("🛠️ Tool Calls", data.get("tool_calls", 0))
m5.metric("⏱️ Latency", f"{data.get('latency_s', 0):.1f}s")

st.markdown("<br>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["🔬 Investigation Evidence", "⚡ Next Best Actions", "📋 SAR Details"])

# ── Tab 1: Evidence ───────────────────────────────────────────────────────────
with tab1:
    st.markdown("### 🔬 Investigation Evidence")
    col_ev, col_txn = st.columns([3, 2])

    with col_ev:
        st.markdown("**Evidence Collected**")
        for item in case.get("evidence", []) or ["No evidence recorded."]:
            st.markdown(f'<div class="evidence-item">🔹 {item}</div>', unsafe_allow_html=True)
        st.markdown("<br>**📬 Evidence Requests Made**", unsafe_allow_html=True)
        render_evidence_requests(data.get("evidence_requests", []))

    with col_txn:
        st.markdown("**🏷️ Affected Transaction IDs**")
        txn_ids = case.get("affected_txn_ids", [])
        if txn_ids:
            for txn in txn_ids:
                st.markdown(
                    f'<div style="background:rgba(56,189,248,0.08);border:1px solid rgba(56,189,248,0.2);'
                    f'border-radius:8px;padding:10px 16px;margin:6px 0;font-family:monospace;'
                    f'font-size:1rem;color:#7dd3fc;">TXN # {txn}</div>', unsafe_allow_html=True)
        else:
            st.caption("No affected transactions identified.")

        st.markdown("<br>**🕵️ Similar Prior Cases**", unsafe_allow_html=True)
        prior = case.get("similar_prior_cases", [])
        if prior:
            chips = "".join([f'<span class="prior-case-chip">⚠️ {p}</span>' for p in prior])
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.caption("No similar prior cases found.")

        st.markdown("<br>**🛑 Stop Reason**", unsafe_allow_html=True)
        st.markdown(
            f'<div style="background:rgba(148,163,184,0.08);border-radius:8px;padding:10px 14px;'
            f'color:#94a3b8;font-size:0.88rem;">{data.get("stop_reason","None")}</div>', unsafe_allow_html=True)

# ── Tab 2: Actions ────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### ⚡ Next Best Actions")
    nba = data.get("next_best_actions", {})
    ci, cf = st.columns(2)

    with ci:
        st.markdown("#### 🟡 Initial Actions")
        st.markdown('<div style="color:#64748b;font-size:0.78rem;margin-bottom:8px;">Before evidence gathered</div>', unsafe_allow_html=True)
        render_actions(nba.get("initial", []))

    with cf:
        st.markdown("#### 🟢 Final Actions")
        st.markdown('<div style="color:#64748b;font-size:0.78rem;margin-bottom:8px;">After full investigation</div>', unsafe_allow_html=True)
        render_actions(nba.get("final", []))

    what_changed = nba.get("what_changed", "")
    if what_changed:
        st.markdown(
            f'<div class="what-changed-box">'
            f'<div style="color:#fbbf24;font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">💡 What Changed</div>'
            f'<div style="color:#e2e8f0;font-size:0.92rem;">{what_changed}</div>'
            f'</div>', unsafe_allow_html=True)

# ── Tab 3: SAR ────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 📋 Suspicious Activity Report")
    sar = data.get("sar", {})

    if sar.get("file"):
        st.markdown(
            '<div style="background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.3);border-radius:10px;padding:20px;">',
            unsafe_allow_html=True)
        st.markdown("#### 🚨 SAR Filed — Regulatory Action Required")
        sc1, sc2 = st.columns(2)
        sc1.metric("Total Amount", f"${sar.get('total_amount_usd', 0):,.2f}")
        sc2.metric("Subjects", ", ".join(sar.get("subjects", [])))
        st.markdown("**Regulatory Narrative:**")
        st.markdown(
            f'<div style="background:rgba(239,68,68,0.05);border-radius:8px;padding:14px;'
            f'color:#fca5a5;font-size:0.9rem;line-height:1.7;">{sar.get("narrative","")}</div>',
            unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.3);border-radius:10px;padding:20px;">',
            unsafe_allow_html=True)
        st.markdown("#### ✅ No SAR Filing Required")
        st.markdown(
            f'<div style="color:#86efac;font-size:0.9rem;margin-bottom:12px;">{sar.get("reason","No reason provided.")}</div>',
            unsafe_allow_html=True)
        subjects = sar.get("subjects", [])
        narrative = sar.get("narrative", "")
        if subjects:
            st.markdown(f'**Subjects of Interest:** {", ".join(subjects)}')
        if narrative:
            st.markdown("**Investigation Narrative:**")
            st.markdown(
                f'<div style="background:rgba(34,197,94,0.04);border-radius:8px;padding:14px;'
                f'color:#cbd5e1;font-size:0.88rem;line-height:1.7;">{narrative}</div>',
                unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.divider()
st.markdown(
    '<div style="text-align:center;color:#334155;font-size:0.75rem;">'
    'FraudSight · TigerGraph Agentic Fraud Investigation · Hacker House Goa 2026 · mistral-nemo + LangGraph + TigerGraph MCP'
    '</div>', unsafe_allow_html=True)
