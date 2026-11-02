# -*- coding: utf-8 -*-
"""
FinNet - Single-Screen Banking Transaction Network Dashboard
============================================================
Author      : Saransh Singh
Reg. No.    : 12301472
Roll No.    : 35
Section     : RK2P23FHB36
Institution : Lovely Professional University (LPU)
Dataset     : PaySim Synthetic Financial Dataset

Architecture:
- 100% Single-Screen Cockpit (Zero Page Scrolling)
- Top Bar: Title + Compact Inline Controls + One-Click Export Button
- Upper Window: Large Interactive Directed Graph (PyVis Force-Directed)
- Lower Window: 4 Symmetrical Analytical Cards (KPIs, Most Active, Leaderboard, Top Transfers)
"""

import os
import json
import pandas as pd
import numpy as np
import networkx as nx
import streamlit as st
import streamlit.components.v1 as components

import visualizations as vz
import banking_transaction_network as btn

# ----------------------------------------------------------------------------
# Page Config & Viewport Fit CSS
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="FinNet • Banking Network Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Utility: guarantees HTML is never treated as markdown code-block
def render_html(html_str: str):
    clean = "\n".join(line.strip() for line in html_str.strip().splitlines() if line.strip())
    st.markdown(clean, unsafe_allow_html=True)


# Strict styling to eliminate all Streamlit paddings and lock to 1 viewport
render_html("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
    }

    /* Remove Streamlit default header, footer, and padding */
    header[data-testid="stHeader"] { display: none !important; }
    footer { display: none !important; }
    #MainMenu { visibility: hidden; }
    
    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 0.2rem !important;
        padding-left: 1.2rem !important;
        padding-right: 1.2rem !important;
        max-width: 100% !important;
    }

    /* Top Bar */
    .top-badge {
        font-size: 10.5px;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 6px;
        background: #1e1b4b;
        color: #a5b4fc;
        border: 1px solid #3730a3;
        white-space: nowrap;
    }

    /* Graph Container Box */
    .graph-box {
        background: #090d16;
        border: 1px solid #1e293b;
        border-radius: 8px 8px 0 0;
        overflow: hidden;
    }
    .graph-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #0f172a;
        padding: 6px 14px;
        border-bottom: 1px solid #1e293b;
        font-size: 12px;
        font-weight: 700;
        color: #cbd5e1;
    }

    /* Bottom Cards */
    .compact-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 9px 12px;
        height: 215px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.25);
    }
    .compact-card-title {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #94a3b8;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 4px;
        border-bottom: 1px solid #1e293b;
        padding-bottom: 3px;
    }
    
    .metric-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 2.5px 0;
        border-bottom: 1px dashed rgba(30, 41, 59, 0.7);
        font-size: 11.5px;
    }
    .metric-row:last-child {
        border-bottom: none;
    }
    .metric-row-label {
        color: #94a3b8;
    }
    .metric-row-val {
        font-weight: 700;
        color: #f1f5f9;
    }

    /* Minimalist Selectbox & Button adjustments */
    div[data-baseweb="select"] > div {
        min-height: 28px !important;
        height: 28px !important;
        font-size: 11.5px !important;
        border-radius: 6px !important;
        background-color: #0f172a !important;
        border-color: #1e293b !important;
    }
    
    button[data-testid="baseButton-secondary"] {
        min-height: 28px !important;
        height: 28px !important;
        padding: 0 10px !important;
        font-size: 11.5px !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
        background-color: #1e1b4b !important;
        border-color: #3730a3 !important;
        color: #a5b4fc !important;
    }
    button[data-testid="baseButton-secondary"]:hover {
        background-color: #2e1065 !important;
        color: #ffffff !important;
        border-color: #6366f1 !important;
    }
</style>
""")


# ----------------------------------------------------------------------------
# Data Loading & Processing (Cached)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(nrows=200000):
    candidates = [
        "PS_20174392719_1491204439457_log.csv",
        "PS_20174392719_1491204439457_log_1.csv",
        "sample_transactions.csv"
    ]
    csv_path = next((p for p in candidates if os.path.exists(p)), None)
    if not csv_path:
        st.error("❌ Dataset not found. Please provide a CSV file.")
        st.stop()

    cols = ["step", "type", "amount", "nameOrig", "nameDest", "isFraud"]
    dtypes = {"step": "int16", "type": "category", "amount": "float64", "isFraud": "int8"}
    df = pd.read_csv(csv_path, usecols=cols, dtype=dtypes, nrows=nrows if nrows > 0 else None)
    transfers = df[df["type"] == "TRANSFER"].reset_index(drop=True)
    return df, transfers


@st.cache_data(show_spinner=False)
def build_network(transfers_df):
    G = nx.from_pandas_edgelist(
        transfers_df,
        source="nameOrig",
        target="nameDest",
        edge_attr=["amount", "step", "isFraud"],
        create_using=nx.MultiDiGraph
    )
    sent = dict(G.out_degree())
    received = dict(G.in_degree())

    tbl = pd.DataFrame({"customer": list(G.nodes())})
    tbl["sent"] = tbl["customer"].map(sent).fillna(0).astype(int)
    tbl["received"] = tbl["customer"].map(received).fillna(0).astype(int)
    tbl["total"] = tbl["sent"] + tbl["received"]
    tbl["customer_id"] = tbl["customer"]
    tbl["outgoing_tx_count"] = tbl["sent"]
    tbl["incoming_tx_count"] = tbl["received"]
    tbl = tbl.sort_values(["total", "customer"], ascending=[False, True]).reset_index(drop=True)
    return G, tbl


def format_money(val: float) -> str:
    if val >= 1e9: return f"${val/1e9:,.2f}B"
    if val >= 1e6: return f"${val/1e6:,.2f}M"
    if val >= 1e3: return f"${val/1e3:,.1f}K"
    return f"${val:,.0f}"


# ----------------------------------------------------------------------------
# Top Navigation Bar with Inline Compact Controls
# ----------------------------------------------------------------------------
c_title, c_hubs, c_percentile, c_sample, c_export = st.columns([3.8, 1.3, 1.8, 1.6, 1.5])

with c_title:
    render_html("""
    <div style="display: flex; align-items: center; gap: 8px; height: 28px;">
        <span style="font-size: 16px; font-weight: 800; color: #38BDF8;">🏦 FinNet</span>
        <span style="font-size: 12px; font-weight: 600; color: #94A3B8;">Banking Transaction Network</span>
        <span class="top-badge">Saransh Singh • 12301472 • LPU</span>
    </div>
    """)

with c_hubs:
    top_hubs_count = st.selectbox("Hubs", [5, 8, 10, 15], index=2, label_visibility="collapsed")

with c_percentile:
    percentile = st.selectbox(
        "Percentile",
        [(90.0, "Top 10% Cutoff"), (95.0, "Top 5% Cutoff (95%)"), (99.0, "Top 1% Cutoff")],
        index=1,
        format_func=lambda x: x[1],
        label_visibility="collapsed"
    )[0]

with c_sample:
    sample_choice = st.selectbox(
        "Sample",
        [(100000, "100k Rows (Fast)"), (200000, "200k Rows (Balanced)"), (500000, "500k Rows (Deep)")],
        index=1,
        format_func=lambda x: x[1],
        label_visibility="collapsed"
    )[0]

# Load Data
raw_df, transfers_df = load_data(nrows=sample_choice)
G, customer_table = build_network(transfers_df)

# Threshold & Top Customer computations
threshold = float(transfers_df["amount"].quantile(percentile / 100.0))
n_hv = int((transfers_df["amount"] >= threshold).sum())
total_volume = float(transfers_df["amount"].sum())

top_customer = customer_table.iloc[0]
top_id = top_customer["customer"]
top_tot = int(top_customer["total"])
top_sent = int(top_customer["sent"])
top_recv = int(top_customer["received"])
top_recv_val = float(transfers_df.loc[transfers_df["nameDest"] == top_id, "amount"].sum())
top_sent_val = float(transfers_df.loc[transfers_df["nameOrig"] == top_id, "amount"].sum())

with c_export:
    if st.button("⚡ Export PNGs", width="stretch"):
        out_dir = "outputs"
        os.makedirs(out_dir, exist_ok=True)
        summary = {
            "customers": int(G.number_of_nodes()),
            "transfers": int(G.number_of_edges()),
            "total_value": float(total_volume),
            "threshold": float(threshold),
            "high_value_transfers": int(n_hv),
            "most_active": str(top_id),
            "most_active_total": int(top_tot)
        }
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        customer_table.to_csv(os.path.join(out_dir, "transactions_per_customer.csv"), index=False)
        btn.plot_network(G, customer_table, threshold, os.path.join(out_dir, "network_graph.png"))
        btn.plot_ego(G, top_id, threshold, os.path.join(out_dir, "most_active_customer.png"))
        btn.plot_dashboard(G, customer_table, transfers_df, threshold, summary, os.path.join(out_dir, "dashboard.png"))
        st.toast("✅ Report PNGs saved to outputs/!")


# ----------------------------------------------------------------------------
# 1. UPPER SECTION: NETWORK GRAPH WINDOW (TOP HALF OF SCREEN)
# ----------------------------------------------------------------------------
top_hubs = list(customer_table["customer"].head(top_hubs_count))
subgraph = vz.build_interactive_subgraph(G, hubs=top_hubs, threshold=threshold, max_neighbors_per_hub=25)

sub_nodes = subgraph.number_of_nodes()
sub_edges = subgraph.number_of_edges()
sub_hv = sum(1 for _, _, d in subgraph.edges(data=True) if d.get("is_high_value", False))

render_html(f"""
<div class="graph-box">
    <div class="graph-header">
        <span>🌐 <b>DIRECTED TRANSACTION GRAPH</b> ({sub_nodes} Accounts, {sub_edges} Transfers)</span>
        <span style="font-size: 11px;">
            <b style="color: #F59E0B;">👑 Most Active:</b> {top_id} &nbsp;|&nbsp; 
            <b style="color: #EF4444;">🔴 High-Value Link:</b> ≥ {format_money(threshold)} ({sub_hv} in view) &nbsp;|&nbsp; 
            <b style="color: #64748B;">⚪ Regular Transfer:</b> Slate
        </span>
    </div>
</div>
""")

# Generate PyVis HTML with compact 370px height so everything fits cleanly
pyvis_html = vz.generate_pyvis_network_html(
    subgraph,
    high_value_threshold=threshold,
    most_active_customer_id=top_id,
    height="365px",
    physics_enabled=True,
    dark_mode=True
)

components.html(pyvis_html, height=370, scrolling=False)


# ----------------------------------------------------------------------------
# 2. LOWER SECTION: ALL DATA & CARDS (BOTTOM HALF - 4 COLUMNS)
# ----------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns([1, 1, 1.1, 1.1])

# --- Card 1: Network KPIs ---
with col1:
    render_html(f"""
    <div class="compact-card" style="border-left: 3px solid #3B82F6;">
        <div>
            <div class="compact-card-title">📊 Key Metrics <span style="color:#3B82F6;">●</span></div>
            <div class="metric-row">
                <span class="metric-row-label">Total Customers (Nodes):</span>
                <span class="metric-row-val">{G.number_of_nodes():,}</span>
            </div>
            <div class="metric-row">
                <span class="metric-row-label">Total Transfers (Edges):</span>
                <span class="metric-row-val">{G.number_of_edges():,}</span>
            </div>
            <div class="metric-row">
                <span class="metric-row-label">Total Volume Moved:</span>
                <span class="metric-row-val" style="color: #10B981;">{format_money(total_volume)}</span>
            </div>
            <div class="metric-row">
                <span class="metric-row-label">High-Value Cutoff:</span>
                <span class="metric-row-val" style="color: #EF4444;">≥ {format_money(threshold)}</span>
            </div>
            <div class="metric-row">
                <span class="metric-row-label">High-Value Transfers:</span>
                <span class="metric-row-val">{n_hv:,}</span>
            </div>
        </div>
        <div style="font-size: 9.5px; color: #64748B; text-align: center; border-top: 1px solid #1e293b; padding-top: 2px;">
            Directed Graph (Nodes = Customers, Edges = Transfers)
        </div>
    </div>
    """)

# --- Card 2: Most Active Customer Spotlight ---
with col2:
    render_html(f"""
    <div class="compact-card" style="border-left: 3px solid #F59E0B;">
        <div>
            <div class="compact-card-title">👑 Most Active Customer <span style="color:#F59E0B;">#1 Rank</span></div>
            <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.25); border-radius: 5px; padding: 4px 6px; text-align: center; margin-bottom: 4px;">
                <div style="font-size: 13.5px; font-weight: 800; color: #FBBF24;">{top_id}</div>
                <div style="font-size: 10.5px; color: #CBD5E1;"><b>{top_tot} Total Transactions</b></div>
            </div>
            <div class="metric-row">
                <span class="metric-row-label">Incoming Transfers:</span>
                <span class="metric-row-val" style="color: #10B981;">{top_recv} ({format_money(top_recv_val)})</span>
            </div>
            <div class="metric-row">
                <span class="metric-row-label">Outgoing Transfers:</span>
                <span class="metric-row-val" style="color: #3B82F6;">{top_sent} ({format_money(top_sent_val)})</span>
            </div>
            <div class="metric-row">
                <span class="metric-row-label">Account Flow Role:</span>
                <span class="metric-row-val" style="color: #A855F7;">Beneficiary Hub</span>
            </div>
        </div>
        <div style="font-size: 9.5px; color: #64748B; text-align: center; border-top: 1px solid #1e293b; padding-top: 2px;">
            Spotlighted in Gold node in top network
        </div>
    </div>
    """)

# --- Card 3: Top Active Customers Table ---
with col3:
    table_rows = "".join([
        f"""<tr style="border-bottom: 1px solid #1e293b; font-size: 11px;">
            <td style="color: #64748B; font-weight: 700; padding: 3px 0;">#{idx+1}</td>
            <td style="color: #F8FAFC; font-weight: 600;">{r['customer']}</td>
            <td style="text-align: right; color: #3B82F6;">{r['sent']}</td>
            <td style="text-align: right; color: #10B981;">{r['received']}</td>
            <td style="text-align: right; color: #F59E0B; font-weight: 700;">{r['total']}</td>
        </tr>""" for idx, r in customer_table.head(5).iterrows()
    ])

    render_html(f"""
    <div class="compact-card" style="border-left: 3px solid #8B5CF6;">
        <div>
            <div class="compact-card-title">🏆 Top Active Customers <span style="color:#8B5CF6;">Ranked</span></div>
            <table style="width: 100%; border-collapse: collapse; margin-top: 2px;">
                <thead>
                    <tr style="border-bottom: 1px solid #334155; font-size: 10px; color: #94A3B8; text-transform: uppercase;">
                        <th style="text-align: left; padding-bottom: 2px;">Rank</th>
                        <th style="text-align: left;">Customer</th>
                        <th style="text-align: right;">Sent</th>
                        <th style="text-align: right;">Recv</th>
                        <th style="text-align: right;">Total</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>
        <div style="font-size: 9.5px; color: #64748B; text-align: center; border-top: 1px solid #1e293b; padding-top: 2px;">
            Sorted by Total Transactions (Sent + Received)
        </div>
    </div>
    """)

# --- Card 4: Top Largest Transfers Table ---
with col4:
    tx_rows = "".join([
        f"""<tr style="border-bottom: 1px solid #1e293b; font-size: 11px;">
            <td style="color: #94A3B8; font-family: monospace; font-size: 10px; padding: 3px 0;">{r['nameOrig'][:10]}</td>
            <td style="color: #94A3B8; font-family: monospace; font-size: 10px;">➔ {r['nameDest'][:10]}</td>
            <td style="text-align: right; color: #EF4444; font-weight: 700;">{format_money(r['amount'])}</td>
        </tr>""" for _, r in transfers_df.nlargest(5, "amount").iterrows()
    ])

    render_html(f"""
    <div class="compact-card" style="border-left: 3px solid #EF4444;">
        <div>
            <div class="compact-card-title">🔥 Largest Transfers <span style="color:#EF4444;">Top 5</span></div>
            <table style="width: 100%; border-collapse: collapse; margin-top: 2px;">
                <thead>
                    <tr style="border-bottom: 1px solid #334155; font-size: 10px; color: #94A3B8; text-transform: uppercase;">
                        <th style="text-align: left; padding-bottom: 2px;">Sender</th>
                        <th style="text-align: left;">Receiver</th>
                        <th style="text-align: right;">Amount</th>
                    </tr>
                </thead>
                <tbody>
                    {tx_rows}
                </tbody>
            </table>
        </div>
        <div style="font-size: 9.5px; color: #64748B; text-align: center; border-top: 1px solid #1e293b; padding-top: 2px;">
            High-value transfers highlighted in red in graph
        </div>
    </div>
    """)


# ----------------------------------------------------------------------------
# Bottom Compact Footer Bar
# ----------------------------------------------------------------------------
render_html("""
<div style="font-size: 10px; color: #64748B; text-align: center; margin-top: 4px; padding-top: 2px; border-top: 1px solid #1e293b;">
    Tasks Completed: <b>1.</b> Load Dataset &nbsp;|&nbsp; <b>2.</b> Construct Directed Graph &nbsp;|&nbsp; <b>3.</b> Transactions per Customer &nbsp;|&nbsp; <b>4.</b> Most Active Customer &nbsp;|&nbsp; <b>5.</b> High-Value Visual Highlight
</div>
""")
