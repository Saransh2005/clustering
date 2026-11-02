# -*- coding: utf-8 -*-
"""
Banking Transaction Network Analysis
====================================
Name      : Saransh Singh
Reg. No.  : 12301472
Roll No.  : 35
Section   : RK2P23FHB36
Dataset   : PaySim - Synthetic Financial Datasets For Fraud Detection
            https://www.kaggle.com/datasets/ealaxi/paysim1

Tasks
  1. Load a dataset of bank transactions between customers.
  2. Construct a directed graph (nodes = customers, edges = money transfers).
  3. Compute the total number of transactions per customer.
  4. Identify the most active customer.
  5. Visualize the network with high-value transactions highlighted.
"""

import json
import os
import time

import matplotlib
matplotlib.use("Agg")                      # draw to files (no display needed)
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
CSV_PATH = "PS_20174392719_1491204439457_log.csv" if os.path.exists("PS_20174392719_1491204439457_log.csv") else "PS_20174392719_1491204439457_log_1.csv"
OUT_DIR = "outputs"
SAMPLE_ROWS = None            # use e.g. 1_000_000 on a low-RAM PC; None = all rows
HIGH_VALUE_PERCENTILE = 0.95  # top 5 % of transfer amounts are "high-value"
TOP_N = 10                    # most active customers drawn in the network plot

RED, GOLD, BLUE, GREY = "#e03131", "#f59f00", "#1c7ed6", "#adb5bd"
os.makedirs(OUT_DIR, exist_ok=True)


def money(x):
    """Format a number as 1.23B / 4.56M / 789.0K for chart labels."""
    if x >= 1e9:
        return f"{x / 1e9:,.2f}B"
    if x >= 1e6:
        return f"{x / 1e6:,.2f}M"
    if x >= 1e3:
        return f"{x / 1e3:,.1f}K"
    return f"{x:,.0f}"


def banner(text):
    print("\n" + "=" * 70 + f"\n {text}\n" + "=" * 70)


# ----------------------------------------------------------------------------
# Step 1 - Load the dataset
# ----------------------------------------------------------------------------
def load_dataset(path, nrows=None):
    cols = ["step", "type", "amount", "nameOrig", "nameDest", "isFraud"]
    dtypes = {"step": "int16", "type": "category",
              "amount": "float64", "isFraud": "int8"}
    return pd.read_csv(path, usecols=cols, dtype=dtypes, nrows=nrows)


# ----------------------------------------------------------------------------
# Step 2 - Build the directed graph
# ----------------------------------------------------------------------------
def build_graph(transfers):
    """Nodes = customers, directed edge = one money transfer (sender -> receiver).
    MultiDiGraph keeps every transfer, even repeated ones between the same pair."""
    edge_attrs = ["amount", "step"]
    if "isFraud" in transfers.columns:
        edge_attrs.append("isFraud")
    return nx.from_pandas_edgelist(
        transfers, source="nameOrig", target="nameDest",
        edge_attr=edge_attrs, create_using=nx.MultiDiGraph)


# ----------------------------------------------------------------------------
# Step 3 - Transactions per customer
# ----------------------------------------------------------------------------
def transactions_per_customer(G):
    sent = dict(G.out_degree())       # transfers made by the customer
    received = dict(G.in_degree())    # transfers received by the customer
    table = pd.DataFrame({"customer": list(G.nodes())})
    table["sent"] = table["customer"].map(sent)
    table["received"] = table["customer"].map(received)
    table["total_transactions"] = table["sent"] + table["received"]
    return (table.sort_values(["total_transactions", "customer"],
                              ascending=[False, True])
                 .reset_index(drop=True))


# ----------------------------------------------------------------------------
# Step 5 - Visualisation helpers
# ----------------------------------------------------------------------------
def simple_view(G, nodes, threshold):
    """Collapse parallel edges into one drawable DiGraph.
    An edge is 'high' if any transfer on that pair reaches the threshold."""
    D = nx.DiGraph()
    D.add_nodes_from(nodes)
    for u, v, d in G.subgraph(nodes).edges(data=True):
        if D.has_edge(u, v):
            D[u][v]["amount"] = max(D[u][v]["amount"], d["amount"])
            D[u][v]["n"] += 1
        else:
            D.add_edge(u, v, amount=d["amount"], n=1)
    for u, v in D.edges():
        D[u][v]["high"] = D[u][v]["amount"] >= threshold
    return D


def ring_layout(D, hubs):
    """Place the hubs evenly on a ring; their counter-parties are pulled towards
    them by the spring layout, so every hub forms its own readable cluster."""
    ring = {h: (1.25 * np.cos(2 * np.pi * i / len(hubs)),
                1.25 * np.sin(2 * np.pi * i / len(hubs))) for i, h in enumerate(hubs)}
    return nx.spring_layout(D, pos=ring, fixed=hubs, k=0.12, iterations=300, seed=42)


def draw_network(ax, G, table, threshold, top_n=TOP_N, labels=True, small=False):
    """Draw the top-N most active customers plus all of their neighbours."""
    hubs = list(table["customer"].head(top_n))
    nodes = set(hubs)
    for h in hubs:
        nodes.update(G.predecessors(h))
        nodes.update(G.successors(h))
    D = simple_view(G, nodes, threshold)
    pos = ring_layout(D, hubs)

    total = dict(zip(table["customer"], table["total_transactions"]))
    normal = [(u, v) for u, v, d in D.edges(data=True) if not d["high"]]
    high = [(u, v) for u, v, d in D.edges(data=True) if d["high"]]

    sizes = [(total[n] * (10 if small else 18)) if n in hubs else (6 if small else 14)
             for n in D.nodes()]
    colors = [GOLD if n == hubs[0] else BLUE if n in hubs else GREY for n in D.nodes()]

    nx.draw_networkx_edges(D, pos, edgelist=normal, ax=ax, edge_color="#868e96",
                           alpha=0.35, width=0.6, arrows=True,
                           arrowsize=6 if small else 8, node_size=sizes)
    nx.draw_networkx_edges(D, pos, edgelist=high, ax=ax, edge_color=RED,
                           alpha=0.95, width=1.1 if small else 1.9, arrows=True,
                           arrowsize=8 if small else 13, node_size=sizes)
    nx.draw_networkx_nodes(D, pos, ax=ax, node_size=sizes, node_color=colors,
                           edgecolors="black", linewidths=0.4)
    if labels:
        for h in hubs:                       # label on the empty side, towards the centre
            x, y = pos[h]
            ax.text(x * 0.78, y * 0.78, h, ha="center", va="center", fontsize=8,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    ax.set_axis_off()
    return D, len(high), len(normal)


def network_legend(ax, threshold):
    handles = [
        Line2D([0], [0], color=RED, lw=2.2,
               label=f"High-value transfer (amount >= {money(threshold)})"),
        Line2D([0], [0], color="#868e96", lw=1, label="Regular transfer"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=GOLD,
               markeredgecolor="black", markersize=11, label="Most active customer"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=BLUE,
               markeredgecolor="black", markersize=9, label="Other top customers"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=GREY,
               markeredgecolor="black", markersize=6, label="Other customers"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0),
              ncol=3, fontsize=9, frameon=False)


def plot_network(G, table, threshold, path):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    D, n_high, n_norm = draw_network(ax, G, table, threshold)
    network_legend(ax, threshold)
    ax.set_title(f"Banking Transaction Network - top {TOP_N} most active customers "
                 f"and their counter-parties\n({D.number_of_nodes()} customers, "
                 f"{D.number_of_edges()} connections, {n_high} high-value)",
                 fontsize=12, fontweight="bold")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return D, n_high


def plot_ego(G, customer, threshold, path):
    """Everything the most active customer sent (right) and received (left)."""
    D = simple_view(G, set(G.predecessors(customer)) | set(G.successors(customer))
                    | {customer}, threshold)
    senders = sorted(D.predecessors(customer), key=lambda n: -D[n][customer]["amount"])
    receivers = sorted(D.successors(customer), key=lambda n: -D[customer][n]["amount"])
    if not receivers:                      # customer only receives money
        hub_x, xlim = 0.6, (-1.6, 1.0)
    elif not senders:                      # customer only sends money
        hub_x, xlim = -0.6, (-1.0, 1.6)
    else:
        hub_x, xlim = 0.0, (-1.5, 1.5)
    pos = {customer: (hub_x, 0)}
    for side, group in ((-1, senders), (1, receivers)):
        ys = np.linspace(1, -1, max(len(group), 2))
        for n, y in zip(group, ys):
            pos[n] = (side, y)

    fig, ax = plt.subplots(figsize=(11, 8))
    amounts = [d["amount"] for _, _, d in D.edges(data=True)]
    widths = {e: 0.6 + 4.0 * np.log10(1 + D.edges[e]["amount"]) / np.log10(1 + max(amounts))
              for e in D.edges()}
    for e in D.edges():
        hv = D.edges[e]["high"]
        nx.draw_networkx_edges(D, pos, edgelist=[e], ax=ax, width=widths[e],
                               edge_color=RED if hv else "#868e96",
                               alpha=0.95 if hv else 0.45, arrows=True, arrowsize=12,
                               node_size=300, connectionstyle="arc3,rad=0.05")
    nx.draw_networkx_nodes(D, pos, ax=ax, node_size=[1300 if n == customer else 150
                                                     for n in D.nodes()],
                           node_color=[GOLD if n == customer else GREY for n in D.nodes()],
                           edgecolors="black", linewidths=0.6)
    for n, (x, y) in pos.items():
        if n == customer:
            ax.text(x, y - 0.18, customer, ha="center", fontsize=10, fontweight="bold")
        else:
            ax.text(x + (-0.04 if x < 0 else 0.04), y, n, fontsize=7,
                    ha="right" if x < 0 else "left", va="center")
    for u, v, d in D.edges(data=True):     # amount tag beside each high-value edge
        if d["high"]:
            other = u if v == customer else v
            x, y = pos[other]
            ax.text(x + (0.28 if x < 0 else -0.28), y, money(d["amount"]), color=RED,
                    fontsize=9, fontweight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=RED, lw=0.8))
    if senders:
        ax.text(-1, 1.18, f"RECEIVED FROM ({len(senders)})", ha="center",
                fontweight="bold", fontsize=10)
    if receivers:
        ax.text(1, 1.18, f"SENT TO ({len(receivers)})", ha="center",
                fontweight="bold", fontsize=10)
    ax.set_xlim(*xlim)
    ax.set_ylim(-1.25, 1.3)
    ax.set_axis_off()
    network_legend(ax, threshold)
    ax.set_title(f"Transaction neighbourhood of the most active customer ({customer})\n"
                 "edge thickness = transfer amount", fontsize=12, fontweight="bold")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_dashboard(G, table, transfers, threshold, summary, path):
    with plt.rc_context({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11}):
        _dashboard(G, table, transfers, threshold, summary, path)


def _dashboard(G, table, transfers, threshold, summary, path):
    fig = plt.figure(figsize=(16, 10), facecolor="#f8f9fa")
    gs = fig.add_gridspec(3, 3, height_ratios=[0.42, 1.5, 1.5],
                          hspace=0.55, wspace=0.24, left=0.055, right=0.975,
                          top=0.90, bottom=0.06)
    fig.suptitle("BANKING TRANSACTION NETWORK - DASHBOARD", fontsize=21,
                 fontweight="bold", y=0.975)

    # ---- KPI cards ---------------------------------------------------------
    kpis = [("Customers (nodes)", f"{summary['customers']:,}", BLUE),
            ("Transfers (edges)", f"{summary['transfers']:,}", BLUE),
            ("Total value moved", money(summary["total_value"]), BLUE),
            (f"High-value transfers (>= {money(threshold)})",
             f"{summary['high_value_transfers']:,}", RED),
            ("Most active customer",
             f"{summary['most_active']}\n{summary['most_active_total']} transactions", GOLD)]
    inner = gs[0, :].subgridspec(1, 5, wspace=0.08)
    for i, (label, value, color) in enumerate(kpis):
        ax = fig.add_subplot(inner[0, i])
        ax.set_facecolor("white")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(color); s.set_linewidth(2)
        ax.text(0.5, 0.64, value, ha="center", va="center", transform=ax.transAxes,
                fontsize=18 if "\n" not in value else 13, fontweight="bold", color=color)
        ax.text(0.5, 0.17, label, ha="center", va="center", transform=ax.transAxes,
                fontsize=9.5, color="#495057")

    # ---- 1. distribution of transactions per customer ----------------------
    ax = fig.add_subplot(gs[1, 0])
    dist = table["total_transactions"].value_counts().sort_index()
    ax.bar(dist.index, dist.values, color=BLUE)
    ax.set_yscale("log")
    ax.set_title("Transactions per customer (distribution)", fontweight="bold")
    ax.set_xlabel("Total transactions (sent + received)")
    ax.set_ylabel("Number of customers (log scale)")

    # ---- 2. top-10 most active customers -----------------------------------
    ax = fig.add_subplot(gs[1, 1])
    t10 = table.head(10).iloc[::-1]
    ax.barh(t10["customer"], t10["sent"], color=BLUE, label="Sent")
    ax.barh(t10["customer"], t10["received"], left=t10["sent"], color="#20c997",
            label="Received")
    for i, tot in enumerate(t10["total_transactions"]):
        ax.text(tot + 0.3, i, str(tot), va="center", fontsize=9, fontweight="bold")
    ax.set_title("Top 10 most active customers", fontweight="bold")
    ax.set_xlabel("Number of transactions")
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xlim(0, 39)                       # leave room for the legend
    ax.legend(loc="lower right")

    # ---- 3. transfer-amount distribution -----------------------------------
    ax = fig.add_subplot(gs[1, 2])
    bins = np.logspace(0, np.log10(transfers["amount"].max()), 50)
    ax.hist(transfers["amount"].clip(lower=1), bins=bins, color="#74c0fc")
    ax.hist(transfers.loc[transfers["amount"] >= threshold, "amount"], bins=bins,
            color=RED)
    ax.axvline(threshold, color="black", ls="--", lw=1.2)
    ax.set_xscale("log")
    ax.set_title("Transfer amount distribution", fontweight="bold")
    ax.set_xlabel("Amount (log scale)")
    ax.set_ylabel("Number of transfers")
    ax.legend(handles=[Line2D([0], [0], color="#74c0fc", lw=6, label="Regular"),
                       Line2D([0], [0], color=RED, lw=6, label="High-value"),
                       Line2D([0], [0], color="black", ls="--",
                              label=f"Threshold {money(threshold)}")], fontsize=8)

    # ---- 4. transfers per day ----------------------------------------------
    ax = fig.add_subplot(gs[2, 0])
    day = (transfers["step"] - 1) // 24 + 1          # 1 step = 1 hour
    hv = transfers["amount"] >= threshold
    per_day = pd.DataFrame({"regular": day[~hv].value_counts(),
                            "high": day[hv].value_counts()}).fillna(0).sort_index()
    ax.bar(per_day.index, per_day["regular"], color="#74c0fc", label="Regular")
    ax.bar(per_day.index, per_day["high"], bottom=per_day["regular"], color=RED,
           label="High-value")
    ax.set_title("Transfers per day", fontweight="bold")
    ax.set_xlabel("Day of simulation")
    ax.set_ylabel("Number of transfers")
    ax.legend()

    # ---- 5. mini network ----------------------------------------------------
    ax = fig.add_subplot(gs[2, 1])
    draw_network(ax, G, table, threshold, top_n=TOP_N, labels=False, small=True)
    ax.set_title("Network of top customers (red = high-value)", fontweight="bold")

    # ---- 6. five largest transfers (table) ---------------------------------
    ax = fig.add_subplot(gs[2, 2])
    ax.set_axis_off()
    ax.set_title("Five largest transfers", fontweight="bold")
    big = transfers.nlargest(5, "amount")
    cells = [[r.nameOrig, r.nameDest, money(r.amount), f"{(r.step - 1) // 24 + 1}"]
             for r in big.itertuples()]
    tbl = ax.table(cellText=cells, colLabels=["Sender", "Receiver", "Amount", "Day"],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 2.1)
    for (r, _), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#343a40")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#ffe3e3")

    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main program
# ----------------------------------------------------------------------------
def main():
    t0 = time.time()
    banner("BANKING TRANSACTION NETWORK ANALYSIS")

    # ---- Step 1 ------------------------------------------------------------
    print("[Step 1] Loading the dataset ...")
    df = load_dataset(CSV_PATH, SAMPLE_ROWS)
    rows_loaded = len(df)
    print(f"  File                  : {CSV_PATH}")
    print(f"  Rows loaded           : {rows_loaded:,}")
    print("  Transactions by type  :")
    for t, c in df["type"].value_counts().items():
        print(f"      {t:<10} {c:>10,}")
    transfers = df[df["type"] == "TRANSFER"].reset_index(drop=True)
    print(f"  Money transfers used for the graph (type = TRANSFER): {len(transfers):,}")

    # ---- Step 2 ------------------------------------------------------------
    print("\n[Step 2] Constructing the directed graph ...")
    G = build_graph(transfers)
    print(f"  Graph type            : {type(G).__name__} (directed, parallel edges allowed)")
    print(f"  Nodes (customers)     : {G.number_of_nodes():,}")
    print(f"  Edges (transfers)     : {G.number_of_edges():,}")
    print(f"  Self-loops            : {nx.number_of_selfloops(G):,}")

    # ---- Step 3 ------------------------------------------------------------
    print("\n[Step 3] Total number of transactions per customer (sent + received)")
    table = transactions_per_customer(G)
    table.to_csv(os.path.join(OUT_DIR, "transactions_per_customer.csv"), index=False)
    print(table.head(10).to_string(index=False))
    once = int((table["total_transactions"] == 1).sum())
    print(f"  Customers with exactly 1 transaction : {once:,} "
          f"({once / len(table):.1%} of all customers)")
    print(f"  Average transactions per customer    : "
          f"{table['total_transactions'].mean():.2f}")

    # ---- Step 4 ------------------------------------------------------------
    banner("[Step 4] MOST ACTIVE CUSTOMER")
    best = table["total_transactions"].max()
    leaders = table[table["total_transactions"] == best]
    top = leaders.iloc[0]
    sent_value = transfers.loc[transfers["nameOrig"] == top.customer, "amount"].sum()
    recv_value = transfers.loc[transfers["nameDest"] == top.customer, "amount"].sum()
    print(f"  Customer ID           : {top.customer}")
    print(f"  Total transactions    : {top.total_transactions}")
    print(f"  Transfers sent        : {top.sent}   (value {sent_value:,.2f})")
    print(f"  Transfers received    : {top.received}   (value {recv_value:,.2f})")
    if len(leaders) > 1:
        print(f"  (tie: {len(leaders)} customers share {best} transactions)")

    # Extra check: count every transaction type (merchants 'M...' are not customers)
    all_cnt = df["nameOrig"].value_counts().add(df["nameDest"].value_counts(),
                                                fill_value=0)
    all_cnt = all_cnt[all_cnt.index.str.startswith("C")].sort_values(ascending=False)
    print("\n  Extended check - counting ALL five transaction types:")
    print(f"      most active customer = {all_cnt.index[0]} "
          f"with {int(all_cnt.iloc[0])} transactions")
    ext_id, ext_total = all_cnt.index[0], int(all_cnt.iloc[0])
    del df, all_cnt                      # free memory - not needed any more

    # ---- Step 5 ------------------------------------------------------------
    banner("[Step 5] HIGH-VALUE TRANSACTIONS AND VISUALISATION")
    threshold = float(transfers["amount"].quantile(HIGH_VALUE_PERCENTILE))
    n_high = int((transfers["amount"] >= threshold).sum())
    print(f"  High-value threshold  : {threshold:,.2f} "
          f"(top {1 - HIGH_VALUE_PERCENTILE:.0%} of transfer amounts)")
    print(f"  High-value transfers  : {n_high:,}")
    print(f"  Largest transfer      : {transfers['amount'].max():,.2f}")

    summary = {
        "rows_loaded": int(rows_loaded),
        "transfers": int(len(transfers)),
        "customers": int(G.number_of_nodes()),
        "total_value": float(transfers["amount"].sum()),
        "threshold": threshold,
        "high_value_transfers": n_high,
        "most_active": top.customer,
        "most_active_total": int(top.total_transactions),
        "most_active_sent": int(top.sent),
        "most_active_received": int(top.received),
        "once_customers": once,
        "extended_most_active": ext_id,
        "extended_most_active_total": ext_total,
    }

    D, n_high_drawn = plot_network(G, table, threshold,
                                   os.path.join(OUT_DIR, "network_graph.png"))
    print(f"  Saved network_graph.png  ({D.number_of_nodes()} customers drawn, "
          f"{n_high_drawn} high-value connections in red)")
    plot_ego(G, top.customer, threshold, os.path.join(OUT_DIR, "most_active_customer.png"))
    print("  Saved most_active_customer.png")
    plot_dashboard(G, table, transfers, threshold, summary,
                   os.path.join(OUT_DIR, "dashboard.png"))
    print("  Saved dashboard.png")

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAll files saved in '{OUT_DIR}/'.  Finished in {time.time() - t0:.0f} s.")


if __name__ == "__main__":
    main()
