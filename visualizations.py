"""
visualizations.py
Interactive visualizations for bank transaction networks and analytics.
Provides PyVis force-directed physics network, Plotly network graphs,
activity leaderboards, amount distributions, and customer ego networks.
"""

import json
import networkx as nx
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pyvis.network import Network
from typing import Optional, Dict, Any


def generate_pyvis_network_html(
    G: nx.DiGraph,
    high_value_threshold: float,
    most_active_customer_id: Optional[str] = None,
    height: str = "650px",
    physics_enabled: bool = True,
    dark_mode: bool = True
) -> str:
    """
    Generates an interactive force-directed directed network graph using PyVis with
    high-value transactions highlighted in bright crimson/gold and normal transactions in sleek slate.
    
    Args:
        G: NetworkX directed subgraph
        high_value_threshold: Cutoff amount for high-value transfers
        most_active_customer_id: ID of the #1 most active customer to spotlight
        height: Canvas height
        physics_enabled: Whether physics simulation is active
        dark_mode: Whether to style for dark theme
        
    Returns:
        str: Raw HTML code of the interactive network
    """
    bg_color = "#0F172A" if dark_mode else "#FFFFFF"
    font_color = "#F8FAFC" if dark_mode else "#1E293B"

    net = Network(
        height=height,
        width="100%",
        directed=True,
        notebook=False,
        bgcolor=bg_color,
        font_color=font_color,
        cdn_resources="in_line"
    )

    # Add Nodes
    for node in G.nodes():
        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)
        total_deg = in_deg + out_deg
        
        node_data = G.nodes[node]
        sent_amt = node_data.get('total_sent', 0.0)
        recv_amt = node_data.get('total_received', 0.0)

        # Style by role and activity
        is_most_active = (node == most_active_customer_id)
        
        if is_most_active:
            color = "#F59E0B"  # Bright Gold / Amber for #1 Most Active
            border_color = "#FEF08A"
            size = 28
            label = f"👑 {node} (Top Active)"
            shape = "dot"
        elif in_deg > 0 and out_deg > 0:
            color = "#8B5CF6"  # Intermediary Hub
            border_color = "#C4B5FD"
            size = min(30, max(14, 12 + total_deg * 2))
            label = str(node)
            shape = "dot"
        elif out_deg > 0:
            color = "#3B82F6"  # Payer / Sender
            border_color = "#93C5FD"
            size = min(26, max(12, 10 + out_deg * 2))
            label = str(node)
            shape = "dot"
        else:
            color = "#10B981"  # Beneficiary / Receiver
            border_color = "#6EE7B7"
            size = min(26, max(12, 10 + in_deg * 2))
            label = str(node)
            shape = "dot"

        tooltip = (
            f"<div style='font-family:sans-serif; padding:6px; min-width:180px;'>"
            f"<strong style='color:{color}; font-size:14px;'>Customer: {node}</strong><br/>"
            f"{'🌟 <b>MOST ACTIVE CUSTOMER</b><br/>' if is_most_active else ''}"
            f"<b>Total Transactions:</b> {total_deg}<br/>"
            f"<b>Outgoing Transfers:</b> {out_deg} (${sent_amt:,.2f})<br/>"
            f"<b>Incoming Transfers:</b> {in_deg} (${recv_amt:,.2f})<br/>"
            f"</div>"
        )

        net.add_node(
            node,
            label=label,
            title=tooltip,
            color={'background': color, 'border': border_color, 'highlight': {'background': '#FF0055', 'border': '#FFFFFF'}},
            size=size,
            shape=shape,
            borderWidth=3 if is_most_active else 1.5,
            font={'color': font_color, 'size': 13 if is_most_active else 11, 'face': 'Inter, system-ui, sans-serif'}
        )

    # Add Directed Edges with High-Value Highlighting
    for u, v, data in G.edges(data=True):
        weight = data.get('weight', 0.0)
        count = data.get('count', 1)
        tx_type = data.get('type', 'TRANSFER')
        is_fraud = data.get('is_fraud', 0) == 1
        is_high = data.get('is_high_value', False) or (weight >= high_value_threshold if high_value_threshold > 0 else False)

        if is_fraud:
            edge_color = "#DC2626"  # Crimson warning
            edge_width = 4.5
            title_prefix = "🚨 FRAUDULENT TRANSFER"
        elif is_high:
            edge_color = "#EF4444"  # Neon Red / Crimson highlight
            edge_width = 4.0
            title_prefix = "🔥 HIGH-VALUE TRANSFER"
        else:
            edge_color = "#64748B80" if dark_mode else "#94A3B8B0"  # Subtle slate
            edge_width = 1.5
            title_prefix = "Money Transfer"

        tooltip = (
            f"<div style='font-family:sans-serif; padding:5px;'>"
            f"<b style='color:{edge_color};'>{title_prefix}</b><br/>"
            f"<b>From:</b> {u} ➔ <b>To:</b> {v}<br/>"
            f"<b>Amount:</b> ${weight:,.2f}<br/>"
            f"<b>Tx Count:</b> {count}<br/>"
            f"<b>Type:</b> {tx_type}"
            f"</div>"
        )

        net.add_edge(
            u, v,
            title=tooltip,
            color=edge_color,
            width=edge_width,
            arrows={'to': {'enabled': True, 'scaleFactor': 0.8}},
            smooth={'type': 'curvedCW', 'roundness': 0.12}
        )

    # Configure physics engine
    options = {
        "physics": {
            "enabled": physics_enabled,
            "stabilization": {
                "enabled": True,
                "iterations": 150,
                "updateInterval": 25,
                "fit": True
            },
            "barnesHut": {
                "gravitationalConstant": -4500,
                "centralGravity": 0.25,
                "springLength": 100,
                "springConstant": 0.05,
                "damping": 0.14,
                "avoidOverlap": 0.45
            },
            "minVelocity": 0.75
        },
        "interaction": {
            "hover": True,
            "tooltipDelay": 80,
            "navigationButtons": False,
            "keyboard": False,
            "zoomView": True
        }
    }
    net.set_options(json.dumps(options))
    return net.generate_html()


def create_plotly_network_figure(
    G: nx.DiGraph,
    high_value_threshold: float,
    most_active_customer_id: Optional[str] = None
) -> go.Figure:
    """
    Creates an interactive 2D Plotly network graph with separate traces for normal
    and highlighted high-value transactions.
    """
    if G.number_of_nodes() == 0:
        fig = go.Figure()
        fig.add_annotation(text="No nodes to display in current filter", showarrow=False)
        return fig

    # Layout positioning
    pos = nx.spring_layout(G, seed=42, k=0.35, iterations=50)

    # Separate edges into Normal and High-Value for clean legend toggling
    normal_edge_x, normal_edge_y = [], []
    high_edge_x, high_edge_y = [], []
    fraud_edge_x, fraud_edge_y = [], []

    for u, v, d in G.edges(data=True):
        if u not in pos or v not in pos:
            continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        amt = d.get('weight', 0.0)
        is_fraud = d.get('is_fraud', 0) == 1
        is_high = d.get('is_high_value', False) or (amt >= high_value_threshold if high_value_threshold > 0 else False)

        if is_fraud:
            fraud_edge_x.extend([x0, x1, None])
            fraud_edge_y.extend([y0, y1, None])
        elif is_high:
            high_edge_x.extend([x0, x1, None])
            high_edge_y.extend([y0, y1, None])
        else:
            normal_edge_x.extend([x0, x1, None])
            normal_edge_y.extend([y0, y1, None])

    traces = []

    # Normal Edges Trace
    if normal_edge_x:
        traces.append(go.Scatter(
            x=normal_edge_x, y=normal_edge_y,
            line=dict(width=1.2, color='rgba(148, 163, 184, 0.45)'),
            hoverinfo='none',
            mode='lines',
            name='Normal Transfers'
        ))

    # High-Value Edges Trace (Highlighted)
    if high_edge_x:
        traces.append(go.Scatter(
            x=high_edge_x, y=high_edge_y,
            line=dict(width=3.5, color='#EF4444'),
            hoverinfo='none',
            mode='lines',
            name='🔥 High-Value Transfers'
        ))

    # Fraud Edges Trace
    if fraud_edge_x:
        traces.append(go.Scatter(
            x=fraud_edge_x, y=fraud_edge_y,
            line=dict(width=4.0, color='#DC2626', dash='dash'),
            hoverinfo='none',
            mode='lines',
            name='🚨 Fraud Transfers'
        ))

    # Nodes Trace
    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for node in G.nodes():
        if node not in pos:
            continue
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)
        tot_deg = in_deg + out_deg
        is_most_active = (node == most_active_customer_id)
        
        prefix = "👑 MOST ACTIVE: " if is_most_active else ""
        node_text.append(f"{prefix}Customer: {node}<br>Total Tx: {tot_deg}<br>Out: {out_deg} | In: {in_deg}")
        
        if is_most_active:
            node_color.append('#F59E0B')
            node_size.append(26)
        elif in_deg > 0 and out_deg > 0:
            node_color.append('#8B5CF6')
            node_size.append(max(10, min(24, 8 + tot_deg * 2)))
        elif out_deg > 0:
            node_color.append('#3B82F6')
            node_size.append(max(8, min(20, 8 + out_deg * 2)))
        else:
            node_color.append('#10B981')
            node_size.append(max(8, min(20, 8 + in_deg * 2)))

    traces.append(go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            color=node_color,
            size=node_size,
            line=dict(width=1.5, color='#FFFFFF')
        ),
        name='Customers'
    ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text="Directed Bank Transaction Network (Plotly View)", x=0.5, font=dict(size=16)),
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20, l=20, r=20, t=50),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        template='plotly_dark',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def create_top_active_customers_bar_chart(customer_stats_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """
    Creates a grouped bar chart displaying Outgoing vs Incoming transactions for top active customers.
    """
    top_df = customer_stats_df.head(top_n).copy()
    c_col = 'customer_id' if 'customer_id' in top_df.columns else 'customer'
    out_col = 'outgoing_tx_count' if 'outgoing_tx_count' in top_df.columns else 'sent'
    in_col = 'incoming_tx_count' if 'incoming_tx_count' in top_df.columns else 'received'
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top_df[c_col],
        y=top_df[out_col],
        name='Outgoing Transfers (Sent)',
        marker_color='#3B82F6',
        text=top_df[out_col],
        textposition='auto'
    ))
    fig.add_trace(go.Bar(
        x=top_df[c_col],
        y=top_df[in_col],
        name='Incoming Transfers (Received)',
        marker_color='#10B981',
        text=top_df[in_col],
        textposition='auto'
    ))

    fig.update_layout(
        barmode='stack',
        title=f"Top {top_n} Most Active Customers (Outgoing vs Incoming Transactions)",
        xaxis_title="Customer Account ID",
        yaxis_title="Number of Transactions",
        template='plotly_dark',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


def create_transaction_amount_distribution(df: pd.DataFrame, high_value_threshold: float) -> go.Figure:
    """
    Creates a histogram with log-scale of transaction amounts and a vertical threshold line
    highlighting high-value transactions.
    """
    fig = px.histogram(
        df,
        x='amount',
        nbins=60,
        log_y=True,
        color_discrete_sequence=['#3B82F6'],
        title="Transaction Amount Distribution (Log Frequency Scale)",
        labels={'amount': 'Transaction Amount ($)', 'count': 'Transaction Count (Log Scale)'}
    )

    # Add vertical line for high-value threshold
    fig.add_vline(
        x=high_value_threshold,
        line_width=3,
        line_dash="dash",
        line_color="#EF4444",
        annotation_text=f"High-Value Cutoff: ${high_value_threshold:,.0f}",
        annotation_position="top right",
        annotation_font_color="#EF4444"
    )

    fig.update_layout(
        template='plotly_dark',
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


def create_pareto_curve(df: pd.DataFrame) -> go.Figure:
    """
    Creates a Pareto Chart (Cumulative Volume % vs Cumulative Transaction Count %).
    Demonstrates concentration of wealth transfer in top high-value transactions.
    """
    sorted_df = df.sort_values(by='amount', ascending=False).reset_index(drop=True)
    sorted_df['cum_amount'] = sorted_df['amount'].cumsum()
    total_amount = sorted_df['cum_amount'].iloc[-1] if not sorted_df.empty else 1.0
    
    sorted_df['cum_amount_pct'] = (sorted_df['cum_amount'] / total_amount) * 100
    sorted_df['cum_tx_pct'] = ((sorted_df.index + 1) / len(sorted_df)) * 100

    # Downsample for snappy chart rendering if rows are large
    step_size = max(1, len(sorted_df) // 200)
    chart_df = sorted_df.iloc[::step_size]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=chart_df['cum_tx_pct'],
        y=chart_df['cum_amount_pct'],
        mode='lines',
        name='Cumulative Volume %',
        line=dict(color='#F59E0B', width=3)
    ))
    # 45-degree reference line
    fig.add_trace(go.Scatter(
        x=[0, 100], y=[0, 100],
        mode='lines',
        name='Even Distribution Line',
        line=dict(color='#64748B', dash='dash')
    ))

    # Add 80/20 guideline
    fig.add_hline(y=80, line_dash="dot", line_color="#EF4444", annotation_text="80% Total Volume", annotation_font_color="#EF4444")

    fig.update_layout(
        title="Pareto Concentration of Transfer Volume (80/20 Analysis)",
        xaxis_title="Top Transactions (% of Total Count)",
        yaxis_title="Cumulative Volume (% of Total $)",
        template='plotly_dark',
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def create_transaction_types_donut(df: pd.DataFrame) -> go.Figure:
    """
    Creates a modern donut chart of transaction types.
    """
    if 'type' not in df.columns:
        return go.Figure()
        
    counts = df['type'].value_counts().reset_index()
    counts.columns = ['type', 'count']

    colors = {
        'TRANSFER': '#3B82F6',
        'CASH_OUT': '#EF4444',
        'PAYMENT': '#10B981',
        'CASH_IN': '#8B5CF6',
        'DEBIT': '#F59E0B'
    }
    color_seq = [colors.get(t, '#94A3B8') for t in counts['type']]

    fig = go.Figure(data=[go.Pie(
        labels=counts['type'],
        values=counts['count'],
        hole=.55,
        marker=dict(colors=color_seq, line=dict(color='#0F172A', width=2))
    )])
    fig.update_layout(
        title="Transaction Breakdown by Type",
        template='plotly_dark',
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=True
    )
    return fig


def create_customer_flow_breakdown(customer_id: str, df: pd.DataFrame) -> go.Figure:
    """
    Creates a side-by-side volume and count bar breakdown for a selected customer.
    """
    sent = df[df['nameOrig'] == customer_id]
    recv = df[df['nameDest'] == customer_id]

    categories = ['Outgoing Transfers (Sent)', 'Incoming Transfers (Received)']
    counts = [len(sent), len(recv)]
    amounts = [sent['amount'].sum() if not sent.empty else 0.0, recv['amount'].sum() if not recv.empty else 0.0]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=categories,
        y=amounts,
        text=[f"${a:,.2f}" for a in amounts],
        textposition='auto',
        marker_color=['#3B82F6', '#10B981']
    ))
    fig.update_layout(
        title=f"Financial Volume Flow for Customer {customer_id}",
        yaxis_title="Total Volume ($)",
        template='plotly_dark',
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def build_interactive_subgraph(
    G: nx.MultiDiGraph,
    hubs: list,
    threshold: float,
    max_neighbors_per_hub: int = 50
) -> nx.DiGraph:
    """
    Extracts a connected subgraph of top hubs and their 1-hop counter-parties,
    collapsing multiple parallel transactions into aggregate weighted edges.
    """
    nodes = set(hubs)
    for h in hubs:
        if G.has_node(h):
            preds = list(G.predecessors(h))
            succs = list(G.successors(h))
            nodes.update(preds[:max_neighbors_per_hub])
            nodes.update(succs[:max_neighbors_per_hub])

    sub = nx.DiGraph()
    for u, v, d in G.subgraph(nodes).edges(data=True):
        amt = float(d.get('amount', 0.0))
        fraud = int(d.get('isFraud', 0))
        if sub.has_edge(u, v):
            sub[u][v]['weight'] += amt
            sub[u][v]['count'] += 1
            if fraud == 1:
                sub[u][v]['is_fraud'] = 1
        else:
            sub.add_edge(u, v, weight=amt, count=1, is_fraud=fraud, is_high_value=False)

    for u, v in sub.edges():
        sub[u][v]['is_high_value'] = (sub[u][v]['weight'] >= threshold)

    for n in sub.nodes():
        sub.nodes[n]['total_sent'] = sum(d['weight'] for _, _, d in sub.out_edges(n, data=True))
        sub.nodes[n]['total_received'] = sum(d['weight'] for _, _, d in sub.in_edges(n, data=True))

    return sub


def extract_customer_ego_subgraph(
    G: nx.MultiDiGraph,
    customer_id: str,
    threshold: float
) -> nx.DiGraph:
    """
    Extracts the direct ego-network for a single customer (all immediate senders and receivers).
    """
    return build_interactive_subgraph(G, hubs=[customer_id], threshold=threshold, max_neighbors_per_hub=100)


