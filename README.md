# 🏦 FinNet: Banking Transaction Network Analysis Dashboard

A directed-graph modeling and transaction network analytics dashboard built using **Streamlit**, **NetworkX**, **PyVis**, and **Plotly**.

## 📌 Project Overview
- **Author**: Saransh Singh
- **Registration No.**: 12301472
- **Roll No.**: 35
- **Section**: RK2P23FHB36
- **Institution**: Lovely Professional University (LPU)
- **Dataset**: PaySim Synthetic Financial Dataset

## 🚀 Key Features
1. **Directed Graph Construction**: Nodes represent customer accounts; directed edges represent money transfers ($Sender \to Receiver$).
2. **Interactive Force-Directed Network**: Built with PyVis and WebGL physics simulation.
3. **High-Value Transfers Highlighting**: Top 5% (or customizable percentile) highlighted in vibrant crimson.
4. **Customer Activity & Degree Analysis**: Automatic discovery of the most active account and top leaderboards.
5. **Zero-Scroll Executive Cockpit UI**: Full single-screen viewport layout.

## 🛠️ Local Installation & Run
```bash
# Clone the repository
git clone https://github.com/<your-username>/banking-transaction-network.git
cd banking-transaction-network

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit dashboard
streamlit run app.py
```

