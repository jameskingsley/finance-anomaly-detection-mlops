import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import time
from datetime import datetime

# Page Config 
st.set_page_config(page_title="James Tech: Anomaly Dashboard", layout="wide", page_icon="🛡️")

#  CUSTOM CSS 
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    [data-testid="stMetric"] {
        background-color: #161b22;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #30363d;
    }
    [data-testid="stMetricLabel"] p {
        color: #FFFFFF !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricValue"] div {
        color: #FFFFFF !important;
        font-size: 2rem !important;
        font-weight: 900 !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title(" Real-Time Finance Anomaly Detection")

# Sidebar 
st.sidebar.header(" Control Panel")
API_URL = st.sidebar.text_input("FastAPI Endpoint", "https://finance-anomaly-detection-mlops.onrender.com")
REFRESH_RATE = st.sidebar.slider("Refresh Interval (s)", 2, 60, 5)
SYMBOL = st.sidebar.selectbox("Market Symbol", ["BTC-USD", "ETH-USD"])

# Main Dashboard Container 
placeholder = st.empty()

while True:
    try:
        response = requests.get(f"{API_URL}/detect", timeout=60)
        response.raise_for_status()
        data = response.json()

        if "all_data" in data:
            df = pd.DataFrame(data["all_data"])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            df_plot = df[df['symbol'] == SYMBOL]
            df_anomalies = df_plot[df_plot['anomaly_signal'] == -1]

            with placeholder.container():
                # Metrics Row
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Market", SYMBOL)
                m2.metric("Last Price", f"${df_plot['price'].iloc[-1]:,.2f}")
                m3.metric("Ticks Analyzed", data["total_ticks"])
                m4.metric("Anomalies Found", data["anomaly_count"])

                # Plotly Chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_plot['timestamp'], 
                    y=df_plot['price'],
                    mode='lines',
                    name='Market Price',
                    line=dict(color='#00d4ff', width=3)
                ))

                if not df_anomalies.empty:
                    fig.add_trace(go.Scatter(
                        x=df_anomalies['timestamp'], 
                        y=df_anomalies['price'],
                        mode='markers',
                        name=' ANOMALY',
                        marker=dict(color='#FF4B4B', size=15, symbol='circle', line=dict(color='white', width=2))
                    ))

                fig.update_layout(
                    template="plotly_dark",
                    height=550,
                    margin=dict(l=0, r=0, t=30, b=0),
                    hovermode="x unified"
                )

                # unique key using timestamp 
                chart_key = f"chart_{SYMBOL}_{time.time()}"
                st.plotly_chart(fig, use_container_width=True, key=chart_key)

                if not df_anomalies.empty:
                    st.subheader("Incident Log")
                    st.table(df_anomalies[['timestamp', 'price', 'price_change']].tail(5))
                else:
                    st.success(f" Market Stable: No anomalies detected.")

    except Exception as e:
        # Prevent the screen from flashing too much on connection errors
        with placeholder.container():
            st.error(" Offline: Waiting for FastAPI...")
            st.warning(f"Technical Detail: {e}")
            st.info("Ensure uvicorn src.api.main:app --reload is running.")

    time.sleep(REFRESH_RATE)