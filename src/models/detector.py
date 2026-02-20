import pandas as pd
from google.cloud import bigquery
import os
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()

def detect_anomalies():
    client = bigquery.Client()
    table_id = f"{os.getenv('BQ_PROJECT_ID')}.{os.getenv('BQ_DATASET_ID')}.{os.getenv('BQ_TABLE_ID')}"

    #   Pull the latest 200 ticks for better visualization context
    query = f"""
        SELECT timestamp, symbol, price 
        FROM `{table_id}` 
        ORDER BY timestamp DESC 
        LIMIT 200
    """
    df = client.query(query).to_dataframe()

    if df.empty:
        print("No data found.")
        return

    df = df.sort_values('timestamp')

    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].copy()
        
        # Calculate Rolling Stats (Window of 20)
        window = 20
        symbol_df['mean'] = symbol_df['price'].rolling(window=window).mean()
        symbol_df['std'] = symbol_df['price'].rolling(window=window).std()
        symbol_df['z_score'] = (symbol_df['price'] - symbol_df['mean']) / symbol_df['std']

        # Identify Anomalies
        anomalies = symbol_df[symbol_df['z_score'].abs() > 3]

        # --- Plotly Visualization ---
        fig = go.Figure()

        # Add Price Line
        fig.add_trace(go.Scatter(
            x=symbol_df['timestamp'], y=symbol_df['price'],
            mode='lines', name='Price', line=dict(color='blue', width=1)
        ))

        # Add Rolling Mean (The "Normal" baseline)
        fig.add_trace(go.Scatter(
            x=symbol_df['timestamp'], y=symbol_df['mean'],
            mode='lines', name='Rolling Mean (20)', line=dict(color='orange', dash='dash')
        ))

        # Add Anomalies as Red Dots
        if not anomalies.empty:
            fig.add_trace(go.Scatter(
                x=anomalies['timestamp'], y=anomalies['price'],
                mode='markers', name='Anomaly',
                marker=dict(color='red', size=10, symbol='x')
            ))
            print(f"{symbol}: {len(anomalies)} anomalies found!")
        else:
            print(f"{symbol}: No anomalies detected.")

        fig.update_layout(
            title=f"Real-time Anomaly Detection: {symbol}",
            xaxis_title="Time",
            yaxis_title="Price (USD)",
            template="plotly_white"
        )
        
        # This will open a tab in your browser
        fig.show()

if __name__ == "__main__":
    detect_anomalies()