import pandas as pd
from google.cloud import bigquery
import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

def get_feature_data(limit=1000):
    """Fetch raw data and transform it into features for ML"""
    # Explicitly get project ID to prevent Authentication errors
    project_id = os.getenv('BQ_PROJECT_ID')
    client = bigquery.Client(project=project_id)
    
    table_id = f"{project_id}.{os.getenv('BQ_DATASET_ID')}.{os.getenv('BQ_TABLE_ID')}"
    
    # Query BigQuery
    query = f"SELECT * FROM `{table_id}` ORDER BY timestamp DESC LIMIT {limit}"
    
    try:
        df = client.query(query).to_dataframe()
    except Exception as e:
        print(f" Error fetching data: {e}")
        return pd.DataFrame()
    
    if df.empty:
        return df

    # --- Feature Engineering ---
    # Sort for time-series analysis (crucial for rolling windows)
    df = df.sort_values(['symbol', 'timestamp'])
    
    # Feature 1: Rolling Mean (The 'Trend')
    df['rolling_mean'] = df.groupby('symbol')['price'].transform(
        lambda x: x.rolling(window=20).mean()
    )
    
    # Feature 2: Rolling Std (The 'Volatility')
    df['rolling_std'] = df.groupby('symbol')['price'].transform(
        lambda x: x.rolling(window=20).std()
    )
    
    # Feature 3: Price Change (The 'Velocity')
    df['price_change'] = df.groupby('symbol')['price'].diff()
    
    # Drop rows where rolling windows haven't filled yet (first 19 rows)
    return df.dropna()

if __name__ == "__main__":
    data = get_feature_data()
    if not data.empty:
        print("Features Engineered Successfully:")
        print(data[['symbol', 'price', 'rolling_mean', 'rolling_std', 'price_change']].tail())
    else:
        print(" No data available. Make sure your collector is running and BQ has records.")