import pandas as pd
from google.cloud import bigquery
import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

def get_feature_data(limit=1000):
    """Fetch raw data and transform it into features for ML"""
    
    # CREDENTIAL RESOLUTION 
    # Check if Render's environment variable is already set
    # If not, look for the local file in the config folder
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        local_creds = os.path.join(os.getcwd(), "config", "google_creds.json")
        if os.path.exists(local_creds):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = local_creds
            print(f" Using local credentials at: {local_creds}")
    else:
        print(f"Using Render/Environment credentials at: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}")

    # Explicitly get project ID to prevent Authentication errors
    project_id = os.getenv('BQ_PROJECT_ID')
    
    # Initializing client without passing credentials directly—it will 
    # automatically pick up the path from GOOGLE_APPLICATION_CREDENTIALS
    client = bigquery.Client(project=project_id)
    
    table_id = f"{project_id}.{os.getenv('BQ_DATASET_ID')}.{os.getenv('BQ_TABLE_ID')}"
    
    # Query BigQuery
    query = f"SELECT * FROM `{table_id}` ORDER BY timestamp DESC LIMIT {limit}"
    
    try:
        df = client.query(query).to_dataframe()
    except Exception as e:
        print(f"Error fetching data from BigQuery: {e}")
        return pd.DataFrame()
    
    if df.empty:
        return df

    # Feature Engineering 
    # Sort for time-series analysis
    df = df.sort_values(['symbol', 'timestamp'])
    
    # Rolling Mean 
    df['rolling_mean'] = df.groupby('symbol')['price'].transform(
        lambda x: x.rolling(window=20).mean()
    )
    
    # Rolling Std 
    df['rolling_std'] = df.groupby('symbol')['price'].transform(
        lambda x: x.rolling(window=20).std()
    )
    
    # Price Change 
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