import pandas as pd
from google.cloud import bigquery
import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

def get_feature_data(limit=1000):
    """Fetch raw data and transform it into features for ML"""
    
    # SUPER-SAFE 
    # hunt for the JSON key in all possible Render/Local locations
    env_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    
    if env_path and os.path.exists(env_path):
        print(f"[AUTH] Using path from Environment Variable: {env_path}")
    
    elif os.path.exists("/opt/render/project/src/google_creds.json"):
        # This is the standard path for Render Secret Files
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/opt/render/project/src/google_creds.json"
        print("[AUTH] Found credentials in Render Project Root.")
        
    elif os.path.exists(os.path.join(os.getcwd(), "google_creds.json")):
        # Backup: Check if it's in the current working directory
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(os.getcwd(), "google_creds.json")
        print(" [AUTH] Found credentials in Current Working Directory.")

    else:
        # Local Development Fallback
        local_creds = os.path.join(os.getcwd(), "config", "google_creds.json")
        if os.path.exists(local_creds):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = local_creds
            print(f" [AUTH] Using local config credentials: {local_creds}")
        else:
            print("[AUTH] WARNING: No Google Credentials file found in any known location!")

    # Explicitly get project ID
    project_id = os.getenv('BQ_PROJECT_ID')
    
    # The client will now use the path we set in os.environ above
    try:
        client = bigquery.Client(project=project_id)
    except Exception as auth_err:
        print(f" [AUTH] Critical Failure initializing BigQuery Client: {auth_err}")
        return pd.DataFrame()
    
    table_id = f"{project_id}.{os.getenv('BQ_DATASET_ID')}.{os.getenv('BQ_TABLE_ID')}"
    
    # Query BigQuery
    query = f"SELECT * FROM `{table_id}` ORDER BY timestamp DESC LIMIT {limit}"
    
    try:
        df = client.query(query).to_dataframe()
    except Exception as e:
        print(f" [DATA] Error fetching data from BigQuery: {e}")
        return pd.DataFrame()
    
    if df.empty:
        return df

    #  Feature Engineering 
    df = df.sort_values(['symbol', 'timestamp'])
    
    df['rolling_mean'] = df.groupby('symbol')['price'].transform(
        lambda x: x.rolling(window=20).mean()
    )
    
    df['rolling_std'] = df.groupby('symbol')['price'].transform(
        lambda x: x.rolling(window=20).std()
    )
    
    df['price_change'] = df.groupby('symbol')['price'].diff()
    
    return df.dropna()

if __name__ == "__main__":
    data = get_feature_data()
    if not data.empty:
        print(" Features Engineered Successfully:")
        print(data[['symbol', 'price', 'rolling_mean', 'rolling_std', 'price_change']].tail())
    else:
        print("No data available.")