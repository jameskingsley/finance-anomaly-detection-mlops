from google.cloud import bigquery
from dotenv import load_dotenv
import os

# Load variables from .env
load_dotenv()

def bootstrap_bigquery():
    # 1. Connect to Google Cloud using your JSON key
    client = bigquery.Client()
    
    project_id = os.getenv("BQ_PROJECT_ID")
    dataset_name = os.getenv("BQ_DATASET_ID")
    table_name = os.getenv("BQ_TABLE_ID")
    
    # This is the full path: project-id.dataset_name
    dataset_id = f"{project_id}.{dataset_name}"
    
    # 2. Define the Dataset (The "Folder")
    dataset = bigquery.Dataset(dataset_id)
    dataset.location = "US" 
    
    print(f"Attempting to create dataset: {dataset_id}")
    try:
        client.create_dataset(dataset, exists_ok=True)
        print(f"Success: Dataset '{dataset_name}' created in {project_id}.")
    except Exception as e:
        print(f"Failed to create dataset: {e}")
        return

    # 3. Define the Table Schema (The "Spreadsheet")
    table_id = f"{dataset_id}.{table_name}"
    schema = [
        bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("symbol", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("price", "FLOAT", mode="REQUIRED"),
        bigquery.SchemaField("volume", "FLOAT", mode="NULLABLE"),
    ]
    
    table = bigquery.Table(table_id, schema=schema)
    
    try:
        client.create_table(table, exists_ok=True)
        print(f"Success: Table '{table_name}' created inside '{dataset_name}'.")
    except Exception as e:
        print(f"Failed to create table: {e}")

if __name__ == "__main__":
    bootstrap_bigquery()