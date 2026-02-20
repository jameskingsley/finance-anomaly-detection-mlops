from prefect import flow, task
# Use a relative-style import to force Python to stay within this package
from src.ingestion.collector import start_ingestion

@task(retries=3, retry_delay_seconds=10)
def stream_crypto_data():
    """Task to handle the continuous stream to BigQuery"""
    print("Data Engineering: Starting Coinbase WebSocket Stream...")
    start_ingestion()

@flow(name="Data-Engineering-Pipeline")
def run_ingestion_pipeline():
    stream_crypto_data()

if __name__ == "__main__":
    # Register it to Prefect and start the process
    run_ingestion_pipeline.serve(name="Data-Ingestion-Service")