import asyncio
import json
import websockets
import os
import io
from google.cloud import bigquery
from dotenv import load_dotenv

load_dotenv()

class CoinbaseIngestor:
    def __init__(self):
        self.client = bigquery.Client()
        self.table_id = f"{os.getenv('BQ_PROJECT_ID')}.{os.getenv('BQ_DATASET_ID')}.{os.getenv('BQ_TABLE_ID')}"
        self.uri = "wss://ws-feed.exchange.coinbase.com"
        self.buffer = []

    async def fetch_and_load(self, batch_size=20):
        # Outer loop ensures that if the connection drops, we try to reconnect
        while True:
            try:
                print(" Connecting to Coinbase WebSocket...")
                async with websockets.connect(
                    self.uri,
                    ping_interval=30,  # Sends a ping every 30s to keep connection alive
                    ping_timeout=10    # Waits 10s for a pong before timing out
                ) as ws:
                    subscribe_msg = {
                        "type": "subscribe",
                        "channels": [{"name": "ticker", "product_ids": ["BTC-USD", "ETH-USD"]}]
                    }
                    await ws.send(json.dumps(subscribe_msg))
                    
                    while True:
                        message = await ws.recv()
                        data = json.loads(message)
                        
                        if data.get("type") == "ticker":
                            row = {
                                "timestamp": data.get("time"),
                                "symbol": data.get("product_id"),
                                "price": float(data.get("price")),
                                "volume": float(data.get("last_size"))
                            }
                            self.buffer.append(row)

                            if len(self.buffer) >= batch_size:
                                self.upload_batch()
                                
            except (websockets.exceptions.ConnectionClosedError, 
                    websockets.exceptions.ConnectionClosedOK) as e:
                print(f" Connection lost ({e}). Reconnecting in 5 seconds...")
                await asyncio.sleep(5)
            except Exception as e:
                print(f" Unexpected error: {e}. Retrying...")
                await asyncio.sleep(5)

    def upload_batch(self):
        try:
            job_config = bigquery.LoadJobConfig(
                source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
            )
            json_data = "\n".join([json.dumps(r) for r in self.buffer])
            job = self.client.load_table_from_file(
                io.StringIO(json_data), 
                self.table_id, 
                job_config=job_config
            )
            job.result()  # Wait for the job to complete
            print(f" Data Engineering: Uploaded {len(self.buffer)} rows to BigQuery.")
            self.buffer = []
        except Exception as e:
            print(f" Failed to upload batch: {e}")

def start_ingestion():
    ingestor = CoinbaseIngestor()
    asyncio.run(ingestor.fetch_and_load())