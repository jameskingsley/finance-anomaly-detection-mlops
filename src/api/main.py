from fastapi import FastAPI
from clearml import Model
import joblib
import pandas as pd
from src.models.features import get_feature_data
import os

app = FastAPI(title="Finance Anomaly Detection API")

# Global variable to hold the loaded model
ml_model = None

@app.on_event("startup")
def load_latest_model():
    global ml_model
    print(" Fetching latest model from ClearML...")
    
    try:
        # Query Model Registry for our specific model name
        models = Model.query_models(
            project_name='Finance Anomaly Detection',
            model_name='isolation_forest_model', 
            only_published=False,
            max_results=1
        )
        
        if not models:
            print(" No model found in ClearML Registry.")
            return

        model_object = models[0]
        local_path = model_object.get_local_copy()
        
        # Load the serialized model
        ml_model = joblib.load(local_path)
        print(f" Model Loaded Successfully from ClearML! (ID: {model_object.id})")
        
    except Exception as e:
        print(f"Failed to load model: {e}")

@app.get("/")
def home():
    return {
        "status": "online", 
        "model_loaded": ml_model is not None,
        "environment": os.getenv("RENDER", "local")
    }

@app.get("/detect")
def detect_anomalies():
    """Returns full window for visualization and specific flagged anomalies"""
    if ml_model is None:
        return {"error": "ML Model is not initialized."}

    # 1. Pull the latest 100 ticks
    df = get_feature_data(limit=100)
    
    if df.empty:
        return {"message": "No data found in BigQuery."}

    # 2. Prepare features for prediction
    features = ['price', 'rolling_mean', 'rolling_std', 'price_change']
    
    # 3. Perform Inference on the entire batch
    # 1 = Normal, -1 = Anomaly
    df['anomaly_signal'] = ml_model.predict(df[features])
    
    # 4. Clean up timestamp for JSON serialization
    df['timestamp_str'] = df['timestamp'].astype(str)
    
    # 5. Extract only the anomalies for the alert table
    anomalies = df[df['anomaly_signal'] == -1].copy()

    return {
        "total_ticks": len(df),
        "anomaly_count": len(anomalies),
        # all_data used for the line chart
        "all_data": df.to_dict(orient="records"),
        # anomalies used for red markers and alert logs
        "anomalies": anomalies.to_dict(orient="records")
    }