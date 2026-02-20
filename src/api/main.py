from fastapi import FastAPI
from clearml import Model
import joblib
import pandas as pd
from src.models.features import get_feature_data
import os
from contextlib import asynccontextmanager

# Use Global State for the model
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events cleanly."""
    print(" [LIFESPAN] Fetching latest model from ClearML...")
    try:
        # Search for published models first, then fallback to any
        # This project name must match EXACTLY what is in the ClearML UI
        models = Model.query_models(
            project_name='Finance Anomaly Detection',
            model_name='isolation_forest_model', 
            only_published=True,  
            max_results=1
        )
        
        # Fallback if no 'published' model is found
        if not models:
            print(" No PUBLISHED model found. Trying to find any available version...")
            models = Model.query_models(
                project_name='Finance Anomaly Detection',
                model_name='isolation_forest_model', 
                only_published=False,
                max_results=1
            )

        if models:
            model_object = models[0]
            # Force download ensure we get the file even if cache is weird on Render
            local_path = model_object.get_local_copy(force_download=True)
            
            if local_path:
                ml_models["anomaly_detector"] = joblib.load(local_path)
                print(f" Model Loaded Successfully! ID: {model_object.id}")
            else:
                print(" ClearML returned None for local_path.")
        else:
            print(" No model found at all in the ClearML Registry.")

    except Exception as e:
        print(f" Critical Error during model loading: {e}")
    
    yield
    # Shutdown logic
    ml_models.clear()

app = FastAPI(title="Finance Anomaly Detection API", lifespan=lifespan)

@app.get("/")
def home():
    return {
        "status": "online", 
        "model_loaded": "anomaly_detector" in ml_models,
        "environment": os.getenv("RENDER", "local")
    }

@app.get("/detect")
def detect_anomalies():
    """Returns full window for visualization and specific flagged anomalies"""
    detector = ml_models.get("anomaly_detector")
    
    if detector is None:
        return {"error": "ML Model is not initialized. Check server logs."}

    #  Pull data
    df = get_feature_data(limit=100)
    if df.empty:
        return {"message": "No data found in BigQuery."}

    #  Prediction
    features = ['price', 'rolling_mean', 'rolling_std', 'price_change']
    # Ensure columns match training order
    df['anomaly_signal'] = detector.predict(df[features])
    
    # Serialize for JSON (BigQuery timestamps are tricky)
    df['timestamp'] = df['timestamp'].astype(str)
    
    #  Filter results
    anomalies = df[df['anomaly_signal'] == -1].copy()

    return {
        "total_ticks": len(df),
        "anomaly_count": len(anomalies),
        "all_data": df.to_dict(orient="records"),
        "anomalies": anomalies.to_dict(orient="records")
    }