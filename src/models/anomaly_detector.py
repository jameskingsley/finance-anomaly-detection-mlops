import pandas as pd
import os
import joblib
import plotly.express as px
from sklearn.ensemble import IsolationForest
from clearml import Task, OutputModel
from src.models.features import get_feature_data
from prefect import flow, task 

@task(name="Fetch-Data-From-BigQuery", retries=2, retry_delay_seconds=30)
def fetch_data():
    """Task to pull feature data for training"""
    df = get_feature_data(limit=1000)
    if df.empty:
        raise ValueError("No data fetched from BigQuery. Training aborted.")
    return df

@task(name="Train-And-Register-Model")
def train_and_register(df: pd.DataFrame):
    """Task to train Isolation Forest and push to ClearML Model Registry"""
    
    # Cloud-hosted model storage enabled via output_uri
    cl_task = Task.init(
        project_name='Finance Anomaly Detection', 
        task_name='Isolation Forest Training',
        task_type=Task.TaskTypes.training,
        reuse_last_task_id=False,
        output_uri=True 
    )

    params = {
        "contamination": 0.02,
        "n_estimators": 100,
        "random_state": 42
    }
    cl_task.connect(params)

    features = ['price', 'rolling_mean', 'rolling_std', 'price_change']
    X = df[features]
    model = IsolationForest(**params)
    
    df['anomaly_signal'] = model.fit_predict(X)
    df['status'] = df['anomaly_signal'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')

    num_anomalies = int((df['anomaly_signal'] == -1).sum())
    cl_task.get_logger().report_single_value(name='Anomaly Count', value=num_anomalies)

    fig = px.scatter(
        df, x="timestamp", y="price", color="status",
        symbol="status", color_discrete_map={'Normal': 'blue', 'Anomaly': 'red'},
        title="Crypto Anomalies Detected by Isolation Forest"
    )
    cl_task.get_logger().report_plotly(
        title="Anomaly Detection Visualization",
        series="Price vs Time",
        figure=fig
    )

    model_path = "isolation_forest.pkl"
    joblib.dump(model, model_path)
    
    # Syncs local file to ClearML Cloud HTTPS URL
    output_model = OutputModel(task=cl_task, name="isolation_forest_model")
    output_model.update_weights(weights_filename=model_path, auto_delete_file=False)
    
    # Finalize and Publish so API picks it up automatically
    output_model.publish()
    cl_task.close()
    
    return f"Success: Found {num_anomalies} anomalies. Model Registered and Published."

@flow(name="MLOps-Training-Pipeline")
def train_model_flow():
    """The Main MLOps Entry Point"""
    raw_data = fetch_data()
    result = train_and_register(raw_data)
    print(result)

if __name__ == "__main__":
    # This creates a deployment on Prefect and schedules it
    print("Starting Prefect Server deployment...")
    train_model_flow.serve(
        name="Model-Retraining-Job",
        cron="0 0 * * *", # Runs every day at 00:00 (Midnight)
    )