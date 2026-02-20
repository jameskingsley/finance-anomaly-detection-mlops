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
    
    # Initialize ClearML Task inside the Prefect Task
    cl_task = Task.init(
        project_name='Finance Anomaly Detection', 
        task_name='Isolation Forest Training',
        task_type=Task.TaskTypes.training,
        reuse_last_task_id=False 
    )

    #  Model Parameters
    params = {
        "contamination": 0.02,
        "n_estimators": 100,
        "random_state": 42
    }
    cl_task.connect(params)

    #  Train Model
    features = ['price', 'rolling_mean', 'rolling_std', 'price_change']
    X = df[features]
    model = IsolationForest(**params)
    
    # 1 for normal, -1 for anomaly
    df['anomaly_signal'] = model.fit_predict(X)
    df['status'] = df['anomaly_signal'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')

    #  Log Metrics to ClearML
    num_anomalies = int((df['anomaly_signal'] == -1).sum())
    cl_task.get_logger().report_single_value(name='Anomaly Count', value=num_anomalies)

    #  Create and Log Plotly Chart
    fig = px.scatter(
        df, 
        x="timestamp", 
        y="price", 
        color="status",
        symbol="status",
        color_discrete_map={'Normal': 'blue', 'Anomaly': 'red'},
        title="Crypto Anomalies Detected by Isolation Forest",
        hover_data=['price_change', 'symbol']
    )
    
    cl_task.get_logger().report_plotly(
        title="Anomaly Detection Visualization",
        series="Price vs Time",
        figure=fig
    )

    # Save and Register Model
    model_path = "isolation_forest.pkl"
    joblib.dump(model, model_path)
    
    output_model = OutputModel(task=cl_task, name="isolation_forest_model")
    output_model.update_weights(weights_filename=model_path)
    
    # Cleanup task
    cl_task.close()
    
    return f"Success: Found {num_anomalies} anomalies. Model Registered."

@flow(name="MLOps-Training-Pipeline")
def train_model_flow():
    """The Main MLOps Entry Point"""
    raw_data = fetch_data()
    result = train_and_register(raw_data)
    print(result)

if __name__ == "__main__":
    # This registers the flow and starts a local 'worker' 
    # right here in this terminal window.
    train_model_flow.serve(
        name="Model-Retraining-Job",
        cron="0 0 * * *", # Run at midnight
    )