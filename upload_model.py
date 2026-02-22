import joblib
from clearml import Task, OutputModel

# 1. Initialize Task with Cloud Storage enabled
task = Task.init(
    project_name='Finance Anomaly Detection', 
    task_name='Scheduled Model Update',
    output_uri=True  # 👈 This forces the upload to ClearML's server
)

# 2. Upload and SYNC
print("🚀 Uploading model...")
output_model = OutputModel(task=task, name='isolation_forest_model', framework='ScikitLearn')
# auto_delete_file=False keeps your local copy safe
output_model.update_weights(weights_filename='isolation_forest.pkl', auto_delete_file=False)

# 3. Publish and Close
output_model.publish() # 👈 Makes it visible to Render
task.close()           # 👈 Flushes all buffers to ensure upload completes

print(f"Success! New Model ID: {output_model.id}")