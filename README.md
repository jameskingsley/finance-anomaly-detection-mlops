# Finance Anomaly Detection MLOps System

A production-grade MLOps pipeline designed to detect financial anomalies in real-time cryptocurrency data. This project demonstrates a complete machine learning lifecycle, from automated data engineering to model orchestration and live dashboarding.

### Live Demo
* Web Dashboard: (https://finance-anomaly-detection-mlops-wtwuzqrxwdmd3jkucfpbhs.streamlit.app/)

## Tech Stack
Data Engineering: Python, WebSockets, Google BigQuery.

Orchestration: Prefect 3.0 (Automated Ingestion & Retraining schedules).

Machine Learning: Scikit-Learn (Isolation Forest), Pandas.

Model Management: ClearML (Experiment tracking & Model Registry).

Deployment: FastAPI (Backend), Streamlit (Frontend), Render (Hosting).

## Architecture
The system consists of three independent but interconnected layers:

* The Heart (Data Pipeline): A continuous WebSocket listener that streams BTC/ETH ticker data from Coinbase directly into a BigQuery data warehouse.

* The Brain (MLOps): A Prefect-orchestrated pipeline that runs scheduled retraining jobs, evaluates model performance, and registers the best versions to ClearML.

* The Face (UI/API): A FastAPI backend that serves model predictions and a Streamlit dashboard that visualizes anomalies and market trends.

## Local Setup
*  Clone the Repository
* Bash
git clone https://github.com/YOUR_USERNAME/finance-anomaly-detection-mlops.git
cd finance-anomaly-detection-mlops
##  Environment Configuration
Create a .env file in the root directory and populate it with your credentials:
#####
#####

3. Installation
This project uses modular requirements files for efficient deployment:

Development: pip install -r requirements.txt

API/Render: pip install -r requirements_api.txt

Dashboard: pip install -r requirements_ui.txt

## Dashboard
streamlit run src/ui/app.py

## Monitoring
* Orchestration: All jobs are managed via the Prefect Cloud Dashboard.

* Experiments: Model metrics and anomaly plots are logged in the ClearML Console.