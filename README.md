# Enrollment Forecast Project

This project builds an enrollment forecasting system for educational data using machine learning and time-series models. It includes a backend API for serving predictions and a frontend interface for interacting with the system.

## Project Overview

- Predict student enrollment trends over time
- Use feature engineering and time-series modeling
- Provide a simple API and web interface

## Project Structure

- backend/: contains the API server, data processing scripts, and model logic
- frontend/: contains the web interface
- data/: contains processed datasets and feature engineering outputs
```
root/ 
├── render.yaml                                         # Deployment configuration for Render
├── xgboost.pkl                                         # Trained XGBoost model
├── backend/                                            # Backend API and data processing
│   ├── API.py                                          # Backend API entry point
│   ├── data/                                    
│   │   ├── data_transform/                             # Data transformation scripts
│   │   │   ├── ETL.py                                  # Extract-Transform-Load pipeline
│   │   │   ├── synthetic_data.py                       # Synthetic data generation
│   │   └── feature/                                    # Feature-engineered datasets
│   │       ├── feature_engineer.py                     # Feature engineering script
│   │       └── feature_engineered_test_day_splits/     # Test data split by day
│   └── model/                                          # Enrollment forecasting model
│       ├── EnrollmentForecastModel.py                  # Forecasting model logic
│       └── model_dev.ipynb                             # Model development notebook
└── frontend/                                           # User interface
    ├── app.py                                          # Frontend application entry point
    └── requirements.txt                                # Frontend dependencies
```
## Code running process 

### Step 0: Environment settings

```bash
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

### Step 1: Download OULAD dataset

Download [OULAD](https://analyse.kmi.open.ac.uk/open_dataset) dataset and move it to backend/data/OULAD/

### Step 2: Generate synthetic data

```bash
python backend/data/data_transform/synthetic_data.py
```

**Output:**

- `backend/data/synthetic/synthetic_marketing_data.csv`
- `backend/data/synthetic/synthetic_data_dictionary.json`

### Step 3: ETL — merge OULAD + synthetic

```bash
python backend/data/data_transform/ETL.py
```

**Output:** `backend/data/feature/feature_table.csv`

### Step 4: Feature engineering

```bash
python backend/data/feature/feature_engineer.py
```

**Output:**

- `backend/data/feature/feature_engineered_train.csv`
- `backend/data/feature/feature_engineered_test.csv`
- `backend/data/feature/feature_engineered_test_day_splits/feature_engineered_test_day_*.csv`

### Step 5: Model training

**Output:** `xgboost.pkl`.

### Step 6: Run the Dashboard

```bash
streamlit run frontend/app.py
```
Front-end: `http://localhost:8501`

### Step 7 (tùy chọn): Run Backend API

```bash
uvicorn backend.API:app --reload
```

API at `http://127.0.0.1:8000`.

## Model and Data Notes

- The backend includes training and evaluation notebooks for several forecasting approaches.
- Pretrained model file `xgboost.pkl` are stored in the project root.
- Feature-engineered datasets are stored under backend/data/feature/.

## Deployment

The project includes a Render deployment configuration in render.yaml.
