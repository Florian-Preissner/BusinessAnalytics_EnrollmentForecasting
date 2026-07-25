# Project Structure

> This file describes the main directory structure of the project for easier tracking and development.

```text
BI/
├── README.md                      # Project documentation
├── render.yaml                    # Deployment configuration for Render
├── lightgbm.pkl                  # Trained LightGBM model
├── xgboost.pkl                   # Trained XGBoost model
├── backend/                      # Backend API and data processing
│   ├── API.py                     # Backend API entry point
│   ├── requirements.txt          # Required Python libraries
│   ├── data/                      # Data and ETL/feature engineering pipeline
│   │   ├── data_transform/        # Data transformation scripts
│   │   │   ├── ETL.py             # Extract-Transform-Load pipeline
│   │   │   ├── synthetic_data.py  # Synthetic data generation
│   │   │   └── data_transformation.ipynb  # Data analysis notebook
│   │   └── feature/               # Feature-engineered datasets
│   │       ├── feature_engineer.py  # Feature engineering script
│   │       ├── feature_engineered_table.csv
│   │       ├── feature_engineered_train.csv
│   │       ├── feature_engineered_test.csv
│   │       └── feature_engineered_test_day_splits/  # Test data split by day
│   │           ├── feature_engineered_test_day_0.csv
│   │           ├── feature_engineered_test_day_1.csv
│   │           ├── ...
│   │           └── feature_engineered_test_day_133.csv
│   └── model/                     # Enrollment forecasting model
│       ├── EnrollmentForecastModel.py  # Forecasting model logic
│       └── model_dev.ipynb        # Model development notebook
└── frontend/                      # User interface
    ├── app.py                     # Frontend application entry point
    └── requirements.txt          # Frontend dependencies
```
