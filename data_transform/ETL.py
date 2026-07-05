from pathlib import Path
import json
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "feature"
OUTPUT_DIR.mkdir(exist_ok=True)
OULAD_DIR = ROOT / "OULAD"
SYNTHETIC_DIR = ROOT / "synthetic"

def extract_oulad_data(data_dir=OULAD_DIR):
    if not data_dir.exists():
        raise FileNotFoundError(f"OULAD directory not found: {data_dir}")

    assessment_data = pd.read_csv(OULAD_DIR / 'assessments.csv')
    courses_data = pd.read_csv(OULAD_DIR / 'courses.csv')
    studentInfo_data = pd.read_csv(OULAD_DIR / 'studentInfo.csv')
    vle_data = pd.read_csv(OULAD_DIR / 'vle.csv')
    
    # Demographic feature
    demographics = studentInfo_data.groupby(['code_module', 'code_presentation']).agg(
        historical_student_count=('id_student', 'count'),
        pct_male=('gender', lambda x: (x == 'M').mean()),
        pct_higher_ed=('imd_band', lambda x: x.isin(['70-80%', '80-90%', '90-100%']).mean())
    ).reset_index()

    # Course difficulty feature
    assessment_features = assessment_data.groupby(['code_module', 'code_presentation']).agg(
        total_assignments=('id_assessment', 'count'),
        is_exam_included=('assessment_type', lambda x: (x == 'Exam').any().astype(int))
    ).reset_index()

    # Abundance of learning materials feature
    vle_features = vle_data.groupby('code_module').agg(
        total_learning_materials=('id_site', 'count')
    ).reset_index()

    # Connect to courses table
    oulad_features = courses_data.copy()
    
    # Left join features
    oulad_features = pd.merge(oulad_features, demographics, on=['code_module', 'code_presentation'], how='left')
    oulad_features = pd.merge(oulad_features, assessment_features, on=['code_module', 'code_presentation'], how='left')
    oulad_features = pd.merge(oulad_features, vle_features, on=['code_module'], how='left')

    # Missing values
    fill_values = {
        'historical_student_count': 0,
        'pct_male': oulad_features['pct_male'].mean(),
        'pct_higher_ed': oulad_features['pct_higher_ed'].mean(),
        'total_assignments': 0,
        'is_exam_included': 0,
        'total_learning_materials': 0
    }
    oulad_features.fillna(value=fill_values, inplace=True)
    
    return oulad_features


def build_feature_table(oulad_df, synthetic_df):
    # Merge oulad_df and synthetic_df
    df_master = pd.merge(
        oulad_df, 
        synthetic_df, 
        on=['code_module', 'code_presentation'], 
        how='left'
    )

    # Fill missing values after the merge
    df_master = df_master.fillna({
        'historical_student_count': 0,
        'total_assignments': 0,
        'total_learning_materials': 0
    })

    # Create basic time-based features
    df_master['day_of_week'] = df_master['day_offset'] % 7
    df_master['is_weekend'] = df_master['day_of_week'].isin([5, 6]).astype(int)

    return df_master


def save_etl_outputs(df_master):
    etl_path = OUTPUT_DIR / "feature_table.csv"
    df_master.to_csv(etl_path, index=False)
    print(f"ETL combined data saved to {etl_path}")
    
if __name__ == "__main__":
    oulad_df = extract_oulad_data()
    synthetic_df = pd.read_csv(SYNTHETIC_DIR / "synthetic_marketing_data.csv")
    df_master = build_feature_table(oulad_df, synthetic_df)
    synthetic_dictionary = save_etl_outputs(df_master)