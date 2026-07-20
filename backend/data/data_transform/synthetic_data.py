from pathlib import Path
import json
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "synthetic"
OUTPUT_DIR.mkdir(exist_ok=True)
COURSES_PATH = ROOT / "OULAD" / "courses.csv"

def generate_synthetic_data(courses_df, days_per_presentation=250, seed=42):
    """
    Generate time-series data by course level.
    """
    rng = np.random.default_rng(seed)
    all_series = []

    for _, row in courses_df.iterrows():
        module = row['code_module']
        presentation = row['code_presentation']

        # Create a Time Index for the semester
        t = np.arange(days_per_presentation)

        # Generate the daily enrollment numbers based on various factors
        # Trend
        trend = -0.05 * t + rng.uniform(10, 30) 
        trend = np.clip(trend, 0, None)
        
        # Weekly Seasonality
        weekly_seasonality = 8 * np.sin(2 * np.pi * t / 7 - np.pi/2) 
        
        # Academic Calendar Effects
        semester_start_flag = np.where(t < 14, 1, 0)
        semester_start_effect = semester_start_flag * rng.uniform(20, 50, size=days_per_presentation)
        
        semester_end_flag = np.where(t > days_per_presentation - 14, 1, 0)
        
        # Holidays
        holiday_flag = np.where((t % 100 >= 90) & (t % 100 <= 95), 1, 0)
        holiday_effect = holiday_flag * -15

        # Marketing-campaign spikes
        marketing_flag = rng.choice([0, 1], size=days_per_presentation, p=[0.97, 0.03])
        marketing_effect = marketing_flag * rng.uniform(30, 80, size=days_per_presentation)
        
        # Random Noise
        noise = rng.normal(0, 4, days_per_presentation)

        # Daily Enrollments
        daily_enrollments = np.maximum(0, (
            trend 
            + weekly_seasonality 
            + semester_start_effect 
            + holiday_effect 
            + marketing_effect 
            + noise
        )).astype(int)
        
        # Corresponding marketing costs
        marketing_spend = np.where(marketing_flag == 1, rng.uniform(500, 2000), rng.uniform(10, 50)).round(2)
        
        # Tạo DataFrame cho khóa học này
        df_temp = pd.DataFrame({
            "code_module": module,
            "code_presentation": presentation,
            "day_offset": t,
            "marketing_spend": marketing_spend,
            "marketing_campaign_flag": marketing_flag,
            "holiday_flag": holiday_flag,
            "semester_start_flag": semester_start_flag,
            "semester_end_flag": semester_end_flag,
            "enrollment_num_daily": daily_enrollments
        })
        all_series.append(df_temp)

    # Nối tất cả các chuỗi thời gian của mọi môn học lại với nhau
    combined_ts = pd.concat(all_series, ignore_index=True)
    return combined_ts

def build_data_dictionary():
    """
    Data dictionary containing metadata for the synthetic dataset
    """
    dict_records = [
        {
            "column_name": "code_module",
            "data_type": "string",
            "unit": "N/A",
            "valid_range": "OULAD module codes (e.g., AAA, BBB)",
            "generation_logic": "Extracted directly from OULAD courses.csv table."
        },
        {
            "column_name": "code_presentation",
            "data_type": "string",
            "unit": "N/A",
            "valid_range": "OULAD presentation terms (e.g., 2013J, 2014B)",
            "generation_logic": "Extracted directly from OULAD courses.csv table."
        },
        {
            "column_name": "day_offset",
            "data_type": "integer",
            "unit": "Days",
            "valid_range": "0 to 249",
            "generation_logic": "Represents the chronological day index from the start of the presentation."
        },
        {
            "column_name": "marketing_spend",
            "data_type": "float",
            "unit": "USD",
            "valid_range": "10.0 to 2000.0",
            "generation_logic": "Base cost of 10-50, spiking to 500-2000 on days where marketing_campaign_flag is 1."
        },
        {
            "column_name": "marketing_campaign_flag",
            "data_type": "integer",
            "unit": "Binary Flag",
            "valid_range": "0 or 1",
            "generation_logic": "Randomly generated with a 3% probability per day to simulate sporadic ad campaigns."
        },
        {
            "column_name": "holiday_flag",
            "data_type": "integer",
            "unit": "Binary Flag",
            "valid_range": "0 or 1",
            "generation_logic": "Set to 1 during predetermined recurring 5-day periods (e.g., days 90-95, 190-195) to simulate public holidays."
        },
        {
            "column_name": "semester_start_flag",
            "data_type": "integer",
            "unit": "Binary Flag",
            "valid_range": "0 or 1",
            "generation_logic": "Set to 1 for the first 14 days (day_offset < 14) of the presentation."
        },
        {
            "column_name": "semester_end_flag",
            "data_type": "integer",
            "unit": "Binary Flag",
            "valid_range": "0 or 1",
            "generation_logic": "Set to 1 for the last 14 days of the presentation."
        },
        {
            "column_name": "enrollment_num_daily",
            "data_type": "integer",
            "unit": "Students",
            "valid_range": ">= 0",
            "generation_logic": "Sum of linear trend, weekly sine-wave seasonality, semester start spikes, holiday drops, marketing spikes, and Gaussian noise. Clipped at 0."
        }
    ]
    return dict_records


def save_synthetic_outputs(cleaned_df, dictionary):
    synthetic_path = OUTPUT_DIR / "synthetic_marketing_data.csv"
    dictionary_path = OUTPUT_DIR / "synthetic_data_dictionary.json"

    cleaned_df.to_csv(synthetic_path, index=False)
    with dictionary_path.open("w", encoding="utf-8") as f:
        json.dump(dictionary, f, indent=2)

    print(f"Synthetic data saved to {synthetic_path}")
    print(f"Synthetic dictionary saved to {dictionary_path}")


def main():
    synthetic_df, synthetic_dictionary = load_synthetic_data(250)
    save_synthetic_outputs(synthetic_df, synthetic_dictionary)

    return 

def load_synthetic_data(number_days: int):
    courses_df = pd.read_csv(COURSES_PATH)
    synthetic_df = generate_synthetic_data(courses_df, days_per_presentation=number_days)
    synthetic_dictionary = build_data_dictionary()
    return synthetic_df, synthetic_dictionary

if __name__ == "__main__":
    main()