import os
import pandas as pd

FEATURE_DIR = os.path.dirname(__file__)
INPUT_CSV = os.path.join(FEATURE_DIR, "feature_table.csv")
OUTPUT_CSV = os.path.join(FEATURE_DIR, "feature_engineered_table.csv")

feature_table = pd.read_csv(INPUT_CSV)
df = feature_table.copy()

# -----------------------------
# Create a date from day_offset
# -----------------------------
# Choose the dataset start date (change if known)
start_date = pd.Timestamp("2024-01-01")

df["date"] = start_date + pd.to_timedelta(df["day_offset"], unit="D")

# -----------------------------
# Time Features
# -----------------------------
df["day_of_week"] = df["date"].dt.dayofweek
df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
df["month"] = df["date"].dt.month
df["quarter"] = df["date"].dt.quarter

# -----------------------------
# Target column
# -----------------------------
target = "enrollment_num_daily"

# -----------------------------
# Lag Features
# -----------------------------
df["lag_1"] = df[target].shift(1)
df["lag_7"] = df[target].shift(7)
df["lag_14"] = df[target].shift(14)
df["lag_30"] = df[target].shift(30)

# -----------------------------
# Rolling Features
# -----------------------------
df["rolling_mean_7"] = df[target].rolling(7).mean()
df["rolling_mean_30"] = df[target].rolling(30).mean()
df["rolling_std_30"] = df[target].rolling(30).std()

# -----------------------------
# Rename event columns
# -----------------------------
df.rename(columns={
    "marketing_campaign_flag": "campaign_flag",
    "semester_start_flag": "semester_start"
}, inplace=True)

# Remove rows with NaN created by lag/rolling
df = df.dropna().reset_index(drop=True)

# Save engineered feature table
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved engineered feature table to: {OUTPUT_CSV}")

# Ensure time-based split uses the date order
if "date" in df.columns:
    df = df.sort_values("date").reset_index(drop=True)

# Select features and target for time-based split
X = df.drop(columns=[target, "date"])
y = df[target]

# Time-based split (80% train, 20% test)
split = int(len(df) * 0.8)
X_train = X.iloc[:split]
X_test = X.iloc[split:]
y_train = y.iloc[:split]
y_test = y.iloc[split:]

TRAIN_CSV = os.path.join(FEATURE_DIR, "feature_engineered_train.csv")
TEST_CSV = os.path.join(FEATURE_DIR, "feature_engineered_test.csv")

train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

train_df.to_csv(TRAIN_CSV, index=False)
test_df.to_csv(TEST_CSV, index=False)
print(f"Saved train set to: {TRAIN_CSV}")
print(f"Saved test set to: {TEST_CSV}")

# Split the test set into separate day_offset files
TEST_DAY_DIR = os.path.join(FEATURE_DIR, "feature_engineered_test_day_splits")
os.makedirs(TEST_DAY_DIR, exist_ok=True)
for day_offset, subset in test_df.groupby("day_offset"):
    day_file = os.path.join(TEST_DAY_DIR, f"feature_engineered_test_day_{day_offset}.csv")
    subset.to_csv(day_file, index=False)
print(f"Saved per-day test split files to: {TEST_DAY_DIR}")

# Display engineered features
print(df[[
    "date",
    "day_of_week",
    "week_of_year",
    "month",
    "quarter",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_30",
    "rolling_mean_7",
    "rolling_mean_30",
    "rolling_std_30",
    "holiday_flag",
    "campaign_flag",
    "semester_start"
]].head())