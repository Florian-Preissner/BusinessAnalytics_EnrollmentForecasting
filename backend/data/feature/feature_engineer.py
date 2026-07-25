import os
import pandas as pd

FEATURE_DIR = os.path.dirname(__file__)
INPUT_CSV = os.path.join(FEATURE_DIR, "feature_table.csv")
OUTPUT_CSV = os.path.join(FEATURE_DIR, "feature_engineered_table.csv")

feature_table = pd.read_csv(INPUT_CSV)
df = feature_table.copy()

df = feature_table.copy()
TARGET = "enrollment_num_daily"

# -----------------------------
# Create a date from day_offset
# -----------------------------
# Choose the dataset start date (change if known)
start_date = pd.Timestamp("2014-01-01")

df["date"] = start_date + pd.to_timedelta(df["day_offset"], unit="D")

# -----------------------------
# Create a date from day_offset
# -----------------------------
df["code_module_original"] = df["code_module"]
df = pd.get_dummies(df, columns=['code_module'], prefix='module') # module: one-hot
df.rename(columns={"code_module_original": "code_module"}, inplace=True)

# -----------------------------
# Time Features
# -----------------------------
df["day_of_week"] = df["date"].dt.dayofweek
df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
df["month"] = df["date"].dt.month
df["quarter"] = df["date"].dt.quarter
df["year"] = df['code_presentation'].str[:4].astype(int)
df['semester'] = df['code_presentation'].str[-1]

df = pd.get_dummies(df, columns=['semester'], prefix='sem') # semester: one-hot
# -----------------------------
# Lag Features
# -----------------------------
df["lag_1"] = df[TARGET].shift(1)
df["lag_7"] = df[TARGET].shift(7)
df["lag_14"] = df[TARGET].shift(14)
df["lag_30"] = df[TARGET].shift(30)

# -----------------------------
# Rolling Features
# -----------------------------
df["rolling_mean_7"] = df[TARGET].rolling(7).mean()
df["rolling_mean_30"] = df[TARGET].rolling(30).mean()
df["rolling_std_30"] = df[TARGET].rolling(30).std()

# -----------------------------
# Rename event columns
# -----------------------------
df.rename(columns={
    "marketing_campaign_flag": "campaign_flag",
    "semester_start_flag": "semester_start"
}, inplace=True)

# Remove rows with NaN created by lag/rolling
df = df.dropna().reset_index(drop=True)
# drop unnecessary cols
df = df.drop(columns=["module_presentation_length", 'campaign_flag','is_exam_included','quarter', "date", 'week_of_year'], errors="ignore")

# Time-based split (80% train, 20% test)
# Select features and target for time-based split
X = df.drop(columns=[TARGET])
y = df[TARGET]

print(X.columns)

split = int(len(df) * 0.8)
X_train = X.iloc[:split]
X_test = X.iloc[split:]
y_train = y.iloc[:split]
y_test = y.iloc[split:]

print(X_train.columns)
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

