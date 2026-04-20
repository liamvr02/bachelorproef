from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error
import numpy as np

# === 1. Load CSV ===
path_to_csv = Path(__file__).parent.parent / "sample_stream_output.csv"

df = pd.read_csv(path_to_csv)

# === 2. Define columns ===
target_column = "temperature"

feature_columns = [
    "dhm1_elevation",
    "trees_count50m_count",
    "ua_ua_dense_built_up_100m_frac",
    "ua_ua_mixed_urban_100m_frac",
    "ua_ua_transport_infrastructure_100m_frac",
    "ua_ua_bare_sparse_100m_frac",
    "ua_ua_water_wetlands_100m_frac",
    "ua_ua_dense_built_up_30m_frac",
    "ua_ua_mixed_urban_30m_frac",
    "ua_ua_transport_infrastructure_30m_frac",
    "ua_ua_bare_sparse_30m_frac",
    "ua_ua_water_wetlands_30m_frac",
    "timestamp"   # include raw timestamp column
]

# === 3. Convert timestamp ===
# Adjust format if needed, e.g. format="%Y-%m-%d %H:%M:%S"
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# Extract useful time features
df["year"] = df["timestamp"].dt.year
df["day_of_year"] = df["timestamp"].dt.dayofyear
df["hour"] = df["timestamp"].dt.hour

# Drop original timestamp if not needed
df = df.drop(columns=["timestamp"])

# Update feature list
feature_columns = feature_columns[:-1] + ["year", "day_of_year", "hour"]

# === 4. Clean data ===
df = df.sort_values(["year", "day_of_year"])

# === 5. Time-based split ===
# Example: train on earlier data, test on later data

split_year = df["year"].quantile(0.8)  # or choose explicitly
split_doy = 200  # optional fine control within a year

train_df = df[
    (df["year"] < split_year) |
    ((df["year"] == split_year) & (df["day_of_year"] <= split_doy))
]

test_df = df.drop(train_df.index)

X_train = train_df[feature_columns]
y_train = train_df[target_column]

X_test = test_df[feature_columns]
y_test = test_df[target_column]

# === 6. Train model ===
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# === 7. Evaluate ===
y_pred = model.predict(X_test)

y_pred = model.predict(X_test)

mse = np.mean((y_test - y_pred) ** 2)
rmse = np.sqrt(mse)

print("R2:", r2_score(y_test, y_pred))
print("MSE:", mse)
print("RMSE:", rmse)

# === 8. Feature importance ===
importances = pd.Series(model.feature_importances_, index=feature_columns)
print("\nFeature Importances:")
print(importances.sort_values(ascending=False))