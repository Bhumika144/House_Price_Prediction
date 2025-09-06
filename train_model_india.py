# train_model_india.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import pickle

# Load dataset
df = pd.read_csv("india_housing_prices.csv")

# Select features + target
features = ["City", "Locality", "Property_Type", "BHK", "Size_in_SqFt",
            "Furnished_Status", "Parking_Space", "Year_Built"]
target = "Price_in_Lakhs"

X = df[features]
y = df[target]

# Handle missing values
X = X.fillna("Unknown")
y = y.fillna(y.median())

# Define column groups
categorical_cols = ["City", "Locality", "Property_Type", "Furnished_Status", "Parking_Space"]
numeric_cols = ["BHK", "Size_in_SqFt", "Year_Built"]

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", StandardScaler(), numeric_cols)
])

# Model pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=30, random_state=42))  # keep it fast
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
print("🚀 Training model...")
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"✅ Model trained. R² Score: {r2:.3f}, MAE: {mae:.2f} Lakhs")

# Save trained pipeline (model + preprocessing together)
with open("house_model_india.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("📂 Model saved as house_model_india.pkl")
