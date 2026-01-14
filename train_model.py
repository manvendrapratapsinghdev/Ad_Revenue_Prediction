# Simple XGBoost Model Training - Generate Pickle File
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings
warnings.filterwarnings('ignore')

print("Loading dataset...")
df = pd.read_csv('Dataset.csv')

# Basic preprocessing
df['date'] = pd.to_datetime(df['date'])
df['hour'] = df['date'].dt.hour
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Fill missing values
df['total_revenue'].fillna(0, inplace=True)

# Prepare features
exclude_cols = ['date', 'total_revenue']
feature_cols = [col for col in df.columns if col not in exclude_cols]
X = df[feature_cols].copy()
y = df['total_revenue'].copy()

# Encode categorical columns
label_encoders = {}
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("Training model...")
model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Test
score = model.score(X_test, y_test)
print(f"Model R² Score: {score:.4f}")

# Save as pickle
model_data = {
    'model': model,
    'label_encoders': label_encoders,
    'feature_cols': feature_cols
}

with open('model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("✓ Model saved to 'model.pkl'")
