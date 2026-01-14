# Simple Prediction Script
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load model
print("Loading model...")
with open('model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
label_encoders = model_data['label_encoders']
feature_cols = model_data['feature_cols']

# Load test data
print("Loading test data...")
df_sample = pd.read_csv('Dataset.csv', nrows=5)

# Preprocess (same as training)
df_sample['date'] = pd.to_datetime(df_sample['date'])
# Change year to 2025
df_sample['date'] = df_sample['date'].apply(lambda x: x.replace(year=2025))
df_sample['hour'] = df_sample['date'].dt.hour
df_sample['day_of_week'] = df_sample['date'].dt.dayofweek
df_sample['is_weekend'] = df_sample['day_of_week'].isin([5, 6]).astype(int)

# Prepare features
X_sample = df_sample[feature_cols].copy()

# Apply label encoding
for col in label_encoders.keys():
    if col in X_sample.columns:
        le = label_encoders[col]
        X_sample[col] = X_sample[col].astype(str)
        X_sample[col] = X_sample[col].apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )

# Predict
print("Making predictions...")
predictions = model.predict(X_sample)

# Show results with inputs
print("\n" + "="*80)
print("PREDICTION RESULTS")
print("="*80)

for i in range(len(predictions)):
    print(f"\nSample {i+1}:")
    print("-" * 40)
    
    # Show input parameters
    print("INPUT:")
    print(f"  Date: {df_sample['date'].iloc[i]}")
    print(f"  Site ID: {df_sample['site_id'].iloc[i]}")
    print(f"  Ad Type ID: {df_sample['ad_type_id'].iloc[i]}")
    print(f"  Geo ID: {df_sample['geo_id'].iloc[i]}")
    print(f"  Device Category ID: {df_sample['device_category_id'].iloc[i]}")
    print(f"  Total Impressions: {df_sample['total_impressions'].iloc[i]}")
    print(f"  Viewable Impressions: {df_sample['viewable_impressions'].iloc[i]}")
    print(f"  Measurable Impressions: {df_sample['measurable_impressions'].iloc[i]}")
    
    # Show predicted output
    print(f"\nPREDICTED REVENUE: ${predictions[i]:.6f}")

print("\n" + "="*80)
print("✓ Done!")
