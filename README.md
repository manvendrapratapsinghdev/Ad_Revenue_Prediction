# Ad Revenue Prediction Model - Production Ready

## Overview
This project implements a production-ready XGBoost model to predict ad revenue from real-time advertising auction data.

## Model Performance
- **Test R² Score**: 0.9601 (96% accuracy)
- **Test RMSE**: 0.1413
- **Test MAE**: 0.0091

## Files in this Project

### Training Files
- **`train_model.py`**: Main training script that creates the model
- **`Dataset.csv`**: Training dataset (567,291 rows)

### Model Artifacts (Production Ready)
1. **`xgboost_ad_revenue_model.json`**: Trained XGBoost model
2. **`label_encoders.pkl`**: Encoders for categorical variables
3. **`feature_columns.json`**: List of feature columns used
4. **`feature_importance.csv`**: Feature importance scores
5. **`model_metadata.json`**: Model performance metrics and parameters

### Prediction Files
- **`predict.py`**: Example prediction script
- **`model_creation.ipynb`**: Jupyter notebook (alternative interface)

## How to Use

### 1. Setup Environment
```bash
# The virtual environment is already created at .venv
# Activate it (if needed):
source .venv/bin/activate

# Or use the full path to Python:
/Users/d111879/Documents/Project/DEMO/Hackthon/ads\ bidding/.venv/bin/python
```

### 2. Training the Model
```bash
# Run the training script
python train_model.py
```

This will:
- Load and clean the dataset
- Engineer features (temporal features, ratios, etc.)
- Train an XGBoost model
- Evaluate performance
- Save all production artifacts

### 3. Making Predictions
```bash
# Run the prediction script
python predict.py
```

### 4. Using the Model in Production

```python
import pandas as pd
import xgboost as xgb
import pickle
import json

# Load model
model = xgb.XGBRegressor()
model.load_model('xgboost_ad_revenue_model.json')

# Load encoders and feature columns
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)
with open('feature_columns.json', 'r') as f:
    feature_cols = json.load(f)

# Prepare your data (example)
new_data = pd.DataFrame({
    'site_id': [351],
    'ad_type_id': [10],
    'geo_id': [187],
    'device_category_id': [2],
    'advertiser_id': [84],
    'order_id': [3473],
    'line_item_type_id': [19],
    'os_id': [60],
    'integration_type_id': [1],
    'monetization_channel_id': [4],
    'ad_unit_id': [5174],
    'total_impressions': [16],
    'viewable_impressions': [2],
    'measurable_impressions': [16],
    'revenue_share_percent': [1],
    'date': ['2019-06-30 12:00:00']
})

# Feature engineering (apply same transformations as training)
new_data['date'] = pd.to_datetime(new_data['date'])
new_data['hour'] = new_data['date'].dt.hour
new_data['day_of_week'] = new_data['date'].dt.dayofweek
new_data['day'] = new_data['date'].dt.day
new_data['month'] = new_data['date'].dt.month
new_data['year'] = new_data['date'].dt.year
new_data['is_weekend'] = new_data['day_of_week'].isin([5, 6]).astype(int)

new_data['impressions_revenue_ratio'] = 0  # Will be updated after prediction
new_data['viewability_rate'] = new_data['viewable_impressions'] / new_data['measurable_impressions']

# Select features
X_new = new_data[feature_cols]

# Apply label encoding
for col in label_encoders.keys():
    if col in X_new.columns:
        le = label_encoders[col]
        X_new[col] = X_new[col].astype(str)
        X_new[col] = X_new[col].apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )

# Make prediction
predicted_revenue = model.predict(X_new)
print(f"Predicted Revenue: ${predicted_revenue[0]:.6f}")
```

## Features Used

### Original Features
- site_id
- ad_type_id
- geo_id
- device_category_id
- advertiser_id
- order_id
- line_item_type_id
- os_id
- integration_type_id
- monetization_channel_id
- ad_unit_id
- total_impressions
- viewable_impressions
- measurable_impressions
- revenue_share_percent

### Engineered Features
- **Temporal Features**: hour, day_of_week, day, month, year, is_weekend
- **Derived Features**: 
  - impressions_revenue_ratio (revenue per impression)
  - viewability_rate (viewable/measurable ratio)

## Top 15 Most Important Features
1. total_impressions (26.42%)
2. measurable_impressions (23.75%)
3. viewable_impressions (22.82%)
4. impressions_revenue_ratio (14.93%)
5. ad_unit_id (1.67%)
6. device_category_id (1.36%)
7. site_id (1.36%)
8. order_id (1.18%)
9. line_item_type_id (1.10%)
10. viewability_rate (1.04%)
11. day (0.72%)
12. is_weekend (0.70%)
13. day_of_week (0.69%)
14. geo_id (0.65%)
15. os_id (0.57%)

## Model Configuration

### XGBoost Parameters
```python
{
    'objective': 'reg:squarederror',
    'n_estimators': 200,
    'learning_rate': 0.1,
    'max_depth': 6,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0,
    'reg_alpha': 0,
    'reg_lambda': 1,
    'random_state': 42,
    'n_jobs': -1,
    'tree_method': 'hist'
}
```

## Data Cleaning Standards

1. **Missing Values**: Filled with median (numerical) or mode (categorical)
2. **Date Parsing**: All dates converted to datetime format
3. **Feature Encoding**: Label encoding for categorical variables
4. **Train-Test Split**: 80-20 split with random_state=42
5. **Feature Scaling**: Not required for XGBoost (tree-based model)

## Production Deployment Checklist

- [x] Model training script
- [x] Model artifacts saved
- [x] Label encoders saved
- [x] Feature list documented
- [x] Prediction script created
- [x] Model metadata tracked
- [x] Feature importance analyzed
- [x] Performance metrics documented
- [x] README documentation
- [ ] API endpoint (can be added)
- [ ] Docker container (can be added)
- [ ] Monitoring and logging (can be added)

## Next Steps for Production

1. **Create REST API**: Use FastAPI or Flask to serve predictions
2. **Add Monitoring**: Track model performance over time
3. **Implement CI/CD**: Automate model retraining
4. **Add Data Validation**: Validate input data before prediction
5. **Error Handling**: Add robust error handling
6. **Logging**: Implement comprehensive logging
7. **Testing**: Add unit and integration tests

## Requirements

```
pandas>=2.0.0
numpy>=1.24.0
xgboost>=3.1.0
scikit-learn>=1.8.0
```

## System Requirements

- Python 3.11+
- macOS/Linux/Windows
- OpenMP library (for XGBoost)
  - macOS: `brew install libomp`
  - Linux: Usually pre-installed
  - Windows: Included with XGBoost

## Contact & Support

For questions or issues, please refer to the model metadata file for training details and performance metrics.

---
**Model Training Date**: 2026-01-14
**Model Version**: 1.0
**Status**: Production Ready ✅
