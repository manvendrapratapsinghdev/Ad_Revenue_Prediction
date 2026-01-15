# Sprint Requirements: Ad Revenue Prediction Web Application

**Project Name:** Interactive Ad Revenue Predictor  
**Sprint Duration:** 1-2 Days  
**Developer:** TBD  
**Sprint Start Date:** January 14, 2026

---

## 📋 Executive Summary

Build an interactive Streamlit web application that loads the pre-trained XGBoost model (`model.pkl`) and provides a user-friendly interface for predicting ad revenue. Users can select a date, configure ad parameters, and input impression counts to receive revenue predictions in real-time.

---

## 🎯 Sprint Goals

1. ✅ Create production-ready Streamlit web application
2. ✅ Implement interactive date picker with automatic time-based feature extraction
3. ✅ Build organized input sections with proper labeling and importance indicators
4. ✅ Display predictions with comprehensive input summaries
5. ✅ Include feature importance rankings for user education
6. ✅ Ensure all categorical inputs use only training data options

---

## 📊 Technical Specifications

### Model Information
- **Model Type:** XGBoost Regressor
- **Model File:** `model.pkl` (pre-trained, 17 features)
- **Performance Metrics:**
  - R² Score: 83.41%
  - RMSE: 0.003159
  - MAE: 0.001629

### Input Features (17 Total)

#### Date-Derived Features (Auto-Generated)
1. `hour` - Hour of day (0-23) extracted from datetime picker
2. `day_of_week` - Day of week (0=Monday, 6=Sunday) from date picker
3. `is_weekend` - Binary flag (1=weekend, 0=weekday) auto-calculated

#### Impression Features (User Input - Critical)
4. `measurable_impressions` - Most important feature (42.85% importance) ⭐
5. `total_impressions` - 3rd most important (18.01% importance) ⭐
6. `viewable_impressions` - 6th most important (3.41% importance)
7. `revenue_share_percent` - 7th most important (2.95% importance)

#### Categorical ID Features (Dropdowns - Training Data Only)
8. `ad_type_id` - 2nd most important (18.87% importance) ⭐
9. `geo_id` - 4th most important (4.25% importance)
10. `advertiser_id` - 5th most important (3.73% importance)
11. `monetization_channel_id` - 8th most important (2.54% importance)
12. `ad_unit_id` - 9th most important (1.93% importance)
13. `site_id` - 10th most important (1.87% importance)
14. `device_category_id`
15. `line_item_type_id`
16. `os_id`
17. `integration_type_id`

---

## 🎨 UI/UX Design Specifications

### Sidebar (Left Panel)
**Title:** "📅 Date & Time Selection"

**Components:**
- **Date Picker:** Default to today (January 14, 2026)
- **Time Slider:** Hour selection (0-23), default to current hour
- **Auto-Display:** Show calculated `day_of_week` and `is_weekend` as badges

**Layout Example:**
```
┌─────────────────────────┐
│ 📅 Date & Time Selection│
├─────────────────────────┤
│ Select Date:            │
│ [📆 Jan 14, 2026]      │
│                         │
│ Select Hour:            │
│ [━━━●━━━━━━━] 12       │
│                         │
│ 📊 Calculated:         │
│ • Day: Tuesday (1)      │
│ • Weekend: No           │
└─────────────────────────┘
```

### Main Area - Section 1: "🎯 Critical Impression Inputs"
**Subtitle:** "These features have the highest impact on predictions"

**Components:**
- `measurable_impressions` with ⭐ 42.85% badge - Number input (default: 25)
- `total_impressions` with ⭐ 18.01% badge - Number input (default: 30)
- `viewable_impressions` with 3.41% badge - Number input (default: 20)
- `revenue_share_percent` with 2.95% badge - Number input (default: 1)

**Layout:** 2-column grid with importance indicators

### Main Area - Section 2: "🎛️ Ad Configuration"
**Subtitle:** "High-priority ad parameters"

**Components:**
- `ad_type_id` dropdown with ⭐ 18.87% badge - Options from Dataset.csv
- `geo_id` dropdown with 4.25% badge - Options from Dataset.csv
- `advertiser_id` dropdown with 3.73% badge - Options from Dataset.csv

**Layout:** 3-column grid or vertical stacking

### Main Area - Section 3: "⚙️ Additional Parameters"
**Type:** Expandable/Collapsible Section (collapsed by default)

**Components:**
- `site_id` dropdown
- `device_category_id` dropdown
- `line_item_type_id` dropdown
- `os_id` dropdown
- `integration_type_id` dropdown
- `monetization_channel_id` dropdown
- `ad_unit_id` dropdown

**Layout:** 2-column grid inside expander

### Main Area - Section 4: "🔮 Prediction"
**Components:**
- **Large Predict Button:** "Predict Ad Revenue" (primary color, full width)
- **Result Display:** Large formatted text showing "$X.XXXXXX" in success banner
- **Collapsible "Input Summary":** Table showing all 17 feature values used
- **Collapsible "Model Performance":** Display R², RMSE, MAE metrics

### Main Area - Section 5: "📊 Feature Importance Rankings"
**Type:** Expandable section (collapsed by default)

**Components:**
- Horizontal bar chart or table showing top 10 features
- Format: "Feature Name: XX.XX%"
- Color-coded bars (green for high, yellow for medium, gray for low)

**Content:**
1. measurable_impressions: 42.85%
2. ad_type_id: 18.87%
3. total_impressions: 18.01%
4. geo_id: 4.25%
5. advertiser_id: 3.73%
6. viewable_impressions: 3.41%
7. revenue_share_percent: 2.95%
8. monetization_channel_id: 2.54%
9. ad_unit_id: 1.93%
10. site_id: 1.87%

---

## 🔧 Technical Implementation Requirements

### File Structure
```
ads bidding/
├── app.py                    # Main Streamlit application (NEW)
├── model.pkl                 # Pre-trained model (EXISTS)
├── Dataset.csv               # Training data for dropdown options (EXISTS)
├── requirements.txt          # Updated dependencies (UPDATE)
├── SPRINT_REQUIREMENTS.md    # This file (NEW)
├── model_creation.ipynb      # Model training notebook (EXISTS)
└── README.md                 # Project documentation (EXISTS)
```

### Dependencies (requirements.txt)
```
streamlit>=1.31.0
xgboost>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
```

### Core Functions in app.py

#### 1. `load_model_and_data()`
- Load `model.pkl` (model, label_encoders, feature_cols, performance)
- Load `Dataset.csv` to extract unique values for dropdowns
- Cache with `@st.cache_resource` for performance

#### 2. `extract_date_features(selected_date, selected_hour)`
- Extract hour, day_of_week, is_weekend from datetime
- Return dictionary of date-derived features

#### 3. `preprocess_inputs(input_dict, label_encoders, feature_cols)`
- Apply label encoding to categorical features
- Construct feature array in correct order
- Handle any missing or invalid values
- Return numpy array ready for prediction

#### 4. `predict_revenue(model, features_array)`
- Call `model.predict(features_array)`
- Return predicted revenue value

#### 5. `display_feature_importance()`
- Show expandable section with top 10 features
- Create horizontal bar chart or table
- Color-code by importance level

---

## ✅ Acceptance Criteria

### Must-Have Features
- [x] Streamlit app loads successfully without errors
- [x] Date picker defaults to today's date (January 14, 2026)
- [x] Time slider allows hour selection (0-23)
- [x] Day of week and weekend status are auto-calculated and displayed
- [x] All categorical dropdowns show only options from training data
- [x] Impression inputs have default values and importance badges
- [x] Predict button triggers prediction and shows formatted result
- [x] Result displays as "$X.XXXXXX" format
- [x] Input summary shows all 17 features used for prediction
- [x] Model performance metrics are displayed
- [x] Feature importance section is expandable and shows top 10 features
- [x] Application handles all edge cases gracefully

### User Experience Requirements
- [x] UI is clean, professional, and easy to navigate
- [x] Inputs are logically organized by importance/category
- [x] Loading states are shown during model loading
- [x] Error messages are user-friendly
- [x] Importance indicators help users prioritize inputs
- [x] Responsive layout works on different screen sizes

### Technical Requirements
- [x] Model loads only once using caching
- [x] Label encoding matches training data
- [x] Feature array order matches model expectations
- [x] No hardcoded values for dropdown options
- [x] Code is well-commented and maintainable
- [x] Error handling for missing files or invalid inputs

---

## 🚀 Deployment Considerations

### Running Locally
```bash
cd "ads bidding"
pip install -r requirements.txt
streamlit run app.py
```

### Cloud Deployment Options
- **Streamlit Cloud:** Direct GitHub integration
- **Heroku:** Add Procfile with `web: streamlit run app.py`
- **AWS/GCP:** Docker containerization recommended

### Environment Variables
- None required (all data in local files)

---

## 📝 Testing Checklist

### Functional Testing
- [ ] Date picker works and extracts correct features
- [ ] All dropdowns populate with correct values
- [ ] Impression inputs accept valid numeric values
- [ ] Predict button triggers prediction successfully
- [ ] Prediction result displays in correct format
- [ ] Input summary matches actual inputs
- [ ] Feature importance chart displays correctly

### Edge Case Testing
- [ ] Handle missing model.pkl file
- [ ] Handle missing Dataset.csv file
- [ ] Handle invalid input values (negative numbers, etc.)
- [ ] Handle extreme dates (past/future)
- [ ] Handle unknown categorical values (shouldn't happen with dropdowns)

### Performance Testing
- [ ] Model loads in < 2 seconds
- [ ] Prediction completes in < 1 second
- [ ] UI remains responsive during operations

---

## 📈 Success Metrics

1. **Application Launch:** Successfully starts without errors
2. **Prediction Accuracy:** Matches predictions from notebook testing
3. **User Experience:** Intuitive interface requiring no instructions
4. **Performance:** Fast load times and responsive interactions
5. **Robustness:** Handles edge cases without crashing

---

## 🔄 Future Enhancements (Out of Scope)

- Batch prediction from CSV upload
- Historical prediction tracking
- A/B testing different model parameters
- Integration with live ad serving systems
- Export predictions to CSV/Excel
- Advanced visualizations (charts, trends)
- User authentication and session management

---

## 👥 Stakeholders

- **Development Team:** Implementation and testing
- **Data Science Team:** Model validation and feature interpretation
- **Business Users:** End users for revenue predictions
- **Product Manager:** Requirements and acceptance testing

---

## 📞 Support & Documentation
- **Technical Issues:** Reference model_creation.ipynb for training details
- **Model Questions:** Check performance metrics in model.pkl
- **Feature Questions:** See feature importance rankings in app

---

**Document Version:** 1.0  
**Last Updated:** January 14, 2026  
**Status:** Ready for Development ✅
