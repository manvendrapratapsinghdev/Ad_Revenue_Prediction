"""
Interactive Ad Revenue Prediction Application
==============================================
A Streamlit web application for predicting ad revenue using a pre-trained XGBoost model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, date, time
import warnings
warnings.filterwarnings('ignore')

# ============================
# PAGE CONFIGURATION
# ============================

st.set_page_config(
    page_title="Ad Revenue Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================
# LOAD MODEL AND DATA
# ============================

@st.cache_resource
def load_model_and_data():
    """Load the trained model and extract dropdown options from training data"""
    try:
        # Load model
        with open('model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        # Load dataset for dropdown options
        df = pd.read_csv('Dataset.csv')
        
        # Extract unique values for categorical features
        categorical_options = {}
        categorical_features = [
            'site_id', 'ad_type_id', 'geo_id', 'device_category_id', 
            'advertiser_id', 'line_item_type_id', 'os_id', 
            'integration_type_id', 'monetization_channel_id', 'ad_unit_id'
        ]
        
        for feature in categorical_features:
            if feature in df.columns:
                categorical_options[feature] = sorted(df[feature].unique().tolist())
        
        # Calculate default values for impression features
        impression_defaults = {
            'measurable_impressions': int(df['measurable_impressions'].median()),
            'total_impressions': int(df['total_impressions'].median()),
            'viewable_impressions': int(df['viewable_impressions'].median())
        }
        
        # Calculate ranges for impression features
        impression_ranges = {
            'measurable_impressions': (int(df['measurable_impressions'].min()), int(df['measurable_impressions'].max())),
            'total_impressions': (int(df['total_impressions'].min()), int(df['total_impressions'].max())),
            'viewable_impressions': (int(df['viewable_impressions'].min()), int(df['viewable_impressions'].max()))
        }
        
        return model_data, categorical_options, impression_defaults, impression_ranges
    
    except FileNotFoundError as e:
        st.error("🚫 **Application Setup Required**")
        st.warning("""
        **Missing Required Files**
        
        Please ensure the following files are in the application directory:
        - `model.pkl` - The trained machine learning model
        - `Dataset.csv` - The training dataset for reference values
        
        **To set up the application:**
        1. Run `python train_model.py` to train and generate the model
        2. Ensure `Dataset.csv` is in the same directory
        3. Restart the application
        """)
        st.info(f"📁 Missing file: `{str(e).split(': ')[1] if ': ' in str(e) else e}`")
        st.stop()
    except Exception as e:
        st.error("⚠️ **Application Error**")
        st.warning("""
        **Failed to Load Model Resources**
        
        There was an error loading the required model files. This could be due to:
        - Corrupted model file (`model.pkl`)
        - Incompatible model version
        - Missing dependencies
        
        **To resolve:**
        1. Delete `model.pkl` if it exists
        2. Run `python train_model.py` to regenerate the model
        3. Restart the application
        """)
        st.info(f"🔍 Technical details: `{str(e)}`")
        st.stop()

# Load resources
model_data, categorical_options, impression_defaults, impression_ranges = load_model_and_data()
model = model_data['model']
label_encoders = model_data['label_encoders']
feature_cols = model_data['feature_cols']
performance = model_data['performance']

# ============================
# HELPER FUNCTIONS
# ============================

def extract_date_features(selected_date, selected_hour):
    """Extract time-based features from selected date and hour"""
    # Create datetime
    dt = datetime.combine(selected_date, time(hour=selected_hour))
    
    # Extract features
    hour = dt.hour
    day_of_week = dt.weekday()  # 0=Monday, 6=Sunday
    is_weekend = 1 if day_of_week >= 5 else 0
    
    return {
        'hour': hour,
        'day_of_week': day_of_week,
        'is_weekend': is_weekend
    }

def preprocess_inputs(input_dict, label_encoders, feature_cols):
    """Preprocess inputs and apply label encoding"""
    try:
        # Create a copy of inputs
        processed = input_dict.copy()
        
        # Apply label encoding to categorical features
        for col, encoder in label_encoders.items():
            if col in processed:
                value = str(processed[col])
                try:
                    processed[col] = encoder.transform([value])[0]
                except ValueError:
                    # Handle unknown category (shouldn't happen with dropdowns)
                    st.warning(f"⚠️ Unknown value for {col}: {value}. Using default.")
                    processed[col] = 0
        
        # Create feature array in correct order
        feature_array = np.array([processed[col] for col in feature_cols]).reshape(1, -1)
        
        return feature_array
    
    except Exception as e:
        st.error(f"❌ Error preprocessing inputs: {e}")
        return None

def predict_revenue(model, features_array):
    """Make revenue prediction"""
    try:
        prediction = model.predict(features_array)[0]
        return prediction
    except Exception as e:
        st.error(f"❌ Error making prediction: {e}")
        return None

# ============================
# MAIN APPLICATION UI
# ============================

# Header
st.title("💰 Ad Revenue Prediction System")
st.markdown("Powered by XGBoost")
st.markdown("---")

# ============================
# SIDEBAR - DATE & TIME SELECTION
# ============================

with st.sidebar:
    st.header("📅 Date & Time Selection")
    st.markdown("Select when the ad will be shown")
    
    # Date picker (default to today)
    selected_date = st.date_input(
        "Select Date",
        value=date(2026, 1, 14),  # Today's date
        help="Choose the date for ad impression"
    )
    
    # Hour slider
    selected_hour = st.slider(
        "Select Hour (0-23)",
        min_value=0,
        max_value=23,
        value=12,
        help="Choose the hour of day (0 = midnight, 12 = noon, 23 = 11 PM)"
    )
    
    # Extract and display calculated features
    date_features = extract_date_features(selected_date, selected_hour)
    
    st.markdown("---")
    st.subheader("📊 Calculated Features")
    
    # Day of week names
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_name = day_names[date_features['day_of_week']]
    
    # Display in separate rows
    st.metric("Day", f"{day_name}")
    st.markdown(f"Day of Week: {date_features['day_of_week']}")
    
    weekend_status = "Yes ✅" if date_features['is_weekend'] else "No ❌"
    st.markdown(f"Weekend: {weekend_status}")
    
    st.markdown("---")
    st.markdown("### 📈 Model Performance")
    st.metric("R² Score", f"{performance['test_r2']:.2%}")
    st.metric("RMSE", f"₹{performance['test_rmse']*85:.4f}")
    st.metric("MAE", f"₹{performance['test_mae']*85:.4f}")

# ============================
# MAIN AREA - INPUT SECTIONS
# ============================

# Section 1: Critical Impression Inputs
st.header("🎯 Critical Impression Inputs")
st.markdown("*These features have the highest impact on predictions*")

col1, col2, col3 = st.columns(3)

with col1:
    measurable_impressions = st.number_input(
        "Measurable Impressions ⭐",
        min_value=0,
        value=impression_defaults['measurable_impressions'],
        help="Most important feature (42.85% importance)"
    )
    st.caption("⭐ Importance: 42.85%")
    st.caption(f"📊 Range: {impression_ranges['measurable_impressions'][0]:,} - {impression_ranges['measurable_impressions'][1]:,}")

with col2:
    total_impressions = st.number_input(
        "Total Impressions ⭐",
        min_value=0,
        value=impression_defaults['total_impressions'],
        help="3rd most important feature (18.01% importance)"
    )
    st.caption("⭐ Importance: 18.01%")
    st.caption(f"📊 Range: {impression_ranges['total_impressions'][0]:,} - {impression_ranges['total_impressions'][1]:,}")

with col3:
    viewable_impressions = st.number_input(
        "Viewable Impressions",
        min_value=0,
        value=impression_defaults['viewable_impressions'],
        help="6th most important feature (3.41% importance)"
    )
    st.caption("Importance: 3.41%")
    st.caption(f"📊 Range: {impression_ranges['viewable_impressions'][0]:,} - {impression_ranges['viewable_impressions'][1]:,}")

# Revenue share percent is hardcoded to 1 (always constant)
revenue_share_percent = 1

st.markdown("---")

# Section 2: Ad Configuration
st.header("🎛️ Ad Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    ad_type_id = st.selectbox(
        "Ad Type ID ⭐",
        options=categorical_options.get('ad_type_id', []),
        help="2nd most important feature (18.87% importance)"
    )
    st.caption("⭐ Importance: 18.87%")

with col2:
    geo_id = st.selectbox(
        "Geographic ID",
        options=categorical_options.get('geo_id', []),
        help="4th most important feature (4.25% importance)"
    )
    st.caption("Importance: 4.25%")

with col3:
    advertiser_id = st.selectbox(
        "Advertiser ID",
        options=categorical_options.get('advertiser_id', []),
        help="5th most important feature (3.73% importance)"
    )
    st.caption("Importance: 3.73%")

st.markdown("---")

# Section 3: Additional Parameters (Expandable)
with st.expander("⚙️ Additional Parameters", expanded=False):
    st.markdown("*Other configuration options*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        site_id = st.selectbox(
            "Site ID",
            options=categorical_options.get('site_id', []),
            help="Website/platform identifier (1.87% importance)"
        )
        
        device_category_id = st.selectbox(
            "Device Category ID",
            options=categorical_options.get('device_category_id', []),
            help="Device type (mobile, desktop, tablet)"
        )
        
        line_item_type_id = st.selectbox(
            "Line Item Type ID",
            options=categorical_options.get('line_item_type_id', []),
            help="Type of line item"
        )
        
        os_id = st.selectbox(
            "Operating System ID",
            options=categorical_options.get('os_id', []),
            help="Operating system identifier"
        )
    
    with col2:
        integration_type_id = st.selectbox(
            "Integration Type ID",
            options=categorical_options.get('integration_type_id', []),
            help="Integration method"
        )
        
        monetization_channel_id = st.selectbox(
            "Monetization Channel ID",
            options=categorical_options.get('monetization_channel_id', []),
            help="Revenue channel (2.54% importance)"
        )
        
        ad_unit_id = st.selectbox(
            "Ad Unit ID",
            options=categorical_options.get('ad_unit_id', []),
            help="Ad unit identifier (1.93% importance)"
        )

st.markdown("---")

# ============================
# PREDICTION SECTION
# ============================

st.header("🔮 Revenue Prediction")

# Large predict button
predict_button = st.button("🚀 Predict Ad Revenue", type="primary", use_container_width=True)

if predict_button:
    # Collect all inputs
    input_dict = {
        # Date features
        'hour': date_features['hour'],
        'day_of_week': date_features['day_of_week'],
        'is_weekend': date_features['is_weekend'],
        
        # Impression features
        'measurable_impressions': measurable_impressions,
        'total_impressions': total_impressions,
        'viewable_impressions': viewable_impressions,
        'revenue_share_percent': revenue_share_percent,
        
        # Categorical features
        'ad_type_id': ad_type_id,
        'geo_id': geo_id,
        'advertiser_id': advertiser_id,
        'site_id': site_id,
        'device_category_id': device_category_id,
        'line_item_type_id': line_item_type_id,
        'os_id': os_id,
        'integration_type_id': integration_type_id,
        'monetization_channel_id': monetization_channel_id,
        'ad_unit_id': ad_unit_id
    }
    
    # Preprocess inputs
    with st.spinner("Processing inputs..."):
        features_array = preprocess_inputs(input_dict, label_encoders, feature_cols)
    
    if features_array is not None:
        # Make prediction
        with st.spinner("Generating prediction..."):
            prediction = predict_revenue(model, features_array)
        
        if prediction is not None:
            # Display result in success banner
            prediction_inr = prediction * 85
            st.success(f"### 💵 Predicted Revenue: ₹{prediction_inr:.4f}")
            st.caption(f"(Approximately ${prediction:.6f} USD at exchange rate: 1 USD = 85 INR)")
            st.markdown("<p style='font-size: 0.85em; color: #888;'>Note: Original model was trained on USD values. Predictions are converted to INR for better understanding.</p>", unsafe_allow_html=True)

            
            # Create columns for additional info
            # col1, col2 = st.columns(2)
            
            # with col1:
                # Input Summary
            with st.expander("📋 Input Summary", expanded=False):
                    st.markdown("**Date & Time Features:**")
                    st.write(f"- Date: {selected_date}")
                    st.write(f"- Hour: {date_features['hour']}")
                    st.write(f"- Day of Week: {day_name} ({date_features['day_of_week']})")
                    st.write(f"- Weekend: {'Yes' if date_features['is_weekend'] else 'No'}")
                    
                    st.markdown("**Impression Features:**")
                    st.write(f"- Measurable Impressions: {measurable_impressions:,}")
                    st.write(f"- Total Impressions: {total_impressions:,}")
                    st.write(f"- Viewable Impressions: {viewable_impressions:,}")
                    st.write(f"- Revenue Share: {revenue_share_percent}% (fixed)")
                    
                    st.markdown("**Categorical Features:**")
                    st.write(f"- Ad Type ID: {ad_type_id}")
                    st.write(f"- Geo ID: {geo_id}")
                    st.write(f"- Advertiser ID: {advertiser_id}")
                    st.write(f"- Site ID: {site_id}")
                    st.write(f"- Device Category: {device_category_id}")
                    st.write(f"- Other IDs: {line_item_type_id}, {os_id}, {integration_type_id}, {monetization_channel_id}, {ad_unit_id}")
            
            # with col2:
            #     # Model Performance
            #     with st.expander("📊 Model Performance Metrics", expanded=True):
            #         st.metric("R² Score (Accuracy)", f"{performance['test_r2']:.2%}")
            #         st.metric("RMSE", f"₹{performance['test_rmse']*85:.4f}")
            #         st.metric("MAE", f"₹{performance['test_mae']*85:.4f}")
                    
            #         st.markdown("---")
            #         st.caption("These metrics show how well the model performed on test data.")
            #         st.caption("💱 Currency: INR (Indian Rupees) | Exchange Rate: 1 USD = 85 INR")

st.markdown("---")

# ============================
# FEATURE IMPORTANCE SECTION
# ============================

with st.expander("📊 Feature Importance Rankings", expanded=False):
    st.markdown("### Understanding Feature Impact on Predictions")
    st.markdown("*Higher percentages indicate greater influence on revenue predictions*")
    
    # Feature importance data (top 10)
    importance_data = {
        'Feature': [
            'Measurable Impressions',
            'Ad Type ID',
            'Total Impressions',
            'Geographic ID',
            'Advertiser ID',
            'Viewable Impressions',
            'Revenue Share %',
            'Monetization Channel ID',
            'Ad Unit ID',
            'Site ID'
        ],
        'Importance (%)': [42.85, 18.87, 18.01, 4.25, 3.73, 3.41, 2.95, 2.54, 1.93, 1.87],
        'Category': [
            'Impression', 'Categorical', 'Impression', 'Categorical', 'Categorical',
            'Impression', 'Impression', 'Categorical', 'Categorical', 'Categorical'
        ]
    }
    
    importance_df = pd.DataFrame(importance_data)
    
    # Display as horizontal bar chart
    st.bar_chart(importance_df.set_index('Feature')['Importance (%)'], use_container_width=True)
    
    # Display as table
    st.markdown("### Detailed Rankings")
    
    # Color-code by importance level
    def color_importance(val):
        if val > 15:
            return 'background-color: #90EE90'  # Light green
        elif val > 5:
            return 'background-color: #FFD700'  # Gold
        else:
            return 'background-color: #D3D3D3'  # Light gray
    
    styled_df = importance_df.style.applymap(color_importance, subset=['Importance (%)'])
    st.dataframe(styled_df, use_container_width=True)
    
    st.markdown("---")
    st.info("💡 **Tip:** Focus on the top 3 features (Measurable Impressions, Ad Type ID, and Total Impressions) as they account for nearly 80% of the prediction power!")

# ============================
# FOOTER
# ============================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p style='margin-top: 15px; font-size: 0.9em;'>Developed with ❤️ by <a href='https://www.linkedin.com/in/manvendrapratapsinghdev/' target='_blank' style='color: #0077B5; text-decoration: none;'>Manvendra</a></p>
    </div>
    """,
    unsafe_allow_html=True
)
