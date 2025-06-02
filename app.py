import streamlit as st
import numpy as np
import joblib

# Load model and scalers
model = joblib.load('india_house_price.pkl')
scaler = joblib.load('scaler.pkl')       # For input features
y_scaler = joblib.load('y_scaler.pkl')   # For target (Price)

# App config
st.set_page_config(page_title="India House Price Prediction", page_icon="🏡")
st.title("🏡 House Price Prediction App")
st.subheader("By Meet Patel (E041)")
st.write("This app uses a machine learning model to predict house prices based on features like bedrooms, bathrooms, condition, grade, etc.")

st.divider()

# User Inputs
bedrooms = st.number_input("🛏️ Number of Bedrooms", min_value=0, value=2)
bathrooms = st.number_input("🛁 Number of Bathrooms", min_value=0, value=2)
grade = st.slider("🏗️ Grade of the House (1 = Basic, 13 = Luxury)", 1, 13, value=7)
living_area = st.number_input("📐 Living Area (sq ft)", min_value=100, value=2000)
condition = st.slider("🔧 Condition of the House (1 = Poor, 5 = Excellent)", 1, 5, value=3)
schools = st.number_input("🏫 Number of Schools Nearby", min_value=0, value=2)

# Prepare input & scale
x = np.array([[bedrooms, bathrooms, grade, living_area, condition, schools]])
x_scaled = scaler.transform(x)

st.divider()

# Explanation Section
with st.expander("ℹ️ What is 'Grade of the House'?"):
    st.markdown("""
    - **1–3**: Poor/substandard  
    - **4–6**: Basic to average  
    - **7–10**: Good to very good  
    - **11–13**: High-end/luxury or custom-built
    """)

# Feature Importance Section
with st.expander("📊 Feature Importance Chart"):
    try:
        importances = model.feature_importances_
        labels = ["Bedrooms", "Bathrooms", "Grade", "Living Area", "Condition", "Schools"]
        st.bar_chart(data=dict(zip(labels, importances)))
    except AttributeError:
        st.warning("Feature importance is not available for this model.")

# Model Info Section
with st.expander("📈 Model & Dataset Info"):
    st.markdown("""
    **🔢 Model Used**: Random Forest Regressor  
    **📊 RMSE**: ₹2,29,809  
    **🎯 Features**:
    - Bedrooms
    - Bathrooms
    - Grade
    - Living Area
    - Condition
    - Schools Nearby  
    **📂 Data**: [House Prices India (Kaggle)](https://www.kaggle.com/datasets/sukhmandeepsinghbrar/house-prices-india)
    """)

# Prediction Button
if st.button("🔍 Predict House Price"):
    y_scaled = model.predict(x_scaled)
    y_pred = y_scaler.inverse_transform(y_scaled.reshape(-1, 1))[0][0]
    st.success(f"🏷️ Predicted House Price: ₹ {y_pred:,.2f}")
else:
    st.info("Please enter all details above and click 'Predict' to see the price.")
