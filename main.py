# House Price Prediction App using Random Forest 

#Libraries used
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

#  Generating synthetic housing data 
def generate_house_data(n_samples=100):
    np.random.seed(42)
    size = np.random.normal(1400, 100, n_samples)
    bedrooms = np.random.randint(1, 6, n_samples)
    bathrooms = np.random.randint(1, 4, n_samples)
    location = np.random.choice(['Mumbai', 'Pune', 'Nagpur', 'Delhi'], n_samples)
    price = size * 50 + bedrooms * 20000 + bathrooms * 15000 + np.random.normal(0, 50000, n_samples)
    return pd.DataFrame({
        'size': size,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'location': location,
        'price': price
    })

#  Location factor 
def location_factor(loc):
    factors = {'Mumbai': 1.5, 'Pune': 1.2, 'Nagpur': 1.0, 'Delhi': 1.3}
    return factors.get(loc, 1.0)

# Training
def train_model():
    df = generate_house_data()
    df['loc_factor'] = df['location'].apply(location_factor)
    X = df[['size', 'bedrooms', 'bathrooms', 'loc_factor']]
    y = df['price']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Store training data and metrics for evaluation
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    rmse = mean_squared_error(y, y_pred, squared=False)
    return model, mae, rmse, X.columns, model.feature_importances_, df, y_pred, y

# App
def main():
    st.set_page_config(page_title="House Price Predictor - RF", layout="centered")
    st.title("House Price Prediction App")

    model, mae, rmse, features, importances, X, y_pred, y = train_model()

    st.header("🔍 Predict Price of a Single House")
    size = st.number_input("House Size (sq ft)", min_value=500, max_value=10000, value=1500)
    bedrooms = st.slider("Bedrooms", 1, 5, 3)
    bathrooms = st.slider("Bathrooms", 1, 4, 2)
    location = st.selectbox("Location", ['Mumbai', 'Pune', 'Nagpur', 'Delhi'])
    currency = st.radio("Select Currency", ["\u20b9 INR", "$ USD"])

    if st.button("Predict Price"):
        loc_fact = location_factor(location)
        features_input = [[size, bedrooms, bathrooms, loc_fact]]
        price = model.predict(features_input)[0]

        error_margin = 20000
        lower = price - error_margin
        upper = price + error_margin

        if currency == "\u20b9 INR":
            price *= 83
            lower *= 83
            upper *= 83
            symbol = "\u20b9"
        else:
            symbol = "$"

        st.success(f"Estimated Price: {symbol}{price:,.2f}")
        st.info(f"Confidence Range: {symbol}{lower:,.0f} – {symbol}{upper:,.0f}")

    st.markdown("---")
    st.header("\U0001F4C9 Actual vs Predicted (Training Data)")
    fig2 = px.scatter(x=y, y=y_pred, labels={'x': 'Actual Price', 'y': 'Predicted Price'},
                     title="\U0001F4C8 Model Fit (Training Data)")
    st.plotly_chart(fig2)

    st.markdown("---")
    st.header("\U0001F4CB Feature Importance")
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    fig = px.bar(importance_df, x='Feature', y='Importance', title="\U0001F9E0 Feature Importance (Random Forest)")
    st.plotly_chart(fig)

    st.markdown("---")
    st.header("\U0001F4CA Model Evaluation")
    st.write(f"🔹 Mean Absolute Error (MAE): ₹{mae:,.0f}")
    st.write(f"🔹 Root Mean Squared Error (RMSE): ₹{rmse:,.0f}")

if __name__ == '__main__':
    main()


