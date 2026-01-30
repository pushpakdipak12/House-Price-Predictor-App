import streamlit as st
import pandas as pd
import joblib

# ---------------- LOAD MODEL ----------------
model = joblib.load("xgboost_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")
st.title("🏠 California House Price Predictor")
st.write("Enter house details below to estimate the price.")

# ---------------- USER INPUTS ----------------
longitude = st.number_input("Longitude", value=-120.0, format="%.5f")
latitude = st.number_input("Latitude", value=37.0, format="%.5f")
housing_median_age = st.number_input("House Age", min_value=1, max_value=100, value=20)
total_rooms = st.number_input("Total Rooms", min_value=1, value=1500)
total_bedrooms = st.number_input("Total Bedrooms", min_value=1, value=300)
population = st.number_input("Population", min_value=1, value=800)
households = st.number_input("Households", min_value=1, value=300)
median_income = st.number_input("Median Income", min_value=0.1, max_value=15.0, value=5.0)

ocean_proximity = st.selectbox(
    "Ocean Proximity",
    ["NEAR BAY", "INLAND", "1H OCEAN", "NEAR OCEAN", "ISLAND"]
)

# ---------------- PREDICTION ----------------
if st.button("Predict Price"):

    # Safety check (avoid divide by zero)
    if households == 0 or total_rooms == 0:
        st.error("Households and Total Rooms must be greater than zero.")
        st.stop()

    # Create input dictionary
    data = {
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity
    }

    df = pd.DataFrame([data])

    # -------- Feature Engineering (same as training) --------
    df['rooms_per_household'] = df['total_rooms'] / df['households']
    df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
    df['population_per_household'] = df['population'] / df['households']

    # -------- One-Hot Encoding --------
    df = pd.get_dummies(df)

    # -------- Fix any possible column mismatch --------
    df.columns = df.columns.str.replace("<", "", regex=False)

    # -------- Align with training columns --------
    df = df.reindex(columns=model_columns, fill_value=0)

    # -------- Prediction --------
    prediction = model.predict(df)[0]

    st.success(f"💰 Estimated House Price: ${prediction:,.2f}")

    # Optional: Show input summary
    with st.expander("See Input Data"):
        st.dataframe(df)
