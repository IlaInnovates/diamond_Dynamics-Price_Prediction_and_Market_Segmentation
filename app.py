import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Diamond Analytics", layout="centered")
st.title("💎 Diamond Price Prediction & Market Segmentation")

# ---------------------------------------------------
# Load models safely
# ---------------------------------------------------
REQUIRED_FILES = {
    "Price Model": "price_model.pkl",
    "Preprocessor": "preprocessor.pkl",
    "Cluster Model": "cluster_model.pkl"
}

for name, file in REQUIRED_FILES.items():
    if not os.path.exists(file):
        st.error(f"❌ {file} not found. Please place it in the same folder as app.py")
        st.stop()

price_model = joblib.load("price_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
cluster_model = joblib.load("cluster_model.pkl")

st.success("✅ All models loaded successfully")

# ---------------------------------------------------
# User Inputs
# ---------------------------------------------------
st.subheader("🔢 Enter Diamond Details")

carat = st.number_input("Carat", 0.1, 5.0, 1.0)
cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
color = st.selectbox("Color", ["J", "I", "H", "G", "F", "E", "D"])
clarity = st.selectbox("Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])

depth = st.number_input("Depth", 40.0, 80.0, 60.0)
table = st.number_input("Table", 40.0, 80.0, 55.0)

x = st.number_input("X (length)", 0.0, 10.0, 5.0)
y = st.number_input("Y (width)", 0.0, 10.0, 5.0)
z = st.number_input("Z (depth)", 0.0, 10.0, 3.0)

# ---------------------------------------------------
# Feature Engineering (MUST MATCH TRAINING)
# ---------------------------------------------------
volume = x * y * z
dimension_ratio = (x + y) / (2 * z) if z != 0 else 0
price_per_carat = 0  # dummy value (not used for prediction input)
carat_category = "Medium"  # placeholder (handled by encoder)

input_df = pd.DataFrame([{
    "carat": carat,
    "cut": cut,
    "color": color,
    "clarity": clarity,
    "depth": depth,
    "table": table,
    "x": x,
    "y": y,
    "z": z,
    "volume": volume,
    "dimension_ratio": dimension_ratio,
    "price_per_carat": price_per_carat,
    "carat_category": carat_category
}])

# ---------------------------------------------------
# PRICE PREDICTION
# ---------------------------------------------------
st.subheader("💰 Price Prediction")

if st.button("Predict Price"):
    try:
        X_processed = preprocessor.transform(input_df)
        price = price_model.predict(X_processed)[0]
        st.success(f"💎 Predicted Diamond Price: ₹ {price:,.2f}")
    except Exception as e:
        st.error("❌ Price prediction failed")
        st.write(e)

# ---------------------------------------------------
# MARKET SEGMENT PREDICTION
# ---------------------------------------------------
st.subheader("📊 Market Segment Prediction")

if st.button("Predict Market Segment"):
    try:
        cluster_features = input_df[
            ["carat", "depth", "table", "x", "y", "z", "volume"]
        ]

        cluster_id = cluster_model.predict(cluster_features)[0]

        cluster_names = {
            0: "Affordable Small Diamonds",
            1: "Mid-range Balanced Diamonds",
            2: "Premium Heavy Diamonds"
        }

        st.info(f"📌 Cluster ID: {cluster_id}")
        st.success(f"🏷 Market Segment: {cluster_names.get(cluster_id)}")

    except Exception as e:
        st.error("❌ Market segmentation failed")
        st.write(e)
	