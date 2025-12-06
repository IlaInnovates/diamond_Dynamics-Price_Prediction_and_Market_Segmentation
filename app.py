import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# -----------------------------
# Load Saved Artifacts
# -----------------------------
def load_file(path):
    with open(path, "rb") as f:
        return pickle.load(f)

reg_model = load_file("/content/regression_model.pkl")
clust_model = load_file("/content/clustering_model.pkl")
scaler = load_file("/content/scaler.pkl")
encoders = load_file("/content/encoders.pkl")
cluster_names = load_file("/content/cluster_names.pkl")

# -----------------------------
# Streamlit Setup
# -----------------------------
st.set_page_config(page_title="Diamond Dynamics", layout="wide")
st.title("💎 Diamond Dynamics: Price Prediction & Market Segmentation")

tab_price, tab_cluster, tab_info = st.tabs(
    ["💰 Price Prediction", "📊 Market Segment", "ℹ️ Info"]
)

# -----------------------------
# Input Form (Unique Keys)
# -----------------------------
def diamond_form(prefix=""):
    col1, col2, col3 = st.columns(3)

    with col1:
        carat = st.number_input("Carat", 0.1, 5.0, 0.7, 0.01, key=f"carat_{prefix}")
        x = st.number_input("Length (x)", 0.1, 15.0, 5.5, 0.1, key=f"x_{prefix}")
        y = st.number_input("Width (y)", 0.1, 15.0, 5.6, 0.1, key=f"y_{prefix}")

    with col2:
        z = st.number_input("Depth (z)", 0.1, 15.0, 3.5, 0.1, key=f"z_{prefix}")
        depth = st.number_input("Depth %", 40.0, 80.0, 61.0, 0.1, key=f"depth_{prefix}")
        table = st.number_input("Table %", 40.0, 80.0, 57.0, 0.1, key=f"table_{prefix}")

    with col3:
        cut = st.selectbox("Cut", ["Fair","Good","Very Good","Premium","Ideal"], key=f"cut_{prefix}")
        color = st.selectbox("Color", ["D","E","F","G","H","I","J"], key=f"color_{prefix}")
        clarity = st.selectbox("Clarity", ["IF","VVS1","VVS2","VS1","VS2","SI1","SI2","I1"], key=f"clarity_{prefix}")

    return {
        "carat":carat, "depth":depth, "table":table, "x":x, "y":y, "z":z,
        "cut":cut, "color":color, "clarity":clarity
    }

# -----------------------------
# Build Feature Vector
# -----------------------------
def build_features(data):
    df = pd.DataFrame([data])
    df["volume"] = df["x"] * df["y"] * df["z"]
    df["dimension_ratio"] = (df["x"] + df["y"]) / (2 * df["z"])
    df["carat_category"] = pd.cut(df["carat"], [0,0.5,1.5,10], labels=["Light","Medium","Heavy"])

    for col in ["cut", "color", "clarity", "carat_category"]:
        df[col] = encoders[col].transform(df[[col]])

    cols = [
        "carat","depth","table","x","y","z","cut","color","clarity",
        "volume","dimension_ratio","carat_category"
    ]

    X = df[cols].astype(float).values
    return scaler.transform(X)

# -----------------------------
# Price Prediction
# -----------------------------
with tab_price:
    st.subheader("Price Prediction (INR)")
    inputs = diamond_form("p")

    if st.button("Predict Price", key="pp"):
        X = build_features(inputs)
        pred_usd = reg_model.predict(X)[0]
        pred_inr = pred_usd * 83
        st.success(f"Predicted Price: ₹{pred_inr:,.2f}")

# -----------------------------
# Clustering
# -----------------------------
with tab_cluster:
    st.subheader("Market Segment")
    inputs2 = diamond_form("c")

    if st.button("Predict Segment", key="cs"):
        X2 = build_features(inputs2)
        label = int(clust_model.predict(X2)[0])
        st.success(f"Market Segment: {cluster_names[label]}")

# -----------------------------
# Info Tab
# -----------------------------
with tab_info:
    st.markdown(
        """
        ### 📘 Project Overview
        **Diamond Dynamics** predicts diamond prices using regression
        and classifies diamonds into 3 major **market segments** using clustering.

        This app demonstrates:
        - Data preprocessing
        - Feature engineering
        - Price prediction
        - Market segmentation
        - Streamlit UI
        """
    )
