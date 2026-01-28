💎 Diamond Price Prediction & Market Segmentation
📌 Project Overview

This project is a complete end-to-end Machine Learning application that predicts the price of diamonds and categorizes them into different market segments using advanced data analysis and machine learning techniques. The solution combines both supervised learning (regression) and unsupervised learning (clustering) and is deployed as an interactive Streamlit web application.

The primary goal of this project is to help users, sellers, and businesses estimate a fair diamond price and understand the market positioning of a diamond based on its physical and quality attributes.

🎯 Problem Statement

Diamond pricing depends on multiple factors such as carat weight, cut, color, clarity, and physical dimensions. Manual price estimation is often subjective and inconsistent. Additionally, businesses require an efficient way to segment diamonds into different market categories such as affordable, mid-range, and premium for better inventory and pricing strategies.

This project addresses both challenges by:

Predicting diamond prices accurately using machine learning

Segmenting diamonds into meaningful market groups using clustering

🧠 Solution Approach

The project is divided into two main components:

Price Prediction (Regression)
A supervised machine learning model is trained to predict the diamond price (in INR) based on its attributes.

Market Segmentation (Clustering)
An unsupervised learning model groups diamonds into different market segments without predefined labels.

The final models are integrated into a Streamlit application, allowing users to interactively input diamond details and get real-time predictions.

📊 Dataset Description

Source: Diamond Dataset

Shape: 53,940 rows × 10 features

Key Features:

carat: Weight of the diamond

cut: Quality of the cut

color: Diamond color grade

clarity: Purity of the diamond

depth: Total depth percentage

table: Table width percentage

x, y, z: Physical dimensions of the diamond

price: Diamond price (converted from USD to INR)

🔧 Data Preprocessing

To ensure high-quality data for modeling, the following preprocessing steps were applied:

Handled missing values

Replaced invalid zero values in x, y, and z

Removed outliers using the IQR method

Converted price from USD to INR

Ensured all values are suitable for machine learning algorithms

🛠 Feature Engineering

Several new features were created to enhance model performance and interpretability:

Volume: x × y × z (represents actual diamond size)

Price per Carat: Industry-standard pricing metric

Dimension Ratio: Captures shape proportion

Carat Category: Light, Medium, Heavy

These engineered features help the model capture real-world relationships more effectively.

🔢 Encoding & Scaling

Ordinal Encoding was applied to cut, color, and clarity since they have a natural order.

One-Hot Encoding was used for nominal features such as carat_category.

StandardScaler was applied to numerical features to ensure all features contribute equally, especially for clustering.

🤖 Model Building
🔹 Regression Models (Price Prediction)

Multiple models were trained and evaluated:

Linear Regression

Decision Tree

Random Forest

KNN

XGBoost (Best Performing Model)

Evaluation metrics used:

MAE

RMSE

R² Score

The best-performing model was saved using joblib.

🔹 Clustering Model (Market Segmentation)

Algorithm: KMeans Clustering

Number of Clusters: 3 (selected using the Elbow Method)

Cluster Interpretation:

Cluster 0: Affordable Small Diamonds

Cluster 1: Mid-range Balanced Diamonds

Cluster 2: Premium Heavy Diamonds

🌐 Streamlit Web Application

The project is deployed using Streamlit, providing a simple and interactive interface.

Features:

User input for diamond attributes

Real-time price prediction (in INR)

Market segment prediction

Clear display of cluster name

Clean and user-friendly UI

Streamlit enables quick deployment and makes the machine learning models accessible to non-technical users.

📦 Project Structure
├── app.py
├── price_model.pkl
├── preprocessor.pkl
├── cluster_model.pkl
├── requirements.txt
├── README.md

🚀 How to Run the Project

Clone the repository:

git clone https://github.com/your-username/diamond-price-prediction.git


Install dependencies:

pip install -r requirements.txt


Run the Streamlit app:

streamlit run app.py

📈 Future Enhancements

Hyperparameter tuning for better accuracy

Model explainability using SHAP

Deployment on cloud platforms

Advanced visualization of clusters

✅ Conclusion

This project demonstrates a complete machine learning workflow from data preprocessing to deployment. By combining regression and clustering techniques, it provides both accurate price prediction and valuable market insights, making it useful for both academic learning and real-world applications.
