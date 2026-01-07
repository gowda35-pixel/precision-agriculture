import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Data Visualizations", page_icon="📊", layout="wide")
st.title("📊 Data Visualizations")

st.write("Explore the crop dataset visually using the charts below.")

# Get the path to the data file
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
data_path = os.path.join(base_dir, 'data', 'crop_recommendation.csv')
df = pd.read_csv(data_path)

# Crop Distribution
if st.checkbox("Show Crop Distribution"):
    st.subheader("🌾 Crop Distribution")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(x="label", data=df, order=df["label"].value_counts().index, ax=ax)
    plt.xticks(rotation=90)
    st.pyplot(fig)

# Feature Correlation
if st.checkbox("Show Feature Correlation Heatmap"):
    st.subheader("📈 Feature Correlation Heatmap")
    numeric_df = df.drop(columns=["label"])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
