import pandas as pd
import streamlit as st

st.set_page_config(page_title="A/B Test Dashboard", layout="wide")

st.title("📊 A/B Testing Results for ML Models")

# Load log file
try:
    df = pd.read_csv("logs/predictions.log", header=None,
                     names=["timestamp", "model", "features", "prediction", "feedback"])

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["feedback"] = df["feedback"].astype(str).str.lower() == "true"

    st.sidebar.subheader("Summary")
    st.sidebar.write(f"Total Predictions: {len(df)}")
    st.sidebar.write(f"With Feedback: {df['feedback'].notnull().sum()}")

    st.subheader("Prediction Counts")
    st.bar_chart(df["model"].value_counts())

    st.subheader("Accuracy by Model")
    acc = df.groupby("model")["feedback"].mean().sort_values(ascending=False)
    st.dataframe(acc.rename("Accuracy (%)") * 100)

    st.subheader("Accuracy Over Time (Optional)")
    df["date"] = df["timestamp"].dt.date
    daily_acc = df.groupby(["date", "model"])["feedback"].mean().unstack()
    st.line_chart(daily_acc)

except Exception as e:
    st.error(f"Failed to load data: {e}")