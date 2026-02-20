import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(
    page_title="Traffic Anomaly Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# Title
# ---------------------------------------------------
st.title("🚗 Traffic Flow Anomaly Detection")
st.markdown("Detect anomalies in network traffic using Isolation Forest algorithm")

# ---------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------
st.sidebar.header("⚙️ Configuration")

contamination = st.sidebar.slider(
    "Contamination Rate",
    0.01, 0.2, 0.04, 0.01
)

n_estimators = st.sidebar.slider(
    "Number of Trees",
    50, 200, 100, 10
)

# ---------------------------------------------------
# Dataset Loading
# ---------------------------------------------------
st.sidebar.header("📁 Data")

uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")
use_sample = st.sidebar.checkbox("Use sample dataset", value=True)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✓ File uploaded successfully")

elif use_sample:
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(BASE_DIR, "embedded_system_network_security_dataset.csv")
        df = pd.read_csv(file_path)
        st.sidebar.success("✓ Sample dataset loaded")
    except Exception as e:
        st.error(f"Sample dataset not found. Error: {e}")
        st.stop()

else:
    st.error("Please upload a CSV file or enable sample dataset.")
    st.stop()

# ---------------------------------------------------
# Data Preprocessing
# ---------------------------------------------------
st.sidebar.header("⚙️ Data Processing")

features = df.drop(columns=['label'], errors='ignore')

# Convert bool to int
for col in features.columns:
    if features[col].dtype == 'bool':
        features[col] = features[col].astype(int)

# Fill missing values
features = features.fillna(features.mean())

# Scale features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
scaled_df = pd.DataFrame(scaled_features, columns=features.columns)

# ---------------------------------------------------
# Train Isolation Forest
# ---------------------------------------------------
model = IsolationForest(
    n_estimators=n_estimators,
    contamination=contamination,
    max_samples=256,
    random_state=42
)

model.fit(scaled_df)
anomaly_labels = model.predict(scaled_df)
scaled_df['anomaly'] = anomaly_labels

# ---------------------------------------------------
# Tabs
# ---------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Overview", "📈 Visualizations", "📋 Details", "📥 Export"]
)

# ---------------------------------------------------
# TAB 1 — Overview
# ---------------------------------------------------
with tab1:

    normal_count = len(scaled_df[scaled_df['anomaly'] == 1])
    anomaly_count = len(scaled_df[scaled_df['anomaly'] == -1])
    total_count = len(scaled_df)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", total_count)
    col2.metric("Normal", normal_count, f"{normal_count/total_count*100:.1f}%")
    col3.metric("Anomalies", anomaly_count, f"{anomaly_count/total_count*100:.1f}%")
    col4.metric("Anomaly Rate", f"{contamination*100:.1f}%")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Data")
        st.dataframe(df.head(), use_container_width=True)

    with col2:
        st.subheader("With Anomaly Labels")
        st.dataframe(scaled_df.head(), use_container_width=True)

# ---------------------------------------------------
# TAB 2 — Visualizations
# ---------------------------------------------------
with tab2:

    normal = scaled_df[scaled_df['anomaly'] == 1]
    anomaly = scaled_df[scaled_df['anomaly'] == -1]

    numeric_cols = scaled_df.columns.drop('anomaly').tolist()

    # 2D Plot
    if len(numeric_cols) >= 2:

        feat1 = st.selectbox("X-axis:", numeric_cols)
        feat2 = st.selectbox("Y-axis:", numeric_cols, index=1)

        fig, ax = plt.subplots()

        ax.scatter(normal[feat1], normal[feat2],
                   c='blue', label='Normal', alpha=0.6)

        ax.scatter(anomaly[feat1], anomaly[feat2],
                   c='red', label='Anomaly', marker='x')

        ax.set_xlabel(feat1)
        ax.set_ylabel(feat2)
        ax.legend()

        st.pyplot(fig)

    # 3D Plot
    if len(numeric_cols) >= 3:

        feat_x = st.selectbox("3D X-axis:", numeric_cols, key="3dx")
        feat_y = st.selectbox("3D Y-axis:", numeric_cols, index=1, key="3dy")
        feat_z = st.selectbox("3D Z-axis:", numeric_cols, index=2, key="3dz")

        fig3d = go.Figure()

        fig3d.add_trace(go.Scatter3d(
            x=normal[feat_x],
            y=normal[feat_y],
            z=normal[feat_z],
            mode='markers',
            name='Normal',
            marker=dict(size=4, color='blue')
        ))

        fig3d.add_trace(go.Scatter3d(
            x=anomaly[feat_x],
            y=anomaly[feat_y],
            z=anomaly[feat_z],
            mode='markers',
            name='Anomaly',
            marker=dict(size=8, color='red')
        ))

        fig3d.update_layout(height=600)

        st.plotly_chart(fig3d, use_container_width=True)

# ---------------------------------------------------
# TAB 3 — Details
# ---------------------------------------------------
with tab3:

    st.subheader("Statistics")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Normal Data")
        st.dataframe(normal.describe())

    with col2:
        st.write("Anomaly Data")
        st.dataframe(anomaly.describe())

# ---------------------------------------------------
# TAB 4 — Export
# ---------------------------------------------------
with tab4:

    results_df = df.copy()
    results_df['anomaly_prediction'] = anomaly_labels
    results_df['anomaly_type'] = results_df['anomaly_prediction'].map(
        {1: 'Normal', -1: 'Anomaly'}
    )

    csv = results_df.to_csv(index=False)

    st.download_button(
        label="📥 Download Results",
        data=csv,
        file_name="anomaly_results.csv",
        mime="text/csv"
    )

# ---------------------------------------------------
# Footer
# ---------------------------------------------------
st.divider()
st.caption("🔬 Traffic Flow Anomaly Detection System | Powered by Streamlit")
