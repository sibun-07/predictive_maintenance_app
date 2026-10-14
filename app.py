import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load trained model, columns, and dataset
with open('model.pkl', 'rb') as file:
    model, model_columns = pickle.load(file)

df = pd.read_csv('ai4i2020.csv')

st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")

st.title("⚙️ Industry 4.0: Predictive Maintenance & Failure Mode Analytics")
st.write("This dashboard predicts operational machine health and analyzes specific failure modes under Industry 4.0 standards.")

# --- SIDEBAR FOR PREDICTIONS ---
st.sidebar.header("🔧 Machine Sensor Inputs")

machine_type = st.sidebar.selectbox("Machine Type", ["L (Low)", "M (Medium)", "H (High)"])
air_temp = st.sidebar.slider("Air temperature [K]", 295.0, 305.0, 298.0)
process_temp = st.sidebar.slider("Process temperature [K]", 305.0, 315.0, 308.0)
rotational_speed = st.sidebar.slider("Rotational speed [rpm]", 1100, 2900, 1500)
torque = st.sidebar.slider("Torque [Nm]", 3.0, 80.0, 40.0)
tool_wear = st.sidebar.slider("Tool wear [min]", 0, 250, 50)

type_L = 1 if "Low" in machine_type else 0
type_M = 1 if "Medium" in machine_type else 0
type_H = 1 if "High" in machine_type else 0

input_data = pd.DataFrame({
    'Air temperature [K]': [air_temp],
    'Process temperature [K]': [process_temp],
    'Rotational speed [rpm]': [rotational_speed],
    'Torque [Nm]': [torque],
    'Tool wear [min]': [tool_wear],
    'Type_L': [type_L],
    'Type_M': [type_M],
    'Type_H': [type_H]
})

input_data = input_data.reindex(columns=model_columns, fill_value=0)

if st.sidebar.button("Predict Machine Status"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)
    
    st.sidebar.subheader("📊 Prediction Result")
    if prediction[0] == 1:
        st.sidebar.error(f"🚨 High Risk of Failure!\nFailure Probability: {probability[0][1]*100:.2f}%")
    else:
        st.sidebar.success(f"🟢 Machine is Healthy.\nSuccess Probability: {probability[0][0]*100:.2f}%")

# --- MAIN PAGE: DATA VISUALIZATIONS & FAILURE MODES ---
st.markdown("---")
st.subheader("📈 Operational Data Visualizations & Failure Mode Breakdown")

col1, col2 = st.columns(2)

with col1:
    st.write("### Torque vs. Rotational Speed (by Failure)")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(
        data=df.sample(1000), 
        x='Rotational speed [rpm]', 
        y='Torque [Nm]', 
        hue='Machine failure', 
        palette={0: 'green', 1: 'red'}, 
        alpha=0.6,
        ax=ax
    )
    plt.title("Machine Failures Clustering")
    st.pyplot(fig)

with col2:
    st.write("### Breakdown of Historical Failure Types")
    # Sum up the counts of each specific failure mode in the dataset
    failure_counts = {
        'Tool Wear (TWF)': df['TWF'].sum(),
        'Heat Dissipation (HDF)': df['HDF'].sum(),
        'Power (PWF)': df['PWF'].sum(),
        'Overstrain (OSF)': df['OSF'].sum(),
        'Random (RNF)': df['RNF'].sum()
    }
    
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.barplot(x=list(failure_counts.keys()), y=list(failure_counts.values()), palette='viridis', ax=ax2)
    plt.xticks(rotation=30, ha='right')
    plt.ylabel("Occurrences")
    plt.title("Frequency by Specific Failure Mode")
    st.pyplot(fig2)