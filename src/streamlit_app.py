import streamlit as st
import pandas as pd
import requests
import joblib

# Streamlit App
st.title("Intrusion Detection System Dashboard")

# Sidebar for user input
st.sidebar.header("Input Network Traffic Data")


def user_input_features():
    duration = st.sidebar.number_input('Duration', min_value=0, max_value=5000, value=0)
    protocol_type = st.sidebar.selectbox('Protocol Type', ('tcp', 'udp', 'icmp'))
    service = st.sidebar.selectbox('Service', ('http', 'smtp', 'ftp', 'telnet', 'dns', 'other'))
    flag = st.sidebar.selectbox('Flag', ('SF', 'S0', 'REJ', 'S1', 'RSTR', 'S2', 'RSTO', 'S3', 'OTH'))
    src_bytes = st.sidebar.number_input('Source Bytes', min_value=0, max_value=5000, value=0)
    dst_bytes = st.sidebar.number_input('Destination Bytes', min_value=0, max_value=5000, value=0)
    count_no = st.sidebar.number_input('Count No', min_value=0, max_value=5000, value=0)
    same_srv_rate = st.sidebar.number_input('Same Service Rate (%)', min_value=0, max_value=100, value=0)
    diff_srv_rate = st.sidebar.number_input('Different Service Rate (%)', min_value=0, max_value=100, value=0)
    dst_host_same_srv_rate = st.sidebar.number_input('Dst Host Same Service Rate (%)', min_value=0, max_value=100,
                                                     value=0)
    dst_host_diff_srv_rate = st.sidebar.number_input('Dst Host Different Service Rate (%)', min_value=0, max_value=100,
                                                     value=0)
    logged_in = st.sidebar.selectbox('Logged In', (0, 1))

    data = {
        'duration': [duration],
        'protocol_type': [protocol_type],
        'service': [service],
        'flag': [flag],
        'src_bytes': [src_bytes],
        'dst_bytes': [dst_bytes],
        'count_no': [count_no],
        'same_srv_rate': [same_srv_rate],
        'diff_srv_rate': [diff_srv_rate],
        'dst_host_same_srv_rate': [dst_host_same_srv_rate],
        'dst_host_diff_srv_rate': [dst_host_diff_srv_rate],
        'logged_in': [logged_in]
    }
    return pd.DataFrame(data)


input_df = user_input_features()

# Display user input
st.subheader("User Input Features")
st.write(input_df)

# Prediction
if st.button('Predict'):
    # Send the data to the Flask API for prediction
    try:
        response = requests.post('http://localhost:5000/predict', json=input_df.to_dict(orient='records'))
        if response.status_code == 200:
            prediction = response.json()
            st.subheader("Prediction Results")
            st.write(f"**Logistic Regression**:   {prediction['Logistic Regression']}")
            st.write(f"**SVM**: {prediction['SVM']}")
            st.write(f"**KNN**: {prediction['KNN']}")
        else:
            st.error("Error in prediction: " + response.text)
    except Exception as e:
        st.error(f"Error connecting to the backend: {e}")

# Option to visualize input data or results
st.subheader("Visualize Input Data")
if st.checkbox('Show Input Data'):
    st.write(input_df)

# Performance Visualization (Placeholder)
st.subheader("Intrusion Detection Performance")
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
performance_data = np.random.rand(10)  # Replace with real performance data
ax.plot(performance_data, marker='o')
ax.set_title("Model Performance Over Time")
ax.set_xlabel("Time")
ax.set_ylabel("Performance Metric")
st.pyplot(fig)
