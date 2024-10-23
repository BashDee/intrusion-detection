import streamlit as st
import requests
import json

st.title('Intrusion Detection System')

# Create input fields for each feature
duration = st.number_input('Duration', value=0)
src_bytes = st.number_input('Source Bytes', value=0)
dst_bytes = st.number_input('Destination Bytes', value=0)
wrong_fragment = st.number_input('Wrong Fragment', value=0)
hot = st.number_input('Hot', value=0)
count = st.number_input('Count', value=0)
srv_count = st.number_input('Service Count', value=0)
dst_host_count = st.number_input('Destination Host Count', value=0)
dst_host_srv_count = st.number_input('Destination Host Service Count', value=0)
service_ecr_i = st.number_input('Service ECR I', value=0)

# Prepare the input data for prediction
data = [{
    "duration": duration,
    "src_bytes": src_bytes,
    "dst_bytes": dst_bytes,
    "wrong_fragment": wrong_fragment,
    "hot": hot,
    "count": count,
    "srv_count": srv_count,
    "dst_host_count": dst_host_count,
    "dst_host_srv_count": dst_host_srv_count,
    "service_ecr_i": service_ecr_i
}]

# Button to make the prediction
if st.button('Predict'):
    response = requests.post('http://127.0.0.1:5000/predict', json=data)
    
    if response.status_code == 200:
        result = response.json()
        st.write('Prediction Results:')
        st.write(result)
    else:
        st.error(f"Error connecting to the backend: {response.text}")
