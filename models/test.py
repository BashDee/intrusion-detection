import joblib

# Load the .pkl file
with open('knn_model.pkl', 'rb') as file:
    columns = joblib.load(file)

# Print the contents
print(columns)
