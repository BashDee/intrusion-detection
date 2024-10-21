from flask import Flask, request, jsonify
import joblib
import pandas as pd
import logging 

# Initialize Flask app
app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)

# Load the feature selector, scaler, and models
selector = joblib.load('../models/feature_selector.pkl')
scaler = joblib.load('../models/scaler.pkl')
lr_model = joblib.load('../models/logistic_regression.pkl')
svm_model = joblib.load('../models/svm_model.pkl')
knn_model = joblib.load('../models/knn_model.pkl')

# Load the saved selected columns (these are the columns used during training)
selected_columns = joblib.load('../models/selected_columns.pkl')

# Define the categorical columns
categorical_columns = ['protocol_type', 'service', 'flag']

# Function to align features with training columns
def align_features(df, reference_columns):
    # One-hot encode the categorical features
    df_encoded = pd.get_dummies(df, columns=categorical_columns)
    
    # Reindex to match the columns used during training (fill missing columns with 0)
    df_aligned = df_encoded.reindex(columns=reference_columns, fill_value=0)
    
    return df_aligned

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from POST request
        data = request.json
        logging.debug(f"Input Data: {data}")
        
        # Convert input data to DataFrame
        df = pd.DataFrame(data)
        
        # Align input data columns with training data columns
        df_aligned = align_features(df, selected_columns)

        # Apply feature selection
        X_selected = selector.transform(df_aligned)
        logging.debug(f"Transformed Data: {X_selected}")

        # Normalize the features
        X_scaled = scaler.transform(X_selected)

        # Make predictions with each model
        lr_pred = lr_model.predict(X_scaled)[0]
        svm_pred = svm_model.predict(X_scaled)[0]
        knn_pred = knn_model.predict(X_scaled)[0]

        # Return predictions
        return jsonify({
            'Logistic Regression': int(lr_pred),
            'SVM': int(svm_pred),
            'KNN': int(knn_pred)
        })

    except Exception as e:
        logging.debug(f"Error During Prediction: {e}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
