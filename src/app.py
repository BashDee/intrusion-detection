from flask import Flask, request, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Load pre-trained models and feature scaler
scaler = joblib.load('../models/scaler.pkl')
columns = joblib.load('../models/columns.pkl')  # The columns used during training
lr_model = joblib.load('../models/logistic_regression.pkl')
svm_model = joblib.load('../models/svm_model.pkl')
knn_model = joblib.load('../models/knn_model.pkl')

def align_features(df, reference_columns):
    """Align input DataFrame columns to match the trained model columns."""
    # df_encoded = pd.get_dummies(df, columns=['protocol_type', 'service', 'flag'])
    df_aligned = df_encoded.reindex(columns=reference_columns, fill_value=0)
    return df_aligned

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive JSON data
        data = request.json
        logging.debug(f"Received input data: {data}")

        # Convert JSON data to DataFrame
        df = pd.DataFrame(data)
        
        # Align the input data to the training data columns
        df_aligned = df.reindex(columns=columns, fill_value=0)
        logging.debug(f"Aligned DataFrame columns: {df_aligned.columns}")

         # Feature selection
        # X = selector.transform(df_aligned)
        
        # Scale the input data
        X_scaled = scaler.transform(df_aligned)
        
        # Make predictions
        lr_pred = lr_model.predict(X_scaled)[0]
        svm_pred = svm_model.predict(X_scaled)[0]
        knn_pred = knn_model.predict(X_scaled)[0]
        
        # Return predictions in JSON format
        return jsonify({
            'Logistic Regression': lr_pred,
            'SVM': svm_pred,
            'KNN': knn_pred
        })

    except Exception as e:
        # Log the exception
        logging.debug(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
